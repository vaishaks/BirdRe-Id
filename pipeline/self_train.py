# pipeline/self_train.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from pipeline.config import EMBEDDINGS_DIR, MODELS_DIR, EMBED_DIM


class BirdSessionDataset(Dataset):
    """Dataset that yields pairs of crops from the same session."""

    def __init__(self, session_index: pd.DataFrame, crop_index: pd.DataFrame, transform=None):
        self.transform = transform
        self.sessions = []

        crop_path_map = dict(zip(crop_index["crop_id"], crop_index["crop_path"]))

        for _, row in session_index.iterrows():
            crop_ids = row["crop_ids"]
            if isinstance(crop_ids, str):
                crop_ids = crop_ids.split(",")
            paths = [crop_path_map[cid] for cid in crop_ids if cid in crop_path_map]
            if len(paths) >= 2:  # need at least 2 crops for a pair
                self.sessions.append(paths)

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        paths = self.sessions[idx]
        # Randomly pick 2 different crops from the same session
        indices = torch.randperm(len(paths))[:2]
        img1 = Image.open(paths[indices[0]]).convert("RGB")
        img2 = Image.open(paths[indices[1]]).convert("RGB")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""

    def __init__(self, input_dim=384, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ContrastiveModel(nn.Module):
    """DINOv2 backbone + projection head."""

    def __init__(self, backbone, proj_head, freeze_backbone=True):
        super().__init__()
        self.backbone = backbone
        self.proj_head = proj_head
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        with torch.no_grad() if not any(p.requires_grad for p in self.backbone.parameters()) else torch.enable_grad():
            features = self.backbone(x)
        projected = self.proj_head(features)
        return F.normalize(projected, dim=1)

    def embed(self, x):
        """Get projected features (for final embedding, not backbone-only)."""
        with torch.no_grad():
            features = self.backbone(x)
            projected = self.proj_head(features)
        return F.normalize(projected, dim=1)


def nt_xent_loss(z1, z2, temperature=0.1):
    """NT-Xent (normalized temperature-scaled cross entropy) loss."""
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)

    # Positive pairs: (i, i+B) and (i+B, i)
    pos_indices = torch.arange(batch_size, device=z.device)
    labels = torch.cat([pos_indices + batch_size, pos_indices], dim=0)

    loss = F.cross_entropy(sim, labels)
    return loss


def train_contrastive(
    session_index: pd.DataFrame,
    crop_index: pd.DataFrame,
    num_epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    temperature: float = 0.1,
):
    """Train contrastive model using session-based positive pairs."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = BirdSessionDataset(session_index, crop_index, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    print(f"Training sessions with >= 2 crops: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")

    # Load backbone
    print("Loading DINOv2 backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval()

    proj_head = ProjectionHead(input_dim=EMBED_DIM, hidden_dim=256, output_dim=128)
    model = ContrastiveModel(backbone, proj_head, freeze_backbone=True)
    model = model.to(device)

    optimizer = torch.optim.Adam(proj_head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for img1, img2 in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            img1, img2 = img1.to(device), img2.to(device)

            z1 = model(img1)
            z2 = model(img2)

            loss = nt_xent_loss(z1, z2, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "proj_head_state_dict": proj_head.state_dict(),
        "config": {"input_dim": EMBED_DIM, "hidden_dim": 256, "output_dim": 128},
    }, MODELS_DIR / "contrastive_head.pt")
    print(f"Saved projection head to {MODELS_DIR / 'contrastive_head.pt'}")

    return model


def reembed_crops(
    model,
    crop_index: pd.DataFrame,
    batch_size: int = 32,
) -> np.ndarray:
    """Re-embed all crops using the fine-tuned model."""
    device = next(model.parameters()).device
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    crop_paths = crop_index["crop_path"].tolist()
    all_embeddings = []

    for i in tqdm(range(0, len(crop_paths), batch_size), desc="Re-embedding crops"):
        batch_paths = crop_paths[i:i + batch_size]
        images = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            images.append(transform(img))

        batch = torch.stack(images).to(device)
        with torch.no_grad():
            embeddings = model.embed(batch)
        all_embeddings.append(embeddings.cpu().numpy())

    embeddings = np.vstack(all_embeddings)
    return embeddings


def run_self_training(num_epochs: int = 20):
    """Run the full self-training pipeline."""
    session_index = pd.read_csv(EMBEDDINGS_DIR / "session_index.csv")
    crop_index = pd.read_csv(EMBEDDINGS_DIR / "crop_index.csv")

    # Train
    model = train_contrastive(session_index, crop_index, num_epochs=num_epochs)

    # Re-embed
    print("\nRe-embedding all crops...")
    new_embeddings = reembed_crops(model, crop_index)

    # L2 normalize
    norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
    new_embeddings = new_embeddings / norms

    # Save
    np.save(EMBEDDINGS_DIR / "crop_embeddings_v2.npy", new_embeddings)
    print(f"Saved new embeddings: {new_embeddings.shape} to crop_embeddings_v2.npy")

    # Recompute session embeddings
    from pipeline.group import run_grouping
    session_embs, session_df = run_grouping(crop_index, new_embeddings)
    print(f"Recomputed session embeddings: {session_embs.shape}")

    return new_embeddings


if __name__ == "__main__":
    import sys
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    run_self_training(num_epochs=epochs)
