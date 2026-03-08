# pipeline/embed.py
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from pipeline.config import DINO_MODEL, EMBED_DIM, EMBED_BATCH_SIZE, EMBEDDINGS_DIR


def load_model():
    """Load DINOv2-small model and preprocessing transform."""
    model = torch.hub.load("facebookresearch/dinov2", DINO_MODEL)
    model.eval()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform


def embed_crops(
    model,
    transform,
    crop_paths: list[str],
    batch_size: int = EMBED_BATCH_SIZE,
) -> np.ndarray:
    """Compute DINOv2 embeddings for a list of crop images."""
    device = next(model.parameters()).device
    all_embeddings = []

    for i in range(0, len(crop_paths), batch_size):
        batch_paths = crop_paths[i : i + batch_size]
        images = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            images.append(transform(img))

        batch = torch.stack(images).to(device)
        with torch.no_grad():
            embeddings = model(batch)

        embeddings = embeddings.cpu().numpy()
        all_embeddings.append(embeddings)

    embeddings = np.vstack(all_embeddings)
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings


def run_embedding(
    crops_df: pd.DataFrame,
    embeddings_dir: Path = EMBEDDINGS_DIR,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Run embedding extraction on all crops."""
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    print("Loading DINOv2 model...")
    model, transform = load_model()

    crop_paths = crops_df["crop_path"].tolist()
    print(f"Embedding {len(crop_paths)} crops...")
    embeddings = embed_crops(model, transform, crop_paths)

    # Save
    np.save(embeddings_dir / "crop_embeddings.npy", embeddings)
    index_df = crops_df[["crop_id", "frame_id", "video_id", "crop_path"]].copy()
    index_df.to_csv(embeddings_dir / "crop_index.csv", index=False)

    print(f"Saved embeddings: {embeddings.shape}")
    return embeddings, index_df


if __name__ == "__main__":
    from pipeline.config import DATA_DIR
    crops_df = pd.read_csv(DATA_DIR / "crops_metadata.csv")
    run_embedding(crops_df)
