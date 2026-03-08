# pipeline/explain.py
"""Explainability: DINOv2 attention heatmaps showing discriminative bird features."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pipeline.config import DINO_MODEL, EMBEDDINGS_DIR, CLUSTERS_DIR
import pandas as pd


def get_attention_maps(model, image_tensor, device):
    """Extract CLS attention from last transformer block.

    Returns attention map of shape (num_patches_h, num_patches_w) showing
    how much the CLS token attends to each spatial patch.
    """
    model.eval()
    x = image_tensor.unsqueeze(0).to(device)

    # Forward through patch embed + pos embed
    x = model.prepare_tokens_with_masks(x)

    # Forward through all blocks except last
    for blk in model.blocks[:-1]:
        x = blk(x)

    # Last block: manually compute attention weights
    last_block = model.blocks[-1]
    # Layer norm before attention
    x_normed = last_block.norm1(x)

    B, N, C = x_normed.shape
    qkv = last_block.attn.qkv(x_normed).reshape(B, N, 3, last_block.attn.num_heads, C // last_block.attn.num_heads)
    q, k, v = torch.unbind(qkv, 2)  # each (B, N, num_heads, head_dim)
    q = q.transpose(1, 2)  # (B, num_heads, N, head_dim)
    k = k.transpose(1, 2)

    # Compute attention weights
    scale = (C // last_block.attn.num_heads) ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)  # (B, num_heads, N, N)

    # CLS token attention to patches: average across heads
    # attn[0, :, 0, 1:] = attention from CLS to each patch token
    cls_attn = attn[0, :, 0, 1:].mean(dim=0)  # (num_patches,)

    # Reshape to spatial grid
    patch_size = model.patch_size  # 14
    h = w = 224 // patch_size  # 16x16
    attn_map = cls_attn.reshape(h, w).detach().cpu().numpy()

    return attn_map


def create_attention_overlay(image_path, attn_map, alpha=0.5):
    """Overlay attention heatmap on original image.

    Returns PIL Image with heatmap overlay.
    """
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_np = np.array(img) / 255.0

    # Upsample attention map to image size
    attn_resized = np.array(Image.fromarray(attn_map).resize((224, 224), Image.BILINEAR))

    # Normalize to [0, 1]
    attn_min, attn_max = attn_resized.min(), attn_resized.max()
    if attn_max > attn_min:
        attn_resized = (attn_resized - attn_min) / (attn_max - attn_min)

    # Apply colormap
    heatmap = cm.jet(attn_resized)[:, :, :3]

    # Blend
    blended = img_np * (1 - alpha) + heatmap * alpha
    blended = (blended * 255).clip(0, 255).astype(np.uint8)

    return Image.fromarray(blended), attn_resized


def generate_explainability_gallery(
    assignments_df: pd.DataFrame,
    session_index: pd.DataFrame,
    crop_index: pd.DataFrame,
    output_path: Path,
    samples_per_bird: int = 4,
):
    """Generate HTML gallery with attention heatmap overlays per bird cluster."""
    print("Loading DINOv2 for attention extraction...")
    model = torch.hub.load("facebookresearch/dinov2", DINO_MODEL)
    model.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    crop_path_map = dict(zip(crop_index["crop_id"], crop_index["crop_path"]))

    # Create output directory for heatmap images
    heatmap_dir = output_path.parent / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)

    html = """<!DOCTYPE html>
<html><head><style>
body { font-family: sans-serif; margin: 20px; background: #f5f5f5; }
.species-group { margin: 30px 0; }
.species-group h2 { color: #333; border-bottom: 2px solid #666; padding-bottom: 5px; }
.bird { margin: 15px 0; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.bird h3 { margin-top: 0; color: #555; }
.bird .stats { color: #888; font-size: 14px; margin-bottom: 10px; }
.pair-grid { display: flex; flex-wrap: wrap; gap: 12px; }
.pair { display: flex; flex-direction: column; align-items: center; gap: 2px; }
.pair img { width: 140px; height: 140px; object-fit: cover; border-radius: 4px; border: 1px solid #ddd; }
.pair .label { font-size: 11px; color: #999; }
.noise { opacity: 0.5; }
.legend { padding: 15px; background: white; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.legend img { height: 20px; width: 200px; }
</style></head><body>
<h1>Bird Identity Gallery — Attention Explainability</h1>
<div class="legend">
<p><strong>How to read:</strong> Each bird shows original crops (top) and attention heatmaps (bottom).
<span style="color:red">Red/yellow</span> = regions the model focuses on most for identity.
<span style="color:blue">Blue</span> = regions ignored. Look for consistent patterns across crops of the same bird.</p>
</div>
"""

    total_crops_processed = 0

    for species in sorted(assignments_df["species"].unique()):
        sp_df = assignments_df[assignments_df["species"] == species]
        html += f'<div class="species-group"><h2>{species}</h2>\n'

        for bird_id in sorted(sp_df["bird_id"].unique()):
            is_noise = "noise" in bird_id
            bird_df = sp_df[sp_df["bird_id"] == bird_id]
            n_sessions = len(bird_df)
            css_class = "bird noise" if is_noise else "bird"

            html += f'<div class="{css_class}"><h3>{bird_id}</h3>\n'
            html += f'<div class="stats">{n_sessions} sessions</div>\n'
            html += '<div class="pair-grid">\n'

            # Collect crop paths from sessions
            all_crop_paths = []
            for _, row in bird_df.iterrows():
                session_row = session_index[session_index["session_id"] == row["session_id"]]
                if len(session_row) == 0:
                    continue
                crop_ids = session_row.iloc[0]["crop_ids"]
                if isinstance(crop_ids, str):
                    crop_ids = crop_ids.split(",")
                for cid in crop_ids:
                    cp = crop_path_map.get(cid)
                    if cp:
                        all_crop_paths.append((cid, cp))

            # Sample
            if len(all_crop_paths) > samples_per_bird:
                chosen = rng.choice(len(all_crop_paths), samples_per_bird, replace=False)
                all_crop_paths = [all_crop_paths[i] for i in chosen]

            for crop_id, crop_path in all_crop_paths:
                # Compute attention heatmap
                img_tensor = transform(Image.open(crop_path).convert("RGB"))
                attn_map = get_attention_maps(model, img_tensor, device)
                overlay, _ = create_attention_overlay(crop_path, attn_map)

                # Save heatmap
                heatmap_path = heatmap_dir / f"{crop_id}_attn.jpg"
                overlay.save(str(heatmap_path), quality=90)
                total_crops_processed += 1

                html += f'<div class="pair">'
                html += f'<img src="file://{crop_path}" alt="original">'
                html += f'<div class="label">original</div>'
                html += f'<img src="file://{heatmap_path.resolve()}" alt="attention">'
                html += f'<div class="label">attention</div>'
                html += f'</div>\n'

            html += '</div></div>\n'

        html += '</div>\n'

    html += '</body></html>'

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"\nGenerated explainability gallery: {output_path}")
    print(f"Processed {total_crops_processed} attention heatmaps")
    print(f"Open in browser: file://{output_path.resolve()}")


if __name__ == "__main__":
    assignments_df = pd.read_csv(CLUSTERS_DIR / "cluster_assignments.csv")
    session_index = pd.read_csv(EMBEDDINGS_DIR / "session_index.csv")
    crop_index = pd.read_csv(EMBEDDINGS_DIR / "crop_index_v3.csv")

    generate_explainability_gallery(
        assignments_df, session_index, crop_index,
        output_path=CLUSTERS_DIR / "explainability_gallery.html",
    )
