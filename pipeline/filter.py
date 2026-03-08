# pipeline/filter.py
"""Crop quality filtering and flip-invariant embedding computation."""
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn.decomposition import PCA
from pipeline.config import EMBEDDINGS_DIR, MODELS_DIR, EMBED_DIM

# Quality thresholds
MIN_CONFIDENCE = 0.65
MIN_BBOX_AREA = 200 * 200  # minimum bbox area in pixels
ASPECT_RATIO_RANGE = (0.4, 2.5)  # reject extremely elongated crops

# Orientation filtering
ORIENTATION_OUTLIER_PERCENTILE = 5  # drop crops at extremes of PC0 (back/top views)


def filter_low_quality_crops(
    crops_df: pd.DataFrame,
    min_confidence: float = MIN_CONFIDENCE,
    min_bbox_area: float = MIN_BBOX_AREA,
    aspect_ratio_range: tuple = ASPECT_RATIO_RANGE,
) -> pd.DataFrame:
    """Remove low-quality crops: low confidence, too small, extreme aspect ratio."""
    n_before = len(crops_df)

    # Confidence filter
    mask = crops_df["confidence"] >= min_confidence

    # Bbox area filter
    bbox_w = crops_df["bbox_x2"] - crops_df["bbox_x1"]
    bbox_h = crops_df["bbox_y2"] - crops_df["bbox_y1"]
    area = bbox_w * bbox_h
    mask &= area >= min_bbox_area

    # Aspect ratio filter (reject extremely elongated)
    aspect = bbox_w / bbox_h.clip(lower=1)
    mask &= (aspect >= aspect_ratio_range[0]) & (aspect <= aspect_ratio_range[1])

    filtered = crops_df[mask].reset_index(drop=True)
    n_after = len(filtered)
    print(f"Quality filter: {n_before} -> {n_after} crops "
          f"(removed {n_before - n_after}: "
          f"{(~(crops_df['confidence'] >= min_confidence)).sum()} low-conf, "
          f"{(area < min_bbox_area).sum()} small, "
          f"{((aspect < aspect_ratio_range[0]) | (aspect > aspect_ratio_range[1])).sum()} bad-aspect)")
    return filtered


def compute_flip_invariant_embeddings(
    model,
    crop_paths: list[str],
    batch_size: int = 32,
) -> np.ndarray:
    """Embed each crop as average of original + horizontally flipped."""
    device = next(model.parameters()).device
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_embeddings = []

    for i in tqdm(range(0, len(crop_paths), batch_size), desc="Flip-invariant embedding"):
        batch_paths = crop_paths[i:i + batch_size]
        originals = []
        flipped = []

        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            originals.append(transform(img))
            flipped.append(transform(img.transpose(Image.FLIP_LEFT_RIGHT)))

        orig_batch = torch.stack(originals).to(device)
        flip_batch = torch.stack(flipped).to(device)

        with torch.no_grad():
            emb_orig = model.embed(orig_batch) if hasattr(model, 'embed') else model(orig_batch)
            emb_flip = model.embed(flip_batch) if hasattr(model, 'embed') else model(flip_batch)

        # Average original + flipped
        avg_emb = (emb_orig + emb_flip) / 2.0
        # Re-normalize
        avg_emb = avg_emb / avg_emb.norm(dim=1, keepdim=True)
        all_embeddings.append(avg_emb.cpu().numpy())

    return np.vstack(all_embeddings)


def filter_orientation_outliers(
    embeddings: np.ndarray,
    crop_ids: list[str],
    video_ids: list[str],
    percentile: float = ORIENTATION_OUTLIER_PERCENTILE,
) -> tuple[np.ndarray, list[str]]:
    """Remove crops at extremes of top PCA component (back/top-down views).

    Operates per-species-group (approximated by per-video to avoid needing labels).
    Uses global PCA to find orientation axis, then removes crops at the extremes.
    """
    n_before = len(embeddings)

    # Global PCA to find orientation axis
    pca = PCA(n_components=min(10, embeddings.shape[0] - 1, embeddings.shape[1]))
    transformed = pca.fit_transform(embeddings)

    # PC0 is typically orientation — remove extreme values
    pc0 = transformed[:, 0]
    low_cut = np.percentile(pc0, percentile)
    high_cut = np.percentile(pc0, 100 - percentile)
    keep_mask = (pc0 >= low_cut) & (pc0 <= high_cut)

    filtered_embs = embeddings[keep_mask]
    filtered_ids = [cid for cid, k in zip(crop_ids, keep_mask) if k]

    n_after = len(filtered_embs)
    print(f"Orientation filter: {n_before} -> {n_after} crops "
          f"(removed {n_before - n_after} extreme-orientation crops, "
          f"PC0 explains {pca.explained_variance_ratio_[0]*100:.1f}% variance)")

    return filtered_embs, filtered_ids


def run_filtering(
    crops_df: pd.DataFrame,
    model,
    embeddings_dir=EMBEDDINGS_DIR,
):
    """Full filtering pipeline: quality -> flip-invariant embed -> orientation filter."""
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Step C: Quality filtering
    print("\n=== Step C: Quality Filtering ===")
    filtered_crops = filter_low_quality_crops(crops_df)

    # Step A: Flip-invariant embeddings
    print("\n=== Step A: Flip-Invariant Embeddings ===")
    crop_paths = filtered_crops["crop_path"].tolist()
    embeddings = compute_flip_invariant_embeddings(model, crop_paths)

    # Step B: Orientation filtering
    print("\n=== Step B: Orientation Filtering ===")
    crop_ids = filtered_crops["crop_id"].tolist()
    video_ids = filtered_crops["video_id"].tolist()
    embeddings, kept_crop_ids = filter_orientation_outliers(
        embeddings, crop_ids, video_ids,
    )

    # Build filtered crop index
    kept_set = set(kept_crop_ids)
    final_crops = filtered_crops[filtered_crops["crop_id"].isin(kept_set)].reset_index(drop=True)

    # Ensure embeddings align with final_crops order
    crop_id_to_emb = dict(zip(kept_crop_ids, embeddings))
    aligned_embeddings = np.array([crop_id_to_emb[cid] for cid in final_crops["crop_id"]])

    # Save
    np.save(embeddings_dir / "crop_embeddings_v3.npy", aligned_embeddings)
    index_df = final_crops[["crop_id", "frame_id", "video_id", "crop_path"]].copy()
    index_df.to_csv(embeddings_dir / "crop_index_v3.csv", index=False)

    print(f"\n=== Final: {len(aligned_embeddings)} crops, {aligned_embeddings.shape[1]}-dim embeddings ===")
    return aligned_embeddings, index_df
