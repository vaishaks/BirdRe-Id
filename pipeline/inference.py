# pipeline/inference.py
"""Inference pipeline: identify birds in a new video."""
import json
import numpy as np
import pandas as pd
import torch
import faiss
import cv2
from PIL import Image
from torchvision import transforms
from pathlib import Path
from ultralytics import YOLO
from pipeline.config import (
    YOLO_MODEL, BIRD_CLASS_ID, DETECTION_CONFIDENCE,
    BBOX_PADDING, CROP_SIZE, DINO_MODEL, EMBED_DIM,
    FRAME_SKIP, SIMILARITY_THRESHOLD, MODELS_DIR,
    GROUPING_DISTANCE_THRESHOLD,
)
from pipeline.filter import (
    MIN_CONFIDENCE, MIN_BBOX_AREA, ASPECT_RATIO_RANGE,
)
from pipeline.self_train import ProjectionHead, ContrastiveModel
from pipeline.group import group_crops_in_video


def load_inference_model(models_dir: Path = MODELS_DIR):
    """Load the full inference model: DINOv2 + projection head + FAISS index."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # DINOv2 backbone
    backbone = torch.hub.load("facebookresearch/dinov2", DINO_MODEL)
    backbone.eval()

    # Projection head
    checkpoint = torch.load(models_dir / "contrastive_head.pt", weights_only=True)
    proj_head = ProjectionHead(**checkpoint["config"])
    proj_head.load_state_dict(checkpoint["proj_head_state_dict"])

    model = ContrastiveModel(backbone, proj_head, freeze_backbone=True)
    model = model.to(device)
    model.eval()

    # FAISS index
    index = faiss.read_index(str(models_dir / "faiss_index.bin"))

    # Bird ID mapping
    with open(models_dir / "faiss_bird_ids.json") as f:
        bird_ids = json.load(f)

    # Bird database
    with open(models_dir / "bird_database.json") as f:
        database = json.load(f)

    # YOLO detector
    detector = YOLO(YOLO_MODEL)

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return {
        "model": model,
        "detector": detector,
        "index": index,
        "bird_ids": bird_ids,
        "database": database,
        "transform": transform,
        "device": device,
    }


def extract_frames(video_path: str, frame_skip: int = FRAME_SKIP) -> list[np.ndarray]:
    """Extract frames from video at regular intervals."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % frame_skip == 0:
            frames.append(frame)
        frame_num += 1
    cap.release()
    return frames


def detect_and_crop(detector, frames: list[np.ndarray]) -> list[dict]:
    """Detect birds and extract crops from frames."""
    crops = []
    for i, frame in enumerate(frames):
        results = detector(frame, verbose=False)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id != BIRD_CLASS_ID or conf < DETECTION_CONFIDENCE:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                h, w = frame.shape[:2]

                # Quality filter: confidence
                if conf < MIN_CONFIDENCE:
                    continue

                # Expand bbox
                bw, bh = x2 - x1, y2 - y1

                # Quality filter: bbox area
                if bw * bh < MIN_BBOX_AREA:
                    continue

                # Quality filter: aspect ratio
                aspect = bw / max(bh, 1)
                if aspect < ASPECT_RATIO_RANGE[0] or aspect > ASPECT_RATIO_RANGE[1]:
                    continue

                pad_x = bw * BBOX_PADDING
                pad_y = bh * BBOX_PADDING
                x1 = max(0, int(x1 - pad_x))
                y1 = max(0, int(y1 - pad_y))
                x2 = min(w, int(x2 + pad_x))
                y2 = min(h, int(y2 + pad_y))

                crop = frame[y1:y2, x1:x2]
                crop_resized = cv2.resize(crop, CROP_SIZE)
                crop_pil = Image.fromarray(cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB))

                crops.append({
                    "frame_idx": i,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "image": crop_pil,
                })

    return crops


def embed_crops_flip_invariant(model, transform, crops: list[dict], device: str) -> np.ndarray:
    """Compute flip-invariant embeddings for detected crops."""
    if not crops:
        return np.array([])

    all_embeddings = []
    batch_size = 32

    for i in range(0, len(crops), batch_size):
        batch = crops[i:i + batch_size]
        originals = []
        flipped = []
        for c in batch:
            img = c["image"]
            originals.append(transform(img))
            flipped.append(transform(img.transpose(Image.FLIP_LEFT_RIGHT)))

        orig_t = torch.stack(originals).to(device)
        flip_t = torch.stack(flipped).to(device)

        with torch.no_grad():
            emb_orig = model.embed(orig_t)
            emb_flip = model.embed(flip_t)

        avg = (emb_orig + emb_flip) / 2.0
        avg = avg / avg.norm(dim=1, keepdim=True)
        all_embeddings.append(avg.cpu().numpy())

    return np.vstack(all_embeddings)


def identify_birds(
    video_path: str,
    inference_bundle: dict,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict]:
    """Full inference pipeline for a single video.

    Returns list of identified birds with:
    - bird_id (or "new_candidate" if below threshold)
    - species
    - similarity score
    - number of crops
    - frame indices
    """
    model = inference_bundle["model"]
    detector = inference_bundle["detector"]
    index = inference_bundle["index"]
    bird_ids = inference_bundle["bird_ids"]
    database = inference_bundle["database"]
    transform = inference_bundle["transform"]
    device = inference_bundle["device"]

    # Step 1: Extract frames
    frames = extract_frames(video_path)
    if not frames:
        return []

    # Step 2: Detect and crop birds
    crops = detect_and_crop(detector, frames)
    if not crops:
        return []

    # Step 3: Embed with flip-invariance
    embeddings = embed_crops_flip_invariant(model, transform, crops, device)
    if len(embeddings) == 0:
        return []

    # Step 4: Within-video grouping
    crop_ids = [f"inf_crop_{i}" for i in range(len(crops))]
    groups = group_crops_in_video(
        crop_ids, embeddings,
        distance_threshold=GROUPING_DISTANCE_THRESHOLD,
    )

    # Step 5: For each group, compute session embedding and query FAISS
    results = []
    for group in groups:
        group_indices = [int(cid.split("_")[-1]) for cid in group["crop_ids"]]
        group_embs = embeddings[group_indices]

        # Session embedding: mean, L2-normalized
        session_emb = group_embs.mean(axis=0)
        session_emb /= np.linalg.norm(session_emb)
        query = session_emb.reshape(1, -1).astype(np.float32)

        # FAISS search
        similarities, indices = index.search(query, k=3)
        top_sim = float(similarities[0, 0])
        top_idx = int(indices[0, 0])

        if top_sim >= similarity_threshold:
            matched_bird_id = bird_ids[top_idx]
            species = database[matched_bird_id]["species"]
        else:
            matched_bird_id = "new_candidate"
            # Use closest match species as best guess
            species = database[bird_ids[top_idx]]["species"] + " (uncertain)"

        # Collect frame indices for this group
        frame_indices = sorted(set(crops[i]["frame_idx"] for i in group_indices))

        results.append({
            "bird_id": matched_bird_id,
            "species": species,
            "similarity": round(top_sim, 4),
            "n_crops": len(group_indices),
            "frame_indices": frame_indices,
            "top_3_matches": [
                {
                    "bird_id": bird_ids[int(indices[0, k])],
                    "similarity": round(float(similarities[0, k]), 4),
                }
                for k in range(min(3, len(bird_ids)))
            ],
        })

    return results


def run_inference(video_path: str):
    """Run inference on a video and print results."""
    print(f"Loading inference model...")
    bundle = load_inference_model()

    print(f"\nProcessing: {video_path}")
    results = identify_birds(video_path, bundle)

    if not results:
        print("No birds detected in video.")
        return results

    print(f"\nIdentified {len(results)} bird(s):\n")
    for i, r in enumerate(results):
        status = "MATCHED" if r["bird_id"] != "new_candidate" else "NEW?"
        print(f"  Bird {i+1}: [{status}] {r['bird_id']}")
        print(f"    Species: {r['species']}")
        print(f"    Similarity: {r['similarity']:.4f}")
        print(f"    Crops: {r['n_crops']}, Frames: {r['frame_indices']}")
        print(f"    Top 3 matches:")
        for m in r["top_3_matches"]:
            print(f"      {m['bird_id']}: {m['similarity']:.4f}")
        print()

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.inference <video_path>")
        sys.exit(1)
    run_inference(sys.argv[1])
