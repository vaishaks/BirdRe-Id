# Bird Re-Identification

Unsupervised visual re-identification pipeline for identifying individual birds from motion-triggered bird feeder camera videos.

## Goal

Given high-FPS MP4 videos from a bird feeder camera, identify which individual birds visit the feeder and track them across multiple visits — without requiring labeled training data.

## Species

7 known species with imbalanced distribution:
- Dark-eyed Junco
- House Finch
- Chestnut-backed Chickadee
- Golden-crowned Sparrow
- Black-capped Chickadee
- Song Sparrow
- House Sparrow

## Pipeline

1. **Ingest** — Extract every 10th frame from high-FPS videos, deduplicate by file hash
2. **Detect** — YOLO bird detection, crop and standardize to 224×224
3. **Embed** — DINOv2-small (ViT-S/14) to produce 384-dim embeddings
4. **Group** — Agglomerative clustering to group detections of the same bird within a video
5. **Species** — k-means clustering for species proxy; minimal manual labeling (~7 clusters)
6. **Cluster** — PCA (384→50 dims) + per-species HDBSCAN for individual identity
7. **Refine** — Temporal consistency checks, centroid merging, variance-based splitting
8. **Identify** — FAISS index for fast similarity search; new videos assigned bird IDs (threshold: 0.85)

## Tech Stack

| Purpose | Tool |
|---|---|
| Vision embeddings | DINOv2-small (PyTorch, MPS) |
| Bird detection | YOLOv8 nano |
| Clustering | HDBSCAN |
| Similarity search | FAISS |
| Data | CSVs + numpy arrays |
| Compute | Apple M4 Mac Pro |

## Data

- `raw_data/` — 786 source MP4 video files (~5.3 GB)

## Status

Planning phase. Implementation not yet started.
