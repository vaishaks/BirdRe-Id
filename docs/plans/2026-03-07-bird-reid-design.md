# Bird Re-Identification System Design

## Problem

Unsupervised visual re-identification of individual birds from feeder camera videos. Learn an embedding where images of the same individual are close together, without labels.

## Data

- ~391 high-FPS MP4 videos (`*_1.mp4`) from motion-triggered feeder camera
- Two cameras per event; we use high-FPS (`_1`) only
- 7 known species: Dark-eyed Junco, House Finch, Chestnut-backed Chickadee, Golden-crowned Sparrow, Black-capped Chickadee, Song Sparrow, House Sparrow
- Some videos contain multiple birds (simultaneously or sequentially, camera may pan)
- Species distribution is imbalanced

## Approach: Simplified Hybrid Pipeline

DINOv2 embeddings + YOLO detection + embedding-based within-video grouping + species-aware HDBSCAN clustering. No ByteTrack, no UMAP for clustering (UMAP reserved for visualization only).

## Compute

Apple M4 Mac Pro, 16GB RAM. Use MPS acceleration. DINOv2-small (ViT-S/14, 384-dim) to fit memory.

---

## Pipeline Phases

### Phase 1: Data Ingestion

- Scan `raw_data/*_1.mp4`, deduplicate by file hash (handle `(1)` copies)
- Extract every 10th frame using OpenCV
- Save to `data/frames/<video_hash>/frame_NNNNN.jpg`
- Output: `data/frames_metadata.csv` — columns: `frame_id, video_id, frame_number, image_path`

### Phase 2: Bird Detection & Cropping

- YOLOv8n (nano) pretrained on COCO, filter class=bird (ID 14), confidence > 0.5
- Expand bbox by 10% padding, crop and resize to 224x224
- Save to `data/crops/<video_id>/crop_NNNNN.jpg`
- Output: `data/crops_metadata.csv` — columns: `crop_id, frame_id, video_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence, crop_path`

### Phase 3: Feature Extraction

- DINOv2-small (`dinov2_vits14`) via torch hub
- Extract CLS token (384-dim) per crop, batch size ~32 on MPS
- Output: `embeddings/crop_embeddings.npy` + `embeddings/crop_index.csv`

### Phase 4: Within-Video Grouping

- For each video, cluster its crop embeddings using agglomerative clustering with cosine distance threshold (~0.3)
- Each within-video cluster = one sub-session (one bird's appearance in that video)
- Session embedding = mean of crop embeddings per sub-session
- Output: `embeddings/session_embeddings.npy` + `embeddings/session_index.csv` — columns: `session_id, video_id, crop_ids`

### Phase 5: Species Classification

- Quick k-means (k=7) on all crop embeddings as rough species proxy
- Labeling UI (Jupyter notebook or HTML): display grid of sample crops per cluster, user types species name once per cluster
- Propagate labels to all crops in that cluster
- Output: `data/species_labels.csv` — columns: `crop_id, session_id, species`

### Phase 6: Species-Aware Clustering

- Per species: PCA 384 -> 50 dims on session embeddings
- Per species: HDBSCAN (tunable `min_cluster_size` per species — smaller for rare species)
- Each cluster = one individual bird
- Refinement:
  - Temporal consistency: flag sessions in same cluster that overlap in time
  - Merge clusters with centroid cosine similarity > 0.9
  - Split clusters with high intra-cluster variance (> 2 std devs)
- Output: `clusters/cluster_assignments.csv` — columns: `session_id, video_id, species, bird_id`
- Output: `clusters/cluster_centroids.npy`

### Phase 7: Bird Identity Database & Inference

- Per bird_id: store representative embedding (centroid), example crop paths, visit history (session IDs + timestamps)
- FAISS index (IndexFlatIP on L2-normalized vectors) over cluster centroids
- New video inference: detect -> crop -> embed -> within-video group -> PCA transform -> FAISS nearest neighbor -> assign bird_id if cosine sim > 0.85, else flag as new candidate
- Output: `models/pca_model.pkl`, `models/faiss_index.bin`, `models/bird_database.json`

### Phase 8 (Later): Visualization

- UMAP 2D projection of session embeddings
- Color by bird_id and/or species
- Interactive plots for exploration

### Phase 9 (Optional): Self-Training

- Use cluster assignments as pseudo-labels
- Contrastive fine-tuning (SimCLR/MoCo)
- Recompute embeddings, recluster, repeat until stable

---

## Directory Structure

```
BirdRe-Id/
  raw_data/              # source MP4 videos
  data/
    frames/              # extracted frames per video
    crops/               # bird crops per video
    frames_metadata.csv
    crops_metadata.csv
    species_labels.csv
  embeddings/
    crop_embeddings.npy
    crop_index.csv
    session_embeddings.npy
    session_index.csv
  clusters/
    cluster_assignments.csv
    cluster_centroids.npy
  models/
    pca_model.pkl
    faiss_index.bin
    bird_database.json
  pipeline/
    ingest.py
    detect.py
    embed.py
    group.py
    label_species.py
    cluster.py
    inference.py
  docs/plans/
```

## Libraries

```
torch
ultralytics
opencv-python
scikit-learn
hdbscan
faiss-cpu
umap-learn
pandas
numpy
jupyter
```

## Key Design Decisions

1. High-FPS camera only — better temporal resolution for grouping
2. No ByteTrack — embedding-based within-video grouping handles multi-bird and camera-pan scenarios
3. Species-first clustering — prevents imbalanced species from distorting individual-level clustering
4. Minimal manual effort — label ~7 species clusters, everything else is unsupervised
5. File-based storage — CSVs + numpy + pickle, no database server needed at this scale
6. UMAP for visualization only, not in the clustering pipeline
