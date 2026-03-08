# Feeder Friends — Bird Re-Identification

Unsupervised individual bird re-identification from feeder camera videos. Learns visual embeddings where images of the same bird are close together, without any labels. Identifies returning individuals across visits using DINOv2 + contrastive learning + clustering.

## Quick Start

### Identify birds in a new video

```bash
python identify.py video.mp4
```

Output:
```
Bird 1: [✓] Scarlet
  Species:    House Finch
  Confidence: 93%
  Crops:      36
```

### More options

```bash
# Process a directory of videos
python identify.py /path/to/videos/

# Adjust matching strictness (lower = more matches)
python identify.py video.mp4 --threshold 0.6

# JSON output for scripting
python identify.py video.mp4 --json > results.json
```

## Requirements

- Python 3.11+
- Apple Silicon Mac recommended (MPS acceleration), also works on CPU/CUDA
- ~4 GB disk for models and embeddings

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies: PyTorch, ultralytics (YOLOv8), DINOv2 (via torch hub), scikit-learn, FAISS, UMAP, hdbscan, OpenCV, pandas, numpy, matplotlib.

## Full Pipeline

Run these steps in order to build the bird database from raw videos. Each step is idempotent — re-running overwrites previous outputs.

### 1. Data Ingestion — Extract frames from videos

```bash
python -m pipeline.ingest
```

Scans `raw_data/*_1.mp4` (high-FPS camera), deduplicates by MD5 hash, extracts every 10th frame.

**Output:** `data/frames/`, `data/frames_metadata.csv`

### 2. Bird Detection — Crop birds from frames

```bash
python -m pipeline.detect
```

Runs YOLOv8-nano pretrained on COCO, filters for bird class (ID 14), confidence > 0.5. Expands bounding boxes by 10% and resizes crops to 224x224.

**Output:** `data/crops/`, `data/crops_metadata.csv`

### 3. Feature Extraction — DINOv2 embeddings

```bash
python -m pipeline.embed
```

Extracts 384-dim CLS token embeddings from DINOv2-small (ViT-S/14) for each crop. L2-normalized.

**Output:** `embeddings/crop_embeddings.npy`, `embeddings/crop_index.csv`

### 4. Within-Video Grouping — Group crops into sessions

```bash
python -m pipeline.group
```

Clusters crops within each video by cosine similarity (agglomerative, threshold 0.3). Each cluster = one bird's appearance in that video. Session embedding = mean of crop embeddings.

**Output:** `embeddings/session_embeddings.npy`, `embeddings/session_index.csv`

### 5. Species Labeling — Manual labeling via HTML UI

```bash
python -m pipeline.label_species
```

Runs k-means (k=20) on crop embeddings, generates an HTML page with sample crop grids per cluster. You label each cluster with a species name, then the script propagates labels to all crops and sessions.

**Output:** `data/species_labels.csv`, `data/species_labeling.html`

### 6. Self-Training — Contrastive learning for better embeddings

```bash
python -m pipeline.self_train
```

Trains a projection head (384 -> 256 -> 128) on top of frozen DINOv2 using NT-Xent contrastive loss. Positive pairs = two random crops from the same session (same bird, different poses). 20 epochs. Then re-embeds all crops and re-runs grouping.

**Output:** `embeddings/crop_embeddings_v2.npy`, `models/contrastive_head.pt`

### 7. Quality Filtering — Remove noise and normalize orientation

```bash
python -c "
from pipeline.filter import run_filtering
from pipeline.self_train import ProjectionHead, ContrastiveModel
from pipeline.config import MODELS_DIR
import torch, pandas as pd

backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
backbone.eval()
cp = torch.load(MODELS_DIR / 'contrastive_head.pt', weights_only=True)
proj = ProjectionHead(**cp['config'])
proj.load_state_dict(cp['proj_head_state_dict'])
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = ContrastiveModel(backbone, proj, freeze_backbone=True).to(device)

crops_df = pd.read_csv('data/crops_metadata.csv')
run_filtering(crops_df, model)
"
```

Three filters:
- **Quality:** removes low-confidence detections (< 0.65), small crops, extreme aspect ratios
- **Flip-invariant embeddings:** averages original + horizontally flipped to eliminate left/right bias
- **Orientation filtering:** removes crops at PCA extremes (back/top-down views that lack identity features)

Then re-run grouping and apply species labels:

```bash
python -m pipeline.group  # uses crop_embeddings_v3 / crop_index_v3
```

**Output:** `embeddings/crop_embeddings_v3.npy`, `embeddings/crop_index_v3.csv`

### 8. Clustering — Identify individuals

```bash
python -m pipeline.cluster
```

Per-species agglomerative clustering on session embeddings. For large groups (>= 30 sessions): PCA to 30 dims, drop top 3 PCs (pose/orientation), cosine threshold 0.8. For small groups: raw embeddings, threshold 0.35.

**Output:** `clusters/cluster_assignments.csv`, `clusters/cluster_centroids.npy`

### 9. Build Identity Database

```bash
python -m pipeline.identity_db
```

Creates a FAISS index (inner product on L2-normalized 128-dim vectors) over cluster centroids, plus a JSON database with per-bird metadata.

**Output:** `models/faiss_index.bin`, `models/bird_database.json`

### 10. Explainability — Attention heatmaps

```bash
python -m pipeline.explain
```

Extracts DINOv2 CLS-token attention maps from the last transformer block. Generates overlay heatmaps showing which image regions the model focuses on for identity.

**Output:** `clusters/heatmaps/`, `clusters/explainability_gallery.html`

### 11. Dashboard

```bash
python -m pipeline.dashboard
open clusters/dashboard.html
```

Interactive HTML dashboard ("Feeder Friends") with bird photo cards, species filters, detail overlays with photo galleries, attention heatmap toggles, co-occurring bird networks, and a collapsible UMAP scatter plot.

## Project Structure

```
BirdRe-Id/
├── identify.py              # Quick inference script
├── requirements.txt
├── pipeline/
│   ├── config.py            # All paths and hyperparameters
│   ├── ingest.py            # Frame extraction
│   ├── detect.py            # YOLOv8 bird detection
│   ├── embed.py             # DINOv2 feature extraction
│   ├── group.py             # Within-video crop grouping
│   ├── label_species.py     # Species labeling UI
│   ├── self_train.py        # Contrastive self-training
│   ├── filter.py            # Quality + orientation filtering
│   ├── cluster.py           # Species-aware clustering
│   ├── identity_db.py       # FAISS index + bird database
│   ├── inference.py         # Full video inference pipeline
│   ├── explain.py           # Attention heatmap explainability
│   ├── visualize.py         # UMAP plots + cluster gallery
│   ├── dashboard.py         # Interactive HTML dashboard
│   └── bird_names.py        # Cute name assignment
├── raw_data/                # Source MP4 videos
├── data/
│   ├── frames/              # Extracted frames per video
│   ├── crops/               # Bird crops per video
│   ├── frames_metadata.csv
│   ├── crops_metadata.csv
│   └── species_labels.csv
├── embeddings/
│   ├── crop_embeddings.npy      # Original DINOv2 (384-dim)
│   ├── crop_embeddings_v2.npy   # Self-trained (128-dim)
│   ├── crop_embeddings_v3.npy   # Filtered + flip-invariant
│   ├── session_embeddings.npy
│   ├── crop_index.csv
│   └── session_index.csv
├── clusters/
│   ├── cluster_assignments.csv
│   ├── cluster_centroids.npy
│   ├── dashboard.html
│   ├── heatmaps/
│   └── *.png
├── models/
│   ├── contrastive_head.pt
│   ├── faiss_index.bin
│   ├── bird_database.json
│   └── pca_models.pkl
├── tests/
└── docs/plans/
```

## How It Works

1. **Detect** birds in video frames using YOLOv8
2. **Embed** each crop using DINOv2 (self-supervised vision transformer)
3. **Self-train** a projection head with contrastive learning to make embeddings pose-invariant
4. **Filter** low-quality crops and normalize orientation (flip-invariant embeddings)
5. **Group** crops within each video into sessions (one bird per session)
6. **Cluster** session embeddings per species to discover individuals
7. **Index** cluster centroids with FAISS for fast nearest-neighbor inference

## Key Design Decisions

- **DINOv2-small** (ViT-S/14, 384-dim) — strong visual features, fits in 16 GB RAM
- **Embedding-based grouping** instead of spatial tracking — handles camera panning and multi-bird scenes
- **Species-first clustering** — prevents dominant species from distorting individual-level clusters
- **Pose correction** — dropping top PCA components that capture orientation instead of identity
- **Flip-invariant embeddings** — averaging original + horizontally flipped eliminates left/right bias
- **Contrastive self-training** — learns to ignore pose variation using session-based positive pairs
- **File-based storage** — CSVs, numpy arrays, pickle, JSON. No database server needed

## Configuration

All hyperparameters are in `pipeline/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FRAME_SKIP` | 10 | Extract every Nth frame |
| `DETECTION_CONFIDENCE` | 0.5 | YOLO confidence threshold |
| `BBOX_PADDING` | 0.10 | Bounding box expansion ratio |
| `GROUPING_DISTANCE_THRESHOLD` | 0.3 | Cosine distance for within-video grouping |
| `SIMILARITY_THRESHOLD` | 0.50 | FAISS match threshold for inference |

## Tests

```bash
python -m pytest tests/ -v
```
