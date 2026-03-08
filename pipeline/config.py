# pipeline/config.py
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
DATA_DIR = PROJECT_ROOT / "data"
FRAMES_DIR = DATA_DIR / "frames"
CROPS_DIR = DATA_DIR / "crops"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
CLUSTERS_DIR = PROJECT_ROOT / "clusters"
MODELS_DIR = PROJECT_ROOT / "models"

# Frame extraction
FRAME_SKIP = 10
HIGH_FPS_SUFFIX = "_1"

# Detection
YOLO_MODEL = "yolov8n.pt"
BIRD_CLASS_ID = 14
DETECTION_CONFIDENCE = 0.5
BBOX_PADDING = 0.10
CROP_SIZE = (224, 224)

# Embedding
DINO_MODEL = "dinov2_vits14"
EMBED_DIM = 384
EMBED_BATCH_SIZE = 32

# Within-video grouping
GROUPING_DISTANCE_THRESHOLD = 0.3

# Species
NUM_SPECIES = 7

# Clustering
PCA_COMPONENTS = 50
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 3

# Inference
SIMILARITY_THRESHOLD = 0.85

# Cluster refinement
MERGE_SIMILARITY_THRESHOLD = 0.9
SPLIT_VARIANCE_THRESHOLD = 2.0
