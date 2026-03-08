# Bird Re-Identification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an unsupervised pipeline that identifies individual birds from feeder camera videos using DINOv2 embeddings and species-aware clustering.

**Architecture:** YOLO detects birds in video frames, DINOv2 encodes crops into 384-dim embeddings, agglomerative clustering groups crops within videos, species are labeled via a simple UI, and HDBSCAN clusters individuals per species. FAISS enables inference on new videos.

**Tech Stack:** Python 3.13, PyTorch (MPS), ultralytics (YOLOv8), DINOv2 (torch hub), scikit-learn, hdbscan, faiss-cpu, OpenCV, pandas, numpy, Jupyter

---

## Milestone 1: Project Setup & Dependencies

### Task 1: Initialize project and install dependencies

**Files:**
- Create: `requirements.txt`
- Create: `pipeline/__init__.py`
- Create: `pipeline/config.py`

**Step 1: Create requirements.txt**

```txt
torch
torchvision
ultralytics
opencv-python
scikit-learn
hdbscan
faiss-cpu
umap-learn
pandas
numpy
jupyter
matplotlib
Pillow
tqdm
```

**Step 2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully

**Step 3: Create config module**

```python
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
FRAME_SKIP = 10  # extract every Nth frame
HIGH_FPS_SUFFIX = "_1"  # high-FPS camera identifier

# Detection
YOLO_MODEL = "yolov8n.pt"
BIRD_CLASS_ID = 14  # COCO bird class
DETECTION_CONFIDENCE = 0.5
BBOX_PADDING = 0.10  # 10% padding
CROP_SIZE = (224, 224)

# Embedding
DINO_MODEL = "dinov2_vits14"
EMBED_DIM = 384
EMBED_BATCH_SIZE = 32

# Within-video grouping
GROUPING_DISTANCE_THRESHOLD = 0.3  # cosine distance

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
SPLIT_VARIANCE_THRESHOLD = 2.0  # std devs
```

**Step 4: Create __init__.py**

```python
# pipeline/__init__.py
```

**Step 5: Create directory structure**

Run: `mkdir -p data/frames data/crops embeddings clusters models pipeline tests`

**Step 6: Verify setup**

Run: `python -c "import torch; import ultralytics; import cv2; import sklearn; import hdbscan; import faiss; print('All imports OK'); print(f'MPS available: {torch.backends.mps.is_available()}')"`
Expected: "All imports OK" and "MPS available: True"

**Step 7: Commit**

```bash
git init
git add requirements.txt pipeline/config.py pipeline/__init__.py
git commit -m "feat: initialize project with dependencies and config"
```

---

## Milestone 2: Data Ingestion (Phase 1)

**Pause after this milestone for user inspection.**

### Task 2: Write frame extraction tests

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_ingest.py`

**Step 1: Write tests**

```python
# tests/test_ingest.py
import tempfile
import os
from pathlib import Path
import numpy as np
import cv2
import pandas as pd


def create_test_video(path, num_frames=30, fps=30, width=640, height=480):
    """Create a minimal test video."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (i * 8 % 256, 100, 100)
        writer.write(frame)
    writer.release()


def test_find_high_fps_videos():
    from pipeline.ingest import find_high_fps_videos

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        (Path(tmpdir) / "abc_1.mp4").touch()
        (Path(tmpdir) / "abc.mp4").touch()
        (Path(tmpdir) / "def_1.MP4").touch()
        (Path(tmpdir) / "def.MP4").touch()
        (Path(tmpdir) / "ghi_1 (1).mp4").touch()  # duplicate

        videos = find_high_fps_videos(Path(tmpdir))
        # Should find _1 files, case insensitive, no (1) dupes
        assert len(videos) >= 2
        assert all("_1" in v.stem for v in videos)


def test_deduplicate_videos():
    from pipeline.ingest import deduplicate_videos

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two identical files
        content = b"fake video content"
        p1 = Path(tmpdir) / "abc_1.mp4"
        p2 = Path(tmpdir) / "abc_1 (1).mp4"
        p1.write_bytes(content)
        p2.write_bytes(content)
        # Create a different file
        p3 = Path(tmpdir) / "def_1.mp4"
        p3.write_bytes(b"different content")

        unique = deduplicate_videos([p1, p2, p3])
        assert len(unique) == 2


def test_extract_frames():
    from pipeline.ingest import extract_frames

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "test_1.mp4"
        create_test_video(video_path, num_frames=30)
        output_dir = Path(tmpdir) / "frames"

        frames = extract_frames(video_path, output_dir, frame_skip=10)
        assert len(frames) == 3  # frames 0, 10, 20
        assert all(Path(f["image_path"]).exists() for f in frames)
        assert all("frame_id" in f for f in frames)
        assert all("video_id" in f for f in frames)


def test_ingest_pipeline():
    from pipeline.ingest import run_ingestion

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw_data"
        raw_dir.mkdir()
        create_test_video(raw_dir / "vid1_1.mp4", num_frames=20)
        create_test_video(raw_dir / "vid2_1.mp4", num_frames=20)
        (raw_dir / "vid1.mp4").touch()  # non-high-fps, should be ignored

        output_dir = Path(tmpdir) / "data"
        metadata_df = run_ingestion(raw_dir, output_dir / "frames", frame_skip=10)

        assert isinstance(metadata_df, pd.DataFrame)
        assert len(metadata_df) > 0
        assert set(metadata_df.columns) >= {"frame_id", "video_id", "frame_number", "image_path"}
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.ingest'`

### Task 3: Implement frame extraction

**Files:**
- Create: `pipeline/ingest.py`

**Step 1: Implement ingest module**

```python
# pipeline/ingest.py
import hashlib
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm
from pipeline.config import FRAME_SKIP, HIGH_FPS_SUFFIX


def find_high_fps_videos(raw_dir: Path) -> list[Path]:
    """Find all high-FPS camera videos (*_1.mp4)."""
    videos = []
    for ext in ("*.mp4", "*.MP4"):
        for p in raw_dir.glob(ext):
            stem = p.stem
            # Skip (1) duplicates
            if stem.endswith(" (1)") or stem.endswith("(1)"):
                continue
            if HIGH_FPS_SUFFIX in stem:
                videos.append(p)
    return sorted(videos)


def _file_hash(path: Path) -> str:
    """Compute MD5 hash of file contents."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def deduplicate_videos(videos: list[Path]) -> list[Path]:
    """Remove duplicate videos by file hash."""
    seen = {}
    unique = []
    for v in videos:
        h = _file_hash(v)
        if h not in seen:
            seen[h] = v
            unique.append(v)
    return unique


def _video_id_from_path(path: Path) -> str:
    """Extract video ID from filename (hash portion before _1)."""
    stem = path.stem
    idx = stem.find(HIGH_FPS_SUFFIX)
    if idx != -1:
        return stem[:idx]
    return stem


def extract_frames(
    video_path: Path,
    output_dir: Path,
    frame_skip: int = FRAME_SKIP,
) -> list[dict]:
    """Extract every Nth frame from a video."""
    video_id = _video_id_from_path(video_path)
    vid_output_dir = output_dir / video_id
    vid_output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: Could not open {video_path}")
        return []

    frames = []
    frame_num = 0
    extracted = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % frame_skip == 0:
            frame_id = f"{video_id}_f{frame_num:05d}"
            filename = f"frame_{extracted:05d}.jpg"
            frame_path = vid_output_dir / filename
            cv2.imwrite(str(frame_path), frame)
            frames.append({
                "frame_id": frame_id,
                "video_id": video_id,
                "frame_number": frame_num,
                "image_path": str(frame_path),
            })
            extracted += 1
        frame_num += 1

    cap.release()
    return frames


def run_ingestion(
    raw_dir: Path,
    frames_dir: Path,
    frame_skip: int = FRAME_SKIP,
) -> pd.DataFrame:
    """Run the full ingestion pipeline."""
    print("Finding high-FPS videos...")
    videos = find_high_fps_videos(raw_dir)
    print(f"Found {len(videos)} high-FPS videos")

    print("Deduplicating...")
    videos = deduplicate_videos(videos)
    print(f"{len(videos)} unique videos after dedup")

    frames_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []
    for video_path in tqdm(videos, desc="Extracting frames"):
        frames = extract_frames(video_path, frames_dir, frame_skip)
        all_frames.extend(frames)

    df = pd.DataFrame(all_frames)
    metadata_path = frames_dir.parent / "frames_metadata.csv"
    df.to_csv(metadata_path, index=False)
    print(f"Extracted {len(df)} frames from {len(videos)} videos")
    print(f"Metadata saved to {metadata_path}")
    return df


if __name__ == "__main__":
    from pipeline.config import RAW_DATA_DIR, FRAMES_DIR
    run_ingestion(RAW_DATA_DIR, FRAMES_DIR)
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_ingest.py -v`
Expected: All 4 tests PASS

**Step 3: Commit**

```bash
git add pipeline/ingest.py tests/test_ingest.py tests/__init__.py
git commit -m "feat: add frame extraction pipeline with deduplication"
```

### Task 4: Run ingestion on real data

**Step 1: Run the pipeline**

Run: `python -m pipeline.ingest`
Expected: Extracts frames from ~391 unique videos into `data/frames/`

**Step 2: Inspect output**

Run: `wc -l data/frames_metadata.csv && head -5 data/frames_metadata.csv && ls data/frames/ | head -10`
Expected: CSV with thousands of rows, frame images in subdirectories

**Step 3: Commit data metadata (not images)**

```bash
git add data/frames_metadata.csv
echo "data/frames/" >> .gitignore
echo "data/crops/" >> .gitignore
echo "embeddings/*.npy" >> .gitignore
echo "models/" >> .gitignore
echo "raw_data/" >> .gitignore
git add .gitignore
git commit -m "feat: run ingestion, add frames metadata and gitignore"
```

**PAUSE: Inspect frame extraction results before continuing.**

---

## Milestone 3: Bird Detection & Cropping (Phase 2)

**Pause after this milestone for user inspection.**

### Task 5: Write detection tests

**Files:**
- Create: `tests/test_detect.py`

**Step 1: Write tests**

```python
# tests/test_detect.py
import tempfile
from pathlib import Path
import numpy as np
import cv2
import pandas as pd


def create_test_frame(path, width=640, height=480):
    """Create a test image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[100:300, 200:400] = (0, 150, 200)  # colored rectangle
    cv2.imwrite(str(path), img)


def test_expand_bbox():
    from pipeline.detect import expand_bbox

    bbox = (100, 100, 200, 200)
    expanded = expand_bbox(bbox, padding=0.1, img_width=640, img_height=480)
    x1, y1, x2, y2 = expanded
    assert x1 < 100
    assert y1 < 100
    assert x2 > 200
    assert y2 > 200
    # Should be clamped to image bounds
    assert x1 >= 0
    assert y1 >= 0
    assert x2 <= 640
    assert y2 <= 480


def test_crop_and_resize():
    from pipeline.detect import crop_and_resize

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[100:300, 200:400] = 255
    crop = crop_and_resize(img, (200, 100, 400, 300), size=(224, 224))
    assert crop.shape == (224, 224, 3)


def test_detect_returns_dataframe():
    from pipeline.detect import run_detection

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal frames metadata
        frames_dir = Path(tmpdir) / "frames" / "testvid"
        frames_dir.mkdir(parents=True)
        frame_path = frames_dir / "frame_00000.jpg"
        create_test_frame(frame_path)

        frames_df = pd.DataFrame([{
            "frame_id": "testvid_f00000",
            "video_id": "testvid",
            "frame_number": 0,
            "image_path": str(frame_path),
        }])

        crops_dir = Path(tmpdir) / "crops"
        result_df = run_detection(frames_df, crops_dir)

        assert isinstance(result_df, pd.DataFrame)
        expected_cols = {"crop_id", "frame_id", "video_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "confidence", "crop_path"}
        assert expected_cols <= set(result_df.columns)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_detect.py -v`
Expected: FAIL — `ModuleNotFoundError`

### Task 6: Implement detection

**Files:**
- Create: `pipeline/detect.py`

**Step 1: Implement detection module**

```python
# pipeline/detect.py
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from pipeline.config import (
    YOLO_MODEL, BIRD_CLASS_ID, DETECTION_CONFIDENCE,
    BBOX_PADDING, CROP_SIZE,
)


def expand_bbox(
    bbox: tuple[int, int, int, int],
    padding: float,
    img_width: int,
    img_height: int,
) -> tuple[int, int, int, int]:
    """Expand bounding box by padding fraction, clamped to image bounds."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(img_width, x2 + pad_x),
        min(img_height, y2 + pad_y),
    )


def crop_and_resize(
    img: np.ndarray,
    bbox: tuple[int, int, int, int],
    size: tuple[int, int] = CROP_SIZE,
) -> np.ndarray:
    """Crop image to bbox and resize."""
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, size)


def run_detection(
    frames_df: pd.DataFrame,
    crops_dir: Path,
) -> pd.DataFrame:
    """Run YOLO bird detection on all frames and save crops."""
    model = YOLO(YOLO_MODEL)
    crops_dir.mkdir(parents=True, exist_ok=True)

    all_crops = []
    crop_counter = 0

    for _, row in tqdm(frames_df.iterrows(), total=len(frames_df), desc="Detecting birds"):
        img = cv2.imread(row["image_path"])
        if img is None:
            continue

        h, w = img.shape[:2]
        results = model(img, verbose=False)

        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                conf = float(boxes.conf[i])

                if cls != BIRD_CLASS_ID or conf < DETECTION_CONFIDENCE:
                    continue

                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                expanded = expand_bbox(bbox, BBOX_PADDING, w, h)
                crop = crop_and_resize(img, expanded)

                video_id = row["video_id"]
                vid_crop_dir = crops_dir / video_id
                vid_crop_dir.mkdir(parents=True, exist_ok=True)

                crop_id = f"{video_id}_c{crop_counter:05d}"
                crop_filename = f"crop_{crop_counter:05d}.jpg"
                crop_path = vid_crop_dir / crop_filename
                cv2.imwrite(str(crop_path), crop)

                all_crops.append({
                    "crop_id": crop_id,
                    "frame_id": row["frame_id"],
                    "video_id": video_id,
                    "bbox_x1": expanded[0],
                    "bbox_y1": expanded[1],
                    "bbox_x2": expanded[2],
                    "bbox_y2": expanded[3],
                    "confidence": conf,
                    "crop_path": str(crop_path),
                })
                crop_counter += 1

    df = pd.DataFrame(all_crops)
    metadata_path = crops_dir.parent / "crops_metadata.csv"
    df.to_csv(metadata_path, index=False)
    print(f"Detected {len(df)} bird crops from {len(frames_df)} frames")
    print(f"Metadata saved to {metadata_path}")
    return df


if __name__ == "__main__":
    from pipeline.config import DATA_DIR, CROPS_DIR
    frames_df = pd.read_csv(DATA_DIR / "frames_metadata.csv")
    run_detection(frames_df, CROPS_DIR)
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_detect.py -v`
Expected: All 3 tests PASS (note: test_detect_returns_dataframe may return 0 crops since test image isn't a real bird — that's OK, we test the schema)

**Step 3: Commit**

```bash
git add pipeline/detect.py tests/test_detect.py
git commit -m "feat: add YOLO bird detection and cropping pipeline"
```

### Task 7: Run detection on real data

**Step 1: Run the pipeline**

Run: `python -m pipeline.detect`
Expected: Detects bird crops from frames, saves to `data/crops/`

**Step 2: Inspect output**

Run: `wc -l data/crops_metadata.csv && head -5 data/crops_metadata.csv`
Expected: CSV with crop records. Check a few crop images visually.

**Step 3: Commit metadata**

```bash
git add data/crops_metadata.csv
git commit -m "feat: run bird detection, save crop metadata"
```

**PAUSE: Inspect detection results — check some crop images to verify quality.**

---

## Milestone 4: Feature Extraction (Phase 3)

### Task 8: Write embedding tests

**Files:**
- Create: `tests/test_embed.py`

**Step 1: Write tests**

```python
# tests/test_embed.py
import tempfile
from pathlib import Path
import numpy as np
import cv2
import pandas as pd


def create_test_crop(path, size=224):
    """Create a test crop image."""
    img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_load_dino_model():
    from pipeline.embed import load_model

    model, transform = load_model()
    assert model is not None
    assert transform is not None


def test_embed_single_crop():
    from pipeline.embed import load_model, embed_crops

    with tempfile.TemporaryDirectory() as tmpdir:
        crop_path = Path(tmpdir) / "crop.jpg"
        create_test_crop(crop_path)

        model, transform = load_model()
        embeddings = embed_crops(model, transform, [str(crop_path)])
        assert embeddings.shape == (1, 384)


def test_embed_batch():
    from pipeline.embed import load_model, embed_crops

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for i in range(5):
            p = Path(tmpdir) / f"crop_{i}.jpg"
            create_test_crop(p)
            paths.append(str(p))

        model, transform = load_model()
        embeddings = embed_crops(model, transform, paths)
        assert embeddings.shape == (5, 384)
        # Embeddings should be L2-normalized
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_embed.py -v`
Expected: FAIL — `ModuleNotFoundError`

### Task 9: Implement embedding extraction

**Files:**
- Create: `pipeline/embed.py`

**Step 1: Implement embedding module**

```python
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
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_embed.py -v`
Expected: All 3 tests PASS (first run downloads DINOv2 model — may take a minute)

**Step 3: Commit**

```bash
git add pipeline/embed.py tests/test_embed.py
git commit -m "feat: add DINOv2 embedding extraction"
```

### Task 10: Run embedding on real data

**Step 1: Run the pipeline**

Run: `python -m pipeline.embed`
Expected: Embeds all crops, saves to `embeddings/crop_embeddings.npy`

**Step 2: Inspect output**

Run: `python -c "import numpy as np; e = np.load('embeddings/crop_embeddings.npy'); print(f'Shape: {e.shape}, Norm sample: {np.linalg.norm(e[0]):.4f}')"`
Expected: Shape (N, 384), Norm ~1.0

**Step 3: Commit**

```bash
git add embeddings/crop_index.csv
git commit -m "feat: run embedding extraction, save index"
```

---

## Milestone 5: Within-Video Grouping (Phase 4)

### Task 11: Write grouping tests

**Files:**
- Create: `tests/test_group.py`

**Step 1: Write tests**

```python
# tests/test_group.py
import numpy as np
import pandas as pd


def test_group_single_bird_video():
    from pipeline.group import group_crops_in_video

    # 5 crops, all similar embeddings -> 1 group
    embeddings = np.random.randn(5, 384).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Make them very similar
    base = embeddings[0]
    for i in range(1, 5):
        embeddings[i] = base + np.random.randn(384) * 0.01
        embeddings[i] /= np.linalg.norm(embeddings[i])

    crop_ids = [f"vid_c{i:05d}" for i in range(5)]
    groups = group_crops_in_video(crop_ids, embeddings, distance_threshold=0.3)
    assert len(groups) == 1
    assert len(groups[0]["crop_ids"]) == 5


def test_group_two_bird_video():
    from pipeline.group import group_crops_in_video

    # 6 crops: 3 similar + 3 very different
    rng = np.random.RandomState(42)
    base_a = rng.randn(384).astype(np.float32)
    base_b = -base_a  # opposite direction = very different

    embeddings = np.zeros((6, 384), dtype=np.float32)
    for i in range(3):
        embeddings[i] = base_a + rng.randn(384) * 0.01
    for i in range(3, 6):
        embeddings[i] = base_b + rng.randn(384) * 0.01
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    crop_ids = [f"vid_c{i:05d}" for i in range(6)]
    groups = group_crops_in_video(crop_ids, embeddings, distance_threshold=0.3)
    assert len(groups) == 2


def test_compute_session_embeddings():
    from pipeline.group import compute_session_embeddings

    crop_embeddings = np.random.randn(10, 384).astype(np.float32)
    crop_embeddings = crop_embeddings / np.linalg.norm(crop_embeddings, axis=1, keepdims=True)

    sessions = [
        {"session_id": "s0", "video_id": "v0", "crop_ids": ["c0", "c1", "c2"]},
        {"session_id": "s1", "video_id": "v0", "crop_ids": ["c3", "c4"]},
    ]
    crop_id_to_idx = {f"c{i}": i for i in range(10)}

    session_embs = compute_session_embeddings(sessions, crop_embeddings, crop_id_to_idx)
    assert session_embs.shape == (2, 384)
    # Should be L2-normalized
    norms = np.linalg.norm(session_embs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_group.py -v`
Expected: FAIL

### Task 12: Implement within-video grouping

**Files:**
- Create: `pipeline/group.py`

**Step 1: Implement grouping module**

```python
# pipeline/group.py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from pipeline.config import GROUPING_DISTANCE_THRESHOLD, EMBEDDINGS_DIR


def group_crops_in_video(
    crop_ids: list[str],
    embeddings: np.ndarray,
    distance_threshold: float = GROUPING_DISTANCE_THRESHOLD,
) -> list[dict]:
    """Group crops within a single video by embedding similarity."""
    if len(crop_ids) == 1:
        return [{"crop_ids": crop_ids}]

    # Cosine distance = 1 - cosine_similarity
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)

    groups = []
    for label in sorted(set(labels)):
        mask = labels == label
        group_crop_ids = [crop_ids[i] for i in range(len(crop_ids)) if mask[i]]
        groups.append({"crop_ids": group_crop_ids})
    return groups


def compute_session_embeddings(
    sessions: list[dict],
    crop_embeddings: np.ndarray,
    crop_id_to_idx: dict[str, int],
) -> np.ndarray:
    """Compute mean embedding per session, L2-normalized."""
    session_embs = []
    for session in sessions:
        indices = [crop_id_to_idx[cid] for cid in session["crop_ids"]]
        mean_emb = crop_embeddings[indices].mean(axis=0)
        mean_emb /= np.linalg.norm(mean_emb)
        session_embs.append(mean_emb)
    return np.array(session_embs)


def run_grouping(
    crop_index_df: pd.DataFrame,
    crop_embeddings: np.ndarray,
    embeddings_dir: Path = EMBEDDINGS_DIR,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Group crops within each video and compute session embeddings."""
    crop_id_to_idx = {cid: i for i, cid in enumerate(crop_index_df["crop_id"])}
    all_sessions = []
    session_counter = 0

    for video_id, group in tqdm(crop_index_df.groupby("video_id"), desc="Grouping within videos"):
        crop_ids = group["crop_id"].tolist()
        indices = [crop_id_to_idx[cid] for cid in crop_ids]
        embs = crop_embeddings[indices]

        groups = group_crops_in_video(crop_ids, embs)

        for g in groups:
            session_id = f"session_{session_counter:05d}"
            all_sessions.append({
                "session_id": session_id,
                "video_id": video_id,
                "crop_ids": ",".join(g["crop_ids"]),
            })
            session_counter += 1

    sessions_df = pd.DataFrame(all_sessions)
    session_embeddings = compute_session_embeddings(
        all_sessions, crop_embeddings, crop_id_to_idx,
    )

    # Save
    np.save(embeddings_dir / "session_embeddings.npy", session_embeddings)
    sessions_df.to_csv(embeddings_dir / "session_index.csv", index=False)

    print(f"Created {len(sessions_df)} sessions from {len(crop_index_df)} crops")
    print(f"Session embeddings shape: {session_embeddings.shape}")
    return session_embeddings, sessions_df


if __name__ == "__main__":
    crop_index_df = pd.read_csv(EMBEDDINGS_DIR / "crop_index.csv")
    crop_embeddings = np.load(EMBEDDINGS_DIR / "crop_embeddings.npy")
    run_grouping(crop_index_df, crop_embeddings)
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_group.py -v`
Expected: All 3 tests PASS

**Step 3: Commit**

```bash
git add pipeline/group.py tests/test_group.py
git commit -m "feat: add within-video crop grouping into sessions"
```

### Task 13: Run grouping on real data

**Step 1: Run**

Run: `python -m pipeline.group`

**Step 2: Inspect**

Run: `python -c "import numpy as np; import pandas as pd; s=pd.read_csv('embeddings/session_index.csv'); print(f'Sessions: {len(s)}'); print(s['video_id'].value_counts().describe())"`
Expected: More sessions than videos if multi-bird videos exist

**Step 3: Commit**

```bash
git add embeddings/session_index.csv
git commit -m "feat: run within-video grouping, save session index"
```

---

## Milestone 6: Species Classification (Phase 5)

**Pause after this milestone — user needs to label species.**

### Task 14: Write species labeling tests

**Files:**
- Create: `tests/test_label_species.py`

**Step 1: Write tests**

```python
# tests/test_label_species.py
import numpy as np


def test_cluster_for_species():
    from pipeline.label_species import cluster_for_species

    # 21 embeddings that should form ~3 clusters
    rng = np.random.RandomState(42)
    centers = rng.randn(3, 384).astype(np.float32)
    embeddings = []
    for c in centers:
        for _ in range(7):
            e = c + rng.randn(384) * 0.1
            embeddings.append(e / np.linalg.norm(e))
    embeddings = np.array(embeddings)

    labels = cluster_for_species(embeddings, n_clusters=3)
    assert len(labels) == 21
    assert len(set(labels)) == 3


def test_get_sample_crops_per_cluster():
    from pipeline.label_species import get_sample_crops_per_cluster
    import pandas as pd

    crop_index = pd.DataFrame({
        "crop_id": [f"c{i}" for i in range(10)],
        "crop_path": [f"/path/crop_{i}.jpg" for i in range(10)],
    })
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    samples = get_sample_crops_per_cluster(labels, crop_index, samples_per_cluster=3)
    assert len(samples) == 2  # 2 clusters
    assert all(len(v) <= 3 for v in samples.values())
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_label_species.py -v`
Expected: FAIL

### Task 15: Implement species labeling helper

**Files:**
- Create: `pipeline/label_species.py`

**Step 1: Implement module**

```python
# pipeline/label_species.py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pipeline.config import NUM_SPECIES, DATA_DIR, EMBEDDINGS_DIR


def cluster_for_species(
    embeddings: np.ndarray,
    n_clusters: int = NUM_SPECIES,
) -> np.ndarray:
    """K-means clustering as rough species proxy."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels


def get_sample_crops_per_cluster(
    labels: np.ndarray,
    crop_index: pd.DataFrame,
    samples_per_cluster: int = 20,
) -> dict[int, list[str]]:
    """Get sample crop paths for each cluster."""
    samples = {}
    rng = np.random.RandomState(42)
    for cluster_id in sorted(set(labels)):
        indices = np.where(np.array(labels) == cluster_id)[0]
        n_samples = min(samples_per_cluster, len(indices))
        chosen = rng.choice(indices, n_samples, replace=False)
        samples[cluster_id] = crop_index.iloc[chosen]["crop_path"].tolist()
    return samples


def generate_labeling_html(
    samples: dict[int, list[str]],
    output_path: Path,
):
    """Generate an HTML page showing sample crops per cluster for labeling."""
    html = """<!DOCTYPE html>
<html><head><style>
body { font-family: sans-serif; margin: 20px; }
.cluster { margin: 30px 0; padding: 20px; border: 2px solid #ccc; border-radius: 8px; }
.cluster h2 { margin-top: 0; }
.grid { display: flex; flex-wrap: wrap; gap: 8px; }
.grid img { width: 112px; height: 112px; object-fit: cover; border-radius: 4px; }
input[type="text"] { font-size: 18px; padding: 8px; margin-top: 10px; width: 300px; }
button { font-size: 18px; padding: 10px 30px; margin-top: 20px; cursor: pointer; }
</style></head><body>
<h1>Species Labeling</h1>
<p>Type the species name for each cluster, then click Save.</p>
<form id="form">
"""
    for cluster_id, paths in sorted(samples.items()):
        html += f'<div class="cluster"><h2>Cluster {cluster_id} ({len(paths)} samples)</h2>\n'
        html += '<div class="grid">\n'
        for p in paths:
            html += f'<img src="file://{p}" alt="crop">\n'
        html += '</div>\n'
        html += f'<br><label>Species: <input type="text" name="cluster_{cluster_id}" placeholder="e.g. House Finch"></label>\n'
        html += '</div>\n'

    html += """
<button type="button" onclick="saveLabels()">Save Labels</button>
<pre id="output"></pre>
<script>
function saveLabels() {
    const data = {};
    const inputs = document.querySelectorAll('input[type=text]');
    inputs.forEach(input => {
        const cluster = input.name.replace('cluster_', '');
        data[cluster] = input.value.trim();
    });
    document.getElementById('output').textContent = JSON.stringify(data, null, 2);
    // Copy to clipboard
    navigator.clipboard.writeText(JSON.stringify(data));
    alert('Labels copied to clipboard! Paste into terminal when prompted.');
}
</script>
</body></html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Labeling page saved to {output_path}")
    print(f"Open in browser: file://{output_path.resolve()}")


def apply_labels(
    labels: np.ndarray,
    species_map: dict[int, str],
    crop_index: pd.DataFrame,
    session_index: pd.DataFrame,
) -> pd.DataFrame:
    """Apply species labels to all crops and sessions."""
    crop_species = [species_map.get(int(l), "unknown") for l in labels]
    crop_index = crop_index.copy()
    crop_index["species"] = crop_species

    # Map crop species to sessions
    crop_to_species = dict(zip(crop_index["crop_id"], crop_index["species"]))
    session_species = []
    for _, row in session_index.iterrows():
        crop_ids = row["crop_ids"].split(",")
        species_votes = [crop_to_species.get(cid, "unknown") for cid in crop_ids]
        # Majority vote
        from collections import Counter
        most_common = Counter(species_votes).most_common(1)[0][0]
        session_species.append(most_common)

    session_index = session_index.copy()
    session_index["species"] = session_species

    return crop_index, session_index


def run_species_labeling(
    crop_embeddings: np.ndarray,
    crop_index: pd.DataFrame,
    session_index: pd.DataFrame,
):
    """Run the species labeling workflow."""
    print("Clustering crops for species labeling...")
    labels = cluster_for_species(crop_embeddings)

    print(f"Found {len(set(labels))} clusters")
    for cl in sorted(set(labels)):
        count = sum(1 for l in labels if l == cl)
        print(f"  Cluster {cl}: {count} crops")

    samples = get_sample_crops_per_cluster(labels, crop_index)
    html_path = DATA_DIR / "species_labeling.html"
    generate_labeling_html(samples, html_path)

    print("\n--- ACTION REQUIRED ---")
    print(f"1. Open in browser: file://{html_path.resolve()}")
    print("2. Label each cluster with a species name")
    print("3. Click 'Save Labels' (copies JSON to clipboard)")
    print("4. Paste the JSON below:\n")

    raw = input("Paste species labels JSON: ")
    import json
    species_map = {int(k): v for k, v in json.loads(raw).items()}
    print(f"\nLabels: {species_map}")

    crop_index, session_index = apply_labels(labels, species_map, crop_index, session_index)

    # Save
    species_labels_path = DATA_DIR / "species_labels.csv"
    crop_index[["crop_id", "species"]].to_csv(species_labels_path, index=False)
    session_index.to_csv(EMBEDDINGS_DIR / "session_index.csv", index=False)

    print(f"\nSpecies distribution (sessions):")
    print(session_index["species"].value_counts().to_string())
    print(f"\nSaved to {species_labels_path}")
    return crop_index, session_index


if __name__ == "__main__":
    crop_embeddings = np.load(EMBEDDINGS_DIR / "crop_embeddings.npy")
    crop_index = pd.read_csv(EMBEDDINGS_DIR / "crop_index.csv")
    session_index = pd.read_csv(EMBEDDINGS_DIR / "session_index.csv")
    run_species_labeling(crop_embeddings, crop_index, session_index)
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_label_species.py -v`
Expected: All 2 tests PASS

**Step 3: Commit**

```bash
git add pipeline/label_species.py tests/test_label_species.py
git commit -m "feat: add species labeling helper with HTML UI"
```

### Task 16: Run species labeling

**Step 1: Run**

Run: `python -m pipeline.label_species`
Expected: Opens HTML labeling page, waits for user input

**Step 2: User labels species in browser, pastes JSON**

**Step 3: Commit**

```bash
git add data/species_labels.csv embeddings/session_index.csv
git commit -m "feat: label species, save species assignments"
```

**PAUSE: Review species distribution before clustering.**

---

## Milestone 7: Species-Aware Clustering (Phase 6)

**Pause after this milestone for user inspection.**

### Task 17: Write clustering tests

**Files:**
- Create: `tests/test_cluster.py`

**Step 1: Write tests**

```python
# tests/test_cluster.py
import numpy as np
import pandas as pd


def test_cluster_species():
    from pipeline.cluster import cluster_species

    # Create 20 embeddings with 4 clear clusters
    rng = np.random.RandomState(42)
    centers = rng.randn(4, 50).astype(np.float32)
    embeddings = []
    for c in centers:
        for _ in range(5):
            e = c + rng.randn(50) * 0.05
            embeddings.append(e)
    embeddings = np.array(embeddings)

    labels = cluster_species(embeddings, min_cluster_size=3, min_samples=2)
    assert len(labels) == 20
    # Should find some clusters (not all noise)
    non_noise = sum(1 for l in labels if l >= 0)
    assert non_noise > 10


def test_pca_reduction():
    from pipeline.cluster import reduce_dimensions

    embeddings = np.random.randn(50, 384).astype(np.float32)
    reduced, pca = reduce_dimensions(embeddings, n_components=50)
    assert reduced.shape == (50, 50)
    assert pca is not None
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cluster.py -v`
Expected: FAIL

### Task 18: Implement clustering

**Files:**
- Create: `pipeline/cluster.py`

**Step 1: Implement clustering module**

```python
# pipeline/cluster.py
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
from pipeline.config import (
    PCA_COMPONENTS, HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES,
    MERGE_SIMILARITY_THRESHOLD, EMBEDDINGS_DIR, CLUSTERS_DIR, MODELS_DIR,
)


def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = PCA_COMPONENTS,
) -> tuple[np.ndarray, PCA]:
    """PCA dimensionality reduction."""
    n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    return reduced, pca


def cluster_species(
    embeddings: np.ndarray,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
) -> np.ndarray:
    """Run HDBSCAN clustering on embeddings."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embeddings)
    return labels


def refine_clusters(
    labels: np.ndarray,
    embeddings: np.ndarray,
    merge_threshold: float = MERGE_SIMILARITY_THRESHOLD,
) -> np.ndarray:
    """Refine clusters: merge similar clusters."""
    unique_labels = [l for l in set(labels) if l >= 0]
    if len(unique_labels) <= 1:
        return labels

    # Compute centroids
    centroids = {}
    for l in unique_labels:
        mask = labels == l
        centroids[l] = embeddings[mask].mean(axis=0)

    # Find pairs to merge
    centroid_matrix = np.array([centroids[l] for l in unique_labels])
    sim_matrix = cosine_similarity(centroid_matrix)

    merge_map = {}
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            if sim_matrix[i, j] > merge_threshold:
                li, lj = unique_labels[i], unique_labels[j]
                target = merge_map.get(li, li)
                merge_map[lj] = target

    if merge_map:
        new_labels = labels.copy()
        for old, new in merge_map.items():
            new_labels[labels == old] = new
        return new_labels

    return labels


def run_clustering(
    session_embeddings: np.ndarray,
    session_index: pd.DataFrame,
):
    """Run species-aware clustering."""
    CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    all_assignments = []
    all_centroids = {}
    pca_models = {}

    species_list = session_index["species"].unique()
    print(f"Clustering {len(species_list)} species...")

    for species in sorted(species_list):
        mask = session_index["species"] == species
        species_idx = np.where(mask)[0]
        species_embs = session_embeddings[species_idx]
        species_sessions = session_index[mask].reset_index(drop=True)

        print(f"\n--- {species} ({len(species_embs)} sessions) ---")

        if len(species_embs) < 3:
            print(f"  Too few sessions, assigning all to one bird")
            for _, row in species_sessions.iterrows():
                all_assignments.append({
                    "session_id": row["session_id"],
                    "video_id": row["video_id"],
                    "species": species,
                    "bird_id": f"{species}_bird_0",
                })
            continue

        # PCA
        n_comp = min(PCA_COMPONENTS, len(species_embs) - 1)
        reduced, pca = reduce_dimensions(species_embs, n_components=n_comp)
        pca_models[species] = pca

        # Adaptive HDBSCAN parameters
        min_cs = max(3, len(species_embs) // 20)
        min_s = max(2, min_cs // 2)
        print(f"  PCA: {species_embs.shape[1]} -> {reduced.shape[1]}")
        print(f"  HDBSCAN params: min_cluster_size={min_cs}, min_samples={min_s}")

        labels = cluster_species(reduced, min_cluster_size=min_cs, min_samples=min_s)
        labels = refine_clusters(labels, reduced)

        n_clusters = len(set(labels) - {-1})
        n_noise = sum(1 for l in labels if l == -1)
        print(f"  Found {n_clusters} individuals, {n_noise} noise points")

        # Compute centroids (on original embeddings for FAISS later)
        for l in set(labels):
            if l < 0:
                continue
            bird_id = f"{species}_bird_{l}"
            cluster_mask = labels == l
            centroid = species_embs[cluster_mask].mean(axis=0)
            centroid /= np.linalg.norm(centroid)
            all_centroids[bird_id] = centroid

        for i, (_, row) in enumerate(species_sessions.iterrows()):
            label = labels[i]
            bird_id = f"{species}_bird_{label}" if label >= 0 else f"{species}_noise"
            all_assignments.append({
                "session_id": row["session_id"],
                "video_id": row["video_id"],
                "species": species,
                "bird_id": bird_id,
            })

    # Save assignments
    assignments_df = pd.DataFrame(all_assignments)
    assignments_df.to_csv(CLUSTERS_DIR / "cluster_assignments.csv", index=False)

    # Save centroids
    if all_centroids:
        bird_ids = sorted(all_centroids.keys())
        centroid_matrix = np.array([all_centroids[bid] for bid in bird_ids])
        np.save(CLUSTERS_DIR / "cluster_centroids.npy", centroid_matrix)
        pd.DataFrame({"bird_id": bird_ids}).to_csv(
            CLUSTERS_DIR / "centroid_index.csv", index=False
        )

    # Save PCA models
    with open(MODELS_DIR / "pca_models.pkl", "wb") as f:
        pickle.dump(pca_models, f)

    print(f"\n=== Summary ===")
    print(f"Total sessions: {len(assignments_df)}")
    print(f"Total individuals: {len(all_centroids)}")
    print(f"\nPer species:")
    for species in sorted(species_list):
        sp_df = assignments_df[assignments_df["species"] == species]
        n_birds = sp_df["bird_id"].nunique()
        n_noise = len(sp_df[sp_df["bird_id"].str.contains("noise")])
        print(f"  {species}: {n_birds} individuals, {n_noise} noise sessions")

    return assignments_df


if __name__ == "__main__":
    session_embeddings = np.load(EMBEDDINGS_DIR / "session_embeddings.npy")
    session_index = pd.read_csv(EMBEDDINGS_DIR / "session_index.csv")
    run_clustering(session_embeddings, session_index)
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_cluster.py -v`
Expected: All 2 tests PASS

**Step 3: Commit**

```bash
git add pipeline/cluster.py tests/test_cluster.py
git commit -m "feat: add species-aware HDBSCAN clustering with refinement"
```

### Task 19: Run clustering on real data

**Step 1: Run**

Run: `python -m pipeline.cluster`
Expected: Per-species cluster summary

**Step 2: Inspect**

Run: `python -c "import pandas as pd; df=pd.read_csv('clusters/cluster_assignments.csv'); print(df.groupby('species')['bird_id'].nunique()); print(f'\nTotal: {df[\"bird_id\"].nunique()} individuals')"`

**Step 3: Commit**

```bash
git add clusters/cluster_assignments.csv clusters/centroid_index.csv
git commit -m "feat: run clustering, save bird identity assignments"
```

**PAUSE: Review clustering results. Check if individual counts per species make sense.**

---

## Milestone 8: Bird Identity Database & Inference (Phase 7)

### Task 20: Write inference tests

**Files:**
- Create: `tests/test_inference.py`

**Step 1: Write tests**

```python
# tests/test_inference.py
import numpy as np


def test_build_faiss_index():
    from pipeline.inference import build_faiss_index

    centroids = np.random.randn(10, 384).astype(np.float32)
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    bird_ids = [f"bird_{i}" for i in range(10)]

    index, id_map = build_faiss_index(centroids, bird_ids)
    assert index.ntotal == 10
    assert len(id_map) == 10


def test_query_identity():
    from pipeline.inference import build_faiss_index, query_identity

    rng = np.random.RandomState(42)
    centroids = rng.randn(5, 384).astype(np.float32)
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    bird_ids = [f"bird_{i}" for i in range(5)]

    index, id_map = build_faiss_index(centroids, bird_ids)

    # Query with something very similar to centroid 0
    query = centroids[0] + rng.randn(384) * 0.01
    query = query / np.linalg.norm(query)

    result = query_identity(index, id_map, query, threshold=0.5)
    assert result["bird_id"] == "bird_0"
    assert result["similarity"] > 0.9
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_inference.py -v`
Expected: FAIL

### Task 21: Implement inference pipeline

**Files:**
- Create: `pipeline/inference.py`

**Step 1: Implement inference module**

```python
# pipeline/inference.py
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from pipeline.config import (
    SIMILARITY_THRESHOLD, MODELS_DIR, CLUSTERS_DIR,
    EMBEDDINGS_DIR, DATA_DIR, CROPS_DIR,
)
from pipeline.ingest import extract_frames
from pipeline.detect import run_detection
from pipeline.embed import load_model, embed_crops
from pipeline.group import group_crops_in_video, compute_session_embeddings


def build_faiss_index(
    centroids: np.ndarray,
    bird_ids: list[str],
) -> tuple[faiss.IndexFlatIP, dict[int, str]]:
    """Build a FAISS inner-product index from cluster centroids."""
    centroids = centroids.astype(np.float32)
    # Ensure L2-normalized for cosine similarity via inner product
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / norms

    dim = centroids.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(centroids)

    id_map = {i: bird_ids[i] for i in range(len(bird_ids))}
    return index, id_map


def query_identity(
    index: faiss.IndexFlatIP,
    id_map: dict[int, str],
    embedding: np.ndarray,
    threshold: float = SIMILARITY_THRESHOLD,
) -> dict:
    """Query the FAISS index for the nearest bird identity."""
    embedding = embedding.astype(np.float32).reshape(1, -1)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    similarities, indices = index.search(embedding, 1)
    sim = float(similarities[0][0])
    idx = int(indices[0][0])

    if sim >= threshold:
        return {"bird_id": id_map[idx], "similarity": sim, "is_new": False}
    else:
        return {"bird_id": None, "similarity": sim, "is_new": True}


def build_bird_database(
    assignments_df: pd.DataFrame,
    session_index: pd.DataFrame,
    crop_index: pd.DataFrame,
) -> dict:
    """Build the bird identity database with example images and visit history."""
    database = {}

    for bird_id in assignments_df["bird_id"].unique():
        if "noise" in bird_id:
            continue

        bird_sessions = assignments_df[assignments_df["bird_id"] == bird_id]
        species = bird_sessions["species"].iloc[0]
        video_ids = bird_sessions["video_id"].tolist()

        # Get example crop paths
        session_ids = bird_sessions["session_id"].tolist()
        example_crops = []
        for sid in session_ids[:5]:  # up to 5 sessions
            session_row = session_index[session_index["session_id"] == sid]
            if len(session_row) == 0:
                continue
            crop_ids = session_row.iloc[0]["crop_ids"].split(",")
            for cid in crop_ids[:2]:  # up to 2 crops per session
                crop_row = crop_index[crop_index["crop_id"] == cid]
                if len(crop_row) > 0:
                    example_crops.append(crop_row.iloc[0]["crop_path"])

        database[bird_id] = {
            "bird_id": bird_id,
            "species": species,
            "num_visits": len(bird_sessions),
            "video_ids": video_ids,
            "example_crops": example_crops[:10],
        }

    return database


def save_inference_artifacts(
    centroids: np.ndarray,
    bird_ids: list[str],
    database: dict,
):
    """Save FAISS index, PCA models, and bird database."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    index, id_map = build_faiss_index(centroids, bird_ids)
    faiss.write_index(index, str(MODELS_DIR / "faiss_index.bin"))

    with open(MODELS_DIR / "faiss_id_map.json", "w") as f:
        json.dump({str(k): v for k, v in id_map.items()}, f)

    with open(MODELS_DIR / "bird_database.json", "w") as f:
        json.dump(database, f, indent=2)

    print(f"Saved FAISS index ({index.ntotal} birds)")
    print(f"Saved bird database ({len(database)} identities)")


def infer_video(
    video_path: Path,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict]:
    """Run inference on a new video."""
    # Load models
    print("Loading models...")
    dino_model, dino_transform = load_model()

    with open(MODELS_DIR / "pca_models.pkl", "rb") as f:
        pca_models = pickle.load(f)

    index = faiss.read_index(str(MODELS_DIR / "faiss_index.bin"))
    with open(MODELS_DIR / "faiss_id_map.json") as f:
        id_map = {int(k): v for k, v in json.load(f).items()}

    with open(MODELS_DIR / "bird_database.json") as f:
        database = json.load(f)

    # Extract frames
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        frames_dir = Path(tmpdir) / "frames"
        frames = extract_frames(video_path, frames_dir)
        if not frames:
            print("No frames extracted")
            return []

        frames_df = pd.DataFrame(frames)

        # Detect
        crops_dir = Path(tmpdir) / "crops"
        crops_df = run_detection(frames_df, crops_dir)
        if len(crops_df) == 0:
            print("No birds detected")
            return []

        # Embed
        crop_paths = crops_df["crop_path"].tolist()
        crop_embeddings = embed_crops(dino_model, dino_transform, crop_paths)

        # Group within video
        crop_ids = crops_df["crop_id"].tolist()
        groups = group_crops_in_video(crop_ids, crop_embeddings)

        crop_id_to_idx = {cid: i for i, cid in enumerate(crop_ids)}

        results = []
        for g in groups:
            indices = [crop_id_to_idx[cid] for cid in g["crop_ids"]]
            session_emb = crop_embeddings[indices].mean(axis=0)
            session_emb /= np.linalg.norm(session_emb)

            result = query_identity(index, id_map, session_emb, threshold)
            result["num_crops"] = len(g["crop_ids"])

            if result["bird_id"]:
                bird_info = database.get(result["bird_id"], {})
                result["species"] = bird_info.get("species", "unknown")
            else:
                result["species"] = "unknown"

            results.append(result)
            status = f"matched {result['bird_id']}" if not result["is_new"] else "NEW bird candidate"
            print(f"  Bird group ({result['num_crops']} crops): {status} (sim={result['similarity']:.3f})")

        return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Inference mode
        video_path = Path(sys.argv[1])
        results = infer_video(video_path)
    else:
        # Build mode
        print("Building bird identity database...")
        centroids = np.load(CLUSTERS_DIR / "cluster_centroids.npy")
        centroid_index = pd.read_csv(CLUSTERS_DIR / "centroid_index.csv")
        assignments_df = pd.read_csv(CLUSTERS_DIR / "cluster_assignments.csv")
        session_index = pd.read_csv(EMBEDDINGS_DIR / "session_index.csv")
        crop_index = pd.read_csv(EMBEDDINGS_DIR / "crop_index.csv")

        bird_ids = centroid_index["bird_id"].tolist()
        database = build_bird_database(assignments_df, session_index, crop_index)
        save_inference_artifacts(centroids, bird_ids, database)
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_inference.py -v`
Expected: All 2 tests PASS

**Step 3: Commit**

```bash
git add pipeline/inference.py tests/test_inference.py
git commit -m "feat: add FAISS inference pipeline and bird identity database"
```

### Task 22: Build identity database and test inference

**Step 1: Build database**

Run: `python -m pipeline.inference`
Expected: Saves FAISS index and bird database to `models/`

**Step 2: Test on a known video**

Run: `python -m pipeline.inference raw_data/<pick_a_video>_1.mp4`
Expected: Identifies bird(s) with similarity scores

**Step 3: Commit**

```bash
git add models/faiss_id_map.json models/bird_database.json
git commit -m "feat: build bird identity database and FAISS index"
```

**PAUSE: Test inference on a few videos. Check if identifications look reasonable.**

---

## Milestone 9 (Later): Visualization

### Task 23: Add UMAP visualization

**Files:**
- Create: `pipeline/visualize.py`

**Step 1: Implement visualization**

```python
# pipeline/visualize.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from pipeline.config import EMBEDDINGS_DIR, CLUSTERS_DIR


def visualize_embeddings(
    session_embeddings: np.ndarray,
    assignments_df: pd.DataFrame,
    color_by: str = "species",
    output_path: Path = None,
):
    """UMAP 2D visualization of session embeddings."""
    reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine")
    coords = reducer.fit_transform(session_embeddings)

    categories = assignments_df[color_by].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
    color_map = dict(zip(categories, colors))

    fig, ax = plt.subplots(figsize=(14, 10))
    for cat in sorted(categories):
        mask = assignments_df[color_by] == cat
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[color_map[cat]], label=cat, alpha=0.6, s=20,
        )

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_title(f"Bird Sessions — colored by {color_by}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    session_embeddings = np.load(EMBEDDINGS_DIR / "session_embeddings.npy")
    assignments_df = pd.read_csv(CLUSTERS_DIR / "cluster_assignments.csv")

    CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)

    visualize_embeddings(
        session_embeddings, assignments_df,
        color_by="species",
        output_path=CLUSTERS_DIR / "umap_by_species.png",
    )
    visualize_embeddings(
        session_embeddings, assignments_df,
        color_by="bird_id",
        output_path=CLUSTERS_DIR / "umap_by_bird_id.png",
    )
```

**Step 2: Run**

Run: `python -m pipeline.visualize`
Expected: Two UMAP plots saved to `clusters/`

**Step 3: Commit**

```bash
git add pipeline/visualize.py
git commit -m "feat: add UMAP visualization of bird embeddings"
```

---

## Summary of Milestones & Pause Points

| Milestone | Phase | Pause? |
|-----------|-------|--------|
| 1. Project Setup | Setup | No |
| 2. Data Ingestion | Phase 1 | **Yes** — inspect frame extraction |
| 3. Bird Detection | Phase 2 | **Yes** — inspect crop quality |
| 4. Feature Extraction | Phase 3 | No |
| 5. Within-Video Grouping | Phase 4 | No |
| 6. Species Classification | Phase 5 | **Yes** — user labels species |
| 7. Clustering | Phase 6 | **Yes** — review individual counts |
| 8. Identity Database | Phase 7 | **Yes** — test inference |
| 9. Visualization | Phase 8 | No |
