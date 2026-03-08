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

    columns = [
        "crop_id", "frame_id", "video_id",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "confidence", "crop_path",
    ]
    df = pd.DataFrame(all_crops, columns=columns)
    metadata_path = crops_dir.parent / "crops_metadata.csv"
    df.to_csv(metadata_path, index=False)
    print(f"Detected {len(df)} bird crops from {len(frames_df)} frames")
    print(f"Metadata saved to {metadata_path}")
    return df


if __name__ == "__main__":
    from pipeline.config import DATA_DIR, CROPS_DIR
    frames_df = pd.read_csv(DATA_DIR / "frames_metadata.csv")
    run_detection(frames_df, CROPS_DIR)
