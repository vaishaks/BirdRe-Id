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
