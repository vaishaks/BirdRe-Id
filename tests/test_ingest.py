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
        (Path(tmpdir) / "abc_1.mp4").touch()
        (Path(tmpdir) / "abc.mp4").touch()
        (Path(tmpdir) / "def_1.MP4").touch()
        (Path(tmpdir) / "def.MP4").touch()
        (Path(tmpdir) / "ghi_1 (1).mp4").touch()

        videos = find_high_fps_videos(Path(tmpdir))
        assert len(videos) >= 2
        assert all("_1" in v.stem for v in videos)


def test_deduplicate_videos():
    from pipeline.ingest import deduplicate_videos

    with tempfile.TemporaryDirectory() as tmpdir:
        content = b"fake video content"
        p1 = Path(tmpdir) / "abc_1.mp4"
        p2 = Path(tmpdir) / "abc_1 (1).mp4"
        p1.write_bytes(content)
        p2.write_bytes(content)
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
        assert len(frames) == 3
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
        (raw_dir / "vid1.mp4").touch()

        output_dir = Path(tmpdir) / "data"
        metadata_df = run_ingestion(raw_dir, output_dir / "frames", frame_skip=10)

        assert isinstance(metadata_df, pd.DataFrame)
        assert len(metadata_df) > 0
        assert set(metadata_df.columns) >= {"frame_id", "video_id", "frame_number", "image_path"}
