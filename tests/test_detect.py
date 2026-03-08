# tests/test_detect.py
import tempfile
from pathlib import Path
import numpy as np
import cv2
import pandas as pd


def create_test_frame(path, width=640, height=480):
    """Create a test image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[100:300, 200:400] = (0, 150, 200)
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
