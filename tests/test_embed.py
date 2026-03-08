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
