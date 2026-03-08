# tests/test_group.py
import numpy as np
import pandas as pd


def test_group_single_bird_video():
    from pipeline.group import group_crops_in_video

    # 5 crops, all similar embeddings -> 1 group
    embeddings = np.random.randn(5, 384).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
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
    base_b = -base_a

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
    norms = np.linalg.norm(session_embs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
