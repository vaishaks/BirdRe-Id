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
    assert len(samples) == 2
    assert all(len(v) <= 3 for v in samples.values())
