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
