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

        # PCA (saved for inference, but clustering uses raw embeddings)
        n_comp = min(20, len(species_embs) - 1)
        reduced, pca = reduce_dimensions(species_embs, n_components=n_comp)
        pca_models[species] = pca

        # Agglomerative clustering on raw embeddings with cosine distance
        # Threshold tuned per group size:
        #   Small groups (<30): tighter threshold to avoid over-merging
        #   Large groups (30+): threshold that finds ~10-30 individuals
        from sklearn.cluster import AgglomerativeClustering
        if len(species_embs) < 30:
            dist_threshold = 0.35
        else:
            dist_threshold = 0.25

        agg = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=dist_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = agg.fit_predict(species_embs)

        # Filter out tiny clusters as noise
        from collections import Counter
        cluster_counts = Counter(labels)
        min_size = 3 if len(species_embs) >= 30 else 2
        for i in range(len(labels)):
            if cluster_counts[labels[i]] < min_size:
                labels[i] = -1

        # For mixed-species groups, ensure at least 2 clusters
        if "mixture" in species.lower() or "mixed" in species.lower() or species == "Chickadee":
            unique_non_noise = len(set(labels) - {-1})
            if unique_non_noise < 2:
                agg2 = AgglomerativeClustering(n_clusters=2, metric="cosine", linkage="average")
                labels = agg2.fit_predict(species_embs)

        print(f"  Agglomerative (threshold={dist_threshold}): {len(set(labels) - {-1})} individuals")

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
