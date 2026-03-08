# pipeline/identity_db.py
"""Bird identity database: FAISS index + metadata for fast bird lookup."""
import json
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from pipeline.config import CLUSTERS_DIR, EMBEDDINGS_DIR, MODELS_DIR


def build_database(
    clusters_dir: Path = CLUSTERS_DIR,
    embeddings_dir: Path = EMBEDDINGS_DIR,
    models_dir: Path = MODELS_DIR,
) -> dict:
    """Build bird identity database from cluster results.

    Creates:
    - FAISS index over cluster centroids
    - Bird database JSON with metadata per bird
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    centroids = np.load(clusters_dir / "cluster_centroids.npy")
    centroid_index = pd.read_csv(clusters_dir / "centroid_index.csv")
    assignments = pd.read_csv(clusters_dir / "cluster_assignments.csv")
    session_index = pd.read_csv(embeddings_dir / "session_index.csv")
    crop_index = pd.read_csv(embeddings_dir / "crop_index_v3.csv")

    bird_ids = centroid_index["bird_id"].tolist()
    crop_path_map = dict(zip(crop_index["crop_id"], crop_index["crop_path"]))

    # Build FAISS index (inner product on L2-normalized vectors = cosine similarity)
    dim = centroids.shape[1]
    index = faiss.IndexFlatIP(dim)
    # Ensure centroids are L2-normalized
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids_normed = (centroids / norms).astype(np.float32)
    index.add(centroids_normed)

    faiss.write_index(index, str(models_dir / "faiss_index.bin"))
    print(f"FAISS index: {index.ntotal} birds, {dim}-dim")

    # Build bird database
    database = {}
    for bird_id in bird_ids:
        bird_assignments = assignments[assignments["bird_id"] == bird_id]
        species = bird_assignments["species"].iloc[0]
        session_ids = bird_assignments["session_id"].tolist()
        video_ids = bird_assignments["video_id"].unique().tolist()

        # Collect example crop paths
        example_crops = []
        for sid in session_ids:
            session_row = session_index[session_index["session_id"] == sid]
            if len(session_row) == 0:
                continue
            crop_ids = session_row.iloc[0]["crop_ids"]
            if isinstance(crop_ids, str):
                crop_ids = crop_ids.split(",")
            for cid in crop_ids[:2]:  # up to 2 crops per session
                cp = crop_path_map.get(cid)
                if cp:
                    example_crops.append(cp)
            if len(example_crops) >= 8:
                break

        database[bird_id] = {
            "species": species,
            "n_sessions": len(session_ids),
            "n_videos": len(video_ids),
            "session_ids": session_ids,
            "video_ids": video_ids,
            "example_crops": example_crops[:8],
        }

    with open(models_dir / "bird_database.json", "w") as f:
        json.dump(database, f, indent=2)

    # Save bird_id order (maps FAISS index position to bird_id)
    with open(models_dir / "faiss_bird_ids.json", "w") as f:
        json.dump(bird_ids, f)

    print(f"Bird database: {len(database)} individuals")
    for species in sorted(set(d["species"] for d in database.values())):
        sp_birds = [k for k, v in database.items() if v["species"] == species]
        total_sessions = sum(database[b]["n_sessions"] for b in sp_birds)
        print(f"  {species}: {len(sp_birds)} birds, {total_sessions} sessions")

    return database


if __name__ == "__main__":
    build_database()
