# pipeline/group.py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from pipeline.config import GROUPING_DISTANCE_THRESHOLD, EMBEDDINGS_DIR


def group_crops_in_video(
    crop_ids: list[str],
    embeddings: np.ndarray,
    distance_threshold: float = GROUPING_DISTANCE_THRESHOLD,
) -> list[dict]:
    """Group crops within a single video by embedding similarity."""
    if len(crop_ids) == 1:
        return [{"crop_ids": crop_ids}]

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)

    groups = []
    for label in sorted(set(labels)):
        mask = labels == label
        group_crop_ids = [crop_ids[i] for i in range(len(crop_ids)) if mask[i]]
        groups.append({"crop_ids": group_crop_ids})
    return groups


def compute_session_embeddings(
    sessions: list[dict],
    crop_embeddings: np.ndarray,
    crop_id_to_idx: dict[str, int],
) -> np.ndarray:
    """Compute mean embedding per session, L2-normalized."""
    session_embs = []
    for session in sessions:
        crop_ids = session["crop_ids"]
        if isinstance(crop_ids, str):
            crop_ids = crop_ids.split(",")
        indices = [crop_id_to_idx[cid] for cid in crop_ids]
        mean_emb = crop_embeddings[indices].mean(axis=0)
        mean_emb /= np.linalg.norm(mean_emb)
        session_embs.append(mean_emb)
    return np.array(session_embs)


def run_grouping(
    crop_index_df: pd.DataFrame,
    crop_embeddings: np.ndarray,
    embeddings_dir: Path = EMBEDDINGS_DIR,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Group crops within each video and compute session embeddings."""
    crop_id_to_idx = {cid: i for i, cid in enumerate(crop_index_df["crop_id"])}
    all_sessions = []
    session_counter = 0

    for video_id, group in tqdm(crop_index_df.groupby("video_id"), desc="Grouping within videos"):
        crop_ids = group["crop_id"].tolist()
        indices = [crop_id_to_idx[cid] for cid in crop_ids]
        embs = crop_embeddings[indices]

        groups = group_crops_in_video(crop_ids, embs)

        for g in groups:
            session_id = f"session_{session_counter:05d}"
            all_sessions.append({
                "session_id": session_id,
                "video_id": video_id,
                "crop_ids": ",".join(g["crop_ids"]),
            })
            session_counter += 1

    sessions_df = pd.DataFrame(all_sessions)
    session_embeddings = compute_session_embeddings(
        all_sessions, crop_embeddings, crop_id_to_idx,
    )

    # Save
    np.save(embeddings_dir / "session_embeddings.npy", session_embeddings)
    sessions_df.to_csv(embeddings_dir / "session_index.csv", index=False)

    print(f"Created {len(sessions_df)} sessions from {len(crop_index_df)} crops")
    print(f"Session embeddings shape: {session_embeddings.shape}")
    return session_embeddings, sessions_df


if __name__ == "__main__":
    crop_index_df = pd.read_csv(EMBEDDINGS_DIR / "crop_index.csv")
    crop_embeddings = np.load(EMBEDDINGS_DIR / "crop_embeddings.npy")
    run_grouping(crop_index_df, crop_embeddings)
