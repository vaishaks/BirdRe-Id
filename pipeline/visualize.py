# pipeline/visualize.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import umap
from pipeline.config import EMBEDDINGS_DIR, CLUSTERS_DIR


def plot_umap(
    session_embeddings: np.ndarray,
    assignments_df: pd.DataFrame,
    color_by: str = "species",
    output_path: Path = None,
):
    """UMAP 2D visualization of session embeddings."""
    print(f"Computing UMAP projection (color_by={color_by})...")
    reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine")
    coords = reducer.fit_transform(session_embeddings)

    categories = sorted(assignments_df[color_by].unique())
    cmap = plt.cm.tab20 if len(categories) <= 20 else plt.cm.gist_ncar
    colors = cmap(np.linspace(0, 1, len(categories)))
    color_map = dict(zip(categories, colors))

    fig, ax = plt.subplots(figsize=(14, 10))
    for cat in categories:
        mask = assignments_df[color_by] == cat
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[color_map[cat]], label=cat, alpha=0.6, s=20,
        )

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7, ncol=1 + len(categories) // 25)
    ax.set_title(f"Bird Sessions — colored by {color_by}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    plt.close()


def generate_cluster_gallery(
    assignments_df: pd.DataFrame,
    session_index: pd.DataFrame,
    crop_index: pd.DataFrame,
    output_path: Path,
    samples_per_bird: int = 8,
):
    """Generate an HTML gallery showing example crops per bird_id cluster."""
    # Build crop_id -> crop_path mapping
    crop_path_map = dict(zip(crop_index["crop_id"], crop_index["crop_path"]))

    html = """<!DOCTYPE html>
<html><head><style>
body { font-family: sans-serif; margin: 20px; background: #f5f5f5; }
.species-group { margin: 30px 0; }
.species-group h2 { color: #333; border-bottom: 2px solid #666; padding-bottom: 5px; }
.bird { margin: 15px 0; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.bird h3 { margin-top: 0; color: #555; }
.bird .stats { color: #888; font-size: 14px; margin-bottom: 10px; }
.grid { display: flex; flex-wrap: wrap; gap: 8px; }
.grid img { width: 112px; height: 112px; object-fit: cover; border-radius: 4px; border: 1px solid #ddd; }
.noise { opacity: 0.5; }
</style></head><body>
<h1>Bird Identity Gallery</h1>
"""

    rng = np.random.RandomState(42)

    for species in sorted(assignments_df["species"].unique()):
        sp_df = assignments_df[assignments_df["species"] == species]
        html += f'<div class="species-group"><h2>{species}</h2>\n'

        for bird_id in sorted(sp_df["bird_id"].unique()):
            is_noise = "noise" in bird_id
            bird_df = sp_df[sp_df["bird_id"] == bird_id]
            n_sessions = len(bird_df)
            css_class = "bird noise" if is_noise else "bird"

            html += f'<div class="{css_class}"><h3>{bird_id}</h3>\n'
            html += f'<div class="stats">{n_sessions} sessions</div>\n'
            html += '<div class="grid">\n'

            # Collect crop paths from sessions
            all_crop_paths = []
            for _, row in bird_df.iterrows():
                session_row = session_index[session_index["session_id"] == row["session_id"]]
                if len(session_row) == 0:
                    continue
                crop_ids = session_row.iloc[0]["crop_ids"]
                if isinstance(crop_ids, str):
                    crop_ids = crop_ids.split(",")
                for cid in crop_ids:
                    cp = crop_path_map.get(cid)
                    if cp:
                        all_crop_paths.append(cp)

            # Sample
            if len(all_crop_paths) > samples_per_bird:
                chosen = rng.choice(len(all_crop_paths), samples_per_bird, replace=False)
                all_crop_paths = [all_crop_paths[i] for i in chosen]

            for cp in all_crop_paths:
                html += f'<img src="file://{cp}" alt="crop">\n'

            html += '</div></div>\n'

        html += '</div>\n'

    html += '</body></html>'

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Gallery saved to {output_path}")
    print(f"Open in browser: file://{output_path.resolve()}")


if __name__ == "__main__":
    session_embeddings = np.load(EMBEDDINGS_DIR / "session_embeddings.npy")
    assignments_df = pd.read_csv(CLUSTERS_DIR / "cluster_assignments.csv")
    session_index = pd.read_csv(EMBEDDINGS_DIR / "session_index.csv")
    crop_index = pd.read_csv(EMBEDDINGS_DIR / "crop_index.csv")

    CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)

    plot_umap(
        session_embeddings, assignments_df,
        color_by="species",
        output_path=CLUSTERS_DIR / "umap_by_species.png",
    )
    plot_umap(
        session_embeddings, assignments_df,
        color_by="bird_id",
        output_path=CLUSTERS_DIR / "umap_by_bird_id.png",
    )
    generate_cluster_gallery(
        assignments_df, session_index, crop_index,
        output_path=CLUSTERS_DIR / "cluster_gallery.html",
    )
