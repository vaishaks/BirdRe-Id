# pipeline/label_species.py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pipeline.config import NUM_SPECIES, DATA_DIR, EMBEDDINGS_DIR


def cluster_for_species(
    embeddings: np.ndarray,
    n_clusters: int = NUM_SPECIES,
) -> np.ndarray:
    """K-means clustering as rough species proxy."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels


def get_sample_crops_per_cluster(
    labels: np.ndarray,
    crop_index: pd.DataFrame,
    samples_per_cluster: int = 20,
) -> dict[int, list[str]]:
    """Get sample crop paths for each cluster."""
    samples = {}
    rng = np.random.RandomState(42)
    for cluster_id in sorted(set(labels)):
        indices = np.where(np.array(labels) == cluster_id)[0]
        n_samples = min(samples_per_cluster, len(indices))
        chosen = rng.choice(indices, n_samples, replace=False)
        samples[cluster_id] = crop_index.iloc[chosen]["crop_path"].tolist()
    return samples


def generate_labeling_html(
    samples: dict[int, list[str]],
    output_path: Path,
):
    """Generate an HTML page showing sample crops per cluster for labeling."""
    html = """<!DOCTYPE html>
<html><head><style>
body { font-family: sans-serif; margin: 20px; }
.cluster { margin: 30px 0; padding: 20px; border: 2px solid #ccc; border-radius: 8px; }
.cluster h2 { margin-top: 0; }
.grid { display: flex; flex-wrap: wrap; gap: 8px; }
.grid img { width: 112px; height: 112px; object-fit: cover; border-radius: 4px; }
input[type="text"] { font-size: 18px; padding: 8px; margin-top: 10px; width: 300px; }
button { font-size: 18px; padding: 10px 30px; margin-top: 20px; cursor: pointer; }
</style></head><body>
<h1>Species Labeling</h1>
<p>Type the species name for each cluster, then click Save.</p>
<form id="form">
"""
    for cluster_id, paths in sorted(samples.items()):
        html += f'<div class="cluster"><h2>Cluster {cluster_id} ({len(paths)} samples)</h2>\n'
        html += '<div class="grid">\n'
        for p in paths:
            html += f'<img src="file://{p}" alt="crop">\n'
        html += '</div>\n'
        html += f'<br><label>Species: <input type="text" name="cluster_{cluster_id}" placeholder="e.g. House Finch"></label>\n'
        html += '</div>\n'

    html += """
<button type="button" onclick="saveLabels()">Save Labels</button>
<pre id="output"></pre>
<script>
function saveLabels() {
    const data = {};
    const inputs = document.querySelectorAll('input[type=text]');
    inputs.forEach(input => {
        const cluster = input.name.replace('cluster_', '');
        data[cluster] = input.value.trim();
    });
    document.getElementById('output').textContent = JSON.stringify(data, null, 2);
    navigator.clipboard.writeText(JSON.stringify(data));
    alert('Labels copied to clipboard! Paste into terminal when prompted.');
}
</script>
</body></html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Labeling page saved to {output_path}")
    print(f"Open in browser: file://{output_path.resolve()}")


def apply_labels(
    labels: np.ndarray,
    species_map: dict[int, str],
    crop_index: pd.DataFrame,
    session_index: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply species labels to all crops and sessions."""
    crop_species = [species_map.get(int(l), "unknown") for l in labels]
    crop_index = crop_index.copy()
    crop_index["species"] = crop_species

    # Map crop species to sessions
    crop_to_species = dict(zip(crop_index["crop_id"], crop_index["species"]))
    session_species = []
    for _, row in session_index.iterrows():
        crop_ids = row["crop_ids"]
        if isinstance(crop_ids, str):
            crop_ids = crop_ids.split(",")
        species_votes = [crop_to_species.get(cid, "unknown") for cid in crop_ids]
        from collections import Counter
        most_common = Counter(species_votes).most_common(1)[0][0]
        session_species.append(most_common)

    session_index = session_index.copy()
    session_index["species"] = session_species

    return crop_index, session_index


def run_species_labeling(
    crop_embeddings: np.ndarray,
    crop_index: pd.DataFrame,
    session_index: pd.DataFrame,
):
    """Run the species labeling workflow."""
    print("Clustering crops for species labeling...")
    labels = cluster_for_species(crop_embeddings)

    print(f"Found {len(set(labels))} clusters")
    for cl in sorted(set(labels)):
        count = sum(1 for l in labels if l == cl)
        print(f"  Cluster {cl}: {count} crops")

    samples = get_sample_crops_per_cluster(labels, crop_index)
    html_path = DATA_DIR / "species_labeling.html"
    generate_labeling_html(samples, html_path)

    print("\n--- ACTION REQUIRED ---")
    print(f"1. Open in browser: file://{html_path.resolve()}")
    print("2. Label each cluster with a species name")
    print("3. Click 'Save Labels' (copies JSON to clipboard)")
    print("4. Paste the JSON below:\n")

    raw = input("Paste species labels JSON: ")
    import json
    species_map = {int(k): v for k, v in json.loads(raw).items()}
    print(f"\nLabels: {species_map}")

    crop_index, session_index = apply_labels(labels, species_map, crop_index, session_index)

    # Save
    species_labels_path = DATA_DIR / "species_labels.csv"
    crop_index[["crop_id", "species"]].to_csv(species_labels_path, index=False)
    session_index.to_csv(EMBEDDINGS_DIR / "session_index.csv", index=False)

    print(f"\nSpecies distribution (sessions):")
    print(session_index["species"].value_counts().to_string())
    print(f"\nSaved to {species_labels_path}")
    return crop_index, session_index


if __name__ == "__main__":
    crop_embeddings = np.load(EMBEDDINGS_DIR / "crop_embeddings.npy")
    crop_index = pd.read_csv(EMBEDDINGS_DIR / "crop_index.csv")
    session_index = pd.read_csv(EMBEDDINGS_DIR / "session_index.csv")
    run_species_labeling(crop_embeddings, crop_index, session_index)
