# pipeline/dashboard.py
"""Generate interactive bird re-identification dashboard as self-contained HTML."""
import json
import numpy as np
import pandas as pd
import umap
from pathlib import Path
from pipeline.config import CLUSTERS_DIR, EMBEDDINGS_DIR, MODELS_DIR
from pipeline.bird_names import assign_names


def _collect_bird_crops(bird_id, assignments, session_index, crop_path_map, max_crops=12):
    """Collect crop paths for a bird, sampling across sessions."""
    bird_sessions = assignments[assignments["bird_id"] == bird_id]
    all_crop_paths = []
    for _, row in bird_sessions.iterrows():
        srow = session_index[session_index["session_id"] == row["session_id"]]
        if len(srow) == 0:
            continue
        crop_ids = srow.iloc[0]["crop_ids"]
        if isinstance(crop_ids, str):
            crop_ids = crop_ids.split(",")
        for cid in crop_ids[:2]:
            cp = crop_path_map.get(cid)
            if cp:
                all_crop_paths.append(cp)
        if len(all_crop_paths) >= max_crops:
            break
    return all_crop_paths[:max_crops]


def _check_heatmap(crop_path, heatmaps_dir):
    """Check if an attention heatmap exists for a crop."""
    crop_id = Path(crop_path).stem
    heatmap = heatmaps_dir / f"{crop_id}_attn.jpg"
    if heatmap.exists():
        return str(heatmap.resolve())
    return None


def build_dashboard(
    output_path: Path = None,
    clusters_dir: Path = CLUSTERS_DIR,
    embeddings_dir: Path = EMBEDDINGS_DIR,
    models_dir: Path = MODELS_DIR,
):
    """Build interactive HTML dashboard."""
    if output_path is None:
        output_path = clusters_dir / "dashboard.html"

    print("Loading data...")
    assignments = pd.read_csv(clusters_dir / "cluster_assignments.csv")
    session_index = pd.read_csv(embeddings_dir / "session_index.csv")
    crop_index = pd.read_csv(embeddings_dir / "crop_index_v3.csv")
    session_embs = np.load(embeddings_dir / "session_embeddings.npy")

    with open(models_dir / "bird_database.json") as f:
        bird_db = json.load(f)

    crop_path_map = dict(zip(crop_index["crop_id"], crop_index["crop_path"]))
    heatmaps_dir = clusters_dir / "heatmaps"

    # UMAP projection
    print("Computing UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine")
    coords = reducer.fit_transform(session_embs)

    # Assign cute names
    all_bird_ids = list(bird_db.keys())
    species_map = {bid: info["species"] for bid, info in bird_db.items()}
    bird_names = assign_names(all_bird_ids, species_map)
    # Also name noise birds
    for bid in assignments["bird_id"].unique():
        if bid not in bird_names:
            bird_names[bid] = "Unknown Visitor"

    print(f"Assigned names to {len(bird_names)} birds")

    # Build JSON data for the dashboard
    # 1. UMAP points
    umap_points = []
    for i, (_, row) in enumerate(assignments.iterrows()):
        umap_points.append({
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "session_id": row["session_id"],
            "bird_id": row["bird_id"],
            "bird_name": bird_names.get(row["bird_id"], row["bird_id"]),
            "species": row["species"],
            "video_id": row["video_id"],
        })

    # 2. Bird profiles
    bird_profiles = {}
    for bird_id, info in bird_db.items():
        crop_paths = _collect_bird_crops(
            bird_id, assignments, session_index, crop_path_map,
        )
        crops_with_heatmaps = []
        for cp in crop_paths:
            hm = _check_heatmap(cp, heatmaps_dir)
            crops_with_heatmaps.append({
                "original": f"file://{cp}",
                "heatmap": f"file://{hm}" if hm else None,
            })

        bird_profiles[bird_id] = {
            "name": bird_names.get(bird_id, bird_id),
            "species": info["species"],
            "n_sessions": info["n_sessions"],
            "n_videos": info["n_videos"],
            "crops": crops_with_heatmaps,
        }

    # 3. Species summary
    species_summary = {}
    for species in sorted(assignments["species"].unique()):
        sp_df = assignments[assignments["species"] == species]
        birds = sp_df["bird_id"].unique()
        n_noise = sum(1 for b in birds if "noise" in b)
        species_summary[species] = {
            "n_birds": len(birds) - n_noise,
            "n_noise_birds": n_noise,
            "n_sessions": len(sp_df),
            "n_videos": sp_df["video_id"].nunique(),
        }

    # 4. Video timeline data
    video_birds = {}
    for vid in assignments["video_id"].unique():
        vid_df = assignments[assignments["video_id"] == vid]
        video_birds[vid] = sorted(vid_df["bird_id"].unique().tolist())

    dashboard_data = {
        "umap_points": umap_points,
        "bird_profiles": bird_profiles,
        "species_summary": species_summary,
        "video_birds": video_birds,
    }

    print("Generating dashboard HTML...")
    html = _generate_html(dashboard_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Dashboard saved to: {output_path}")
    print(f"Open in browser: file://{output_path.resolve()}")


def _generate_html(data):
    data_json = json.dumps(data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Feeder Friends - Bird Re-ID</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Nunito', sans-serif; background: #fef9f0; color: #4a3728; }}

/* Header */
.header {{
    background: linear-gradient(135deg, #f8e8d0 0%, #fce4c4 50%, #f5ddb8 100%);
    padding: 20px 32px;
    border-bottom: 2px solid #e8d0b0;
    display: flex; align-items: center; gap: 16px;
}}
.header h1 {{
    font-size: 26px; font-weight: 800; color: #6b4226;
    text-shadow: 0 1px 0 rgba(255,255,255,0.5);
}}
.header h1 .icon {{ font-size: 28px; }}
.header .stats {{
    font-size: 13px; color: #8b6b4a; display: flex; gap: 10px; margin-left: auto;
}}
.header .stats span {{
    background: rgba(255,255,255,0.6); padding: 5px 12px; border-radius: 20px;
    font-weight: 600; backdrop-filter: blur(4px);
}}

/* Filter bar */
.top-bar {{
    background: #fff8f0; padding: 14px 32px; border-bottom: 1px solid #ecdcc8;
    display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
}}
.filter-label {{ font-size: 13px; color: #a08060; margin-right: 4px; font-weight: 600; }}
.filter-btn {{
    padding: 6px 14px; border-radius: 20px; border: 2px solid #e0c8a8;
    background: transparent; color: #8b6b4a; cursor: pointer;
    font-size: 12px; font-weight: 700; font-family: 'Nunito', sans-serif;
    transition: all 0.25s ease;
}}
.filter-btn:hover {{ background: #fff0e0; border-color: #d0a878; transform: translateY(-1px); }}
.filter-btn.active {{ background: #6b4226; border-color: #6b4226; color: #fff; }}

/* Main layout */
.main-content {{ padding: 24px 32px; }}

/* View toggle */
.view-toggle {{
    display: flex; gap: 8px; margin-bottom: 20px;
}}
.view-btn {{
    padding: 8px 18px; border-radius: 20px; border: 2px solid #e0c8a8;
    background: transparent; color: #8b6b4a; cursor: pointer;
    font-size: 13px; font-weight: 700; font-family: 'Nunito', sans-serif;
    transition: all 0.25s ease;
}}
.view-btn:hover {{ background: #fff0e0; }}
.view-btn.active {{ background: #6b4226; border-color: #6b4226; color: #fff; }}

/* Bird card grid */
.bird-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 20px;
}}
.bird-card {{
    background: #fff; border-radius: 16px; overflow: hidden;
    box-shadow: 0 2px 12px rgba(107,66,38,0.08);
    border: 2px solid transparent;
    cursor: pointer; transition: all 0.3s ease;
}}
.bird-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(107,66,38,0.15);
    border-color: #e0c8a8;
}}
.bird-card .card-photos {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 2px;
    height: 180px; overflow: hidden;
}}
.bird-card .card-photos img {{
    width: 100%; height: 100%; object-fit: cover;
}}
.bird-card .card-photos.single {{ grid-template-columns: 1fr; }}
.bird-card .card-body {{ padding: 14px 16px; }}
.bird-card .card-name {{
    font-size: 18px; font-weight: 800; color: #6b4226; margin-bottom: 2px;
}}
.bird-card .card-species {{
    font-size: 12px; font-weight: 600; margin-bottom: 8px;
}}
.bird-card .card-stats {{
    display: flex; gap: 12px; font-size: 12px; color: #a08060;
}}
.bird-card .card-stats span {{ display: flex; align-items: center; gap: 4px; }}

/* Detail overlay */
.detail-overlay {{
    display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(74,55,40,0.4); backdrop-filter: blur(6px);
    z-index: 1000; justify-content: center; align-items: flex-start;
    padding: 40px; overflow-y: auto;
}}
.detail-overlay.visible {{ display: flex; }}
.detail-card {{
    background: #fff; border-radius: 20px; max-width: 720px; width: 100%;
    box-shadow: 0 20px 60px rgba(107,66,38,0.2);
    overflow: hidden; animation: slideUp 0.3s ease;
}}
@keyframes slideUp {{
    from {{ transform: translateY(30px); opacity: 0; }}
    to {{ transform: translateY(0); opacity: 1; }}
}}
.detail-hero {{
    position: relative; height: 280px; overflow: hidden;
}}
.detail-hero img {{
    width: 100%; height: 100%; object-fit: cover;
}}
.detail-hero .hero-overlay {{
    position: absolute; bottom: 0; left: 0; right: 0;
    background: linear-gradient(transparent, rgba(0,0,0,0.6));
    padding: 20px 24px 16px;
}}
.detail-hero .hero-name {{
    font-size: 28px; font-weight: 800; color: #fff;
    text-shadow: 0 2px 8px rgba(0,0,0,0.3);
}}
.detail-hero .hero-species {{
    font-size: 14px; font-weight: 600; color: rgba(255,255,255,0.85);
}}
.detail-close {{
    position: absolute; top: 16px; right: 16px;
    width: 36px; height: 36px; border-radius: 50%;
    background: rgba(255,255,255,0.9); border: none; cursor: pointer;
    font-size: 18px; color: #6b4226; display: flex; align-items: center;
    justify-content: center; transition: all 0.2s;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}}
.detail-close:hover {{ background: #fff; transform: scale(1.1); }}
.detail-body {{ padding: 24px; }}
.detail-stats {{
    display: flex; gap: 16px; margin-bottom: 24px;
}}
.detail-stat {{
    flex: 1; background: #fef5e8; border-radius: 12px; padding: 14px;
    text-align: center;
}}
.detail-stat .stat-value {{ font-size: 24px; font-weight: 800; color: #6b4226; }}
.detail-stat .stat-label {{ font-size: 11px; font-weight: 600; color: #a08060; text-transform: uppercase; letter-spacing: 0.5px; }}

.detail-section {{ margin-bottom: 24px; }}
.detail-section h3 {{
    font-size: 14px; font-weight: 700; color: #8b6b4a;
    text-transform: uppercase; letter-spacing: 0.5px;
    margin-bottom: 12px; padding-bottom: 8px;
    border-bottom: 2px solid #f0e0d0;
}}

/* Photo gallery in detail */
.photo-gallery {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px;
}}
.photo-item {{
    position: relative; border-radius: 10px; overflow: hidden;
    aspect-ratio: 1; cursor: pointer;
}}
.photo-item img {{
    width: 100%; height: 100%; object-fit: cover;
    transition: transform 0.3s;
}}
.photo-item:hover img {{ transform: scale(1.05); }}
.photo-item .heatmap-toggle {{
    position: absolute; bottom: 6px; right: 6px;
    background: rgba(107,66,38,0.8); border: none; color: #fff;
    font-size: 10px; padding: 4px 8px; border-radius: 8px; cursor: pointer;
    font-weight: 700; font-family: 'Nunito', sans-serif;
    backdrop-filter: blur(4px); transition: all 0.2s;
}}
.photo-item .heatmap-toggle:hover {{ background: #6b4226; }}
.photo-item .heatmap-toggle.showing-heatmap {{ background: #d4763a; }}

/* Friends list */
.friends-list {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.friend-chip {{
    display: flex; align-items: center; gap: 6px;
    padding: 6px 12px; border-radius: 20px;
    background: #fef5e8; border: 1px solid #e8d0b0;
    cursor: pointer; font-size: 12px; font-weight: 600;
    transition: all 0.2s;
}}
.friend-chip:hover {{ background: #fce4c4; transform: translateY(-1px); }}
.friend-chip .friend-count {{ color: #a08060; font-weight: 400; }}

/* UMAP panel (collapsible) */
.umap-section {{
    background: #fff; border-radius: 16px; padding: 20px;
    box-shadow: 0 2px 12px rgba(107,66,38,0.08);
    margin-bottom: 24px;
}}
.umap-header {{
    display: flex; justify-content: space-between; align-items: center;
    cursor: pointer; user-select: none;
}}
.umap-header h3 {{
    font-size: 14px; font-weight: 700; color: #8b6b4a;
    text-transform: uppercase; letter-spacing: 0.5px;
    border: none; padding: 0; margin: 0;
}}
.umap-toggle {{
    font-size: 13px; color: #a08060; font-weight: 600;
    transition: transform 0.3s;
}}
.umap-toggle.collapsed {{ transform: rotate(-90deg); }}
.umap-body {{ margin-top: 16px; display: none; }}
.umap-body.expanded {{ display: block; }}
.umap-canvas-wrap {{ position: relative; height: 400px; border-radius: 12px; overflow: hidden; background: #fef9f0; }}
.umap-canvas {{ width: 100%; height: 100%; cursor: crosshair; }}

/* Tooltip */
.tooltip {{
    position: fixed; background: #fff; padding: 10px 14px; border-radius: 10px;
    font-size: 12px; pointer-events: none; z-index: 2000;
    border: 2px solid #e8d0b0;
    box-shadow: 0 4px 16px rgba(107,66,38,0.15); display: none;
}}
.tooltip .tt-bird {{ color: #6b4226; font-weight: 800; font-size: 14px; }}
.tooltip .tt-species {{ font-weight: 600; }}
.tooltip .tt-meta {{ color: #a08060; }}

/* Species summary cards */
.species-cards {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px; margin-bottom: 24px;
}}
.species-card {{
    background: #fff; border-radius: 12px; padding: 16px;
    box-shadow: 0 2px 8px rgba(107,66,38,0.06);
    border-left: 4px solid;
    cursor: pointer; transition: all 0.2s;
}}
.species-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 16px rgba(107,66,38,0.12); }}
.species-card .sp-name {{ font-size: 14px; font-weight: 700; color: #6b4226; }}
.species-card .sp-stats {{ font-size: 12px; color: #a08060; margin-top: 4px; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 8px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: #dcc8a8; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: #c0a880; }}

/* Responsive */
@media (max-width: 768px) {{
    .bird-grid {{ grid-template-columns: 1fr 1fr; gap: 12px; }}
    .main-content {{ padding: 16px; }}
    .photo-gallery {{ grid-template-columns: repeat(3, 1fr); }}
}}
</style>
</head>
<body>

<div class="header">
    <h1><span class="icon">&#x1F426;</span> Feeder Friends</h1>
    <div class="stats" id="header-stats"></div>
</div>

<div class="top-bar" id="filter-bar">
    <span class="filter-label">Species:</span>
    <button class="filter-btn active" data-species="all">All Friends</button>
</div>

<div class="main-content">
    <div class="species-cards" id="species-cards"></div>
    <div class="bird-grid" id="bird-grid"></div>

    <div class="umap-section">
        <div class="umap-header" id="umap-header">
            <h3>&#x1F50D; Embedding Space</h3>
            <span class="umap-toggle collapsed" id="umap-toggle">&#x25BC;</span>
        </div>
        <div class="umap-body" id="umap-body">
            <div class="umap-canvas-wrap">
                <canvas class="umap-canvas" id="umap-canvas"></canvas>
            </div>
        </div>
    </div>
</div>

<div class="detail-overlay" id="detail-overlay">
    <div class="detail-card" id="detail-card">
        <div class="detail-hero" id="detail-hero">
            <img id="detail-hero-img" src="" alt="">
            <button class="detail-close" id="detail-close">&#x2715;</button>
            <div class="hero-overlay">
                <div class="hero-name" id="detail-name"></div>
                <div class="hero-species" id="detail-species"></div>
            </div>
        </div>
        <div class="detail-body">
            <div class="detail-stats" id="detail-stats"></div>
            <div class="detail-section">
                <h3>&#x1F4F8; Photo Gallery</h3>
                <div class="photo-gallery" id="detail-gallery"></div>
            </div>
            <div class="detail-section">
                <h3>&#x1F91D; Feeder Friends</h3>
                <div class="friends-list" id="detail-friends"></div>
            </div>
        </div>
    </div>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
const DATA = {data_json};

// --- State ---
let selectedSpecies = "all";
let selectedBird = null;

// --- Species colors (warm palette) ---
const SPECIES_COLORS = {{}};
const SPECIES_LIST = Object.keys(DATA.species_summary).sort();
const COLOR_PALETTE = [
    "#d4763a", "#6b8e4e", "#c45b84", "#8b6bb0", "#3d8fa6",
    "#c4873a", "#7a6b52", "#5b8e8e", "#9e5b5b", "#5b7e9e",
];
SPECIES_LIST.forEach((sp, i) => {{
    SPECIES_COLORS[sp] = COLOR_PALETTE[i % COLOR_PALETTE.length];
}});

// --- Header ---
function initHeader() {{
    const totalBirds = Object.keys(DATA.bird_profiles).length;
    const totalSessions = DATA.umap_points.length;
    const totalVideos = new Set(DATA.umap_points.map(p => p.video_id)).size;
    document.getElementById("header-stats").innerHTML =
        `<span>&#x1F425; ${{totalBirds}} birds</span><span>&#x1F3AC; ${{totalSessions}} sightings</span><span>&#x1F4F9; ${{totalVideos}} videos</span>`;
}}

// --- Species filters ---
function initFilters() {{
    const bar = document.getElementById("filter-bar");
    SPECIES_LIST.forEach(sp => {{
        const btn = document.createElement("button");
        btn.className = "filter-btn";
        btn.dataset.species = sp;
        btn.textContent = sp;
        btn.addEventListener("click", () => setSpeciesFilter(sp));
        bar.appendChild(btn);
    }});
    document.querySelector('[data-species="all"]').addEventListener("click", () => setSpeciesFilter("all"));
}}

function setSpeciesFilter(sp) {{
    selectedSpecies = sp;
    document.querySelectorAll(".filter-btn").forEach(b => {{
        b.classList.toggle("active", b.dataset.species === sp);
    }});
    renderSpeciesCards();
    renderBirdGrid();
    if (umapExpanded) drawUMAP();
}}

// --- Species cards ---
function renderSpeciesCards() {{
    const container = document.getElementById("species-cards");
    let html = "";
    for (const [sp, stats] of Object.entries(DATA.species_summary)) {{
        if (selectedSpecies !== "all" && sp !== selectedSpecies) continue;
        const color = SPECIES_COLORS[sp];
        html += `<div class="species-card" style="border-left-color:${{color}}" onclick="setSpeciesFilter('${{sp}}')">
            <div class="sp-name">${{sp}}</div>
            <div class="sp-stats">${{stats.n_birds}} individuals &middot; ${{stats.n_sessions}} sightings</div>
        </div>`;
    }}
    container.innerHTML = html;
}}

// --- Bird grid ---
function renderBirdGrid() {{
    const container = document.getElementById("bird-grid");
    const birds = Object.entries(DATA.bird_profiles)
        .filter(([id, info]) => selectedSpecies === "all" || info.species === selectedSpecies)
        .filter(([id]) => !id.includes("noise"))
        .sort((a, b) => b[1].n_sessions - a[1].n_sessions);

    let html = "";
    for (const [birdId, info] of birds) {{
        const color = SPECIES_COLORS[info.species];
        const photos = info.crops.slice(0, 4);
        const gridClass = photos.length === 1 ? "card-photos single" : "card-photos";

        html += `<div class="bird-card" onclick="openDetail('${{birdId}}')">`;
        html += `<div class="${{gridClass}}">`;
        for (const crop of photos) {{
            html += `<img src="${{crop.original}}" alt="${{info.name}}" loading="lazy">`;
        }}
        html += `</div>`;
        html += `<div class="card-body">`;
        html += `<div class="card-name">${{info.name}}</div>`;
        html += `<div class="card-species" style="color:${{color}}">${{info.species}}</div>`;
        html += `<div class="card-stats">`;
        html += `<span>&#x1F441; ${{info.n_sessions}} sightings</span>`;
        html += `<span>&#x1F4F9; ${{info.n_videos}} videos</span>`;
        html += `</div></div></div>`;
    }}
    container.innerHTML = html;
}}

// --- Detail overlay ---
function openDetail(birdId) {{
    selectedBird = birdId;
    const info = DATA.bird_profiles[birdId];
    if (!info) return;

    const overlay = document.getElementById("detail-overlay");
    overlay.classList.add("visible");
    document.body.style.overflow = "hidden";

    // Hero image
    const heroImg = document.getElementById("detail-hero-img");
    heroImg.src = info.crops.length > 0 ? info.crops[0].original : "";
    document.getElementById("detail-name").textContent = info.name;
    document.getElementById("detail-species").textContent = info.species;

    // Stats
    document.getElementById("detail-stats").innerHTML = `
        <div class="detail-stat"><div class="stat-value">${{info.n_sessions}}</div><div class="stat-label">Sightings</div></div>
        <div class="detail-stat"><div class="stat-value">${{info.n_videos}}</div><div class="stat-label">Videos</div></div>
        <div class="detail-stat"><div class="stat-value">${{info.crops.length}}</div><div class="stat-label">Photos</div></div>
    `;

    // Photo gallery with heatmap toggle
    const gallery = document.getElementById("detail-gallery");
    let gHtml = "";
    for (const crop of info.crops) {{
        const toggleBtn = crop.heatmap
            ? `<button class="heatmap-toggle" onclick="toggleHeatmap(event, this)">&#x1F525; Features</button>`
            : "";
        gHtml += `<div class="photo-item" data-original="${{crop.original}}" data-heatmap="${{crop.heatmap || ""}}">
            <img src="${{crop.original}}" alt="photo">
            ${{toggleBtn}}
        </div>`;
    }}
    gallery.innerHTML = gHtml;

    // Friends
    const friendsContainer = document.getElementById("detail-friends");
    const birdVideos = new Set();
    DATA.umap_points.filter(p => p.bird_id === birdId).forEach(p => birdVideos.add(p.video_id));
    const coOccur = {{}};
    for (const vid of birdVideos) {{
        const others = DATA.video_birds[vid] || [];
        for (const other of others) {{
            if (other !== birdId && !other.includes("noise")) {{
                coOccur[other] = (coOccur[other] || 0) + 1;
            }}
        }}
    }}
    const coList = Object.entries(coOccur).sort((a, b) => b[1] - a[1]).slice(0, 12);
    if (coList.length > 0) {{
        let fHtml = "";
        for (const [otherId, count] of coList) {{
            const otherInfo = DATA.bird_profiles[otherId];
            const color = SPECIES_COLORS[otherInfo?.species] || "#888";
            const name = otherInfo?.name || otherId;
            fHtml += `<div class="friend-chip" onclick="event.stopPropagation(); closeDetail(); setTimeout(() => openDetail('${{otherId}}'), 350);">
                <span style="color:${{color}}">${{name}}</span>
                <span class="friend-count">${{count}}x</span>
            </div>`;
        }}
        friendsContainer.innerHTML = fHtml;
    }} else {{
        friendsContainer.innerHTML = '<span style="color:#a08060; font-size:13px;">Prefers solo visits</span>';
    }}
}}

function closeDetail() {{
    selectedBird = null;
    document.getElementById("detail-overlay").classList.remove("visible");
    document.body.style.overflow = "";
}}

document.getElementById("detail-close").addEventListener("click", closeDetail);
document.getElementById("detail-overlay").addEventListener("click", (e) => {{
    if (e.target === document.getElementById("detail-overlay")) closeDetail();
}});

// --- Heatmap toggle ---
function toggleHeatmap(event, btn) {{
    event.stopPropagation();
    const item = btn.closest(".photo-item");
    const img = item.querySelector("img");
    const original = item.dataset.original;
    const heatmap = item.dataset.heatmap;
    if (!heatmap) return;

    if (!btn.classList.contains("showing-heatmap")) {{
        img.src = heatmap;
        btn.innerHTML = "&#x1F4F7; Photo";
        btn.classList.add("showing-heatmap");
    }} else {{
        img.src = original;
        btn.innerHTML = "&#x1F525; Features";
        btn.classList.remove("showing-heatmap");
    }}
}}

// --- UMAP (collapsible) ---
let umapExpanded = false;
const canvas = document.getElementById("umap-canvas");
const ctx = canvas.getContext("2d");
const tooltipEl = document.getElementById("tooltip");
let umapTransform = {{ scale: 1, offsetX: 0, offsetY: 0 }};
let isDragging = false;
let dragStart = {{ x: 0, y: 0 }};
let hoveredPoint = null;

document.getElementById("umap-header").addEventListener("click", () => {{
    umapExpanded = !umapExpanded;
    document.getElementById("umap-body").classList.toggle("expanded", umapExpanded);
    document.getElementById("umap-toggle").classList.toggle("collapsed", !umapExpanded);
    if (umapExpanded) {{ setTimeout(resizeCanvas, 50); }}
}});

function resizeCanvas() {{
    const wrap = canvas.parentElement;
    const rect = wrap.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    canvas.style.width = rect.width + "px";
    canvas.style.height = rect.height + "px";
    ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
    drawUMAP();
}}

function getVisiblePoints() {{
    return DATA.umap_points.filter(p => selectedSpecies === "all" || p.species === selectedSpecies);
}}

function mapToCanvas(x, y) {{
    const points = DATA.umap_points;
    const xs = points.map(p => p.x), ys = points.map(p => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const pad = 40;
    const w = canvas.width / window.devicePixelRatio - pad * 2;
    const h = canvas.height / window.devicePixelRatio - pad * 2;
    const scale = Math.min(w / (maxX - minX || 1), h / (maxY - minY || 1)) * umapTransform.scale;
    return {{
        cx: (x - (minX + maxX) / 2) * scale + canvas.width / window.devicePixelRatio / 2 + umapTransform.offsetX,
        cy: (y - (minY + maxY) / 2) * -scale + canvas.height / window.devicePixelRatio / 2 + umapTransform.offsetY,
    }};
}}

function drawUMAP() {{
    const w = canvas.width / window.devicePixelRatio;
    const h = canvas.height / window.devicePixelRatio;
    ctx.clearRect(0, 0, w, h);
    const visible = getVisiblePoints();
    for (const p of visible) {{
        const {{ cx, cy }} = mapToCanvas(p.x, p.y);
        const isHov = hoveredPoint && p.bird_id === hoveredPoint;
        ctx.beginPath();
        ctx.arc(cx, cy, isHov ? 5 : 3.5, 0, Math.PI * 2);
        ctx.fillStyle = (SPECIES_COLORS[p.species] || "#888") + (isHov ? "ff" : "99");
        ctx.fill();
        if (isHov) {{ ctx.strokeStyle = "#6b4226"; ctx.lineWidth = 1.5; ctx.stroke(); }}
    }}
}}

canvas.addEventListener("mousemove", (e) => {{
    if (isDragging) {{ umapTransform.offsetX += e.movementX; umapTransform.offsetY += e.movementY; drawUMAP(); return; }}
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    let closest = null, closestDist = 12;
    for (const p of getVisiblePoints()) {{
        const {{ cx, cy }} = mapToCanvas(p.x, p.y);
        const d = Math.hypot(cx - mx, cy - my);
        if (d < closestDist) {{ closest = p; closestDist = d; }}
    }}
    if (closest) {{
        hoveredPoint = closest.bird_id;
        const info = DATA.bird_profiles[closest.bird_id];
        tooltipEl.style.display = "block";
        tooltipEl.style.left = (e.clientX + 14) + "px";
        tooltipEl.style.top = (e.clientY - 10) + "px";
        tooltipEl.innerHTML = `<div class="tt-bird">${{info?.name || closest.bird_id}}</div>
            <div class="tt-species" style="color:${{SPECIES_COLORS[closest.species]}}">${{closest.species}}</div>
            <div class="tt-meta">${{info?.n_sessions || "?"}} sightings</div>`;
        drawUMAP();
    }} else if (hoveredPoint) {{
        hoveredPoint = null; tooltipEl.style.display = "none"; drawUMAP();
    }}
}});
canvas.addEventListener("mousedown", (e) => {{ isDragging = true; dragStart = {{x: e.clientX, y: e.clientY}}; }});
canvas.addEventListener("mouseup", (e) => {{
    isDragging = false;
    if (Math.abs(e.clientX - dragStart.x) < 3 && Math.abs(e.clientY - dragStart.y) < 3) {{
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;
        let closest = null, closestDist = 12;
        for (const p of getVisiblePoints()) {{
            const {{ cx, cy }} = mapToCanvas(p.x, p.y);
            if (Math.hypot(cx - mx, cy - my) < closestDist) {{ closest = p; closestDist = Math.hypot(cx - mx, cy - my); }}
        }}
        if (closest && !closest.bird_id.includes("noise")) openDetail(closest.bird_id);
    }}
}});
canvas.addEventListener("wheel", (e) => {{ e.preventDefault(); umapTransform.scale *= e.deltaY > 0 ? 0.9 : 1.1; umapTransform.scale = Math.max(0.5, Math.min(10, umapTransform.scale)); drawUMAP(); }});
canvas.addEventListener("mouseleave", () => {{ isDragging = false; hoveredPoint = null; tooltipEl.style.display = "none"; drawUMAP(); }});

// --- Keyboard ---
document.addEventListener("keydown", (e) => {{
    if (e.key === "Escape") {{
        if (selectedBird) closeDetail();
        else setSpeciesFilter("all");
    }}
}});

// --- Init ---
initHeader();
initFilters();
renderSpeciesCards();
renderBirdGrid();
window.addEventListener("resize", () => {{ if (umapExpanded) resizeCanvas(); }});
</script>
</body>
</html>"""


if __name__ == "__main__":
    build_dashboard()
