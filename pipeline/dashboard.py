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
<title>Bird Re-ID Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f1117; color: #e0e0e0; }}

/* Header */
.header {{ background: #1a1d27; padding: 16px 24px; border-bottom: 1px solid #2a2d37; display: flex; align-items: center; gap: 16px; }}
.header h1 {{ font-size: 20px; font-weight: 600; color: #fff; }}
.header .stats {{ font-size: 13px; color: #888; display: flex; gap: 16px; }}
.header .stats span {{ background: #2a2d37; padding: 4px 10px; border-radius: 12px; }}

/* Layout */
.container {{ display: grid; grid-template-columns: 1fr 380px; grid-template-rows: auto 1fr; height: calc(100vh - 60px); }}
.top-bar {{ grid-column: 1 / -1; background: #1a1d27; padding: 12px 24px; border-bottom: 1px solid #2a2d37; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}

/* Species filters */
.filter-btn {{ padding: 5px 12px; border-radius: 16px; border: 1px solid #3a3d47; background: transparent; color: #aaa; cursor: pointer; font-size: 12px; transition: all 0.2s; }}
.filter-btn:hover {{ border-color: #5a5d67; color: #ddd; }}
.filter-btn.active {{ background: #2563eb; border-color: #2563eb; color: #fff; }}
.filter-label {{ font-size: 12px; color: #666; margin-right: 4px; }}

/* UMAP panel */
.umap-panel {{ padding: 16px; overflow: hidden; position: relative; }}
.umap-canvas {{ width: 100%; height: 100%; cursor: crosshair; }}

/* Sidebar */
.sidebar {{ background: #1a1d27; border-left: 1px solid #2a2d37; overflow-y: auto; padding: 0; }}
.sidebar-section {{ padding: 16px; border-bottom: 1px solid #2a2d37; }}
.sidebar-section h3 {{ font-size: 13px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }}

/* Bird card */
.bird-card {{ background: #22252f; border-radius: 8px; padding: 14px; margin-bottom: 10px; cursor: pointer; border: 1px solid transparent; transition: all 0.2s; }}
.bird-card:hover {{ border-color: #3a3d47; }}
.bird-card.selected {{ border-color: #2563eb; background: #1e2338; }}
.bird-card .bird-name {{ font-size: 14px; font-weight: 600; color: #fff; margin-bottom: 4px; }}
.bird-card .bird-meta {{ font-size: 12px; color: #888; }}
.bird-card .bird-species {{ font-size: 11px; color: #2563eb; margin-bottom: 4px; }}

/* Detail panel */
.detail-panel {{ display: none; }}
.detail-panel.visible {{ display: block; }}
.detail-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
.detail-header h2 {{ font-size: 16px; color: #fff; }}
.back-btn {{ background: none; border: none; color: #2563eb; cursor: pointer; font-size: 13px; }}
.crop-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-bottom: 12px; }}
.crop-item {{ position: relative; }}
.crop-item img {{ width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 4px; border: 1px solid #2a2d37; }}
.crop-item .toggle-heatmap {{ position: absolute; top: 4px; right: 4px; background: rgba(0,0,0,0.7); border: none; color: #fff; font-size: 10px; padding: 2px 6px; border-radius: 4px; cursor: pointer; }}
.crop-item .toggle-heatmap:hover {{ background: #2563eb; }}

/* Species stats */
.species-stat {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #2a2d37; }}
.species-stat:last-child {{ border-bottom: none; }}
.species-stat .name {{ font-size: 13px; color: #ddd; }}
.species-stat .count {{ font-size: 13px; color: #888; }}

/* Bird list */
.bird-list {{ max-height: 50vh; overflow-y: auto; }}

/* Tooltip */
.tooltip {{ position: absolute; background: #2a2d37; padding: 8px 12px; border-radius: 6px; font-size: 12px; pointer-events: none; z-index: 100; border: 1px solid #3a3d47; box-shadow: 0 4px 12px rgba(0,0,0,0.4); display: none; }}
.tooltip .tt-bird {{ color: #fff; font-weight: 600; }}
.tooltip .tt-species {{ color: #2563eb; }}
.tooltip .tt-meta {{ color: #888; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: #3a3d47; border-radius: 3px; }}
</style>
</head>
<body>

<div class="header">
    <h1>Bird Re-Identification Dashboard</h1>
    <div class="stats" id="header-stats"></div>
</div>

<div class="container">
    <div class="top-bar">
        <span class="filter-label">Species:</span>
        <button class="filter-btn active" data-species="all">All</button>
    </div>

    <div class="umap-panel">
        <canvas class="umap-canvas" id="umap-canvas"></canvas>
        <div class="tooltip" id="tooltip"></div>
    </div>

    <div class="sidebar">
        <div id="list-view">
            <div class="sidebar-section">
                <h3>Species Summary</h3>
                <div id="species-stats"></div>
            </div>
            <div class="sidebar-section">
                <h3>Individuals <span id="bird-count"></span></h3>
                <div class="bird-list" id="bird-list"></div>
            </div>
        </div>

        <div class="detail-panel" id="detail-view">
            <div class="sidebar-section">
                <div class="detail-header">
                    <h2 id="detail-name"></h2>
                    <button class="back-btn" id="back-btn">Back</button>
                </div>
                <div id="detail-species" class="bird-species" style="margin-bottom:8px;"></div>
                <div id="detail-meta" class="bird-meta" style="margin-bottom:14px;"></div>
                <h3 style="margin-bottom:8px;">Example Crops</h3>
                <div class="crop-grid" id="detail-crops"></div>
                <h3 style="margin-bottom:8px;">Co-occurring Birds</h3>
                <div id="detail-cooccur"></div>
            </div>
        </div>
    </div>
</div>

<script>
const DATA = {data_json};

// --- State ---
let selectedSpecies = "all";
let selectedBird = null;
let hoveredPoint = null;

// --- Species colors ---
const SPECIES_COLORS = {{}};
const SPECIES_LIST = Object.keys(DATA.species_summary).sort();
const COLOR_PALETTE = [
    "#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed",
    "#db2777", "#0891b2", "#65a30d", "#ea580c", "#4f46e5",
];
SPECIES_LIST.forEach((sp, i) => {{
    SPECIES_COLORS[sp] = COLOR_PALETTE[i % COLOR_PALETTE.length];
}});

// --- Bird colors (unique per bird within species) ---
function getBirdColor(birdId) {{
    const sp = DATA.bird_profiles[birdId]?.species || "";
    const baseColor = SPECIES_COLORS[sp] || "#888";
    // Vary brightness per bird
    const birdNum = parseInt(birdId.match(/\\d+$/)?.[0] || "0");
    const lighten = (birdNum % 5) * 12;
    return baseColor;  // Keep species color for UMAP, distinguish in list
}}

// --- Init header stats ---
function initHeader() {{
    const totalBirds = Object.keys(DATA.bird_profiles).length;
    const totalSessions = DATA.umap_points.length;
    const totalVideos = new Set(DATA.umap_points.map(p => p.video_id)).size;
    document.getElementById("header-stats").innerHTML =
        `<span>${{totalBirds}} individuals</span><span>${{totalSessions}} sessions</span><span>${{totalVideos}} videos</span><span>${{SPECIES_LIST.length}} species</span>`;
}}

// --- Species filters ---
function initFilters() {{
    const bar = document.querySelector(".top-bar");
    SPECIES_LIST.forEach(sp => {{
        const btn = document.createElement("button");
        btn.className = "filter-btn";
        btn.dataset.species = sp;
        btn.textContent = sp;
        btn.style.borderColor = SPECIES_COLORS[sp];
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
    renderBirdList();
    drawUMAP();
}}

// --- Species stats ---
function initSpeciesStats() {{
    const container = document.getElementById("species-stats");
    let html = "";
    for (const [sp, stats] of Object.entries(DATA.species_summary)) {{
        html += `<div class="species-stat">
            <span class="name" style="color:${{SPECIES_COLORS[sp]}}">${{sp}}</span>
            <span class="count">${{stats.n_birds}} birds / ${{stats.n_sessions}} sessions</span>
        </div>`;
    }}
    container.innerHTML = html;
}}

// --- Bird list ---
function renderBirdList() {{
    const container = document.getElementById("bird-list");
    const birds = Object.entries(DATA.bird_profiles)
        .filter(([id, info]) => selectedSpecies === "all" || info.species === selectedSpecies)
        .filter(([id]) => !id.includes("noise"))
        .sort((a, b) => b[1].n_sessions - a[1].n_sessions);

    document.getElementById("bird-count").textContent = `(${{birds.length}})`;

    let html = "";
    for (const [birdId, info] of birds) {{
        const sel = selectedBird === birdId ? "selected" : "";
        html += `<div class="bird-card ${{sel}}" data-bird="${{birdId}}" onclick="selectBird('${{birdId}}')">
            <div class="bird-species" style="color:${{SPECIES_COLORS[info.species]}}">${{info.species}}</div>
            <div class="bird-name">${{info.name}}</div>
            <div class="bird-meta">${{info.n_sessions}} sessions across ${{info.n_videos}} videos</div>
        </div>`;
    }}
    container.innerHTML = html;
}}

// --- Bird detail ---
function selectBird(birdId) {{
    selectedBird = birdId;
    const info = DATA.bird_profiles[birdId];
    if (!info) return;

    document.getElementById("list-view").style.display = "none";
    document.getElementById("detail-view").classList.add("visible");

    document.getElementById("detail-name").textContent = info.name;
    document.getElementById("detail-species").textContent = info.species;
    document.getElementById("detail-species").style.color = SPECIES_COLORS[info.species];
    document.getElementById("detail-meta").textContent =
        `${{info.n_sessions}} sessions across ${{info.n_videos}} videos`;

    // Crops
    const cropsContainer = document.getElementById("detail-crops");
    let cropsHtml = "";
    for (const crop of info.crops) {{
        const heatmapBtn = crop.heatmap
            ? `<button class="toggle-heatmap" onclick="toggleHeatmap(event, this)">Attention</button>`
            : "";
        cropsHtml += `<div class="crop-item" data-original="${{crop.original}}" data-heatmap="${{crop.heatmap || ""}}">
            <img src="${{crop.original}}" alt="crop">
            ${{heatmapBtn}}
        </div>`;
    }}
    cropsContainer.innerHTML = cropsHtml;

    // Co-occurring birds
    const coContainer = document.getElementById("detail-cooccur");
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
    const coList = Object.entries(coOccur).sort((a, b) => b[1] - a[1]).slice(0, 10);
    if (coList.length > 0) {{
        let coHtml = "";
        for (const [otherId, count] of coList) {{
            const otherInfo = DATA.bird_profiles[otherId];
            const sp = otherInfo?.species || "";
            const otherName = otherInfo?.name || otherId;
            coHtml += `<div class="species-stat" style="cursor:pointer" onclick="selectBird('${{otherId}}')">
                <span class="name" style="color:${{SPECIES_COLORS[sp]}}">${{otherName}}</span>
                <span class="count">${{count}} shared videos</span>
            </div>`;
        }}
        coContainer.innerHTML = coHtml;
    }} else {{
        coContainer.innerHTML = '<div class="bird-meta">No co-occurring birds found</div>';
    }}

    drawUMAP();
}}

function goBack() {{
    selectedBird = null;
    document.getElementById("list-view").style.display = "block";
    document.getElementById("detail-view").classList.remove("visible");
    renderBirdList();
    drawUMAP();
}}
document.getElementById("back-btn").addEventListener("click", goBack);

// --- Heatmap toggle ---
function toggleHeatmap(event, btn) {{
    event.stopPropagation();
    const item = btn.closest(".crop-item");
    const img = item.querySelector("img");
    const original = item.dataset.original;
    const heatmap = item.dataset.heatmap;
    if (!heatmap) return;

    if (img.src === original || img.getAttribute("src") === original) {{
        img.src = heatmap;
        btn.textContent = "Original";
        btn.style.background = "#2563eb";
    }} else {{
        img.src = original;
        btn.textContent = "Attention";
        btn.style.background = "rgba(0,0,0,0.7)";
    }}
}}

// --- UMAP Canvas ---
const canvas = document.getElementById("umap-canvas");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");

let transform = {{ scale: 1, offsetX: 0, offsetY: 0 }};
let isDragging = false;
let dragStart = {{ x: 0, y: 0 }};

function resizeCanvas() {{
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    canvas.style.width = rect.width + "px";
    canvas.style.height = rect.height + "px";
    ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
    drawUMAP();
}}

function getVisiblePoints() {{
    return DATA.umap_points.filter(p => {{
        if (selectedSpecies !== "all" && p.species !== selectedSpecies) return false;
        return true;
    }});
}}

function mapToCanvas(x, y) {{
    const points = DATA.umap_points;
    const xs = points.map(p => p.x);
    const ys = points.map(p => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const padding = 40;

    const w = canvas.width / window.devicePixelRatio - padding * 2;
    const h = canvas.height / window.devicePixelRatio - padding * 2;

    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const scale = Math.min(w / rangeX, h / rangeY) * transform.scale;

    const cx = (x - (minX + maxX) / 2) * scale + canvas.width / window.devicePixelRatio / 2 + transform.offsetX;
    const cy = (y - (minY + maxY) / 2) * -scale + canvas.height / window.devicePixelRatio / 2 + transform.offsetY;
    return {{ cx, cy }};
}}

function drawUMAP() {{
    const w = canvas.width / window.devicePixelRatio;
    const h = canvas.height / window.devicePixelRatio;
    ctx.clearRect(0, 0, w, h);

    const visible = getVisiblePoints();

    // Draw non-selected points first (dimmed)
    for (const p of visible) {{
        const {{ cx, cy }} = mapToCanvas(p.x, p.y);
        const isSelected = selectedBird && p.bird_id === selectedBird;
        const isHighlighted = hoveredPoint && p.bird_id === hoveredPoint;

        if (isSelected || isHighlighted) continue;

        ctx.beginPath();
        ctx.arc(cx, cy, selectedBird ? 2 : 3, 0, Math.PI * 2);
        const color = SPECIES_COLORS[p.species] || "#888";
        ctx.fillStyle = selectedBird ? color + "30" : color + "aa";
        ctx.fill();
    }}

    // Draw selected/highlighted points on top
    for (const p of visible) {{
        const {{ cx, cy }} = mapToCanvas(p.x, p.y);
        const isSelected = selectedBird && p.bird_id === selectedBird;
        const isHighlighted = hoveredPoint && p.bird_id === hoveredPoint;

        if (!isSelected && !isHighlighted) continue;

        ctx.beginPath();
        ctx.arc(cx, cy, isSelected ? 5 : 4, 0, Math.PI * 2);
        ctx.fillStyle = SPECIES_COLORS[p.species] || "#888";
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }}
}}

// --- Mouse interactions ---
canvas.addEventListener("mousemove", (e) => {{
    if (isDragging) {{
        transform.offsetX += e.movementX;
        transform.offsetY += e.movementY;
        drawUMAP();
        return;
    }}

    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    let closest = null;
    let closestDist = 10;
    for (const p of getVisiblePoints()) {{
        const {{ cx, cy }} = mapToCanvas(p.x, p.y);
        const d = Math.sqrt((cx - mx) ** 2 + (cy - my) ** 2);
        if (d < closestDist) {{
            closest = p;
            closestDist = d;
        }}
    }}

    if (closest) {{
        hoveredPoint = closest.bird_id;
        tooltip.style.display = "block";
        tooltip.style.left = (e.clientX - canvas.parentElement.getBoundingClientRect().left + 12) + "px";
        tooltip.style.top = (e.clientY - canvas.parentElement.getBoundingClientRect().top - 10) + "px";
        const info = DATA.bird_profiles[closest.bird_id];
        const displayName = info?.name || closest.bird_name || closest.bird_id;
        tooltip.innerHTML = `
            <div class="tt-bird">${{displayName}}</div>
            <div class="tt-species">${{closest.species}}</div>
            <div class="tt-meta">${{info ? info.n_sessions + " sessions" : closest.session_id}}</div>
        `;
        drawUMAP();
    }} else {{
        if (hoveredPoint) {{
            hoveredPoint = null;
            tooltip.style.display = "none";
            drawUMAP();
        }}
    }}
}});

canvas.addEventListener("mousedown", (e) => {{
    isDragging = true;
    dragStart = {{ x: e.clientX, y: e.clientY }};
}});

canvas.addEventListener("mouseup", (e) => {{
    const dx = e.clientX - dragStart.x;
    const dy = e.clientY - dragStart.y;
    isDragging = false;

    // If it was a click (not drag), select bird
    if (Math.abs(dx) < 3 && Math.abs(dy) < 3) {{
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        let closest = null;
        let closestDist = 10;
        for (const p of getVisiblePoints()) {{
            const {{ cx, cy }} = mapToCanvas(p.x, p.y);
            const d = Math.sqrt((cx - mx) ** 2 + (cy - my) ** 2);
            if (d < closestDist) {{
                closest = p;
                closestDist = d;
            }}
        }}

        if (closest && !closest.bird_id.includes("noise")) {{
            selectBird(closest.bird_id);
        }}
    }}
}});

canvas.addEventListener("wheel", (e) => {{
    e.preventDefault();
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    transform.scale *= factor;
    transform.scale = Math.max(0.5, Math.min(10, transform.scale));
    drawUMAP();
}});

canvas.addEventListener("mouseleave", () => {{
    isDragging = false;
    hoveredPoint = null;
    tooltip.style.display = "none";
    drawUMAP();
}});

// --- Keyboard ---
document.addEventListener("keydown", (e) => {{
    if (e.key === "Escape") {{
        if (selectedBird) goBack();
        else setSpeciesFilter("all");
    }}
}});

// --- Init ---
initHeader();
initFilters();
initSpeciesStats();
renderBirdList();
window.addEventListener("resize", resizeCanvas);
resizeCanvas();
</script>
</body>
</html>"""


if __name__ == "__main__":
    build_dashboard()
