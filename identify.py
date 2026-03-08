#!/usr/bin/env python3
"""Identify birds in a video file or directory of videos.

Usage:
    python identify.py video.mp4
    python identify.py /path/to/videos/
    python identify.py video.mp4 --threshold 0.6
    python identify.py video.mp4 --json
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Identify birds in video(s)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python identify.py my_video.mp4
  python identify.py ~/bird_videos/
  python identify.py clip.mp4 --threshold 0.6
  python identify.py clip.mp4 --json > results.json
        """,
    )
    parser.add_argument("path", help="Video file or directory of videos")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Similarity threshold for matching (default: 0.50)")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()

    path = Path(args.path).expanduser().resolve()
    if not path.exists():
        print(f"Error: {path} does not exist", file=sys.stderr)
        sys.exit(1)

    # Collect video files
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if path.is_file():
        videos = [path]
    else:
        videos = sorted(p for p in path.iterdir() if p.suffix.lower() in video_exts)

    if not videos:
        print(f"No video files found in {path}", file=sys.stderr)
        sys.exit(1)

    # Load model once
    from pipeline.inference import load_inference_model, identify_birds

    if not args.json:
        print(f"Loading model...", file=sys.stderr)
    bundle = load_inference_model()

    all_results = {}
    for video_path in videos:
        if not args.json:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"Processing: {video_path.name}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)

        results = identify_birds(
            str(video_path), bundle,
            similarity_threshold=args.threshold,
        )

        all_results[str(video_path)] = results

        if not args.json:
            if not results:
                print("  No birds detected.")
                continue

            for i, r in enumerate(results):
                matched = r["bird_id"] != "new_candidate"
                name = r["bird_id"]

                # Look up cute name
                if matched and name in bundle["database"]:
                    db_entry = bundle["database"][name]
                    # Try to find cute name from bird_names
                    try:
                        from pipeline.bird_names import assign_names
                        species_map = {bid: info["species"] for bid, info in bundle["database"].items()}
                        names = assign_names(list(bundle["database"].keys()), species_map)
                        cute_name = names.get(name, name)
                    except Exception:
                        cute_name = name
                else:
                    cute_name = "Unknown visitor"

                icon = "✓" if matched else "?"
                print(f"\n  Bird {i+1}: [{icon}] {cute_name}")
                print(f"    Species:    {r['species']}")
                print(f"    Confidence: {r['similarity']:.0%}")
                print(f"    Crops:      {r['n_crops']}")
                if not matched:
                    top = r["top_3_matches"][0]
                    print(f"    Closest:    {top['bird_id']} ({top['similarity']:.0%})")

    if args.json:
        print(json.dumps(all_results, indent=2))
    elif len(videos) > 1:
        print(f"\n{'='*60}")
        total_birds = sum(len(r) for r in all_results.values())
        print(f"Processed {len(videos)} videos, detected {total_birds} birds total")


if __name__ == "__main__":
    main()
