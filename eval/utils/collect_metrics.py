"""
Aggregate overall_metrics JSON files produced by eval_camera.py / eval.py.

Usage:
    # Single file
    python eval/utils/collect_metrics.py path/to/overall_metrics_*.json

    # Multiple files / glob
    python eval/utils/collect_metrics.py eval_cam_results/tum/causalvggt/overall_metrics/*.json

    # Recursive: find all overall_metrics*.json under a directory
    python eval/utils/collect_metrics.py --dir eval_cam_results/
"""

import argparse
import json
import glob
import os
import sys
from collections import defaultdict


TRAJ_METRICS = ["ate", "rpe_rot", "rpe_trans"]
TRAJ_STATS   = ["mean", "rmse", "std", "median"]
TIME_KEYS    = ["time_seconds", "fps"]
MEMORY_KEYS  = ["peak_memory_used"]


def _mean(values):
    return sum(values) / len(values) if values else float("nan")


def aggregate_file(path: str) -> dict:
    """Return per-scene rows and averaged summary for one JSON file."""
    with open(path) as f:
        data = json.load(f)

    rows = []
    accum = defaultdict(lambda: defaultdict(list))  # metric -> stat -> [values]
    time_accum  = defaultdict(list)
    mem_accum   = defaultdict(list)

    for scene, info in data.items():
        traj = info.get("trajectory", {})
        timecost = info.get("timecost", {})
        memory   = info.get("memory", {})

        row = {"scene": scene}

        # --- trajectory metrics ---
        for metric in TRAJ_METRICS:
            m = traj.get(metric, {})
            for stat in TRAJ_STATS:
                v = m.get(stat)
                key = f"{metric}.{stat}"
                row[key] = v
                if v is not None:
                    accum[metric][stat].append(v)

        # --- timing ---
        for k in TIME_KEYS:
            v = timecost.get(k)
            row[k] = v
            if v is not None:
                time_accum[k].append(v)

        # --- memory ---
        for k in MEMORY_KEYS:
            v = memory.get(k)
            row[k] = v
            if v is not None:
                mem_accum[k].append(v)

        rows.append(row)

    # Build averaged summary
    summary = {"scene": f"MEAN ({len(rows)} scenes)"}
    for metric in TRAJ_METRICS:
        for stat in TRAJ_STATS:
            vals = accum[metric][stat]
            summary[f"{metric}.{stat}"] = _mean(vals)
    for k in TIME_KEYS:
        summary[k] = _mean(time_accum[k])
    for k in MEMORY_KEYS:
        summary[k] = _mean(mem_accum[k])

    return {"path": path, "rows": rows, "summary": summary}


def print_table(result: dict, show_scenes: bool = True):
    path    = result["path"]
    rows    = result["rows"]
    summary = result["summary"]

    # Determine columns (keep insertion order)
    cols = list(summary.keys())

    # Column widths
    col_w = {c: max(len(c), 8) for c in cols}
    for row in rows + [summary]:
        for c in cols:
            v = row.get(c)
            s = f"{v:.4f}" if isinstance(v, float) else str(v) if v is not None else "-"
            col_w[c] = max(col_w[c], len(s))

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v) if v is not None else "-"

    sep   = "  "
    header = sep.join(c.ljust(col_w[c]) for c in cols)
    hline  = sep.join("-" * col_w[c] for c in cols)

    print(f"\n{'='*80}")
    print(f"File : {path}")
    print(f"{'='*80}")
    print(header)
    print(hline)

    if show_scenes:
        for row in rows:
            line = sep.join(fmt(row.get(c)).ljust(col_w[c]) for c in cols)
            print(line)
        print(hline)

    # Summary row
    line = sep.join(fmt(summary.get(c)).ljust(col_w[c]) for c in cols)
    print(line)
    print()

    # Compact key-metric summary
    print("  Key metrics (mean over scenes):")
    for metric in TRAJ_METRICS:
        rmse = summary.get(f"{metric}.rmse", float("nan"))
        mean = summary.get(f"{metric}.mean", float("nan"))
        print(f"    {metric:<12}  rmse={rmse:.4f}  mean={mean:.4f}")
    fps = summary.get("fps")
    mem = summary.get("peak_memory_used")
    if fps is not None:
        print(f"    fps          = {fps:.2f}")
    if mem is not None:
        print(f"    peak_mem(MB) = {mem:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate eval_camera metrics JSON files")
    parser.add_argument("files", nargs="*",
                        help="JSON files or glob patterns")
    parser.add_argument("--dir", type=str, default=None,
                        help="Recursively find all overall_metrics*.json under this directory")
    parser.add_argument("--no-scenes", action="store_true",
                        help="Only print the averaged summary row, skip per-scene rows")
    args = parser.parse_args()

    paths = []
    if args.dir:
        paths = sorted(glob.glob(os.path.join(args.dir, "**", "overall_metrics*.json"),
                                 recursive=True))
    for pattern in args.files:
        matched = sorted(glob.glob(pattern, recursive=True))
        if matched:
            paths.extend(matched)
        elif os.path.isfile(pattern):
            paths.append(pattern)

    paths = list(dict.fromkeys(paths))  # deduplicate, preserve order

    if not paths:
        print("No JSON files found.", file=sys.stderr)
        sys.exit(1)

    for path in paths:
        result = aggregate_file(path)
        print_table(result, show_scenes=not args.no_scenes)


if __name__ == "__main__":
    main()
