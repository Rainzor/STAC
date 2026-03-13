"""
Compare reconstruction metrics across multiple overall_metrics JSON files.
Displays mean and median in adjacent columns (e.g. Acc_mean, Acc_med) per run.
Usage: python compare_overall_metrics.py [--dir DIR] [--metric METRIC] [--scene NAME]
"""
import argparse
import json
from pathlib import Path

# Metric groups: (display_name, mean_key, median_key)
METRIC_GROUPS = [
    ("Acc", "accuracy", "accuracy_median"),
    ("Comp", "completion", "completion_median"),
    ("NC1", "normal_consistency_1", "normal_consistency_1_median"),
    ("NC2", "normal_consistency_2", "normal_consistency_2_median"),
]

ALL_KEYS = []
for _dn, mk, mdk in METRIC_GROUPS:
    ALL_KEYS.append(mk)
    ALL_KEYS.append(mdk)

# Lower is better for acc/comp; higher is better for normal_consistency
LOWER_BETTER = {"accuracy", "completion", "accuracy_median", "completion_median"}


def short_name(filename: str) -> str:
    """Strip path, date suffix and .json to get a compact run label."""
    stem = Path(filename).stem  # e.g. overall_metrics_w4h2r2c4_20260310_2230
    parts = stem.split("_")
    cleaned = []
    for p in parts:
        if p in ("overall", "metrics"):
            continue
        if len(p) == 8 and p.isdigit():
            break
        cleaned.append(p)
    return "_".join(cleaned) if cleaned else stem


def load_files(directory: str):
    directory = Path(directory)
    runs = {}
    for f in sorted(directory.glob("*.json")):
        if f.name == "compare_results.json":
            continue
        with open(f) as fp:
            data = json.load(fp)
        label = short_name(f.name)
        runs[label] = {"path": str(f), "data": data}
    return runs


def extract_metrics(data: dict):
    """Return {scene: {metric: value}}."""
    scene_metrics = {}
    for scene, entry in data.items():
        if not isinstance(entry, dict) or "reconstruction" not in entry:
            continue
        scene_metrics[scene] = entry["reconstruction"]
    return scene_metrics


def _get_timing(entry: dict) -> dict | None:
    """Get timing dict from a scene entry (supports Time(ms) or timing)."""
    return entry.get("Time(ms)") or entry.get("timing")


def _get_memory(entry: dict) -> dict | None:
    """Get memory_details dict from a scene entry (supports VoxelCache/Memory(MB) or merger/memory_details)."""
    vc = entry.get("VoxelCache") or entry.get("merger") or {}
    return vc.get("Memory(MB)") or vc.get("memory_details")


# Backbone-only time: aggregator infer + KV position + retrieval + prune_merge
BACKBONE_TIME_KEYS = (
    "aggregator_infer_time",
    "kv_position_time",
    "kv_retrieval_time",
    "kv_evict_merge_time",
)
# Memory: use backend total_usage/total_alloc; actual = temporal + spatial (attention working set, MB)
# Supports new keys (temporal_cache_usage, spatial_cache_usage) and legacy (hot_cache_usage, retrieval_memory)


def compute_time_stats(timing: dict) -> dict | None:
    """From timing dict compute total_time_ms, fps, backbone_time_ms, backbone_fps."""
    if not timing:
        return None
    fps = timing.get("infer_fps")
    if fps is not None and fps > 0:
        total_ms = 1000.0 / fps
    else:
        total_ms = sum(
            timing.get(k, 0) for k in timing if k != "infer_fps" and k.endswith("_time")
        )
    backbone_ms = sum(timing.get(k, 0) for k in BACKBONE_TIME_KEYS)
    return {
        "total_time_ms": total_ms,
        "backbone_time_ms": backbone_ms,
    }


def compute_mem_stats(mem: dict) -> dict | None:
    """From memory_details: total_usage from backend; actual = temporal + spatial (MB)."""
    if not mem:
        return None
    total_usage = mem.get("total_usage")
    actual = (
        mem.get("temporal_cache_usage", 0) + mem.get("`spatial_cache_usage`", 0)
    )
    return {
        "total_usage_mb": total_usage,
        "actual_mem_mb": actual,
    }


def compute_averages(scene_metrics: dict) -> dict:
    totals = {}
    counts = {}
    for metrics in scene_metrics.values():
        for k, v in metrics.items():
            if k in ALL_KEYS:
                totals[k] = totals.get(k, 0.0) + v
                counts[k] = counts.get(k, 0) + 1
    return {k: totals[k] / counts[k] for k in totals}


def rank_values(vals: list, metric: str):
    """Return (best, second_best) scalar values."""
    if not vals:
        return None, None
    lower_better = metric in LOWER_BETTER
    sorted_vals = sorted(vals, reverse=not lower_better)
    best = sorted_vals[0]
    second = sorted_vals[1] if len(sorted_vals) > 1 else best
    return best, second


def format_cell(v, best_val, second_val, width=10):
    """Return (raw_str, colored_str) for alignment and display. Bold green best, underline yellow second."""
    if v is None:
        raw = "N/A"
    else:
        raw = f"{v:.5f}"
    if v is not None and v == best_val:
        colored = f"\033[1;32m{raw}\033[0m"
    elif v is not None and v == second_val and v != best_val:
        colored = f"\033[4;33m{raw}\033[0m"
    else:
        colored = raw
    padded = colored + " " * (width - len(raw))
    return padded


def print_table(runs: dict, metric_groups: list, scene_filter=None):
    run_labels = list(runs.keys())

    # Per-run averages
    avg_per_run = {}
    for label, info in runs.items():
        sm = extract_metrics(info["data"])
        avg_per_run[label] = compute_averages(sm)

    col_w = 10
    metric_w = max(4, max(len(dn) for dn, _, _ in metric_groups) + 1)

    # Header: Metric | run_mean run_med | run_mean run_med | ...
    def build_header():
        parts = [f"{'Metric':<{metric_w}}"]
        for label in run_labels:
            parts.append(f"{label}_mean".ljust(col_w))
            parts.append(f"{label}_med".ljust(col_w))
        return "  ".join(parts)

    header = build_header()
    sep_len = len(header)

    print("\n" + "=" * sep_len)
    print("  AVERAGE METRICS ACROSS ALL SCENES (mean | med adjacent per run)")
    print("=" * sep_len)
    print(header)
    print("-" * sep_len)

    for display_name, mean_key, med_key in metric_groups:
        vals_mean = [avg_per_run[l].get(mean_key) for l in run_labels]
        vals_med = [avg_per_run[l].get(med_key) for l in run_labels]
        valid_mean = [v for v in vals_mean if v is not None]
        valid_med = [v for v in vals_med if v is not None]
        best_mean, second_mean = rank_values(valid_mean, mean_key)
        best_med, second_med = rank_values(valid_med, med_key)

        row = f"{display_name:<{metric_w}}"
        for i, label in enumerate(run_labels):
            vm = vals_mean[i]
            vd = vals_med[i]
            row += "  " + format_cell(vm, best_mean, second_mean, col_w) + format_cell(vd, best_med, second_med, col_w)
        print(row)

    # Per-scene breakdown
    all_scenes = set()
    for info in runs.values():
        all_scenes.update(extract_metrics(info["data"]).keys())
    all_scenes = sorted(all_scenes)

    if scene_filter:
        all_scenes = [s for s in all_scenes if scene_filter.lower() in s.lower()]

    for scene in all_scenes:
        print("\n" + "=" * sep_len)
        print(f"  SCENE: {scene}")
        print("=" * sep_len)
        print(header)
        print("-" * sep_len)

        for display_name, mean_key, med_key in metric_groups:
            vals_mean = []
            vals_med = []
            for label in run_labels:
                sm = extract_metrics(runs[label]["data"])
                m = sm.get(scene, {})
                vals_mean.append(m.get(mean_key))
                vals_med.append(m.get(med_key))
            valid_mean = [v for v in vals_mean if v is not None]
            valid_med = [v for v in vals_med if v is not None]
            best_mean, second_mean = rank_values(valid_mean, mean_key)
            best_med, second_med = rank_values(valid_med, med_key)

            row = f"{display_name:<{metric_w}}"
            for i in range(len(run_labels)):
                row += "  " + format_cell(vals_mean[i], best_mean, second_mean, col_w) + format_cell(vals_med[i], best_med, second_med, col_w)
            print(row)

    print("\n\033[1;32mGreen/bold\033[0m = best,  \033[4;33mYellow/underline\033[0m = 2nd best")
    print("Lower is better: Acc, Comp (mean & med). Higher is better: NC1, NC2 (mean & med).")


def print_time_memory_tables(runs: dict, scene_filter=None):
    """Print Time (total, FPS, backbone-only time/FPS) and Memory (total MB, actual MB) per run."""
    run_labels = list(runs.keys())
    col_w = max(12, max(len(l) for l in run_labels) + 2)
    metric_w = 18

    # Collect per-scene time & mem stats for each run
    all_scenes = set()
    for info in runs.values():
        for scene, entry in info["data"].items():
            if not isinstance(entry, dict):
                continue
            if entry.get("reconstruction") is not None:
                all_scenes.add(scene)
    all_scenes = sorted(all_scenes)
    if scene_filter:
        all_scenes = [s for s in all_scenes if scene_filter.lower() in s.lower()]

    time_keys = ["total_time_ms", "backbone_time_ms"]
    mem_keys = ["total_usage_mb", "actual_mem_mb"]

    avg_time = {label: {} for label in run_labels}
    avg_mem = {label: {} for label in run_labels}
    for label in run_labels:
        data = runs[label]["data"]
        time_vals = {k: [] for k in time_keys}
        mem_vals = {k: [] for k in mem_keys}
        for scene in all_scenes:
            entry = data.get(scene)
            if not isinstance(entry, dict):
                continue
            t = compute_time_stats(_get_timing(entry))
            if t:
                for k in time_keys:
                    time_vals[k].append(t[k])
            m = compute_mem_stats(_get_memory(entry))
            if m:
                for k in mem_keys:
                    mem_vals[k].append(m[k])
        for k in time_keys:
            avg_time[label][k] = sum(time_vals[k]) / len(time_vals[k]) if time_vals[k] else None
        for k in mem_keys:
            avg_mem[label][k] = sum(mem_vals[k]) / len(mem_vals[k]) if mem_vals[k] else None

    # Time table: lower total_time / backbone_time is better; higher fps is better
    header = f"{'Metric':<{metric_w}}" + "".join(f"{l:>{col_w}}" for l in run_labels)
    sep_len = len(header)
    print("\n" + "=" * sep_len)
    print("  TIME (ms): total, backbone-only (aggregator_infer + kv_position + kv_retrieval + kv_prune_merge)")
    print("=" * sep_len)
    print(header)
    print("-" * sep_len)
    for row_name, key in [
        ("Total time (ms)", "total_time_ms"),
        ("Backbone time (ms)", "backbone_time_ms"),
    ]:
        vals = [avg_time[l].get(key) for l in run_labels]
        valid = [v for v in vals if v is not None]
        lower_better = key.endswith("_ms")  # time ms: lower better; fps: higher better
        if valid:
            sorted_v = sorted(valid, reverse=not lower_better)
            best, second = sorted_v[0], sorted_v[1] if len(sorted_v) > 1 else sorted_v[0]
        else:
            best = second = None
        line = f"{row_name:<{metric_w}}"
        for v in vals:
            if v is None:
                line += f"{'N/A':>{col_w}}"
            else:
                raw = f"{v:.2f}"
                if v == best:
                    raw = f"\033[1;32m{raw}\033[0m"
                elif v == second and v != best:
                    raw = f"\033[4;33m{raw}\033[0m"
                line += f"{raw:>{col_w}}"
        print(line)

    # Memory table: backend total_usage/total_alloc; actual = temporal_cache + spatial_cache (attention working set)
    print("\n" + "=" * sep_len)
    print("  MEMORY (MB): total_usage; actual = temporal + spatial (attention working set)")
    print("=" * sep_len)
    print(header)
    print("-" * sep_len)
    for row_name, key in [
        ("Total usage (MB)", "total_usage_mb"),
        ("Actual/working set (MB)", "actual_mem_mb"),
    ]:
        vals = [avg_mem[l].get(key) for l in run_labels]
        valid = [v for v in vals if v is not None]
        best = min(valid) if valid else None
        second = sorted(valid)[1] if len(valid) > 1 else best
        line = f"{row_name:<{metric_w}}"
        for v in vals:
            if v is None:
                line += f"{'N/A':>{col_w}}"
            else:
                raw = f"{v:.1f}"
                if v == best:
                    raw = f"\033[1;32m{raw}\033[0m"
                elif v == second and v != best:
                    raw = f"\033[4;33m{raw}\033[0m"
                line += f"{raw:>{col_w}}"
        print(line)


def print_run_configs(runs: dict):
    print("\n=== MODEL CONFIGS ===")
    for label, info in runs.items():
        data = info["data"]
        first_scene = next(iter(data.values()), {})
        model = first_scene.get("model", {})
        print(f"\n[{label}]  ({Path(info['path']).name})")
        for k, v in model.items():
            print(f"  {k}: {v}")


def main():
    default_dir = Path(__file__).resolve().parent.parent.parent / "eval_recon" / "NRGBD" / "causalvggt" / "overall_metrics"
    parser = argparse.ArgumentParser(
        description="Compare overall_metrics JSON files (mean | med adjacent per run)."
    )
    parser.add_argument(
        "--dir",
        default=str(default_dir),
        help="Directory containing overall_metrics JSON files",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["Acc", "Comp", "NC1", "NC2"],
        choices=[dn for dn, _, _ in METRIC_GROUPS],
        help="Metric groups to display (Acc, Comp, NC1, NC2)",
    )
    parser.add_argument(
        "--scene",
        default=None,
        help="Filter scenes by substring (e.g. 'chess')",
    )
    parser.add_argument(
        "--configs",
        action="store_true",
        help="Also print model configs for each run",
    )
    args = parser.parse_args()

    runs = load_files(args.dir)
    if not runs:
        print(f"No JSON files found in {args.dir}")
        return

    groups = [g for g in METRIC_GROUPS if g[0] in args.metrics]

    print(f"\nLoaded {len(runs)} run(s) from: {args.dir}")
    for label, info in runs.items():
        print(f"  [{label}]  {Path(info['path']).name}")

    if args.configs:
        print_run_configs(runs)

    print_table(runs, groups, scene_filter=args.scene)
    print_time_memory_tables(runs, scene_filter=args.scene)


if __name__ == "__main__":
    main()
