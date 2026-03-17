"""
Compare reconstruction metrics across multiple overall_metrics JSON files.
Displays mean and median in adjacent columns (e.g. Acc_mean, Acc_med) per run.

Run labels are by default model_tag (e.g. stream3r_stac). Rows are sorted by model then tag.
Deduplication still uses tag + CONFIG_LABEL_KEYS; same config keeps the run with the latest clock.
--tag X only loads JSON files whose filename contains X.

Usage: python compare.py [--dir DIR] [--tag TAG] [--metrics ...] [--scene NAME] [--all] [--label-from config|filename]
"""
import argparse
import json
import re
from pathlib import Path

# Metric groups: (display_name, mean_key, median_key). Keys can be str or tuple of two keys for (k1+k2)/2.
METRIC_GROUPS = [
    ("Acc", "accuracy", "accuracy_median"),
    ("Comp", "completion", "completion_median"),
    ("NC", ("normal_consistency_1", "normal_consistency_2"), ("normal_consistency_1_median", "normal_consistency_2_median")),
]

ALL_KEYS = [
    "accuracy", "accuracy_median",
    "completion", "completion_median",
    "normal_consistency_1", "normal_consistency_2",
    "normal_consistency_1_median", "normal_consistency_2_median",
]

# Lower is better for Acc, Comp (mean & med); higher is better for NC (mean & med)
LOWER_BETTER = {"accuracy", "completion", "accuracy_median", "completion_median", "Acc", "Comp"}

# Keys used to distinguish runs: tag (from filename) + model config from JSON
CONFIG_LABEL_KEYS = (
    "base_model",
    "mode",
    "window_size",
    "hh_size",
    "retrieval_size",
    "chunk_size",
    "voxel_backend",
)
# Tag is extracted from filename and prepended to config identity (tag + CONFIG_LABEL_KEYS)

# Short abbreviations for mode in labels
MODE_ABBREV = {
    "full": "full",
    "causal": "causal",
    "window": "win",
    "window_kv": "wkv",
    "window_chunk": "wc",
    "window_merge": "wm",
    "window_chunk_merge": "wcm",
}


# Tag = part between "overall_metrics_" and trailing "_YYYYMMDD_HHMM" (e.g. overall_metrics_stac_cuda_samp_25_20260314_2216 -> stac_cuda_samp_25)
_TIMESTAMP_SUFFIX_RE = re.compile(r"_\d{8}_\d{4}$")


def short_name(filename: str) -> str:
    """Strip path, overall_metrics prefix, time suffix and .json to get the tag as run label."""
    return tag_from_filename(filename) or Path(filename).stem


def tag_from_filename(filename: str) -> str:
    """Extract tag from filename: text between 'overall_metrics_' and '_YYYYMMDD_HHMM' (e.g. overall_metrics_stac_cuda_samp_25_20260314_2216 -> 'stac_cuda_samp_25')."""
    stem = Path(filename).stem
    if not stem.startswith("overall_metrics_"):
        return ""
    mid = stem[len("overall_metrics_"):]
    mid = _TIMESTAMP_SUFFIX_RE.sub("", mid)
    return mid.strip("_") if mid else ""


def config_signature(model: dict, filename_tag: str) -> tuple:
    """Unique tuple (filename_tag, base_model, mode, ...) for deduplication; same (tag + CONFIG_LABEL_KEYS) keeps latest clock."""
    head = (filename_tag,)
    if not model:
        return head
    return head + tuple(model.get(k) for k in CONFIG_LABEL_KEYS)


def config_to_label(model: dict, filename_tag: str = "") -> str:
    """Build run label for display: model_tag (e.g. stream3r_stac). Rows sort by model then tag."""
    if not model:
        return f"unknown_{filename_tag}" if filename_tag else "unknown"
    base = model.get("base_model")
    if base is None:
        return f"unknown_{filename_tag}" if filename_tag else "unknown"
    if filename_tag:
        return f"{base}_{filename_tag}"
    return str(base)


def _latest_clock(data: dict) -> str:
    """Return the latest clock string from any scene entry (format YYYY-MM-DD-HH-MM)."""
    clocks = []
    for entry in data.values():
        if isinstance(entry, dict) and "clock" in entry:
            clocks.append(entry["clock"])
    return max(clocks) if clocks else ""


def load_files(directory: str, label_from: str = "config", tag: str | None = None):
    directory = Path(directory)
    candidates = []
    for f in sorted(directory.glob("*.json")):
        if f.name == "compare_results.json":
            continue
        filename_tag = tag_from_filename(f.name)
        if tag is not None and tag not in f.name:
            continue
        with open(f) as fp:
            data = json.load(fp)
        first_entry = next(iter(data.values()), {})
        model = first_entry.get("model", {}) if isinstance(first_entry, dict) else {}
        sig = config_signature(model, filename_tag)
        clock = _latest_clock(data)
        candidates.append((f, data, model, filename_tag, sig, clock))

    # Same (tag + CONFIG_LABEL_KEYS) → keep only the file with latest clock
    by_config = {}
    for f, data, model, filename_tag, sig, clock in candidates:
        if sig not in by_config or clock > by_config[sig][4]:
            by_config[sig] = (f, data, model, filename_tag, clock)

    runs = {}
    seen_labels = {}
    for (f, data, model, filename_tag, _clock) in by_config.values():
        if label_from == "config":
            label = config_to_label(model, filename_tag)
            if label in seen_labels:
                seen_labels[label] += 1
                label = f"{label}_{seen_labels[label]}"
            else:
                seen_labels[label] = 0
        else:
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
    """Get memory_details dict from a scene entry (supports Merger/Memory(MB) or merger/memory_details)."""
    vc = entry.get("Merger") or entry.get("merger") or {}
    return vc.get("Memory(MB)") or vc.get("memory_details")


def _get_num_frames(entry: dict) -> float:
    """Get scene frame count for weighting; fallback to 1.0 if missing/invalid."""
    n = entry.get("num_frames")
    if isinstance(n, (int, float)) and n > 0:
        return float(n)
    return 1.0


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
        mem.get("temporal_cache_usage", 0) + mem.get("spatial_cache_usage", 0)
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


def _metric_value(m: dict, key) -> float | None:
    """Get scalar for one metric: m[key] if key is str, else (m[k1]+m[k2])/2 if key is (k1,k2)."""
    if isinstance(key, str):
        return m.get(key)
    k1, k2 = key
    v1, v2 = m.get(k1), m.get(k2)
    if v1 is not None and v2 is not None:
        return (v1 + v2) / 2.0
    return None


def rank_values(vals: list, metric_name: str):
    """Return (best, second_best) scalar values."""
    if not vals:
        return None, None
    lower_better = metric_name in LOWER_BETTER
    sorted_vals = sorted(vals, reverse=not lower_better)
    best = sorted_vals[0]
    second = sorted_vals[1] if len(sorted_vals) > 1 else best
    return best, second


def format_cell(v, best_val, second_val, width=12):
    """Return cell string with visible width=width for alignment. Bold green best, underline yellow second."""
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
    # Pad so visible width equals width (pad after content so columns align)
    pad = width - len(raw)
    return colored + (" " * pad) if pad > 0 else colored


def print_table(runs: dict, metric_groups: list, scene_filter=None, show_all_scenes=False):
    run_labels = sorted(runs.keys())  # order: model then tag (e.g. stream3r_stac, stream3r_win8, streamvggt_stac)

    # Per-run averages
    avg_per_run = {}
    all_scenes_union = set()
    for label, info in runs.items():
        sm = extract_metrics(info["data"])
        avg_per_run[label] = compute_averages(sm)
        all_scenes_union.update(sm.keys())
    all_scenes_sorted = sorted(all_scenes_union)
    if scene_filter:
        all_scenes_sorted = [s for s in all_scenes_sorted if scene_filter.lower() in s.lower()]
    n_scenes = len(all_scenes_sorted)
    scenes_line = ", ".join(all_scenes_sorted) if all_scenes_sorted else "(none)"
    if n_scenes > 10:
        scenes_line = ", ".join(all_scenes_sorted[:8]) + f", ... ({n_scenes} scenes)"
    else:
        scenes_line = ", ".join(all_scenes_sorted) + f" ({n_scenes} scene{'s' if n_scenes != 1 else ''})"

    # Column widths: first column = run label; then Acc_mean, Acc_med, Comp_mean, ...
    num_col_w = 12
    col_w = num_col_w
    run_col_w = max(6, max(len(str(l)) for l in run_labels))

    # Header: Run | Acc_mean | Acc_med | Comp_mean | Comp_med | NC_mean | NC_med
    def build_header():
        parts = [f"{'Run':<{run_col_w}}"]
        for display_name, _mean_key, _med_key in metric_groups:
            parts.append(f"{display_name}_mean".ljust(col_w))
            parts.append(f"{display_name}_med".ljust(col_w))
        return "  ".join(parts)

    header = build_header()
    sep_len = len(header)

    print("\n" + "=" * sep_len)
    print("  AVERAGE METRICS ACROSS ALL SCENES (rows = runs, columns = Acc_mean, Acc_med, ...)")
    print("  Averaged over: " + scenes_line)
    print("=" * sep_len)
    print(header)
    print("-" * sep_len)

    # Per-metric best/second for coloring (over runs)
    best_second = {}
    for display_name, mean_key, med_key in metric_groups:
        vals_mean = [_metric_value(avg_per_run[l], mean_key) for l in run_labels]
        vals_med = [_metric_value(avg_per_run[l], med_key) for l in run_labels]
        valid_mean = [v for v in vals_mean if v is not None]
        valid_med = [v for v in vals_med if v is not None]
        best_second[(display_name, "mean")] = rank_values(valid_mean, display_name)
        best_second[(display_name, "med")] = rank_values(valid_med, display_name)

    # One row per run
    for label in run_labels:
        row = f"{label:<{run_col_w}}"
        for display_name, mean_key, med_key in metric_groups:
            vm = _metric_value(avg_per_run[label], mean_key)
            vd = _metric_value(avg_per_run[label], med_key)
            best_mean, second_mean = best_second[(display_name, "mean")]
            best_med, second_med = best_second[(display_name, "med")]
            row += "  " + format_cell(vm, best_mean, second_mean, col_w) + "  " + format_cell(vd, best_med, second_med, col_w)
        print(row)

    # Per-scene breakdown: only when --all or --scene is set
    if not show_all_scenes and not scene_filter:
        print("\n\033[1;32mGreen/bold\033[0m = best,  \033[4;33mYellow/underline\033[0m = 2nd best")
        print("Lower is better: Acc, Comp. Higher is better: NC=(NC1+NC2)/2.")
        return

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

        # Best/second per metric for this scene
        best_second_scene = {}
        for display_name, mean_key, med_key in metric_groups:
            vals_mean = [_metric_value(extract_metrics(runs[l]["data"]).get(scene, {}), mean_key) for l in run_labels]
            vals_med = [_metric_value(extract_metrics(runs[l]["data"]).get(scene, {}), med_key) for l in run_labels]
            valid_mean = [v for v in vals_mean if v is not None]
            valid_med = [v for v in vals_med if v is not None]
            best_second_scene[(display_name, "mean")] = rank_values(valid_mean, display_name)
            best_second_scene[(display_name, "med")] = rank_values(valid_med, display_name)

        for label in run_labels:
            sm = extract_metrics(runs[label]["data"])
            m = sm.get(scene, {})
            row = f"{label:<{run_col_w}}"
            for display_name, mean_key, med_key in metric_groups:
                vm = _metric_value(m, mean_key)
                vd = _metric_value(m, med_key)
                best_mean, second_mean = best_second_scene[(display_name, "mean")]
                best_med, second_med = best_second_scene[(display_name, "med")]
                row += "  " + format_cell(vm, best_mean, second_mean, col_w) + "  " + format_cell(vd, best_med, second_med, col_w)
            print(row)

    print("\n\033[1;32mGreen/bold\033[0m = best,  \033[4;33mYellow/underline\033[0m = 2nd best")
    print("Lower is better: Acc, Comp (mean & med). Higher is better: NC=(NC1+NC2)/2 (mean & med).")


def _cell_right(s: str, width: int, best_val=None, second_val=None, value=None) -> str:
    """Format cell with visible width=width (pad left). Optionally color if value is best/second."""
    pad = width - len(s)
    pad_str = " " * pad if pad > 0 else ""
    if value is not None and best_val is not None and value == best_val:
        return pad_str + f"\033[1;32m{s}\033[0m"
    if value is not None and second_val is not None and value == second_val and value != best_val:
        return pad_str + f"\033[4;33m{s}\033[0m"
    return pad_str + s


def print_time_memory_tables(runs: dict, scene_filter=None):
    """Print Time/Memory (frame-weighted by num_frames) per run."""
    run_labels = sorted(runs.keys())  # order: model then tag

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
    frame_totals = {label: 0.0 for label in run_labels}
    for label in run_labels:
        data = runs[label]["data"]
        time_weighted_sums = {k: 0.0 for k in time_keys}
        time_weight_sums = {k: 0.0 for k in time_keys}
        mem_weighted_sums = {k: 0.0 for k in mem_keys}
        mem_weight_sums = {k: 0.0 for k in mem_keys}
        for scene in all_scenes:
            entry = data.get(scene)
            if not isinstance(entry, dict):
                continue
            scene_weight = _get_num_frames(entry)
            frame_totals[label] += scene_weight
            t = compute_time_stats(_get_timing(entry))
            if t:
                for k in time_keys:
                    if t[k] is not None:
                        time_weighted_sums[k] += t[k] * scene_weight
                        time_weight_sums[k] += scene_weight
            m = compute_mem_stats(_get_memory(entry))
            if m:
                for k in mem_keys:
                    if m[k] is not None:
                        mem_weighted_sums[k] += m[k] * scene_weight
                        mem_weight_sums[k] += scene_weight
        for k in time_keys:
            avg_time[label][k] = (
                time_weighted_sums[k] / time_weight_sums[k]
                if time_weight_sums[k] > 0
                else None
            )
        for k in mem_keys:
            avg_mem[label][k] = (
                mem_weighted_sums[k] / mem_weight_sums[k]
                if mem_weight_sums[k] > 0
                else None
            )

    # Column layout: Run | metric1 | metric2 | ... (same as main table)
    run_col_w = max(6, max(len(l) for l in run_labels))
    num_col_w = 12

    # Frame totals table (for weighted averages)
    frame_col_name = "Frames(total)"
    frame_col_w = max(num_col_w, len(frame_col_name))
    frame_header = "  ".join([f"{'Run':<{run_col_w}}", frame_col_name.ljust(frame_col_w)])
    frame_sep_len = len(frame_header)
    print("\n" + "=" * frame_sep_len)
    print("  FRAME TOTALS used for weighted averaging")
    print("=" * frame_sep_len)
    print(frame_header)
    print("-" * frame_sep_len)
    for label in run_labels:
        total_frames = int(round(frame_totals[label]))
        print(f"{label:<{run_col_w}}  {str(total_frames).rjust(frame_col_w)}")

    # Time table: header Run | Total time (ms) | Backbone time (ms); one row per run (lower is better)
    time_rows = [
        ("Total time (ms)", "total_time_ms", ".2f"),
        ("Backbone time (ms)", "backbone_time_ms", ".2f"),
    ]
    time_col_w = max(num_col_w, max(len(name) for name, _, _ in time_rows))
    time_header_parts = [f"{'Run':<{run_col_w}}"] + [f"{name}".ljust(time_col_w) for name, _key, _fmt in time_rows]
    time_header = "  ".join(time_header_parts)
    sep_len = len(time_header)
    print("\n" + "=" * sep_len)
    print("  TIME (ms): frame-weighted by num_frames; total, backbone-only (aggregator_infer + kv_position + kv_retrieval + kv_prune_merge)")
    print("=" * sep_len)
    print(time_header)
    print("-" * sep_len)
    time_best_second = {}
    for row_name, key, fmt in time_rows:
        vals = [avg_time[l].get(key) for l in run_labels]
        valid = [v for v in vals if v is not None]
        if valid:
            sorted_v = sorted(valid)
            time_best_second[key] = (sorted_v[0], sorted_v[1] if len(sorted_v) > 1 else sorted_v[0])
        else:
            time_best_second[key] = (None, None)
    for label in run_labels:
        line = f"{label:<{run_col_w}}"
        for _row_name, key, fmt in time_rows:
            v = avg_time[label].get(key)
            raw = "N/A" if v is None else f"{v:{fmt}}"
            best, second = time_best_second[key]
            line += "  " + _cell_right(raw, time_col_w, best_val=best, second_val=second, value=v)
        print(line)

    # Memory table: header Run | Total usage (MB) | Actual/working set (MB); one row per run (lower is better)
    mem_rows = [
        ("Total usage (MB)", "total_usage_mb", ".1f"),
        ("Actual/working set (MB)", "actual_mem_mb", ".1f"),
    ]
    mem_col_w = max(num_col_w, max(len(name) for name, _, _ in mem_rows))
    mem_header_parts = [f"{'Run':<{run_col_w}}"] + [f"{name}".ljust(mem_col_w) for name, _key, _fmt in mem_rows]
    mem_header = "  ".join(mem_header_parts)
    sep_len = max(sep_len, len(mem_header))
    print("\n" + "=" * sep_len)
    print("  MEMORY (MB): frame-weighted by num_frames; total_usage; actual = temporal + spatial (attention working set)")
    print("=" * sep_len)
    print(mem_header)
    print("-" * sep_len)
    mem_best_second = {}
    for _row_name, key, _fmt in mem_rows:
        vals = [avg_mem[l].get(key) for l in run_labels]
        valid = [v for v in vals if v is not None]
        if valid:
            sorted_v = sorted(valid)
            mem_best_second[key] = (sorted_v[0], sorted_v[1] if len(sorted_v) > 1 else sorted_v[0])
        else:
            mem_best_second[key] = (None, None)
    for label in run_labels:
        line = f"{label:<{run_col_w}}"
        for _row_name, key, fmt in mem_rows:
            v = avg_mem[label].get(key)
            raw = "N/A" if v is None else f"{v:{fmt}}"
            best, second = mem_best_second[key]
            line += "  " + _cell_right(raw, mem_col_w, best_val=best, second_val=second, value=v)
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
        default=["Acc", "Comp", "NC"],
        choices=[dn for dn, _, _ in METRIC_GROUPS],
        help="Metric groups to display (Acc, Comp, NC=(NC1+NC2)/2)",
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
    parser.add_argument(
        "--all",
        action="store_true",
        help="Print per-scene breakdown for all scenes (default: only average across scenes)",
    )
    parser.add_argument(
        "--label-from",
        choices=["config", "filename"],
        default="config",
        help="Run label: from JSON model config (base_model, mode, window_size, hh_size, retrieval_size, chunk_size, voxel_backend) or from filename (default: config)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Only consider JSON files whose filename contains this tag; among tag-matched files, same config (CONFIG_LABEL_KEYS) keeps the run with latest clock",
    )
    args = parser.parse_args()

    runs = load_files(args.dir, label_from=args.label_from, tag=args.tag)
    if not runs:
        msg = f"No JSON files found in {args.dir}"
        if args.tag:
            msg += f" (with tag '{args.tag}' in filename)"
        print(msg)
        return

    groups = [g for g in METRIC_GROUPS if g[0] in args.metrics]

    print(f"\nLoaded {len(runs)} run(s) from: {args.dir}" + (f" (filename contains '{args.tag}')" if args.tag else ""))
    for label in sorted(runs.keys()):
        print(f"  [{label}]  {Path(runs[label]['path']).name}")

    if args.configs:
        print_run_configs(runs)

    print_table(runs, groups, scene_filter=args.scene, show_all_scenes=args.all)
    print_time_memory_tables(runs, scene_filter=args.scene)


if __name__ == "__main__":
    main()
