#!/usr/bin/env python3
"""
Aggregate results across runs: compute mean ± std per (Train_Dataset, Test_Dataset)
and save to experiments/aggregates/<config_prefix>.csv.

Usage (from 3GPP_Article directory):
  python scripts/aggregate_runs.py [experiments_dir]
  Default experiments_dir: ./experiments
"""

import re
import sys
from pathlib import Path

import pandas as pd


def find_run_dirs(experiments_dir: Path) -> dict[str, list[Path]]:
    """Group experiment dirs by config prefix (strip _runN). Returns {prefix: [run1_path, run2_path, ...]}."""
    pattern = re.compile(r"^(.+)_run(\d+)$")
    groups: dict[str, list[tuple[int, Path]]] = {}

    for d in experiments_dir.iterdir():
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue
        prefix, run_num = m.group(1), int(m.group(2))
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append((run_num, d))

    # Sort runs by number and keep only paths
    out = {}
    for prefix, run_list in groups.items():
        run_list.sort(key=lambda x: x[0])
        out[prefix] = [p for _, p in run_list]
    return out


def aggregate_one_group(run_dirs: list[Path], out_path: Path, metric_cols: list[str]) -> bool:
    """Load all_results.csv from each run dir, compute mean and std, save to out_path."""
    dfs = []
    for d in run_dirs:
        csv_path = d / "all_results.csv"
        if not csv_path.exists():
            print(f"  Skip (missing all_results.csv): {d.name}")
            continue
        df = pd.read_csv(csv_path)
        dfs.append(df)

    if not dfs:
        return False

    combined = pd.concat(dfs, ignore_index=True)

    # Group by (Train_Dataset, Test_Dataset)
    group_cols = ["Train_Dataset", "Test_Dataset"]
    metric_cols_present = [c for c in metric_cols if c in combined.columns]
    if not metric_cols_present:
        return False
    agg = combined.groupby(group_cols, as_index=False)[metric_cols_present].agg(["mean", "std"])

    # Flatten column names: (SGCS, mean) -> SGCS_mean, (SGCS, std) -> SGCS_std
    new_cols = []
    for c in agg.columns:
        if isinstance(c, tuple) and len(c) == 2 and c[1]:
            new_cols.append(f"{c[0]}_{c[1]}")
        else:
            new_cols.append(c if not isinstance(c, tuple) else c[0])
    agg.columns = new_cols

    # Reorder: Train_Dataset, Test_Dataset, then for each metric mean then std
    cols = list(agg.columns)
    metric_order = []
    for m in metric_cols_present:
        if f"{m}_mean" in cols:
            metric_order.append(f"{m}_mean")
        if f"{m}_std" in cols:
            metric_order.append(f"{m}_std")
    final_cols = group_cols + metric_order
    agg = agg[[c for c in final_cols if c in agg.columns]]

    agg.insert(2, "n_runs", len(dfs))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False)
    return True


def main():
    experiments_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments")
    if not experiments_dir.is_dir():
        print(f"Not a directory: {experiments_dir}")
        sys.exit(1)

    aggregates_dir = experiments_dir / "aggregates"
    aggregates_dir.mkdir(parents=True, exist_ok=True)

    metric_cols = ["SGCS", "Loss", "Recon_Loss"]
    # Contrast_Loss may be missing for baseline
    run_dirs_by_prefix = find_run_dirs(experiments_dir)

    print(f"Found {len(run_dirs_by_prefix)} config(s) with run subdirs under {experiments_dir}")
    for prefix, run_dirs in sorted(run_dirs_by_prefix.items()):
        n = len(run_dirs)
        if n < 2:
            print(f"  {prefix}: {n} run(s), skip (need ≥2 for aggregate)")
            continue
        out_path = aggregates_dir / f"{prefix}.csv"
        # Include Contrast_Loss if present in first run's CSV
        first_csv = run_dirs[0] / "all_results.csv"
        if first_csv.exists():
            sample = pd.read_csv(first_csv, nrows=1)
            if "Contrast_Loss" in sample.columns:
                metrics = metric_cols + ["Contrast_Loss"]
            else:
                metrics = metric_cols
        else:
            metrics = metric_cols
        if aggregate_one_group(run_dirs, out_path, metrics):
            print(f"  {prefix}: {n} runs -> {out_path}")
        else:
            print(f"  {prefix}: no data written")

    print(f"\nAggregates saved under {aggregates_dir}")


if __name__ == "__main__":
    main()
