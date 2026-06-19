#!/usr/bin/env python3
"""Rank tuning results and pick the best config per pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TUNING_DIR = Path(__file__).resolve().parent


def rank_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "rank" in df.columns:
        return df.sort_values("rank")
    if "composite_score" in df.columns:
        return df.sort_values("composite_score", ascending=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["no_target", "all_qi", "both"], default="both")
    args = parser.parse_args()

    sources = []
    if args.pipeline in ("no_target", "both"):
        sources.append(("No_target", TUNING_DIR / "results" / "No_target_experiment_ranking.csv", ROOT / "No_target" / "results" / "experiment_ranking.csv"))
    if args.pipeline in ("all_qi", "both"):
        sources.append(("all_qi_pipeline", TUNING_DIR / "results" / "all_qi_pipeline_experiment_ranking.csv", ROOT / "all_qi_pipeline" / "results" / "experiment_ranking.csv"))

    summaries = []
    for label, tuned, fallback in sources:
        path = tuned if tuned.exists() else fallback
        if not path.exists():
            print(f"[{label}] No results at {path}")
            continue
        df = rank_file(path)
        top = df.iloc[0]
        summaries.append({
            "pipeline": label,
            "source": str(path),
            "rank": int(top.get("rank", 1)),
            "folder": top.get("folder"),
            "composite_score": top.get("composite_score"),
            "overall_pass": top.get("overall_pass"),
            "tvd_pass_rate": top.get("tvd_pass_rate"),
            "ks_pass_rate": top.get("ks_pass_rate"),
            "mean_corr_drift": top.get("mean_corr_drift"),
            "mean_pair_drift": top.get("mean_pair_drift"),
            "auc_retention_ratio": top.get("auc_retention_ratio"),
            "n_suppressed": top.get("n_suppressed"),
            "runtime_sec": top.get("runtime_sec"),
        })
        print(f"\n=== {label} top config ===")
        print(f"  folder: {top.get('folder')}")
        print(f"  composite: {top.get('composite_score')} | pass: {top.get('overall_pass')}")
        print(f"  TVD pass: {top.get('tvd_pass_rate')} | KS pass: {top.get('ks_pass_rate')}")
        if pd.notna(top.get("auc_retention_ratio")):
            print(f"  AUC retention: {top.get('auc_retention_ratio')}")
        if pd.notna(top.get("mean_pair_drift")):
            print(f"  pair drift: {top.get('mean_pair_drift')}")
        elif pd.notna(top.get("mean_corr_drift")):
            print(f"  corr drift: {top.get('mean_corr_drift')}")

    if summaries:
        out = TUNING_DIR / "results" / "tuning_summary.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summaries).to_csv(out, index=False)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
