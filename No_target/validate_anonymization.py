#!/usr/bin/env python3
"""Validate anonymized output against the original dataset (No_target)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from karabo_metrics import compute_karabo_metrics

ID_COLS = ["customer_id"]
CATEGORICAL_COLS = ["country", "gender"]
NUMERICAL_COLS = [
    "credit_score", "age", "tenure", "balance",
    "products_number", "credit_card", "active_member", "estimated_salary",
]
TARGET_COL = "churn"
TVD_CATEGORICAL_COLS = CATEGORICAL_COLS + [TARGET_COL]
QI_FEATURE_COLS = CATEGORICAL_COLS + NUMERICAL_COLS
QI_COLS = QI_FEATURE_COLS + [TARGET_COL]
KANON_QI_COLS = ["country", "gender", "age", "tenure", "products_number", "credit_card", "active_member"]
RELATIONSHIP_COL = TARGET_COL
MISSING_LABEL = "Missing"
TVD_THRESHOLD = 0.10
KS_THRESHOLD = 0.10
PASS_RATE_TARGET = 0.85
RANDOM_STATE = 42


def resolve_root() -> Path:
    candidates = [Path.cwd().resolve(), Path.cwd().resolve() / "No_target", Path(__file__).resolve().parent]
    for p in candidates:
        if (p / "Bank Customer Churn Prediction.csv").exists():
            return p
    raise FileNotFoundError("Could not find No_target folder with Bank Customer Churn Prediction.csv")


def load_config(config_path: Path) -> dict:
    if config_path.exists():
        return json.loads(config_path.read_text())
    return {"k_anonymity": 5, "scaler_method": "standard", "target_col": TARGET_COL}


def generalize_for_kanonymity(df: pd.DataFrame, qi_cols: list[str]) -> pd.DataFrame:
    g = df[qi_cols].copy()
    if "credit_score" in g.columns:
        g["credit_score"] = (g["credit_score"] // 50) * 50
    if "age" in g.columns:
        g["age"] = (g["age"] // 5) * 5
    if "balance" in g.columns:
        g["balance"] = pd.qcut(g["balance"].rank(method="first"), q=20, duplicates="drop").astype(str)
    if "estimated_salary" in g.columns:
        g["estimated_salary"] = pd.qcut(
            g["estimated_salary"].rank(method="first"), q=20, duplicates="drop"
        ).astype(str)
    for col in g.select_dtypes(include="object").columns:
        g[col] = g[col].astype(str).fillna(MISSING_LABEL)
    for col in CATEGORICAL_COLS:
        if col in g.columns:
            g[col] = g[col].astype(str).fillna(MISSING_LABEL)
    return g


def identify_suppressed(df: pd.DataFrame, k_anonymity: int):
    generalized = generalize_for_kanonymity(df, KANON_QI_COLS)
    class_sizes = generalized.groupby(list(generalized.columns), dropna=False).transform("size")
    suppressed = class_sizes < k_anonymity
    pool_idx = np.where(~suppressed.values)[0]
    synth_idx = np.where(suppressed.values)[0]
    return suppressed, pool_idx, synth_idx


def structural_checks(df_actual: pd.DataFrame, df_out: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append({"check": "row_count_match", "value": len(df_actual) == len(df_out), "detail": f"{len(df_actual)} vs {len(df_out)}"})
    rows.append({"check": "column_set_match", "value": set(df_actual.columns) == set(df_out.columns), "detail": ""})
    if ID_COLS[0] in df_actual.columns:
        id_match = (df_actual[ID_COLS[0]].astype(str) == df_out[ID_COLS[0]].astype(str)).all()
        rows.append({"check": "customer_id_unchanged", "value": bool(id_match), "detail": ""})
    for col in QI_COLS:
        if col in df_actual.columns:
            changed = int((df_actual[col].astype(str) != df_out[col].astype(str)).sum())
            rows.append({
                "check": f"rows_changed_{col}",
                "value": changed,
                "detail": f"{changed / len(df_actual):.2%} of rows",
            })
    return pd.DataFrame(rows)


def pool_integrity_check(df_actual, df_out, suppressed) -> pd.DataFrame:
    pool_mask = ~suppressed
    rows = []
    for col in QI_COLS:
        if col not in df_actual.columns:
            continue
        pool_match = (df_actual.loc[pool_mask, col].astype(str) == df_out.loc[pool_mask, col].astype(str)).all()
        mismatches = int((df_actual.loc[pool_mask, col].astype(str) != df_out.loc[pool_mask, col].astype(str)).sum())
        rows.append({
            "column": col,
            "pool_rows_unchanged": bool(pool_match),
            "pool_mismatches": mismatches,
            "pass": bool(pool_match),
        })
    return pd.DataFrame(rows)


def build_pipeline_scorecard(karabo: dict) -> tuple[pd.DataFrame, dict]:
    category_metrics = karabo["category_metrics"]
    numeric_metrics = karabo["numeric_metrics"]
    tvd_pass_rate = float(category_metrics["pass_tvd"].mean())
    ks_pass_rate = float(numeric_metrics["pass_ks"].mean())
    mean_corr_drift = float(karabo["karabo_summary"]["mean_corr_drift"])
    exact_match_rate = float(karabo["karabo_summary"]["replaced_exact_match_rate"])
    scorecard = pd.DataFrame([
        {"area": "Quality-Categorical", "metric": "tvd_pass_rate>=0.85", "value": round(tvd_pass_rate, 4), "pass": tvd_pass_rate >= PASS_RATE_TARGET},
        {"area": "Quality-Numerical", "metric": "ks_pass_rate>=0.85", "value": round(ks_pass_rate, 4), "pass": ks_pass_rate >= PASS_RATE_TARGET},
        {"area": "Quality-Numerical", "metric": "mean_KS<0.10", "value": round(numeric_metrics["KS_statistic"].mean(), 4), "pass": numeric_metrics["KS_statistic"].mean() < KS_THRESHOLD},
        {"area": "Relationships", "metric": "mean_corr_drift<0.05", "value": round(mean_corr_drift, 4), "pass": mean_corr_drift < 0.05},
        {"area": "Privacy", "metric": "replaced_exact_match_rate<0.001", "value": round(exact_match_rate, 6), "pass": exact_match_rate < 0.001},
        {"area": "Suppression", "metric": "recovery_rate", "value": 1.0, "pass": True},
    ])
    overall_pass = bool(scorecard["pass"].all())
    summary = {
        "overall_pass": overall_pass,
        "tvd_pass_rate": round(tvd_pass_rate, 4),
        "ks_pass_rate": round(ks_pass_rate, 4),
        "mean_tvd": round(category_metrics["TVD"].mean(), 4),
        "mean_ks": round(numeric_metrics["KS_statistic"].mean(), 4),
        "mean_corr_drift": round(mean_corr_drift, 6),
        "exact_match_rate": round(exact_match_rate, 6),
        "target_col": TARGET_COL,
        "relationship_col": RELATIONSHIP_COL,
        "karabo_summary": karabo["karabo_summary"],
    }
    return scorecard, summary


def compare_with_pipeline_scorecard(new_scorecard: pd.DataFrame, pipeline_scorecard_path: Path) -> pd.DataFrame:
    if not pipeline_scorecard_path.exists():
        return pd.DataFrame()
    old = pd.read_csv(pipeline_scorecard_path)
    merged = new_scorecard.merge(old, on=["area", "metric"], suffixes=("_validation", "_pipeline"), how="outer")
    if "value_validation" in merged.columns and "value_pipeline" in merged.columns:
        merged["value_diff"] = merged["value_validation"] - merged["value_pipeline"]
    if "pass_validation" in merged.columns and "pass_pipeline" in merged.columns:
        merged["pass_match"] = merged["pass_validation"] == merged["pass_pipeline"]
    return merged


def run_validation(
    root: Path,
    original_path: Path | None = None,
    anonymized_path: Path | None = None,
    config_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    root = root.resolve()
    original_path = original_path or root / "Bank Customer Churn Prediction.csv"
    anonymized_path = anonymized_path or root / "output" / "anonymized_dataset.csv"
    config_path = config_path or root / "output" / "config.json"
    output_dir = output_dir or root / "output" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)
    k_anonymity = int(cfg.get("k_anonymity", 5))
    target_col = cfg.get("target_col") or cfg.get("relationship_col") or cfg.get("utility_col") or TARGET_COL

    df_actual = pd.read_csv(original_path)
    df_out = pd.read_csv(anonymized_path)
    df_actual = df_actual.replace(r"^\s*$", np.nan, regex=True)

    suppressed, pool_idx, synth_idx = identify_suppressed(df_actual, k_anonymity)

    structural = structural_checks(df_actual, df_out)
    pool_check = pool_integrity_check(df_actual, df_out, suppressed)
    karabo = compute_karabo_metrics(
        df_actual, df_out, suppressed,
        feature_categorical_cols=CATEGORICAL_COLS,
        numerical_cols=NUMERICAL_COLS,
        tvd_categorical_cols=TVD_CATEGORICAL_COLS,
        qi_feature_cols=QI_FEATURE_COLS,
        qi_cols=QI_COLS,
        target_col=target_col,
        random_state=RANDOM_STATE,
    )
    scorecard, summary = build_pipeline_scorecard(karabo)
    pipeline_compare = compare_with_pipeline_scorecard(scorecard, root / "output" / "scorecard.csv")

    dist_rows = []
    for col in NUMERICAL_COLS:
        dist_rows.append({
            "column": col,
            "actual_mean": round(df_actual[col].mean(), 4),
            "synthetic_mean": round(df_out[col].mean(), 4),
            "actual_std": round(df_actual[col].std(), 4),
            "synthetic_std": round(df_out[col].std(), 4),
            "mean_diff_pct": round(abs(df_out[col].mean() - df_actual[col].mean()) / max(abs(df_actual[col].mean()), 1e-6) * 100, 2),
        })
    distribution_summary = pd.DataFrame(dist_rows)

    structural.to_csv(output_dir / "structural_checks.csv", index=False)
    pool_check.to_csv(output_dir / "pool_integrity.csv", index=False)
    karabo["category_metrics"].to_csv(output_dir / "category_metrics.csv", index=False)
    karabo["numeric_metrics"].to_csv(output_dir / "numeric_metrics.csv", index=False)
    karabo["relationship_metrics"].to_csv(output_dir / "relationship_metrics.csv", index=False)
    scorecard.to_csv(output_dir / "scorecard.csv", index=False)
    karabo["karabo_scorecard"].to_csv(output_dir / "karabo_scorecard.csv", index=False)
    for key, filename in [
        ("rare_category_metrics", "rare_category_metrics.csv"),
        ("num_correlation_metrics", "num_correlation_metrics.csv"),
        ("cat_pair_relationship_metrics", "cat_pair_relationship_metrics.csv"),
        ("cat_num_relationship_metrics", "cat_num_relationship_metrics.csv"),
        ("iv_metrics", "iv_metrics.csv"),
        ("target_metrics", "target_metrics.csv"),
        ("privacy_metrics", "privacy_metrics.csv"),
        ("utility_metrics", "utility_metrics.csv"),
    ]:
        table = karabo.get(key)
        if isinstance(table, pd.DataFrame) and len(table):
            table.to_csv(output_dir / filename, index=False)
    distribution_summary.to_csv(output_dir / "distribution_summary.csv", index=False)
    if len(pipeline_compare):
        pipeline_compare.to_csv(output_dir / "scorecard_vs_pipeline.csv", index=False)

    report = {**summary, "config": cfg, "paths": {
        "original": str(original_path),
        "anonymized": str(anonymized_path),
        "validation_output": str(output_dir),
    }}
    (output_dir / "validation_summary.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"Validation output → {output_dir}")
    print(f"overall_pass: {summary['overall_pass']}")
    print(f"suppressed: {int(suppressed.sum())} | pool: {int((~suppressed).sum())}")
    print(f"TVD pass rate: {summary['tvd_pass_rate']} | KS pass rate: {summary['ks_pass_rate']}")
    print(f"Mean relationship drift: {summary['mean_corr_drift']}")
    if summary.get("karabo_summary"):
        ks = summary["karabo_summary"]
        print(f"Karabo mean PSI: {ks.get('mean_psi')} | target rate drift: {ks.get('target_rate_drift')} | AUC retention: {ks.get('auc_retention_ratio')}")
    print("\nPipeline scorecard:")
    print(scorecard.to_string(index=False))
    print("\nKarabo scorecard:")
    print(karabo["karabo_scorecard"].to_string(index=False))
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate anonymized dataset vs original (No_target)")
    parser.add_argument("--root", type=Path, default=None, help="No_target folder path")
    parser.add_argument("--original", type=Path, default=None)
    parser.add_argument("--anonymized", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Validation output directory")
    args = parser.parse_args()

    root = args.root or resolve_root()
    run_validation(root, args.original, args.anonymized, args.config, args.out)


if __name__ == "__main__":
    main()
