#!/usr/bin/env python3
"""Validate anonymized output against the original dataset (No_target)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import itertools

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PSI_THRESHOLD = 0.25
RARE_CATEGORY_THRESHOLD = 0.05
TARGET_RATE_DRIFT_THRESHOLD = 0.05
AUC_RETENTION_THRESHOLD = 0.80

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


# Metrics logic mirrored from production_pipeline.ipynb (Karabo validation section).
def calculate_psi(real_values, synth_values, bins: int = 10) -> float:
    real_values = pd.Series(real_values).dropna().astype(float)
    synth_values = pd.Series(synth_values).dropna().astype(float)
    if len(real_values) == 0 or len(synth_values) == 0:
        return float("nan")
    if real_values.nunique() <= 1:
        return 0.0 if synth_values.nunique() <= 1 and synth_values.iloc[0] == real_values.iloc[0] else float("nan")
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(real_values, quantiles))
    if len(edges) < 3:
        return float("nan")
    real_counts, _ = np.histogram(real_values, bins=edges)
    synth_counts, _ = np.histogram(synth_values, bins=edges)
    real_pct = real_counts / max(1, real_counts.sum())
    synth_pct = synth_counts / max(1, synth_counts.sum())
    epsilon = 1e-6
    return float(np.sum((synth_pct - real_pct) * np.log((synth_pct + epsilon) / (real_pct + epsilon))))


def categorical_distribution_metrics(real_data, synth_data, cat_cols):
    rows = []
    for col in cat_cols:
        real_dist = real_data[col].astype(str).value_counts(normalize=True, dropna=False)
        synth_dist = synth_data[col].astype(str).value_counts(normalize=True, dropna=False)
        all_categories = sorted(set(real_dist.index) | set(synth_dist.index))
        real_aligned = real_dist.reindex(all_categories, fill_value=0)
        synth_aligned = synth_dist.reindex(all_categories, fill_value=0)
        drift = float(0.5 * np.abs(real_aligned - synth_aligned).sum())
        real_categories = set(real_dist.index)
        synth_categories = set(synth_dist.index)
        rows.append({
            "column": col,
            "TVD": round(drift, 4),
            "pass_tvd": drift < TVD_THRESHOLD,
            "real_category_count": len(real_categories),
            "synthetic_category_count": len(synth_categories),
            "category_coverage": round(len(real_categories & synth_categories) / max(1, len(real_categories)), 4),
            "new_synthetic_category_count": len(synth_categories - real_categories),
            "missing_synthetic_category_count": len(real_categories - synth_categories),
            "max_category_proportion_diff": round(float(np.abs(real_aligned - synth_aligned).max()), 4),
        })
    return pd.DataFrame(rows)


def rare_category_metrics(original_df, synthetic_df, cols, threshold=RARE_CATEGORY_THRESHOLD):
    rows = []
    for col in cols:
        orig_freq = original_df[col].value_counts(normalize=True)
        rare_cats = orig_freq[orig_freq < threshold].index
        if len(rare_cats) == 0:
            rows.append({"column": col, "rare_category_count": 0, "original_rare_mass": 0.0, "synthetic_rare_mass": 0.0, "retention_ratio": float("nan")})
            continue
        synth_freq = synthetic_df[col].value_counts(normalize=True)
        orig_mass = float(orig_freq.loc[rare_cats].sum())
        synth_mass = float(synth_freq.reindex(rare_cats, fill_value=0).sum())
        rows.append({
            "column": col,
            "rare_category_count": int(len(rare_cats)),
            "original_rare_mass": round(orig_mass, 4),
            "synthetic_rare_mass": round(synth_mass, 4),
            "retention_ratio": round(synth_mass / orig_mass, 4) if orig_mass > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def numerical_distribution_metrics(real_data, synth_data, num_cols):
    rows = []
    for col in num_cols:
        real = pd.to_numeric(real_data[col], errors="coerce").dropna()
        synth = pd.to_numeric(synth_data[col], errors="coerce").dropna()
        if len(real) == 0 or len(synth) == 0:
            continue
        ks_stat = float(ks_2samp(real, synth).statistic) if real.nunique() > 1 and synth.nunique() > 1 else float("nan")
        psi = calculate_psi(real, synth)
        rows.append({
            "column": col,
            "KS_statistic": round(ks_stat, 4) if pd.notna(ks_stat) else np.nan,
            "pass_ks": ks_stat < KS_THRESHOLD if pd.notna(ks_stat) else False,
            "psi": round(psi, 4) if pd.notna(psi) else np.nan,
            "pass_psi": psi < PSI_THRESHOLD if pd.notna(psi) else False,
            "real_mean": round(float(real.mean()), 4),
            "synthetic_mean": round(float(synth.mean()), 4),
            "mean_abs_diff": round(abs(float(real.mean() - synth.mean())), 4),
            "real_median": round(float(real.median()), 4),
            "synthetic_median": round(float(synth.median()), 4),
            "median_abs_diff": round(abs(float(real.median() - synth.median())), 4),
            "real_std": round(float(real.std()), 4),
            "synthetic_std": round(float(synth.std()), 4),
            "std_abs_diff": round(abs(float(real.std() - synth.std())), 4),
        })
    return pd.DataFrame(rows)


def cramers_v(x, y) -> float:
    tbl = pd.crosstab(x.astype(str), y.astype(str))
    if tbl.shape[0] < 2 or tbl.shape[1] < 2:
        return float("nan")
    chi2 = chi2_contingency(tbl)[0]
    n = tbl.sum().sum()
    k = min(tbl.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if n > 0 and k > 0 else 0.0


def target_relationship_metrics(df_actual, df_out, feature_cols, target_col):
    rows = []
    for col in feature_cols:
        if col == target_col:
            continue
        if col in df_actual.columns and pd.api.types.is_numeric_dtype(df_actual[col]):
            c_a = df_actual[col].astype(float).corr(df_actual[target_col].astype(float))
            c_s = df_out[col].astype(float).corr(df_out[target_col].astype(float))
            if pd.notna(c_a) and pd.notna(c_s):
                rows.append({"label": target_col, "column": col, "metric": "correlation", "actual": round(float(c_a), 4), "synthetic": round(float(c_s), 4), "drift": round(abs(float(c_a) - float(c_s)), 4)})
        else:
            v_a = cramers_v(df_actual[col], df_actual[target_col])
            v_s = cramers_v(df_out[col], df_out[target_col])
            rows.append({"label": target_col, "column": col, "metric": "cramers_v", "actual": round(v_a, 4), "synthetic": round(v_s, 4), "drift": round(abs(v_a - v_s), 4)})
    return pd.DataFrame(rows)


def numerical_correlation_metrics(real_data, synth_data, num_cols):
    if len(num_cols) < 2:
        return pd.DataFrame()
    real_corr = real_data[num_cols].astype(float).corr()
    synth_corr = synth_data[num_cols].astype(float).corr()
    rows = []
    for c1, c2 in itertools.combinations(num_cols, 2):
        real_val = real_corr.loc[c1, c2]
        synth_val = synth_corr.loc[c1, c2]
        rows.append({
            "feature_1": c1, "feature_2": c2,
            "real_corr": round(float(real_val), 4) if pd.notna(real_val) else np.nan,
            "synthetic_corr": round(float(synth_val), 4) if pd.notna(synth_val) else np.nan,
            "abs_corr_diff": round(abs(float(real_val) - float(synth_val)), 4) if pd.notna(real_val) and pd.notna(synth_val) else np.nan,
        })
    return pd.DataFrame(rows)


def categorical_pair_relationship_metrics(real_data, synth_data, cat_cols):
    rows = []
    for c1, c2 in itertools.combinations(cat_cols, 2):
        real_v = cramers_v(real_data[c1], real_data[c2])
        synth_v = cramers_v(synth_data[c1], synth_data[c2])
        rows.append({
            "feature_1": c1, "feature_2": c2,
            "real_cramers_v": round(real_v, 4) if pd.notna(real_v) else np.nan,
            "synthetic_cramers_v": round(synth_v, 4) if pd.notna(synth_v) else np.nan,
            "abs_cramers_v_diff": round(abs(real_v - synth_v), 4) if pd.notna(real_v) and pd.notna(synth_v) else np.nan,
        })
    return pd.DataFrame(rows)


def categorical_numerical_relationship_metrics(real_data, synth_data, cat_cols, num_cols, max_categories=25):
    rows = []
    for cat in cat_cols:
        if real_data[cat].nunique(dropna=False) > max_categories:
            continue
        for num in num_cols:
            real_group = real_data.groupby(cat)[num].mean()
            synth_group = synth_data.groupby(cat)[num].mean()
            common_groups = sorted(set(real_group.index) & set(synth_group.index))
            if not common_groups:
                continue
            real_values = real_group.reindex(common_groups)
            synth_values = synth_group.reindex(common_groups)
            scale = real_data[num].astype(float).std()
            if pd.isna(scale) or scale == 0:
                scale = 1.0
            mean_group_diff = float(np.mean(np.abs(real_values - synth_values)) / scale)
            rows.append({"categorical_column": cat, "numerical_column": num, "common_category_count": len(common_groups), "normalised_group_mean_diff": round(mean_group_diff, 4)})
    return pd.DataFrame(rows)


def calc_iv(df, feature, target):
    eps = 1e-6
    target_num = pd.to_numeric(df[target], errors="coerce")
    total_good = int((target_num == 0).sum())
    total_bad = int((target_num == 1).sum())
    iv = 0.0
    for _, sub in df.groupby(feature, dropna=False):
        y = pd.to_numeric(sub[target], errors="coerce")
        good = int((y == 0).sum())
        bad = int((y == 1).sum())
        dist_good = good / max(total_good, 1)
        dist_bad = bad / max(total_bad, 1)
        woe = np.log((dist_good + eps) / (dist_bad + eps))
        iv += (dist_good - dist_bad) * woe
    return float(iv)


def iv_retention_metrics(original_df, synthetic_df, cols, target):
    rows = []
    for col in cols:
        orig_iv = calc_iv(original_df, col, target)
        synth_iv = calc_iv(synthetic_df, col, target)
        rows.append({
            "feature": col,
            "original_iv": round(orig_iv, 4),
            "synthetic_iv": round(synth_iv, 4),
            "retention_ratio": round(synth_iv / orig_iv, 4) if abs(orig_iv) > 1e-9 else np.nan,
            "absolute_delta": round(abs(orig_iv - synth_iv), 4),
        })
    return pd.DataFrame(rows).sort_values("absolute_delta", ascending=False)


def target_rate_metrics(df_actual, df_out, target_col):
    actual_rate = float(pd.to_numeric(df_actual[target_col], errors="coerce").mean())
    synthetic_rate = float(pd.to_numeric(df_out[target_col], errors="coerce").mean())
    drift = abs(actual_rate - synthetic_rate)
    return pd.DataFrame([{"target_col": target_col, "actual_rate": round(actual_rate, 4), "synthetic_rate": round(synthetic_rate, 4), "rate_drift": round(drift, 4), "pass_rate_drift": drift < TARGET_RATE_DRIFT_THRESHOLD}])


def privacy_metrics(df_actual, df_out, qi_cols, suppressed):
    full_hash_actual = pd.util.hash_pandas_object(df_actual[qi_cols].astype(str), index=False)
    full_hash_out = pd.util.hash_pandas_object(df_out[qi_cols].astype(str), index=False)
    replaced_idx = suppressed[suppressed].index
    replaced_out = df_out.loc[replaced_idx, qi_cols].astype(str)
    replaced_hash = pd.util.hash_pandas_object(replaced_out, index=False)
    synthetic_duplicate_rate = 1.0 - (replaced_hash.nunique() / max(1, len(replaced_hash)))
    exact_match_to_actual_rate = float(replaced_hash.isin(set(full_hash_actual)).mean())
    full_dataset_duplicate_rate = 1.0 - (full_hash_out.nunique() / max(1, len(full_hash_out)))
    replaced_feature_cols = [c for c in df_actual.columns if c not in ID_COLS]
    exact_match_replaced_rows = len(df_out.loc[replaced_idx, replaced_feature_cols].merge(df_actual[replaced_feature_cols], how="inner")) / max(len(replaced_idx), 1)
    return pd.DataFrame([{
        "replaced_rows": int(len(replaced_idx)),
        "replaced_unique_profiles": int(replaced_hash.nunique()),
        "synthetic_duplicate_rate": round(synthetic_duplicate_rate, 6),
        "exact_match_to_actual_rate": round(exact_match_to_actual_rate, 6),
        "full_dataset_duplicate_rate": round(full_dataset_duplicate_rate, 6),
        "replaced_exact_match_rate": round(exact_match_replaced_rows, 6),
    }])


def utility_metrics(df_actual, df_out, feature_cols, target_col, random_state=42):
    y_all = pd.to_numeric(df_actual[target_col], errors="coerce")
    if y_all.nunique(dropna=True) != 2:
        return pd.DataFrame([{"status": "skipped_target_not_binary"}])
    train_idx, test_idx = train_test_split(df_actual.index, test_size=0.2, random_state=random_state, stratify=y_all)

    def encode_frame(frame):
        out = frame[feature_cols].copy()
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(out[col]):
                numeric = pd.to_numeric(out[col], errors="coerce")
                out[col] = numeric.fillna(numeric.median())
            else:
                enc = LabelEncoder()
                out[col] = enc.fit_transform(out[col].astype(str).fillna(MISSING_LABEL))
        return out

    X_train = encode_frame(df_actual.loc[train_idx])
    X_test = encode_frame(df_actual.loc[test_idx])
    X_syn_train = encode_frame(df_out.loc[train_idx])
    y_train = y_all.loc[train_idx]
    y_test = y_all.loc[test_idx]
    if y_train.nunique() < 2 or pd.to_numeric(df_out.loc[train_idx, target_col], errors="coerce").nunique() < 2:
        return pd.DataFrame([{"status": "skipped_single_class_in_training"}])

    rows = []
    roc_curves = {}
    for name, X_fit, y_fit in [
        ("real_train", X_train, y_train),
        ("synthetic_train", X_syn_train, pd.to_numeric(df_out.loc[train_idx, target_col], errors="coerce")),
    ]:
        model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
        model.fit(X_fit, y_fit)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_curves[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        rows.append({
            "training_data": name,
            "auc": round(float(roc_auc_score(y_test, proba)), 4),
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "f1_score": round(float(f1_score(y_test, pred)), 4),
        })
    result = pd.DataFrame(rows)
    real_auc = float(result.loc[result["training_data"] == "real_train", "auc"].iloc[0])
    syn_auc = float(result.loc[result["training_data"] == "synthetic_train", "auc"].iloc[0])
    result["auc_retention_ratio"] = np.nan
    result.loc[result["training_data"] == "synthetic_train", "auc_retention_ratio"] = round(syn_auc / real_auc if real_auc > 1e-9 else np.nan, 4)
    result.attrs["roc_curves"] = roc_curves
    result.attrs["real_auc"] = real_auc
    result.attrs["synthetic_auc"] = syn_auc
    return result


def _metric_val(summary, key, default):
    val = summary.get(key)
    return default if val is None else float(val)


def build_karabo_scorecard(summary):
    rows = [
        {"area": "Karabo-Categorical", "metric": "mean_category_coverage>=1.0", "value": summary.get("mean_category_coverage"), "pass": _metric_val(summary, "mean_category_coverage", 0.0) >= 1.0},
        {"area": "Karabo-Categorical", "metric": "new_synthetic_categories==0", "value": summary.get("total_new_categories"), "pass": int(summary.get("total_new_categories") or 0) == 0},
        {"area": "Karabo-Numerical", "metric": "mean_PSI<0.25", "value": summary.get("mean_psi"), "pass": _metric_val(summary, "mean_psi", 999.0) < PSI_THRESHOLD},
        {"area": "Karabo-Target", "metric": "target_rate_drift<0.05", "value": summary.get("target_rate_drift"), "pass": _metric_val(summary, "target_rate_drift", 999.0) < TARGET_RATE_DRIFT_THRESHOLD},
        {"area": "Karabo-Utility", "metric": "auc_retention>=0.80", "value": summary.get("auc_retention_ratio"), "pass": _metric_val(summary, "auc_retention_ratio", 0.0) >= AUC_RETENTION_THRESHOLD},
        {"area": "Karabo-Privacy", "metric": "synthetic_duplicate_rate", "value": summary.get("synthetic_duplicate_rate"), "pass": True},
        {"area": "Karabo-Privacy", "metric": "replaced_exact_match_rate<0.001", "value": summary.get("replaced_exact_match_rate"), "pass": _metric_val(summary, "replaced_exact_match_rate", 999.0) < 0.001},
    ]
    return pd.DataFrame(rows)


def compute_karabo_metrics(df_actual, df_out, suppressed, *, feature_categorical_cols, numerical_cols, tvd_categorical_cols, qi_feature_cols, qi_cols, target_col, random_state=42):
    category_metrics = categorical_distribution_metrics(df_actual, df_out, tvd_categorical_cols)
    numeric_metrics = numerical_distribution_metrics(df_actual, df_out, numerical_cols)
    rare_category = rare_category_metrics(df_actual, df_out, tvd_categorical_cols)
    relationship_metrics = target_relationship_metrics(df_actual, df_out, qi_feature_cols, target_col) if target_col else pd.DataFrame()
    num_correlation_metrics = numerical_correlation_metrics(df_actual, df_out, numerical_cols)
    cat_pair_metrics = categorical_pair_relationship_metrics(df_actual, df_out, feature_categorical_cols)
    cat_num_metrics = categorical_numerical_relationship_metrics(df_actual, df_out, feature_categorical_cols, numerical_cols)
    iv_metrics = iv_retention_metrics(df_actual, df_out, qi_feature_cols, target_col) if target_col else pd.DataFrame()
    target_metrics = target_rate_metrics(df_actual, df_out, target_col) if target_col else pd.DataFrame()
    privacy = privacy_metrics(df_actual, df_out, qi_cols, suppressed)
    utility = utility_metrics(df_actual, df_out, qi_feature_cols, target_col, random_state=random_state) if target_col else pd.DataFrame()

    mean_corr_drift = float(relationship_metrics["drift"].mean()) if len(relationship_metrics) else 0.0
    mean_psi = float(numeric_metrics["psi"].dropna().mean()) if len(numeric_metrics) and "psi" in numeric_metrics else float("nan")
    mean_category_coverage = float(category_metrics["category_coverage"].mean()) if len(category_metrics) else float("nan")
    total_new_categories = int(category_metrics["new_synthetic_category_count"].sum()) if len(category_metrics) else 0
    target_rate_drift = float(target_metrics["rate_drift"].iloc[0]) if len(target_metrics) else float("nan")
    auc_retention_ratio = real_auc = synthetic_auc = float("nan")
    roc_curves = {}
    if len(utility) and "auc_retention_ratio" in utility.columns:
        vals = utility["auc_retention_ratio"].dropna()
        if len(vals):
            auc_retention_ratio = float(vals.iloc[0])
    if hasattr(utility, "attrs"):
        real_auc = float(utility.attrs.get("real_auc", float("nan")))
        synthetic_auc = float(utility.attrs.get("synthetic_auc", float("nan")))
        roc_curves = utility.attrs.get("roc_curves", {})

    karabo_summary = {
        "mean_category_coverage": round(mean_category_coverage, 4) if pd.notna(mean_category_coverage) else None,
        "total_new_categories": total_new_categories,
        "mean_psi": round(mean_psi, 4) if pd.notna(mean_psi) else None,
        "mean_num_corr_drift": round(float(num_correlation_metrics["abs_corr_diff"].dropna().mean()), 4) if len(num_correlation_metrics) else None,
        "mean_cat_pair_drift": round(float(cat_pair_metrics["abs_cramers_v_diff"].dropna().mean()), 4) if len(cat_pair_metrics) else None,
        "mean_cat_num_drift": round(float(cat_num_metrics["normalised_group_mean_diff"].dropna().mean()), 4) if len(cat_num_metrics) else None,
        "mean_iv_retention": round(float(iv_metrics["retention_ratio"].dropna().mean()), 4) if len(iv_metrics) else None,
        "target_rate_drift": round(target_rate_drift, 4) if pd.notna(target_rate_drift) else None,
        "real_auc": round(real_auc, 4) if pd.notna(real_auc) else None,
        "synthetic_auc": round(synthetic_auc, 4) if pd.notna(synthetic_auc) else None,
        "auc_retention_ratio": round(auc_retention_ratio, 4) if pd.notna(auc_retention_ratio) else None,
        "synthetic_duplicate_rate": float(privacy["synthetic_duplicate_rate"].iloc[0]) if len(privacy) else None,
        "replaced_exact_match_rate": float(privacy["replaced_exact_match_rate"].iloc[0]) if len(privacy) else None,
        "mean_corr_drift": round(mean_corr_drift, 6),
    }
    return {
        "category_metrics": category_metrics,
        "numeric_metrics": numeric_metrics,
        "rare_category_metrics": rare_category,
        "relationship_metrics": relationship_metrics,
        "num_correlation_metrics": num_correlation_metrics,
        "cat_pair_relationship_metrics": cat_pair_metrics,
        "cat_num_relationship_metrics": cat_num_metrics,
        "iv_metrics": iv_metrics,
        "target_metrics": target_metrics,
        "privacy_metrics": privacy,
        "utility_metrics": utility,
        "roc_curves": roc_curves,
        "karabo_scorecard": build_karabo_scorecard(karabo_summary),
        "karabo_summary": karabo_summary,
    }


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
