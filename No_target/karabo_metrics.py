"""Karabo-style validation metrics for the No_target anonymization pipeline."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

PSI_THRESHOLD = 0.25
RARE_CATEGORY_THRESHOLD = 0.05
TARGET_RATE_DRIFT_THRESHOLD = 0.05
AUC_RETENTION_THRESHOLD = 0.80


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


def categorical_distribution_metrics(real_data: pd.DataFrame, synth_data: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
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
            "pass_tvd": drift < 0.10,
            "real_category_count": len(real_categories),
            "synthetic_category_count": len(synth_categories),
            "category_coverage": round(len(real_categories & synth_categories) / max(1, len(real_categories)), 4),
            "new_synthetic_category_count": len(synth_categories - real_categories),
            "missing_synthetic_category_count": len(real_categories - synth_categories),
            "max_category_proportion_diff": round(float(np.abs(real_aligned - synth_aligned).max()), 4),
        })
    return pd.DataFrame(rows)


def rare_category_metrics(
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    cols: list[str],
    threshold: float = RARE_CATEGORY_THRESHOLD,
) -> pd.DataFrame:
    rows = []
    for col in cols:
        orig_freq = original_df[col].value_counts(normalize=True)
        rare_cats = orig_freq[orig_freq < threshold].index
        if len(rare_cats) == 0:
            rows.append({
                "column": col,
                "rare_category_count": 0,
                "original_rare_mass": 0.0,
                "synthetic_rare_mass": 0.0,
                "retention_ratio": float("nan"),
            })
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


def numerical_distribution_metrics(real_data: pd.DataFrame, synth_data: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
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
            "pass_ks": ks_stat < 0.10 if pd.notna(ks_stat) else False,
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


def target_relationship_metrics(
    df_actual: pd.DataFrame,
    df_out: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        if col == target_col:
            continue
        if col in df_actual.columns and pd.api.types.is_numeric_dtype(df_actual[col]):
            c_a = df_actual[col].astype(float).corr(df_actual[target_col].astype(float))
            c_s = df_out[col].astype(float).corr(df_out[target_col].astype(float))
            if pd.notna(c_a) and pd.notna(c_s):
                rows.append({
                    "label": target_col,
                    "column": col,
                    "metric": "correlation",
                    "actual": round(float(c_a), 4),
                    "synthetic": round(float(c_s), 4),
                    "drift": round(abs(float(c_a) - float(c_s)), 4),
                })
        else:
            v_a = cramers_v(df_actual[col], df_actual[target_col])
            v_s = cramers_v(df_out[col], df_out[target_col])
            rows.append({
                "label": target_col,
                "column": col,
                "metric": "cramers_v",
                "actual": round(v_a, 4),
                "synthetic": round(v_s, 4),
                "drift": round(abs(v_a - v_s), 4),
            })
    return pd.DataFrame(rows)


def numerical_correlation_metrics(real_data: pd.DataFrame, synth_data: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    if len(num_cols) < 2:
        return pd.DataFrame()
    real_corr = real_data[num_cols].astype(float).corr()
    synth_corr = synth_data[num_cols].astype(float).corr()
    rows = []
    for c1, c2 in itertools.combinations(num_cols, 2):
        real_val = real_corr.loc[c1, c2]
        synth_val = synth_corr.loc[c1, c2]
        rows.append({
            "feature_1": c1,
            "feature_2": c2,
            "real_corr": round(float(real_val), 4) if pd.notna(real_val) else np.nan,
            "synthetic_corr": round(float(synth_val), 4) if pd.notna(synth_val) else np.nan,
            "abs_corr_diff": round(abs(float(real_val) - float(synth_val)), 4)
            if pd.notna(real_val) and pd.notna(synth_val) else np.nan,
        })
    return pd.DataFrame(rows)


def categorical_pair_relationship_metrics(real_data: pd.DataFrame, synth_data: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    rows = []
    for c1, c2 in itertools.combinations(cat_cols, 2):
        real_v = cramers_v(real_data[c1], real_data[c2])
        synth_v = cramers_v(synth_data[c1], synth_data[c2])
        rows.append({
            "feature_1": c1,
            "feature_2": c2,
            "real_cramers_v": round(real_v, 4) if pd.notna(real_v) else np.nan,
            "synthetic_cramers_v": round(synth_v, 4) if pd.notna(synth_v) else np.nan,
            "abs_cramers_v_diff": round(abs(real_v - synth_v), 4) if pd.notna(real_v) and pd.notna(synth_v) else np.nan,
        })
    return pd.DataFrame(rows)


def categorical_numerical_relationship_metrics(
    real_data: pd.DataFrame,
    synth_data: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    max_categories: int = 25,
) -> pd.DataFrame:
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
            rows.append({
                "categorical_column": cat,
                "numerical_column": num,
                "common_category_count": len(common_groups),
                "normalised_group_mean_diff": round(mean_group_diff, 4),
            })
    return pd.DataFrame(rows)


def calc_iv(df: pd.DataFrame, feature: str, target: str) -> float:
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


def iv_retention_metrics(original_df: pd.DataFrame, synthetic_df: pd.DataFrame, cols: list[str], target: str) -> pd.DataFrame:
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


def target_rate_metrics(df_actual: pd.DataFrame, df_out: pd.DataFrame, target_col: str) -> pd.DataFrame:
    actual_rate = float(pd.to_numeric(df_actual[target_col], errors="coerce").mean())
    synthetic_rate = float(pd.to_numeric(df_out[target_col], errors="coerce").mean())
    drift = abs(actual_rate - synthetic_rate)
    return pd.DataFrame([{
        "target_col": target_col,
        "actual_rate": round(actual_rate, 4),
        "synthetic_rate": round(synthetic_rate, 4),
        "rate_drift": round(drift, 4),
        "pass_rate_drift": drift < TARGET_RATE_DRIFT_THRESHOLD,
    }])


def privacy_metrics(df_actual: pd.DataFrame, df_out: pd.DataFrame, qi_cols: list[str], suppressed: pd.Series) -> pd.DataFrame:
    full_hash_actual = pd.util.hash_pandas_object(df_actual[qi_cols].astype(str), index=False)
    full_hash_out = pd.util.hash_pandas_object(df_out[qi_cols].astype(str), index=False)
    replaced_idx = suppressed[suppressed].index
    replaced_out = df_out.loc[replaced_idx, qi_cols].astype(str)
    replaced_hash = pd.util.hash_pandas_object(replaced_out, index=False)
    synthetic_duplicate_rate = 1.0 - (replaced_hash.nunique() / max(1, len(replaced_hash)))
    exact_match_to_actual_rate = float(replaced_hash.isin(set(full_hash_actual)).mean())
    full_dataset_duplicate_rate = 1.0 - (full_hash_out.nunique() / max(1, len(full_hash_out)))
    replaced_feature_cols = [c for c in df_actual.columns if c != "customer_id"]
    exact_match_replaced_rows = len(
        df_out.loc[replaced_idx, replaced_feature_cols].merge(df_actual[replaced_feature_cols], how="inner")
    ) / max(len(replaced_idx), 1)
    return pd.DataFrame([{
        "replaced_rows": int(len(replaced_idx)),
        "replaced_unique_profiles": int(replaced_hash.nunique()),
        "synthetic_duplicate_rate": round(synthetic_duplicate_rate, 6),
        "exact_match_to_actual_rate": round(exact_match_to_actual_rate, 6),
        "full_dataset_duplicate_rate": round(full_dataset_duplicate_rate, 6),
        "replaced_exact_match_rate": round(exact_match_replaced_rows, 6),
    }])


def utility_metrics(
    df_actual: pd.DataFrame,
    df_out: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    random_state: int = 42,
) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    y_all = pd.to_numeric(df_actual[target_col], errors="coerce")
    if y_all.nunique(dropna=True) != 2:
        return pd.DataFrame([{"status": "skipped_target_not_binary"}])

    train_idx, test_idx = train_test_split(
        df_actual.index, test_size=0.2, random_state=random_state, stratify=y_all
    )

    def encode_frame(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[feature_cols].copy()
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(out[col]):
                numeric = pd.to_numeric(out[col], errors="coerce")
                out[col] = numeric.fillna(numeric.median())
            else:
                enc = LabelEncoder()
                out[col] = enc.fit_transform(out[col].astype(str).fillna("Missing"))
        return out

    X_train = encode_frame(df_actual.loc[train_idx])
    X_test = encode_frame(df_actual.loc[test_idx])
    X_syn_train = encode_frame(df_out.loc[train_idx])
    y_train = y_all.loc[train_idx]
    y_test = y_all.loc[test_idx]

    if y_train.nunique() < 2 or pd.to_numeric(df_out.loc[train_idx, target_col], errors="coerce").nunique() < 2:
        return pd.DataFrame([{"status": "skipped_single_class_in_training"}])

    rows = []
    for name, X_fit, y_fit in [
        ("real_train", X_train, y_train),
        ("synthetic_train", X_syn_train, pd.to_numeric(df_out.loc[train_idx, target_col], errors="coerce")),
    ]:
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
        model.fit(X_fit, y_fit)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
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
    result.loc[result["training_data"] == "synthetic_train", "auc_retention_ratio"] = round(
        syn_auc / real_auc if real_auc > 1e-9 else np.nan, 4
    )
    return result


def _metric_val(summary: dict, key: str, default: float) -> float:
    val = summary.get(key)
    return default if val is None else float(val)


def build_karabo_scorecard(summary: dict) -> pd.DataFrame:
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


def compute_karabo_metrics(
    df_actual: pd.DataFrame,
    df_out: pd.DataFrame,
    suppressed: pd.Series,
    *,
    feature_categorical_cols: list[str],
    numerical_cols: list[str],
    tvd_categorical_cols: list[str],
    qi_feature_cols: list[str],
    qi_cols: list[str],
    target_col: str | None,
    random_state: int = 42,
) -> dict[str, Any]:
    category_metrics = categorical_distribution_metrics(df_actual, df_out, tvd_categorical_cols)
    numeric_metrics = numerical_distribution_metrics(df_actual, df_out, numerical_cols)
    rare_category = rare_category_metrics(df_actual, df_out, tvd_categorical_cols)
    relationship_metrics = (
        target_relationship_metrics(df_actual, df_out, qi_feature_cols, target_col)
        if target_col else pd.DataFrame()
    )
    num_correlation_metrics = numerical_correlation_metrics(df_actual, df_out, numerical_cols)
    cat_pair_metrics = categorical_pair_relationship_metrics(df_actual, df_out, feature_categorical_cols)
    cat_num_metrics = categorical_numerical_relationship_metrics(
        df_actual, df_out, feature_categorical_cols, numerical_cols
    )
    iv_metrics = (
        iv_retention_metrics(df_actual, df_out, qi_feature_cols, target_col)
        if target_col else pd.DataFrame()
    )
    target_metrics = target_rate_metrics(df_actual, df_out, target_col) if target_col else pd.DataFrame()
    privacy = privacy_metrics(df_actual, df_out, qi_cols, suppressed)
    utility = (
        utility_metrics(df_actual, df_out, qi_feature_cols, target_col, random_state=random_state)
        if target_col else pd.DataFrame()
    )

    mean_corr_drift = float(relationship_metrics["drift"].mean()) if len(relationship_metrics) else 0.0
    mean_psi = float(numeric_metrics["psi"].dropna().mean()) if len(numeric_metrics) and "psi" in numeric_metrics else float("nan")
    mean_category_coverage = float(category_metrics["category_coverage"].mean()) if len(category_metrics) else float("nan")
    total_new_categories = int(category_metrics["new_synthetic_category_count"].sum()) if len(category_metrics) else 0
    target_rate_drift = float(target_metrics["rate_drift"].iloc[0]) if len(target_metrics) else float("nan")
    auc_retention_ratio = float("nan")
    if len(utility) and "auc_retention_ratio" in utility.columns:
        vals = utility["auc_retention_ratio"].dropna()
        if len(vals):
            auc_retention_ratio = float(vals.iloc[0])

    karabo_summary = {
        "mean_category_coverage": round(mean_category_coverage, 4) if pd.notna(mean_category_coverage) else None,
        "total_new_categories": total_new_categories,
        "mean_psi": round(mean_psi, 4) if pd.notna(mean_psi) else None,
        "mean_num_corr_drift": round(float(num_correlation_metrics["abs_corr_diff"].dropna().mean()), 4)
        if len(num_correlation_metrics) else None,
        "mean_cat_pair_drift": round(float(cat_pair_metrics["abs_cramers_v_diff"].dropna().mean()), 4)
        if len(cat_pair_metrics) else None,
        "mean_cat_num_drift": round(float(cat_num_metrics["normalised_group_mean_diff"].dropna().mean()), 4)
        if len(cat_num_metrics) else None,
        "mean_iv_retention": round(float(iv_metrics["retention_ratio"].dropna().mean()), 4) if len(iv_metrics) else None,
        "target_rate_drift": round(target_rate_drift, 4) if pd.notna(target_rate_drift) else None,
        "auc_retention_ratio": round(auc_retention_ratio, 4) if pd.notna(auc_retention_ratio) else None,
        "synthetic_duplicate_rate": float(privacy["synthetic_duplicate_rate"].iloc[0]) if len(privacy) else None,
        "replaced_exact_match_rate": float(privacy["replaced_exact_match_rate"].iloc[0]) if len(privacy) else None,
        "mean_corr_drift": round(mean_corr_drift, 6),
    }
    karabo_scorecard = build_karabo_scorecard(karabo_summary)

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
        "karabo_scorecard": karabo_scorecard,
        "karabo_summary": karabo_summary,
    }
