"""
Shared library for Story 02 KNN anonymization notebooks.
Imported by all experiment notebooks in this folder.
"""

from __future__ import annotations

import json
import time
import tracemalloc
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

# --- Paths ---
STORY_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STORY_ROOT.parent
DATA_PATH = PROJECT_ROOT / "Bank Customer Churn Prediction.csv"
ITERATIONS_DIR = STORY_ROOT / "iterations"
RESULTS_DIR = STORY_ROOT / "results"

# --- Columns ---
ID_COLS = ["customer_id"]
TARGET_COL = "churn"
CATEGORICAL_COLS = ["country", "gender"]
NUMERICAL_COLS = [
    "credit_score",
    "age",
    "tenure",
    "balance",
    "products_number",
    "credit_card",
    "active_member",
    "estimated_salary",
]
QI_COLS = CATEGORICAL_COLS + NUMERICAL_COLS
FEATURE_COLS = QI_COLS + [TARGET_COL]

# Columns used for k-anonymity equivalence classes (exclude ultra-high-cardinality numerics)
KANON_QI_COLS = CATEGORICAL_COLS + [
    "age",
    "tenure",
    "products_number",
    "credit_card",
    "active_member",
]

MISSING_LABEL = "Missing"
HIGH_CARDINALITY_THRESHOLD = 0.5
TVD_THRESHOLD = 0.10
KS_THRESHOLD = 0.10
PASS_RATE_TARGET = 0.85

# Distance design space
NUM_DISTANCE_METRICS = ["euclidean", "manhattan", "minkowski"]
CAT_DISTANCE_METRICS = ["hamming", "jaccard", "overlap"]
DISTANCE_MODES = ["weighted_sum", "gower"]
DEFAULT_MINKOWSKI_P = 3


def _utility_model(random_state: int):
    return LogisticRegression(max_iter=500, random_state=random_state)


@dataclass
class ExperimentConfig:
    name: str
    folder: str
    k_anonymity: int = 5
    k_neighbors: int = 15
    cat_weight: float = 1.0
    num_weight: float = 1.0
    scaler_method: str = "standard"
    cat_gen_method: str = "weighted_mode"
    target_gen_method: str = "probability"
    num_gen_method: str = "interpolation"
    random_state: int = 42
    description: str = ""
    distance_mode: str = "weighted_sum"
    num_distance_metric: str = "euclidean"
    cat_distance_metric: str = "hamming"
    minkowski_p: int = DEFAULT_MINKOWSKI_P


from itertools import product

# Story 02 design-space axes (Area 2 + Area 3)
K_NEIGHBORS_GRID = [5, 10, 15, 25]
CAT_GEN_GRID = ["mode", "weighted_mode", "probability"]
NUM_GEN_GRID = ["interpolation", "weighted_mean"]
SCALER_GRID = ["standard", "minmax", "robust"]
TARGET_GEN_GRID = ["mode", "weighted_mode", "probability"]
DISTANCE_GRID = [
    ("balanced", 1.0, 1.0),
    ("num_heavy", 2.0, 0.5),
    ("cat_heavy", 0.5, 2.0),
]


def _short(name: str) -> str:
    return name.replace("_", "-")


def build_experiment_grid() -> list[ExperimentConfig]:
    """Curated Story 02 grid — readable folder names under iterations/."""
    specs = [
        # k, cat_gen, num_gen, scaler, dist_tag, nw, cw, tgt_gen, seed, display_name, folder
        (
            15,
            "weighted_mode",
            "interpolation",
            "standard",
            "balanced",
            1.0,
            1.0,
            "probability",
            42,
            "Baseline",
            "01_baseline",
        ),
        (
            25,
            "weighted_mode",
            "interpolation",
            "standard",
            "balanced",
            1.0,
            1.0,
            "probability",
            42,
            "Neighbors K25",
            "03_neighbors_k25",
        ),
        (
            15,
            "weighted_mode",
            "weighted_mean",
            "standard",
            "balanced",
            1.0,
            1.0,
            "probability",
            42,
            "Numeric Weighted Mean",
            "05_numeric_weighted_mean",
        ),
        (
            15,
            "mode",
            "interpolation",
            "standard",
            "balanced",
            1.0,
            1.0,
            "probability",
            42,
            "Category Mode",
            "06_category_mode",
        ),
        (
            15,
            "probability",
            "interpolation",
            "standard",
            "balanced",
            1.0,
            1.0,
            "probability",
            42,
            "Category Probability",
            "07_category_probability",
        ),
        (
            15,
            "weighted_mode",
            "interpolation",
            "robust",
            "balanced",
            1.0,
            1.0,
            "probability",
            42,
            "Scaler Robust",
            "08_scaler_robust",
        ),
        (
            15,
            "weighted_mode",
            "interpolation",
            "minmax",
            "balanced",
            1.0,
            1.0,
            "probability",
            42,
            "Scaler MinMax",
            "09_scaler_minmax",
        ),
        (
            15,
            "weighted_mode",
            "interpolation",
            "standard",
            "num_heavy",
            2.0,
            0.5,
            "probability",
            42,
            "Distance Numeric Heavy",
            "10_distance_numeric_heavy",
        ),
        (
            15,
            "weighted_mode",
            "interpolation",
            "standard",
            "cat_heavy",
            0.5,
            2.0,
            "probability",
            42,
            "Distance Category Heavy",
            "11_distance_category_heavy",
        ),
        (
            15,
            "weighted_mode",
            "interpolation",
            "standard",
            "balanced",
            1.0,
            1.0,
            "mode",
            42,
            "Target Mode",
            "12_target_mode",
        ),
    ]
    configs: list[ExperimentConfig] = []
    for (
        k,
        cat_gen,
        num_gen,
        scaler,
        dist_tag,
        nw,
        cw,
        tgt_gen,
        seed,
        display_name,
        folder,
    ) in specs:
        configs.append(
            ExperimentConfig(
                name=display_name,
                folder=folder,
                k_neighbors=k,
                cat_weight=cw,
                num_weight=nw,
                scaler_method=scaler,
                cat_gen_method=cat_gen,
                num_gen_method=num_gen,
                target_gen_method=tgt_gen,
                random_state=seed,
                description=(
                    f"K={k}, cat={cat_gen}, num={num_gen}, scaler={scaler}, "
                    f"dist={dist_tag}, target={tgt_gen}, seed={seed}"
                ),
            )
        )

    distance_variants = [
        (
            "Distance Manhattan",
            "13_distance_manhattan",
            dict(num_distance_metric="manhattan"),
            "K=15, numeric distance=manhattan (L1), hamming categorical, weighted_sum",
        ),
        (
            "Distance Minkowski",
            "14_distance_minkowski",
            dict(num_distance_metric="minkowski", minkowski_p=DEFAULT_MINKOWSKI_P),
            f"K=15, numeric distance=minkowski (p={DEFAULT_MINKOWSKI_P}), hamming categorical, weighted_sum",
        ),
        (
            "Distance Gower",
            "15_distance_gower",
            dict(distance_mode="gower"),
            "K=15, mixed Gower distance (range-normalized numeric + categorical mismatch)",
        ),
        (
            "Distance Categorical Jaccard",
            "16_distance_cat_jaccard",
            dict(cat_distance_metric="jaccard"),
            "K=15, categorical distance=jaccard, weighted_sum",
        ),
        (
            "Distance Categorical Overlap",
            "17_distance_cat_overlap",
            dict(cat_distance_metric="overlap"),
            "K=15, categorical distance=overlap, weighted_sum",
        ),
    ]
    for display_name, folder, dist_overrides, desc in distance_variants:
        configs.append(
            ExperimentConfig(
                name=display_name,
                folder=folder,
                k_neighbors=15,
                cat_weight=1.0,
                num_weight=1.0,
                scaler_method="standard",
                cat_gen_method="weighted_mode",
                num_gen_method="interpolation",
                target_gen_method="probability",
                random_state=42,
                description=desc,
                **dist_overrides,
            )
        )
    return configs


def build_full_experiment_grid() -> list[ExperimentConfig]:
    """Full factorial grid: 4×3×2×3×3×3 = 648 configs."""
    configs: list[ExperimentConfig] = []
    idx = 1
    for k, cat_gen, num_gen, scaler, (dist_tag, nw, cw), tgt_gen in product(
        K_NEIGHBORS_GRID,
        CAT_GEN_GRID,
        NUM_GEN_GRID,
        SCALER_GRID,
        DISTANCE_GRID,
        TARGET_GEN_GRID,
    ):
        folder = (
            f"{idx:03d}_k{k}_cat-{cat_gen}_num-{num_gen}_"
            f"scale-{scaler}_dist-{dist_tag}_tgt-{tgt_gen}"
        )
        configs.append(
            ExperimentConfig(
                name=(
                    f"K={k} cat={cat_gen} num={num_gen} scale={scaler} "
                    f"dist={dist_tag} tgt={tgt_gen}"
                ),
                folder=folder,
                k_neighbors=k,
                cat_weight=cw,
                num_weight=nw,
                scaler_method=scaler,
                cat_gen_method=cat_gen,
                num_gen_method=num_gen,
                target_gen_method=tgt_gen,
                description=(
                    f"K={k}, cat={cat_gen}, num={num_gen}, scaler={scaler}, "
                    f"dist={dist_tag} (num={nw}, cat={cw}), target={tgt_gen}"
                ),
            )
        )
        idx += 1
    return configs


# Axes aligned with results/parameter_reference.csv
REF_K_NEIGHBORS = [15, 25]
REF_CAT_GEN = ["mode", "weighted_mode", "probability"]
REF_NUM_GEN = ["interpolation", "weighted_mean"]
REF_TARGET_GEN = ["mode", "weighted_mode", "probability"]
REF_SCALERS = ["standard", "minmax", "robust"]
REF_DISTANCE_MODES = ["weighted_sum", "gower"]
REF_NUM_METRICS = ["euclidean", "manhattan", "minkowski"]
REF_CAT_METRICS = ["hamming", "jaccard", "overlap"]
REF_DISTANCE_PROFILES = [
    ("balanced", 1.0, 1.0),
    ("num_heavy", 2.0, 0.5),
    ("cat_heavy", 0.5, 2.0),
]


def _reference_config_key(cfg: ExperimentConfig) -> tuple:
    return (
        cfg.k_neighbors,
        cfg.cat_gen_method,
        cfg.num_gen_method,
        cfg.target_gen_method,
        cfg.scaler_method,
        cfg.distance_mode,
        cfg.num_distance_metric,
        cfg.cat_distance_metric,
        cfg.num_weight,
        cfg.cat_weight,
        cfg.minkowski_p,
    )


def build_reference_experiment_grid() -> list[ExperimentConfig]:
    """Curated grid from parameter_reference.csv — includes cat_distance_metric sweep."""
    seen: dict[tuple, ExperimentConfig] = {}
    idx = 1

    def add(
        k: int,
        cat_gen: str,
        num_gen: str,
        target_gen: str,
        scaler: str,
        dist_mode: str,
        num_metric: str,
        profile: str,
        nw: float,
        cw: float,
        cat_metric: str = "hamming",
    ) -> None:
        nonlocal idx
        cfg = ExperimentConfig(
            name=(
                f"K={k} cat={cat_gen} num={num_gen} tgt={target_gen} "
                f"scale={scaler} {dist_mode}/{num_metric}/{cat_metric}/{profile}"
            ),
            folder=(
                f"{idx:03d}_k{k}_cat-{cat_gen}_num-{num_gen}_tgt-{target_gen}_"
                f"scale-{scaler}_{dist_mode}-{num_metric}-cat-{cat_metric}_w-{profile}"
            ),
            k_neighbors=k,
            cat_gen_method=cat_gen,
            num_gen_method=num_gen,
            target_gen_method=target_gen,
            scaler_method=scaler,
            distance_mode=dist_mode,
            num_distance_metric=num_metric,
            cat_distance_metric=cat_metric,
            num_weight=nw,
            cat_weight=cw,
            description=(
                f"K={k}, cat={cat_gen}, num={num_gen}, target={target_gen}, "
                f"scaler={scaler}, mode={dist_mode}, num_metric={num_metric}, "
                f"cat_metric={cat_metric}, weights={profile} ({nw}/{cw})"
            ),
        )
        key = _reference_config_key(cfg)
        if key not in seen:
            seen[key] = cfg
            idx += 1

    # Block 1: primary factorial — k × scaler × num metric × profile × cat metric + gower
    for k, scaler in product(REF_K_NEIGHBORS, REF_SCALERS):
        for num_metric, (profile, nw, cw), cat_metric in product(
            REF_NUM_METRICS, REF_DISTANCE_PROFILES, REF_CAT_METRICS
        ):
            add(
                k,
                "weighted_mode",
                "interpolation",
                "probability",
                scaler,
                "weighted_sum",
                num_metric,
                profile,
                nw,
                cw,
                cat_metric,
            )
        add(
            k,
            "weighted_mode",
            "interpolation",
            "probability",
            scaler,
            "gower",
            "euclidean",
            "balanced",
            1.0,
            1.0,
            "hamming",
        )

    # Block 2: generation-method sweep at baseline distance settings × cat metrics
    for cat_gen, num_gen, target_gen, cat_metric in product(
        REF_CAT_GEN, REF_NUM_GEN, REF_TARGET_GEN, REF_CAT_METRICS
    ):
        add(
            15,
            cat_gen,
            num_gen,
            target_gen,
            "standard",
            "weighted_sum",
            "euclidean",
            "balanced",
            1.0,
            1.0,
            cat_metric,
        )

    configs = list(seen.values())
    for i, cfg in enumerate(configs, start=1):
        if cfg.num_weight == cfg.cat_weight == 1.0:
            wlabel = "balanced"
        elif cfg.num_weight > cfg.cat_weight:
            wlabel = "num_heavy"
        else:
            wlabel = "cat_heavy"
        cat_tag = (
            f"-cat-{cfg.cat_distance_metric}"
            if cfg.distance_mode == "weighted_sum"
            else ""
        )
        cfg.folder = (
            f"{i:03d}_k{cfg.k_neighbors}_cat-{cfg.cat_gen_method}_"
            f"num-{cfg.num_gen_method}_tgt-{cfg.target_gen_method}_"
            f"scale-{cfg.scaler_method}_{cfg.distance_mode}-"
            f"{cfg.num_distance_metric}{cat_tag}_w-{wlabel}"
        )
    return configs


REFERENCE_EXPERIMENT_GRID: list[ExperimentConfig] = build_reference_experiment_grid()


EXPERIMENT_GRID: list[ExperimentConfig] = build_experiment_grid()
FULL_EXPERIMENT_GRID: list[ExperimentConfig] = build_full_experiment_grid()


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA_PATH)
    df = df.replace(r"^\s*$", np.nan, regex=True)
    empty = [c for c in df.columns if df[c].isna().all()]
    if empty:
        df = df.drop(columns=empty)
    return df.reset_index(drop=True)


def dataset_profile(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    rows = []
    for col in df.columns:
        n_unique = df[col].nunique()
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "role": (
                    "id"
                    if col in ID_COLS
                    else (
                        "target"
                        if col == TARGET_COL
                        else (
                            "categorical"
                            if col in CATEGORICAL_COLS
                            else "numerical" if col in NUMERICAL_COLS else "other"
                        )
                    )
                ),
                "missing_pct": round(100 * df[col].isna().mean(), 2),
                "n_unique": n_unique,
                "cardinality_ratio": round(n_unique / n, 4),
                "high_cardinality": n_unique / n > HIGH_CARDINALITY_THRESHOLD,
            }
        )
    return pd.DataFrame(rows)


def generalize_for_kanonymity(df: pd.DataFrame, qi_cols: list[str]) -> pd.DataFrame:
    g = df[qi_cols].copy()
    if "credit_score" in g.columns:
        g["credit_score"] = (g["credit_score"] // 50) * 50
    if "age" in g.columns:
        g["age"] = (g["age"] // 5) * 5
    if "balance" in g.columns:
        g["balance"] = pd.qcut(
            g["balance"].rank(method="first"), q=20, duplicates="drop"
        ).astype(str)
    if "estimated_salary" in g.columns:
        g["estimated_salary"] = pd.qcut(
            g["estimated_salary"].rank(method="first"), q=20, duplicates="drop"
        ).astype(str)
    for col in g.select_dtypes(include="object").columns:
        g[col] = g[col].astype(str).fillna(MISSING_LABEL)
    return g


def identify_suppressed(df: pd.DataFrame, k: int = 5) -> tuple[pd.Series, pd.DataFrame]:
    """Step A: k-anonymity on generalized quasi-identifiers."""
    generalized = generalize_for_kanonymity(df, KANON_QI_COLS)
    class_sizes = generalized.groupby(
        list(generalized.columns), dropna=False
    ).transform("size")
    suppressed = class_sizes < k
    single_col_unique = pd.Series(False, index=df.index)
    for col in KANON_QI_COLS:
        single_col_unique |= df.groupby(col)[col].transform("size") == 1
    summary = pd.DataFrame(
        {
            "n_rows": len(df),
            "k_anonymity": k,
            "n_suppressed_tuple": int(suppressed.sum()),
            "pct_suppressed_tuple": round(100 * suppressed.mean(), 2),
            "n_single_column_unique": int(single_col_unique.sum()),
            "pct_single_column_unique": round(100 * single_col_unique.mean(), 2),
        },
        index=[0],
    )
    return suppressed, summary


def fit_scaler(method: str, X: np.ndarray):
    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }
    return scalers[method]().fit(X)


def fit_preprocessors(df_pool: pd.DataFrame, cfg: ExperimentConfig):
    cat_encoders = {}
    for col in CATEGORICAL_COLS:
        enc = LabelEncoder()
        enc.fit(df_pool[col].astype(str).fillna(MISSING_LABEL))
        cat_encoders[col] = enc
    cat_domains = {c: list(cat_encoders[c].classes_) for c in CATEGORICAL_COLS}
    num_medians = df_pool[NUMERICAL_COLS].median()
    num_scaler = fit_scaler(
        cfg.scaler_method, df_pool[NUMERICAL_COLS].fillna(num_medians).values
    )
    num_p01 = {c: np.percentile(df_pool[c].dropna(), 1) for c in NUMERICAL_COLS}
    num_p99 = {c: np.percentile(df_pool[c].dropna(), 99) for c in NUMERICAL_COLS}
    return cat_encoders, cat_domains, num_medians, num_scaler, num_p01, num_p99


def encode_cats(frame: pd.DataFrame, cat_encoders: dict) -> np.ndarray:
    return np.column_stack(
        [
            cat_encoders[c].transform(frame[c].astype(str).fillna(MISSING_LABEL))
            for c in CATEGORICAL_COLS
        ]
    )


def hamming_cat_distance(cat_a: np.ndarray, cat_b: np.ndarray) -> np.ndarray:
    if cat_a.ndim == 1:
        cat_a = cat_a.reshape(1, -1)
    if cat_b.ndim == 1:
        cat_b = cat_b.reshape(1, -1)
    return np.mean(cat_a != cat_b, axis=1)


def numeric_pairwise_distance(
    pool_num: np.ndarray,
    synth_num: np.ndarray,
    metric: str,
    p: int = DEFAULT_MINKOWSKI_P,
) -> np.ndarray:
    """Pairwise numeric distances: pool (n_pool, f) vs synth (n_synth, f) → (n_synth, n_pool)."""
    diff = pool_num[np.newaxis, :, :] - synth_num[:, np.newaxis, :]
    if metric == "euclidean":
        return np.linalg.norm(diff, axis=2)
    if metric == "manhattan":
        return np.sum(np.abs(diff), axis=2)
    if metric == "minkowski":
        return np.sum(np.abs(diff) ** p, axis=2) ** (1.0 / p)
    raise ValueError(f"Unknown numeric distance metric: {metric}")


def categorical_pairwise_distance(
    pool_cat: np.ndarray,
    synth_cat: np.ndarray,
    metric: str,
) -> np.ndarray:
    mismatch = pool_cat[np.newaxis, :, :] != synth_cat[:, np.newaxis, :]
    n_cat = pool_cat.shape[1]
    if n_cat == 0:
        return np.zeros((len(synth_cat), len(pool_cat)))
    matches = (~mismatch).sum(axis=2).astype(float)
    if metric == "hamming":
        return np.mean(mismatch, axis=2)
    if metric == "jaccard":
        union = 2.0 * n_cat - matches
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0, matches / union, 1.0)
        return 1.0 - sim
    if metric == "overlap":
        return 1.0 - matches / n_cat
    raise ValueError(f"Unknown categorical distance metric: {metric}")


def gower_pairwise_distance(
    pool_num: np.ndarray,
    synth_num: np.ndarray,
    pool_cat: np.ndarray,
    synth_cat: np.ndarray,
    num_ranges: np.ndarray,
) -> np.ndarray:
    """Gower distance averaged across numeric + categorical columns."""
    n_synth, n_pool = len(synth_num), len(pool_num)
    n_num = pool_num.shape[1]
    n_cat = pool_cat.shape[1]
    n_cols = n_num + n_cat
    if n_cols == 0:
        return np.zeros((n_synth, n_pool))

    total = np.zeros((n_synth, n_pool))
    safe_ranges = np.where(num_ranges < 1e-8, 1.0, num_ranges)
    for j in range(n_num):
        total += (
            np.abs(pool_num[np.newaxis, :, j] - synth_num[:, np.newaxis, j])
            / safe_ranges[j]
        )
    for j in range(n_cat):
        total += (pool_cat[np.newaxis, :, j] != synth_cat[:, np.newaxis, j]).astype(
            float
        )
    return total / n_cols


def mixed_pairwise_distance(
    pool_num: np.ndarray,
    synth_num: np.ndarray,
    pool_cat: np.ndarray,
    synth_cat: np.ndarray,
    num_ranges: np.ndarray,
    cfg: ExperimentConfig,
) -> np.ndarray:
    if cfg.distance_mode == "gower":
        return gower_pairwise_distance(
            pool_num, synth_num, pool_cat, synth_cat, num_ranges
        )
    if cfg.distance_mode != "weighted_sum":
        raise ValueError(f"Unknown distance_mode: {cfg.distance_mode}")

    num_dist = numeric_pairwise_distance(
        pool_num, synth_num, cfg.num_distance_metric, cfg.minkowski_p
    )
    cat_dist = categorical_pairwise_distance(
        pool_cat, synth_cat, cfg.cat_distance_metric
    )
    return cfg.num_weight * num_dist + cfg.cat_weight * cat_dist


def neighbor_cache_key(cfg: ExperimentConfig) -> tuple:
    return (
        cfg.scaler_method,
        cfg.num_weight,
        cfg.cat_weight,
        cfg.distance_mode,
        cfg.num_distance_metric,
        cfg.cat_distance_metric,
        cfg.minkowski_p if cfg.num_distance_metric == "minkowski" else 0,
    )


def find_neighbours(
    base_num: np.ndarray,
    base_cat: np.ndarray,
    pool_num: np.ndarray,
    pool_cat: np.ndarray,
    pool_idx: np.ndarray,
    k: int,
    cfg: ExperimentConfig,
    num_ranges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if num_ranges is None:
        num_ranges = np.ones(base_num.shape[-1], dtype=float)
    total = mixed_pairwise_distance(
        pool_num,
        base_num.reshape(1, -1),
        pool_cat,
        base_cat.reshape(1, -1),
        num_ranges,
        cfg,
    )[0]
    idx_local = np.argpartition(total, min(k, len(total) - 1))[:k]
    order = idx_local[np.argsort(total[idx_local])]
    return total[order], pool_idx[order]


def neighbor_weights(distances: np.ndarray) -> np.ndarray:
    w = 1.0 / (distances + 1e-8)
    return w / w.sum()


def generate_categorical(
    values: np.ndarray, weights: np.ndarray, method: str, rng: np.random.Generator
) -> str:
    if method == "mode":
        return Counter(values).most_common(1)[0][0]
    if method == "weighted_mode":
        counts: dict[str, float] = defaultdict(float)
        for v, w in zip(values, weights):
            counts[str(v)] += w
        return max(counts, key=counts.get)
    if method == "probability":
        probs = weights / weights.sum()
        return str(rng.choice(values, p=probs))
    raise ValueError(method)


K_MAX = max(K_NEIGHBORS_GRID)


def precompute_neighbor_cache(
    ctx: dict,
    cfg: ExperimentConfig,
    k_max: int = K_MAX,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Batch-precompute top-k neighbours for every suppressed row."""
    pool_idx = ctx["pool_idx"]
    synth_idx = ctx["synth_idx"]
    X_num, X_cat = ctx["X_num"], ctx["X_cat"]
    pool_cat = X_cat[pool_idx]
    synth_cat = X_cat[synth_idx]

    if cfg.distance_mode == "gower":
        pool_num = ctx["num_raw"][pool_idx]
        synth_num = ctx["num_raw"][synth_idx]
    else:
        pool_num = X_num[pool_idx]
        synth_num = X_num[synth_idx]

    total = mixed_pairwise_distance(
        pool_num,
        synth_num,
        pool_cat,
        synth_cat,
        ctx["num_ranges"],
        cfg,
    )

    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    k = min(k_max, len(pool_idx))
    for local_i, global_i in enumerate(synth_idx):
        row_dist = total[local_i]
        idx_local = np.argpartition(row_dist, k - 1)[:k]
        order = idx_local[np.argsort(row_dist[idx_local])]
        cache[int(global_i)] = (row_dist[order], pool_idx[order])
    return cache


def synthesize_row(
    row_idx: int,
    df: pd.DataFrame,
    X_num: np.ndarray,
    X_cat: np.ndarray,
    num_scaler,
    num_p01: dict,
    num_p99: dict,
    cfg: ExperimentConfig,
    rng: np.random.Generator,
    neighbor_cache: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    if neighbor_cache is not None:
        dists_full, nbrs_full = neighbor_cache[int(row_idx)]
        k = min(cfg.k_neighbors, len(dists_full))
        dists, nbrs = dists_full[:k], nbrs_full[:k]
    else:
        pool_idx = np.arange(len(df))  # unused fallback
        raise ValueError("neighbor_cache required")

    w = neighbor_weights(dists)
    out = {}

    for col in CATEGORICAL_COLS:
        vals = df.loc[nbrs, col].astype(str).fillna(MISSING_LABEL).values
        out[col] = generate_categorical(vals, w, cfg.cat_gen_method, rng)

    synth_scaled = np.zeros(len(NUMERICAL_COLS))
    for j in range(len(NUMERICAL_COLS)):
        if cfg.num_gen_method == "interpolation":
            nj = rng.choice(nbrs)
            t = rng.random()
            synth_scaled[j] = X_num[row_idx, j] + t * (X_num[nj, j] - X_num[row_idx, j])
        elif cfg.num_gen_method == "weighted_mean":
            synth_scaled[j] = float(np.dot(w, X_num[nbrs, j]))
        else:
            raise ValueError(cfg.num_gen_method)

    synth_num = num_scaler.inverse_transform(synth_scaled.reshape(1, -1)).flatten()
    for j, col in enumerate(NUMERICAL_COLS):
        out[col] = float(np.clip(synth_num[j], num_p01[col], num_p99[col]))

    tvals = df.loc[nbrs, TARGET_COL].astype(str).values
    out[TARGET_COL] = int(
        float(generate_categorical(tvals, w, cfg.target_gen_method, rng))
    )
    out[ID_COLS[0]] = df.loc[row_idx, ID_COLS[0]]
    return out


def prepare_context(df: pd.DataFrame, cfg: ExperimentConfig) -> dict:
    """Precompute suppression mask and feature matrices (reused across configs with same scaler)."""
    suppressed, suppression_summary = identify_suppressed(df, cfg.k_anonymity)
    pool_idx = np.where(~suppressed.values)[0]
    synth_idx = np.where(suppressed.values)[0]
    if len(pool_idx) == 0:
        raise ValueError(
            "No non-suppressed neighbour pool — relax generalization or lower k."
        )
    cat_encoders, cat_domains, num_medians, num_scaler, num_p01, num_p99 = (
        fit_preprocessors(df, cfg)
    )
    X_cat = encode_cats(df, cat_encoders)
    X_num = num_scaler.transform(df[NUMERICAL_COLS].fillna(num_medians).values)
    num_raw = df[NUMERICAL_COLS].fillna(num_medians).values.astype(float)
    num_ranges = np.ptp(num_raw, axis=0)
    return {
        "suppressed": suppressed,
        "suppression_summary": suppression_summary,
        "pool_idx": pool_idx,
        "synth_idx": synth_idx,
        "cat_encoders": cat_encoders,
        "cat_domains": cat_domains,
        "num_medians": num_medians,
        "num_scaler": num_scaler,
        "num_p01": num_p01,
        "num_p99": num_p99,
        "X_cat": X_cat,
        "X_num": X_num,
        "num_raw": num_raw,
        "num_ranges": num_ranges,
    }


def run_pipeline(
    df: pd.DataFrame,
    cfg: ExperimentConfig,
    ctx: dict | None = None,
    neighbor_cache: dict | None = None,
    f1_baseline: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Full Story 02 pipeline: k-anon → replace suppressed → validate metrics."""
    tracemalloc.start()
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.random_state)

    if ctx is None:
        ctx = prepare_context(df, cfg)

    suppressed = ctx["suppressed"]
    pool_idx = ctx["pool_idx"]
    synth_idx = ctx["synth_idx"]
    cat_encoders = ctx["cat_encoders"]
    cat_domains = ctx["cat_domains"]
    num_medians = ctx["num_medians"]
    num_scaler = ctx["num_scaler"]
    num_p01 = ctx["num_p01"]
    num_p99 = ctx["num_p99"]
    X_cat = ctx["X_cat"]
    X_num = ctx["X_num"]
    suppression_summary = ctx["suppression_summary"]

    df_out = df.copy()
    int_cols = {c for c in NUMERICAL_COLS if pd.api.types.is_integer_dtype(df[c])}
    rows_out: list[tuple[int, dict]] = []
    for i in synth_idx:
        rows_out.append(
            (
                int(i),
                synthesize_row(
                    i,
                    df,
                    X_num,
                    X_cat,
                    num_scaler,
                    num_p01,
                    num_p99,
                    cfg,
                    rng,
                    neighbor_cache,
                ),
            )
        )
    idx_list = [i for i, _ in rows_out]
    for col in QI_COLS + [TARGET_COL]:
        vals = [row[col] for _, row in rows_out]
        if col in int_cols:
            vals = [int(round(v)) for v in vals]
        df_out.loc[idx_list, col] = vals

    runtime_sec = time.perf_counter() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    metrics = compute_all_metrics(
        df,
        df_out,
        suppressed,
        cat_domains,
        cat_encoders,
        num_medians,
        cfg,
        f1_baseline,
    )
    metrics["runtime_sec"] = round(runtime_sec, 3)
    metrics["peak_memory_mb"] = round(peak_mem / 1024 / 1024, 2)
    metrics["suppression"] = suppression_summary.iloc[0].to_dict()
    metrics["n_suppressed_replaced"] = int(len(synth_idx))
    metrics["n_kept_unchanged"] = int(len(pool_idx))
    metrics["suppression_recovery_rate"] = round(
        len(synth_idx) / max(len(synth_idx), 1), 4
    )
    metrics["config"] = asdict(cfg)
    return df_out, metrics


def tvd(actual: pd.Series, synthetic: pd.Series) -> float:
    a = actual.astype(str).value_counts(normalize=True)
    s = synthetic.astype(str).value_counts(normalize=True)
    keys = set(a.index) | set(s.index)
    return 0.5 * sum(abs(s.get(k, 0) - a.get(k, 0)) for k in keys)


def cramers_v(x, y) -> float:
    tbl = pd.crosstab(x, y)
    chi2 = chi2_contingency(tbl)[0]
    n = tbl.sum().sum()
    r, k = tbl.shape
    return float(np.sqrt(chi2 / (n * (min(r - 1, k - 1) + 1e-8))))


def compute_all_metrics(
    df_actual: pd.DataFrame,
    df_out: pd.DataFrame,
    suppressed: pd.Series,
    cat_domains: dict,
    cat_encoders: dict,
    num_medians: pd.Series,
    cfg: ExperimentConfig,
    f1_baseline: float | None = None,
) -> dict:
    # Only evaluate on replaced rows + compare full distributions
    cat_rows = []
    for col in CATEGORICAL_COLS:
        act, syn = df_actual[col], df_out[col]
        tv = tvd(act, syn)
        cat_rows.append(
            {
                "column": col,
                "TVD": round(tv, 4),
                "pass_tvd": tv < TVD_THRESHOLD,
                "invalid_categories": len(set(syn.astype(str)) - set(cat_domains[col])),
                "cardinality_actual": act.nunique(),
                "cardinality_synthetic": syn.nunique(),
                "high_cardinality": act.nunique() / len(df_actual)
                > HIGH_CARDINALITY_THRESHOLD,
            }
        )
    cat_df = pd.DataFrame(cat_rows)
    tvd_pass_rate = cat_df["pass_tvd"].mean()

    num_rows = []
    for col in NUMERICAL_COLS:
        act, syn = df_actual[col].astype(float), df_out[col].astype(float)
        ks_stat, ks_p = ks_2samp(act, syn)
        num_rows.append(
            {
                "column": col,
                "KS_statistic": round(ks_stat, 4),
                "KS_pvalue": round(ks_p, 4),
                "pass_ks": ks_stat < KS_THRESHOLD,
                "mean_diff_pct": round(
                    100 * abs(syn.mean() - act.mean()) / act.mean(), 2
                ),
            }
        )
    num_df = pd.DataFrame(num_rows)
    ks_pass_rate = num_df["pass_ks"].mean()

    # Replaced-row fidelity
    replaced = df_out.loc[suppressed]
    replaced_act = df_actual.loc[suppressed]
    mean_replaced_pct_change = np.mean(
        100
        * np.abs(replaced[NUMERICAL_COLS].values - replaced_act[NUMERICAL_COLS].values)
        / replaced_act[NUMERICAL_COLS].values
    )

    corr_cols = NUMERICAL_COLS + [TARGET_COL]
    corr_diff = (df_actual[corr_cols].corr() - df_out[corr_cols].corr()).abs()
    triu = corr_diff.values[np.triu_indices_from(corr_diff.values, 1)]

    cramers_rows = []
    for col in CATEGORICAL_COLS:
        v_act = cramers_v(df_actual[col].astype(str), df_actual[TARGET_COL])
        v_syn = cramers_v(df_out[col].astype(str), df_out[TARGET_COL])
        cramers_rows.append(
            {
                "column": col,
                "cramers_v_actual": round(v_act, 4),
                "cramers_v_synthetic": round(v_syn, 4),
                "drift": round(abs(v_act - v_syn), 4),
            }
        )

    # Utility
    def encode_model(frame):
        out = frame[QI_COLS].copy()
        out[NUMERICAL_COLS] = out[NUMERICAL_COLS].fillna(num_medians)
        for c in CATEGORICAL_COLS:
            out[c] = cat_encoders[c].transform(out[c].astype(str).fillna(MISSING_LABEL))
        return out

    train_idx, test_idx = train_test_split(
        df_actual.index, test_size=0.2, random_state=cfg.random_state
    )
    if f1_baseline is None:
        rf_base = _utility_model(cfg.random_state)
        rf_base.fit(
            encode_model(df_actual.loc[train_idx]), df_actual.loc[train_idx, TARGET_COL]
        )
        X_te = encode_model(df_actual.loc[test_idx])
        f1_base = f1_score(df_actual.loc[test_idx, TARGET_COL], rf_base.predict(X_te))
    else:
        f1_base = f1_baseline
        X_te = encode_model(df_actual.loc[test_idx])
    rf_syn = _utility_model(cfg.random_state)
    rf_syn.fit(encode_model(df_out.loc[train_idx]), df_out.loc[train_idx, TARGET_COL])
    f1_syn = f1_score(
        df_actual.loc[test_idx, TARGET_COL],
        rf_syn.predict(encode_model(df_out.loc[test_idx])),
    )

    feature_cols = [c for c in df_actual.columns if c not in ID_COLS]
    replaced_idx = suppressed[suppressed].index
    n_exact_replaced = len(
        df_out.loc[replaced_idx, feature_cols].merge(
            df_actual[feature_cols], how="inner"
        )
    )
    exact_match_rate = n_exact_replaced / max(len(replaced_idx), 1)

    scorecard = pd.DataFrame(
        [
            {
                "area": "Quality-Categorical",
                "metric": "tvd_pass_rate>=0.85",
                "value": round(tvd_pass_rate, 4),
                "pass": tvd_pass_rate >= PASS_RATE_TARGET,
            },
            {
                "area": "Quality-Numerical",
                "metric": "ks_pass_rate>=0.85",
                "value": round(ks_pass_rate, 4),
                "pass": ks_pass_rate >= PASS_RATE_TARGET,
            },
            {
                "area": "Quality-Numerical",
                "metric": "mean_KS<0.10",
                "value": round(num_df["KS_statistic"].mean(), 4),
                "pass": num_df["KS_statistic"].mean() < KS_THRESHOLD,
            },
            {
                "area": "Relationships",
                "metric": "mean_corr_drift<0.05",
                "value": round(float(triu.mean()), 4),
                "pass": triu.mean() < 0.05,
            },
            {
                "area": "Utility",
                "metric": "F1>=80% baseline",
                "value": round(f1_syn, 4),
                "pass": f1_syn >= 0.8 * f1_base,
            },
            {
                "area": "Privacy",
                "metric": "replaced_exact_match_rate<0.001",
                "value": round(exact_match_rate, 6),
                "pass": exact_match_rate < 0.001,
            },
            {
                "area": "Suppression",
                "metric": "recovery_rate",
                "value": 1.0,
                "pass": suppressed.sum() > 0,
            },
        ]
    )

    return {
        "categorical_metrics": cat_df,
        "numerical_metrics": num_df,
        "cramers_v": pd.DataFrame(cramers_rows),
        "scorecard": scorecard,
        "overall_pass": bool(scorecard["pass"].all()),
        "tvd_pass_rate": round(tvd_pass_rate, 4),
        "ks_pass_rate": round(ks_pass_rate, 4),
        "mean_tvd": round(cat_df["TVD"].mean(), 4),
        "mean_ks": round(num_df["KS_statistic"].mean(), 4),
        "mean_corr_drift": round(float(triu.mean()), 6),
        "f1_baseline": round(f1_base, 4),
        "f1_synthetic": round(f1_syn, 4),
        "exact_match_rate": round(exact_match_rate, 6),
        "mean_replaced_num_pct_change": round(float(mean_replaced_pct_change), 2),
    }


def save_iteration_outputs(
    folder: Path, df_out: pd.DataFrame, metrics: dict, cfg: ExperimentConfig
):
    folder.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(folder / "anonymized_dataset.csv", index=False)
    metrics["categorical_metrics"].to_csv(folder / "category_metrics.csv", index=False)
    metrics["numerical_metrics"].to_csv(folder / "numeric_metrics.csv", index=False)
    metrics["cramers_v"].to_csv(folder / "relationship_metrics.csv", index=False)
    metrics["scorecard"].to_csv(folder / "scorecard.csv", index=False)

    summary = {k: v for k, v in metrics.items() if not isinstance(v, pd.DataFrame)}
    with open(folder / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(folder / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)


def _distance_profile(cfg: ExperimentConfig) -> str:
    if cfg.num_weight == cfg.cat_weight == 1.0:
        return "balanced"
    if cfg.num_weight > cfg.cat_weight:
        return "num_heavy"
    return "cat_heavy"


def _ranking_row(cfg: ExperimentConfig, metrics: dict) -> dict:
    return {
        "folder": cfg.folder,
        "name": cfg.name,
        "k_anonymity": cfg.k_anonymity,
        "k_neighbors": cfg.k_neighbors,
        "cat_gen_method": cfg.cat_gen_method,
        "num_gen_method": cfg.num_gen_method,
        "target_gen_method": cfg.target_gen_method,
        "scaler_method": cfg.scaler_method,
        "num_weight": cfg.num_weight,
        "cat_weight": cfg.cat_weight,
        "distance_profile": _distance_profile(cfg),
        "distance_mode": cfg.distance_mode,
        "num_distance_metric": cfg.num_distance_metric,
        "cat_distance_metric": cfg.cat_distance_metric,
        "minkowski_p": cfg.minkowski_p,
        "random_state": cfg.random_state,
        "runtime_sec": metrics.get("runtime_sec"),
        "peak_memory_mb": metrics.get("peak_memory_mb"),
        "n_suppressed": metrics.get(
            "n_suppressed_replaced", metrics.get("n_suppressed")
        ),
        "tvd_pass_rate": metrics.get("tvd_pass_rate"),
        "ks_pass_rate": metrics.get("ks_pass_rate"),
        "mean_tvd": metrics.get("mean_tvd"),
        "mean_ks": metrics.get("mean_ks"),
        "mean_corr_drift": metrics.get("mean_corr_drift"),
        "f1_baseline": metrics.get("f1_baseline"),
        "f1_synthetic": metrics.get("f1_synthetic"),
        "exact_match_rate": metrics.get("exact_match_rate"),
        "overall_pass": metrics.get("overall_pass"),
    }


def _finalize_ranking(ranking_rows: list[dict]) -> pd.DataFrame:
    ranking = pd.DataFrame(ranking_rows)
    ranking["composite_score"] = (
        ranking["tvd_pass_rate"]
        + ranking["ks_pass_rate"]
        + ranking["f1_synthetic"] / ranking["f1_baseline"].clip(lower=0.01)
        - ranking["mean_ks"]
        - ranking["runtime_sec"] / 100
    )
    ranking = ranking.sort_values("composite_score", ascending=False).reset_index(
        drop=True
    )
    ranking["rank"] = ranking.index + 1
    return ranking


def run_experiment_grid(
    df: pd.DataFrame | None = None,
    *,
    grid: list[ExperimentConfig] | None = None,
    save_outputs: bool = True,
    iterations_dir: Path | None = None,
    results_path: Path | None = None,
    checkpoint_every: int = 25,
) -> pd.DataFrame:
    """Run a grid of experiments; optionally skip writing per-run iteration files."""
    if df is None:
        df = load_dataset()
    grid = grid or EXPERIMENT_GRID
    out_root = iterations_dir or ITERATIONS_DIR
    results_path = results_path or (RESULTS_DIR / "experiment_ranking.csv")

    if save_outputs:
        out_root.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    completed: set[str] = set()
    ranking_rows: list[dict] = []
    if results_path.exists():
        prev = pd.read_csv(results_path)
        if "folder" in prev.columns:
            completed = set(prev["folder"].astype(str))
            ranking_rows = prev.to_dict("records")
            print(f"Resuming: {len(completed)} configs already in {results_path.name}")

    total = len(grid)
    print(f"Total experiments: {total}")

    prep = prepare_context(df, ExperimentConfig("prep", "prep"))
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)

    def _encode(frame):
        out = frame[QI_COLS].copy()
        out[NUMERICAL_COLS] = out[NUMERICAL_COLS].fillna(prep["num_medians"])
        for c in CATEGORICAL_COLS:
            out[c] = prep["cat_encoders"][c].transform(
                out[c].astype(str).fillna(MISSING_LABEL)
            )
        return out

    rf_base = _utility_model(42)
    rf_base.fit(_encode(df.loc[train_idx]), df.loc[train_idx, TARGET_COL])
    f1_baseline = float(
        f1_score(
            df.loc[test_idx, TARGET_COL], rf_base.predict(_encode(df.loc[test_idx]))
        )
    )
    print(f"F1 baseline (precomputed): {f1_baseline:.4f}")

    scaler_contexts: dict[str, dict] = {}
    neighbor_caches: dict[tuple, dict] = {}
    new_runs = 0

    for i, cfg in enumerate(grid, start=1):
        if cfg.folder in completed:
            if i % 50 == 0 or i == total:
                print(f"[{i}/{total}] {cfg.folder} — skip (in ranking)")
            continue

        if cfg.scaler_method not in scaler_contexts:
            prep_cfg = ExperimentConfig("prep", "prep", scaler_method=cfg.scaler_method)
            scaler_contexts[cfg.scaler_method] = prepare_context(df, prep_cfg)

        cache_key = neighbor_cache_key(cfg)
        if cache_key not in neighbor_caches:
            print(
                f"  Precomputing neighbours: scaler={cfg.scaler_method} "
                f"mode={cfg.distance_mode} num={cfg.num_distance_metric} "
                f"num_w={cfg.num_weight} cat_w={cfg.cat_weight} ..."
            )
            neighbor_caches[cache_key] = precompute_neighbor_cache(
                scaler_contexts[cfg.scaler_method],
                cfg,
            )

        if save_outputs:
            summary_path = out_root / cfg.folder / "summary.json"
            if summary_path.exists():
                print(f"[{i}/{total}] {cfg.folder} — skip (already done)")
                with open(summary_path) as f:
                    prev = json.load(f)
                ranking_rows.append(_ranking_row(cfg, prev))
                completed.add(cfg.folder)
                continue

        print(f"[{i}/{total}] {cfg.folder}")
        df_out, metrics = run_pipeline(
            df,
            cfg,
            scaler_contexts[cfg.scaler_method],
            neighbor_caches[cache_key],
            f1_baseline,
        )
        if save_outputs:
            save_iteration_outputs(out_root / cfg.folder, df_out, metrics, cfg)
        ranking_rows.append(_ranking_row(cfg, metrics))
        completed.add(cfg.folder)
        new_runs += 1

        if i % 10 == 0 or i == total:
            print(
                f"  [{i}/{total}] TVD={metrics['tvd_pass_rate']:.0%} "
                f"KS={metrics['ks_pass_rate']:.0%} pass={metrics['overall_pass']} "
                f"runtime={metrics['runtime_sec']}s"
            )
        if new_runs and new_runs % checkpoint_every == 0:
            _finalize_ranking(ranking_rows).to_csv(results_path, index=False)
            print(f"  Checkpoint saved → {results_path}")

    ranking = _finalize_ranking(ranking_rows)
    ranking.to_csv(results_path, index=False)
    return ranking


def run_all_experiments(df: pd.DataFrame | None = None) -> pd.DataFrame:
    return run_experiment_grid(df, grid=EXPERIMENT_GRID, save_outputs=True)
