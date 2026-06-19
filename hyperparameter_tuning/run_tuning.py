#!/usr/bin/env python3
"""Deploy tuning grids and run batch experiments for a KNN pipeline."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TUNING_DIR = Path(__file__).resolve().parent
PYTHON = ROOT / "No_target" / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

PIPELINE_COLS_NO_TARGET = [
    "k_anonymity", "k_neighbors", "cat_gen_method", "num_gen_method", "target_gen_method",
    "scaler_method", "distance_mode", "num_distance_metric", "minkowski_p",
    "cat_distance_metric", "num_weight", "cat_weight", "distance_profile",
]
PIPELINE_COLS_ALL_QI = [
    "k_anonymity", "k_neighbors", "cat_gen_method", "num_gen_method",
    "scaler_method", "distance_mode", "num_distance_metric", "minkowski_p",
    "cat_distance_metric", "num_weight", "cat_weight", "distance_profile",
]

SKIP_SUBSTRINGS = (
    "display_metrics_report(report_df_actual",
    "report_df_actual = df\n",
)


def load_grid(pipeline: str) -> tuple[Path, list[str]]:
    if pipeline == "no_target":
        src = TUNING_DIR / "parameter_combinations_no_target.csv"
        cols = PIPELINE_COLS_NO_TARGET
    elif pipeline == "all_qi":
        src = TUNING_DIR / "parameter_combinations_all_qi.csv"
        cols = PIPELINE_COLS_ALL_QI
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")
    if not src.exists():
        raise FileNotFoundError(src)
    return src, cols


def deploy_grid(pipeline_dir: Path, pipeline: str, backup: bool = True) -> Path:
    import pandas as pd

    src, cols = load_grid(pipeline)
    df = pd.read_csv(src)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Grid missing columns: {missing}")

    dest = pipeline_dir / "parameter_combinations.csv"
    if backup and dest.exists():
        stamp = time.strftime("%Y%m%d_%H%M%S")
        shutil.copy2(dest, TUNING_DIR / f"backup_{pipeline_dir.name}_parameter_combinations_{stamp}.csv")

    df[cols].to_csv(dest, index=False)
    archive = TUNING_DIR / "deployed" / pipeline_dir.name
    archive.mkdir(parents=True, exist_ok=True)
    df.to_csv(archive / "parameter_combinations.csv", index=False)
    return dest


def run_notebook_batch(pipeline_dir: Path, batch_limit: int | None = None) -> Path:
    os.chdir(pipeline_dir)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    nb_path = pipeline_dir / "production_pipeline.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    globs: dict = {"__name__": "__main__"}

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if any(s in src for s in SKIP_SUBSTRINGS):
            continue
        if "RUN_MODE = " in src and "RUN_MODE =" in src.split("\n")[0:5]:
            src = src.replace('RUN_MODE = "single"', 'RUN_MODE = "batch"')
            src = src.replace("RUN_MODE = 'single'", "RUN_MODE = 'batch'")
        if batch_limit is not None and "BATCH_LIMIT = " in src:
            for line in src.split("\n"):
                if line.strip().startswith("BATCH_LIMIT"):
                    src = src.replace(line, f"BATCH_LIMIT = {batch_limit}")
                    break
        exec(compile(src, f"{nb_path.name}:cell", "exec"), globs)

    ranking = pipeline_dir / "results" / "experiment_ranking.csv"
    if ranking.exists():
        out = TUNING_DIR / "results" / f"{pipeline_dir.name}_experiment_ranking.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ranking, out)
    return ranking


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning grids")
    parser.add_argument(
        "pipeline",
        choices=["no_target", "all_qi", "both"],
        help="Which pipeline to tune",
    )
    parser.add_argument(
        "--deploy-only",
        action="store_true",
        help="Copy grid to pipeline folder without running experiments",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override BATCH_LIMIT (default: run full tuning grid)",
    )
    parser.add_argument("--no-backup", action="store_true", help="Skip backup of existing parameter_combinations.csv")
    args = parser.parse_args()

    targets = []
    if args.pipeline in ("no_target", "both"):
        targets.append(("no_target", ROOT / "No_target"))
    if args.pipeline in ("all_qi", "both"):
        targets.append(("all_qi", ROOT / "all_qi_pipeline"))

    for name, pipeline_dir in targets:
        dest = deploy_grid(pipeline_dir, name, backup=not args.no_backup)
        n_rows = sum(1 for _ in dest.open(encoding="utf-8")) - 1
        print(f"Deployed {n_rows} configs -> {dest}")
        if args.deploy_only:
            continue
        print(f"Running {name} tuning batch...")
        t0 = time.perf_counter()
        ranking = run_notebook_batch(pipeline_dir, batch_limit=args.limit)
        elapsed = time.perf_counter() - t0
        print(f"Done in {elapsed:.1f}s | ranking -> {ranking}")


if __name__ == "__main__":
    main()
