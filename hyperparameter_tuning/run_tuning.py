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

for _venv in (ROOT / ".venv", ROOT / "No_target" / ".venv"):
    _python = _venv / "Scripts" / "python.exe"
    if _python.exists():
        PYTHON = _python
        break
else:
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

# Production pipelines used for hyperparameter tuning (not the simplified full/ notebooks).
PIPELINES = {
    "no_target": {
        "dir": ROOT / "No_target",
        "notebook": "production_pipeline.ipynb",
        "result_name": "No_target",
    },
    "all_qi": {
        "dir": ROOT / "all_qi_pipeline",
        "notebook": "all_qi_production_pipeline.ipynb",
        "result_name": "all_qi_pipeline",
    },
}

SKIP_SUBSTRINGS = (
    "display_metrics_report(",
    "plot_validation_charts(",
    "report_df_actual = df",
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


def _patch_config_cell(src: str, batch_limit: int | None) -> str:
    lines = src.split("\n")
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("RUN_MODE"):
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f'{indent}RUN_MODE = "batch"')
        elif batch_limit is not None and stripped.startswith("BATCH_LIMIT"):
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}BATCH_LIMIT = {batch_limit}")
        else:
            out.append(line)
    return "\n".join(out)


def resolve_notebook(pipeline_dir: Path, notebook_name: str) -> Path:
    nb_path = pipeline_dir / notebook_name
    if nb_path.exists():
        return nb_path
    candidates = sorted(pipeline_dir.glob("*production_pipeline.ipynb"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"No notebook found in {pipeline_dir} (expected {notebook_name})")


def run_notebook_batch(pipeline_dir: Path, notebook_name: str, batch_limit: int | None = None) -> Path:
    os.chdir(pipeline_dir)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    nb_path = resolve_notebook(pipeline_dir, notebook_name)
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    globs: dict = {"__name__": "__main__"}

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if any(s in src for s in SKIP_SUBSTRINGS):
            continue
        if "RUN_MODE" in src or "BATCH_LIMIT" in src:
            src = _patch_config_cell(src, batch_limit)
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
        help="Which pipeline to tune (No_target or all_qi_pipeline)",
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
        targets.append(("no_target", PIPELINES["no_target"]))
    if args.pipeline in ("all_qi", "both"):
        targets.append(("all_qi", PIPELINES["all_qi"]))

    for name, cfg in targets:
        pipeline_dir = cfg["dir"]
        if not pipeline_dir.exists():
            raise FileNotFoundError(f"Pipeline folder not found: {pipeline_dir}")

        dest = deploy_grid(pipeline_dir, name, backup=not args.no_backup)
        n_rows = sum(1 for _ in dest.open(encoding="utf-8")) - 1
        print(f"Deployed {n_rows} configs -> {dest}")
        if args.deploy_only:
            continue
        print(f"Running {name} tuning batch ({pipeline_dir.name})...")
        t0 = time.perf_counter()
        ranking = run_notebook_batch(pipeline_dir, cfg["notebook"], batch_limit=args.limit)
        elapsed = time.perf_counter() - t0
        print(f"Done in {elapsed:.1f}s | ranking -> {ranking}")


if __name__ == "__main__":
    main()
