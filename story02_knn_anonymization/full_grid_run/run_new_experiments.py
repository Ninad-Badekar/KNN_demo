#!/usr/bin/env python3
"""Run reference-grid experiments (parameter_reference.csv) — metrics only."""
import sys
from pathlib import Path

FULL_GRID_ROOT = Path(__file__).resolve().parent
STORY_ROOT = FULL_GRID_ROOT.parent
sys.path.insert(0, str(STORY_ROOT))

from knn_lib import (  # noqa: E402
    REFERENCE_EXPERIMENT_GRID,
    RESULTS_DIR,
    load_dataset,
    run_experiment_grid,
)

RESULTS_PATH = RESULTS_DIR / "experiment_ranking.csv"


if __name__ == "__main__":
    print(f"Reference grid: {len(REFERENCE_EXPERIMENT_GRID)} experiments")
    print(f"Output: {RESULTS_PATH}")
    print("Metrics only — no files written under iterations/\n")

    ranking = run_experiment_grid(
        load_dataset(),
        grid=REFERENCE_EXPERIMENT_GRID,
        save_outputs=False,
        results_path=RESULTS_PATH,
        checkpoint_every=10,
    )
    print(f"\nDone. {len(ranking)} rows ranked.")
    print(f"Top: {ranking.iloc[0]['folder']}")
    print(f"Passing all checks: {ranking['overall_pass'].sum()}/{len(ranking)}")
