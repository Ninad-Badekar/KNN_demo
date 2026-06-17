#!/usr/bin/env python3
"""Run reference-grid experiments (parameter_reference.csv) → results/experiment_ranking.csv."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from knn_lib import (  # noqa: E402
    REFERENCE_EXPERIMENT_GRID,
    load_dataset,
    run_experiment_grid,
)

if __name__ == "__main__":
    print(f"Reference grid: {len(REFERENCE_EXPERIMENT_GRID)} experiments")
    ranking = run_experiment_grid(
        load_dataset(),
        grid=REFERENCE_EXPERIMENT_GRID,
        save_outputs=False,
        checkpoint_every=10,
    )
    print(f"\nDone. {len(ranking)} configs ranked.")
    print(f"Top: {ranking.iloc[0]['folder']}")
    print(f"Passing all checks: {ranking['overall_pass'].sum()}/{len(ranking)}")
