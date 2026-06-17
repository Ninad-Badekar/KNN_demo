# Story 02 — KNN Anonymization

Mixed-variable KNN anonymization on **Bank Customer Churn Prediction.csv**.

## Quick start

| Step | File | Purpose |
|------|------|---------|
| 1 | `00_overview.ipynb` | Dataset profile & k-anonymity analysis |
| 2 | `01_run_experiments.ipynb` | Run reference-grid experiments |
| 3 | `02_compare_results.ipynb` | Rank runs & Gate 3 recommendation |
| 4 | **`production_pipeline.ipynb`** | **Standalone production run → `output/`** |

```bash
python run_all.py                              # metrics only → results/experiment_ranking.csv
python full_grid_run/run_new_experiments.py    # same grid, same output (resume/checkpoint)
```

## Core files

| Path | Description |
|------|-------------|
| `knn_lib.py` | Experiment grid library (notebooks 00–02, runners) |
| `run_all.py` | CLI for reference grid (metrics only) |
| `results/parameter_reference.csv` | Experiment parameter axes & defaults |
| `results/experiment_ranking.csv` | Ranked metrics from completed runs |
| `results/gate3_recommendation.md` | Top config summary & Gate 3 decision |
| `full_grid_run/run_new_experiments.py` | Same grid runner with resume checkpoints |
| `production_pipeline.ipynb` | Standalone production notebook → `output/` |
| `KNN_Anonymization_Project_Guide.docx` | Full beginner's guide to the project |

## Outputs

| Path | Contents |
|------|----------|
| `results/experiment_ranking.csv` | All experiment metrics & rankings |
| `output/` | Production exports (`anonymized_dataset.csv`, metrics, `config.json`) |

## Parameter axes

See `results/parameter_reference.csv` for tunable values (k=15|25, scalers, distance modes, `num_gen_method=weighted_mean`, cat distance metrics, etc.).
