# Hyperparameter tuning

Curated search grids for the bank-churn KNN anonymization pipelines, built from batch results in `No_target/` and `all_qi_pipeline/` plus story02 overlap/jaccard winners.

## Recommended production configs (pre-tuning evidence)

| Pipeline | Config | Why |
|----------|--------|-----|
| **No_target** | `k=3, kn=15, weighted_mode, interpolation, target=mode, standard, weighted_sum/euclidean/hamming` | Highest `composite_score` (0.690) in completed experiments |
| **all_qi** | `k=3, kn=15, weighted_mode, interpolation, standard, weighted_sum/euclidean/hamming` | Highest `composite_score` (0.717) in all-QI batch |

Single-row CSVs: `recommended_best_no_target.csv`, `recommended_best_all_qi.csv`.

## Tuning grids

| File | Rows | Purpose |
|------|------|---------|
| `parameter_combinations_no_target.csv` | 20 | Karabo pipeline (separate `churn` target) |
| `parameter_combinations_all_qi.csv` | 18 | All-column QI pipeline |

Grids are **focused** (~20 configs vs 80 in the exploratory grid):

1. **Champion backbone** — empirically best synthesis + distance settings
2. **Target / synthesis refinements** — `target_gen_method`, `num_gen_method`, cat generators
3. **Privacy tradeoffs** — `k_anonymity` 5 and 7
4. **Neighbour count** — `k_neighbors` 10, 15, 20
5. **Distance / scaler** — Gower, overlap, manhattan, minmax, robust, weight profiles

Extra columns `tier` and `notes` are documentation only; they are stripped when deploying to a pipeline.

## Quick start

```powershell
cd C:\Users\admin\OneDrive\Documents\GitHub\KNN_demo

# Deploy grid only (copies to pipeline folder, backs up old grid)
.\No_target\.venv\Scripts\python.exe hyperparameter_tuning\run_tuning.py no_target --deploy-only

# Run full tuning batch for No_target (~20 experiments, ~20 min)
.\No_target\.venv\Scripts\python.exe hyperparameter_tuning\run_tuning.py no_target

# Run both pipelines
.\No_target\.venv\Scripts\python.exe hyperparameter_tuning\run_tuning.py both

# Summarize results after a run
.\No_target\.venv\Scripts\python.exe hyperparameter_tuning\summarize_tuning.py
```

Results are copied to `hyperparameter_tuning/results/`. The pipeline's own `results/experiment_ranking.csv` is updated in place.

## Evidence notes

- On the **k=3 / standard / euclidean / hamming** block, synthesis hyperparameters barely changed validation metrics in batch runs — neighbour search and k-anonymity dominate outcomes.
- **`target_gen_method`** was not swept on the winning backbone in the original 80-row grid; the tuning set adds `probability` and `weighted_mode` variants.
- **`k=5`** increases suppression (~9.3k rows) but partial runs showed better AUC retention; included for privacy/utility tradeoff exploration.
- **Gower** uses a different neighbour cache key — worth testing when weighted_sum results plateau.

## Folder layout

```
hyperparameter_tuning/
  parameter_combinations_no_target.csv   # tuning grid (Karabo)
  parameter_combinations_all_qi.csv      # tuning grid (all QI)
  recommended_best_*.csv                 # single best pick per pipeline
  run_tuning.py                          # deploy + run batch
  summarize_tuning.py                    # rank results after tuning
  results/                               # copied rankings + summary
  deployed/                              # archived grids per deploy
```
