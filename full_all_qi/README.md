# full_all_qi

Self-contained **All-QI** KNN anonymization pipeline (simplified from `all_qi_pipeline/`).

- Single notebook: `No_model_production_pipeline.ipynb`
- Set `DATA_FILE`, parameter lists, and batch controls in the notebook — schema is inferred automatically
- Every quasi-identifier (including binary columns like `churn`) is used in distance and synthesis — no separate target handling
- No ML model in metrics (distribution, relationship, and privacy checks only)
- Output: `experiment_ranking.csv` only (no `output/`, `results/`, or extra exports)

Run all cells from this folder.
