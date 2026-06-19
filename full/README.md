# full

Self-contained **No Target** KNN anonymization pipeline (simplified from `No_target/`).

- Single notebook: `production_pipeline.ipynb`
- Set `DATA_FILE`, parameter lists, and batch controls in the notebook — schema is inferred automatically
- Target column (e.g. `churn`) is synthesized separately from feature QIs (Karabo-style)
- Output: `experiment_ranking.csv` only (no `output/`, `results/`, or extra exports)

Run all cells from this folder.
