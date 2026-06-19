# No_target — KNN Anonymization (Karabo-style target)

This folder is a **self-contained** KNN anonymization pipeline for the bank churn dataset.

**`churn` is a target column (Karabo-style):** excluded from KNN neighbour distance, synthesized separately from neighbours via `target_gen_method`. Feature QI columns are `country`, `gender`, and the numerical fields.

## Files

| File | Purpose |
|------|---------|
| `production_pipeline.ipynb` | Main batch/single anonymization pipeline |
| `bank_churn_eda.ipynb` | Exploratory analysis |
| `Bank Customer Churn Prediction.csv` | Dataset |
| `parameter_combinations.csv` | Experiment grid |
| `parameter_reference.csv` | Parameter definitions |
| `validate_anonymization.py` | Compare original vs `output/anonymized_dataset.csv` |
| `results/` | Batch experiment rankings |
| `output/` | Single-run / top-config exports |

## Quick start

```bash
cd No_target
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook production_pipeline.ipynb
```

Run with working directory set to `No_target/`.

## Column config

```python
CATEGORICAL_COLS = ["country", "gender"]   # feature QI — used in KNN distance
TARGET_COL = "churn"                       # excluded from distance; sampled from neighbours
NUMERICAL_COLS = ["credit_score", "age", ...]
ROW_SYNTHESIS_MODE = "donor"               # neighbour-coherent feature synthesis
```

## Karabo-style target handling

- `target_gen_method` in the parameter grid controls how `churn` is picked from neighbours (`mode`, `weighted_mode`, `probability`)
- `cat_gen_method` / `num_gen_method` apply only to feature QI columns
- Relationship drift is measured against `TARGET_COL`
