#!/usr/bin/env python3
"""Quick A/B: independent vs donor row synthesis on the bank churn dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # No_target/


def load_pipeline_namespace() -> dict:
  nb = json.loads((ROOT / "production_pipeline.ipynb").read_text())
  ns: dict = {}
  for cell in nb["cells"]:
    if cell["cell_type"] != "code":
      continue
    src = "".join(cell["source"])
    if any(
      token in src
      for token in (
        "PIPELINE_ROOT",
        "def identify_suppressed",
        "def compute_metrics",
        "def synthesize_dataset",
      )
    ):
      exec(src, ns)
  return ns


def run_once(ns: dict, mode: str) -> dict:
  df = pd.read_csv(ns["DATA_PATH"])
  df = df.replace(r"^\s*$", np.nan, regex=True)
  cfg = {
    "k_anonymity": 5,
    "k_neighbors": 15,
    "cat_gen_method": "probability",
    "num_gen_method": "interpolation",
    "scaler_method": "minmax",
    "distance_mode": "gower",
    "num_distance_metric": "euclidean",
    "cat_distance_metric": "hamming",
    "minkowski_p": 3,
    "num_weight": 1.0,
    "cat_weight": 1.0,
    "random_state": 42,
    "row_synthesis_mode": mode,
  }
  suppressed, pool_idx, synth_idx = ns["identify_suppressed"](df, cfg["k_anonymity"])
  prep = ns["fit_preprocessors"](df, cfg["scaler_method"])
  cache = ns["build_neighbor_cache"](cfg, prep, pool_idx, synth_idx)
  df_out = ns["synthesize_dataset"](df, cfg, prep, suppressed, pool_idx, synth_idx, cache)
  metrics = ns["compute_metrics"](df, df_out, suppressed, ns["RELATIONSHIP_COL"])
  return {
    "mode": mode,
    "n_suppressed": int(suppressed.sum()),
    "overall_pass": metrics["overall_pass"],
    "mean_corr_drift": metrics["mean_corr_drift"],
    "tvd_pass_rate": metrics["tvd_pass_rate"],
    "ks_pass_rate": metrics["ks_pass_rate"],
    "exact_match_rate": metrics["exact_match_rate"],
  }


def main():
  ns = load_pipeline_namespace()
  rows = [run_once(ns, mode) for mode in ("independent", "donor")]
  print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
  main()
