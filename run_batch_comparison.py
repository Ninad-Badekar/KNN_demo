"""Run batch experiments for No_target and all_qi_pipeline notebooks."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / "No_target" / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

BATCH_LIMIT = 10
SKIP_SUBSTRINGS = (
    "display_metrics_report(report_df_actual",
    "report_df_actual = df\n",
)


def run_notebook(pipeline_dir: Path, label: str) -> Path:
    os.chdir(pipeline_dir)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ranking = pipeline_dir / "results" / "experiment_ranking.csv"
    if ranking.exists():
        ranking.unlink()

    nb = json.loads((pipeline_dir / "production_pipeline.ipynb").read_text(encoding="utf-8"))
    globs: dict = {"__name__": "__main__"}

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if any(s in src for s in SKIP_SUBSTRINGS):
            continue
        if "RUN_MODE = " in src and "EXPERIMENT_RANKING_CSV" in src:
            src = src.replace('RUN_MODE = "single"', 'RUN_MODE = "batch"')
            src = src.replace('RUN_MODE = "batch"', 'RUN_MODE = "batch"', 1)
            if "BATCH_LIMIT = None" in src:
                src = src.replace("BATCH_LIMIT = None", f"BATCH_LIMIT = {BATCH_LIMIT}")
            elif "BATCH_LIMIT = " in src:
                import re
                src = re.sub(r"BATCH_LIMIT = \d+", f"BATCH_LIMIT = {BATCH_LIMIT}", src)
        if "PARAMETER_COMBINATIONS_RAW = " in src and "PARAMETER_COMBINATIONS_CSV" not in src:
            src = src.replace(
                'PARAMETER_COMBINATIONS_RAW = PIPELINE_ROOT / "parameter_combinations.csv"\n',
                'PARAMETER_COMBINATIONS_RAW = PIPELINE_ROOT / "parameter_combinations.csv"\n'
                'PARAMETER_COMBINATIONS_CSV = PARAMETER_COMBINATIONS_RAW\n',
            )
        exec(compile(src, f"<{label}>", "exec"), globs)

    if not ranking.exists():
        raise FileNotFoundError(f"{label}: ranking not written to {ranking}")
    return ranking


def main():
    results = {}
    pipelines = [
        ("No_target (Karabo target)", "No_target", False),
        ("all_qi_pipeline (all QI)", "all_qi_pipeline", True),
    ]
    for name, subdir, run in pipelines:
        pipeline_dir = ROOT / subdir
        ranking = pipeline_dir / "results" / "experiment_ranking.csv"
        if not run and ranking.exists():
            import pandas as pd
            df = pd.read_csv(ranking)
            if len(df) >= BATCH_LIMIT:
                print(f"Using existing {name} ranking ({len(df)} rows)")
                results[name] = df
                continue
        print(f"\n{'=' * 72}\nRunning {name} — {BATCH_LIMIT} experiments\n{'=' * 72}")
        t0 = time.perf_counter()
        ranking_path = run_notebook(pipeline_dir, name)
        elapsed = time.perf_counter() - t0
        import pandas as pd

        df = pd.read_csv(ranking_path)
        results[name] = df
        print(f"Done in {elapsed:.1f}s | rows={len(df)} | passing={int(df['overall_pass'].sum())}")

    import pandas as pd

    out_dir = ROOT / "comparison_results"
    out_dir.mkdir(exist_ok=True)

    summary_rows = []
    for name, df in results.items():
        df.to_csv(out_dir / f"{name.split()[0].lower()}_ranking_10.csv", index=False)
        summary_rows.append({
            "pipeline": name,
            "experiments": len(df),
            "passing": int(df["overall_pass"].sum()),
            "best_composite_score": round(float(df["composite_score"].max()), 4),
            "mean_tvd_pass_rate": round(float(df["tvd_pass_rate"].mean()), 4),
            "mean_ks_pass_rate": round(float(df["ks_pass_rate"].mean()), 4),
            "mean_tvd": round(float(df["mean_tvd"].mean()), 4),
            "mean_ks": round(float(df["mean_ks"].mean()), 4),
            "mean_exact_match_rate": round(float(df["exact_match_rate"].mean()), 6),
            "mean_psi": round(float(df["mean_psi"].dropna().mean()), 4) if "mean_psi" in df else None,
        })
        if "mean_corr_drift" in df.columns:
            summary_rows[-1]["mean_relationship_drift"] = round(float(df["mean_corr_drift"].mean()), 6)
        if "mean_pair_drift" in df.columns:
            summary_rows[-1]["mean_pair_drift"] = round(float(df["mean_pair_drift"].mean()), 6)
        if "auc_retention_ratio" in df.columns:
            summary_rows[-1]["mean_auc_retention"] = round(float(df["auc_retention_ratio"].dropna().mean()), 4)
        if "target_rate_drift" in df.columns:
            summary_rows[-1]["mean_target_rate_drift"] = round(float(df["target_rate_drift"].dropna().mean()), 4)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "batch_10_summary.csv", index=False)
    print("\n\nSUMMARY\n", summary.to_string(index=False))
    print(f"\nSaved -> {out_dir}/")
    return results


if __name__ == "__main__":
    main()
