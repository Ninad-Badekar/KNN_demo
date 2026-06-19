from pathlib import Path
import pandas as pd

IN_CSV = Path(__file__).resolve().parent / "parameter_combinations.csv"
OUT_CSV = Path(__file__).resolve().parent / "parameter_combinations_filtered.csv"


def strict_keep_mask(df: pd.DataFrame) -> pd.Series:
    gower_mask = (
        (df["distance_mode"] == "gower")
        & (
            (df["cat_distance_metric"] != "hamming")
            | (df["scaler_method"] != "minmax")
            | (df["num_distance_metric"] != "euclidean")
            | (df["num_weight"].astype(float) != 1.0)
            | (df["cat_weight"].astype(float) != 1.0)
        )
    )
    invalid_minkowski_mask = (
        (df["num_distance_metric"] != "minkowski")
        & (df["minkowski_p"].astype(int) != 3)
    )
    return ~(gower_mask | invalid_minkowski_mask)


def main() -> None:
    df = pd.read_csv(IN_CSV)
    keep = strict_keep_mask(df)
    out = df.loc[keep].reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Input rows   : {len(df):,}")
    print(f"Output rows  : {len(out):,}")
    print(f"Removed rows : {len(df) - len(out):,}")
    print(f"Saved to     : {OUT_CSV}")


if __name__ == "__main__":
    main()
