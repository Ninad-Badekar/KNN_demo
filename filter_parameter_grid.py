from pathlib import Path

import pandas as pd

IN_CSV = Path("final_folder/parameter_combinations.csv")
OUT_CSV = Path("final_folder/parameter_combinations_filtered.csv")


def strict_keep_mask(df: pd.DataFrame) -> pd.Series:
    # Canonical constraints for Gower rows
    gower_mask = (df["distance_mode"] == "gower") & (
        (df["cat_distance_metric"] != "hamming")
        | (df["scaler_method"] != "minmax")  # speed tradeoff choice
        | (df["num_distance_metric"] != "euclidean")
        | (df["num_weight"].astype(float) != 1.0)
        | (df["cat_weight"].astype(float) != 1.0)
    )

    # Global minkowski consistency constraint
    invalid_minkowski_mask = (df["num_distance_metric"] != "minkowski") & (
        df["minkowski_p"].astype(int) != 3
    )

    # Keep only rows that violate none of the constraints
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
