# Gate 3 Recommendation (Bank Churn — Story 02)

**Decision:** Profile-driven / No-Go

## Top configuration (reference grid, weighted_mean only)
- Folder: `201_k15_cat-weighted_mode_num-weighted_mean_tgt-probability_scale-standard_weighted_sum-euclidean-cat-overlap_w-balanced`
- K neighbours: 15
- Cat gen: weighted_mode | Num gen: weighted_mean
- Scaler: standard | Distance: balanced (weighted_sum/euclidean/overlap)
- TVD pass: 50.0% | KS pass: 75.0%
- F1: 0.1652 / 0.2923
- Configs passing all checks: 0/36
