**Table 2-s — Bitwidth-aware kernel PPA scaling (post-processing model).**

| Config | Precision preset | Scenario | adc_share/dac_share | Scaling | k | E_kernel (mJ, scaled) |
|--------|------------------|----------|----------------------|---------|---|------------------------|
| baseline_swin | standard | Conservative | 0.7/0.3 | exp | 0.3 | 1.162 |
| baseline_swin | standard | Nominal | 0.5/0.2 | linear | 0.2 | 0.920 |
| baseline_swin | standard | Aggressive | 0.3/0.1 | linear | 0.1 | 0.907 |
| full_opt_swin | balanced | Conservative | 0.7/0.3 | exp | 0.3 | 0.581 |
| full_opt_swin | balanced | Nominal | 0.5/0.2 | linear | 0.2 | 0.830 |
| full_opt_swin | balanced | Aggressive | 0.3/0.1 | linear | 0.1 | 0.856 |
| baseline_vit | standard | Conservative | 0.7/0.3 | exp | 0.3 | N/A |
| baseline_vit | standard | Nominal | 0.5/0.2 | linear | 0.2 | N/A |
| baseline_vit | standard | Aggressive | 0.3/0.1 | linear | 0.1 | N/A |
| optimized_vit | standard | Conservative | 0.7/0.3 | exp | 0.3 | N/A |
| optimized_vit | standard | Nominal | 0.5/0.2 | linear | 0.2 | N/A |
| optimized_vit | standard | Aggressive | 0.3/0.1 | linear | 0.1 | N/A |

*Full sensitivity grid (all adc_share/dac_share/scaling_model/k combinations) is provided in `kernel_ppa_scaled.csv`.*
