# CV leaderboard (3 folds x 28 days, WRMSSE, lower is better)

| Model | Fold 1 | Fold 2 | Fold 3 | Mean | Bias |
|---|---|---|---|---|---|
| LightGBM 0.5 + N-HiTS 0.5 | 0.791 | 0.765 | 0.751 | **0.769** | |
| LightGBM 0.5 + CatBoost 0.3 + GLM 0.2 | 0.794 | 0.775 | 0.758 | 0.775 | -1.8% |
| LightGBM (Tweedie) | 0.797 | 0.780 | 0.762 | 0.780 | -1.4% |
| N-HiTS (GPU) | 0.806 | 0.772 | 0.762 | 0.780 | -5...-9% |
| CatBoost CPU (4M rows) | 0.804 | 0.782 | 0.765 | 0.784 | -1.3% |
| LightGBM without item_id | 0.804 | 0.783 | 0.768 | 0.785 | -0.8% |
| Poisson GLM with offset | 0.814 | 0.794 | 0.775 | 0.794 | -3.3% |
| Croston SBA (rule) | 0.814 | 0.787 | 0.786 | 0.796 | -5% |
| Mean 28 x weekly profile (rule) | 0.811 | 0.795 | 0.782 | 0.796 | -3% |
| AutoETS (per series) | 0.815 | 0.789 | 0.789 | 0.798 | -1% |
| ADIDA / IMAPA / AutoTheta | | | | 0.807-0.809 | ~-0.7% |
| CrostonOptimized | 0.819 | 0.798 | 0.803 | 0.807 | +1% |
| DeepAR (StudentT) | 0.845 | 0.811 | 0.805 | 0.820 | -14...-19% |
| CatBoost GPU (Tweedie bug) | 1.168 | 1.133 | 1.121 | 1.141 | -43% |
| AutoARIMA | stopped: >2 h without one fold on 30k series | | | | |
