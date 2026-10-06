"""universal model for the demo, only features any shop has"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from cv import evaluate, folds, load_matrices
from common import FEAT, fold_data, to_matrix

ROOT = Path(__file__).resolve().parents[1]
DEMO_FEATS = ["lag_28", "lag_35", "mean_7", "mean_28", "mean_56", "std_28", "dow_mean_4", "sale_share_28",
              "sale_share_56", "size_56", "wday", "weekend", "mday", "month", "age_days", "is_new"]
PARAMS = dict(objective="tweedie", tweedie_variance_power=1.1, learning_rate=0.08, num_leaves=63,
              min_data_in_leaf=500, feature_fraction=0.9, bagging_fraction=0.7, bagging_freq=1,
              lambda_l2=1.0, n_jobs=12, verbose=-1, seed=0)
ROUNDS = 300
SAMPLE = 4_000_000


def fit(t, rows, seed=0):
    idx = np.random.default_rng(seed).choice(np.flatnonzero(rows), min(SAMPLE, rows.sum()), replace=False)
    y = t["sales"].values[idx].astype(np.float32)
    ds = lgb.Dataset(t[DEMO_FEATS].values[idx].astype(np.float32), np.minimum(y, np.quantile(y, 0.999)))
    return lgb.train(PARAMS, ds, ROUNDS)


def main():
    M = load_matrices()
    t = pd.read_parquet(FEAT, columns=["d", "sales"] + DEMO_FEATS)
    c = folds(M["days"])[-1]
    train, val = fold_data(t, M, c)
    m = fit(t, train)
    p = m.predict(t[DEMO_FEATS].values[val].astype(np.float32))
    res, _, _ = evaluate(to_matrix(p, None, M), M, c)
    print(f"demo model on fold 3: WRMSSE {res['WRMSSE']:.4f}, bias {res['bias']:+.3f} "
          f"(full LightGBM on fold 3: 0.762, best rule: 0.782)")
    imp = pd.Series(m.feature_importance("gain"), index=DEMO_FEATS)
    print((imp / imp.sum() * 100).round(1).sort_values(ascending=False).to_string())

    # refit on everything up to the last dev day for the demo
    last = t["d"].values <= M["days"][-1]
    st_all, _ = fold_data(t, M, len(M["days"]))
    final = fit(t, st_all & last)
    final.save_model(str(ROOT / "data" / "demo_lgbm.txt"))
    import m2cgen
    js = m2cgen.export_to_javascript(final, function_name="demandScore")
    out = ROOT / "docs" / "model.js"
    out.parent.mkdir(exist_ok=True)
    header = (f"// LightGBM (Tweedie) trained on M5 dev data, features: {', '.join(DEMO_FEATS)}\n"
              f"// demandScore returns the raw score; expected demand = Math.exp(score)\n"
              f"const DEMO_FEATURES = {DEMO_FEATS};\n")
    out.write_text(header + js, encoding="utf-8")
    print(f"exported {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
