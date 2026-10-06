"""why CatBoost under-predicts on GPU"""
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from cv import H, evaluate, folds, load_matrices
from features import feature_columns
from common import CAT_COLS, FEAT, fold_data, to_matrix

M = load_matrices()
t = pd.read_parquet(FEAT)
c = folds(M["days"])[-1]
train, val = fold_data(t, M, c)
d = t["d"].values
inner_cut = M["days"][c - 1 - H]
rng = np.random.default_rng(0)
fit_idx = rng.choice(np.flatnonzero(train & (d <= inner_cut)), 2_000_000, replace=False)
stop_idx = rng.choice(np.flatnonzero(train & (d > inner_cut)), 300_000, replace=False)
feats = feature_columns(t)
y = t["sales"].values.astype(np.float32)
cap = np.quantile(y[train], 0.999)


def X_of(idx):
    X = t.iloc[idx][feats].copy() if not isinstance(idx, np.ndarray) or idx.dtype != bool else t.loc[idx, feats].copy()
    for cc in CAT_COLS:
        X[cc] = X[cc].cat.codes.astype(np.int32)
    return X


Xf, Xs, Xv = X_of(fit_idx), X_of(stop_idx), X_of(val)
pf = Pool(Xf, np.minimum(y[fit_idx], cap), cat_features=CAT_COLS)
ps = Pool(Xs, np.minimum(y[stop_idx], cap), cat_features=CAT_COLS)
pv = Pool(Xv, cat_features=CAT_COLS)
print("mean target fit:", round(float(y[fit_idx].mean()), 3), "| mean y val:", round(float(y[val].mean()), 3), flush=True)

VARIANTS = [
    ("tweedie_gpu", dict(loss_function="Tweedie:variance_power=1.1", task_type="GPU", devices="1")),
    ("tweedie_cpu", dict(loss_function="Tweedie:variance_power=1.1", task_type="CPU", thread_count=6)),
    ("poisson_gpu", dict(loss_function="Poisson", task_type="GPU", devices="1")),
    ("rmse_gpu", dict(loss_function="RMSE", task_type="GPU", devices="1")),
]
for name, kw in VARIANTS:
    t0 = time.time()
    try:
        m = CatBoostRegressor(iterations=1500, learning_rate=0.08, depth=8, border_count=128, l2_leaf_reg=3,
                              od_type="Iter", od_wait=100, random_seed=0, verbose=0, **kw)
        m.fit(pf, eval_set=ps, use_best_model=True)
        p = np.maximum(m.predict(pv), 0)
        res, _, _ = evaluate(to_matrix(p, None, M), M, c)
        print(f"{name:12s} trees {m.get_best_iteration():5d} | mean pred {p.mean():.3f} | WRMSSE {res['WRMSSE']:.4f} "
              f"| bias {res['bias']:+.3f} | {time.time() - t0:.0f}s", flush=True)
    except Exception as e:
        print(f"{name:12s} FAILED: {type(e).__name__}: {e}", flush=True)
