"""LightGBM with Tweedie loss"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import CAT_COLS, ROOT, fold_data, save_preds, to_matrix
from cv import H, evaluate, folds
from features import feature_columns

PARAMS = dict(objective="tweedie", tweedie_variance_power=1.1, learning_rate=0.05, num_leaves=255,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1,
              lambda_l2=1.0, cat_smooth=20, max_bin=255, n_jobs=12, verbose=-1, seed=0)


def fit(t, train, inner_cut, feats, cats, rounds=3000):
    y = t["sales"].values.astype(np.float32)
    d = t["d"].values
    fit_rows, stop_rows = train & (d <= inner_cut), train & (d > inner_cut)
    cap = np.quantile(y[train], 0.999)
    dtr = lgb.Dataset(t.loc[fit_rows, feats], np.minimum(y[fit_rows], cap), categorical_feature=cats)
    dst = lgb.Dataset(t.loc[stop_rows, feats], np.minimum(y[stop_rows], cap), reference=dtr)
    return lgb.train(PARAMS, dtr, rounds, valid_sets=[dst],
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])


def run(t, M, drop=(), name="lgbm"):
    feats = [c for c in feature_columns(t) if c not in drop]
    cats = [c for c in CAT_COLS if c not in drop]
    rows, imps = [], []
    for k, c in enumerate(folds(M["days"]), 1):
        t0 = time.time()
        train, val = fold_data(t, M, c)
        model = fit(t, train, M["days"][c - 1 - H], feats, cats)
        pred = to_matrix(model.predict(t.loc[val, feats], num_iteration=model.best_iteration), None, M)
        save_preds(name, k, pred)
        model.save_model(str(ROOT / "data" / f"{name}_fold{k}.txt"))
        res, _, _ = evaluate(pred, M, c)
        rows.append({"fold": k, "model": name, **res, "trees": model.best_iteration, "sec": round(time.time() - t0)})
        print(rows[-1], flush=True)
        imps.append(pd.DataFrame({"feature": feats, "fold": k, "gain": model.feature_importance("gain"),
                                  "split": model.feature_importance("split")}))
    pd.concat(imps).to_csv(ROOT / "reports" / f"{name}_importance.csv", index=False)
    return pd.DataFrame(rows)
