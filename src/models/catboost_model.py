"""CatBoost with Tweedie loss on CPU"""
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from common import CAT_COLS, ROOT, fold_data, save_preds, to_matrix
from cv import H, evaluate, folds
from features import feature_columns


def run(t, M, rounds=1500, device="CPU", sample=4_000_000, seed=0):
    feats = feature_columns(t)
    y = t["sales"].values.astype(np.float32)
    d = t["d"].values
    rows, imps = [], []
    for k, c in enumerate(folds(M["days"]), 1):
        t0 = time.time()
        train, val = fold_data(t, M, c)
        inner_cut = M["days"][c - 1 - H]
        fit, stop = train & (d <= inner_cut), train & (d > inner_cut)
        rng = np.random.default_rng(seed)
        for mask, keep in ((fit, sample), (stop, sample // 8)):
            on = np.flatnonzero(mask)
            if len(on) > keep:
                mask[rng.choice(on, len(on) - keep, replace=False)] = False
        cap = np.quantile(y[train], 0.999)

        def pool(mask, label=True):
            X = t.loc[mask, feats].copy()
            for cc in CAT_COLS:
                X[cc] = X[cc].cat.codes.astype(np.int32)  # integer codes use far less memory than strings
            return Pool(X, np.minimum(y[mask], cap) if label else None, cat_features=CAT_COLS)

        model = CatBoostRegressor(loss_function="Tweedie:variance_power=1.1", iterations=rounds,
                                  learning_rate=0.15 if device == "CPU" else 0.08, depth=8, border_count=128,
                                  l2_leaf_reg=3, task_type=device, thread_count=12, od_type="Iter", od_wait=50,
                                  random_seed=0, verbose=250)
        model.fit(pool(fit), eval_set=pool(stop), use_best_model=True)
        pred = to_matrix(np.maximum(model.predict(pool(val, label=False)), 0), None, M)
        save_preds("catboost", k, pred)
        model.save_model(str(ROOT / "data" / f"catboost_fold{k}.cbm"))
        res, _, _ = evaluate(pred, M, c)
        rows.append({"fold": k, "model": f"catboost_{device.lower()}", **res, "trees": model.get_best_iteration(),
                     "sec": round(time.time() - t0)})
        print(rows[-1], flush=True)
        imps.append(pd.DataFrame({"feature": feats, "fold": k, "importance": model.get_feature_importance()}))
    pd.concat(imps).to_csv(ROOT / "reports" / "catboost_importance.csv", index=False)
    return pd.DataFrame(rows)
