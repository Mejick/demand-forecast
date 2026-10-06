"""Poisson GLM, plain and with the item level as offset"""
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

from common import fold_data, save_preds, to_matrix
from cv import evaluate, folds

HISTORY = ["lag_28", "lag_35", "mean_28", "mean_56", "dow_mean_4", "size_56", "std_28"]


def design(t, fit_cols=None):
    """log1p history features, one-hot calendar and events"""
    X = pd.DataFrame(index=t.index)
    for c in HISTORY:
        X[c] = np.log1p(t[c].fillna(0).clip(lower=0))
    for c in ["sale_share_28", "oos_share_28", "is_new"]:
        X[c] = t[c].fillna(0)
    X["price_rel"] = t["price_rel"].fillna(1).clip(0.3, 3)
    X["log_price"] = np.log(t["price"].fillna(t["price"].median()))
    X["age"] = t["age_days"].clip(lower=0) / 90
    for c in [c for c in t.columns if c.startswith("ev_")]:
        for o in (-1, 0, 1):
            X[f"{c}_{o:+d}"] = (t[c] == o).astype(np.int8)
    dummies = pd.get_dummies(pd.DataFrame({
        "wd_cat": t["cat_id"].astype(str) + "_" + t["wday"].astype(str),
        "snap_cat": t["cat_id"].astype(str) + "_" + t["snap"].astype(str),
        "month": t["month"].astype(str),
        "store": t["store_id"].astype(str),
        "mday_b": pd.cut(t["mday"], [0, 10, 20, 31]).astype(str)}), dtype=np.int8)
    X = pd.concat([X, dummies], axis=1)
    if fit_cols is not None:
        X = X.reindex(columns=fit_cols, fill_value=0)
    return X.astype(np.float32)


def level_of(t):
    """the item's own level"""
    return t["mean_28"].fillna(t["mean_56"]).fillna(0).values.astype(np.float64)


def run(t, M, offset=True, sample=3_000_000, seed=0):
    name = "glm_offset" if offset else "glm"
    lv_all = level_of(t)
    rows = []
    for k, c in enumerate(folds(M["days"]), 1):
        t0 = time.time()
        train, val = fold_data(t, M, c)
        idx = np.flatnonzero(train & (lv_all > 0) if offset else train)
        idx = np.random.default_rng(seed).choice(idx, min(sample, len(idx)), replace=False)
        sub, tv = t.iloc[idx], t[val]
        Xtr, Xv = design(sub), None
        y = t["sales"].values[idx].astype(np.float64)
        y = np.minimum(y, np.quantile(y, 0.999))
        Xv = design(tv, fit_cols=Xtr.columns)
        if offset:  # history features relative to the item's level
            lv, lvv = lv_all[idx], lv_all[val]
            for col in HISTORY:
                Xtr[col] = np.log1p(sub[col].fillna(0).values) - np.log1p(lv)
                Xv[col] = np.log1p(tv[col].fillna(0).values) - np.log1p(lvv)
        sc = StandardScaler().fit(Xtr.values)
        model = PoissonRegressor(alpha=1e-4, max_iter=300)
        if offset:
            model.fit(sc.transform(Xtr.values), y / lv, sample_weight=lv)
            p = lvv * model.predict(sc.transform(Xv.values))
        else:
            model.fit(sc.transform(Xtr.values), y)
            p = model.predict(sc.transform(Xv.values))
        pred = to_matrix(p, tv, M)
        save_preds(name, k, pred)
        res, _, _ = evaluate(pred, M, c)
        rows.append({"fold": k, "model": name, **res, "sec": round(time.time() - t0)})
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)
