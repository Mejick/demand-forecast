"""time-based CV folds and the WRMSSE metric"""
from pathlib import Path

import numpy as np
import pandas as pd

from status import compute_status

ROOT = Path(__file__).resolve().parents[1]
H = 28


def load_matrices():
    path = ROOT / "data" / "matrices.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    dev = pd.read_parquet(ROOT / "data" / "dev.parquet",
                          columns=["item_id", "store_id", "cat_id", "d", "sales", "sell_price", "launched"])
    key = dev["store_id"].astype(str) + "|" + dev["item_id"].astype(str)
    days = np.sort(dev["d"].unique())
    keys = np.asarray(key.unique(), dtype=object)
    n, m = len(keys), len(days)
    assert len(dev) == n * m, "every series must have every day"
    S = dev["sales"].values.reshape(n, m).astype(np.float32)
    P = dev["sell_price"].values.reshape(n, m).astype(np.float32)
    L = dev["launched"].values.reshape(n, m)
    cat = np.asarray(dev["cat_id"].astype(str).to_numpy(dtype=object)).reshape(n, m)[:, 0]
    out = dict(S=S, P=P, L=L, days=days, keys=keys, cat=cat)
    np.savez(path, **out)
    return out


def status_matrix(S, L, upto):
    """Status computed using columns [0, upto) only"""
    n = S.shape[0]
    s, l = S[:, :upto], L[:, :upto]
    sid = np.repeat(np.arange(n), upto)
    return compute_status(s.ravel(), sid, l.ravel()).reshape(n, upto)


def folds(days, n_folds=3):
    """Cutoff column indices: train = [:c], validation = [c:c+H]"""
    m = len(days)
    return [m - H * k for k in range(n_folds, 0, -1)]


def evaluate(pred, M, c, extra_masks=None):
    """metrics for a (n, 28) forecast starting at column c"""
    S, L, P = M["S"], M["L"], M["P"]
    st_eval = status_matrix(S, L, c + H)[:, c:c + H]
    y = S[:, c:c + H]
    ok = (st_eval == 0)
    st_tr = status_matrix(S, L, c)
    hist = np.where(st_tr == 0, S[:, :c], np.nan)
    diff = np.diff(hist, axis=1)
    scale = np.nanmean(diff ** 2, axis=1)
    err2 = np.where(ok, (pred - y) ** 2, np.nan)
    n_ok = ok.sum(1)
    valid = (n_ok > 0) & (scale > 0) & np.isfinite(scale)
    rmsse = np.full(len(S), np.nan)
    rmsse[valid] = np.sqrt(np.nanmean(err2[valid], axis=1) / scale[valid])
    rev = np.nansum(np.where(L[:, c - H:c], S[:, c - H:c] * np.nan_to_num(P[:, c - H:c]), 0), axis=1)
    w = np.where(valid, rev, 0)
    res = {
        "WRMSSE": float(np.nansum(rmsse * w) / w.sum()),
        "RMSSE_mean": float(np.nanmean(rmsse[valid])),
        "MAE": float(np.abs(np.where(ok, pred - y, 0)).sum() / ok.sum()),
        "bias": float(np.where(ok, pred, 0).sum() / np.where(ok, y, 0).sum() - 1),
        "eval_days_share": float(ok.mean()),
    }
    return res, rmsse, w
