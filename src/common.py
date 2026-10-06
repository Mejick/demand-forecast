"""shared paths, feature table and training mask per fold"""
from pathlib import Path

import numpy as np
import pandas as pd

from cv import H, status_matrix
from features import build

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "features.parquet"
PREDS = ROOT / "data" / "preds"
TRAIN_DAYS = 365 * 2
START_D = 1100
CAT_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


def build_table():
    dev = pd.read_parquet(ROOT / "data" / "dev.parquet")
    t = build(dev, START_D)
    t.to_parquet(FEAT, index=False)
    print(t.shape)


def fold_data(t, M, c):
    """training and validation rows for cutoff column c"""
    days = M["days"]
    n = len(M["keys"])
    first_col = int(np.searchsorted(days, START_D))
    per_series = len(days) - first_col
    assert len(t) == n * per_series, "feature table must be series-major with every dev day >= START_D"
    col = first_col + np.tile(np.arange(per_series), n)  # matrix column of each feature row
    ser = np.repeat(np.arange(n), per_series)
    st = status_matrix(M["S"], M["L"], c)  # status known at the cutoff
    train = (col < c) & (col >= c - TRAIN_DAYS)
    train[train] = st[ser[train], col[train]] == 0
    val = (col >= c) & (col < c + H)
    return train, val


def to_matrix(pred_rows, t_val, M):
    return pred_rows.reshape(len(M["keys"]), H)


def save_preds(name, k, pred):
    PREDS.mkdir(parents=True, exist_ok=True)
    np.save(PREDS / f"{name}_fold{k}.npy", pred)
