"""item status per day: on sale, out of stock, delisted, not launched"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WINDOW, MIN_PRIOR, ALPHA = 56, 7, 0.001
DELIST_MIN = 28
ON_SALE, OOS, DELISTED, NOT_LAUNCHED = 0, 1, 2, 3


def _runs(mask, sid):
    """start, length and run id of consecutive True blocks"""
    prev_same = np.r_[False, sid[1:] == sid[:-1]]
    prev_mask = np.r_[False, mask[:-1]]
    start = mask & ~(prev_same & prev_mask)
    run_id = np.cumsum(start) - 1
    starts = np.flatnonzero(start)
    lengths = np.bincount(run_id[mask], minlength=len(starts))
    return starts, lengths, np.where(mask, run_id, -1)


def compute_status(sales, sid, launched=None):
    """status per row, arrays sorted by series then day"""
    sales = np.asarray(sales, dtype=np.float64)
    n = len(sales)
    launched = np.ones(n, dtype=bool) if launched is None else np.asarray(launched, dtype=bool)
    # a series "starts" at its first launched day
    first = launched & ~np.r_[False, (sid[1:] == sid[:-1]) & launched[:-1]]
    series_start = np.maximum.accumulate(np.where(first, np.arange(n), 0))
    last = np.r_[sid[1:] != sid[:-1], True]
    series_end = np.minimum.accumulate(np.where(last, np.arange(n), n)[::-1])[::-1]
    cp = np.r_[0, np.cumsum(sales > 0)]

    status = np.zeros(n, dtype=np.int8)
    zero = (sales == 0) & launched
    if zero.any():
        st, ln, rid = _runs(zero, sid)
        lo = np.maximum(series_start[st], st - WINDOW)
        prior = st - lo
        p = np.where(prior > 0, (cp[st] - cp[lo]) / np.maximum(prior, 1), 0.0)
        oos = (prior >= MIN_PRIOR) & (p > 0) & ((1 - p) ** ln < ALPHA)
        to_end = (st + ln - 1) == series_end[st]
        delisted = oos & to_end & (ln >= DELIST_MIN)
        run_status = np.where(delisted, DELISTED, np.where(oos, OOS, ON_SALE)).astype(np.int8)
        status[zero] = run_status[rid[zero]]
    status[~launched] = NOT_LAUNCHED
    return status


def build():
    df = pd.read_parquet(ROOT / "data" / "long.parquet", columns=["item_id", "store_id", "d", "date", "sales", "launched"])
    df = df[~((df["date"].dt.month == 12) & (df["date"].dt.day == 25))]
    df = df.sort_values(["store_id", "item_id", "d"], kind="stable").reset_index(drop=True)
    sid = (df["store_id"].cat.codes.astype(np.int64) * 100000 + df["item_id"].cat.codes.astype(np.int64)).values
    df["status"] = compute_status(df["sales"].values, sid, df["launched"].values)
    df[["item_id", "store_id", "d", "status"]].to_parquet(ROOT / "data" / "status.parquet", index=False)
    return df, sid


if __name__ == "__main__":
    build()
