"""features for the direct 28-day strategy, history shifted by 28 days"""
from pathlib import Path

import numpy as np
import pandas as pd

from status import ALPHA, MIN_PRIOR, WINDOW

ROOT = Path(__file__).resolve().parents[1]
SHIFT = 28
LAGS = (28, 35, 42, 49, 56)
STRONG_EVENTS = ("Thanksgiving", "NewYear", "IndependenceDay", "LaborDay", "Easter", "Mother's day",
                 "Halloween", "ValentinesDay", "SuperBowl", "MemorialDay")
EVENT_CLIP = 4


def series_id(df):
    return (df["store_id"].cat.codes.astype(np.int64) * 100000 + df["item_id"].cat.codes.astype(np.int64)).values


def _starts(sid):
    first = np.r_[True, sid[1:] != sid[:-1]]
    return np.maximum.accumulate(np.where(first, np.arange(len(sid)), 0))


def _window_sum(cum, idx, lo, hi):
    """sum over rows [idx-hi, idx-lo]"""
    return cum[idx - lo + 1] - cum[idx - hi]


def causal_oos(sales, sid, launched):
    """out-of-stock flag known on the day itself"""
    n = len(sales)
    zero = (sales == 0) & launched
    s0 = _starts(sid)
    first_l = launched & ~np.r_[False, (sid[1:] == sid[:-1]) & launched[:-1]]
    l0 = np.maximum.accumulate(np.where(first_l, np.arange(n), 0))
    cp = np.r_[0, np.cumsum(sales > 0)]
    run_start = zero & ~np.r_[False, (sid[1:] == sid[:-1]) & zero[:-1]]
    rs = np.maximum.accumulate(np.where(run_start, np.arange(n), 0))
    length = np.arange(n) - rs + 1
    lo = np.maximum(np.maximum(l0, s0)[rs], rs - WINDOW)
    prior = rs - lo
    p = np.where(prior > 0, (cp[rs] - cp[lo]) / np.maximum(prior, 1), 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        flag = zero & (prior >= MIN_PRIOR) & (p > 0) & ((1 - p) ** length < ALPHA)
    return flag


def _event_offsets(dates):
    cal = pd.read_csv(ROOT / "data" / "calendar.csv", parse_dates=["date"])
    out = {}
    for ev in STRONG_EVENTS:
        days = cal.loc[(cal["event_name_1"] == ev) | (cal["event_name_2"] == ev), "date"].values
        diff = (dates.values[:, None] - days[None, :]).astype("timedelta64[D]").astype(np.int64)
        near = diff[np.arange(len(dates)), np.abs(diff).argmin(axis=1)]  # signed days to the nearest one
        name = "ev_" + ev.lower().replace("'", "").replace(" ", "_")
        out[name] = np.clip(near, -EVENT_CLIP, EVENT_CLIP).astype(np.int8)
    return pd.DataFrame(out, index=dates.index)


def build(df, start_d):
    df = df[df["d"] >= start_d - 120].reset_index(drop=True)  # room for 56-day windows plus the shift
    sid = series_id(df)
    s0 = _starts(sid)
    n = len(df)
    idx = np.arange(n)
    sales = df["sales"].values.astype(np.float64)
    launched = df["launched"].values.astype(bool)
    oos = causal_oos(sales, sid, launched)
    instock = launched & ~oos

    f = pd.DataFrame(index=df.index)
    pos = idx - s0  # position inside the series

    def shifted(arr, k):
        out = np.full(n, np.nan)
        ok = pos >= k
        out[ok] = arr[idx[ok] - k]
        return out

    for k in LAGS:
        f[f"lag_{k}"] = shifted(np.where(launched, sales, np.nan), k)

    v = np.where(instock, sales, 0.0)
    c = instock.astype(np.float64)
    cv, cc = np.r_[0.0, np.cumsum(v)], np.r_[0.0, np.cumsum(c)]
    cv2 = np.r_[0.0, np.cumsum(v ** 2)]
    pos_sale = (instock & (sales > 0)).astype(np.float64)
    cps = np.r_[0.0, np.cumsum(pos_sale)]
    co = np.r_[0.0, np.cumsum((oos & launched).astype(np.float64))]
    cl = np.r_[0.0, np.cumsum(launched.astype(np.float64))]

    def win(cum, w):
        """sum over w days ending 28 days before the row"""
        hi = np.minimum(SHIFT + w - 1, pos)
        out = np.where(pos >= SHIFT, cum[idx - SHIFT + 1] - cum[idx - hi], np.nan)
        return out

    for w in (7, 28, 56):
        cnt = win(cc, w)
        f[f"mean_{w}"] = np.where(cnt > 0, win(cv, w) / np.maximum(cnt, 1), np.nan)
    cnt28 = win(cc, 28)
    m28 = f["mean_28"].values
    f["std_28"] = np.sqrt(np.maximum(np.where(cnt28 > 1, win(cv2, 28) / np.maximum(cnt28, 1) - m28 ** 2, np.nan), 0))
    for w in (28, 56):
        cnt = win(cc, w)
        f[f"sale_share_{w}"] = np.where(cnt > 0, win(cps, w) / np.maximum(cnt, 1), np.nan)
    ns = win(cps, 56)
    f["size_56"] = np.where(ns > 0, win(cv, 56) / np.maximum(ns, 1), np.nan)
    nl = win(cl, 28)
    f["oos_share_28"] = np.where(nl > 0, win(co, 28) / np.maximum(nl, 1), np.nan)

    same_dow = [shifted(np.where(instock, sales, np.nan), k) for k in (28, 35, 42, 49)]
    f["dow_mean_4"] = np.nanmean(np.vstack(same_dow), axis=0) if n else []

    # calendar (known in advance)
    date = df["date"]
    f["wday"] = date.dt.dayofweek.astype(np.int8)
    f["weekend"] = (f["wday"] >= 5).astype(np.int8)
    f["mday"] = date.dt.day.astype(np.int8)
    f["month"] = date.dt.month.astype(np.int8)
    f["snap"] = df["snap"].astype(np.int8)
    f = f.join(_event_offsets(date))

    # price: planned price is known
    price = df["sell_price"].values.astype(np.float64)
    pv = np.where(launched, price, 0.0)
    cpv = np.r_[0.0, np.cumsum(np.nan_to_num(pv))]
    f["price"] = price
    past_cnt = np.where(pos >= SHIFT, cl[idx - SHIFT + 1] - cl[s0], 0)
    past_sum = np.where(pos >= SHIFT, cpv[idx - SHIFT + 1] - cpv[s0], 0)
    f["price_rel"] = np.where(past_cnt > 0, price / np.where(past_cnt > 0, past_sum / np.maximum(past_cnt, 1), np.nan), np.nan)
    f["price_chg_7"] = price / shifted(price, 7) - 1

    # novelty: days since the first launched day of the series
    first_l = launched & ~np.r_[False, (sid[1:] == sid[:-1]) & launched[:-1]]
    l0 = np.maximum.accumulate(np.where(first_l, idx, -1))
    age = np.where(launched & (l0 >= s0), idx - l0, -1)
    f["age_days"] = np.minimum(age, 90).astype(np.int16)
    f["is_new"] = ((age >= 0) & (age < 28)).astype(np.int8)

    for c_ in ("item_id", "dept_id", "cat_id", "store_id", "state_id"):
        f[c_] = df[c_]
    keys = df[["d", "date", "sales", "launched"]]
    out = pd.concat([keys, f], axis=1)
    out = out[out["d"] >= start_d].reset_index(drop=True)
    num = out.select_dtypes("float64").columns
    out[num] = out[num].astype(np.float32)
    return out


FEATURES = None


def feature_columns(table):
    return [c for c in table.columns if c not in ("d", "date", "sales", "launched")]
