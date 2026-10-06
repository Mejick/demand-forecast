"""simple rules: means, weekly profile, Croston, TSB"""
import numpy as np
import pandas as pd

from common import ROOT, save_preds
from cv import H, evaluate, folds, load_matrices, status_matrix


def instock_hist(M, c):
    st = status_matrix(M["S"], M["L"], c)
    return np.where(st == 0, M["S"][:, :c], np.nan), st


def b_zero(M, c, hist):
    return np.zeros((len(M["S"]), H))


def b_mean(w):
    def f(M, c, hist):
        m = np.nanmean(hist[:, c - w:c], axis=1)
        return np.repeat(np.nan_to_num(m)[:, None], H, axis=1)
    f.__name__ = f"mean_{w}"
    return f


def b_last_week(M, c, hist):
    """Seasonal naive: repeat the last 7 days (raw sales)"""
    last = M["S"][:, c - 7:c]
    return np.tile(last, (1, H // 7))


def b_dow_mean(M, c, hist):
    """Mean of the last 4 same weekdays, on-sale days only"""
    out = np.zeros((len(M["S"]), H))
    for h in range(H):
        cols = [c - 7 * k + (h % 7) for k in range(1, 5)]
        out[:, h] = np.nan_to_num(np.nanmean(hist[:, cols], axis=1))
    return out


def b_mean_dow_profile(M, c, hist):
    """28-day level times the weekly profile of the category"""
    level = np.nan_to_num(np.nanmean(hist[:, c - 28:c], axis=1))
    out = np.zeros((len(M["S"]), H))
    for cat in np.unique(M["cat"]):
        r = M["cat"] == cat
        tot = np.nansum(hist[r, c - 364:c], axis=0)
        prof = np.array([tot[i::7].mean() for i in range(7)])
        prof = prof / prof.mean()
        out[r] = level[r, None] * np.tile(prof, H // 7)[None, :]
    return out


def _croston(M, c, hist, alpha=0.1, tsb=False, beta=0.1):
    n = len(hist)
    z = np.full(n, np.nan); p = np.full(n, np.nan); q = np.ones(n); d = np.full(n, np.nan)
    for t in range(c):
        y = hist[:, t]
        on = ~np.isnan(y)
        dem = on & (y > 0)
        init = dem & np.isnan(z)
        z[init], p[init], d[init] = y[init], q[init], 1.0
        upd = dem & ~init
        z[upd] += alpha * (y[upd] - z[upd])
        p[upd] += alpha * (q[upd] - p[upd])
        if tsb:
            ok = on & ~np.isnan(d) & ~init
            d[ok] += beta * (dem[ok].astype(float) - d[ok])
        q[dem] = 1
        q[on & ~dem] += 1
    f = (d * z) if tsb else (1 - alpha / 2) * z / p
    return np.repeat(np.nan_to_num(f)[:, None], H, axis=1)


def b_croston_sba(M, c, hist):
    return _croston(M, c, hist)


def b_tsb(M, c, hist):
    return _croston(M, c, hist, tsb=True)


BASELINES = [b_zero, b_mean(7), b_mean(28), b_mean(56), b_last_week, b_dow_mean, b_mean_dow_profile,
             b_croston_sba, b_tsb]


def run(M):
    types = pd.read_parquet(ROOT / "data" / "series_types.parquet")
    tkey = types["store_id"].astype(str) + "|" + types["item_id"].astype(str)
    stype = pd.Series(types["type"].values, index=tkey).reindex(M["keys"]).fillna("n/a").values
    rows, seg = [], []
    for k, c in enumerate(folds(M["days"]), 1):
        hist, _ = instock_hist(M, c)
        for b in BASELINES:
            pred = b(M, c, hist)
            save_preds(b.__name__.replace("b_", ""), k, pred)
            res, rmsse, w = evaluate(pred, M, c)
            rows.append({"fold": k, "cutoff_d": int(M["days"][c - 1]), "model": b.__name__.replace("b_", ""), **res})
            for t in np.unique(stype):
                m = (stype == t) & np.isfinite(rmsse) & (w > 0)
                seg.append({"fold": k, "model": b.__name__.replace("b_", ""), "type": t,
                            "WRMSSE": float(np.sum(rmsse[m] * w[m]) / w[m].sum())})
        print(f"fold {k} done")
    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "reports" / "rules_folds.csv", index=False)
    summ = r.groupby("model")[["WRMSSE", "RMSSE_mean", "MAE", "bias"]].mean().sort_values("WRMSSE")
    summ.to_csv(ROOT / "reports" / "rules.csv")
    print(summ.round(4).to_string())
    s = pd.DataFrame(seg).groupby(["model", "type"])["WRMSSE"].mean().unstack()
    s.to_csv(ROOT / "reports" / "rules_by_type.csv")
    print(s.loc[summ.index].round(3).to_string())
    print(r.pivot(index="model", columns="fold", values="WRMSSE").loc[summ.index].round(4).to_string())

