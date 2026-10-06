"""negative binomial around the best forecast and its calibration"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from cv import H, folds, load_matrices, status_matrix

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data" / "preds"
QS = (0.5, 0.8, 0.9, 0.95)
WINDOW = 112


def center(k):
    return 0.5 * np.load(PRED / f"lgbm_fold{k}.npy") + 0.5 * np.load(PRED / f"nhits_fold{k}.npy")


def dispersion(M, c):
    """NB size per series from the last 112 on-sale days"""
    st = status_matrix(M["S"], M["L"], c)[:, c - WINDOW:c]
    y = np.where(st == 0, M["S"][:, c - WINDOW:c], np.nan)
    m, v = np.nanmean(y, axis=1), np.nanvar(y, axis=1)
    excess = v - m
    r = np.where(excess > 1e-6, m ** 2 / np.maximum(excess, 1e-6), 1e3)
    return np.clip(np.nan_to_num(r, nan=1.0), 0.05, 1e3)


def nb_quantile(q, mu, var):
    """NB quantile from mean and variance"""
    mu = np.maximum(mu, 1e-6)
    var = np.maximum(var, mu * (1 + 1e-6))
    r = mu ** 2 / (var - mu)
    return nbinom.ppf(q, r, r / (r + mu))


def coverage(M, c, mu, r, k_week=1.0):
    st = status_matrix(M["S"], M["L"], c + H)[:, c:c + H]
    y = M["S"][:, c:c + H]
    ok = st == 0
    out = {}
    var_d = mu + mu ** 2 / r[:, None]
    for q in QS:
        out[("day", q)] = float((y[ok] <= nb_quantile(q, mu, var_d)[ok]).mean())
    for q in QS:
        hits, n = 0, 0
        for w in range(H // 7):
            s = slice(7 * w, 7 * w + 7)
            full = ok[:, s].all(axis=1)
            mu_w = mu[:, s].sum(1)
            var_w = mu_w + k_week * (var_d[:, s].sum(1) - mu_w)
            hits += int((y[:, s].sum(1)[full] <= nb_quantile(q, mu_w, var_w)[full]).sum())
            n += int(full.sum())
        out[("week", q)] = hits / n
    return out


def main():
    M = load_matrices()
    data = {k: (c, center(k), dispersion(M, c)) for k, c in enumerate(folds(M["days"]), 1)}

    rows = []
    for k, (c, mu, r) in data.items():
        rows.append({"fold": k, "k_week": 1.0, **coverage(M, c, mu, r)})
    # fit the weekly inflation on folds 1-2
    best = None
    for kw in (1.0, 1.5, 2, 3, 4, 5, 6, 8, 10):
        cov = np.mean([coverage(M, data[k][0], data[k][1], data[k][2], kw)[("week", 0.9)] for k in (1, 2)])
        if best is None or abs(cov - 0.9) < abs(best[1] - 0.9):
            best = (kw, cov)
    kw = best[0]
    for k, (c, mu, r) in data.items():
        rows.append({"fold": k, "k_week": kw, **coverage(M, c, mu, r, kw)})
    t = pd.DataFrame(rows)
    t.columns = [f"{a}_{int(b * 100)}" if isinstance(a, str) and b != "" else a for a, b in
                 [(c_ if isinstance(c_, tuple) else (c_, "")) for c_ in t.columns]] if False else \
        ["fold", "k_week"] + [f"{a}_q{int(b * 100)}" for a, b in t.columns[2:]]
    t.to_csv(ROOT / "reports" / "calibration.csv", index=False)
    print(f"weekly inflation fitted on folds 1-2: k = {kw} (fold 1-2 weekly q90 coverage {best[1]:.3f})")
    print(t.round(3).to_string(index=False))
    np.save(ROOT / "data" / "k_week.npy", np.array([kw]))


if __name__ == "__main__":
    main()
