"""risk levels and inventory simulation"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from cv import H, folds, load_matrices, status_matrix
from uncertainty import center, dispersion

ROOT = Path(__file__).resolve().parents[1]
K_WEEK = float(np.load(ROOT / "data" / "k_week.npy")[0])
LEVELS = ["green", "yellow", "red", "black"]


def sum_dist(mu, r, a, b):
    """mean and variance of demand over days [a, b)"""
    m = mu[..., a:b].sum(-1)
    v_day = (mu[..., a:b] + mu[..., a:b] ** 2 / r[..., None]).sum(-1)
    return m, m + K_WEEK * (v_day - m)


def p_exceed(stock, m, v):
    """P(demand > stock) for a negative binomial"""
    m = np.maximum(m, 1e-9)
    v = np.maximum(v, m * (1 + 1e-6))
    rr = m ** 2 / (v - m)
    return nbinom.sf(np.floor(stock), rr, rr / (rr + m))


def risk_level(on_hand, arrive_L, arrive_all, mu, r, t, L):
    if on_hand <= 0:
        return "black", 1.0, 1.0
    end = mu.shape[-1]
    mL, vL = sum_dist(mu, r, t, min(t + L, end))
    mW, vW = sum_dist(mu, r, t, min(t + L + 7, end))
    pL = float(p_exceed(on_hand + arrive_L, mL, vL))
    pW = float(p_exceed(on_hand + arrive_all, mW, vW))
    if pL >= 0.5:
        return "red", pL, pW
    if pW >= 0.1:
        return "yellow", pL, pW
    return "green", pL, pW


def simulate(M, c, mu, r, L, q=0.9, policy="model", mult=1.2):
    """daily review, order up to target, lost sales"""
    st0 = status_matrix(M["S"], M["L"], c + H)[:, c:c + H]
    full = (st0 == 0).all(1)  # items on sale the whole month: actual sales = demand
    idx = np.flatnonzero(full)
    y = M["S"][idx, c:c + H]
    price = np.nan_to_num(M["P"][idx, c - 1])
    hist = np.where(status_matrix(M["S"], M["L"], c)[idx, c - 28:c] == 0, M["S"][idx, c - 28:c], np.nan)
    avg = np.nan_to_num(np.nanmean(hist, axis=1))
    mu_i, r_i = mu[idx], r[idx]
    days = H - L - 1
    on_hand = np.round(avg * (L + 7)).astype(float)
    pipeline = np.zeros((len(idx), days + L + 1))
    lost = sold = demand = 0.0
    stock_value = 0.0
    out_days = 0
    for t in range(days):
        on_hand += pipeline[:, t]
        d = y[:, t]
        s = np.minimum(on_hand, d)
        lost += (d - s).sum(); sold += s.sum(); demand += d.sum()
        out_days += int(((on_hand - s) <= 0).sum())
        on_hand -= s
        stock_value += (on_hand * price).sum()
        position = on_hand + pipeline[:, t + 1:].sum(1)
        if policy == "model":
            m, v = sum_dist(mu_i, r_i, t + 1, t + L + 2)
            m = np.maximum(m, 1e-9); v = np.maximum(v, m * (1 + 1e-6)); rr = m ** 2 / (v - m)
            target = nbinom.ppf(q, rr, rr / (rr + m))
        else:
            target = np.ceil(avg * (L + 1) * mult)
        order = np.maximum(0, target - position)
        pipeline[:, t + L] += order
    n_item_days = len(idx) * days
    return {"policy": policy, "lead_time": L, "items": len(idx), "fill_rate": sold / demand,
            "empty_shelf_share": out_days / n_item_days, "avg_stock_value_$": stock_value / days,
            "lost_units": lost}


def main():
    """service vs money in stock on fold 3"""
    M = load_matrices()
    k, c = 3, folds(M["days"])[-1]
    mu, r = center(k), dispersion(M, c)
    rows = []
    for L in (3, 7):  # with L=14 an order cannot arrive within 28 days
        for q in (0.6, 0.7, 0.8, 0.9, 0.95):
            rows.append({**simulate(M, c, mu, r, L, q=q), "param": f"q={q}"})
        for m in (1.0, 1.2, 1.5, 2.0, 2.5):
            rows.append({**simulate(M, c, mu, r, L, policy="naive", mult=m), "param": f"x{m}"})
    s = pd.DataFrame(rows)[["lead_time", "policy", "param", "fill_rate", "empty_shelf_share", "avg_stock_value_$"]]
    s.to_csv(ROOT / "reports" / "inventory_tradeoff.csv", index=False)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
