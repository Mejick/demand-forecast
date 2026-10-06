"""README figure: synthetic item statuses and hold-out inventory"""
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from demo_model import DEMO_FEATS
from features import build
from inventory import K_WEEK, p_exceed, sum_dist

ROOT = Path(__file__).resolve().parents[1]
SKU, L, H = "ПРО-016", 3, 28


def synthetic_example():
    df = pd.read_csv(ROOT / "docs" / "sample_synthetic_store.csv", comment="#", parse_dates=["date"])
    s = df[df.sku == SKU].sort_values("date").reset_index(drop=True)
    cut = len(s) - H  # forecast the last 28 days
    hist, future = s.iloc[:cut], s.iloc[cut:]
    n = len(s)
    masked = s["sold"].astype("int16").values.copy()
    masked[cut:] = 0
    x = pd.DataFrame({"item_id": pd.Categorical([SKU] * n), "dept_id": pd.Categorical(["P"] * n),
                      "cat_id": pd.Categorical(["P"] * n), "store_id": pd.Categorical(["S"] * n),
                      "state_id": pd.Categorical(["CA"] * n), "d": np.arange(1, n + 1), "date": s["date"],
                      "sales": masked, "sell_price": s["price"].astype("float32"), "snap": np.zeros(n, "int8")})
    x["launched"] = np.arange(n) >= int(np.flatnonzero(s["sold"].values > 0)[0])
    t = build(x, 1)
    mu = lgb.Booster(model_file=str(ROOT / "data" / "demo_lgbm.txt")).predict(
        t[DEMO_FEATS].values[cut:].astype(np.float32))
    h = hist["sold"].values[-112:]
    r = np.clip(h.mean() ** 2 / max(h.var() - h.mean(), 1e-6), 0.05, 1e3)
    stock = round(hist["sold"].values[-28:].mean() * 10)
    rows = []
    for d in range(H):
        if stock <= 0:
            lvl = "black"
        else:
            mL, vL = sum_dist(mu, np.array(r), d, min(d + L, H))
            mW, vW = sum_dist(mu, np.array(r), d, min(d + L + 7, H))
            pL, pW = float(p_exceed(stock, mL, vL)), float(p_exceed(stock, mW, vW))
            lvl = "red" if pL >= 0.5 else "yellow" if pW >= 0.1 else "green"
        demand = int(future["sold"].values[d])
        rows.append({"day": d + 1, "stock": stock, "demand": demand, "forecast_L": mu[d:d + L].sum(), "level": lvl})
        stock -= min(stock, demand)
    return pd.DataFrame(rows)


def main():
    ex = synthetic_example()
    t = pd.read_csv(ROOT / "reports" / "holdout_inventory.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.3, 1]})
    col = {"green": "#27ae60", "yellow": "#f1c40f", "red": "#e74c3c", "black": "#2c3e50"}
    a1.bar(ex["day"], ex["stock"], color=[col[x] for x in ex["level"]])
    a1.plot(ex["day"], ex["demand"], "k.-", lw=1, label="demand per day")
    a1.plot(ex["day"], ex["forecast_L"], color="#8e44ad", ls="--", label=f"forecast for the next {L} days")
    for k, v in col.items():
        a1.bar([0], [0], color=v, label=k)
    a1.set_xlim(0.3, H + 0.7)
    a1.set_xlabel("day"); a1.set_ylabel("units")
    a1.set_title(f"Synthetic item {SKU}: start stock = 10 days, no reorders, lead time {L} days\n"
                 "bar = stock at the start of the day, colour = risk level")
    a1.legend(fontsize=8)
    for pol, m, c in (("model", "o-", "#1f4e79"), ("naive", "s--", "#e67e22")):
        for lt, a in ((3, 1.0), (7, 0.45)):
            x = t[(t.policy == pol) & (t.lead_time == lt)]
            a2.plot(x["avg_stock_value_$"] / 1e3, x["fill_rate"] * 100, m, color=c, alpha=a,
                    label=f"{'forecast + distribution' if pol == 'model' else 'last month average x safety'}, lead {lt} d")
    a2.set_xlabel("average stock on shelves, thousand $ (26 605 items)"); a2.set_ylabel("fill rate, % of demand served")
    a2.set_title("Hold-out: service vs money in stock (up and left is better)")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "12_risk_and_inventory.png", dpi=110)
    print(ex[["day", "stock", "demand", "level"]].to_string(index=False))


if __name__ == "__main__":
    main()
