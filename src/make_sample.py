"""synthetic shop for the demo, not real data"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(42)
dates = pd.date_range("2025-06-01", "2026-09-30", freq="D")
n = len(dates)
wd = dates.dayofweek.values
week = np.where(wd >= 5, 1.2, np.where(wd == 4, 1.0, 0.9))
pay = np.where(np.isin(dates.day, [5, 6, 20, 21]), 1.12, 1.0)  # salary and advance
eve = np.ones(n)
for m_, d_ in [(12, 31), (3, 8), (5, 9), (2, 23)]:
    for y_ in (2025, 2026):
        day = pd.Timestamp(year=y_, month=m_, day=d_)
        for k, f in ((-1, 1.35), (0, 0.8), (-2, 1.1)):
            i = np.flatnonzero(dates == day + pd.Timedelta(days=k))
            eve[i] *= f
season = 1 + 0.1 * np.sin(2 * np.pi * (dates.dayofyear.values - 80) / 365)

items = []
cats = {"Продукты": (25, 40, 0.5, 6), "Бытовая химия": (10, 120, 1.0, 1.2), "Хобби": (6, 450, 2.0, 0.3)}
for cat, (count, price, lump, base) in cats.items():
    for j in range(count):
        level = base * rng.lognormal(0, 0.9)
        items.append(dict(sku=f"{cat[:3].upper()}-{j + 1:03d}", category=cat, level=level, lump=lump,
                          price=round(price * rng.lognormal(0, 0.3), 2)))
rows = []
for k, it in enumerate(items):
    mu = it["level"] * week * pay * eve * season * (1 + 0.0004 * np.arange(n))
    if k == 3:  # new item launched in March 2026, ramps up over ~3 weeks
        start = np.flatnonzero(dates == "2026-03-01")[0]
        ramp = np.clip((np.arange(n) - start) / 21, 0, 1) * 0.5 + 0.5
        mu = np.where(np.arange(n) < start, 0, mu * ramp)
    if it["lump"] > 0.6:  # lumpy: fewer purchases, bigger baskets
        buy = rng.random(n) < np.clip(mu / (1 + it["lump"]), 0, 1)
        sold = np.where(buy, rng.poisson(1 + it["lump"], n), 0)
    else:
        sold = rng.negative_binomial(5, 5 / (5 + mu))
    # stock: replenished weekly up to ~2 weeks of demand
    stock = np.zeros(n, int)
    s = int(it["level"] * 14) + 5
    for t in range(n):
        if wd[t] == 0 and not (k == 7 and dates[t] >= pd.Timestamp("2026-07-01")):
            s = max(s, int(it["level"] * 14 * 1.3) + 3)
        if k == 5 and pd.Timestamp("2026-04-10") <= dates[t] <= pd.Timestamp("2026-04-30"):
            s = 0
        sale = min(s, sold[t])
        sold[t] = sale
        s -= sale
        stock[t] = s
    for t in range(n):
        if k == 3 and dates[t] < pd.Timestamp("2026-03-01"):
            continue
        rows.append((dates[t].date(), it["sku"], it["category"], int(sold[t]), int(stock[t]), it["price"]))

df = pd.DataFrame(rows, columns=["date", "sku", "category", "sold", "stock", "price"])
out = ROOT / "docs" / "sample_synthetic_store.csv"
with open(out, "w", encoding="utf-8", newline="") as fh:
    fh.write("# СИНТЕТИЧЕСКИЕ ДАННЫЕ: сгенерированный магазин для демо, не реальные продажи\n")
    df.to_csv(fh, index=False)
print(df.shape, df["sku"].nunique(), "items;", out, f"{out.stat().st_size / 1e6:.1f} MB")
print(df.groupby("category")["sold"].mean().round(2).to_dict(), "zeros:", round((df["sold"] == 0).mean(), 3))
