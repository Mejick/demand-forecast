"""features must not change when sales after the cutoff change"""
from pathlib import Path

import numpy as np
import pandas as pd

from features import build, feature_columns, series_id
from status import compute_status

ROOT = Path(__file__).resolve().parents[1]
dev = pd.read_parquet(ROOT / "data" / "dev.parquet")
rng = np.random.default_rng(7)
keep = rng.choice(np.unique(series_id(dev)), 1500, replace=False)  # subset of series to keep it quick
dev = dev[np.isin(series_id(dev), keep)].reset_index(drop=True)

T = int(dev["d"].max()) - 3 * 28
start = T - 365
fake = dev.copy()
after = fake["d"] > T
fake.loc[after, "sales"] = rng.integers(0, 50, after.sum()).astype(fake["sales"].dtype)

a = build(dev, start)
b = build(fake, start)
cols = feature_columns(a)
rows = a["d"] <= T + 28
diff = [c for c in cols if not a.loc[rows, c].equals(b.loc[rows, c])]
print(f"features checked: {len(cols)}; rows d <= T+28: {rows.sum():,}")
print("features that changed (leak):", diff or "none")

# training mask: status recomputed on data <= T only
def mask(df):
    x = df[df["d"] <= T]
    st = compute_status(x["sales"].values, series_id(x), x["launched"].values)
    return st == 0
print("training mask identical:", bool((mask(dev) == mask(fake)).all()))

# a deliberately leaky feature must be caught
a["leaky"] = dev.loc[dev["d"] >= start, "sales"].shift(-1).values[: len(a)]
b["leaky"] = fake.loc[fake["d"] >= start, "sales"].shift(-1).values[: len(b)]
print("deliberate leak detected:", not a.loc[rows, "leaky"].equals(b.loc[rows, "leaky"]))
