"""split off the last 28 days as the final hold-out"""
from pathlib import Path

import numpy as np
import pandas as pd

from status import compute_status

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 28


def main():
    df = pd.read_parquet(ROOT / "data" / "long.parquet")
    last_d = int(df["d"].max())
    cut = last_d - HORIZON  # last day of dev
    df = df[~((df["date"].dt.month == 12) & (df["date"].dt.day == 25))]
    df = df.sort_values(["store_id", "item_id", "d"], kind="stable").reset_index(drop=True)

    dev = df[df["d"] <= cut].reset_index(drop=True)
    hold = df[df["d"] > cut].reset_index(drop=True)
    sid = (dev["store_id"].cat.codes.astype(np.int64) * 100000 + dev["item_id"].cat.codes.astype(np.int64)).values
    dev["status"] = compute_status(dev["sales"].values, sid, dev["launched"].values)

    dev.to_parquet(ROOT / "data" / "dev.parquet", index=False)
    hold.to_parquet(ROOT / "data" / "holdout.parquet", index=False)
    print(f"dev: d_1..d_{cut} ({dev['date'].min().date()} - {dev['date'].max().date()}), {len(dev):,} rows")
    print(f"holdout: d_{cut + 1}..d_{last_d} ({hold['date'].min().date()} - {hold['date'].max().date()}), {len(hold):,} rows")
    assert dev["d"].max() < hold["d"].min()


if __name__ == "__main__":
    main()
