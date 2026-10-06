"""three M5 tables into one long table: item x store x day"""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


def main():
    sales = pd.read_csv(DATA / "sales_train_evaluation.csv")
    cal = pd.read_csv(DATA / "calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(DATA / "sell_prices.csv")

    keys = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    for c in keys:
        sales[c] = sales[c].astype("category")
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    long = sales.melt(id_vars=keys, value_vars=day_cols, var_name="d", value_name="sales")
    long["sales"] = long["sales"].astype("int16")
    long["d"] = long["d"].str[2:].astype("int16")

    cal["d"] = cal["d"].str[2:].astype("int16")
    cal_cols = ["d", "date", "wm_yr_wk", "weekday", "month", "year", "event_name_1", "event_type_1",
                "snap_CA", "snap_TX", "snap_WI"]
    long = long.merge(cal[cal_cols], on="d", how="left")

    # one SNAP flag for the row's own state instead of three
    state = long["state_id"].astype(str)
    long["snap"] = 0
    for s in ("CA", "TX", "WI"):
        long.loc[state == s, "snap"] = long.loc[state == s, f"snap_{s}"]
    long = long.drop(columns=["snap_CA", "snap_TX", "snap_WI"])
    long["snap"] = long["snap"].astype("int8")

    for c in ("item_id", "store_id"):
        prices[c] = prices[c].astype(long[c].dtype)
    prices["sell_price"] = prices["sell_price"].astype("float32")
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    # no price yet means the item is not launched
    has_price = long["sell_price"].notna()
    long["launched"] = has_price.groupby([long["item_id"], long["store_id"]], observed=True).cummax()
    print(f"rows: {len(long):,}, pre-launch: {(~long['launched']).sum():,}, "
          f"missing price after launch: {(long['launched'] & ~has_price).sum():,}")

    for c in ("weekday", "event_name_1", "event_type_1"):
        long[c] = long[c].astype("category")
    long = long.reset_index(drop=True)
    long.to_parquet(DATA / "long.parquet", index=False)
    print(long.dtypes)
    print(f"memory: {long.memory_usage(deep=True).sum() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
