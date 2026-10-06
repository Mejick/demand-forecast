"""the browser demo must forecast like the Python model"""
import json
import warnings
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from demo_model import DEMO_FEATS  # noqa: E402
from features import build, causal_oos  # noqa: E402

SAMPLE = ROOT / "docs" / "sample_synthetic_store.csv"
warnings.filterwarnings("ignore", category=DeprecationWarning)


def python_forecast(df, booster):
    out = {}
    for sku, s in df.groupby("sku"):
        s = s.sort_values("date").reset_index(drop=True)
        full = pd.date_range(s["date"].min(), df["date"].max())
        s = s.set_index("date").reindex(full, fill_value=0).rename_axis("date").reset_index()
        sales = s["sold"].astype(float).values
        launched = np.arange(len(s)) >= np.argmax(sales > 0)
        if causal_oos(sales, np.zeros(len(s), np.int64), launched).any() or len(s) < 56:
            continue
        fut = pd.DataFrame({"date": pd.date_range(s["date"].max() + pd.Timedelta(days=1), periods=28), "sold": 0})
        s = pd.concat([s[["date", "sold"]], fut], ignore_index=True)
        n = len(s)
        x = pd.DataFrame({c: pd.Categorical(["x"] * n) for c in ("item_id", "dept_id", "cat_id", "store_id", "state_id")})
        x["d"] = np.arange(1, n + 1)
        x["date"] = s["date"]
        x["sales"] = s["sold"].astype("int16")
        x["sell_price"] = np.float32(1.0)
        x["snap"] = np.int8(0)
        x["launched"] = np.r_[launched, np.ones(28, bool)]
        t = build(x, 1)
        out[sku] = booster.predict(t[DEMO_FEATS].values[-28:].astype(np.float32))
    return out


def main():
    js = json.loads(subprocess.run(["node", str(ROOT / "tests" / "demo_forecast.js"), str(SAMPLE)],
                                   capture_output=True, text=True, check=True, encoding="utf-8").stdout)
    df = pd.read_csv(SAMPLE, comment="#", parse_dates=["date"])
    py = python_forecast(df, lgb.Booster(model_file=str(ROOT / "data" / "demo_lgbm.txt")))
    diffs = {sku: float(np.max(np.abs(np.array(js[sku]) - py[sku]))) for sku in py}
    worst = max(diffs.values())
    print(f"compared {len(diffs)} items (skipped {len(js) - len(diffs)} with inferred empty shelves), "
          f"max abs difference {worst:.2e}")
    assert worst < 1e-3, diffs


if __name__ == "__main__":
    main()
