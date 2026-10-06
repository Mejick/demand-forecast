"""export demo trees to JSON and check them against LightGBM"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from demo_model import DEMO_FEATS
from common import FEAT

ROOT = Path(__file__).resolve().parents[1]


def flatten(tree):
    f, t, l, r, d, v = [], [], [], [], [], []

    def walk(node):
        if "leaf_value" in node:
            v.append(round(node["leaf_value"], 7))
            return ~(len(v) - 1)
        i = len(f)
        f.append(node["split_feature"]); t.append(node["threshold"]); d.append(1 if node["default_left"] else 0)
        assert node["decision_type"] == "<="
        l.append(0); r.append(0)
        l[i] = walk(node["left_child"])
        r[i] = walk(node["right_child"])
        return i

    root = walk(tree["tree_structure"])
    return {"f": f, "t": t, "l": l, "r": r, "d": d, "v": v, "root": root}


def score(trees, x):
    s = 0.0
    for tr in trees:
        n = tr["root"]
        while n >= 0:
            val = x[tr["f"][n]]
            go_left = tr["d"][n] if np.isnan(val) else val <= tr["t"][n]
            n = tr["l"][n] if go_left else tr["r"][n]
        s += tr["v"][~n]
    return s


def main():
    b = lgb.Booster(model_file=str(ROOT / "data" / "demo_lgbm.txt"))
    dump = b.dump_model()
    trees = [flatten(t) for t in dump["tree_info"]]
    out = {"features": DEMO_FEATS, "trees": trees, "note": "LightGBM Tweedie, expected demand = exp(sum of leaves)"}
    path = ROOT / "docs" / "model.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"{len(trees)} trees -> {path} ({path.stat().st_size / 1e6:.2f} MB)")

    t = pd.read_parquet(FEAT, columns=DEMO_FEATS).sample(2000, random_state=0)
    X = t.values.astype(np.float64)
    mine = np.exp([score(trees, x) for x in X])
    ref = b.predict(X)
    print("max abs diff vs LightGBM:", float(np.abs(mine - ref).max()))


if __name__ == "__main__":
    main()
