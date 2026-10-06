"""error correlations and weighted blends of cached predictions"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from cv import H, evaluate, folds, load_matrices, status_matrix

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data" / "preds"
MODELS = ["lgbm", "catboost", "glm_offset", "nhits"]

M = load_matrices()
rows, corr = [], []
for k, c in enumerate(folds(M["days"]), 1):
    P = {m: np.load(PRED / f"{m}_fold{k}.npy") for m in MODELS}
    ok = status_matrix(M["S"], M["L"], c + H)[:, c:c + H] == 0
    y = M["S"][:, c:c + H]
    for a, b in itertools.combinations(MODELS, 2):
        corr.append({"fold": k, "pair": f"{a} + {b}", "err_corr": np.corrcoef((P[a] - y)[ok], (P[b] - y)[ok])[0, 1]})
    blends = {m: {m: 1.0} for m in MODELS}
    for w in (0.8, 0.7, 0.6, 0.5):
        blends[f"lgbm {w} + catboost {1 - w:.1f}"] = {"lgbm": w, "catboost": 1 - w}
        blends[f"lgbm {w} + glm {1 - w:.1f}"] = {"lgbm": w, "glm_offset": 1 - w}
    blends["lgbm 0.5 + catboost 0.3 + glm 0.2"] = {"lgbm": 0.5, "catboost": 0.3, "glm_offset": 0.2}
    for w in (0.6, 0.5):
        blends[f"lgbm {w} + nhits {1 - w:.1f}"] = {"lgbm": w, "nhits": 1 - w}
    blends["lgbm 0.4 + nhits 0.4 + glm 0.2"] = {"lgbm": 0.4, "nhits": 0.4, "glm_offset": 0.2}
    blends["equal thirds (lgbm, catboost, glm)"] = {m: 1 / 3 for m in ("lgbm", "catboost", "glm_offset")}
    for name, w in blends.items():
        res, _, _ = evaluate(sum(v * P[m] for m, v in w.items()), M, c)
        rows.append({"fold": k, "blend": name, "WRMSSE": res["WRMSSE"], "bias": res["bias"]})

r = pd.DataFrame(rows)
r.to_csv(ROOT / "reports" / "blends_folds.csv", index=False)
print(pd.DataFrame(corr).groupby("pair")["err_corr"].mean().round(3).to_string())
t = r.pivot(index="blend", columns="fold", values="WRMSSE")
t["mean"] = t.mean(axis=1)
t["bias"] = r.groupby("blend")["bias"].mean()
print(t.sort_values("mean").round(4).to_string())
