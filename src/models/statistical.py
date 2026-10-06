"""one statistical model per series with statsforecast"""
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import ADIDA, IMAPA, AutoARIMA, AutoETS, AutoTheta, CrostonOptimized

from common import ROOT, save_preds
from cv import H, evaluate, folds, status_matrix
HIST = 364


SETS = {
    "ets": [(AutoETS(season_length=7, model="ZNA"), "AutoETS", "ets"), (CrostonOptimized(), "CrostonOptimized", "croston_opt")],
    "arima": [(AutoARIMA(season_length=7), "AutoARIMA", "arima")],  # ~2 h without finishing one fold on 30k series
    "more": [(AutoTheta(season_length=7), "AutoTheta", "theta"),
             (ADIDA(), "ADIDA", "adida"), (IMAPA(), "IMAPA", "imapa")],
}


def run(M, which="ets"):
    spec = SETS[which]
    n = len(M["keys"])
    rows = []
    for k, c in enumerate(folds(M["days"]), 1):
        st = status_matrix(M["S"], M["L"], c)[:, c - HIST:c]
        y = M["S"][:, c - HIST:c].astype(np.float64)
        on = st == 0
        mean_on = np.where(on.sum(1) > 0, np.where(on, y, 0).sum(1) / np.maximum(on.sum(1), 1), 0)
        y = np.where(on, y, mean_on[:, None])  # demand-like history
        active = on.sum(1) >= 28
        df = pd.DataFrame({"unique_id": np.repeat(np.arange(n), HIST)[np.repeat(active, HIST)],
                           "ds": np.tile(pd.date_range("2000-01-03", periods=HIST), active.sum()),
                           "y": y[active].ravel()})
        sf = StatsForecast(models=[m for m, _, _ in spec], freq="D", n_jobs=12)
        fc = sf.forecast(df=df, h=H).reset_index()
        for _, col, name in spec:
            pred = np.repeat(mean_on[:, None], H, axis=1)
            pred[active] = np.clip(fc[col].values.reshape(active.sum(), H), 0, None)
            save_preds(name, k, pred)
            res, _, _ = evaluate(pred, M, c)
            rows.append({"fold": k, "model": name, **res})
            print(rows[-1], flush=True)
    r = pd.DataFrame(rows)
    print(r.groupby("model")[["WRMSSE", "MAE", "bias"]].mean().round(4).to_string())
    return r
