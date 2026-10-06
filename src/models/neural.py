"""N-HiTS and DeepAR with calendar features, GPU 1"""
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MSE, DistributionLoss
from neuralforecast.models import NHITS, DeepAR

from common import ROOT, save_preds
from cv import H, evaluate, folds, status_matrix
HIST = 728
EXOG = ["wday_sin", "wday_cos", "snap", "event", "event_tomorrow"]


def calendar(days_d, keys):
    cal = pd.read_csv(ROOT / "data" / "calendar.csv", parse_dates=["date"])
    cal["d"] = cal["d"].str[2:].astype(int)
    cal = cal.set_index("d").loc[days_d]
    wd = cal["date"].dt.dayofweek.values
    ev = cal["event_name_1"].notna().astype(np.float32).values
    out = {"date": cal["date"].values, "wday_sin": np.sin(2 * np.pi * wd / 7), "wday_cos": np.cos(2 * np.pi * wd / 7),
           "event": ev, "event_tomorrow": np.r_[ev[1:], 0.0]}
    state = np.array([k.split("|")[0][:2] for k in keys])
    snap = np.vstack([cal[f"snap_{s}"].values for s in state]).astype(np.float32)
    return out, snap


def make_model(name, n_series):
    common = dict(h=H, input_size=56, futr_exog_list=EXOG, max_steps=3000, batch_size=512,
                  val_check_steps=200, early_stop_patience_steps=5, scaler_type="identity", random_seed=0,
                  accelerator="gpu", devices=1, enable_progress_bar=False, enable_model_summary=False)
    if name == "nhits":
        return NHITS(loss=MSE(), learning_rate=1e-3, windows_batch_size=1024, **common)
    return DeepAR(loss=DistributionLoss(distribution="StudentT", level=[80, 90]),  # NegBin capped big items ~17
                  learning_rate=1e-3, lstm_n_layers=2, lstm_hidden_size=64, **common)


def run(M, name, only_fold=None):
    keys, n = M["keys"], len(M["keys"])
    rows = []
    for k, c in enumerate(folds(M["days"]), 1):
        if only_fold and k != only_fold:
            continue
        t0 = time.time()
        st = status_matrix(M["S"], M["L"], c)[:, c - HIST:c]
        y = M["S"][:, c - HIST:c].astype(np.float32)
        on = st == 0
        mean_on = np.where(on.sum(1) > 0, np.where(on, y, 0).sum(1) / np.maximum(on.sum(1), 1), 0).astype(np.float32)
        y = np.where(on, y, mean_on[:, None])
        active = on.sum(1) >= 56
        cal_h, snap_h = calendar(M["days"][c - HIST:c], keys)
        cal_f, snap_f = calendar(M["days"][c:c + H], keys)
        ids = np.flatnonzero(active)

        def frame(cal, snap, values=None):
            T = len(cal["date"])
            df = pd.DataFrame({"unique_id": np.repeat(ids, T), "ds": np.tile(cal["date"], len(ids))})
            for e in ("wday_sin", "wday_cos", "event", "event_tomorrow"):
                df[e] = np.tile(cal[e], len(ids)).astype(np.float32)
            df["snap"] = snap[ids].ravel()
            if values is not None:
                df["y"] = values[ids].ravel()
            return df

        train = frame(cal_h, snap_h, y)
        futr = frame(cal_f, snap_f)
        model = make_model(name, len(ids))
        # per-series scaling over the whole history
        nf = NeuralForecast(models=[model], freq="D", local_scaler_type="standard")
        nf.fit(df=train, val_size=H)
        fc = nf.predict(futr_df=futr)
        col = [c_ for c_ in fc.columns if c_ in (type(model).__name__, f"{type(model).__name__}-median")][0]
        fc = fc.sort_values(["unique_id", "ds"])
        pred = np.repeat(mean_on[:, None], H, axis=1)
        pred[ids] = np.clip(fc[col].values.reshape(len(ids), H), 0, None)
        save_preds(name, k, pred)
        print("pred quantiles 50/99/max:", np.quantile(pred, [0.5, 0.99]).round(2), round(float(pred.max()), 1), flush=True)
        if name == "deepar":  # keep the upper quantiles for the risk layer
            for q in [c_ for c_ in fc.columns if "-hi-" in c_]:
                qm = np.repeat(mean_on[:, None], H, axis=1)
                qm[ids] = np.clip(fc[q].values.reshape(len(ids), H), 0, None)
                np.save(ROOT / "data" / "preds" / f"{name}_{q.split('-', 1)[1]}_fold{k}.npy", qm)
        res, _, _ = evaluate(pred, M, c)
        rows.append({"fold": k, "model": name, **res, "sec": round(time.time() - t0)})
        print(rows[-1], flush=True)
    r = pd.DataFrame(rows)
    print(r[["WRMSSE", "MAE", "bias"]].mean().round(4).to_dict())
    return r
