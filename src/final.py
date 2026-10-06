"""one-shot evaluation on the hold-out"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cv import H, evaluate, status_matrix
from features import build, feature_columns
from common import CAT_COLS, START_D, TRAIN_DAYS
from models.lgbm import PARAMS as LGB_PARAMS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "final"
OUT.mkdir(exist_ok=True)
LOG = []


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)


def matrices(df):
    key = df["store_id"].astype(str) + "|" + df["item_id"].astype(str)
    days = np.sort(df["d"].unique())
    keys = np.asarray(key.unique(), dtype=object)
    n, m = len(keys), len(days)
    assert len(df) == n * m
    return dict(S=df["sales"].values.reshape(n, m).astype(np.float32),
                P=df["sell_price"].values.reshape(n, m).astype(np.float32),
                L=df["launched"].values.reshape(n, m), days=days, keys=keys,
                cat=np.asarray(df["cat_id"].astype(str).to_numpy(dtype=object)).reshape(n, m)[:, 0])


def main():
    cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "d", "date", "sales", "sell_price", "launched",
            "wm_yr_wk", "weekday", "month", "year", "event_name_1", "event_type_1", "snap"]
    dev = pd.read_parquet(ROOT / "data" / "dev.parquet", columns=cols)
    hold = pd.read_parquet(ROOT / "data" / "holdout.parquet", columns=cols)
    full = pd.concat([dev, hold]).sort_values(["store_id", "item_id", "d"], kind="stable").reset_index(drop=True)
    for c_ in ("item_id", "dept_id", "cat_id", "store_id", "state_id"):
        full[c_] = full[c_].astype(dev[c_].dtype)
    M = matrices(full)
    c = int(np.searchsorted(M["days"], hold["d"].min()))
    assert M["days"][c] == hold["d"].min() and len(M["days"]) == c + H
    log(f"cutoff: train <= d_{M['days'][c - 1]}, hold-out d_{M['days'][c]}..d_{M['days'][-1]}")

    masked = full.copy()
    masked.loc[masked["d"] > M["days"][c - 1], "sales"] = 0  # hold-out sales invisible to features
    t0 = time.time()
    t = build(masked, START_D)
    del masked
    log(f"features: {t.shape} in {time.time() - t0:.0f}s")

    n = len(M["keys"])
    first_col = int(np.searchsorted(M["days"], START_D))
    per = len(M["days"]) - first_col
    assert len(t) == n * per
    col = first_col + np.tile(np.arange(per), n)
    ser = np.repeat(np.arange(n), per)
    st = status_matrix(M["S"], M["L"], c)
    train = (col < c) & (col >= c - TRAIN_DAYS)
    train[train] = st[ser[train], col[train]] == 0
    val = col >= c
    y_all = t["sales"].values.astype(np.float32)
    d = t["d"].values
    inner_cut = M["days"][c - 1 - H]
    preds = {}

    # --- LightGBM (with and without item_id)
    import lightgbm as lgb
    for tag, drop in (("lgbm", ()), ("lgbm_noitem", ("item_id",))):
        t1 = time.time()
        feats = [f for f in feature_columns(t) if f not in drop]
        cats = [f for f in CAT_COLS if f not in drop]
        fit, stop = train & (d <= inner_cut), train & (d > inner_cut)
        cap = np.quantile(y_all[train], 0.999)
        dtr = lgb.Dataset(t.loc[fit, feats], np.minimum(y_all[fit], cap), categorical_feature=cats)
        dst = lgb.Dataset(t.loc[stop, feats], np.minimum(y_all[stop], cap), reference=dtr)
        model = lgb.train(LGB_PARAMS, dtr, 3000, valid_sets=[dst],
                          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        p = model.predict(t.loc[val, feats], num_iteration=model.best_iteration).reshape(n, H)
        preds[tag] = p
        np.save(OUT / f"{tag}.npy", p)
        model.save_model(str(OUT / f"{tag}.txt"))
        log(f"{tag}: {model.best_iteration} trees, {time.time() - t1:.0f}s")
    del t

    # --- N-HiTS on GPU 1 (same settings as CV)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    from neuralforecast import NeuralForecast
    from models.neural import HIST, calendar, make_model
    t1 = time.time()
    stw = st[:, c - HIST:c]
    yh = M["S"][:, c - HIST:c].astype(np.float32)
    on = stw == 0
    mean_on = np.where(on.sum(1) > 0, np.where(on, yh, 0).sum(1) / np.maximum(on.sum(1), 1), 0).astype(np.float32)
    yh = np.where(on, yh, mean_on[:, None])
    ids = np.flatnonzero(on.sum(1) >= 56)
    cal_h, snap_h = calendar(M["days"][c - HIST:c], M["keys"])
    cal_f, snap_f = calendar(M["days"][c:c + H], M["keys"])

    def frame(cal, snap, values=None):
        T = len(cal["date"])
        df = pd.DataFrame({"unique_id": np.repeat(ids, T), "ds": np.tile(cal["date"], len(ids))})
        for e in ("wday_sin", "wday_cos", "event", "event_tomorrow"):
            df[e] = np.tile(cal[e], len(ids)).astype(np.float32)
        df["snap"] = snap[ids].ravel()
        if values is not None:
            df["y"] = values[ids].ravel()
        return df

    nf = NeuralForecast(models=[make_model("nhits", len(ids))], freq="D", local_scaler_type="standard")
    nf.fit(df=frame(cal_h, snap_h, yh), val_size=H)
    fc = nf.predict(futr_df=frame(cal_f, snap_f)).sort_values(["unique_id", "ds"])
    p = np.repeat(mean_on[:, None], H, axis=1)
    p[ids] = np.clip(fc["NHITS"].values.reshape(len(ids), H), 0, None)
    preds["nhits"] = p
    np.save(OUT / "nhits.npy", p)
    log(f"nhits: {time.time() - t1:.0f}s, pred max {p.max():.1f}")

    # --- rules (planks)
    from models.rules import b_croston_sba, b_mean_dow_profile, instock_hist
    hist, _ = instock_hist(M, c)
    preds["croston_sba"] = b_croston_sba(M, c, hist)
    preds["mean28_x_profile"] = b_mean_dow_profile(M, c, hist)
    preds["lgbm+nhits"] = 0.5 * preds["lgbm"] + 0.5 * preds["nhits"]

    rows = []
    for name, p in preds.items():
        res, _, _ = evaluate(p, M, c)
        rows.append({"model": name, **res})
    r = pd.DataFrame(rows).sort_values("WRMSSE")
    r.to_csv(ROOT / "reports" / "holdout_results.csv", index=False)
    log("\nHOLD-OUT RESULTS\n" + r[["model", "WRMSSE", "RMSSE_mean", "MAE", "bias"]].round(4).to_string(index=False))

    # --- intervals and inventory with the CV-fitted rules
    import inventory
    import uncertainty
    mu = preds["lgbm+nhits"]
    rr = uncertainty.dispersion(M, c)
    cov = uncertainty.coverage(M, c, mu, rr, inventory.K_WEEK)
    log("coverage (share of actual <= quantile): " + ", ".join(f"{a} q{int(b * 100)}={v:.3f}" for (a, b), v in cov.items()))
    sim = []
    for L in (3, 7):
        for q in (0.8, 0.9, 0.95):
            sim.append({**inventory.simulate(M, c, mu, rr, L, q=q), "param": f"q={q}"})
        for m in (1.2, 1.5, 2.0, 2.5):
            sim.append({**inventory.simulate(M, c, mu, rr, L, policy="naive", mult=m), "param": f"x{m}"})
    s = pd.DataFrame(sim)[["lead_time", "policy", "param", "fill_rate", "empty_shelf_share", "avg_stock_value_$"]]
    s.to_csv(ROOT / "reports" / "holdout_inventory.csv", index=False)
    log("\nINVENTORY ON HOLD-OUT\n" + s.round(3).to_string(index=False))
    (ROOT / "reports" / "holdout_log.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
