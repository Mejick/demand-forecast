"""sanity checks for the status rule"""
import numpy as np
import pandas as pd

from oos import oos_flags
from status import ALPHA, DELISTED, MIN_PRIOR, NOT_LAUNCHED, OOS, WINDOW, build, compute_status

df, sid = build()
s, st, la = df["sales"].values, df["status"].values, df["launched"].values
names = {0: "on sale", 1: "out of stock", 2: "delisted", 3: "not launched"}
print("== status shares (all rows / launched rows)")
for k in range(4):
    print(f"  {names[k]:13s} {(st == k).mean():7.2%} {(st[la] == k).mean():7.2%}")
any_by = lambda m: int(pd.Series(m).groupby(sid).any().sum())
print(f"series with out-of-stock: {any_by(st == OOS):,}; delisted: {any_by(st == DELISTED):,}; "
      f"with pre-launch days: {any_by(st == NOT_LAUNCHED):,} of {len(np.unique(sid)):,}")

print("\n== 1. vectorized rule == loop version on launched days (200 random series)")
rng = np.random.default_rng(0)
bad = 0
for k in rng.choice(np.unique(sid), 200, replace=False):
    m = (sid == k) & la
    loop = oos_flags(s[m], window=WINDOW, alpha=ALPHA, min_prior=MIN_PRIOR, method="v2")
    bad += int((loop != ((st[m] == OOS) | (st[m] == DELISTED))).any())
print(f"series with any mismatch: {bad}")

print("\n== 2. invariants")
print("not launched == launched flag is False:", bool(((st == NOT_LAUNCHED) == ~la).all()))
print("sales on not-launched days:", int(s[st == NOT_LAUNCHED].sum()))
prev_same = np.r_[False, sid[1:] == sid[:-1]]
print("launched -> not launched inside a series:", int((prev_same & np.r_[False, la[:-1]] & ~la).sum()))
print("sales on out-of-stock / delisted days:", int(s[(st == OOS) | (st == DELISTED)].sum()))
last = np.r_[sid[1:] != sid[:-1], True]
dl_start = (st == DELISTED) & ~(prev_same & np.r_[False, st[:-1] == DELISTED])
dl_end = (st == DELISTED) & ~np.r_[(sid[1:] == sid[:-1]) & (st[1:] == DELISTED), False]
print("delisted blocks:", int(dl_start.sum()), "| ending at series end:", int((dl_end & last).sum()),
      "| max per series:", int(pd.Series(dl_start).groupby(sid).sum().max()))

print("\n== 3. false positives: launched sales shuffled in time within each series")
s2 = s.copy()
idx = np.flatnonzero(la)
order = idx[np.lexsort((rng.random(len(idx)), sid[idx]))]
s2[idx] = s[order]
st2 = compute_status(s2, sid, la)
print(f"flagged after shuffle: {(st2[la] % 3 >= 1).mean():.3%} of launched days "
      f"(real: {((st[la] == OOS) | (st[la] == DELISTED)).mean():.2%})")
