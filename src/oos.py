"""first versions of the out-of-stock rule, kept for check_status"""
import numpy as np


def oos_flags(sales, window=56, alpha=0.001, min_prior=14, method="v2", skip=None):
    """0/1 flags for one series"""
    sales = np.asarray(sales, dtype=float)
    n = len(sales)
    skip = np.zeros(n, dtype=bool) if skip is None else np.asarray(skip, dtype=bool)
    flag = np.zeros(n, dtype=np.int8)
    zero = (sales == 0) & ~skip
    csum = np.r_[0.0, np.cumsum(sales)]
    cpos = np.r_[0, np.cumsum(sales > 0)]
    i = 0
    while i < n:
        if not zero[i]:
            i += 1
            continue
        j = i
        while j < n and (zero[j] or skip[j]):
            j += 1
        length = int(zero[i:j].sum())
        start = max(0, i - window)
        if i - start >= min_prior:  # need history before the run to know the usual pace
            if method == "v1":
                rate = (csum[i] - csum[start]) / (i - start)
                p_chance = np.exp(-rate * length) if rate > 0 else 1.0
            else:
                p = (cpos[i] - cpos[start]) / (i - start)
                p_chance = (1 - p) ** length if p > 0 else 1.0
            if p_chance < alpha:
                flag[i:j] = zero[i:j]
        i = j
    return flag
