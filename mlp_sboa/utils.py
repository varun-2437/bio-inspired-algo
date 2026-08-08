import numpy as np


def hd(x, probs=np.arange(0, 1.0001, 0.25)):
    """A simplified Harrell-Davis quantile estimator for 1D arrays.
    Returns a dict of quantiles keyed by prob.
    """
    x = np.sort(np.asarray(x))
    n = len(x)
    m = n + 1
    res = {}
    for p in probs:
        if p <= 0:
            res[p] = x[0]
        elif p >= 1:
            res[p] = x[-1]
        else:
            a = p * m
            b = (1 - p) * m
            # approximate using weighted mean of order statistics
            weights = np.array([np.math.comb(n - 1, k) * (p ** k) * ((1 - p) ** (n - 1 - k)) for k in range(n)])
            weights = weights / weights.sum()
            res[p] = float(np.sum(weights * x))
    return res


def no(q, x):
    # Port of the no() function: simple quantile interpolation
    x = np.sort(np.asarray(x))
    n = len(x)
    if n == 1:
        return x[0]
    # use numpy percentile as fallback
    return float(np.percentile(x, q * 100))
