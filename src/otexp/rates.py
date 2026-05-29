import numpy as np


def beta_rate(n, d):
    """
    beta(n,d), up to multiplicative constants:
        n^{-1/2},                         d=1,2,3
        n^{-1/2} log(n)^{5/2},             d=4
        n^{-2/d} log(n)^{(d+2)/4},         d>=5
    """
    n = np.asarray(n, dtype=float)

    if d in (1, 2, 3):
        return n ** (-0.5)
    if d == 4:
        return n ** (-0.5) * np.log(n) ** 2.5
    return n ** (-2.0 / d) * np.log(n) ** ((d + 2.0) / 4.0)
