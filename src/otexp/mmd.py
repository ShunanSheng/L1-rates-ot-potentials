import numpy as np


def _squared_distance_block(X, Y):
    x_norm = np.sum(X * X, axis=1)[:, None]
    y_norm = np.sum(Y * Y, axis=1)[None, :]
    return np.maximum(x_norm + y_norm - 2.0 * (X @ Y.T), 0.0)


def median_bandwidth(Y_ref, rng, max_points=2000):
    """
    Median pairwise-distance bandwidth from a reference sample.
    """
    Y_ref = np.asarray(Y_ref, dtype=float)
    m = len(Y_ref)
    if m < 2:
        return 1.0

    subset_size = min(int(max_points), m)
    if subset_size < m:
        idx = rng.choice(m, size=subset_size, replace=False)
        Y = Y_ref[idx]
    else:
        Y = Y_ref

    distances = []
    for start in range(0, len(Y), 512):
        block = Y[start:start + 512]
        d2 = _squared_distance_block(block, Y)
        row_idx = np.arange(start, start + len(block))
        mask = row_idx[:, None] < np.arange(len(Y))[None, :]
        if np.any(mask):
            distances.append(np.sqrt(d2[mask]))

    if not distances:
        return 1.0

    sigma0 = float(np.median(np.concatenate(distances)))
    return sigma0 if sigma0 > 0.0 and np.isfinite(sigma0) else 1.0


def gaussian_multiscale_kernel_sum(
    X,
    Y,
    sigma0,
    scales=(0.5, 1.0, 2.0),
    chunk_size=2048,
):
    """
    Sum k(x,y) over all pairs in X x Y for the multiscale Gaussian kernel.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    sigma0 = float(sigma0)
    if sigma0 <= 0.0:
        raise ValueError(f"sigma0 must be positive, got {sigma0}")

    total = 0.0
    scales = tuple(float(scale) for scale in scales)
    for start in range(0, len(X), chunk_size):
        Xb = X[start:start + chunk_size]
        d2 = _squared_distance_block(Xb, Y)
        block_total = 0.0
        for scale in scales:
            sigma = scale * sigma0
            block_total += np.exp(-d2 / (2.0 * sigma * sigma)).sum()
        total += block_total / len(scales)
    return float(total)


def precompute_mmd_reference(Y_ref, sigma0, chunk_size=2048):
    """
    Precompute m^{-1}(m-1)^{-1} sum_{a != b} k(Y_a,Y_b).
    """
    Y_ref = np.asarray(Y_ref, dtype=float)
    m = len(Y_ref)
    if m < 2:
        return np.nan
    total = gaussian_multiscale_kernel_sum(
        Y_ref,
        Y_ref,
        sigma0,
        chunk_size=chunk_size,
    )
    return float((total - m) / (m * (m - 1)))


def compute_mmd_unbiased(X, Y_ref, sigma0, ref_ref_term, chunk_size=2048):
    """
    Unbiased MMD^2 estimator against a fixed reference sample.
    """
    X = np.asarray(X, dtype=float)
    Y_ref = np.asarray(Y_ref, dtype=float)
    n = len(X)
    m = len(Y_ref)
    if n < 2 or m < 2:
        return np.nan

    xx_total = gaussian_multiscale_kernel_sum(X, X, sigma0, chunk_size=chunk_size)
    xy_total = gaussian_multiscale_kernel_sum(X, Y_ref, sigma0, chunk_size=chunk_size)
    xx_term = (xx_total - n) / (n * (n - 1))
    xy_term = xy_total / (n * m)
    return float(xx_term + ref_ref_term - 2.0 * xy_term)
