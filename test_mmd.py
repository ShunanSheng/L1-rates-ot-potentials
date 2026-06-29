#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from otexp.mmd import (
    compute_mmd_unbiased,
    gaussian_multiscale_kernel_sum,
    median_bandwidth,
    precompute_mmd_reference,
)


def explicit_kernel(X, Y, sigma0, scales=(0.5, 1.0, 2.0)):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    d2 = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    K = np.zeros_like(d2)
    for scale in scales:
        sigma = scale * sigma0
        K += np.exp(-d2 / (2.0 * sigma * sigma))
    return K / len(scales)


def explicit_mmd_unbiased(X, Y, sigma0):
    Kxx = explicit_kernel(X, X, sigma0)
    Kyy = explicit_kernel(Y, Y, sigma0)
    Kxy = explicit_kernel(X, Y, sigma0)
    n = len(X)
    m = len(Y)
    xx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    yy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    xy = Kxy.mean()
    return float(xx + yy - 2.0 * xy)


def test_kernel_sum_matches_explicit_matrix():
    X = np.array([[0.0], [1.0], [3.0]])
    Y = np.array([[0.5], [2.0], [4.0], [5.0]])
    sigma0 = 1.3

    expected = float(explicit_kernel(X, Y, sigma0).sum())
    actual = gaussian_multiscale_kernel_sum(X, Y, sigma0, chunk_size=2)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    print("kernel sum")
    print(f"  expected: {expected:.12g}")
    print(f"  actual:   {actual:.12g}")


def test_precomputed_reference_term_matches_explicit_matrix():
    Y = np.array([[0.5], [2.0], [4.0], [5.0]])
    sigma0 = 1.3
    Kyy = explicit_kernel(Y, Y, sigma0)
    expected = float((Kyy.sum() - np.trace(Kyy)) / (len(Y) * (len(Y) - 1)))
    actual = precompute_mmd_reference(Y, sigma0, chunk_size=2)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    print("reference term")
    print(f"  expected: {expected:.12g}")
    print(f"  actual:   {actual:.12g}")


def test_mmd_unbiased_matches_explicit_formula():
    X = np.array([[0.0], [1.0], [3.0]])
    Y = np.array([[0.5], [2.0], [4.0], [5.0]])
    sigma0 = 1.3
    ref_ref_term = precompute_mmd_reference(Y, sigma0, chunk_size=2)

    expected = explicit_mmd_unbiased(X, Y, sigma0)
    actual = compute_mmd_unbiased(
        X,
        Y,
        sigma0=sigma0,
        ref_ref_term=ref_ref_term,
        chunk_size=2,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    print("unbiased MMD")
    print(f"  expected: {expected:.12g}")
    print(f"  actual:   {actual:.12g}")


def test_mmd_chunk_size_invariance():
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(7, 3))
    Y = rng.normal(size=(9, 3))
    sigma0 = 1.1
    ref_ref_term = precompute_mmd_reference(Y, sigma0, chunk_size=1)

    values = [
        compute_mmd_unbiased(X, Y, sigma0, ref_ref_term, chunk_size=chunk_size)
        for chunk_size in (1, 2, 4, 100)
    ]

    np.testing.assert_allclose(values, values[0], rtol=0.0, atol=1e-12)
    print("chunk invariance")
    print(f"  values: {[round(value, 12) for value in values]}")


def test_unbiased_self_mmd_is_not_forced_to_zero():
    X = np.array([[0.0], [1.0], [3.0], [4.0]])
    sigma0 = 1.3
    ref_ref_term = precompute_mmd_reference(X, sigma0, chunk_size=2)
    actual = compute_mmd_unbiased(X, X, sigma0, ref_ref_term, chunk_size=2)
    expected = explicit_mmd_unbiased(X, X, sigma0)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    if np.isclose(actual, 0.0):
        raise AssertionError("This deterministic example should show that unbiased self-MMD is not forced to zero.")

    print("self unbiased MMD")
    print(f"  value: {actual:.12g}")


def test_median_bandwidth_is_deterministic_without_subsampling():
    rng = np.random.default_rng(2026)
    Y = np.array([[0.0], [1.0], [3.0], [4.0]])
    sigma0 = median_bandwidth(Y, rng, max_points=10)

    distances = []
    for i in range(len(Y)):
        for j in range(i + 1, len(Y)):
            distances.append(abs(float(Y[i, 0] - Y[j, 0])))
    expected = float(np.median(distances))

    np.testing.assert_allclose(sigma0, expected, rtol=0.0, atol=1e-12)
    print("median bandwidth")
    print(f"  expected: {expected:.12g}")
    print(f"  actual:   {sigma0:.12g}")


if __name__ == "__main__":
    test_kernel_sum_matches_explicit_matrix()
    test_precomputed_reference_term_matches_explicit_matrix()
    test_mmd_unbiased_matches_explicit_formula()
    test_mmd_chunk_size_invariance()
    test_unbiased_self_mmd_is_not_forced_to_zero()
    test_median_bandwidth_is_deterministic_without_subsampling()
    print("test_mmd.py passed.")
