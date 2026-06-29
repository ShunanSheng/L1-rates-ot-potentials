#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from otexp.core import (
    evaluate_phi_hat_and_assignment,
    evaluate_w2_from_assignment,
    solve_weights,
)
from otexp.sampling import sample_unit_ball_iid


def exact_empirical_w2_squared(X, Y):
    """
    Exact W2^2 between equally weighted empirical measures with equal support
    sizes, computed by the Hungarian algorithm.
    """
    cost = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].mean()), col_ind


def test_optimal_assignment_matches_true_w2_squared():
    X = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    Y = np.array([
        [0.1, 0.0],
        [1.2, 0.0],
        [0.0, 0.7],
    ])

    expected_w2_sq, optimal_assignment = exact_empirical_w2_squared(X, Y)
    actual_w2_sq = evaluate_w2_from_assignment(X, Y, optimal_assignment)

    np.testing.assert_allclose(actual_w2_sq, expected_w2_sq, rtol=0.0, atol=1e-12)
    print("optimal assignment")
    print(f"  assignment: {optimal_assignment.tolist()}")
    print(f"  true W2^2:  {expected_w2_sq:.12g}")
    print(f"  helper W2^2:{actual_w2_sq:.12g}")
    print(f"  true W2:    {np.sqrt(expected_w2_sq):.12g}")


def test_nonoptimal_assignment_does_not_claim_true_w2_squared():
    X = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    Y = np.array([
        [0.1, 0.0],
        [1.2, 0.0],
        [0.0, 0.7],
    ])

    true_w2_sq, optimal_assignment = exact_empirical_w2_squared(X, Y)
    bad_assignment = np.array([1, 2, 0])
    bad_assignment_w2_sq = evaluate_w2_from_assignment(X, Y, bad_assignment)

    if np.isclose(bad_assignment_w2_sq, true_w2_sq):
        raise AssertionError("Non-optimal assignment unexpectedly matched true W2^2.")

    print("non-optimal assignment")
    print(f"  optimal assignment: {optimal_assignment.tolist()}")
    print(f"  bad assignment:     {bad_assignment.tolist()}")
    print(f"  true W2^2:          {true_w2_sq:.12g}")
    print(f"  bad assignment cost:{bad_assignment_w2_sq:.12g}")


def test_solved_semidiscrete_assignment_matches_true_w2_squared():
    """
    Solve the semi-discrete OT dual, use the fitted potential to assign each
    source point to a target point, and compare that induced cost to exact W2^2.

    This test uses 1000 IID source points from the unit ball and a small
    perturbation of that cloud as the target. The exact empirical W2 value is
    computed by the Hungarian algorithm, and the fitted semi-discrete potential
    should induce the same optimal assignment.
    """
    rng = np.random.default_rng(2026)
    n_source = 1000
    d = 3
    X = sample_unit_ball_iid(n_source, d, rng)
    Y = X.copy() + rng.standard_normal((n_source, d)) * 1e-2

    true_w2_sq, optimal_assignment = exact_empirical_w2_squared(X, Y)
    h, res = solve_weights(
        X,
        Y,
        max_iter=500,
        chunk_size=250,
        gtol=1e-10,
        ftol=1e-12,
    )
    _, solved_assignment = evaluate_phi_hat_and_assignment(X, Y, h, chunk_size=256)
    solved_w2_sq = evaluate_w2_from_assignment(X, Y, solved_assignment)

    if not res.success:
        raise AssertionError(f"semi-discrete solve failed: {res.message}")

    np.testing.assert_array_equal(solved_assignment, optimal_assignment)
    np.testing.assert_allclose(solved_w2_sq, true_w2_sq, rtol=0.0, atol=1e-10)

    print("semi-discrete solve")
    print(f"  n_source:           {n_source}")
    print(f"  dimension:          {d}")
    print(f"  solver success:     {res.success}")
    print(f"  matches exact opt.: {bool(np.array_equal(solved_assignment, optimal_assignment))}")
    print(f"  true W2^2:          {true_w2_sq:.12g}")
    print(f"  solved W2^2:        {solved_w2_sq:.12g}")
    print(f"  true W2:            {np.sqrt(true_w2_sq):.12g}")


if __name__ == "__main__":
    test_optimal_assignment_matches_true_w2_squared()
    test_nonoptimal_assignment_does_not_claim_true_w2_squared()
    test_solved_semidiscrete_assignment_matches_true_w2_squared()
    print("All W2 assignment checks passed.")
