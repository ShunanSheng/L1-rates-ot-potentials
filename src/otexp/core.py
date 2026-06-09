import numpy as np
from scipy.optimize import minimize

from .sampling import sample_nu, phi_true


def dual_obj_grad(theta, X, Y, chunk_size=2048):
    """
    Objective and gradient for the discretized semi-discrete OT dual.

    We approximate mu by the source cloud X_1,...,X_M. The empirical target is
    nu_hat_n = n^{-1} sum_i delta_{Y_i}. The empirical potential is

        phi_hat(x) = max_i { x . Y_i - 0.5 ||Y_i||^2 + h_i }.

    The weights h_i are chosen by minimizing

        M^{-1} sum_m max_i {X_m . Y_i - 0.5 ||Y_i||^2 + h_i}
        - n^{-1} sum_i h_i.

    Since h is identifiable only up to constants, theta is centered.
    """
    M = X.shape[0]
    n = Y.shape[0]

    h = theta - theta.mean()
    y_norm = 0.5 * np.sum(Y * Y, axis=1)

    total = 0.0
    counts = np.zeros(n)

    for start in range(0, M, chunk_size):
        Xb = X[start:start + chunk_size]
        scores = Xb @ Y.T - y_norm[None, :] + h[None, :]
        arg = np.argmax(scores, axis=1)
        total += np.max(scores, axis=1).sum()
        counts += np.bincount(arg, minlength=n)

    obj = total / M - h.mean()

    grad_h = counts / M - 1.0 / n
    grad = grad_h - grad_h.mean()
    return obj, grad


def solve_weights(X, Y, max_iter=1000, chunk_size=2048, gtol=1e-5, ftol=1e-10):
    """
    Solve for the semi-discrete dual weights h_i.
    """
    n = Y.shape[0]
    theta0 = np.zeros(n)

    res = minimize(
        lambda th: dual_obj_grad(th, X, Y, chunk_size=chunk_size),
        theta0,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": max_iter,
            "gtol": gtol,
            "ftol": ftol,
            "maxls": 30,
        },
    )

    h = res.x - res.x.mean()
    return h, res


def evaluate_phi_hat(X, Y, h, chunk_size=2048):
    """
    Evaluate phi_hat(x)=max_i{x.Y_i - 0.5||Y_i||^2 + h_i} on X.
    """
    M = X.shape[0]
    y_norm = 0.5 * np.sum(Y * Y, axis=1)
    out = np.empty(M)

    for start in range(0, M, chunk_size):
        Xb = X[start:start + chunk_size]
        scores = Xb @ Y.T - y_norm[None, :] + h[None, :]
        out[start:start + len(Xb)] = np.max(scores, axis=1)

    return out


def one_trial(
    n,
    d,
    X_solve,
    rng=None,
    max_iter=180,
    chunk_size=2048,
    X_eval=None,
):
    """
    One Monte Carlo repetition.

    Computes approximately

        inf_a || phi_hat_n - phi - a ||_{L1(mu)}

    with mu = Unif(B_d), phi(x)=0.5||x||^2.
    """
    if X_eval is None:
        X_eval = X_solve

    Y = sample_nu(n, d, rng)
    h, res = solve_weights(
        X_solve,
        Y,
        max_iter=max_iter,
        chunk_size=chunk_size,
    )

    phi_hat = evaluate_phi_hat(X_eval, Y, h, chunk_size=chunk_size)
    phi = phi_true(X_eval)

    diff = phi_hat - phi

    # For L1, the optimal additive shift is a median.
    a_star = np.median(diff)
    loss = np.mean(np.abs(diff - a_star))

    return {
        "loss": loss,
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "nit": int(res.nit),
        "nfev": int(res.nfev),
        "njev": int(res.njev),
        "fun": float(res.fun),
        "grad_inf": float(np.max(np.abs(res.jac))),
    }
