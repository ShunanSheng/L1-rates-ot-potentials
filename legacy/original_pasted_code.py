import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import qmc, norm


# ============================================================
# Theoretical rate beta(n,d)
# ============================================================

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
    elif d == 4:
        return n ** (-0.5) * np.log(n) ** 2.5
    else:
        return n ** (-2.0 / d) * np.log(n) ** ((d + 2.0) / 4.0)


# ============================================================
# Sampling from the unit ball B_d = {x : ||x|| <= 1}
# ============================================================

def sample_unit_ball_iid(M, d, rng):
    """
    IID samples from the uniform distribution on the unit ball in R^d.

    Method:
        Direction: Z / ||Z||, with Z ~ N(0,I_d).
        Radius: U^{1/d}, with U ~ Unif(0,1).

    Then R * Direction is uniform on the unit ball.
    """
    Z = rng.normal(size=(M, d))
    Z_norm = np.linalg.norm(Z, axis=1, keepdims=True)
    directions = Z / Z_norm

    radii = rng.random(M) ** (1.0 / d)
    X = radii[:, None] * directions

    return X


def sample_unit_ball_qmc(M, d, seed=123):
    """
    Quasi-Monte Carlo source cloud approximately uniform on the unit ball.

    For d=2, this is an exact Sobol polar transform:
        r = sqrt(u_1), theta = 2*pi*u_2.

    For d>=3, we use Sobol points transformed through Gaussian inverse CDF
    to generate directions, and one Sobol coordinate for the radius.
    """
    m_power = int(np.ceil(np.log2(M)))
    sampler = qmc.Sobol(d=d + 1, scramble=True, seed=seed)
    U = sampler.random_base2(m_power)[:M]
    U = np.clip(U, 1e-12, 1 - 1e-12)

    if d == 1:
        # Unit ball in R is [-1,1].
        return 2.0 * U[:, :1] - 1.0

    if d == 2:
        r = np.sqrt(U[:, 0])
        theta = 2.0 * np.pi * U[:, 1]
        X = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        return X

    # d >= 3: QMC-based approximate spherical direction.
    radii = U[:, 0] ** (1.0 / d)

    Z = norm.ppf(U[:, 1:d + 1])
    Z_norm = np.linalg.norm(Z, axis=1, keepdims=True)
    directions = Z / Z_norm

    X = radii[:, None] * directions
    return X


def sample_mu(M, d, rng=None, seed=123, use_qmc=True):
    """
    Source measure:
        mu = Unif(B_d).

    For the source cloud, QMC is preferred to reduce integration noise.
    """
    if use_qmc:
        return sample_unit_ball_qmc(M, d, seed=seed)

    if rng is None:
        rng = np.random.default_rng(seed)
    return sample_unit_ball_iid(M, d, rng)


def sample_nu(n, d, rng):
    """
    Target measure:
        nu = mu = Unif(B_d),

    because we take
        phi(x) = 0.5 ||x||^2,
        grad phi(x) = x.
    """
    return sample_unit_ball_iid(n, d, rng)


def phi_true(X):
    """
    True Brenier potential for the identity map:
        phi(x) = 0.5 ||x||^2.
    """
    return 0.5 * np.sum(X * X, axis=1)


# ============================================================
# Semi-discrete OT dual solver
# ============================================================

def dual_obj_grad(theta, X, Y, chunk_size=2048):
    """
    Objective and gradient for the discretized semi-discrete OT dual.

    We approximate mu by the source cloud X_1,...,X_M.
    The empirical target is
        nu_hat_n = (1/n) sum_i delta_{Y_i}.

    The empirical potential has the form
        phi_hat(x) = max_i { x . Y_i - 0.5 ||Y_i||^2 + h_i }.

    The weights h_i are chosen by minimizing

        (1/M) sum_m max_i {X_m . Y_i - 0.5 ||Y_i||^2 + h_i}
        - (1/n) sum_i h_i.

    Since h is only identifiable up to constants, we center theta.
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

    # Gradient = cell masses - target masses.
    grad_h = counts / M - 1.0 / n

    # Project onto mean-zero subspace.
    grad = grad_h - grad_h.mean()

    return obj, grad


def solve_weights(X, Y, max_iter=180):
    """
    Solve for the semi-discrete dual weights h_i.
    """
    n = Y.shape[0]
    theta0 = np.zeros(n)

    res = minimize(
        lambda th: dual_obj_grad(th, X, Y),
        theta0,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": max_iter,
            "gtol": 1e-6,
            "ftol": 1e-10,
            "maxls": 30,
        },
    )

    h = res.x - res.x.mean()
    return h, res


def evaluate_phi_hat(X, Y, h, chunk_size=2048):
    """
    Evaluate
        phi_hat(x) = max_i { x . Y_i - 0.5 ||Y_i||^2 + h_i }
    on the cloud X.
    """
    M = X.shape[0]
    y_norm = 0.5 * np.sum(Y * Y, axis=1)

    out = np.empty(M)

    for start in range(0, M, chunk_size):
        Xb = X[start:start + chunk_size]
        scores = Xb @ Y.T - y_norm[None, :] + h[None, :]
        out[start:start + len(Xb)] = np.max(scores, axis=1)

    return out


# ============================================================
# One Monte Carlo trial
# ============================================================

def one_trial(n, d, X_source, rng):
    """
    One repetition.

    Computes approximately

        inf_a || phi_hat_n - phi - a ||_{L^1(mu)}

    with mu = Unif(B_d), phi(x)=0.5||x||^2.
    """
    Y = sample_nu(n, d, rng)

    h, res = solve_weights(X_source, Y)

    phi_hat = evaluate_phi_hat(X_source, Y, h)
    phi = phi_true(X_source)

    diff = phi_hat - phi

    # For L1, the optimal additive shift is a median.
    a_star = np.median(diff)

    loss = np.mean(np.abs(diff - a_star))

    return {
        "loss": loss,
        "success": res.success,
        "grad_inf": np.max(np.abs(res.jac)),
    }


# ============================================================
# Small-sample experiment
# ============================================================

def run_ball_experiment(
    d=2,
    n_target=(4, 8, 16, 32, 64),
    B=8,
    n_source=4096,
    seed=2026,
    use_qmc_source=True,
):
    """
    Small sample experiment for mu = Unif(B_d).
    """
    rng = np.random.default_rng(seed)

    X_source = sample_mu(
        n_source,
        d,
        rng=rng,
        seed=seed + 100,
        use_qmc=use_qmc_source,
    )

    rows = []

    for n in n_target:
        losses = []
        successes = []
        grad_infs = []

        for _ in range(B):
            trial_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))

            out = one_trial(n, d, X_source, trial_rng)

            losses.append(out["loss"])
            successes.append(out["success"])
            grad_infs.append(out["grad_inf"])

        losses = np.asarray(losses)

        rows.append({
            "d": d,
            "n": n,
            "mean_loss": losses.mean(),
            "std_loss": losses.std(ddof=1),
            "se_loss": losses.std(ddof=1) / np.sqrt(B),
            "beta": beta_rate(n, d),
            "mean_loss_over_beta": losses.mean() / beta_rate(n, d),
            "success_rate": np.mean(successes),
            "median_grad_inf": np.median(grad_infs),
        })

    df = pd.DataFrame(rows)

    slope_loss, intercept_loss = np.polyfit(
        np.log(df["n"]),
        np.log(df["mean_loss"]),
        deg=1,
    )

    slope_beta, intercept_beta = np.polyfit(
        np.log(df["n"]),
        np.log(df["beta"]),
        deg=1,
    )

    return df, slope_loss, slope_beta


# ============================================================
# Plotting
# ============================================================

def plot_loglog_comparison(df, save_path="small_n_loglog_ot_potential_unit_ball_d2.png"):
    """
    Log-log comparison of the empirical loss with a fitted multiple of beta(n,d).
    """
    d = int(df["d"].iloc[0])

    n = df["n"].to_numpy()
    mean_loss = df["mean_loss"].to_numpy()
    se_loss = df["se_loss"].to_numpy()
    beta = df["beta"].to_numpy()

    # Best multiplicative constant C for C * beta(n,d).
    C_hat = np.sum(mean_loss * beta) / np.sum(beta * beta)

    plt.figure(figsize=(7, 5))

    plt.errorbar(
        n,
        mean_loss,
        yerr=1.96 * se_loss,
        fmt="o-",
        capsize=3,
        label="empirical mean loss",
    )

    plt.plot(
        n,
        C_hat * beta,
        "--",
        label=rf"best fitted $C\beta(n,{d})$",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$n$")
    plt.ylabel(
        r"$\widehat{\mathbb{E}}[\inf_a\|\widehat\varphi_n-\varphi-a\|_{L^1(\mu)}]$"
    )
    plt.title(rf"Small-$n$ log-log comparison on unit ball, $d={d}$")
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, dpi=200)
    plt.show()

    return C_hat


# ============================================================
# Main script
# ============================================================

if __name__ == "__main__":
    d = 2

    n_target = (256, 512, 1024, 2048, 4096)


    df, slope_loss, slope_beta = run_ball_experiment(
        d=d,
        n_target=n_target,
        B= 100,
        n_source= 4096,
        seed=2026,
        use_qmc_source=True,
    )

    print(df)
    print()
    print(f"Empirical log-log slope for mean loss: {slope_loss:.3f}")
    print(f"Log-log slope of beta(n,{d}): {slope_beta:.3f}")

    C_hat = plot_loglog_comparison(
        df,
        save_path=f"figs/small_n_loglog_ot_potential_unit_ball_d{d}.png",
    )

    print(f"Best fitted multiplicative constant C: {C_hat:.4f}")