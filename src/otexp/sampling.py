import numpy as np
from scipy.stats import qmc, norm


def sample_unit_ball_iid(M, d, rng):
    """
    IID samples from the uniform distribution on the unit ball in R^d.
    """
    if d == 1:
        return rng.uniform(-1.0, 1.0, size=(M, 1))

    Z = rng.normal(size=(M, d))
    Z_norm = np.linalg.norm(Z, axis=1, keepdims=True)
    directions = Z / Z_norm

    radii = rng.random(M) ** (1.0 / d)
    return radii[:, None] * directions


def sample_unit_ball_qmc(M, d, seed=123):
    """
    Quasi-Monte Carlo source cloud approximately uniform on the unit ball.

    For d=2, this uses a Sobol polar transform. For d>=3, it uses Sobol
    points transformed through Gaussian inverse CDF to generate directions,
    and one Sobol coordinate for the radius.
    """
    m_power = int(np.ceil(np.log2(M)))
    sampler = qmc.Sobol(d=d + 1, scramble=True, seed=seed)
    U = sampler.random_base2(m_power)[:M]
    U = np.clip(U, 1e-12, 1 - 1e-12)

    if d == 1:
        return 2.0 * U[:, :1] - 1.0

    if d == 2:
        r = np.sqrt(U[:, 0])
        theta = 2.0 * np.pi * U[:, 1]
        return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    radii = U[:, 0] ** (1.0 / d)
    Z = norm.ppf(U[:, 1:d + 1])
    Z_norm = np.linalg.norm(Z, axis=1, keepdims=True)
    directions = Z / Z_norm
    return radii[:, None] * directions


def sample_mu(M, d, rng=None, seed=123, use_qmc=True):
    """
    Source measure mu = Unif(B_d).
    """
    if use_qmc:
        return sample_unit_ball_qmc(M, d, seed=seed)

    if rng is None:
        rng = np.random.default_rng(seed)
    return sample_unit_ball_iid(M, d, rng)


def sample_nu(n, d, rng):
    """
    Target measure nu = mu = Unif(B_d), corresponding to phi(x)=0.5||x||^2.
    """
    return sample_unit_ball_iid(n, d, rng)


def phi_true(X):
    """
    True Brenier potential for the identity map: phi(x)=0.5||x||^2.
    """
    return 0.5 * np.sum(X * X, axis=1)
