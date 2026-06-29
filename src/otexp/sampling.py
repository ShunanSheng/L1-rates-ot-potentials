import numpy as np
from scipy.stats import qmc, norm


GOF_NULLS = ("uniform_ball", "truncated_gaussian", "truncated_elliptical_t")
DEFAULT_T_DF = 5
DEFAULT_T_SIGMA = np.diag([1.0, 1.5**2, 0.7**2])


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


def _collect_rejection_samples(M, sampler, accept, d):
    samples = []
    remaining = int(M)
    batch_size = max(1024, int(np.ceil(1.5 * M)))

    while remaining > 0:
        candidates = sampler(max(batch_size, remaining))
        accepted = candidates[accept(candidates)]
        if len(accepted):
            samples.append(accepted[:remaining])
            remaining -= min(remaining, len(accepted))
        batch_size = max(batch_size, 2 * remaining)

    if not samples:
        return np.empty((0, d))
    return np.vstack(samples)[:M]


def sample_truncated_gaussian(M, d, radius=2.0, rng=None):
    """
    IID samples from N_d(0,I) conditioned on ||Z|| <= radius.
    """
    if rng is None:
        rng = np.random.default_rng()

    def sampler(size):
        return rng.normal(size=(size, d))

    radius_sq = float(radius) ** 2
    return _collect_rejection_samples(
        M,
        sampler,
        lambda X: np.sum(X * X, axis=1) <= radius_sq,
        d,
    )


def sample_truncated_elliptical_t(M, d, radius=2.0, nu=DEFAULT_T_DF, Sigma=None, rng=None):
    """
    IID samples from a centered elliptical t distribution truncated to
    Y^T Sigma^{-1} Y <= radius^2.
    """
    if rng is None:
        rng = np.random.default_rng()
    if Sigma is None:
        Sigma = DEFAULT_T_SIGMA[:d, :d]
    Sigma = np.asarray(Sigma, dtype=float)
    chol = np.linalg.cholesky(Sigma)
    inv_sigma = np.linalg.inv(Sigma)
    radius_sq = float(radius) ** 2

    def sampler(size):
        Z = rng.normal(size=(size, d)) @ chol.T
        G = rng.chisquare(float(nu), size=size)
        return Z / np.sqrt(G[:, None] / float(nu))

    def accept(X):
        q = np.einsum("ij,jk,ik->i", X, inv_sigma, X)
        return q <= radius_sq

    return _collect_rejection_samples(M, sampler, accept, d)


def sample_null(null_name, M, rng=None, use_qmc=False, seed=None):
    """
    IID samples from a supported GOF null distribution in dimension 3.
    """
    if null_name not in GOF_NULLS:
        raise ValueError(f"Unknown null distribution: {null_name}")
    if rng is None:
        rng = np.random.default_rng(seed)

    d = 3
    if null_name == "uniform_ball":
        if use_qmc:
            return sample_unit_ball_qmc(M, d, seed=123 if seed is None else seed)
        return sample_unit_ball_iid(M, d, rng)

    if null_name == "truncated_gaussian":
        return sample_truncated_gaussian(M, d, radius=2.0, rng=rng)

    return sample_truncated_elliptical_t(
        M,
        d,
        radius=2.0,
        nu=DEFAULT_T_DF,
        Sigma=DEFAULT_T_SIGMA,
        rng=rng,
    )


def sample_alternative(null_name, alt_type, level, M, rng=None):
    """
    IID samples from one of the GOF alternative families.
    """
    if rng is None:
        rng = np.random.default_rng()
    level = float(level)
    X0 = sample_null(null_name, M, rng=rng, use_qmc=False)
    e1 = np.array([1.0, 0.0, 0.0])

    if alt_type == "location_shift":
        return X0 + level * e1

    if alt_type == "scale":
        if level <= 0.0:
            raise ValueError(f"Scale level must be positive, got {level}")
        return level * X0

    if alt_type == "mixture_contamination":
        if not 0.0 <= level <= 1.0:
            raise ValueError(f"Mixture level must be in [0, 1], got {level}")
        contaminated = 0.5 * e1 + 0.25 * X0
        use_contaminated = rng.random(M) < level
        Y = X0.copy()
        Y[use_contaminated] = contaminated[use_contaminated]
        return Y

    raise ValueError(f"Unknown alternative type: {alt_type}")


def phi_true(X):
    """
    True Brenier potential for the identity map: phi(x)=0.5||x||^2.
    """
    return 0.5 * np.sum(X * X, axis=1)
