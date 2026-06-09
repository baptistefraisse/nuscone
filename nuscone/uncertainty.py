import numpy as np
from .operators import numerical_jacobian
from .multichance import _global_objective, _decode_probabilities

def mean_error(p, p_err=None, n_events=None):
    nu = np.arange(len(p))
    nubar = np.sum(nu * p)
    sigma = np.sqrt(np.sum(p * (nu - nubar)**2))
    err2 = 0.0
    if p_err is not None:
        err2 += np.sum((nu * p_err)**2)
    if n_events is not None and n_events > 0:
        err2 += sigma**2 / n_events
    return np.sqrt(err2)


def sigma_error(p, p_err=None, n_events=None):
    nu = np.arange(len(p))
    nubar = np.sum(nu * p)
    sigma = np.sqrt(np.sum(p * (nu - nubar)**2))

    err2 = 0.0

    if p_err is not None:
        grad = ((nu - nubar)**2 - sigma**2) / (2 * sigma)
        err2 += np.sum((grad * p_err)**2)

    if n_events is not None and n_events > 0:
        err2 += sigma**2 / (2 * n_events)

    return np.sqrt(err2)


def f2_error(p, p_err):
    nu = np.arange(len(p))
    grad = nu * (nu - 1)
    return np.sqrt(np.sum((grad * p_err)**2))


def f3_error(p, p_err):
    nu = np.arange(len(p))
    grad = nu * (nu - 1) * (nu - 2)
    return np.sqrt(np.sum((grad * p_err)**2))


def tikhonov_inverse_operator(A: np.ndarray, D: np.ndarray, lambda_reg: float) -> np.ndarray:
    lhs = A.T @ A + lambda_reg * (D.T @ D)
    return np.linalg.pinv(lhs) @ A.T


def propagate_distribution_errors(A, D, lambda_reg, y, n_events):
    A_hash = tikhonov_inverse_operator(A, D, lambda_reg)

    y = np.asarray(y, dtype=float)
    N = max(float(n_events), 1.0)

    covariance_y = np.diag(np.maximum(y, 0.0) / N)

    covariance_x = A_hash @ covariance_y @ A_hash.T

    return np.sqrt(np.maximum(np.diag(covariance_x), 0.0))

