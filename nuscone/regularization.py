import numpy as np
from scipy.optimize import nnls


def normalize_distribution(x: np.ndarray) -> np.ndarray:
    total = np.sum(x)
    if total <= 0:
        raise ValueError("Cannot normalize distribution with non-positive sum.")
    return x / total

def solve_tikhonov_nnls(
    A: np.ndarray,
    y: np.ndarray,
    D: np.ndarray,
    lambda_reg: float,
) -> tuple[np.ndarray, float]:
    rhs = np.concatenate([y, np.zeros(D.shape[0])])
    lhs = np.vstack([A, lambda_reg * D])

    x, residual = nnls(lhs, rhs)
    return normalize_distribution(x), residual


def unfold_energy_grid(
    y_grid: np.ndarray,
    operators: list[np.ndarray],
    D: np.ndarray,
    lambdas: list[float],
) -> np.ndarray:
    unfolded = []

    for y, A, lambda_reg in zip(y_grid, operators, lambdas):
        x, _ = solve_tikhonov_nnls(A, y, D, lambda_reg)
        unfolded.append(x)

    return np.asarray(unfolded)

import numpy as np
from scipy.optimize import minimize


def solve_tikhonov_constrained(A, y, D, lam, nubar_target):
    n = A.shape[1]
    nu = np.arange(n, dtype=float)

    def objective(x):
        r = A @ x - y
        s = D @ x
        return np.dot(r, r) + lam**2 * np.dot(s, s)

    def jac(x):
        return (
            2 * A.T @ (A @ x - y)
            + 2 * lam**2 * D.T @ (D @ x)
        )

    constraints = [
        {
            "type": "eq",
            "fun": lambda x: np.sum(x) - 1.0,
            "jac": lambda x: np.ones_like(x),
        },
        {
            "type": "eq",
            "fun": lambda x: np.dot(nu, x) - nubar_target,
            "jac": lambda x: nu,
        },
        {
            "type": "ineq",
            "fun": lambda x: 0.01 - x[0],
            "jac": lambda x: np.r_[-1.0, np.zeros(len(x)-1)],
        }
    ]

    bounds = [(0.0, None)] * n
    x0, _ = solve_tikhonov_nnls(A, y, D, lam)

    res = minimize(
        objective,
        x0,
        jac=jac,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 2000},
    )

    if not res.success:
        raise RuntimeError(res.message)

    return res.x, res