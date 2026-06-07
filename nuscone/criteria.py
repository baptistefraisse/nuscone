import numpy as np
from .regularization import solve_tikhonov_nnls, solve_tikhonov_constrained
from .moments import mean, standard_deviation


def gcv_score(A: np.ndarray, y: np.ndarray, x: np.ndarray, effective_dof: float) -> float:
    numerator = np.linalg.norm(A @ x - y) ** 2
    denominator = max(len(y) - effective_dof, 1e-12) ** 2
    return numerator / denominator


def nubar_difference_score(y: np.ndarray, x: np.ndarray, epsilon: float) -> float:
    observed = mean(y) / epsilon
    unfolded = mean(x)
    return abs(unfolded - observed) / observed


def scan_lambdas(
    A: np.ndarray,
    y: np.ndarray,
    D: np.ndarray,
    lambda_grid: np.ndarray,
    nubar_target=None,
) -> dict:
    xs = []
    residuals = []
    smoothness = []
    sigmas = []
    nubar_diff = []
    gcv = []

    for lam in lambda_grid:
        x, _ = solve_tikhonov_constrained(A, y, D, lam, nubar_target) 
        xs.append(x)
        residuals.append(np.linalg.norm(A @ x - y) ** 2)
        smoothness.append(np.linalg.norm(D @ x) ** 2)
        sigmas.append(standard_deviation(x))
        nubar_diff.append(nubar_difference_score(y, x, 0.73))
        gcv.append(gcv_score(A, y, x, effective_dof=np.sum(x > 0)))
        
    return {
        "lambda": lambda_grid,
        "x": np.asarray(xs),
        "residual": np.asarray(residuals),
        "smoothness": np.asarray(smoothness),
        "sigma": np.asarray(sigmas),
        "nubar_diff": np.asarray(nubar_diff),
        "gcv": np.asarray(gcv),
    }


def choose_lambda(scan: dict, method: str) -> float:
    if method == "gcv":
        idx = np.argmin(scan["gcv"])
    elif method == "sigma_min":
        idx = np.argmin(scan["sigma"])
    elif method == "nubar_diff":
        idx = np.argmin(scan["nubar_diff"])
    elif method == "lcurve":
        idx = _lcurve_corner(scan["residual"], scan["smoothness"])
    else:
        raise ValueError(f"Unknown lambda selection method: {method}")

    return float(scan["lambda"][idx])


def _lcurve_corner(residual: np.ndarray, smoothness: np.ndarray) -> int:
    points = np.column_stack([np.log(residual), np.log(smoothness)])
    first = points[0]
    last = points[-1]
    line = last - first
    line /= np.linalg.norm(line)

    distances = []
    for point in points:
        projection = first + np.dot(point - first, line) * line
        distances.append(np.linalg.norm(point - projection))

    return int(np.argmax(distances))
