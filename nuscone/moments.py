import numpy as np


def multiplicities(p: np.ndarray) -> np.ndarray:
    return np.arange(len(p))


def mean(p: np.ndarray) -> float:
    n = multiplicities(p)
    return float(np.sum(n * p))


def factorial_moment(p: np.ndarray, order: int) -> float:
    n = multiplicities(p)
    coeff = np.ones_like(n, dtype=float)

    for k in range(order):
        coeff *= n - k

    return float(np.sum(coeff * p))


def variance(p: np.ndarray) -> float:
    n = multiplicities(p)
    mu = mean(p)
    return float(np.sum((n - mu) ** 2 * p))


def standard_deviation(p: np.ndarray) -> float:
    return float(np.sqrt(variance(p)))


def skewness(p: np.ndarray) -> float:
    n = multiplicities(p)
    mu = mean(p)
    sigma = standard_deviation(p)
    if sigma == 0:
        return float("nan")
    return float(np.sum(((n - mu) / sigma) ** 3 * p))


def kurtosis_excess(p: np.ndarray) -> float:
    n = multiplicities(p)
    mu = mean(p)
    sigma = standard_deviation(p)
    if sigma == 0:
        return float("nan")
    return float(np.sum(((n - mu) / sigma) ** 4 * p) - 3.0)


def summarize_distribution(p: np.ndarray) -> dict:
    f1 = factorial_moment(p, 1)
    f2 = factorial_moment(p, 2)
    f3 = factorial_moment(p, 3)
    f4 = factorial_moment(p, 4)

    return {
        "nubar": mean(p),
        "sigma": standard_deviation(p),
        "skewness": skewness(p),
        "kurtosis_excess": kurtosis_excess(p),
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "f4": f4,
        "diven_factor": f2 / f1**2 if f1 != 0 else float("nan"),
        "cv": standard_deviation(p) / f1 if f1 != 0 else float("nan"),
    }