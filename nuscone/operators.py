import numpy as np
from scipy.special import binom


def efficiency_operator(nmax: int, epsilon: float) -> np.ndarray:
    size = nmax + 1
    S = np.zeros((size, size))

    for measured in range(size):
        for true in range(size):
            if true >= measured:
                S[measured, true] = (
                    binom(true, measured)
                    * epsilon**measured
                    * (1.0 - epsilon) ** (true - measured)
                )
    return S


def normalize_noise(noise: np.ndarray, nmax: int) -> np.ndarray:
    noise = np.asarray(noise[: nmax + 1], dtype=float)
    total = noise.sum()
    if total <= 0:
        raise ValueError("Noise distribution has non-positive sum.")
    return noise / total


def noise_operator(noise: np.ndarray, nmax: int) -> np.ndarray:
    noise = normalize_noise(noise, nmax)
    size = nmax + 1
    B = np.zeros((size, size))

    for measured in range(size):
        for true in range(size):
            if measured >= true:
                B[measured, true] = noise[measured - true]
    return B


def response_operator(noise: np.ndarray, nmax: int, epsilon: float) -> np.ndarray:
    return noise_operator(noise, nmax) @ efficiency_operator(nmax, epsilon)


def derivative_operator(nmax: int, order: int = 2) -> np.ndarray:
    size = nmax + 1

    if order == 0:
        return np.eye(size)

    if order == 1:
        D = np.zeros((size - 1, size))
        for i in range(size - 1):
            D[i, i] = -1.0
            D[i, i + 1] = 1.0
        return D

    if order == 2:
        # D = np.zeros((size - 2, size))
        # for i in range(size - 2):
        #     D[i, i] = 1.0
        #     D[i, i + 1] = -2.0
        #     D[i, i + 2] = 1.0
        # return D
        return second_derivative_operator_limit_conditions(nmax)

    raise ValueError(f"Unsupported derivative order: {order}")

def second_derivative_operator_limit_conditions(nmax: int) -> np.ndarray:
    D2 = np.zeros((nmax, nmax + 1))

    for i in range(0, nmax):
        for j in range(0, nmax + 1):
            if i == j:
                if i == 0:
                    D2[i][j] = -1
                elif i == nmax - 1:
                    D2[i][j] = -3
                else:
                    D2[i][j] = -2.0

            if i == j + 1:
                D2[i][j] = 1

            if i == j - 1:
                if i == nmax - 1:
                    D2[i][j] = 2
                else:
                    D2[i][j] = 1

    D2[0][0] = 8.0
    D2[0][1] = 0.0
    D2[1][0] = -4.0
    D2[1][1] = 4.0
    D2[1][2] = 0.0

    D2[nmax - 1][nmax] = 10.0
    D2[nmax - 1][nmax - 1] = 0.0
    D2[nmax - 1][nmax - 2] = 0.0

    D2[nmax - 2][nmax] = 0.0
    D2[nmax - 2][nmax - 1] = -1.0
    D2[nmax - 2][nmax - 2] = 1.0
    D2[nmax - 2][nmax - 3] = 0.0

    return D2


def numerical_jacobian(func, x, eps=1e-5):
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(func(x), dtype=float)
    J = np.zeros((f0.size, x.size))

    for j in range(x.size):
        dx = np.zeros_like(x)
        dx[j] = eps * max(1.0, abs(x[j]))

        fp = np.asarray(func(x + dx), dtype=float)
        fm = np.asarray(func(x - dx), dtype=float)

        J[:, j] = (fp - fm) / (2.0 * dx[j])

    return J