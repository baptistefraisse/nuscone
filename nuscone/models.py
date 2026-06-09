import numpy as np


def gaussian(n, mu, sigma):
    return (1.0 / np.sqrt(2.0 * np.pi) / sigma) * np.exp(
        -0.5 * ((n - mu) / sigma) ** 2
    )


def raised_cosine(n, mu, sigma):
    s = sigma / np.sqrt((1.0 / 3.0) - (2.0 / np.pi**2))
    return (
        (n <= mu + s)
        * (n >= mu - s)
        * (1.0 + np.cos(np.pi * (n - mu) / s))
        / (2.0 * s)
    )


def linear_model(E, a, b):
    return a * E + b