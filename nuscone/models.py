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


def delta_tke_to_sigma(delta_tke, sn_en_mean):
    """sigma_nu ~ delta_TKE / <Sn + en> (B3 model)."""
    return delta_tke / sn_en_mean


def delta_tke_to_sigma_err(delta_tke, sn_en_mean, sn_en_std):
    """Incertitude sur sigma_nu par propagation de l'ecart-type sur <Sn+en>."""
    return delta_tke * sn_en_std / sn_en_mean ** 2