from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.optimize import minimize

from .models import gaussian, linear_model, raised_cosine
from .references import load_references


@dataclass
class MultiChanceConfig:
    
    reference_dir: Path = Path("data/references")

    nmax: int = 10

    a_238U: float = 0.11
    b_238U: float = 2.5
    c_238U: float = 0.02
    d_238U: float = 1.09

    a_237U: float = 0.13
    b_237U: float = 2.5
    c_237U: float = 0.02
    d_237U: float = 1.09

    a_236U: float = 0.13
    b_236U: float = 2.5
    c_236U: float = 0.02
    d_236U: float = 1.09

    Sn_239U: float = 4.9
    Sn_238U: float = 6.2

    grid_step_percent: float = 1.0

    w_shape: float = 0.0   # 0
    w_nubar: float = 150   # 150
    w_sigma: float = 1000  # 1000
    w_smooth: float = 1.0  # 1

    b_min: float = 2.45
    b_max: float = 2.55
    c_min: float = 0.0
    c_max: float = 0.08
    sigma_min: float = 0.25

    en_prefission_238U: np.ndarray = field(
    default_factory=lambda: np.array([])
    )

    en_prefission_237U: np.ndarray = field(
        default_factory=lambda: np.array([])
    )


def load_multichance_reference_data(cfg: MultiChanceConfig):
    refs = load_references(cfg.reference_dir)

    en_prefission_238u = refs["EN_PREFISSION_238U"]["value"]
    en_prefission_237u = refs["EN_PREFISSION_237U"]["value"]

    return en_prefission_238u, en_prefission_237u


def read_pnu_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    pnu = df.to_numpy(dtype=float)

    energy = np.arange(1.0, 1.0 + len(pnu), 1.0)

    row_sums = pnu.sum(axis=1)
    pnu = pnu / row_sums[:, None]

    return energy, pnu


def distribution_moments(pnu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.arange(pnu.shape[1])
    mean = pnu @ n
    var = ((n[None, :] - mean[:, None]) ** 2 * pnu).sum(axis=1)
    return mean, np.sqrt(np.maximum(var, 0.0))


def calibrate_first_chance_parameters(
    energy: np.ndarray,
    pnu: np.ndarray,
    cfg: MultiChanceConfig,
    emax: float = 4.0,
) -> MultiChanceConfig:
    nubar_exp, sigma_exp = distribution_moments(pnu)

    mask = energy <= emax

    a, b = np.polyfit(energy[mask], nubar_exp[mask], deg=1)
    c, d = np.polyfit(energy[mask], sigma_exp[mask], deg=1)

    cfg.a_238U = float(a)
    cfg.b_238U = float(b)
    cfg.c_238U = float(c)
    cfg.d_238U = float(d)

    cfg.b_237U = cfg.b_238U
    cfg.b_236U = cfg.b_238U

    cfg.c_237U = cfg.c_238U
    cfg.c_236U = cfg.c_238U

    print("First-chance calibration:")
    print(f"a_238U={cfg.a_238U:.6f}")
    print(f"b_common={cfg.b_238U:.6f}")
    print(f"c_common={cfg.c_238U:.6f}")
    print(f"d_238U={cfg.d_238U:.6f}")

    return cfg


def _component_parameters(E: float, i: int, cfg: MultiChanceConfig):

    if len(cfg.en_prefission_238U) == 0:
        raise RuntimeError("Pre-fission energies not loaded.")

    if len(cfg.en_prefission_237U) == 0:
        raise RuntimeError("Pre-fission energies not loaded.")
    
    en1 = cfg.en_prefission_238U[i]
    en2 = cfg.en_prefission_237U[i]

    b = cfg.b_238U
    c = cfg.c_238U

    E1 = E
    E2 = E - cfg.Sn_239U - en1
    E3 = E - cfg.Sn_239U - en1 - cfg.Sn_238U - en2

    nubar1 = linear_model(E1, cfg.a_238U, b)
    nubar2 = linear_model(E2, cfg.a_237U, b)
    nubar3 = linear_model(E3, cfg.a_236U, b)

    sigma1 = linear_model(E1, c, cfg.d_238U)
    sigma2 = linear_model(E2, c, cfg.d_237U)
    sigma3 = linear_model(E3, c, cfg.d_236U)

    sigma1 = max(float(sigma1), cfg.sigma_min)
    sigma2 = max(float(sigma2), cfg.sigma_min)
    sigma3 = max(float(sigma3), cfg.sigma_min)

    return nubar1, sigma1, nubar2, sigma2, nubar3, sigma3


def model_components(
    E: float,
    i: int,
    p2_percent: float,
    p3_percent: float,
    cfg: MultiChanceConfig,
):
    n = np.arange(cfg.nmax)

    p2 = p2_percent / 100.0
    p3 = p3_percent / 100.0
    p1 = 1.0 - p2 - p3

    p1 = max(p1, 0.0)
    p2 = max(p2, 0.0)
    p3 = max(p3, 0.0)

    s = p1 + p2 + p3
    if s <= 0.0:
        p1, p2, p3 = 1.0, 0.0, 0.0
    else:
        p1, p2, p3 = p1 / s, p2 / s, p3 / s

    nubar1, sigma1, nubar2, sigma2, nubar3, sigma3 = _component_parameters(E, i, cfg)

    g1 = gaussian(n, nubar1, sigma1)
    g2 = gaussian(n - 1, nubar2, sigma2)
    g3 = gaussian(n - 2, nubar3, sigma3)

    g1 = g1 / g1.sum()
    g2 = g2 / g2.sum()
    g3 = g3 / g3.sum()

    c1 = p1 * g1
    c2 = p2 * g2
    c3 = p3 * g3

    total = c1 + c2 + c3
    total = total / total.sum()

    return n, c1, c2, c3, total


def model_pnu(E: float, i: int, p2: float, p3: float, cfg: MultiChanceConfig) -> np.ndarray:
    _, _, _, _, total = model_components(
        E,
        i,
        100.0 * p2,
        100.0 * p3,
        cfg,
    )
    return total


def _softmax2(x: float) -> tuple[float, float]:
    ex = np.exp(np.clip(x, -50.0, 50.0))
    p2 = ex / (1.0 + ex)
    p1 = 1.0 - p2
    return p1, p2


def _softmax3(x2: float, x3: float) -> tuple[float, float, float]:
    z = np.array([0.0, x2, x3])
    z = z - np.max(z)
    ez = np.exp(z)
    p = ez / ez.sum()
    return float(p[0]), float(p[1]), float(p[2])


def _decode_probabilities(
    energy: np.ndarray,
    x: np.ndarray,
    cfg: MultiChanceConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nE = len(energy)

    logits_p2 = x[2:2 + nE]
    logits_p3 = x[2 + nE:2 + 2 * nE]

    p1 = np.zeros(nE)
    p2 = np.zeros(nE)
    p3 = np.zeros(nE)

    for i, E in enumerate(energy):
        if E < cfg.Sn_239U:
            p1[i], p2[i], p3[i] = 1.0, 0.0, 0.0

        elif E < cfg.Sn_239U + cfg.Sn_238U:
            p1[i], p2[i] = _softmax2(logits_p2[i])
            p3[i] = 0.0

        else:
            p1[i], p2[i], p3[i] = _softmax3(logits_p2[i], logits_p3[i])

    return p1, p2, p3


def _smoothness_penalty(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    if len(p1) < 3:
        return 0.0

    return float(
        np.mean(np.diff(p1, n=2) ** 2)
        + np.mean(np.diff(p2, n=2) ** 2)
        + np.mean(np.diff(p3, n=2) ** 2)
    )


def _global_objective(
    x: np.ndarray,
    energy: np.ndarray,
    pnu: np.ndarray,
    cfg: MultiChanceConfig,
) -> float:
    b_common = float(x[0])
    c_common = float(x[1])

    cfg.b_238U = cfg.b_237U = cfg.b_236U = b_common
    cfg.c_238U = cfg.c_237U = cfg.c_236U = c_common

    p1, p2, p3 = _decode_probabilities(energy, x, cfg)

    pnu_fit = []

    for i, E in enumerate(energy):
        _, _, _, _, total = model_components(
            E,
            i,
            100.0 * p2[i],
            100.0 * p3[i],
            cfg,
        )
        pnu_fit.append(total)

    pnu_fit = np.asarray(pnu_fit)

    nubar_exp, sigma_exp = distribution_moments(pnu)
    nubar_fit, sigma_fit = distribution_moments(pnu_fit)

    shape_loss = np.mean((pnu_fit - pnu) ** 2)
    nubar_loss = np.mean((nubar_fit - nubar_exp) ** 2)
    sigma_loss = np.mean((sigma_fit - sigma_exp) ** 2)
    smooth_loss = _smoothness_penalty(p1, p2, p3)

    return (
        cfg.w_shape * shape_loss
        + cfg.w_nubar * nubar_loss
        + cfg.w_sigma * sigma_loss
        + cfg.w_smooth * smooth_loss
    )


def _initial_global_parameters(
    energy: np.ndarray,
    cfg: MultiChanceConfig,
) -> np.ndarray:
    nE = len(energy)

    b0 = np.clip(cfg.b_238U, cfg.b_min, cfg.b_max)
    c0 = np.clip(cfg.c_238U, cfg.c_min, cfg.c_max)

    logits_p2 = np.full(nE, -8.0)
    logits_p3 = np.full(nE, -8.0)

    for i, E in enumerate(energy):
        if E < cfg.Sn_239U:
            logits_p2[i] = -12.0
            logits_p3[i] = -12.0

        elif E < cfg.Sn_239U + cfg.Sn_238U:
            t = (E - cfg.Sn_239U) / max(cfg.Sn_238U, 1e-6)
            t = np.clip(t, 0.02, 0.98)
            logits_p2[i] = np.log(t / (1.0 - t))
            logits_p3[i] = -12.0

        else:
            t3 = (E - (cfg.Sn_239U + cfg.Sn_238U)) / 8.0
            t3 = np.clip(t3, 0.02, 0.70)

            p3 = t3
            p2 = 1.0 - p3
            p1 = 1e-3

            logits_p2[i] = np.log(p2 / p1)
            logits_p3[i] = np.log(p3 / p1)

    return np.r_[b0, c0, logits_p2, logits_p3]


def optimize_global_parameters(
    energy: np.ndarray,
    pnu: np.ndarray,
    cfg: MultiChanceConfig,
    maxiter = 5000,
) -> tuple[MultiChanceConfig, np.ndarray, np.ndarray, np.ndarray, float]:
    x0 = _initial_global_parameters(energy, cfg)

    nE = len(energy)

    bounds = (
        [(cfg.b_min, cfg.b_max), (cfg.c_min, cfg.c_max)]
        + [(-15.0, 15.0)] * nE
        + [(-15.0, 15.0)] * nE
    )

    res = minimize(
        _global_objective,
        x0=x0,
        args=(energy, pnu, cfg),
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": maxiter,
            "ftol": 1e-12,
            "xtol": 1e-8,
            "disp": True,
        },
    )

    x = res.x

    cfg.b_238U = cfg.b_237U = cfg.b_236U = float(x[0])
    cfg.c_238U = cfg.c_237U = cfg.c_236U = float(x[1])

    p1, p2, p3 = _decode_probabilities(energy, x, cfg)

    print("Global fit:")
    print(f"success={res.success}, loss={res.fun:.8e}")
    print(f"b_common={cfg.b_238U:.6f}")
    print(f"c_common={cfg.c_238U:.6f}")

    return cfg, p1, p2, p3, float(res.fun), x


def fit_multichance_for_energy(
    E: float,
    i: int,
    pnu_exp: np.ndarray,
    cfg: MultiChanceConfig,
) -> dict:
    """
    Gardée pour compatibilité.

    Cette fonction fait encore un fit local si appelée seule.
    Dans extract_multichance_probabilities, les probabilités finales viennent
    du fit global optimize_global_parameters.
    """
    pnu_exp = pnu_exp / pnu_exp.sum()

    def objective(y):
        p2, p3 = y

        if p2 < 0.0 or p3 < 0.0 or p2 + p3 > 1.0:
            return 1e12

        pnu_fit = model_pnu(E, i, p2, p3, cfg)

        nubar_exp, sigma_exp = distribution_moments(pnu_exp[None, :])
        nubar_fit, sigma_fit = distribution_moments(pnu_fit[None, :])

        shape_loss = np.mean((pnu_fit - pnu_exp) ** 2)
        nubar_loss = (nubar_fit[0] - nubar_exp[0]) ** 2
        sigma_loss = (sigma_fit[0] - sigma_exp[0]) ** 2

        return (
            cfg.w_shape * shape_loss
            + cfg.w_nubar * nubar_loss
            + cfg.w_sigma * sigma_loss
        )

    if E < cfg.Sn_239U:
        return {"p1": 100.0, "p2": 0.0, "p3": 0.0, "chi2": 0.0}

    if E < cfg.Sn_239U + cfg.Sn_238U:
        x0 = np.array([0.5, 0.0])
        bounds = [(0.0, 1.0), (0.0, 0.0)]
    else:
        x0 = np.array([0.4, 0.2])
        bounds = [(0.0, 1.0), (0.0, 1.0)]

    constraints = ({
        "type": "ineq",
        "fun": lambda y: 1.0 - y[0] - y[1],
    },)

    res = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500},
    )

    p2, p3 = res.x
    p1 = 1.0 - p2 - p3

    return {
        "p1": 100.0 * p1,
        "p2": 100.0 * p2,
        "p3": 100.0 * p3,
        "chi2": float(res.fun),
    }

def bootstrap_pnu_rows(pnu, pnu_err=None, n_events=None, rng=None):
    rng = np.random.default_rng() if rng is None else rng

    pnu = np.asarray(pnu, dtype=float)
    out = np.zeros_like(pnu)

    for i, p in enumerate(pnu):
        if pnu_err is not None:
            err = np.asarray(pnu_err[i], dtype=float)
            q = rng.normal(p, err)
            q = np.clip(q, 0.0, None)
        elif n_events is not None:
            N = int(n_events[i] if np.ndim(n_events) else n_events)
            counts = rng.multinomial(max(N, 1), p / p.sum())
            q = counts / counts.sum()
        else:
            raise ValueError("Need either pnu_err or n_events for bootstrap.")

        s = q.sum()
        if s <= 0.0:
            q = p.copy()
            s = q.sum()

        out[i] = q / s

    return out

def bootstrap_multichance_uncertainties(
    energy,
    pnu,
    cfg,
    n_boot = 5,
    maxiter = 300,
    pnu_err=None,
    n_events=None,
    seed=12345,
):
    rng = np.random.default_rng(seed)

    p1_samples = []
    p2_samples = []
    p3_samples = []

    for iboot in range(n_boot):
        print(f"Bootstrap {iboot+1}/{n_boot}")

        pnu_b = bootstrap_pnu_rows(
            pnu,
            pnu_err=pnu_err,
            n_events=n_events,
            rng=rng,
        )

        cfg_b = deepcopy(cfg)

        cfg_b, p1_b, p2_b, p3_b, _, _ = optimize_global_parameters(
            energy,
            pnu_b,
            cfg_b,
            maxiter=maxiter,
        )

        p1_samples.append(100.0 * p1_b)
        p2_samples.append(100.0 * p2_b)
        p3_samples.append(100.0 * p3_b)

    p1_samples = np.asarray(p1_samples)
    p2_samples = np.asarray(p2_samples)
    p3_samples = np.asarray(p3_samples)

    dp1 = np.std(p1_samples, axis=0, ddof=1)
    dp2 = np.std(p2_samples, axis=0, ddof=1)
    dp3 = np.std(p3_samples, axis=0, ddof=1)

    return dp1, dp2, dp3


def extract_multichance_probabilities(
    pnu_path: Path,
    output_dir: Path,
    cfg: MultiChanceConfig | None = None,
) -> pd.DataFrame:
    if cfg is None:
        cfg = MultiChanceConfig()

    output_dir.mkdir(parents=True, exist_ok=True)

    energy, pnu = read_pnu_table(pnu_path)

    en_prefission_238u, en_prefission_237u = load_multichance_reference_data(cfg)

    cfg.en_prefission_238U = en_prefission_238u
    cfg.en_prefission_237U = en_prefission_237u

    max_points = min(
        len(energy),
        len(en_prefission_238u),
        len(en_prefission_237u),
    )

    energy = energy[:max_points]
    pnu = pnu[:max_points]

    if pnu.shape[1] != cfg.nmax:
        cfg.nmax = pnu.shape[1]

    cfg = calibrate_first_chance_parameters(
        energy,
        pnu,
        cfg,
        emax=4.0,
    )

    cfg, p1, p2, p3, global_loss, x = optimize_global_parameters(
        energy,
        pnu,
        cfg,
        maxiter=100,
    )

    dp1, dp2, dp3 = bootstrap_multichance_uncertainties(
        energy,
        pnu,
        cfg,
        n_boot=10,
        maxiter=100,
        pnu_err=None,
        n_events=1e4, # typical averaged value
    )
    
    rows = []

    for i, E in enumerate(energy):
        _, _, _, _, total = model_components(
            E,
            i,
            100.0 * p2[i],
            100.0 * p3[i],
            cfg,
        )

        local_chi2 = np.mean((total - pnu[i]) ** 2)

        rows.append(
            {
                "energy": E,
                "p1": 100.0 * p1[i],
                "p2": 100.0 * p2[i],
                "p3": 100.0 * p3[i],
                "chi2": local_chi2,
            }
        )

    df = pd.DataFrame(rows)

    df["dp1_fit"] = dp1
    df["dp2_fit"] = dp2
    df["dp3_fit"] = dp3

    df["dp1"] = df["dp1_fit"]
    df["dp2"] = df["dp2_fit"]
    df["dp3"] = df["dp3_fit"]

    pnu_fit = []

    for i, row in df.iterrows():
        _, _, _, _, total = model_components(
            row["energy"],
            i,
            row["p2"],
            row["p3"],
            cfg,
        )
        pnu_fit.append(total)

    pnu_fit = np.asarray(pnu_fit)

    nubar_exp, sigma_exp = distribution_moments(pnu)
    nubar_fit, sigma_fit = distribution_moments(pnu_fit)

    df["nubar_exp"] = nubar_exp
    df["sigma_exp"] = sigma_exp
    df["nubar_fit"] = nubar_fit
    df["sigma_fit"] = sigma_fit

    df["b_common"] = cfg.b_238U
    df["c_common"] = cfg.c_238U
    df["global_loss"] = global_loss

    df.to_csv(output_dir / "multichance_probabilities.txt", sep="\t", index=False)

    df[["energy", "p1"]].to_csv(output_dir / "p1.txt", sep="\t", index=False, header=False)
    df[["energy", "p2"]].to_csv(output_dir / "p2.txt", sep="\t", index=False, header=False)
    df[["energy", "p3"]].to_csv(output_dir / "p3.txt", sep="\t", index=False, header=False)

    np.savetxt(
        output_dir / "multichance_pnu_fit.txt",
        pnu_fit,
        fmt="%.8e",
        delimiter="\t",
    )

    return df

