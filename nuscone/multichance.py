from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.optimize import minimize
from tqdm import tqdm

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

    a_236U: float = 0.11
    b_236U: float = 2.5
    c_236U: float = 0.02
    d_236U: float = 1.09

    # a_235U: float = 0.13
    # b_235U: float = 2.5
    # c_235U: float = 0.02
    # d_235U: float = 1.09

    Sn_239U: float = 4.9 # LANL data
    Sn_238U: float = 6.2 # LANL data
    Sn_237U: float = 5.0 # LANL data

    grid_step_percent: float = 1.0

    w_shape: float  = 1e0    # 1e0
    w_nubar: float  = 1e0    # 1e0
    w_sigma: float  = 1e1    # 1e1
    w_smooth: float = 1e-2   # 3e-2

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

    # en_prefission_236U: np.ndarray = field(
    #     default_factory=lambda: np.array([])
    # )


def load_multichance_reference_data(cfg: MultiChanceConfig):
    refs = load_references(cfg.reference_dir)

    en_prefission_238u = refs["EN_PREFISSION_238U"]["value"]
    en_prefission_237u = refs["EN_PREFISSION_237U"]["value"]
    # en_prefission_236u = refs["EN_PREFISSION_236U"]["value"]

    return en_prefission_238u, en_prefission_237u #, en_prefission_236u


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
    emax: float = 5.0,
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
    # cfg.b_235U = cfg.b_238U

    cfg.c_237U = cfg.c_238U
    cfg.c_236U = cfg.c_238U
    # cfg.c_235U = cfg.c_238U

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

    # if len(cfg.en_prefission_236U) == 0:
    #     raise RuntimeError("Pre-fission energies not loaded.")
    
    en1 = cfg.en_prefission_238U[i]
    en2 = cfg.en_prefission_237U[i]
    # en3 = cfg.en_prefission_236U[i]

    b = cfg.b_238U
    c = cfg.c_238U

    E1 = E
    E2 = E - cfg.Sn_239U - en1
    E3 = E - cfg.Sn_239U - en1 - cfg.Sn_238U - en2
    # E4 = E - cfg.Sn_239U - en1 - cfg.Sn_238U - en2 - cfg.Sn_237U - en3

    nubar1 = linear_model(E1, cfg.a_238U, b)
    nubar2 = linear_model(E2, cfg.a_237U, b)
    nubar3 = linear_model(E3, cfg.a_236U, b)
    # nubar4 = linear_model(E4, cfg.a_235U, b)

    sigma1 = linear_model(E1, c, cfg.d_238U)
    sigma2 = linear_model(E2, c, cfg.d_237U)
    sigma3 = linear_model(E3, c, cfg.d_236U)
    # sigma4 = linear_model(E4, c, cfg.d_235U)

    sigma1 = max(float(sigma1), cfg.sigma_min)
    sigma2 = max(float(sigma2), cfg.sigma_min)
    sigma3 = max(float(sigma3), cfg.sigma_min)
    # sigma4 = max(float(sigma4), cfg.sigma_min)

    return nubar1, sigma1, nubar2, sigma2, nubar3, sigma3 #, nubar4, sigma4


def model_components(
    E: float,
    i: int,
    p2_percent: float,
    p3_percent: float,
    #p4_percent: float,
    cfg: MultiChanceConfig,
):
    n = np.arange(cfg.nmax)

    p2 = p2_percent / 100.0
    p3 = p3_percent / 100.0
    # p4 = p4_percent / 100.0
    p1 = 1.0 - p2 - p3 #- p4

    p1 = max(p1, 0.0)
    p2 = max(p2, 0.0)
    p3 = max(p3, 0.0)
    # p4 = max(p4, 0.0)

    s = p1 + p2 + p3 #+ p4
    if s <= 0.0:
        p1, p2, p3 = 1.0, 0.0, 0.0 # p4, 0.0
    else:
        p1, p2, p3 = p1 / s, p2 / s, p3 / s # p4, p4 / s

    nubar1, sigma1, nubar2, sigma2, nubar3, sigma3 = _component_parameters(E, i, cfg) # , nubar4, sigma4

    g1 = gaussian(n, nubar1, sigma1)
    g2 = gaussian(n - 1, nubar2, sigma2)
    g3 = gaussian(n - 2, nubar3, sigma3)
    # g4 = gaussian(n - 3, nubar4, sigma4)

    g1 = g1 / g1.sum()
    g2 = g2 / g2.sum()
    g3 = g3 / g3.sum()
    # g4 = g4 / g4.sum()

    c1 = p1 * g1
    c2 = p2 * g2
    c3 = p3 * g3
    # c4 = p4 * g4

    total = c1 + c2 + c3 #+ c4
    total = total / total.sum()

    return n, c1, c2, c3, total # c4


def model_pnu(
        E: float, 
        i: int, 
        p2: float, 
        p3: float, 
        # p4: float, 
        cfg: MultiChanceConfig
    ) -> np.ndarray:
    _, _, _, _, total = model_components(
        E,
        i,
        100.0 * p2,
        100.0 * p3,
        # 100.0 * p4,
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

# def _softmax4(x2: float, x3: float, x4: float) -> tuple[float, float, float, float]:
#     z = np.array([0.0, x2, x3, x4])
#     z = z - np.max(z)
#     ez = np.exp(z)
#     p = ez / ez.sum()
#     return float(p[0]), float(p[1]), float(p[2]), float(p[3])

def _decode_probabilities(
    energy: np.ndarray,
    x: np.ndarray,
    cfg: MultiChanceConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: #, np.ndarray]:
    nE = len(energy)

    logits_p2 = x[2:2 + nE]
    logits_p3 = x[2 + nE:2 + 2 * nE]
    #logits_p4 = x[2 + 2 * nE:2 + 3 * nE]

    p1 = np.zeros(nE)
    p2 = np.zeros(nE)
    p3 = np.zeros(nE)
    # p4 = np.zeros(nE)

    Eth2 = cfg.Sn_239U
    Eth3 = cfg.Sn_239U + cfg.Sn_238U
    # Eth4 = cfg.Sn_239U + cfg.Sn_238U + cfg.Sn_237U

    for i, E in enumerate(energy):
        if E < Eth2:
            p1[i], p2[i], p3[i] = 1.0, 0.0, 0.0
        elif E < Eth3:
            p1[i], p2[i] = _softmax2(logits_p2[i])
            p3[i] = 0.0
        else:
            p1[i], p2[i], p3[i] = _softmax3(logits_p2[i], logits_p3[i])

    # for i, E in enumerate(energy):
    #     if E < Eth2:
    #         # p1[i], p2[i], p3[i], p4[i] = 1.0, 0.0, 0.0, 0.0
    #         p1[i], p2[i], p3[i] = 1.0, 0.0, 0.0

    #     elif E < Eth3:
    #         p1[i], p2[i] = _softmax2(logits_p2[i])
    #         # p3[i], p4[i] = 0.0, 0.0
    #         p3[i] = 0.0

    #     elif E < Eth4:
    #         p1[i], p2[i], p3[i] = _softmax3(logits_p2[i], logits_p3[i])
    #         # p4[i] = 0.0

    #     else:
    #         p1[i], p2[i], p3[i], p4[i] = _softmax4(
    #             logits_p2[i],
    #             logits_p3[i],
    #             logits_p4[i],
    #         )

    return p1, p2, p3 #, p4


def _smoothness_penalty(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float: # , p4: np.ndarray
    if len(p1) < 3:
        return 0.0

    return float(
        np.mean(np.diff(p1, n=2) ** 2)
        + np.mean(np.diff(p2, n=2) ** 2)
        + np.mean(np.diff(p3, n=2) ** 2)
        #+ np.mean(np.diff(p4, n=2) ** 2)
    )


def _global_objective(
    x: np.ndarray,
    energy: np.ndarray,
    pnu: np.ndarray,
    cfg: MultiChanceConfig,
) -> float:
    b_common = float(x[0])
    c_common = float(x[1])

    cfg.b_238U = cfg.b_237U = cfg.b_236U = b_common # cfg.b_235U
    cfg.c_238U = cfg.c_237U = cfg.c_236U = c_common # cfg.c_235U

    p1, p2, p3 = _decode_probabilities(energy, x, cfg) # , p4

    pnu_fit = []

    for i, E in enumerate(energy):
        _, _, _, _, total = model_components(
            E,
            i,
            100.0 * p2[i],
            100.0 * p3[i],
            #100.0 * p4[i],
            cfg,
        )
        pnu_fit.append(total)

    pnu_fit = np.asarray(pnu_fit)

    nubar_exp, sigma_exp = distribution_moments(pnu)
    nubar_fit, sigma_fit = distribution_moments(pnu_fit)

    shape_loss = np.mean((pnu_fit - pnu) ** 2)
    nubar_loss = np.mean((nubar_fit - nubar_exp) ** 2)
    sigma_loss = np.mean((sigma_fit - sigma_exp) ** 2)
    smooth_loss = _smoothness_penalty(p1, p2, p3) #, p4)

    # print(f"Debug. w_shape={shape_loss:.4e} | w_nubar={nubar_loss:.4e} | w_sigma={sigma_loss:.4e} | w_smooth={smooth_loss:.4e}")

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
    # logits_p4 = np.full(nE, -8.0)

    for i, E in enumerate(energy):
        if E < cfg.Sn_239U:
            logits_p2[i] = -12.0
            logits_p3[i] = -12.0
            # logits_p4[i] = -12.0

        # elif E < cfg.Sn_239U + cfg.Sn_238U:
        #     t = (E - cfg.Sn_239U) / max(cfg.Sn_238U, 1e-6)
        #     t = np.clip(t, 0.02, 0.98)
        #     logits_p2[i] = np.log(t / (1.0 - t))
        #     logits_p3[i] = -12.0
        #     # logits_p4[i] = -12.0

        # elif E < cfg.Sn_239U + cfg.Sn_238U:
        #     t = (E - cfg.Sn_239U) / 8.0
        #     t = np.clip(t, 0.02, 0.70)
        #     p2 = t
        #     p1 = 1.0 - p2
        #     logits_p2[i] = np.log(p2 / p1)
        #     logits_p3[i] = -12.0

        elif E < cfg.Sn_239U + cfg.Sn_238U:
            t = (E - cfg.Sn_239U) / 6.0
            t = np.clip(t, 0.0, 1.0)
            p2_max = 0.55
            p2 = p2_max * t
            p1 = 1.0 - p2
            logits_p2[i] = np.log(p2 / p1)
            logits_p3[i] = -12.0

        # else:
        #     t = (E - (cfg.Sn_239U + cfg.Sn_238U)) / 6.0
        #     t = np.clip(t, 0.0, 1.0)
        #     p1 = 0.10
        #     p3 = 0.65 * t
        #     p2 = max(1.0 - p1 - p3, 1e-6)
        #     logits_p2[i] = np.log(p2 / p1)
        #     logits_p3[i] = np.log(p3 / p1) if p3 > 0 else -12.0

        else:
            t = (E - (cfg.Sn_239U + cfg.Sn_238U)) / 6.0
            t = np.clip(t, 0.0, 1.0)
            p1 = 0.10 * (1.0 - t) + 0.03
            p3 = 0.45 * t
            p2 = max(1.0 - p1 - p3, 1e-6)
            logits_p2[i] = np.log(p2 / p1)
            logits_p3[i] = np.log(p3 / p1) if p3 > 0 else -12.0

        # else:
        #     t3 = (E - (cfg.Sn_239U + cfg.Sn_238U)) / 30.0
        #     t3 = np.clip(t3, 0.01, 0.15)
        #     p1 = 1e-3
        #     p3 = t3
        #     p2 = max(1.0 - p1 - p3, 1e-6) #-p4
        #     logits_p2[i] = np.log(p2 / p1)
        #     logits_p3[i] = np.log(p3 / p1)

            # Eth4 = cfg.Sn_239U + cfg.Sn_238U + cfg.Sn_237U

            # if E < Eth4:
            #     p1 = 1e-3
            #     # p4 = 1e-6
            #     p3 = t3
            #     p2 = max(1.0 - p1 - p3, 1e-6) #-p4

            # else:
                # t4 = (E - Eth4) / 8.0
                # t4 = np.clip(t4, 0.02, 0.40)
                # p1 = 1e-3
                # p4 = t4
                # p3 = max(0.30, 0.70 - t4)
                # p2 = max(1.0 - p1 - p3 - p4, 1e-6)

                # t4 = (E - Eth4) / 6.0
                # t4 = np.clip(t4, 0.05, 0.60)
                # p1 = 1e-4
                # p4 = t4
                # p3 = max(0.20, 0.65 - 0.7 * t4)
                # p2 = max(1.0 - p1 - p3 - p4, 1e-6)

            # logits_p2[i] = np.log(p2 / p1)
            # logits_p3[i] = np.log(p3 / p1)
            # logits_p4[i] = np.log(p4 / p1)

    return np.r_[b0, c0, logits_p2, logits_p3]#, logits_p4]


def optimize_global_parameters(
    energy: np.ndarray,
    pnu: np.ndarray,
    cfg: MultiChanceConfig,
    maxiter = 5000,
    ) -> tuple[
    MultiChanceConfig,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    # np.ndarray,
    float,
    np.ndarray,
]:
    x0 = _initial_global_parameters(energy, cfg)

    nE = len(energy)

    bounds = (
        [(cfg.b_min, cfg.b_max), (cfg.c_min, cfg.c_max)]
        + [(-15.0, 15.0)] * nE
        + [(-15.0, 15.0)] * nE
        # + [(-15.0, 15.0)] * nE
    )

    pbar = tqdm(
        total=maxiter,
        desc="Powell iterations",
        unit="iter",
    )

    def callback(xk):
        pbar.update(1)
        loss_eval = _global_objective(xk, energy, pnu, cfg)
        pbar.set_postfix(loss=f"{loss_eval:.4e}")
        # tqdm.write(f"  iter {pbar.n:3d} | loss = {loss_eval:.6e}")

    res = minimize(
        _global_objective,
        x0=x0,
        args=(energy, pnu, cfg),
        method="Powell",
        bounds=bounds,
        callback=callback,
        options={
            "maxiter": maxiter,
            "ftol": 1e-7,
            "xtol": 1e-7,
            "disp": True,
        },
    )

    pbar.close()

    x = res.x

    cfg.b_238U = cfg.b_237U = cfg.b_236U = float(x[0]) # cfg.b_235U
    cfg.c_238U = cfg.c_237U = cfg.c_236U = float(x[1]) # cfg.c_235U

    # p1, p2, p3, p4 = _decode_probabilities(energy, x, cfg)
    p1, p2, p3 = _decode_probabilities(energy, x, cfg)

    print("Global fit:")
    print(f"success={res.success}, loss={res.fun:.8e}")
    print(f"b_common={cfg.b_238U:.6f}")
    print(f"c_common={cfg.c_238U:.6f}")

    return cfg, p1, p2, p3, float(res.fun), x #, p4, 


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
    p1_central,
    p2_central,
    p3_central,
    n_boot = 10,
    maxiter = 100,
    pnu_err=None,
    n_events=None,
    seed=12345,
):
    rng = np.random.default_rng(seed)

    p1_samples = []
    p2_samples = []
    p3_samples = []
    nubar_fit_samples = []
    sigma_fit_samples = []

    for iboot in tqdm(range(n_boot), desc="Bootstrap uncertainty quantification", unit="fit"):

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

        pnu_fit_b = np.asarray([
            model_components(E, i, 100.0 * p2_b[i], 100.0 * p3_b[i], cfg_b)[4]
            for i, E in enumerate(energy)
        ])
        nubar_b, sigma_b = distribution_moments(pnu_fit_b)
        nubar_fit_samples.append(nubar_b)
        sigma_fit_samples.append(sigma_b)

    p1_samples = np.asarray(p1_samples)
    p2_samples = np.asarray(p2_samples)
    p3_samples = np.asarray(p3_samples)
    nubar_fit_samples = np.asarray(nubar_fit_samples)
    sigma_fit_samples = np.asarray(sigma_fit_samples)

    c1 = 100.0 * p1_central
    c2 = 100.0 * p2_central
    c3 = 100.0 * p3_central

    # dp1_low = np.maximum(c1 - np.percentile(p1_samples, 16, axis=0), 0.0)
    # dp1_up  = np.maximum(np.percentile(p1_samples, 84, axis=0) - c1, 0.0)
    # dp2_low = np.maximum(c2 - np.percentile(p2_samples, 16, axis=0), 0.0)
    # dp2_up  = np.maximum(np.percentile(p2_samples, 84, axis=0) - c2, 0.0)
    # dp3_low = np.maximum(c3 - np.percentile(p3_samples, 16, axis=0), 0.0)
    # dp3_up  = np.maximum(np.percentile(p3_samples, 84, axis=0) - c3, 0.0)

    p16_1 = np.clip(np.percentile(p1_samples, 16, axis=0), 0.0, 100.0)
    p84_1 = np.clip(np.percentile(p1_samples, 84, axis=0), 0.0, 100.0)
    p16_2 = np.clip(np.percentile(p2_samples, 16, axis=0), 0.0, 100.0)
    p84_2 = np.clip(np.percentile(p2_samples, 84, axis=0), 0.0, 100.0)
    p16_3 = np.clip(np.percentile(p3_samples, 16, axis=0), 0.0, 100.0)
    p84_3 = np.clip(np.percentile(p3_samples, 84, axis=0), 0.0, 100.0)

    dp1_low = np.maximum(c1 - p16_1, 0.0)
    dp1_up  = np.maximum(p84_1 - c1, 0.0)
    dp2_low = np.maximum(c2 - p16_2, 0.0)
    dp2_up  = np.maximum(p84_2 - c2, 0.0)
    dp3_low = np.maximum(c3 - p16_3, 0.0)
    dp3_up  = np.maximum(p84_3 - c3, 0.0)

    nubar_fit_lo = np.percentile(nubar_fit_samples, 16, axis=0)
    nubar_fit_hi = np.percentile(nubar_fit_samples, 84, axis=0)
    sigma_fit_lo = np.percentile(sigma_fit_samples, 16, axis=0)
    sigma_fit_hi = np.percentile(sigma_fit_samples, 84, axis=0)

    return (
        dp1_low, dp1_up, dp2_low, dp2_up, dp3_low, dp3_up,
        nubar_fit_lo, nubar_fit_hi, sigma_fit_lo, sigma_fit_hi,
    )


def extract_multichance_probabilities(
    pnu_path: Path,
    output_dir: Path,
    cfg: MultiChanceConfig | None = None,
    stat=None,
) -> pd.DataFrame:
    if cfg is None:
        cfg = MultiChanceConfig()

    output_dir.mkdir(parents=True, exist_ok=True)

    energy, pnu = read_pnu_table(pnu_path)

    en_prefission_238u, en_prefission_237u = load_multichance_reference_data(cfg) #, en_prefission_236u

    cfg.en_prefission_238U = en_prefission_238u
    cfg.en_prefission_237U = en_prefission_237u
    # cfg.en_prefission_236U = en_prefission_236u

    max_points = min(
        len(energy),
        len(en_prefission_238u),
        len(en_prefission_237u),
        #len(en_prefission_236u),
    )

    #Eth4 = cfg.Sn_239U + cfg.Sn_238U + cfg.Sn_237U + 5 

    # max_points = int(np.searchsorted(energy, Eth4)) # stop before 4th chance
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

    cfg, p1, p2, p3, global_loss, _ = optimize_global_parameters(
        energy,
        pnu,
        cfg,
        maxiter=100,
    )

    (
        dp1_low, dp1_up, dp2_low, dp2_up, dp3_low, dp3_up,
        nubar_fit_lo, nubar_fit_hi, sigma_fit_lo, sigma_fit_hi,
    ) = bootstrap_multichance_uncertainties(
        energy,
        pnu,
        cfg,
        p1, p2, p3,
        n_boot=2,
        maxiter=100,
        pnu_err=None,
        n_events=stat,
    )
    
    rows = []

    for i, E in enumerate(energy):
        _, _, _, _, total = model_components(
            E,
            i,
            100.0 * p2[i],
            100.0 * p3[i],
            # 100.0 * p4[i],
            cfg,
        )

        local_chi2 = np.mean((total - pnu[i]) ** 2)

        rows.append(
            {
                "energy": E,
                "p1": 100.0 * p1[i],
                "p2": 100.0 * p2[i],
                "p3": 100.0 * p3[i],
                # "p4": 100.0 * p4[i],
                "chi2": local_chi2,
            }
        )

    df = pd.DataFrame(rows)

    df["dp1_low"] = dp1_low
    df["dp1_up"]  = dp1_up
    df["dp2_low"] = dp2_low
    df["dp2_up"]  = dp2_up
    df["dp3_low"] = dp3_low
    df["dp3_up"]  = dp3_up

    # moyenne symétrique (utilisée pour dE_exc et autres)
    df["dp1"] = (dp1_low + dp1_up) / 2.0
    df["dp2"] = (dp2_low + dp2_up) / 2.0
    df["dp3"] = (dp3_low + dp3_up) / 2.0
    df["dp1_fit"] = df["dp1"]
    df["dp2_fit"] = df["dp2"]
    df["dp3_fit"] = df["dp3"]

    pnu_fit = []

    for i, row in df.iterrows():
        _, _, _, _, total = model_components(
            row["energy"],
            i,
            row["p2"],
            row["p3"],
            # row["p4"],
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
    df["nubar_fit_lo"] = nubar_fit_lo
    df["nubar_fit_hi"] = nubar_fit_hi
    df["sigma_fit_lo"] = sigma_fit_lo
    df["sigma_fit_hi"] = sigma_fit_hi

    df["b_common"] = cfg.b_238U
    df["c_common"] = cfg.c_238U
    df["global_loss"] = global_loss

    # excitation energy

    E_exc = np.zeros(len(energy))
    for i, En in enumerate(energy):
        en1 = cfg.en_prefission_238U[i]
        en2 = cfg.en_prefission_237U[i]
        E1 = En + cfg.Sn_239U
        E2 = En - en1
        E3 = En - cfg.Sn_238U - en1 - en2
        E_exc[i] = p1[i] * E1 + p2[i] * E2 + p3[i] * E3
    df["E_exc"] = E_exc

    dE_exc_proba = np.zeros(len(energy))
    for i, En in enumerate(energy):
        en1 = cfg.en_prefission_238U[i]
        en2 = cfg.en_prefission_237U[i]
        d2 = ((dp2_low[i] + dp2_up[i]) / 2.0 / 100.0) * (en1 + cfg.Sn_239U)
        d3 = ((dp3_low[i] + dp3_up[i]) / 2.0 / 100.0) * (en1 + cfg.Sn_238U + en2 + cfg.Sn_239U)
        dE_exc_proba[i] = np.sqrt(d2**2 + d3**2)
    df["dE_exc_proba"] = dE_exc_proba

    # saving

    df.to_csv(output_dir / "multichance_probabilities.txt", sep="\t", index=False)

    df[["energy", "p1"]].to_csv(output_dir / "p1.txt", sep="\t", index=False, header=False)
    df[["energy", "p2"]].to_csv(output_dir / "p2.txt", sep="\t", index=False, header=False)
    df[["energy", "p3"]].to_csv(output_dir / "p3.txt", sep="\t", index=False, header=False)
    # df[["energy", "p4"]].to_csv(output_dir / "p4.txt", sep="\t", index=False, header=False)

    np.savetxt(
        output_dir / "multichance_pnu_fit.txt",
        pnu_fit,
        fmt="%.8e",
        delimiter="\t",
    )

    return df

