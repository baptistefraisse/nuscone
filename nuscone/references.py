from pathlib import Path
import numpy as np


def _loadtxt(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing reference file: {path}")
    return np.loadtxt(path)


def factorial_moment_from_pnu(pnu: np.ndarray, order: int) -> np.ndarray:
    nu = np.arange(pnu.shape[1])
    coeff = np.ones_like(nu, dtype=float)
    for k in range(order):
        coeff *= nu - k
    return pnu @ coeff


def load_references(reference_dir: str | Path) -> dict:
    reference_dir = Path(reference_dir)
    refs = {}

    # GEF

    gef = _loadtxt(reference_dir / "gef" / "238U_nubar_sigma_GEF.txt")
    refs["GEF"] = {
        "energy": gef[:, 0],
        "nubar": gef[:, 1],
        "sigma": gef[:, 2],
    }

    gef_pnu = _loadtxt(reference_dir / "gef" / "GEF_pnu.txt")
    gef_energy_pnu = gef_pnu[:, 0]
    gef_pnu_values = gef_pnu[:, 1:]

    refs["GEF_PNU"] = {
        "energy": gef_energy_pnu,
        "pnu": gef_pnu_values,
        "f1": factorial_moment_from_pnu(gef_pnu_values, 1),
        "f2": factorial_moment_from_pnu(gef_pnu_values, 2),
        "f3": factorial_moment_from_pnu(gef_pnu_values, 3),
        "f4": factorial_moment_from_pnu(gef_pnu_values, 4),
    }

    # CGMF

    cgmf = _loadtxt(reference_dir / "cgmf" / "238U_moments_CGMF.txt")
    f1 = cgmf[:, 1]
    f2 = cgmf[:, 2]
    f3 = cgmf[:, 3]
    refs["CGMF"] = {
        "energy": cgmf[:, 0],
        "nubar": f1,
        "sigma": np.sqrt(f1 + f2 - f1**2),
        "f1": f1,
        "f2": f2,
        "f3": f3,
    }

    # JEFF 3.3

    jeff = _loadtxt(reference_dir / "jeff" / "238U_nubar_JEFF.txt")
    refs["JEFF"] = {
        "energy": jeff[:, 0] * 1e-6,
        "nubar": jeff[:, 1],
        "nubar_err": jeff[:, 2],
    }

    # ENDF-BVIII.0

    endf = _loadtxt(reference_dir / "endf" / "238U_nubar_ENDF.txt")
    refs["ENDF"] = {
        "energy": endf[:, 0] * 1e-6,
        "nubar": endf[:, 1],
        "nubar_err": endf[:, 2],
    }

    # EXFOR Frehaut

    frehaut = _loadtxt(reference_dir / "exfor" / "238U_nubar_Frehaut.txt")
    refs["Frehaut"] = {
        "energy": frehaut[:, 0],
        "nubar": frehaut[:, 1],
        "nubar_err": frehaut[:, 2],
    }

    return refs