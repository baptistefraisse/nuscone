import numpy as np
from pathlib import Path


def read_vector(path: str | Path) -> np.ndarray:
    return np.loadtxt(path)


def read_multiplicity_table(
    path: str | Path,
    energies: np.ndarray,
    nmax: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(path)
    file_energies = raw[:, 0]
    counts = raw[:, 1 : nmax + 2]

    selected = []
    selected_energies = []

    for energy in energies:
        idx = np.where(np.isclose(file_energies, energy))[0]
        if len(idx) == 0:
            raise ValueError(f"Energy {energy} MeV not found in {path}")
        selected.append(counts[idx[0]])
        selected_energies.append(file_energies[idx[0]])

    counts = np.asarray(selected, dtype=float)
    stats = counts.sum(axis=1)

    with np.errstate(divide="raise", invalid="raise"):
        probabilities = counts / stats[:, None]

    return np.asarray(selected_energies), probabilities, stats


def write_table(path: str | Path, data: np.ndarray, header: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data, header=header)