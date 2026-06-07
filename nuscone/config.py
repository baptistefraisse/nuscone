from dataclasses import dataclass
from pathlib import Path
import yaml
import numpy as np


@dataclass
class AnalysisConfig:
    isotope: str
    nmax: int
    emin: float
    emax: float
    estep: float
    efficiency: float
    low_energy_threshold: float

    @property
    def energies(self) -> np.ndarray:
        return np.arange(self.emin, self.emax, self.estep)

    @property
    def bins(self) -> np.ndarray:
        return np.arange(0, self.nmax + 1)


@dataclass
class RegularizationConfig:
    lambda_min: float
    lambda_max: float
    lambda_points: int
    derivative_order: int
    lambda_choice: str

    @property
    def lambdas(self) -> np.ndarray:
        return np.geomspace(self.lambda_min, self.lambda_max, self.lambda_points)


@dataclass
class PathConfig:
    raw_high: Path
    raw_low: Path
    noise_high: Path
    noise_low: Path
    reference_dir: Path
    output_dir: Path


@dataclass
class TOFConfig:
    distance: float
    distance_error: float
    time_error: float


@dataclass
class Config:
    analysis: AnalysisConfig
    regularization: RegularizationConfig
    paths: PathConfig
    tof: TOFConfig


def load_config(path: str | Path) -> Config:
    with open(path, "r") as stream:
        raw = yaml.safe_load(stream)

    return Config(
        analysis=AnalysisConfig(**raw["analysis"]),
        regularization=RegularizationConfig(**raw["regularization"]),
        paths=PathConfig(**{k: Path(v) for k, v in raw["paths"].items()}),
        tof=TOFConfig(**raw["tof"]),
    )

