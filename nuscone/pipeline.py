import numpy as np
import pandas as pd

from .config import Config
from .io import read_vector, read_multiplicity_table, write_table
from .operators import response_operator, derivative_operator
from .criteria import scan_lambdas, choose_lambda
from .regularization import solve_tikhonov_nnls, solve_tikhonov_constrained
from .moments import mean, summarize_distribution
from .uncertainty import propagate_distribution_errors, mean_error, sigma_error, f2_error, f3_error
from .tof import energy_error 


def run_analysis(config: Config) -> dict:
    analysis = config.analysis
    reg = config.regularization
    paths = config.paths

    energies = analysis.energies
    bins = analysis.bins

    noise_low = read_vector(paths.noise_low)
    noise_high = read_vector(paths.noise_high)

    _, y_low, stat_low = read_multiplicity_table(paths.raw_low, energies, analysis.nmax)
    _, y_high, stat_high = read_multiplicity_table(paths.raw_high, energies, analysis.nmax)

    A_low = response_operator(noise_low, analysis.nmax, analysis.efficiency)
    A_high = response_operator(noise_high, analysis.nmax, analysis.efficiency)

    D = derivative_operator(analysis.nmax, reg.derivative_order)

    unfolded = []
    unfolded_err = []
    lambda_opt = []
    rows = []

    energy_err = energy_error(
        energies,
        distance=config.tof.distance,
        distance_error=config.tof.distance_error,
        time_error=config.tof.time_error,
    )

    for i, energy in enumerate(energies):

        if energy <= analysis.low_energy_threshold:
            y = y_low[i]
            A = A_low
            noise = noise_low
            stat = stat_low[i]
        else:
            y = y_high[i]
            A = A_high
            noise = noise_high
            stat = stat_high[i]

        noise_mean = mean(noise)
        nubar_target = (mean(y) - noise_mean) / analysis.efficiency

        scan = scan_lambdas(
            A,
            y,
            D,
            reg.lambdas,
            nubar_target=nubar_target,
        )

        lam = choose_lambda(scan, reg.lambda_choice)
        x, _ = solve_tikhonov_constrained(A, y, D, lam, nubar_target)
        xerr = propagate_distribution_errors(A, D, lam, y, stat)
        
        summary = summarize_distribution(x)
        summary["energy"] = energy
        summary["lambda"] = lam
        summary["stat"] = stat
        summary["energy_err"] = energy_err[i]   
        summary["nubar_err"] = mean_error(x, xerr)
        summary["sigma_err"] = sigma_error(x, xerr)
        summary["f2_err"] = f2_error(x, xerr)
        summary["f3_err"] = f3_error(x, xerr)
        unfolded.append(x)
        unfolded_err.append(xerr)
        lambda_opt.append(lam)
        rows.append(summary)

    results = {
        "energies": energies,
        "bins": bins,
        "pnu": np.asarray(unfolded),
        "pnu_err": np.asarray(unfolded_err),
        "lambda": np.asarray(lambda_opt),
        "moments": pd.DataFrame(rows),
    }

    return results


def save_results(results: dict, output_dir) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    write_table(output_dir / "tables" / "pnu.txt", results["pnu"])
    write_table(output_dir / "tables" / "pnu_err.txt", results["pnu_err"])

    results["moments"].to_csv(output_dir / "tables" / "moments.csv", index=False)