import argparse
from pathlib import Path
import numpy as np
from .config import load_config
from .pipeline import run_analysis, save_results
from .references import load_references
from .multichance import extract_multichance_probabilities, MultiChanceConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="NUSCONE neutron multiplicity unfolding")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config file")
    parser.add_argument("--plots", action="store_true", help="Generate standard figures")
    args = parser.parse_args()

    config = load_config(args.config)
    results = run_analysis(config)
    save_results(results, config.paths.output_dir)

    if args.plots:
        from .plotting import (
            plot_nubar_publication,
            plot_sigma_publication,
            plot_pnu_publication,
            plot_lambda_publication,
            plot_pnu_triptych_publication,
            plot_pnu_3d,
            plot_moment_publication,
            plot_multichance_probabilities,
            plot_multichance_pnu_fit,
            plot_multichance_moments,
            plot_multichance_sigma,
            plot_multichance_probabilities,
            plot_multichance_excitation_sigma,
            savefig,
        )

        moments = results["moments"]
        energies = results["energies"]
        bins = results["bins"]
        stat = results["stat"]
        refs = load_references(config.paths.reference_dir)

        # nubar

        fig, _ = plot_nubar_publication(
            moments["energy"].to_numpy(),
            moments["nubar"].to_numpy(),
            nubar_err=moments["nubar_err"].to_numpy(),
            energy_err=moments["energy_err"].to_numpy(),
            references=refs,
        )
        savefig(fig, config.paths.output_dir / "figures" / "238U_SCONE_nubar.pdf")

        # sigma

        fig, _ = plot_sigma_publication(
            moments["energy"].to_numpy(),
            moments["sigma"].to_numpy(),
            sigma_err=moments["sigma_err"].to_numpy(),
            energy_err=moments["energy_err"].to_numpy(),
            references=refs,
        )
        savefig(fig, config.paths.output_dir / "figures" / "238U_SCONE_sigma.pdf")

        # lambda regularization

        fig, _ = plot_lambda_publication(
            moments["energy"].to_numpy(),
            moments["lambda"].to_numpy(),
        )
        savefig(fig, config.paths.output_dir / "figures" / "238U_SCONE_lambda.pdf")

        # pnu separated plots

        # for energy in [2, 5, 10, 14, 20, 25]:
        #     idx = list(energies).index(float(energy))
        #     fig, _ = plot_pnu_publication(
        #         energy,
        #         bins,
        #         results["pnu"][idx],
        #         results["pnu_err"][idx],
        #     )
        #     savefig(fig, config.paths.output_dir / "figures" / f"238U_SCONE_Pnu_{energy}MeV.pdf")

        # pnu same plot

        fig, _ = plot_pnu_triptych_publication(
            energies_to_plot=[5, 14, 20],
            all_energies=energies,
            bins=bins,
            pnu=results["pnu"],
            pnu_err=results["pnu_err"],
            references=refs,
        )
        savefig(
            fig,
            config.paths.output_dir /
            "figures" /
            "238U_SCONE_Pnu_triptych.pdf"
        )

        fig, _ = plot_pnu_3d(energies[:-1], results["pnu"][:-1])
        savefig(fig, config.paths.output_dir / "figures" / "238U_SCONE_pnu_3d.pdf")

        # factorial moments

        for order in [2, 3, 4]:
            fig, _ = plot_moment_publication(
                energies=moments["energy"].to_numpy(),
                values=moments[f"f{order}"].to_numpy(),
                moment_err=moments[f"f{order}_err"].to_numpy(),
                energy_err=moments["energy_err"].to_numpy(),
                order=order,
                references=refs,
            )

            savefig(
                fig,
                config.paths.output_dir / "figures" / f"238U_SCONE_f{order}.pdf",
            )

        # multichance fission

        pnu_path = Path("results/tables/pnu.txt")
        multichance_dir = Path("results/tables")

        multichance = extract_multichance_probabilities(
            pnu_path=pnu_path,
            output_dir=multichance_dir,
            cfg=MultiChanceConfig(),
            stat=stat,
        )

        multichance = multichance.merge(
            results["moments"][["energy", "nubar_err", "sigma_err", "energy_err"]],
            on="energy",
            how="left"
        )
        multichance["dE_exc"] = np.sqrt(
            multichance["energy_err"]**2 + multichance["dE_exc_proba"]**2
        )

        fig, _ = plot_multichance_pnu_fit(
            multichance,
            config.paths.output_dir / "tables" / "pnu.txt",
            chosen_index=9,
        )
        savefig(fig, config.paths.output_dir / "figures" / "238U_SCONE_multichance_pnu_fit.pdf")

        fig, _ = plot_multichance_moments(multichance)
        savefig(fig, config.paths.output_dir / "figures" / "238U_SCONE_multichance_nubar_check.pdf")

        fig, _ = plot_multichance_sigma(multichance)
        savefig(fig, config.paths.output_dir / "figures" / "238U_SCONE_multichance_sigma_check.pdf")

        fig, _ = plot_multichance_probabilities(multichance)
        savefig(fig, config.paths.output_dir / "figures" / "238U_SCONE_multichance_probabilities.pdf")

        # excitation energy studies

        thermal_points = [
            {
                "label": r"$^{235}$U(n$_{\rm th}$,f)",
                "E_exc": 6.545,   # Sn(U-236)
                "sigmas": [0.5*(1.088+1.070)],
                "color": "black",
                "marker": "p",
            },
            {
                "label": r"$^{239}$Pu(n$_{\rm th}$,f)",
                "E_exc": 6.534,   # Sn(Pu-240)
                "sigmas": [1.140],
                "color": "gray",
                "marker": "^",
            },
            {
                "label": r"$^{241}$Pu(n$_{\rm th}$,f)",
                "E_exc": 5.902,   # Sn(Pu-242)
                "sigmas": [1.150],
                "color": "purple",
                "marker": "D",
            },
        ]
        fig, _ = plot_multichance_excitation_sigma(multichance, refs=refs, thermal_points=thermal_points)
        savefig(fig, config.paths.output_dir / "figures" / "238U_SCONE_excitation_sigma.pdf")


if __name__ == "__main__":
    main()