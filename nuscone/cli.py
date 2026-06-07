import argparse
from pathlib import Path

from .config import load_config
from .pipeline import run_analysis, save_results
from .references import load_references

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
            plot_moment_publication,
            savefig,
        )

        moments = results["moments"]
        energies = results["energies"]
        bins = results["bins"]
        refs = load_references(config.paths.reference_dir)

        # nubar

        fig, _ = plot_nubar_publication(
            moments["energy"].to_numpy(),
            moments["nubar"].to_numpy(),
            nubar_err=moments["nubar_err"].to_numpy(),
            energy_err=moments["energy_err"].to_numpy(),
            references=refs,
        )
        savefig(fig, config.paths.output_dir / "figures" / "238U_SCONE_nubar.png")

        # sigma

        fig, _ = plot_sigma_publication(
            moments["energy"].to_numpy(),
            moments["sigma"].to_numpy(),
            sigma_err=moments["sigma_err"].to_numpy(),
            energy_err=moments["energy_err"].to_numpy(),
            references=refs,
        )
        savefig(fig, config.paths.output_dir / "figures" / "238U_SCONE_sigma.png")

        # lambda regularization

        fig, _ = plot_lambda_publication(
            moments["energy"].to_numpy(),
            moments["lambda"].to_numpy(),
        )
        savefig(fig, config.paths.output_dir / "figures" / "238U_SCONE_lambda.png")

        # pnu separated plots

        # for energy in [2, 5, 10, 14, 20, 25]:
        #     idx = list(energies).index(float(energy))
        #     fig, _ = plot_pnu_publication(
        #         energy,
        #         bins,
        #         results["pnu"][idx],
        #         results["pnu_err"][idx],
        #     )
        #     savefig(fig, config.paths.output_dir / "figures" / f"238U_SCONE_Pnu_{energy}MeV.png")

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
            "238U_SCONE_Pnu_triptych.png"
        )

        # factorial moments

        for order in [2, 3]:
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
                config.paths.output_dir / "figures" / f"238U_SCONE_f{order}.png",
            )


if __name__ == "__main__":
    main()