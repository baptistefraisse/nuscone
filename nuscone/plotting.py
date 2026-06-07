from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec


MODEL_STYLES = {
    "SCONE": {
        "color": "red",
        "marker": "s",
        "markersize": 5,
        "linewidth": 3,
        "linestyle": "none",
        "capsize": 5,
        "elinewidth": 2,
        "capthick": 2,
        "zorder": 10,
    },
    "GEF": {
        "color": "black",
        "marker": None,
        "markersize": 7,
        "linewidth": 5,
        "linestyle": "-",
        "capsize": 0,
        "elinewidth": 2,
        "capthick": 2,
        "zorder": 2,
    },
    "CGMF": {
        "color": "blue",
        "marker": None,
        "markersize": 7,
        "linewidth": 5,
        "linestyle": "-",
        "capsize": 0,
        "elinewidth": 2,
        "capthick": 2,
        "zorder": 2,
    },
    # "JEFF": {
    #     "color": "orange",
    #     "marker": "o",
    #     "markersize": 9,
    #     "linewidth": 0,
    #     "linestyle": "none",
    #     "capsize": 6,
    #     "elinewidth": 2,
    #     "capthick": 2,
    #     "zorder": 3,
    # },
    "JEFF": {
        "color": "orange",
        "marker": None,
        "linewidth": 5,
        "linestyle": "-",
        "zorder": 3,
    },
    "ENDF": {
        "color": "purple",
        "marker": "o",
        "markersize": 9,
        "linewidth": 0,
        "linestyle": "none",
        "capsize": 6,
        "elinewidth": 2,
        "capthick": 2,
        "zorder": 3,
    },
    "FREHAUT": {
        "color": "purple",
        "marker": "D",
        "markersize": 9,
        "linewidth": 0,
        "linestyle": "none",
        "capsize": 6,
        "elinewidth": 2,
        "capthick": 2,
        "zorder": 3,
    },
}


PLOT_STYLE = {
    "font_size": 25,
    "label_size": 30,
    "label_size_large": 40,
    "tick_size": 28,
    "tick_size_large": 40,
    "legend_size": 30,
    "legend_size_large": 32,
    "legend_title_size": 30,
    "panel_text_size": 18,
    "axes_linewidth": 3,
    "major_tick_size": 10,
    "major_tick_width": 2,
    "minor_tick_size": 7,
    "minor_tick_width": 1.5,
    "zero_linewidth": 1.5,
}


def _model_name(name):
    aliases = {
        "work": "SCONE",
        "gef": "GEF",
        "cgmf": "CGMF",
        "jeff": "JEFF",
        "endf": "ENDF",
        "frehaut": "FREHAUT",
        "GEF_PNU": "GEF",
    }
    return aliases.get(name, name)


def _style(name):
    return MODEL_STYLES.get(_model_name(name), MODEL_STYLES["SCONE"])


def _fmt(name):
    marker = _style(name).get("marker")
    return "" if marker is None else marker


def setup_publication_style(font_size=None):
    if font_size is None:
        font_size = PLOT_STYLE["font_size"]

    plt.rcParams.update(
        {
            "font.size": font_size,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.linewidth": PLOT_STYLE["axes_linewidth"],
            "legend.frameon": False,
        }
    )


def polish_axes(ax, tick_size=None, spine_width=None):
    if tick_size is None:
        tick_size = PLOT_STYLE["tick_size"]
    if spine_width is None:
        spine_width = PLOT_STYLE["axes_linewidth"]

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_size,
        size=PLOT_STYLE["major_tick_size"],
        width=PLOT_STYLE["major_tick_width"],
        direction="in",
        top=True,
        right=True,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        size=PLOT_STYLE["minor_tick_size"],
        width=PLOT_STYLE["minor_tick_width"],
        direction="in",
        top=True,
        right=True,
    )
    ax.minorticks_on()

    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)


def errorbar_model(ax, model, x, y, *, xerr=None, yerr=None, label=None, **overrides):
    st = {**_style(model), **overrides}

    return ax.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        color=st["color"],
        fmt=_fmt(model),
        linestyle=st.get("linestyle", "none"),
        linewidth=st.get("linewidth", 3),
        markersize=st.get("markersize", 5),
        capsize=st.get("capsize", 5),
        elinewidth=st.get("elinewidth", 2),
        capthick=st.get("capthick", 2),
        label=label,
        zorder=st.get("zorder", 1),
    )


def plot_model(ax, model, x, y, *, label=None, **overrides):
    st = {**_style(model), **overrides}
    marker = st.get("marker")

    return ax.plot(
        x,
        y,
        color=st["color"],
        marker="" if marker is None else marker,
        linestyle=st.get("linestyle", "-"),
        linewidth=st.get("linewidth", 3),
        markersize=st.get("markersize", 5),
        label=label,
        zorder=st.get("zorder", 1),
    )


def savefig(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, transparent=False, bbox_inches="tight")


def plot_nubar_publication(
    energies,
    nubar,
    nubar_err=None,
    energy_err=None,
    references=None,
):
    setup_publication_style()

    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    ax_up = plt.subplot(gs[0])
    ax_down = plt.subplot(gs[1], sharex=ax_up)

    plt.setp(ax_up.get_xticklabels(), visible=False)
    plt.subplots_adjust(hspace=0.0)

    if references:
        if "JEFF" in references:
            ref = references["JEFF"]
            # errorbar_model(
            #     ax_up,
            #     "JEFF",
            #     ref["energy"],
            #     ref["nubar"],
            #     yerr=ref.get("nubar_err"),
            #     label="JEFF-3.3",
            # )
            plot_model(
                ax_up,
                "JEFF",
                ref["energy"],
                ref["nubar"],
                label="JEFF-3.3",
            )

        if "GEF" in references:
            ref = references["GEF"]
            plot_model(ax_up, "GEF", ref["energy"], ref["nubar"], label="GEF")

        if "CGMF" in references:
            ref = references["CGMF"]
            plot_model(ax_up, "CGMF", ref["energy"], ref["nubar"], label="CGMF")

    errorbar_model(
        ax_up,
        "SCONE",
        energies,
        nubar,
        xerr=energy_err,
        yerr=nubar_err,
        label="SCONE",
        linewidth=4,
    )

    if references:
        for name in ["JEFF", "GEF", "CGMF"]:
            if name not in references:
                continue

            ref = references[name]
            common_e, diff = [], []

            for e, y in zip(energies, nubar):
                idx = np.where(np.isclose(ref["energy"], e))[0]
                if len(idx):
                    common_e.append(e)
                    diff.append(100.0 * (y - ref["nubar"][idx[0]]) / y)

            if _style(name).get("marker") is None:
                plot_model(ax_down, name, common_e, diff)
            else:
                errorbar_model(ax_down, name, common_e, diff)

    yerr_residual = None
    if nubar_err is not None:
        yerr_residual = 100.0 * np.asarray(nubar_err) / np.asarray(nubar)

    errorbar_model(
        ax_down,
        "SCONE",
        energies,
        np.zeros_like(energies),
        yerr=yerr_residual,
        xerr=energy_err,
        linewidth=3,
    )

    ax_up.set_ylabel("Average neutron multiplicity", fontsize=PLOT_STYLE["label_size_large"], labelpad=50)
    ax_down.set_ylabel("Difference (%)", fontsize=PLOT_STYLE["label_size_large"], labelpad=25)
    ax_down.set_xlabel("Incident neutron energy (MeV)", fontsize=PLOT_STYLE["label_size_large"])

    ax_up.set_xlim(-0.5, 31.5)
    ax_up.set_ylim(2.25, 6.45)
    ax_down.set_ylim(-7, 7)

    ax_down.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax_up.set_yticks([3, 4, 5, 6])
    ax_down.set_yticks([-5, 0, 5])

    ax_up.legend(
        fontsize=PLOT_STYLE["legend_size_large"],
        loc="upper left",
        frameon=False,
        handlelength=1.6,
        borderpad=0.2,
        labelspacing=0.5,
    )

    for ax in [ax_up, ax_down]:
        polish_axes(ax, tick_size=PLOT_STYLE["tick_size_large"], spine_width=3)

    fig.tight_layout()
    plt.subplots_adjust(hspace=0.0)

    return fig, (ax_up, ax_down)


def plot_sigma_publication(
    energies,
    sigma,
    sigma_err=None,
    energy_err=None,
    references=None,
):
    setup_publication_style()

    fig, ax = plt.subplots(figsize=(10, 10))

    if references:
        for name, ref in references.items():
            if "sigma" not in ref:
                continue
            model = "GEF" if name == "GEF_PNU" else name
            plot_model(ax, model, ref["energy"], ref["sigma"], label=name)

    errorbar_model(
        ax,
        "SCONE",
        energies,
        sigma,
        xerr=energy_err,
        yerr=sigma_err,
        label="SCONE",
    )

    ax.set_xlabel("Incident neutron energy (MeV)", fontsize=PLOT_STYLE["label_size"])
    ax.set_ylabel("Neutron multiplicity standard deviation", fontsize=PLOT_STYLE["label_size"])
    ax.legend(fontsize=PLOT_STYLE["legend_size"], loc="upper left")

    polish_axes(ax)
    fig.tight_layout()

    return fig, ax


def plot_moment_publication(
    energies,
    values,
    order,
    moment_err=None,
    energy_err=None,
    references=None,
):
    setup_publication_style()

    key = f"f{order}"
    ylabel = {
        2: "Second factorial moment",
        3: "Third factorial moment",
        4: "Fourth factorial moment",
    }[order]

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    ax_up = plt.subplot(gs[0])
    ax_down = plt.subplot(gs[1], sharex=ax_up)

    plt.setp(ax_up.get_xticklabels(), visible=False)
    plt.subplots_adjust(hspace=0.0)

    if references:
        for name, ref in references.items():
            if key not in ref:
                continue

            model = "GEF" if name == "GEF_PNU" else name
            label = "GEF" if name == "GEF_PNU" else name

            plot_model(ax_up, model, ref["energy"], ref[key], label=label, linewidth=4)

            common_e, diff = [], []
            for e, y in zip(energies, values):
                idx = np.where(np.isclose(ref["energy"], e))[0]
                if len(idx):
                    common_e.append(e)
                    diff.append(100.0 * (y - ref[key][idx[0]]) / y)

            plot_model(ax_down, model, common_e, diff, linewidth=4)

    errorbar_model(
        ax_up,
        "SCONE",
        energies,
        values,
        xerr=energy_err,
        yerr=moment_err,
        label="SCONE",
    )

    errorbar_model(
        ax_down,
        "SCONE",
        energies,
        np.zeros_like(energies),
        yerr=None if moment_err is None else 100.0 * np.asarray(moment_err) / np.asarray(values),
        xerr=energy_err,
    )

    ax_up.set_ylabel(ylabel, fontsize=PLOT_STYLE["label_size"])
    ax_down.set_ylabel("Difference (%)", fontsize=PLOT_STYLE["label_size"])
    ax_down.set_xlabel("Incident neutron energy (MeV)", fontsize=PLOT_STYLE["label_size"])

    ax_down.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax_up.legend(fontsize=PLOT_STYLE["legend_size"], loc="upper left", frameon=False)

    for ax in [ax_up, ax_down]:
        polish_axes(ax)

    fig.tight_layout()
    plt.subplots_adjust(hspace=0.0)

    return fig, (ax_up, ax_down)


def plot_pnu_publication(
    energy,
    bins,
    pnu,
    pnu_err=None,
    references=None,
):
    setup_publication_style()

    fig, ax = plt.subplots(figsize=(10, 10))

    if references:
        for name, values in references.items():
            model = "GEF" if name == "GEF_PNU" else name
            label = "GEF" if name == "GEF_PNU" else name
            plot_model(ax, model, bins, values, label=label)

    errorbar_model(
        ax,
        "SCONE",
        bins,
        pnu,
        yerr=pnu_err,
        label="SCONE",
        linewidth=5,
        markersize=3,
    )

    ax.set_xlabel("Neutron multiplicity", fontsize=PLOT_STYLE["label_size"])
    ax.set_ylabel("Probability", fontsize=PLOT_STYLE["label_size"])
    ax.set_ylim(-0.05, 0.38)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.legend(
        fontsize=PLOT_STYLE["legend_size"],
        loc="upper right",
        title=f"Uranium 238 ({energy:g} MeV)",
        title_fontsize=PLOT_STYLE["legend_title_size"],
    )

    polish_axes(ax)
    fig.tight_layout()

    return fig, ax


def plot_lambda_publication(energies, lambdas):
    setup_publication_style()

    fig, ax = plt.subplots(figsize=(10, 10))

    st = _style("SCONE")
    ax.scatter(
        energies,
        lambdas,
        color=st["color"],
        marker=st["marker"] or "s",
        s=st["markersize"] ** 2,
        zorder=st["zorder"],
    )

    ax.set_yscale("log")
    ax.set_xlabel("Incident neutron energy (MeV)", fontsize=PLOT_STYLE["label_size"])
    ax.set_ylabel("Regularization parameter", fontsize=PLOT_STYLE["label_size"])

    polish_axes(ax)
    fig.tight_layout()

    return fig, ax


def plot_pnu_triptych_publication(
    energies_to_plot,
    all_energies,
    bins,
    pnu,
    pnu_err=None,
    references=None,
):
    setup_publication_style()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.5, 5.2),
        sharey=True,
        constrained_layout=False,
    )

    for ax, energy in zip(axes, energies_to_plot):
        idx = list(all_energies).index(float(energy))

        if references is not None and "GEF_PNU" in references:
            gef_e = references["GEF_PNU"]["energy"]
            gef_pnu = references["GEF_PNU"]["pnu"]

            idx_gef = np.argmin(np.abs(gef_e - energy))
            nu = np.arange(gef_pnu.shape[1])

            plot_model(
                ax,
                "GEF",
                nu,
                gef_pnu[idx_gef],
                label="GEF" if ax is axes[0] else None,
                linewidth=3,
            )

        errorbar_model(
            ax,
            "SCONE",
            bins,
            pnu[idx],
            yerr=None if pnu_err is None else pnu_err[idx],
            label="SCONE" if ax is axes[0] else None,
            linewidth=1.8,
            markersize=6,
            capsize=3,
        )

        ax.text(
            0.80,
            0.90,
            f"{energy:g} MeV",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=PLOT_STYLE["panel_text_size"],
        )

        ax.set_xlim(-0.5, 12.5)
        ax.set_ylim(-0.01, 0.40)

        ax.set_xticks([0, 5, 10])
        ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
        ax.set_box_aspect(1)

        polish_axes(ax, tick_size=22, spine_width=1.8)

        if ax is axes[0]:
            ax.legend(
                loc="lower left",
                bbox_to_anchor=(0.4, 0.50),
                frameon=False,
                fontsize=PLOT_STYLE["panel_text_size"],
                handlelength=1.4,
            )

    axes[0].set_ylabel("Probability", fontsize=22)
    fig.supxlabel("Neutron multiplicity", fontsize=22, y=0.05)

    fig.subplots_adjust(
        left=0.2,
        right=0.9,
        bottom=0.18,
        top=0.82,
        wspace=0.05,
    )

    return fig, axes