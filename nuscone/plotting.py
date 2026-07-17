from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from .multichance import model_components, read_pnu_table, MultiChanceConfig, load_multichance_reference_data
from .references import load_references


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


def plot_pnu_3d(energies, pnu, figsize=(12, 9)):
    from matplotlib.colors import LinearSegmentedColormap

    biopunk_cmap = LinearSegmentedColormap.from_list(
        "biopunk",
        ["#000814", "#0a1628", "#0f4c0f", "#1a8a1a", "#39ff14", "#b8ff47", "#ffffff"],
    )

    fig = plt.figure(figsize=figsize, facecolor="#000814")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#000814")

    nmax = pnu.shape[1]
    nu = np.arange(nmax)
    NU, EN = np.meshgrid(nu, energies)

    norm = plt.Normalize(vmin=0.0, vmax=float(pnu.max()))

    ax.plot_surface(
        NU, EN, pnu,
        cmap=biopunk_cmap,
        norm=norm,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=True,
        alpha=0.92,
    )

    ax.plot_wireframe(
        NU, EN, pnu,
        color="#39ff14",
        linewidth=0.2,
        alpha=0.18,
        rstride=3,
        cstride=2,
    )

    neon = "#c77dff"

    ax.set_xlabel(r"$\nu$", fontsize=PLOT_STYLE["font_size"], labelpad=16, color=neon)
    ax.set_ylabel(r"$E_n$ (MeV)", fontsize=PLOT_STYLE["font_size"], labelpad=40, color=neon)
    # ax.set_zlabel(r"$P(\nu)$", fontsize=PLOT_STYLE["label_size"], labelpad=4, color=neon)

    ax.set_xlim(0, nmax - 1)
    ax.set_ylim(energies.min(), energies.max())
    ax.set_zlim(0.0, float(pnu.max()) * 1.1)

    ax.set_xticks([0, 5, 10])
    ax.set_yticks([5, 10, 15, 20, 25, 30])
    ax.set_zticks([])

    ax.tick_params(labelsize=PLOT_STYLE["tick_size"] - 4, pad=1, colors=neon)

    ax.view_init(elev=28, azim=-50)
    ax.set_box_aspect([1.2, 3.0, 1.0])

    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = True
        pane.set_facecolor("#000e06")
        pane.set_edgecolor("#0a3318")

    ax.grid(True, color="#0a2211", linewidth=0.4, alpha=0.6)

    ax.set_position([0.0, 0.02, 0.5, 0.96])
    return fig, ax


# multichance plots


def plot_multichance_pnu_fit(df, pnu_path, chosen_index=9, cfg=None):
    setup_publication_style()

    if cfg is None:
        cfg = MultiChanceConfig()

    if (
        len(cfg.en_prefission_238U) == 0
        or len(cfg.en_prefission_237U) == 0
        # or len(cfg.en_prefission_236U) == 0
    ):
        en_prefission_238u, en_prefission_237u = ( #en_prefission_236u
            load_multichance_reference_data(cfg)
        )
        cfg.en_prefission_238U = en_prefission_238u
        cfg.en_prefission_237U = en_prefission_237u
        # cfg.en_prefission_236U = en_prefission_236u

    energy, pnu = read_pnu_table(pnu_path)
    pnu = pnu[:len(df)]

    cfg.nmax = pnu.shape[1]

    row = df.iloc[chosen_index]

    n, c1, c2, c3, total = model_components( # c4
        row["energy"],
        chosen_index,
        row["p2"],
        row["p3"],
        #row["p4"],
        cfg,
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    plot_model(
        ax,
        "SCONE",
        n,
        pnu[chosen_index],
        label="SCONE",
        linestyle="",
        linewidth=0,
        markersize=10,
    )

    ax.plot(n, total, color="dimgrey", linestyle="-", linewidth=4, label="Total")
    ax.plot(n, c1, color="grey", linestyle="-", linewidth=3, label="First chance")
    ax.plot(n, c2, color="darkgrey", linestyle="--", linewidth=3, label="Second chance")
    ax.plot(n, c3, color="silver", linestyle=":", linewidth=3, label="Third chance")
    #ax.plot(n, c4, color="lightgrey", linestyle="-.", linewidth=3, label="Fourth chance")

    ax.set_xlabel("Neutron multiplicity", fontsize=PLOT_STYLE["label_size"])
    ax.set_ylabel("Probability", fontsize=PLOT_STYLE["label_size"])
    ax.set_xlim(-0.5, cfg.nmax - 0.5)
    ax.set_ylim(-0.02, 0.40)
    ax.set_xticks([0, 5, 10])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=PLOT_STYLE["label_size"],
        handlelength=1.6,
        borderpad=0.2,
        labelspacing=0.4,
        title=f"{row['energy']:.1f} MeV",
        title_fontsize=PLOT_STYLE["label_size"],
    )

    polish_axes(ax)
    fig.tight_layout()

    return fig, ax


def plot_multichance_moments(df):
    setup_publication_style()

    fig, ax = plt.subplots(figsize=(10, 10))

    nubar_err = df["nubar_err"].to_numpy() if "nubar_err" in df.columns else None

    errorbar_model(
        ax,
        "SCONE",
        df["energy"],
        df["nubar_exp"],
        yerr=nubar_err,
        label="SCONE",
        linestyle="none",
        linewidth=2,
        markersize=7,
        capsize=3,
        zorder=10,
    )

    color_fit = _style("SCONE")["color"]

    ax.plot(
        df["energy"],
        df["nubar_fit"],
        color=color_fit,
        linestyle="-",
        linewidth=3,
        label="Multi-chance fit",
        zorder=8,
    )

    if "nubar_fit_lo" in df.columns and "nubar_fit_hi" in df.columns:
        ax.fill_between(
            df["energy"],
            df["nubar_fit_lo"],
            df["nubar_fit_hi"],
            color=color_fit,
            alpha=0.25,
            linewidth=0,
            zorder=7,
        )

    ax.set_xlabel(
        "Incident neutron energy (MeV)",
        fontsize=PLOT_STYLE["label_size"],
    )
    ax.set_ylabel(
        "Average neutron multiplicity",
        fontsize=PLOT_STYLE["label_size"],
    )

    ax.set_xlim(0, 19)

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=PLOT_STYLE["legend_size"],
        handlelength=1.6,
        borderpad=0.2,
        labelspacing=0.4,
    )

    polish_axes(ax)
    fig.tight_layout()

    return fig, ax


def plot_multichance_sigma(df):
    setup_publication_style()

    fig, ax = plt.subplots(figsize=(10, 10))

    sigma_err = df["sigma_err"].to_numpy() if "sigma_err" in df.columns else None

    errorbar_model(
        ax,
        "SCONE",
        df["energy"],
        df["sigma_exp"],
        yerr=sigma_err,
        label="SCONE",
        linestyle="none",
        linewidth=2,
        markersize=7,
        capsize=3,
        zorder=10,
    )

    color_fit = _style("SCONE")["color"]

    ax.plot(
        df["energy"],
        df["sigma_fit"],
        color=color_fit,
        linestyle="-",
        linewidth=3,
        label="Multi-chance fit",
        zorder=8,
    )

    if "sigma_fit_lo" in df.columns and "sigma_fit_hi" in df.columns:
        ax.fill_between(
            df["energy"],
            df["sigma_fit_lo"],
            df["sigma_fit_hi"],
            color=color_fit,
            alpha=0.25,
            linewidth=0,
            zorder=7,
        )

    ax.set_xlabel(
        "Incident neutron energy (MeV)",
        fontsize=PLOT_STYLE["label_size"],
    )
    ax.set_ylabel(
        "Neutron multiplicity standard deviation",
        fontsize=PLOT_STYLE["label_size"],
    )

    ax.set_xlim(0, 19)

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=PLOT_STYLE["legend_size"],
        handlelength=1.6,
        borderpad=0.2,
        labelspacing=0.4,
    )

    polish_axes(ax)
    fig.tight_layout()

    return fig, ax

def plot_multichance_probabilities(df, reference_dir=Path("data/references")):

    setup_publication_style()

    refs = load_references(reference_dir)

    gef_ref = refs["GEF_MULTICHANCE"]
    cgmf_ref = refs["CGMF_MULTICHANCE"]

    fig, axes = plt.subplots(
        1,
        3, # 4, 5 for higher chances ...
        figsize=(14.5, 5.2),
        sharey=True,
        constrained_layout=False,
    )

    panel_labels = [
        "First chance",
        "Second chance",
        "Third chance",
        #"Fourth chance",
    ]

    scone_cols = ["p1", "p2", "p3"] #, "p4"]

    e_gef = gef_ref["energy"]
    e_cgmf = cgmf_ref["energy"]

    gef = [
        gef_ref["p1"],
        gef_ref["p2"],
        gef_ref["p3"],
        #gef_ref["p4"] if "p4" in gef_ref else np.zeros_like(e_gef),
    ]

    cgmf = [
        cgmf_ref["p1"],
        cgmf_ref["p2"],
        cgmf_ref["p3"],
        #cgmf_ref["p4"] if "p4" in cgmf_ref else np.zeros_like(e_cgmf),
    ]

    for i, (ax, text, col, gef_i, cgmf_i) in enumerate(
        zip(axes, panel_labels, scone_cols, gef, cgmf)
    ):

        plot_model(
            ax,
            "GEF",
            e_gef,
            gef_i,
            label="GEF" if i == 2 else None,
            linewidth=4,
        )

        plot_model(
            ax,
            "CGMF",
            e_cgmf,
            cgmf_i,
            label="CGMF" if i == 2 else None,
            linewidth=4,
        )

        err_low_col = "d" + col + "_low"
        err_up_col  = "d" + col + "_up"
        if err_low_col in df.columns and err_up_col in df.columns:
            yerr = [df[err_low_col][:17].to_numpy(), df[err_up_col][:17].to_numpy()]
        elif "d" + col in df.columns:
            yerr = df["d" + col][:17].to_numpy()
        else:
            yerr = None

        errorbar_model(
            ax,
            "SCONE",
            df["energy"][:17],
            df[col][:17],
            yerr=yerr,
            label="SCONE" if i == 2 else None,
            linewidth=2,
            markersize=6,
            capsize=3,
            zorder=10,
        )

        ax.set_title(text, fontsize=PLOT_STYLE["label_size"], pad=10)

        ax.set_xlim(0, 19)
        ax.set_ylim(-5, 110)

        ax.set_xticks([5,10,15])
        ax.set_yticks([0, 25, 50, 75, 100])

        polish_axes(
            ax,
            tick_size=PLOT_STYLE["label_size"],
            spine_width=3,
        )

    axes[0].set_ylabel(
        "Fission probability (%)",
        fontsize=PLOT_STYLE["label_size"],
        labelpad=20,
    )

    axes[2].legend(
        loc="upper left",
        frameon=False,
        fontsize=PLOT_STYLE["font_size"] - 2,
        handlelength=1.6,
        borderpad=0.2,
        labelspacing=0.4,
    )

    fig.supxlabel(
        "Incident neutron energy (MeV)",
        fontsize=PLOT_STYLE["label_size"],
        #y=0.04,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.99,
        bottom=0.18,
        top=0.88,
        wspace=0.08,
    )

    return fig, axes


def plot_multichance_excitation_sigma(df, refs=None, sn_en_cf252=8.5, sn_en_cf252_err=0.5, sn_en_238u=8.5, sn_en_238u_err=0.5, emax=18, figsize=(10, 10), thermal_points=None):
    setup_publication_style()
    fig, ax = plt.subplots(figsize=figsize)

    mask = df["energy"].to_numpy() <= emax
    E_exc = df["E_exc"].to_numpy()[mask]
    dE_exc = df["dE_exc"].to_numpy()[mask] if "dE_exc" in df.columns else None
    sigma_exp = df["sigma_exp"].to_numpy()[mask]
    sigma_err  = df["sigma_err"].to_numpy()[mask] if "sigma_err" in df.columns else None
    nubar_exp = df["nubar_exp"].to_numpy()[mask]

    sqrt_nubar_anchor = None
    sqrt_nubar_anchor_err = None
    cf_handle = None
    u238_handle = None
    sqrt_nubar_handle = None

    if refs is not None and "CF252_B3" in refs:
        from .models import delta_tke_to_sigma, delta_tke_to_sigma_err
        E_exc_cf = refs["CF252_B3"]["E_exc"]
        dtke = refs["CF252_B3"]["delta_TKE"]
        sigma_cf = delta_tke_to_sigma(dtke, sn_en_cf252)
        sigma_cf_err = delta_tke_to_sigma_err(dtke, sn_en_cf252, sn_en_cf252_err)
        cf_handle = ax.errorbar(E_exc_cf, sigma_cf, yerr=sigma_cf_err,
                    fmt="o", color="teal", markersize=12,
                    elinewidth=4, capsize=5, capthick=3, zorder=12,
                    label=r"Microscopic calculation $^{252}$Cf")

    if refs is not None and "U238_B3" in refs:
        from .models import delta_tke_to_sigma, delta_tke_to_sigma_err
        E_exc_u238 = refs["U238_B3"]["E_exc"]
        dtke_u238 = refs["U238_B3"]["delta_TKE"]
        sigma_u238 = delta_tke_to_sigma(dtke_u238, sn_en_238u)
        sigma_u238_err = delta_tke_to_sigma_err(dtke_u238, sn_en_238u, sn_en_238u_err)
        u238_handle = ax.errorbar(E_exc_u238[0], sigma_u238[0], yerr=sigma_u238_err[0],
                    fmt="*", color="blue", markersize=12,
                    elinewidth=3, capsize=5, capthick=3, zorder=12,
                    label=r"Microscopic calculation $^{238}$U")
        u238_handle.lines[2][0].set_linestyle("dashed")
        sqrt_nubar_anchor = np.interp(0.0, E_exc_u238, sigma_u238)
        sqrt_nubar_anchor_err = np.interp(0.0, E_exc_u238, sigma_u238_err)

    scone_handle = errorbar_model(
        ax, "SCONE", E_exc, sigma_exp,
        xerr=dE_exc, yerr=sigma_err,
        label=r"SCONE $^{238}$U(n$_{\rm fast}$,f)", linestyle="none",
        linewidth=2, markersize=7, capsize=3, zorder=10,
    )

    sqrt_nubar_band = None
    if sqrt_nubar_anchor is not None:
        E_exc_grid = np.linspace(0.0, 20, 300)

        # Simple model for the average neutron multiplicity
        nubar_0 = 2.5
        dnubar_dE = 0.1
        nubar_model = nubar_0 + dnubar_dE * E_exc_grid

        # Evolution law normalized to 1 at E_exc = 0
        scaling = np.sqrt(nubar_model / nubar_0)

        # Central theoretical prediction
        sigma_model = sqrt_nubar_anchor * scaling

        sqrt_nubar_handle, = ax.plot(
            E_exc_grid,
            sigma_model,
            color="steelblue",
            linestyle="--",
            linewidth=3,
            zorder=9,
            label=r"Statistical evolution model $\propto \sqrt{\bar{\nu}}$",
        )

        # Propagate the two extremities of the theoretical error bar
        if sqrt_nubar_anchor_err is not None:
            sigma_model_low = (
                sqrt_nubar_anchor - sqrt_nubar_anchor_err
            ) * scaling

            sigma_model_high = (
                sqrt_nubar_anchor + sqrt_nubar_anchor_err
            ) * scaling

            sqrt_nubar_band = ax.fill_between(
                E_exc_grid,
                sigma_model_low,
                sigma_model_high,
                color="lightskyblue",
                alpha=0.25,
                linewidth=0,
                zorder=8,
                #label=r"$\sigma \propto \sqrt{\bar{\nu}}$",
            )

    thermal_handles = []
    thermal_labels = []
    if thermal_points is not None:
        seen = set()
        for pt in thermal_points:
            label = pt.get("label", "")
            color = pt.get("color", "gray")
            marker = pt.get("marker", "s")
            E_exc_th = pt["E_exc"]
            sigmas = pt["sigmas"]
            for sigma_val in sigmas:
                h, = ax.plot(
                    E_exc_th, sigma_val,
                    marker=marker, color=color,
                    markersize=10, linestyle="none", zorder=11,
                    label="_nolegend_",
                )
                if label not in seen:
                    thermal_handles.append(h)
                    thermal_labels.append(label)
                    seen.add(label)

    ax.set_xlabel(
        r"Excitation energy (MeV)",
        fontsize=PLOT_STYLE["label_size"],
    )
    ax.set_ylabel(
        r"Neutron multiplicity standard-deviation",
        fontsize=PLOT_STYLE["label_size"],
    )

    # Experimental values: SCONE sigma + thermal isotopes (upper left)
    exp_handles = [scone_handle] + thermal_handles
    exp_labels = [scone_handle.get_label()] + thermal_labels
    exp_legend = ax.legend(
        exp_handles, exp_labels,
        loc="upper left", frameon=False,
        fontsize=PLOT_STYLE["legend_size"],
        handlelength=1.6, borderpad=0.2, labelspacing=0.4,
    )
    ax.add_artist(exp_legend)

    # Models: sqrt(nubar) fit, microscopic 238U, microscopic 252Cf (lower right)
    model_handles = [h for h in [cf_handle, u238_handle, sqrt_nubar_handle] if h is not None]
    model_labels = [h.get_label() for h in model_handles]
    ax.legend(
        model_handles, model_labels,
        loc="lower right", frameon=False,
        fontsize=PLOT_STYLE["legend_size"],
        handlelength=1.6, borderpad=0.2, labelspacing=0.4,
    )

    polish_axes(ax)
    fig.tight_layout()
    ax.set_ylim(0.7,1.7)
    ax.set_xlim(-0.5,20.5)
    ax.set_xticks([0,5,10,15,20])
    ax.set_yticks([1.0, 1.2, 1.4])
    return fig, ax