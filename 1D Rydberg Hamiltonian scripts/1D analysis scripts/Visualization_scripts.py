## Importing relevant libraries
import numpy as np
import scipy
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, eig
from scipy.sparse.linalg import eigs
from itertools import product
from functools import reduce
from pathlib import Path
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.patches import Circle, Rectangle
from scipy.interpolate import PchipInterpolator
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
from matplotlib.ticker import MultipleLocator, FuncFormatter, MaxNLocator
from scipy.signal import find_peaks
import pickle
from collections import Counter
import h5py
from scipy.interpolate import interp1d
from collections import defaultdict, OrderedDict
import csv
import matplotlib.pyplot as plt
from fractions import Fraction
import math
import time
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import jv
from scipy.integrate import quad
from scipy.interpolate import griddata

# Libaries for GL fitting procedure
import math
from numpy.polynomial.laguerre import laggauss
from scipy.optimize import least_squares, minimize
from scipy.optimize import curve_fit

## Simple helper functions for basic visualization
def plot_correlation_vs_distance(selected, rep_points, all_mps, N):
    """
    Plot connected correlation functions C(r) vs distance r
    for only the uniform, period-2, and period-3 phases.
    """

    plt.figure(figsize=(7,5), dpi=600)

    for label in selected:
        Rb, delta = rep_points[label]
        mps = all_mps[(Rb, delta)]
        C_r = correlation_vs_distance(mps, N)
        legend_label = fr"$R_b = {Rb:.2f}$, $\delta = {delta:.2f}$"
        plt.plot(range(N), C_r, 'o-', label=legend_label)

    plt.xlabel(r"Distance $r$", fontsize=19)
    plt.ylabel(r"$C(r) = \langle n_j n_{j+r}\rangle$", fontsize=19)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.legend(loc='best', fontsize=17)
    plt.title("Site-averaged correlation function", fontsize=20)
    plt.grid(True)
    plt.tight_layout(pad=1.0)
    plt.show()

def plot_correlation_maps(selected, rep_points, all_mps, N, max_sites=None):
    """
    Plot density-density connected correlation maps using matplotlib imshow
    for uniform, period-2, and period-3 phases (publication-ready).
    """

    fig, axes = plt.subplots(
        1, len(selected), figsize=(18, 6), dpi=600, constrained_layout=True
    )

    vmax = None
    corr_matrices = {}
    for label in selected:
        Rb, delta = rep_points[label]
        mps = all_mps[(Rb, delta)]
        corr_matrix = get_correlation_matrix(mps, N)
        if max_sites is not None:
            corr_matrix = corr_matrix[:max_sites, :max_sites]
        corr_matrices[label] = corr_matrix
        local_max = np.max(np.abs(corr_matrix))
        vmax = local_max if vmax is None else max(vmax, local_max)

    for ax, label in zip(axes, selected):
        corr_matrix = corr_matrices[label]
        im = ax.imshow(
            corr_matrix,
            cmap="seismic",
            vmin=-vmax,
            vmax=vmax,
            origin="lower",
            aspect="equal",
        )
        Rb, delta = rep_points[label]
        ax.set_title(fr"$R_b = {Rb:.2f}$, $\delta = {delta:.2f}$", fontsize=20)
        ax.set_xlabel("Site $l$", fontsize=19)
        ax.set_ylabel("Site $j$", fontsize=19)
        ax.tick_params(axis="both", which="major", labelsize=17)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
    cbar.set_label(r"$\langle n_j n_l\rangle$", fontsize=19)
    cbar.ax.tick_params(labelsize=17)

    plt.show()

def plot_periodic_sites_lines(labels, all_mps, representative_points, N, max_xticks=10, figsize=(15,4)):
    """
    Plot per-site excitation densities for period-2 and period-3 representative points.
    """

    plt.figure(figsize=figsize, dpi=600)

    for label in labels:
        Rb, delta = representative_points[label]
        mps = all_mps[(Rb, delta)]
        # Compute per-site densities
        n_sites = np.array([get_site_excitation_probability(mps, N, j) for j in range(N)])
        # Plot all sites (no skipping)
        plt.plot(np.arange(N), n_sites, marker='o',
                 label = fr"$R_b = {Rb:.2f}$, $\delta = {delta:.2f}$", linewidth=2)

    plt.xlabel('Site index', fontsize=19)
    plt.ylabel(r'Excitation density $\langle n_j \rangle$', fontsize=19)
    plt.title('Local excitation densities', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=17)

    # Limit the number of x-tick labels for readability
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=max_xticks))

    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=17)
    plt.tight_layout(pad=1.0)
    plt.show()

def plot_density_fluctuations(all_mps, representative_points, N, site_skip=1, figsize=(15,4)):
    """
    Plot per-site density fluctuations for period-2 and period-3 representative points,
    optionally skipping sites for clarity.

    Parameters
    ----------
    all_mps : dict
        Dictionary of MPSs: keys = (Rb, delta), values = MPS
    representative_points : dict
        Representative points for period-2 and period-3:
        keys = label, values = (Rb, delta)
    N : int
        Number of sites
    site_skip : int
        Plot every 'site_skip'-th site for clarity
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize, dpi=600)

    for label in ["Point-2", "Point-3", "Point-4"]:
        Rb, delta = representative_points[label]
        mps = all_mps[(Rb, delta)]
        # Compute per-site densities
        n_sites = np.array([get_site_excitation_probability(mps, N, j) for j in range(N)])
        # Compute mean density
        n_mean = np.mean(n_sites)
        # Compute fluctuations
        delta_n = n_sites - n_mean
        # Select sites for plotting
        sites_to_plot = np.arange(0, N, site_skip)
        plt.plot(sites_to_plot, delta_n[sites_to_plot], marker='o',
                 label=f'{label}', linewidth=2)

    plt.xlabel('Site index', fontsize=19)
    plt.ylabel(r'Fluctuation $\delta n_j = \langle n_j \rangle - \bar{n}$', fontsize=19)
    plt.title(r'$100$-site Rydberg atom chain ($\Omega = 1$)', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.xticks(sites_to_plot)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=17)
    plt.tight_layout(pad=1.0)
    plt.show()

## Helper functions to plot the 1D phase diagram + other panels
def add_zq_schematic(fig, *, left, bottom, width, height, n=10):
    ax_s = fig.add_axes([left, bottom, width, height])
    ax_s.set_axis_off()
    ax_s.set_xlim(0, 1)
    ax_s.set_ylim(0, 1)

    # --- vertical layout (just spacing between rows) ---
    pad_y = 0.15

    # --- horizontal geometry (exact tangency) ---
    r = 1.0 / (2 * n)        # because rectangle width = 1
    dx = 2 * r
    x0 = 0.0

    def draw_row(y_center, period, label):
        y0 = y_center - r

        # rectangle: EXACT height = 2r
        ax_s.add_patch(
            Rectangle((x0, y0), 2 * n * r, 2 * r, fill=False, lw=2.2)
        )

        for i in range(n):
            cx = x0 + (i + 0.5) * dx
            excited = (i % period == 0)

            ax_s.add_patch(
                Circle(
                    (cx, y_center),
                    r,
                    facecolor=("red" if excited else "white"),
                    edgecolor="k",
                    lw=1.4,
                )
            )

        # label to the right
        ax_s.text(
            x0 + 2 * n * r + 0.04,
            y_center,
            label,
            va="center",
            ha="left",
            fontsize=20,
        )

    # centers of the two rows
    draw_row(0.65, period=3, label=r"$\mathbb{Z}_3$")
    draw_row(0.35, period=4, label=r"$\mathbb{Z}_4$")

    # Aspect ratio
    ax_s.set_aspect("equal", adjustable="box")

    return ax_s

def plot_phase_diagram_overlayed_boundaries(
    filename,
    Rb_list,
    n_avg_list,
    long_chain_data,
    title,
    *,
    delta_cut=4.06,
    phase_points=None,
    S_float_thresh=0.4,
):
    # ==================================================
    # 1. Load data
    # ==================================================
    df = pd.read_csv(filename)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)

    scale_factor = 2 ** (1/6)
    Rb_values = sorted(df["Rb"].unique())
    Rb_values = [value / scale_factor for value in Rb_values]
    delta_values = sorted(df["delta"].unique())

    Z_S = df.pivot(index="Rb", columns="delta", values="S_final").values
    Rb_grid, delta_grid = np.meshgrid(delta_values, Rb_values)

    # ==================================================
    # 2. Phase diagram (true Rb coordinates)
    # ==================================================
    cf = ax.contourf(
        Rb_grid,
        delta_grid,
        Z_S,
        levels=20,
        cmap="seismic",
    )

    # ==================================================
    # 3. Overlay Z3 / Z4 boundaries (given)
    # ==================================================
    if phase_points is not None:
        if "Z3" in phase_points:
            pts = np.asarray(phase_points["Z3"])
            ax.scatter(
                pts[:, 1], pts[:, 0] / scale_factor,
                s=70, c="cyan",
                edgecolors="k", linewidths=0.6,
                label=r"$\mathbb{Z}_3$", zorder=10,
            )

        if "Z4" in phase_points:
            pts = np.asarray(phase_points["Z4"])
            ax.scatter(
                pts[:, 1], pts[:, 0] / scale_factor,
                s=70, c="yellow",
                edgecolors="k", linewidths=0.6,
                label=r"$\mathbb{Z}_4$", zorder=10,
            )

    # ==================================================
    # 4. Floating phase: given + entropy-based candidates
    # ==================================================
    if phase_points is not None and "Floating" in phase_points:
        float_given = np.asarray(phase_points["Floating"])

        ax.scatter(
            float_given[:, 1], float_given[:, 0] / scale_factor,
            s=80, c="black",
            edgecolors="k", linewidths=0.7,
            label="Floating",
            zorder=11,
        )

        float_df = pd.DataFrame(float_given, columns=["Rb", "delta"])
        min_float_delta = (
            float_df.groupby("Rb")["delta"]
            .min()
            .to_dict()
        )

        float_candidates = df[df["S_final"] >= S_float_thresh]

        filtered_pts = []
        for _, row in float_candidates.iterrows():
            Rb = row["Rb"]
            delta = row["delta"]
            if Rb in min_float_delta and delta >= min_float_delta[Rb]:
                filtered_pts.append((Rb, delta))

        if filtered_pts:
            filtered_pts = np.asarray(filtered_pts)
            ax.scatter(
                filtered_pts[:, 1], filtered_pts[:, 0] / scale_factor,
                s=35,
                c="black",
                alpha=0.6,
                linewidths=0,
                zorder=9,
            )

    # ==================================================
    # 5. Phase-diagram axis formatting (DISPLAY ONLY)
    # ==================================================
    # scale_factor = 2 ** (1 / 6)

    ax.set_xlim(1.0, 5.0)
    ax.set_ylim(2.76 / scale_factor, 3.85 / scale_factor)

    # Disordered label
    ax.text(
        1.15, 3.37 / scale_factor,
        "Disordered",
        fontsize=21,
        color="white",
        ha="left",
        va="center",
        zorder=21,
    )

    ax.set_xlabel(r"$\delta$", fontsize=25)
    ax.set_ylabel(r"$R_b$", fontsize=25)
    ax.tick_params(labelsize=23)

    # ax.yaxis.set_major_formatter(
    #     FuncFormatter(lambda y, _: f"{y / scale_factor:.2f}")
    #)

    # Set axis limits
    ax.set_xlim(1.0, 5.0)
    # ax.set_ylim(2.76 * scale_factor, 3.85 * scale_factor)

    # ---- Colorbar ----
    cbar = fig.colorbar(
    cf,
    ax=ax,
    orientation="horizontal",
    location="top",
    pad=0.08,
)

    # Set label
    cbar.set_label(
        r"$\mathrm{S}_{v\mathrm{N}}$",
        fontsize=25,
        labelpad=10,
    )

    # Tick parameters
    cbar.ax.tick_params(labelsize=23)

    # Reduce number of tick labels
    cbar.locator = MaxNLocator(nbins=5)
    cbar.update_ticks()

    # --------------------------------------------------
    # Phase-diagram legend (UNCHANGED)
    # --------------------------------------------------
    ax.legend(
        frameon=True,
        fontsize=23,
        loc="upper left",
        bbox_to_anchor=(0.99, 1.05),
    )

    # ==================================================
    # 6. Vertical cut (true Rb)
    # ==================================================
    Rb_list = np.asarray(Rb_list)
    n_avg_list = np.asarray(n_avg_list)

    ax.vlines(
        delta_cut,
        ymin=np.min(Rb_list) / scale_factor,
        ymax=np.max(Rb_list) / scale_factor,
        color="orange",
        linewidth=3.0,
        linestyles="--",
        zorder=6,
    )

    # ==================================================
    # 7. Line plot (UNTOUCHED except title removed)
    # ==================================================
    ax_line = fig.add_axes([1.01, 5e-4, 0.5, 0.5])

    add_zq_schematic(fig, left=1.01, bottom=0.45, width=0.45, height=0.45)

    order = np.argsort(Rb_list)

    ax_line.plot(
        Rb_list[order] / scale_factor,
        n_avg_list[order],
        color="blue",
        linewidth=2.2,
        label=r"$N = 85$",
    )
    ax_line.scatter(
        Rb_list[order] / scale_factor,
        n_avg_list[order],
        color="blue",
        s=35,
        zorder=3,
    )

    Rb_1009 = np.array(sorted(long_chain_data.keys()))
    n_1009 = np.array([long_chain_data[Rb] for Rb in Rb_1009])

    ax_line.plot(
        Rb_1009,
        n_1009,
        color="red",
        linewidth=2.2,
        label=r"$N = 1009$",
    )
    ax_line.scatter(
        Rb_1009,
        n_1009,
        color="red",
        marker="s",
        s=40,
        zorder=4,
    )

    ax_line.set_xlabel(r"$R_b$", fontsize=25)
    ax_line.set_ylabel(r"$\rho = \langle \hat{N} \rangle / N$", fontsize=25)
    ax_line.set_xlim(2.95, 3.30)
    ax_line.grid(True, linestyle="--", alpha=0.5)
    ax_line.tick_params(labelsize=23)
    ax_line.legend(fontsize=20, frameon=True)

    # ==================================================
    # 8. Arrow (true coordinates)
    # ==================================================
    ax.annotate(
        "",
        xy=(0.02, 0.5), xycoords=ax_line.transAxes,
        xytext=(delta_cut, np.mean(Rb_list) / scale_factor),
        textcoords=ax.transData,
        arrowprops=dict(arrowstyle="->", linewidth=3.0, color="orange"),
    )

    plt.tight_layout(pad=1.0)

    fig.savefig(
        "1D_phase_diagram_line_cut_final_version.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=600,
    )

## More specific helper functions for visualization
def plot_dominant_bulk_peaks_interface_configs(
    structure_factor_all_peaks,
    structure_factor_comparison,
    structure_factor_uniform_85=None, # for the 85-site uniform chain
    bc_type="linear_ramp",
    region="bulk",
    n_edge=18,
    delta_boundary=0.1,
    marker_size=5,
    k_max=np.pi,
    save_folder=None,
    filename=None
):
    """
    Plot dominant bulk peak positions (k < π) vs Rb for all δ_boundary cases
    for different interface detuning configurations.

    Overlays:
    - Non-uniform chain (based on interface detuning)
    - Uniform chain (121-site reference)
    - Uniform chain (85-site chain, supplied separately)

    structure_factor_uniform_85: separate dict for 85-site uniform case.
    Expected format:
        structure_factor_uniform_85["uniform"]["full"][0][Rb_value] = peak_data
    """

    # Interface detuning mapping
    interface_labels = {
        "minus_minus": r"($-\alpha$, $-\alpha$)",
        "plus_minus": r"($+\alpha$, $-\alpha$)"
    }

    # Determine which interface configs exist
    available_ifaces = []
    for key in interface_labels.keys():
        try:
            for alpha, iface_dict in structure_factor_all_peaks[bc_type][region][n_edge].items():
                if key in iface_dict:
                    available_ifaces.append(key)
                    break
        except KeyError:
            continue

    if not available_ifaces:
        print("No interface configurations found — nothing to plot.")
        return

    # Set up figure
    n_plots = len(available_ifaces)
    fig, axes = plt.subplots(1, n_plots, figsize=(9 * n_plots, 6), dpi=600, sharey=False)
    if n_plots == 1:
        axes = [axes]
    plt.subplots_adjust(wspace=0.25)

    # Loop over interface configurations
    for ax, iface_key in zip(axes, available_ifaces):
        iface_label = interface_labels[iface_key]

        # -----------------------------
        # Variable-detuning results
        # -----------------------------
        nedge_dict = structure_factor_all_peaks[bc_type][region][n_edge]
        colors = plt.cm.viridis(np.linspace(0, 1, len(nedge_dict)))

        for color, (alpha, iface_dict) in zip(colors, nedge_dict.items()):
            if iface_key not in iface_dict:
                continue

            Rb_dict = iface_dict[iface_key]
            Rb_vals, k_dominant = [], []

            for Rb_value, peak_data in sorted(Rb_dict.items()):
                k_peaks = np.array(peak_data["k_peaks"])
                S_peaks = np.array(peak_data["S_peaks"])
                mask = k_peaks < k_max
                if not np.any(mask):
                    continue
                k_pre_pi = k_peaks[mask]
                k_mode = k_pre_pi[np.argmax(S_peaks[mask])]

                Rb_vals.append(float(Rb_value))
                k_dominant.append(float(k_mode))

            if Rb_vals:
                Rb_vals = [val / 2**(1/6) for val in Rb_vals]
                Rb_sorted, k_sorted = zip(*sorted(zip(Rb_vals, k_dominant)))
                ax.plot(
                    Rb_sorted[30:], np.array(k_sorted[30:]) / (2 * np.pi),
                    marker="o", markersize=marker_size,
                    linewidth=2, color=color,
                    label=fr"$\alpha = {round(alpha,2)}$"
                )

        # -----------------------------
        # Overlay 121-site uniform chain
        # -----------------------------
        if "uniform" in structure_factor_comparison:
            uniform_bulk = structure_factor_comparison["uniform"]["bulk"][n_edge]
            Rb_vals_uniform, k_uniform = [], []

            for Rb_value, peak_data in sorted(uniform_bulk.items()):
                k_peaks = np.array(peak_data["k_peaks"])
                S_peaks = np.array(peak_data["S_peaks"])
                mask = k_peaks < k_max
                if not np.any(mask):
                    continue
                k_mode = k_peaks[mask][np.argmax(S_peaks[mask])]

                Rb_vals_uniform.append(float(Rb_value))
                k_uniform.append(float(k_mode))

            if Rb_vals_uniform:
                Rb_vals_uniform = [val / 2**(1/6) for val in Rb_vals_uniform]
                Rb_sorted_u, k_sorted_u = zip(*sorted(zip(Rb_vals_uniform, k_uniform)))
                ax.plot(
                    Rb_sorted_u[30:], np.array(k_sorted_u[30:]) / (2 * np.pi),
                    color="blue", marker="s", markersize=5, linewidth=2.5,
                    label=r"Uniform chain ($121$ sites)"
                )

        # -------------------------------------------------
        # NEW — Overlay 85-site uniform chain
        # -------------------------------------------------
        if "uniform" in structure_factor_uniform_85:
            uniform_85_data = structure_factor_uniform_85["uniform"]["full"][0]
            Rb_vals_u85, k_u85 = [], []

            for Rb_value, peak_data in sorted(uniform_85_data.items()):
                k_peaks = np.array(peak_data["k_peaks"])
                S_peaks = np.array(peak_data["S_peaks"])
                mask = k_peaks < k_max
                if not np.any(mask):
                    continue
                k_mode = k_peaks[mask][np.argmax(S_peaks[mask])]

                Rb_vals_u85.append(float(Rb_value))
                k_u85.append(float(k_mode))

            if Rb_vals_u85:
                Rb_vals_u85 = [val / 2**(1/6) for val in Rb_vals_u85]
                Rb_sorted_85, k_sorted_85 = zip(*sorted(zip(Rb_vals_u85, k_u85)))
                ax.plot(
                    Rb_sorted_85[30:], np.array(k_sorted_85[30:]) / (2 * np.pi),
                    color="magenta", marker="^", markersize=6, linewidth=2.5,
                    label=r"Uniform chain ($85$ sites)"
                )
            # except KeyError:
            #     print("⚠ No valid 85-site uniform dataset found — skipping.")

        # -----------------------------
        # Formatting
        # -----------------------------
        ax.axhline(1/3, color="gray", linestyle="--", linewidth=1.8, alpha=0.6, label=r"Period-$3$ ($k/(2\pi) = 1/3)$")
        ax.axhline(1/4, color="black", linestyle="--", linewidth=1.8, alpha=0.6, label=r"Period-$4$ ($k/(2\pi) = 1/4)$")
        ax.set_xlabel(r"$R_b$", fontsize=25)
        ax.set_ylabel(r"$k/(2\pi)$", fontsize=25)
        ax.tick_params(axis="both", labelsize=23)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=16)
        ax.set_title(fr"Interface detuning mismatch config.: {iface_label}", fontsize=25)

    # fig.suptitle(
    #     fr"$N_{{\mathrm{{bulk}}}} = 85,~n_{{\mathrm{{boundary}}}} = 18,~\delta_{{\mathrm{{boundary}}}} = {delta_boundary} ~(121 ~\mathrm{{sites}}) + \mathrm{{Uniform}} ~\mathrm{{chain}} ~($85$ ~\mathrm{{sites}})$",
    #     fontsize=27, y=0.95
    # )

    plt.tight_layout(pad=2.0)

    # -----------------------------
    # Save figure as PDF if requested
    # -----------------------------
    if save_folder is not None:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"dominant_bulk_peaks_delta{delta_boundary}.pdf"
        save_path = save_folder / filename
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=600)
        print(f"✅ Figure saved at: {save_path.resolve()}")
    else:
        plt.show()

def plot_dominant_bulk_peaks_dense_Rb_sampling(
    structure_factor_all_peaks,
    bc_type="linear_ramp",
    region="bulk",
    n_edge=18,
    delta_boundary=0.1,
    marker_size=5,
    k_max=np.pi
):
    """
    Plot dominant bulk peak positions (k < π) vs Rb for all δ_boundary cases
    for one or both interface detuning configurations:
        Left  → (-α, -α)  (key: 'minus_minus')
        Right → (α, -α)   (key: 'plus_minus')

    Automatically detects whether one or both configurations are present.
    """

    # Interface detuning mapping
    interface_labels = {
        "minus_minus": r"($-\alpha$, $-\alpha$)",
        "plus_minus": r"($+\alpha$, $-\alpha$)"
    }

    # Check available interface keys in data
    available_ifaces = []
    if bc_type in structure_factor_all_peaks:
        if region in structure_factor_all_peaks[bc_type]:
            if n_edge in structure_factor_all_peaks[bc_type][region]:
                nedge_dict = structure_factor_all_peaks[bc_type][region][n_edge]
                for _, iface_dict in nedge_dict.items():
                    for key in interface_labels.keys():
                        if key in iface_dict and key not in available_ifaces:
                            available_ifaces.append(key)

    if not available_ifaces:
        print("No valid interface configurations found in data.")
        return

    # -----------------------------
    # Figure setup depending on number of configs
    # -----------------------------
    n_plots = len(available_ifaces)
    if n_plots == 1:
        fig, ax_arr = plt.subplots(1, 1, figsize=(9, 6), dpi=600)
        axes = [ax_arr]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=600, sharey=False)
        plt.subplots_adjust(wspace=0.25)

    # -----------------------------
    # Iterate over detected interface configurations
    # -----------------------------
    for ax, iface_key in zip(axes, available_ifaces):
        iface_label = interface_labels[iface_key]
        nedge_dict = structure_factor_all_peaks[bc_type][region][n_edge]
        colors = plt.cm.viridis(np.linspace(0, 1, len(nedge_dict)))

        for color, (alpha, iface_dict) in zip(colors, nedge_dict.items()):
            if iface_key not in iface_dict:
                continue

            Rb_dict = iface_dict[iface_key]
            Rb_vals, k_dominant = [], []

            for Rb_value, peak_data in sorted(Rb_dict.items()):
                k_peaks = np.array(peak_data["k_peaks"])
                S_peaks = np.array(peak_data["S_peaks"])
                mask = k_peaks < k_max
                if not np.any(mask):
                    continue

                k_pre_pi = k_peaks[mask]
                S_pre_pi = S_peaks[mask]
                k_mode = k_pre_pi[np.argmax(S_pre_pi)]

                Rb_vals.append(float(Rb_value))
                k_dominant.append(float(k_mode))

            if Rb_vals:
                Rb_sorted, k_sorted = zip(*sorted(zip(Rb_vals, k_dominant)))
                ax.plot(
                    Rb_sorted[30:],
                    np.array(k_sorted[30:]) / (2 * np.pi),
                    marker="o",
                    markersize=marker_size,
                    linewidth=2,
                    color=color,
                    label=fr"$\alpha = {round(alpha, 2)}$"
                )

        ax.set_xlabel(r"$R_b$", fontsize=19)
        ax.set_ylabel(r"$k/(2\pi)$", fontsize=19)
        ax.tick_params(axis="both", which="major", labelsize=17)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=16)
        ax.set_title(fr"Interface detuning configuration: {iface_label}", fontsize=20)

    # -----------------------------
    # Common title
    # -----------------------------
    fig.suptitle(
        fr"$N_{{\mathrm{{bulk}}}} = 85$,  $n_{{\mathrm{{boundary}}}} = 18$,  $\delta_{{\mathrm{{boundary}}}} = {delta_boundary}$",
        fontsize=21,
        y=0.95
    )

    plt.tight_layout(pad=2.0)
    plt.show()

def plot_dominant_bulk_peaks_dense_Rb_sampling_all_nedge(
    structure_factor_all_peaks,
    bc_type="linear_ramp",
    region="bulk",
    n_edges=(12, 18, 24),
    delta_boundary=0.1,
    marker_size=5,
    k_max=np.pi,
    save_folder=None,
    filename=None,
):
    """
    Plot dominant bulk peak positions (k < π) vs Rb for n_edge = 12, 18, 24
    for one or both interface detuning configurations.

    Handles nested structure:
    structure_factor_all_peaks[bc_type][region][n_edge][alpha][iface][Rb][Rb]
    """

    interface_labels = {
        "minus_minus": r"($-\alpha$, $-\alpha$)",
        "plus_minus": r"($+\alpha$, $-\alpha$)"
    }

    # Validate presence of the required sections
    if bc_type not in structure_factor_all_peaks or region not in structure_factor_all_peaks[bc_type]:
        print("⚠️ Data missing required bc_type/region structure.")
        return

    region_dict = structure_factor_all_peaks[bc_type][region]

    # Collect available interface configurations
    available_ifaces = set()
    for n_edge in n_edges:
        if n_edge not in region_dict:
            continue
        for alpha, iface_dict in region_dict[n_edge].items():
            for iface in interface_labels.keys():
                if iface in iface_dict:
                    available_ifaces.add(iface)

    if not available_ifaces:
        print("⚠️ No valid interface configurations found in data.")
        return

    # Setup figure depending on number of interfaces
    n_plots = len(available_ifaces)
    if n_plots == 1:
        fig, axes = plt.subplots(1, 1, figsize=(9, 6), dpi=600)
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=600, sharey=True)
        plt.subplots_adjust(wspace=0.25)

    colors = {12: "tab:blue", 18: "tab:orange", 24: "tab:green"}

    for ax, iface_key in zip(axes, available_ifaces):
        iface_label = interface_labels[iface_key]

        for n_edge in n_edges:
            if n_edge not in region_dict:
                continue

            for alpha, iface_dict in region_dict[n_edge].items():
                if iface_key not in iface_dict:
                    continue

                Rb_dict = iface_dict[iface_key]
                Rb_vals, k_dominant = [], []

                for Rb_outer, Rb_inner_dict in sorted(Rb_dict.items()):
                    # handle double-nested Rb level
                    inner_data = list(Rb_inner_dict.values())[0]
                    k_peaks = np.array(inner_data["k_peaks"])
                    S_peaks = np.array(inner_data["S_peaks"])
                    mask = k_peaks < k_max
                    if not np.any(mask):
                        continue

                    k_pre_pi = k_peaks[mask]
                    S_pre_pi = S_peaks[mask]
                    k_mode = k_pre_pi[np.argmax(S_pre_pi)]

                    Rb_vals.append(float(Rb_outer))
                    k_dominant.append(float(k_mode))

                if Rb_vals:
                    # Scale Rb values
                    Rb_vals = [val / 2**(1/6) for val in Rb_vals]

                    # Sort and plot
                    Rb_sorted, k_sorted = zip(*sorted(zip(Rb_vals, k_dominant)))
                    ax.plot(
                        Rb_sorted,
                        np.array(k_sorted) / (2 * np.pi),
                        marker="o",
                        markersize=marker_size,
                        linewidth=2,
                        color=colors[n_edge],
                        label=fr"$n_{{\mathrm{{boundary}}}} = {n_edge}$"
                    )

        ax.set_xlabel(r"$R_b$", fontsize=25)
        ax.set_ylabel(r"$k/(2\pi)$", fontsize=25)
        ax.tick_params(axis="both", which="major", labelsize=23)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=21)
        # ax.set_title(fr"Interface detuning configuration: {iface_label}", fontsize=21)

    # fig.suptitle(
    #     fr"$N_{{\mathrm{{full}}}} = 121$;  $\delta_{{\mathrm{{boundary}}}} = {delta_boundary}$; $\alpha = 0.1$",
    #     fontsize=21,
    #     y=0.95
    # )
    plt.tight_layout(pad=2.0)

    # -----------------------------
    # Save figure as PDF if requested
    # -----------------------------
    if save_folder is not None:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"dominant_bulk_peaks_delta{delta_boundary}.pdf"
        save_path = save_folder / filename
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=600)
        print(f"✅ Figure saved at: {save_path.resolve()}")
    else:
        plt.show()

def plot_dominant_bulk_peaks_vs_Rb_no_detuning_vs_detuning(
    structure_factor_nodetuning,
    structure_factor_detuned,
    bc_type="linear_ramp",
    region="bulk",
    delta_boundary=0.1,
    marker_size=5,
    k_max=np.pi,
    save_folder=None,
    filename=None,
):
    """
    Modified version: now compares the no-detuning (alpha=0.0)
    and finite-detuning (alpha=0.1, config = 'minus_minus') cases
    on the same figure.

    ONLY the no-detuning extraction logic has been updated.
    """

    fig, ax = plt.subplots(figsize=(9, 6), dpi=600)

    # -----------------------------------------------------------
    # Helper function for *finite detuning* extraction (unchanged)
    # -----------------------------------------------------------
    def extract_Rb_k_trajectories(structure_dict, alpha_target, config_target=None):
        Rb_vals = []
        k_vals = []

        for n_edge, alpha_dict in structure_dict.get(bc_type,{}).get(region,{}).items():
            if alpha_target not in alpha_dict:
                continue

            config_dict = alpha_dict[alpha_target]

            if config_target is None:
                # Should never be used now (no-detuning handled separately)
                config_keys = list(config_dict.keys())
                if len(config_keys) == 0:
                    continue
                config_key = config_keys[0]
            else:
                if config_target not in config_dict:
                    continue
                config_key = config_target

            Rb_outer = config_dict[config_key]

            for Rb_outer_val, Rb_inner in sorted(Rb_outer.items()):
                for Rb_key, peak_data in Rb_inner.items():

                    k_peaks = np.array(peak_data["k_peaks"])
                    S_peaks = np.array(peak_data["S_peaks"])

                    mask = k_peaks < k_max
                    if not np.any(mask):
                        continue

                    k_pre_pi = k_peaks[mask]
                    S_pre_pi = S_peaks[mask]
                    k_mode = k_pre_pi[np.argmax(S_pre_pi)]

                    Rb_vals.append(float(Rb_key))
                    k_vals.append(float(k_mode))

        if len(Rb_vals) == 0:
            return None, None

        # Scale Rb values
        Rb_vals = [val / 2**(1/6) for val in Rb_vals]

        # Generate arrays
        Rb_sorted, k_sorted = zip(*sorted(zip(Rb_vals, k_vals)))
        return np.array(Rb_sorted), np.array(k_sorted)

    # ------------------------------------------------------------------
    # NEW: Dedicated extraction for **NO DETUNING** (alpha = 0 case)
    # structure: dict[n_edge][delta_boundary][Rb][Rb] → peak_dict
    # ------------------------------------------------------------------
    def extract_nodetuning(structure_dict):
        Rb_vals = []
        k_vals = []

        region_dict = structure_dict.get(bc_type,{}).get(region,{})

        for n_edge, delta_dict in region_dict.items():

            # ensure δ_boundary exists
            if delta_boundary not in delta_dict:
                continue

            Rb_level = delta_dict[delta_boundary]

            # loop through Rb
            for Rb_outer, Rb_inner_dict in sorted(Rb_level.items()):
                # inner key is also Rb
                for Rb_inner, peak_data in Rb_inner_dict.items():

                    k_peaks = np.array(peak_data["k_peaks"])
                    S_peaks = np.array(peak_data["S_peaks"])

                    # select k < π
                    mask = k_peaks < k_max
                    if not np.any(mask):
                        continue

                    k_pre_pi = k_peaks[mask]
                    S_pre_pi = S_peaks[mask]
                    k_mode = k_pre_pi[np.argmax(S_pre_pi)]

                    Rb_vals.append(float(Rb_inner))
                    k_vals.append(float(k_mode))

        if len(Rb_vals) == 0:
            return None, None

        # Scale Rb values
        Rb_vals = [val / 2**(1/6) for val in Rb_vals]

        # Generate arrays
        Rb_sorted, k_sorted = zip(*sorted(zip(Rb_vals, k_vals)))
        return np.array(Rb_sorted), np.array(k_sorted)

    # ------------------------------------------------------------------
    # 1. Extract NO DETUNING
    # ------------------------------------------------------------------
    Rb0, k0 = extract_nodetuning(structure_factor_nodetuning)

    if Rb0 is not None:
        ax.plot(
            Rb0, (k0 / (2*np.pi)),
            marker="o",
            markersize=marker_size,
            linewidth=2,
            label=r"$\alpha = 0.0$",
        )

    # ------------------------------------------------------------------
    # 2. Extract finite detuning (unchanged)
    # ------------------------------------------------------------------
    Rb1, k1 = extract_Rb_k_trajectories(
        structure_factor_detuned,
        alpha_target=0.1,
        config_target="minus_minus"
    )

    if Rb1 is not None:
        ax.plot(
            Rb1, (k1 / (2*np.pi)),
            marker="o",
            markersize=marker_size,
            linewidth=2,
            label=r"$\alpha = 0.1$ [configuration: $(-\alpha,-\alpha)$]",
        )

    # ------------------------------------------------------------------
    # Styling (unchanged)
    # ------------------------------------------------------------------
    ax.set_xlabel(r"$R_b$", fontsize=25)
    ax.set_ylabel(r"$k/(2\pi)$", fontsize=25)
    ax.tick_params(axis="both", which="major", labelsize=23)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=20)

    # fig.suptitle(
    #     fr"$N_{{\mathrm{{full}}}} = 121$; $\delta_{{\mathrm{{boundary}}}} = {delta_boundary}$; "
    #     fr"$N_{{\mathrm{{bulk}}}} = 85$",
    #     fontsize=21,
    #     y=0.95,
    # )

    plt.tight_layout(pad=2.0)

    # Save
    if save_folder is not None:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"dominant_bulk_peaks_comparison_delta{delta_boundary}.pdf"
        save_path = save_folder / filename
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=600)
        print(f"✅ Figure saved at: {save_path.resolve()}")
    else:
        plt.show()

def plot_edge_excitations_no_vs_finite(
    edge_exc_no_detuning,
    edge_exc_finite_detuning,
    delta_boundary,
    alpha,
    config_label,
    n_edge,
    save_folder=None,
    filename=None
):

    # ---------------------------------------------
    # Prepare figure
    # ---------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=600, sharex=False)
    ax_left, ax_right = axes

    # Colors
    color_no = "black"
    color_finite = "tab:blue"

    # ---------------------------------------------
    # Extract Rb arrays
    # ---------------------------------------------
    Rb_no = sorted(edge_exc_no_detuning.keys(), key=float)
    Rb_fin = sorted(edge_exc_finite_detuning.keys(), key=float)

    Rb_no_arr = np.array(Rb_no)
    Rb_fin_arr = np.array(Rb_fin)

    # Scale Rb values
    Rb_no_arr = [val / 2**(1/6) for val in Rb_no_arr]
    Rb_fin_arr = [val / 2**(1/6) for val in Rb_fin_arr]

    # Values
    left_no = np.array([edge_exc_no_detuning[Rb][0] for Rb in Rb_no])
    right_no = np.array([edge_exc_no_detuning[Rb][1] for Rb in Rb_no])

    left_fin = np.array([edge_exc_finite_detuning[Rb][0] for Rb in Rb_fin])
    right_fin = np.array([edge_exc_finite_detuning[Rb][1] for Rb in Rb_fin])

    # ---------------------------------------------
    # Plot curves — No detuning
    # ---------------------------------------------
    ax_left.plot(Rb_no_arr, left_no,
                 label=fr"No detuning mismatch ($\alpha = 0.0$)",
                 color=color_no, linewidth=2, marker="o", markersize=4)

    ax_right.plot(Rb_no_arr, right_no,
                  label=fr"No detuning mismatch ($\alpha = 0.0$)",
                  color=color_no, linewidth=2, marker="o", markersize=4)

    # ---------------------------------------------
    # Plot curves — Finite detuning
    # ---------------------------------------------
    ax_left.plot(Rb_fin_arr, left_fin,
                 label=fr"Finite detuning mismatch ($\alpha = {alpha}$)",
                 color=color_finite, linewidth=2, marker="o", markersize=4)

    ax_right.plot(Rb_fin_arr, right_fin,
                  label=fr"Finite detuning mismatch ($\alpha = {alpha}$)",
                  color=color_finite, linewidth=2, marker="o", markersize=4)

    # ---------------------------------------------
    # Aesthetics
    # ---------------------------------------------
    for ax in [ax_left, ax_right]:
        ax.grid(True, alpha=0.35)
        ax.set_xlabel(r"$R_b$", fontsize=25)
        ax.tick_params(axis="both", labelsize=23)
        ax.legend(loc='best', fontsize=20)

    ax_left.set_ylabel(r"Left interface excitation", fontsize=25)
    ax_right.set_ylabel(r"Right interface excitation", fontsize=25)

    # ax_left.set_title(r"Spin at left interface", fontsize=20)
    # ax_right.set_title(r"Spin at right interface", fontsize=20)

#     # Title
#     fig.suptitle(
#     rf"$N_{{\rm full}} = 121$; "
#     rf"$n_{{\rm boundary}} = {n_edge}$; "
#     rf"$\delta_{{\rm boundary}} = {delta_boundary}$",
#     y=0.98,
#     fontsize=22
# )

    plt.tight_layout(pad=1.0)

    # ---------------------------------------------
    # Save
    # ---------------------------------------------
    if save_folder is not None:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"Edge_excitations_no_vs_finite_detuning_nedge{n_edge}.pdf"

        save_path = save_folder / filename
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=600)
        print(f"✅ Figure saved at: {save_path.resolve()}")

## Helper functions to generate spatial correlation maps + excitation density profiles + structure factor plots + Ornstein-Zernike fits
# Generating spatial correlation maps
def plot_correlation_maps_variable_system_size(
    all_mps_storage,
    alpha=0.1,
    config_label="minus_minus",
    N_values=(97, 109),
    n_edge_dict=None,
    Rb_points=None,
    max_sites=None,
    connected=True,
    figsize=(14,10),
    dpi=300,
    save_folder = None,
    filename = None,
):
    """
    Plot 2x2 correlation maps for two chain sizes (N_values) at two Rb points common to both.
    Matches the storage structure produced by your DMRG script:
        all_mps_storage[(alpha, config_label)] -> mps_storage
        mps_storage[N] -> { n_edge: { Rb: MPS, ... }, ... }

    Parameters
    ----------
    all_mps_storage : dict
        The top-level dictionary built by your script (all_mps_storage).
    alpha : float
        alpha key used (default 0.1).
    config_label : str
        config label (default "minus_minus").
    N_values : tuple
        (N_top, N_bottom) two chain lengths to plot (default (97,109)).
    n_edge_dict : dict or None
        If provided, mapping {N: n_edge}; otherwise the function selects the single available n_edge for each N.
    Rb_points : list or None
        Two Rb values (floats) to plot. If None, picks first and last common Rb between the two chains.
    max_sites : int or None
        Truncate to an M x M corner of correlation matrices for visualization.
    connected : bool
        If True, subtracts product of site densities (⟨n_j⟩⟨n_l⟩) to obtain connected correlations.
    figsize, dpi : figure size and resolution.
    """
    topN, bottomN = N_values

    top_key = (alpha, config_label)
    if top_key not in all_mps_storage:
        raise KeyError(f"Key {top_key} not found in all_mps_storage. Available keys: {list(all_mps_storage.keys())}")

    mps_storage = all_mps_storage[top_key]  # This is what dmrg_sweep_multiple_N returned (mps_storage)

    # Helper to select n_edge for each N
    def select_n_edge_for_N(N):
        if n_edge_dict and (N in n_edge_dict):
            # allow list (user may pass [6]) or single int
            ne = n_edge_dict[N]
            if isinstance(ne, (list, tuple)):
                if len(ne) != 1:
                    raise ValueError(f"n_edge_dict[{N}] must contain a single n_edge for plotting; got {ne}")
                return ne[0]
            return ne
        # otherwise inspect what's stored
        if N not in mps_storage:
            raise KeyError(f"N={N} not present in stored mps (stored Ns: {list(mps_storage.keys())})")
        n_edge_keys = sorted(mps_storage[N].keys())
        if len(n_edge_keys) == 0:
            raise ValueError(f"No n_edge entries found for N={N}")
        if len(n_edge_keys) > 1:
            # warn but choose first if user didn't specify
            print(f"Warning: multiple n_edge values found for N={N}: {n_edge_keys}. Using {n_edge_keys[0]}.")
        return n_edge_keys[0]

    nedge_top = select_n_edge_for_N(topN)
    nedge_bottom = select_n_edge_for_N(bottomN)

    # Extract Rb sets for each N
    Rb_set_top = set(mps_storage[topN][nedge_top].keys())
    Rb_set_bot = set(mps_storage[bottomN][nedge_bottom].keys())

    Rb_common = sorted(Rb_set_top & Rb_set_bot)
    if len(Rb_common) == 0:
        raise ValueError(f"No common Rb values found between N={topN} (n_edge={nedge_top}) and N={bottomN} (n_edge={nedge_bottom}).")

    # Choose Rb_points if not given
    if Rb_points is None:
        # pick two well-separated points: first and last common
        if len(Rb_common) >= 2:
            Rb_points = [Rb_common[0], Rb_common[-1]]
        else:
            # only one common point available
            Rb_points = [Rb_common[0], Rb_common[0]]
            print("Only one common Rb found; plotting it twice.")
    else:
        # validate user supplied points
        for Rb in Rb_points:
            if Rb not in Rb_common:
                raise ValueError(f"Requested Rb={Rb} is not common to both chains. Common values include: {Rb_common[:8]}...")

    # Gather matrices to compute global vmax
    corr_list = []
    for N, nedge, Rb in ((topN, nedge_top, Rb_points[0]),
                         (topN, nedge_top, Rb_points[1]),
                         (bottomN, nedge_bottom, Rb_points[0]),
                         (bottomN, nedge_bottom, Rb_points[1])):
        mps = mps_storage[N][nedge][Rb]
        corr = get_correlation_matrix(mps, N)  # user-provided function
        if connected:
            # subtract mean product if requested
            dens = get_per_site_excitation_densities(mps, N)
            corr = corr - np.outer(dens, dens)
        if max_sites is not None:
            corr = corr[:max_sites, :max_sites]
        corr_list.append(corr)

    vmax = max(np.max(np.abs(c)) for c in corr_list)

    # Make figure 2x2: top row = topN, bottom row = bottomN
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi, constrained_layout=True)

    for row_idx, N in enumerate((topN, bottomN)):
        nedge = nedge_top if row_idx == 0 else nedge_bottom
        for col_idx, Rb in enumerate(Rb_points):
            ax = axes[row_idx, col_idx]
            mps = mps_storage[N][nedge][Rb]
            corr = get_correlation_matrix(mps, N)
            if connected:
                dens = get_per_site_excitation_densities(mps, N)
                corr = corr - np.outer(dens, dens)
            if max_sites is not None:
                corr = corr[:max_sites, :max_sites]

            # Scale Rb value and plot
            Rb = Rb / 2**(1/6)
            im = ax.imshow(
                corr,
                cmap="seismic",
                vmin=-vmax,
                vmax=vmax,
                origin="lower",
                aspect="equal",
            )
            ax.set_title(fr"$N_{{\mathrm{{full}}}}={N},\; n_{{\mathrm{{boundary}}}}={nedge},\; R_b={round(Rb, 2)}$", fontsize=23)
            ax.set_xlabel(r"Site $l$", fontsize=25)
            ax.set_ylabel(r"Site $j$", fontsize=25)
            ax.tick_params(axis="both", which="major", labelsize=23)

    # Define a narrow axis on the right for the colorbar
    # [left, bottom, width, height] in figure coordinates
    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])  # tweak values as needed

    # Create colorbar in the new axis
    cbar = fig.colorbar(im, cax=cbar_ax)

    # Label and ticks
    label = r"$\langle n_j n_l \rangle_c$" if connected else r"$\langle n_j n_l \rangle$"
    cbar.set_label(label, fontsize=25)
    cbar.ax.tick_params(labelsize=23)

    # ------------------------------
    # Save figure if requested
    # ------------------------------
    if save_folder is not None:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)

        if filename is None:
            # auto-generate filename based on parameters
            conn_str = "connected" if connected else "raw"
            filename = f"corrmap_alpha{alpha}_{config_label}_{conn_str}.pdf"

        save_path = save_folder / filename
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi = 600)
        print(f"✅ Figure saved at: {save_path.resolve()}")

# Generating local excitation density profiles
def plot_excitation_density_variable_system_size(
    all_mps_storage,
    alpha=0.1,
    config_label="minus_minus",
    N_values=(97, 109),
    n_edge_dict=None,
    Rb_list=None,
    max_xticks=10,
    figsize=(15,10),
    dpi=300,
    save_folder=None,
    filename=None,
):
    """
    Plot per-site excitation densities for two chain sizes (N_values)
    for multiple Rb values in Rb_list.

    Produces a 2×2 subplot grid:
        Top row: larger N (full, bulk)
        Bottom row: smaller N (full, bulk)
    Each subplot contains curves for all Rb in Rb_list.

    Parameters
    ----------
    Rb_list : list or array-like
        List of Rb values to plot.
    """

    if Rb_list is None or len(Rb_list) == 0:
        raise ValueError("Rb_list must be provided and non-empty.")

    topN, bottomN = N_values
    storage_key = (alpha, config_label)

    if storage_key not in all_mps_storage:
        raise KeyError(f"Key {storage_key} not found in all_mps_storage. "
                       f"Available keys: {list(all_mps_storage.keys())}")

    mps_storage = all_mps_storage[storage_key]

    # Helper: select n_edge
    def select_n_edge(N):
        if n_edge_dict and (N in n_edge_dict):
            ne = n_edge_dict[N]
            if isinstance(ne, (list, tuple)):
                if len(ne) != 1:
                    raise ValueError(f"n_edge_dict[{N}] must contain a single value; got {ne}")
                return ne[0]
            return ne
        # otherwise pick first available
        n_edge_keys = sorted(mps_storage[N].keys())
        if len(n_edge_keys) == 0:
            raise ValueError(f"No n_edge entries found for N={N}")
        return n_edge_keys[0]

    # n_edge for each chain
    nedge_top = select_n_edge(topN)
    nedge_bottom = select_n_edge(bottomN)

    # All available Rb for each chain
    Rb_set_top = set(mps_storage[topN][nedge_top].keys())
    Rb_set_bottom = set(mps_storage[bottomN][nedge_bottom].keys())

    # Ensure Rb_list is valid
    for Rb in Rb_list:
        if Rb not in Rb_set_top:
            raise KeyError(f"Rb={Rb} not found for N={topN}, n_edge={nedge_top}")
        if Rb not in Rb_set_bottom:
            raise KeyError(f"Rb={Rb} not found for N={bottomN}, n_edge={nedge_bottom}")

    # ------------------------
    # Create 2x2 figure
    # ------------------------
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi, constrained_layout=True)

    # Loop over the two chain sizes (top row = larger N)
    for row_idx, N in enumerate((topN, bottomN)):
        n_edge = nedge_top if row_idx == 0 else nedge_bottom

        # Full chain subplot
        ax_full = axes[row_idx, 0]

        # Bulk region subplot
        ax_bulk = axes[row_idx, 1]
        bulk_sites = np.arange(n_edge - 1, N - n_edge)

        # Plot all Rb values
        for Rb in Rb_list:
            mps = mps_storage[N][n_edge][Rb]

            # Full chain densities
            n_sites_full = np.array([get_site_excitation_probability(mps, N, j) for j in range(N)])
            ax_full.plot(np.arange(N), n_sites_full, marker='o', linewidth=2,
                         label=fr"$R_b={Rb / 2**(1/6):.2f}$")

            # Bulk densities
            n_sites_bulk = np.array([get_site_excitation_probability(mps, N, j) for j in bulk_sites])
            ax_bulk.plot(bulk_sites, n_sites_bulk, marker='o', linewidth=2,
                         label=fr"$R_b={Rb / 2**(1/6):.2f}$")

        # Full chain axis formatting
        ax_full.set_title(fr"Full chain ($N={N}$)", fontsize=26)
        ax_full.set_xlabel("Site index $j$ (full chain)", fontsize=25)
        ax_full.set_ylabel(r"$\langle n_j\rangle$", fontsize=25)
        ax_full.set_ylim(0, 1)
        ax_full.grid(True, linestyle="--", alpha=0.5)
        ax_full.legend(fontsize=20)
        ax_full.xaxis.set_major_locator(MaxNLocator(nbins=max_xticks))
        ax_full.tick_params(axis="both", labelsize=23)

        # Bulk axis formatting
        N_bulk = len(bulk_sites) - 1
        ax_bulk.set_title(fr"Bulk region ($N_{{\rm bulk}}={N_bulk}$)", fontsize=26)
        ax_bulk.set_xlabel("Site index $j$ (bulk)", fontsize=25)
        ax_bulk.set_ylabel(r"$\langle n_j\rangle$", fontsize=25)
        ax_bulk.set_ylim(0, 1)
        ax_bulk.grid(True, linestyle="--", alpha=0.5)
        ax_bulk.legend(fontsize=20)
        ax_bulk.xaxis.set_major_locator(MaxNLocator(nbins=max_xticks))
        ax_bulk.tick_params(axis="both", labelsize=23)

    # ------------------------------
    # Save figure if needed
    # ------------------------------
    if save_folder is not None:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"excitation_density_alpha{alpha}_{config_label}_multiRb.pdf"

        save_path = save_folder / filename
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=600)
        print(f"✅ Figure saved at: {save_path.resolve()}")

# Correlation length extraction for different chain lengths
def fit_correlation_vs_distance_bulk_multiple_N(
    mps_dicts,
    N_list,
    Rb_dict,
    n_edge_dict,
    fit_range_ratio=0.6,
    min_fit_points=8,
    diagnostic_plots=True,
    save_folder=None,
    filename=None,
):
    """
    Fit bulk correlation functions for multiple chain lengths (N values),
    handling different MPS storage structures:
      - N = 97, 109: mps_storage[N][n_edge][Rb]
      - N = 121: mps_storage[n_edge][Rb]

    Diagnostic plots appear only for N=97 and 109 in a single row.
    """

    results = {}

    # Determine which N values to plot
    plot_N_list = [N for N in N_list if N in [97, 109]]
    n_plot = len(plot_N_list)

    if diagnostic_plots and n_plot > 0:
        fig, axes = plt.subplots(1, n_plot, figsize=(6*n_plot, 5), dpi=600)
        if n_plot == 1:
            axes = [axes]

    for idx, N in enumerate(N_list):
        n_edge = n_edge_dict[N]
        Rb = Rb_dict[N]

        # Extract MPS depending on storage structure
        if N in [97, 109]:
            # old structure: N -> n_edge -> Rb
            mps_storage_N = mps_dicts[N]  # dict: n_edge -> {Rb -> MPS}
            if n_edge not in mps_storage_N or Rb not in mps_storage_N[n_edge]:
                print(f"⚠ Warning: Missing MPS for N={N}, n_edge={n_edge}, Rb={Rb}")
                continue
            mps = mps_storage_N[n_edge][Rb]

        elif N == 121:
            # new structure: n_edge -> {Rb -> MPS}
            mps_storage_121 = mps_dicts[N]
            if n_edge not in mps_storage_121 or Rb not in mps_storage_121[n_edge]:
                print(f"⚠ Warning: Missing MPS for N={N}, n_edge={n_edge}, Rb={Rb}")
                continue
            mps = mps_storage_121[n_edge][Rb]

        else:
            raise ValueError(f"Unknown storage structure for N={N}")

        # Compute bulk correlation
        C_r_bulk = correlation_vs_distance_bulk(mps, N, n_edge)
        r_vals_bulk = np.arange(len(C_r_bulk))

        # Exclude r=0
        r_vals_bulk = r_vals_bulk[1:]
        C_r_bulk = C_r_bulk[1:]

        # Fit range
        max_fit_distance = int(fit_range_ratio * len(r_vals_bulk))
        max_fit_idx = max(min_fit_points, min(len(r_vals_bulk), max_fit_distance))
        r_fit = r_vals_bulk[:max_fit_idx]
        C_fit = C_r_bulk[:max_fit_idx]

        # Mask invalid points
        valid_mask = np.isfinite(C_fit) & (C_fit > 1e-8) & (C_fit <= 1.0)
        r_fit = r_fit[valid_mask]
        C_fit = C_fit[valid_mask]

        # Initial guesses
        A_guess = np.mean(C_fit[:3])
        xi_guess = estimate_initial_xi(r_fit, C_fit)
        k_guess = np.pi / 2
        phi0_guess = 0.0

        bounds_lower = [1e-3, 0.5, 0.1, -np.pi]
        bounds_upper = [2.0, N * 1.2, np.pi, np.pi]

        # Fit Ornstein-Zernike
        popt, pcov = curve_fit(
            ornstein_zernike_regularized,
            r_fit, C_fit,
            p0=[A_guess, xi_guess, k_guess, phi0_guess],
            bounds=(bounds_lower, bounds_upper),
            maxfev=10000
        )

        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan]*4
        C_pred = ornstein_zernike_regularized(r_fit, *popt)
        residuals = C_fit - C_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((C_fit - np.mean(C_fit))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean(residuals**2))
        xi_fit = popt[1]

        results[N] = {
            'xi': xi_fit,
            'fit_params': popt,
            'fit_errors': perr,
            'fit_quality': {'r_squared': r_squared, 'rmse': rmse},
            'convergence_info': 'success'
        }

        # Diagnostic plots only for N=97 and 109
        if diagnostic_plots and N in plot_N_list:
            i_plot = plot_N_list.index(N)
            ax = axes[i_plot]
            ax.plot(r_vals_bulk, C_r_bulk, 'o', markersize=4, alpha=0.7, label='Numerical data', color='blue')
            r_plot = np.linspace(min(r_vals_bulk), max(r_vals_bulk), 200)
            ax.plot(r_plot, ornstein_zernike_regularized(r_plot, *popt), '-', linewidth=2, color='red',
                    label=fr'Fit: $\xi={xi_fit:.2f}$')
            ax.axvspan(min(r_fit), max(r_fit), alpha=0.2, color='gray', label='Fit region')
            ax.set_xlabel(r"Distance $r$", fontsize=23)
            ax.set_ylabel(r"$C(r)$", fontsize=23)
            ax.tick_params(axis='both', which='major', labelsize=22)
            ax.set_title(
                fr"$N_{{\mathrm{{full}}}} = {N}$, $n_{{\mathrm{{boundary}}}} = {n_edge}$, $R_b = {Rb / 2**(1/6):.3f}$" +
                f"\n$R^2 = {r_squared:.3f}$", fontsize=22
            )
            ax.legend(loc='upper right', fontsize=17)
            ax.grid(True, alpha=0.3)

        # After all plots
        if diagnostic_plots and n_plot > 0:
            # Adjust spacing to avoid overlap
            plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=4.0)
            plt.subplots_adjust(top=0.88)  # leave space for titles if needed

            # ------------------------------
            # Save figure as PDF if requested
            # ------------------------------
            if save_folder is not None:
                save_folder = Path(save_folder)
                save_folder.mkdir(parents=True, exist_ok=True)
                if filename is None:
                    filename = f"excitation_density_alpha{alpha}_{config_label}.pdf"
                save_path = save_folder / filename
                fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=600)
                print(f"✅ Figure saved at: {save_path.resolve()}")
            else:
                # Show the figure if no save folder is provided
                plt.show()

    return results

# Plotting structure factor for N_full = 97 and 109 at representative points in the floating phase
def plot_bulk_structure_factor_variable_boundary_size(
    all_mps_storage,
    alpha=0.1,
    config_label="minus_minus",
    n_edge_dict=None,
    Rb_points=None,
    use_fluct=False,
    oversample=32,
    figsize=(14, 6),
    dpi=400,
    save_folder=None,
    filename=None,
):
    """
    Automatically generate one subplot PER available system size N in the
    MPS dictionary all_mps_storage[(alpha, config_label)].

    Each subplot shows the fine-grained continuous structure factor
    of the bulk excitation density for multiple Rb points.

    Parameters
    ----------
    all_mps_storage : dict
        all_mps_storage[(alpha, config_label)][N][n_edge][Rb] = MPS.
    alpha, config_label : keys in all_mps_storage.
    n_edge_dict : dict, optional
        Mapping N -> n_edge. Otherwise, the first available n_edge is used.
    Rb_points : list of floats
        Rb values to plot.
    use_fluct : bool
        If True, subtract mean density before DFT.
    oversample : int
        Oversampling factor for DFT.
    save_folder : str or Path, optional
        If provided, saves figure as a PDF.
    filename : str
        PDF name if saving.
    """

    # Extract correct branch of the storage
    key = (alpha, config_label)
    if key not in all_mps_storage:
        raise KeyError(f"Key {key} not found in all_mps_storage")
    mps_storage = all_mps_storage[key]

    # All available system sizes
    N_values = sorted(mps_storage.keys())
    n_panels = len(N_values)

    if Rb_points is None:
        raise ValueError("Rb_points must be provided.")

    # Helper to choose n_edge
    def select_n_edge(N):
        if n_edge_dict and (N in n_edge_dict):
            ne = n_edge_dict[N]
            if isinstance(ne, (list, tuple)):
                if len(ne) != 1:
                    raise ValueError(f"n_edge_dict[{N}] must contain a single value. Got {ne}.")
                return ne[0]
            return ne
        # fallback: take first available
        return sorted(mps_storage[N].keys())[0]

    # Create subplots: 1 row, n_panels columns
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(figsize[0] * n_panels / 2.0, figsize[1]),
        dpi=dpi,
        constrained_layout=True
    )

    if n_panels == 1:
        axes = [axes]

    # Colors for different Rb
    colors = plt.cm.plasma(np.linspace(0, 1, len(Rb_points)))

    # Loop over all N values and create one panel each
    for ax, N in zip(axes, N_values):
        n_edge = select_n_edge(N)

        for color, Rb in zip(colors, Rb_points):
            # if Rb not in mps_storage[N][n_edge]:
            #     print(f"⚠️ Rb={Rb:.3f} not found for N={N}, skipping...")
            #     continue

            mps = mps_storage[N][n_edge][Rb]

            # Compute excitation density for all sites
            n_sites_full = np.array([
                get_site_excitation_probability(mps, N, j) for j in range(N)
            ])

            # Bulk selection
            bulk_slice = slice(n_edge - 1, N - n_edge)
            n_bulk = n_sites_full[bulk_slice]

            if use_fluct:
                n_bulk = n_bulk - np.mean(n_bulk)

            # FT
            ks, S_k = continuous_structure_factor(n_bulk, oversample=oversample)

            ax.plot(
                ks / (2 * np.pi),
                S_k,
                linewidth=2,
                color=color,
                label=fr"$R_b={Rb/2**(1/6):.3f}$"
            )

        # Formatting
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"$k / (2\pi)$", fontsize=25)
        ax.set_ylabel(r"$|\mathcal{F}[\langle n_{\mathrm{bulk}} \rangle(k)]|$", fontsize=25)
        ax.set_title(fr"$N_{{\mathrm{{full}}}} = {N}$; $N_{{\mathrm{{bulk}}}} = {N - 2 * n_edge}$", fontsize=26)
        ax.tick_params(axis='both', labelsize=23)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=21, loc='best')

    # Save as PDF
    if save_folder is not None:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"structure_factor_bulk_alpha{alpha}_{config_label}.pdf"
        save_path = save_folder / filename
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=600)
        print(f"✅ Figure saved at: {save_path.resolve()}")

    plt.show()

# Function to plot the structure factor (FT of excitation density profiles)
def plot_bulk_structure_factor_fixed_chain_length_variable_boundary_size(
    all_mps_storage,
    alpha=0.1,
    config_label="minus_minus",
    N_full=None,
    n_edge_list=None,               # NEW: list of n_edge values from user
    Rb_points=None,
    use_fluct=False,
    oversample=32,
    figsize=(14, 6),
    dpi=400,
    save_folder=None,
    filename=None,
):
    """
    Plot continuous fine-grained structure factor of the bulk excitation density
    for a *fixed* N_full, with one subplot per n_edge value.

    New expected storage structure:
        all_mps_storage[(alpha, config_label)][n_edge][Rb] = MPS
    """

    # Pull correct branch
    key = (alpha, config_label)
    if key not in all_mps_storage:
        raise KeyError(f"Key {key} not found in storage")
    mps_storage = all_mps_storage[key]

    if n_edge_list is None:
        raise ValueError("Must provide n_edge_list (list of boundary sizes).")

    if Rb_points is None:
        raise ValueError("Must provide Rb_points.")

    # # Determine fixed N_full by reading an example MPS
    # # (take the first n_edge and first Rb available)
    # example_n_edge = n_edge_list[0]
    # example_Rb = list(mps_storage[example_n_edge].keys())[0]
    # example_mps = mps_storage[example_n_edge][example_Rb]
    # N_full = example_mps.num_sites  # Assumes MPS object has num_sites attribute

    # Number of panels = number of n_edge values
    n_panels = len(n_edge_list)

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(figsize[0] * n_panels / 2.0, figsize[1]),
        dpi=dpi,
        constrained_layout=True
    )

    if n_panels == 1:
        axes = [axes]

    # Colors for Rb curves
    colors = plt.cm.plasma(np.linspace(0, 1, len(Rb_points)))

    # Loop over subplot panels (each one corresponds to one n_edge)
    for ax, n_edge in zip(axes, n_edge_list):

        for color, Rb in zip(colors, Rb_points):

            if Rb not in mps_storage[n_edge]:
                print(f"⚠️ Rb={Rb:.3f} not found for n_edge={n_edge}, skipping...")
                continue

            mps = mps_storage[n_edge][Rb]

            # Compute excitation density
            n_sites_full = np.array([
                get_site_excitation_probability(mps, N_full, j)
                for j in range(N_full)
            ])

            # Extract bulk region just as before
            bulk_slice = slice(n_edge - 1, N_full - n_edge)
            n_bulk = n_sites_full[bulk_slice]

            if use_fluct:
                n_bulk = n_bulk - np.mean(n_bulk)

            # Continuous FT
            ks, S_k = continuous_structure_factor(n_bulk, oversample=oversample)

            ax.plot(
                ks / (2 * np.pi),
                S_k,
                linewidth=2,
                color=color,
                label=fr"$R_b={Rb/2**(1/6):.3f}$"
            )

        # Formatting
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"$k / (2\pi)$", fontsize=19)
        ax.set_ylabel(r"$|\mathcal{F}[n_{\mathrm{bulk}}(k)]|$", fontsize=19)
        ax.set_title(
            fr"$N_{{\mathrm{{full}}}} = {N_full}$;  $n_{{\mathrm{{edge}}}} = {n_edge}$",
            fontsize=20
        )
        ax.tick_params(axis='both', labelsize=17)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=16, loc='best')

    # Save as PDF if requested
    if save_folder is not None:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"structure_factor_fixedN_alpha{alpha}_{config_label}.pdf"
        save_path = save_folder / filename
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=600)
        print(f"✅ Figure saved at: {save_path.resolve()}")

    plt.show()

## Helper function to plot the dominant sampled configurations from the DMRG ground-state MPS in the disordered phase
def plot_top_configs_stack(
    top_configs,
    bulk_width=40,
    top_k_plot=5,
    N_full=121,
    i_start=None,
    i_end=None,
    Rb=None,
    delta=None,
    panel_label="(a)",
    save_folder=None,
    filename=None,
):
    """
    Plot the top configurations, showing only a selected window of sites.

    Parameters
    ----------
    top_configs : list of dicts
        Result from dominant_configs(...) with 'config' and probability fields.
        Each config is assumed to have length N_full.
    bulk_width : int
        Number of central bulk sites to display (used only if i_start/i_end not given).
    top_k_plot : int
        Number of configurations to display.
    N_full : int
        Full chain length (default: 121).
    i_start, i_end : int or None
        Explicit starting and ending site indices to display (end exclusive).
        If None, the central bulk window of width bulk_width is used.
    Rb, delta : numbers
        Used for title annotation (optional).
    """

    # --------------------------------------------------
    # Determine window to plot
    # --------------------------------------------------
    if i_start is None or i_end is None:
        # Default: central bulk window (original behavior)
        assert bulk_width < N_full, "bulk_width must be smaller than N_full"

        center = N_full // 2
        half = bulk_width // 2
        i_start = center - half
        i_end   = i_start + bulk_width
    else:
        # User-specified window
        assert 0 <= i_start < i_end <= N_full, "Invalid (i_start, i_end) window"

    plot_width = i_end - i_start

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(plot_width / 1.5, top_k_plot * 1.5),
        dpi=600
    )

    y_spacing = 1.2

    # --------------------------------------------------
    # Plot configurations
    # --------------------------------------------------
    for idx, item in enumerate(top_configs[:top_k_plot]):
        cfg_full = np.asarray(item["config"])
        exact_p = item.get("exact_prob", np.nan)

        # Extract chosen slice
        cfg = cfg_full[i_start:i_end]

        y_base = (top_k_plot - idx - 1) * y_spacing

        # Bounding rectangle
        rect = Rectangle(
            (0, y_base - 0.5),
            plot_width,
            1.0,
            edgecolor="black",
            facecolor="none",
            lw=1.8,
        )
        ax.add_patch(rect)

        # Draw sites
        for i, state in enumerate(cfg):
            color = "red" if state == 1 else "white"
            circle = Circle(
                (i + 0.5, y_base),
                0.45,
                facecolor=color,
                edgecolor="black",
                lw=1.5,
            )
            ax.add_patch(circle)

    # --------------------------------------------------
    # Formatting
    # --------------------------------------------------
    ax.set_xlim(0, plot_width + 2)
    ax.set_ylim(-0.5, top_k_plot * y_spacing + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # --------------------------------------------------
    # Subfigure label (just above top-left corner)
    # --------------------------------------------------
    if panel_label is not None:
        ax.text(
            -0.02, 0.95, panel_label,
            transform=ax.transAxes,
            fontsize=40,
            ha="left",
            va="bottom",
        )

    plt.tight_layout()

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    if save_folder is not None:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = "dominant_configs_bulk_plot.pdf"

        save_path = save_folder / filename
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=600)
        print(f"✅ Figure saved at: {save_path.resolve()}")