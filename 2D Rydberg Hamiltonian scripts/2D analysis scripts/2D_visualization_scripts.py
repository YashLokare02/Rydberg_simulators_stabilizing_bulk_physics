## Importing relevant libraries
import numpy as np
import scipy
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, eig
from scipy.sparse.linalg import eigs
from functools import reduce
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse, FancyArrowPatch
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
from matplotlib.ticker import MultipleLocator, FuncFormatter, MaxNLocator
from scipy.signal import find_peaks
import pickle
from matplotlib.colors import ListedColormap
from pathlib import Path
import h5py
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
from scipy.interpolate import griddata, UnivariateSpline, make_interp_spline
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Polygon

# Libaries for GL fitting procedure
import math
from numpy.polynomial.laguerre import laggauss
from scipy.optimize import least_squares, minimize
from scipy.optimize import curve_fit
from scipy.interpolate import griddata, PchipInterpolator, interp1d

# Import libraries for 1D long chain wavevector calculations
from pathlib import Path
import re
from scipy.signal import find_peaks

## Helper function to plot square vs. star ordering patterns on a finite 2D Rydberg lattice
def plot_square_vs_star_full_lattice(
    data_square_pkl,
    data_star_pkl,
    Lx,
    Ly,
    cmap="viridis",
    dpi=300,
    save_folder=None,
):
    """
    Plot representative full-lattice density profiles for
    square and star ordered phases.

    Left  : Square ordering
    Right : Star ordering
    """
    # Folder to extract data from
    folder = Path("2D_array_data_final")

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    # For the square solution
    file_path_square = folder / data_square_pkl
    file_path_star = folder / data_star_pkl

    # Load the .pkl files
    with open(file_path_square, "rb") as f:
        data_square_dict = pickle.load(f)

    with open(file_path_star, "rb") as f:
        data_star_dict = pickle.load(f)

    # --------------------------------------------------
    # Convert dicts to arrays
    # --------------------------------------------------
    data_square = np.zeros((Ly, Lx))
    data_star   = np.zeros((Ly, Lx))

    for (y, x), v in data_square_dict.items():
        data_square[y, x] = v

    for (y, x), v in data_star_dict.items():
        data_star[y, x] = v

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=dpi)

    # === Square ===
    im0 = axes[0].imshow(data_square, cmap=cmap, origin="upper")
    # cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    # cbar0.set_label(r"$\langle n_{x,y} \rangle$", fontsize=25)
    # cbar0.ax.tick_params(labelsize=23)

    # axes[0].set_title("Square-ordering pattern", fontsize=27)
    # axes[0].set_xlabel(r"Site index along $x$", fontsize=25)
    # axes[0].set_ylabel(r"Site index along $y$", fontsize=25)
    # axes[0].tick_params(axis="both", labelsize=23)

    # === Star ===
    im1 = axes[1].imshow(data_star, cmap=cmap, origin="upper")
    # cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    # cbar1.set_label(r"$\langle n_{x,y} \rangle$", fontsize=25)
    # cbar1.ax.tick_params(labelsize=23)

    # axes[1].set_title("Star-ordering pattern", fontsize=27)
    # axes[1].set_xlabel(r"Site index along $x$", fontsize=25)
    # axes[1].set_ylabel(r"Site index along $y$", fontsize=25)
    # axes[1].tick_params(axis="both", labelsize=23)

    # Show no axes
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    # --------------------------------------------------
    # Subfigure label (a): top-left, just outside first panel
    # --------------------------------------------------
    axes[0].text(
        -0.04, 0.99, "(a)",
        transform=axes[0].transAxes,
        fontsize=28,
        # fontweight="bold",
        ha="right",
        va="bottom",
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    if save_folder is not None:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = save_folder / "square_vs_star_density_profiles.pdf"
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        print(f"✅ Saved: {save_path.resolve()}")

    plt.show()

## Helper functions to plot ordering patterns retrieved from 2D DMRG simulations (shows full lattice and bulk region results)
# Plotting as a function of \delta_{bulk}
def plot_full_and_bulk_lattices(
    density_dicts,
    delta_values,
    Lx,
    Ly,
    n_edge,
    n_title,
    Rb=None,
    alpha=None,
    cmap="viridis",
    save_folder=None,
    dpi=300,
    print_bulk=True,
    bulk_precision=4,
):
    """
    Plot full lattice and inner bulk region for multiple density dictionaries
    in a single multi-row figure.

    Each row corresponds to one delta value:
        left  : full lattice
        right : bulk region
    """

    assert len(density_dicts) == len(delta_values), \
        "density_dicts and delta_values must have the same length"

    n_cases = len(delta_values)
    bulk_Lx = Lx - 2 * n_edge
    bulk_Ly = Ly - 2 * n_edge

    if save_folder is not None:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Create figure
    # --------------------------------------------------
    fig, axes = plt.subplots(
        n_cases, 2,
        figsize=(16, 6 * n_cases),
        dpi=dpi,
        squeeze=False
    )

    for row, (dens_dict, delta) in enumerate(zip(density_dicts, delta_values)):

        # -----------------------
        # Map dictionary to array
        # -----------------------
        data = np.zeros((Ly, Lx))
        for (y, x), v in dens_dict.items():
            data[y, x] = v

        bulk = data[
            n_edge : Ly - n_edge,
            n_edge : Lx - n_edge
        ]

        # Compute the bulk order parameter
        bulk_op = compute_bulk_order_parameter(bulk)

        # --------------------------------------------------
        # Print bulk excitation values
        # --------------------------------------------------
        if print_bulk:
            print("\n" + "=" * 60)
            print(f"Bulk excitation values")
            print(f"Lattice: {Lx} x {Ly}")
            print(f"Bulk size: {bulk_Lx} x {bulk_Ly}")
            print(f"delta_bulk = {delta}")
            if Rb is not None:
                print(f"Rb = {Rb}")
            if alpha is not None:
                print(f"alpha = {alpha}")
            print("-" * 60)

            with np.printoptions(
                precision=bulk_precision,
                suppress=True
            ):
                print(bulk)

            print("-" * 60)
            print(
                f"Bulk statistics: "
                f"mean = {bulk.mean():.6f}, "
                f"min = {bulk.min():.6f}, "
                f"max = {bulk.max():.6f}"
            )
            print(
                f"Bulk order parameter "
                r"$\sum_{x,y}(n_{x,y}-n_{y,x})^2/N$"
                f" = {bulk_op:.6e}"
            )
            print("=" * 60)

        # -----------------------
        # Full lattice
        # -----------------------
        ax_full = axes[row, 0]
        im1 = ax_full.imshow(data, cmap=cmap, origin="upper")
        cbar1 = plt.colorbar(im1, ax=ax_full, fraction=0.046, pad=0.04)
        cbar1.set_label(r"$\langle n_{x,y} \rangle$", fontsize=25)
        cbar1.ax.tick_params(labelsize=23)

        # ax_full.set_title(
        #     rf"Full ${Lx} \times {Ly}$ lattice",
        #     fontsize=25
        # )
        ax_full.set_xlabel(r"Site index along $x$", fontsize=25)
        ax_full.set_ylabel(r"Site index along $y$", fontsize=25)
        ax_full.tick_params(axis="both", labelsize=23)

        # -----------------------
        # Bulk
        # -----------------------
        ax_bulk = axes[row, 1]
        im2 = ax_bulk.imshow(bulk, cmap=cmap, origin="upper")
        cbar2 = plt.colorbar(im2, ax=ax_bulk, fraction=0.046, pad=0.04)
        cbar2.set_label(
            r"$\langle n_{x_{\mathrm{bulk}},y_{\mathrm{bulk}}} \rangle$",
            fontsize=25
        )
        cbar2.ax.tick_params(labelsize=23)

        # ax_bulk.set_title(
        #     rf"Bulk region (${bulk_Lx} \times {bulk_Ly}$ lattice)",
        #     fontsize=25
        # )
        ax_bulk.set_xlabel(r"Site index along $x_{\mathrm{bulk}}$", fontsize=25)
        ax_bulk.set_ylabel(r"Site index along $y_{\mathrm{bulk}}$", fontsize=25)
        ax_bulk.tick_params(axis="both", labelsize=23)

        # -----------------------
        # Row label (delta)
        # -----------------------
        ax_full.text(
            -0.35, 0.5,
            rf"$\delta_{{\mathrm{{bulk}}}} = {delta}$",
            transform=ax_full.transAxes,
            fontsize=26,
            rotation=90,
            va="center",
            ha="center"
        )

    # --------------------------------------------------
    # Suptitle
    # --------------------------------------------------
    title_parts = []
    if Rb is not None:
        title_parts.insert(1, rf"$R_b = {Rb}$")
    # if alpha is not None:
    #     title_parts.append(rf"$\alpha = {alpha}$")

    fig.suptitle(
        "; ".join(title_parts),
        fontsize=30,
        y=0.95
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    if save_folder is not None:
        fname = f"density_maps_L{Lx}x{Ly}_Rb{Rb}_all_deltas.pdf"
        save_path = save_folder / fname
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        print(f"✅ Saved: {save_path.resolve()}")

    plt.show()

# Plotting as a function of delta_boundary
def plot_full_and_bulk_lattices_varying_delta_boundary(
    density_dicts,
    delta_boundary_values,
    Lx,
    Ly,
    n_edge,
    n_title,
    delta_bulk,
    Rb=None,
    alpha=None,
    cmap="viridis",
    save_folder=None,
    dpi=600,
    print_bulk=True,
    bulk_precision=4,
):
    """
    Plot full lattice and inner bulk region for multiple density dictionaries
    in a single multi-row figure.

    Each row corresponds to one delta_boundary value:
        left  : full lattice
        right : bulk region
    """

    assert len(density_dicts) == len(delta_boundary_values), \
        "density_dicts and delta_boundary_values must have the same length"

    n_cases = len(delta_boundary_values)
    bulk_Lx = Lx - 2 * n_edge
    bulk_Ly = Ly - 2 * n_edge

    if save_folder is not None:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)

    # Initialization
    order_param = []

    # --------------------------------------------------
    # Create figure
    # --------------------------------------------------
    fig, axes = plt.subplots(
        n_cases, 2,
        figsize=(16, 6 * n_cases),
        dpi=dpi,
        squeeze=False
    )

    for row, (dens_dict, delta_boundary) in enumerate(
        zip(density_dicts, delta_boundary_values)
    ):

        # -----------------------
        # Map dictionary to array
        # -----------------------
        data = np.zeros((Ly, Lx))
        for (y, x), v in dens_dict.items():
            data[y, x] = v

        bulk = data[
            n_edge : Ly - n_edge,
            n_edge : Lx - n_edge
        ]

        # Compute the bulk order parameter
        bulk_op = compute_bulk_order_parameter(bulk)
        order_param.append(bulk_op)

        # --------------------------------------------------
        # Print bulk excitation values
        # --------------------------------------------------
        if print_bulk:
            print("\n" + "=" * 60)
            print("Bulk excitation values")
            print(f"Lattice: {Lx} x {Ly}")
            print(f"Bulk size: {bulk_Lx} x {bulk_Ly}")
            print(f"delta_bulk = {delta_bulk}")
            print(f"delta_boundary = {delta_boundary}")
            if Rb is not None:
                print(f"Rb = {Rb}")
            if alpha is not None:
                print(f"alpha = {alpha}")
            print("-" * 60)

            with np.printoptions(
                precision=bulk_precision,
                suppress=True
            ):
                print(bulk)

            print("-" * 60)
            print(
                f"Bulk statistics: "
                f"mean = {bulk.mean():.6f}, "
                f"min = {bulk.min():.6f}, "
                f"max = {bulk.max():.6f}"
            )
            print(
                f"Bulk order parameter "
                r"$\sum_{x,y}(n_{x,y}-n_{y,x})^2/N$"
                f" = {bulk_op:.6e}"
            )
            print("=" * 60)

        # -----------------------
        # Full lattice
        # -----------------------
        ax_full = axes[row, 0]
        im1 = ax_full.imshow(data, cmap=cmap, origin="upper")
        cbar1 = plt.colorbar(im1, ax=ax_full, fraction=0.046, pad=0.04)
        cbar1.set_label(r"$\langle n_{x,y} \rangle$", fontsize=25)
        cbar1.ax.tick_params(labelsize=23)

        # ax_full.set_title(
        #     rf"Full ${Lx} \times {Ly}$ lattice",
        #     fontsize=25
        # )
        ax_full.set_xlabel(r"Site index along $x$", fontsize=25)
        ax_full.set_ylabel(r"Site index along $y$", fontsize=25)
        ax_full.tick_params(axis="both", labelsize=23)

        # -----------------------
        # Bulk
        # -----------------------
        ax_bulk = axes[row, 1]
        im2 = ax_bulk.imshow(bulk, cmap=cmap, origin="upper")
        cbar2 = plt.colorbar(im2, ax=ax_bulk, fraction=0.046, pad=0.04)
        cbar2.set_label(
            r"$\langle n_{x_{\mathrm{bulk}},y_{\mathrm{bulk}}} \rangle$",
            fontsize=25
        )
        cbar2.ax.tick_params(labelsize=23)

        # ax_bulk.set_title(
        #     rf"Bulk region (${bulk_Lx} \times {bulk_Ly}$ lattice)",
        #     fontsize=25
        # )
        ax_bulk.set_xlabel(r"Site index along $x_{\mathrm{bulk}}$", fontsize=25)
        ax_bulk.set_ylabel(r"Site index along $y_{\mathrm{bulk}}$", fontsize=25)
        ax_bulk.tick_params(axis="both", labelsize=23)

        # -----------------------
        # Row label (delta_boundary)
        # -----------------------
        ax_full.text(
            -0.35, 0.5,
            rf"$\delta_{{\mathrm{{boundary}}}} = {delta_boundary}$",
            transform=ax_full.transAxes,
            fontsize=26,
            rotation=90,
            va="center",
            ha="center"
        )

    # --------------------------------------------------
    # Suptitle
    # --------------------------------------------------
    title_parts = [
        rf"$\delta_{{\mathrm{{bulk}}}} = {delta_bulk}$",
    ]
    if Rb is not None:
        title_parts.insert(1, rf"$R_b = {Rb}$")
    # if alpha is not None:
    #     title_parts.append(rf"$\alpha = {alpha}$")

    fig.suptitle(
        "; ".join(title_parts),
        fontsize=30,
        y=0.95
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    if save_folder is not None:
        fname = (
            f"density_maps_L{Lx}x{Ly}_"
            f"deltaBulk{delta_bulk}_varying_deltaBoundary.pdf"
        )
        save_path = save_folder / fname
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        print(f"✅ Saved: {save_path.resolve()}")

    plt.show()

    return order_param

## Helper functions to plot the 2D phase diagram in a restrictive region (shows uniform lattice vs. ramped-detuning results)
## Function to generate the list of bulk data for a given Rb
def order_parameter_list(Lx, Ly, folder, filenames, n_edge=3):
    # Initialize
    ord_param_list = []

    # Compute
    for filename in filenames:
        bulk_data = generate_bulk_data(Lx, Ly, folder, filename, n_edge)
        order_parameter = compute_bulk_order_parameter(bulk_data)
        ord_param_list.append(order_parameter)

    return ord_param_list

def build_op_grid(OP_by_Rb, Rb_values, delta_values):
    """
    OP_by_Rb[Rb] = list of precomputed bulk OP values (one per delta).
    Returns OP_grid with shape (len(Rb_values), len(delta_values)).
    """
    Rb_values = np.array(Rb_values, dtype=float)
    delta_values = np.array(delta_values, dtype=float)

    OP_grid = np.zeros((len(Rb_values), len(delta_values)), dtype=float)

    for i, Rb in enumerate(Rb_values):
        OP_list = OP_by_Rb[Rb]
        assert len(OP_list) == len(delta_values), (
            f"Rb={Rb}: expected {len(delta_values)} OP values, got {len(OP_list)}"
        )
        OP_grid[i, :] = OP_list

    return OP_grid

## Function to ensure that the dashed cyan lines are tangential to the outer DMRG points
def marker_radius_data_units(ax, fig, scatter_s, edge_lw_pts=0.0):
    """
    Convert scatter marker radius (given s in points^2) into data-unit radii (dx, dy)
    at the current axis scaling.

    scatter_s: same 's' you passed to ax.scatter (points^2)
    edge_lw_pts: marker edge linewidth in points (optional, to make tangency match outer edge)
    """
    # marker radius in points: scatter 's' is area in points^2
    r_pts = 0.5 * np.sqrt(scatter_s) + 0.5 * edge_lw_pts

    # points -> pixels
    r_pix = r_pts * fig.dpi / 72.0

    # pixel offset in display coords -> data coords
    inv = ax.transData.inverted()

    # Use an arbitrary reference display point; only differences matter
    x0, y0 = ax.transData.transform((0.0, 0.0))

    # dx: move r_pix in screen-x
    x1, _ = inv.transform((x0 + r_pix, y0))
    x0d, y0d = inv.transform((x0, y0))
    dx_data = x1 - x0d

    # dy: move r_pix in screen-y
    _, y1 = inv.transform((x0, y0 + r_pix))
    dy_data = y1 - y0d

    return dx_data, dy_data

## Function to plot the scatter-point phae diagram (better for sparse phase diagrams)
def plot_scatter_phase_diagram(
    OP_grid_left,          # OP grid for LEFT subplot
    OP_grid_right,         # OP grid for RIGHT subplot (your existing one)
    Rb_values_right,
    delta_values_right,
    Rb_min=1.2,
    Rb_max=1.9,
    delta_min=2.5,
    delta_max=4.9,
    cmap="viridis",
    save_folder=None,
    filename="phase_diagram_scatter_two_panels.pdf",
    dpi=600,
):

    # --------------------------------------------------
    # Define LEFT subplot coordinates internally
    # --------------------------------------------------
    Rb_values_left = np.array([1.6, 1.7, 1.8, 1.9])
    delta_values_left = np.array([3.7, 4.0, 4.3, 4.6, 4.9])

    Rb_values_right = np.array(Rb_values_right, dtype=float)
    delta_values_right = np.array(delta_values_right, dtype=float)

    # --------------------------------------------------
    # Build scatter coordinates
    # --------------------------------------------------
    RR_L, DD_L = np.meshgrid(Rb_values_left, delta_values_left, indexing="ij")
    xL = DD_L.ravel()
    yL = RR_L.ravel()
    colorsL = OP_grid_left.ravel()

    RR_R, DD_R = np.meshgrid(Rb_values_right, delta_values_right, indexing="ij")
    xR = DD_R.ravel()
    yR = RR_R.ravel()
    colorsR = OP_grid_right.ravel()

    # --------------------------------------------------
    # Create figure
    # --------------------------------------------------
    fig, axes = plt.subplots(
        1, 2,
        figsize=(18, 6.5),
        dpi=dpi,
        sharey=False
    )

    # ==================================================
    # LEFT SUBPLOT
    # ==================================================
    scL = axes[0].scatter(
        xL,
        yL,
        c=colorsL,
        cmap=cmap,
        s=450,
        edgecolors="k",
        linewidths=0.8,
        zorder=5,
    )

    cbarL = plt.colorbar(scL, ax=axes[0], fraction=0.046, pad=0.04)
    cbarL.set_label(r"$O_{\mathrm{star}}$", fontsize=29)
    cbarL.ax.tick_params(labelsize=27)

    axes[0].set_xlim(delta_min, delta_max)
    axes[0].set_ylim(Rb_min, Rb_max)
    axes[0].set_xlabel(r"$\delta_{\mathrm{bulk}} ~(= \delta_{\mathrm{uniform}})$", fontsize=29)
    axes[0].set_ylabel(r"$R_b$", fontsize=29)
    axes[0].tick_params(axis="both", labelsize=27)
    axes[0].set_title("Uniform lattice", fontsize=28)

    # Dashed cyan guide lines (RIGHT subplot only, unchanged)
    axes[0].hlines(
        y=Rb_values_left.min() - 0.01,
        xmin=delta_min,
        xmax=delta_max,
        colors="cyan",
        linestyles="--",
        linewidth=3,
        zorder=4,
    )

    axes[0].vlines(
        x=delta_values_left.min() - 0.05,
        ymin=Rb_min,
        ymax=Rb_max,
        colors="cyan",
        linestyles="--",
        linewidth=3,
        zorder=4,
    )

    # ==================================================
    # RIGHT SUBPLOT (UNCHANGED LOGIC)
    # ==================================================
    scR = axes[1].scatter(
        xR,
        yR,
        c=colorsR,
        cmap=cmap,
        s=450,
        edgecolors="k",
        linewidths=0.8,
        zorder=5,
    )

    cbarR = plt.colorbar(scR, ax=axes[1], fraction=0.046, pad=0.04)
    cbarR.set_label(r"$O_{\mathrm{star}}$", fontsize=29)
    cbarR.ax.tick_params(labelsize=27)

    axes[1].set_xlim(delta_min, delta_max)
    axes[1].set_ylim(Rb_min, Rb_max)
    axes[1].set_xlabel(r"$\delta_{\mathrm{bulk}}$", fontsize=29)
    axes[1].tick_params(axis="both", labelsize=27)

    # Dashed cyan guide lines (RIGHT subplot only, unchanged)
    axes[1].hlines(
        y=Rb_values_right.min() - 0.01,
        xmin=delta_min,
        xmax=delta_max,
        colors="cyan",
        linestyles="--",
        linewidth=3,
        zorder=4,
    )

    axes[1].vlines(
        x=delta_values_right.min() - 0.05,
        ymin=Rb_min,
        ymax=Rb_max,
        colors="cyan",
        linestyles="--",
        linewidth=3,
        zorder=4,
    )

    # Thin grey reference line at Rb = 1.65 (RIGHT subplot only)
    Rb_ref = 1.65
    axes[1].hlines(
        y=Rb_ref,
        xmin=delta_min,
        xmax=delta_max,
        colors="grey",
        linestyles="--",
        linewidth=1.5,
        alpha=0.8,
        zorder=3,
    )

    axes[1].text(
        delta_min + 0.02 * (delta_max - delta_min),
        Rb_ref + 0.015,
        r"$R_b = 1.65$",
        color="grey",
        fontsize=23,
        va="bottom",
        ha="left",
    )

    axes[1].set_title("Boundary detuning ramps", fontsize=28)

    plt.tight_layout()

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    if save_folder is not None:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = save_folder / filename
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        print(f"✅ Saved: {save_path.resolve()}")

    plt.show()

## Helper function to generate the guide schematic (2D Rydberg lattice under boundary detuning ramps)
def plot_detuning_and_star_comparison(
    density_uniform,
    density_ramped,
    Lx=13,
    Ly=13,
    n_edge=4,
    delta_bulk=4.9,
    delta_boundary=1.8,
    alpha=0.05,
    cmap="viridis",
    dpi=600,
    save_folder=None,
):
    """
    Left:  large detuning profile schematic
    Right: two stacked density maps (uniform / ramped),
           each half the height of the left panel.
    """

    # --------------------------------------------------
    # Build detuning profile
    # --------------------------------------------------
    delta_interface = delta_bulk - alpha
    detuning = np.zeros((Ly, Lx))

    for y in range(Ly):
        for x in range(Lx):
            d = min(x, Lx - 1 - x, y, Ly - 1 - y)
            if d < n_edge:
                t = d / (n_edge - 1) if n_edge > 1 else 1.0
                detuning[y, x] = (
                    delta_boundary
                    + (delta_interface - delta_boundary) * t
                )
            else:
                detuning[y, x] = delta_bulk

    # ---- NEW: smooth detuning profile ----
    detuning = gaussian_filter(detuning, sigma=0.8)

    # --------------------------------------------------
    # Convert density dicts to arrays
    # --------------------------------------------------
    def dict_to_array(d):
        arr = np.zeros((Ly, Lx))
        for (yy, xx), v in d.items():
            arr[yy, xx] = v
        return arr

    data_uniform = dict_to_array(density_uniform)
    data_ramped  = dict_to_array(density_ramped)

    # --------------------------------------------------
    # Layout: GridSpec
    # --------------------------------------------------
    fig = plt.figure(figsize=(12, 8), dpi=dpi)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[2.0, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.05,
        hspace=0.05,
    )

    ax_left   = fig.add_subplot(gs[:, 0])
    ax_top    = fig.add_subplot(gs[0, 1])
    ax_bottom = fig.add_subplot(gs[1, 1])

    # --------------------------------------------------
    # Left panel: detuning profile
    # --------------------------------------------------
    ax_left.imshow(detuning, cmap=cmap, origin="upper")
    ax_left.set_axis_off()

    # ---- Subfigure label (a) ----
    ax_left.text(
        -0.08, 1.01, "(a)",
        transform=ax_left.transAxes,
        fontsize=24,
        ha="left",
        va="top",
    )

    # ---- Arrows toward bulk (with BIG arrowheads) ----
    arrow_props = dict(
        arrowstyle="->",
        color="black",
        lw=3.5,
        shrinkA=0,
        shrinkB=0,
        mutation_scale=28,   # <<< BIG arrowheads
    )

    bulk_x0 = n_edge - 0.5
    bulk_y0 = n_edge - 0.5
    bulk_x1 = Lx - n_edge - 0.5
    bulk_y1 = Ly - n_edge - 0.5

    bulk_corners_data = [
        (bulk_x0, bulk_y1),
        (bulk_x1, bulk_y1),
        (bulk_x0, bulk_y0),
        (bulk_x1, bulk_y0),
    ]

    start_corners_axes = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
    ]

    for (xa, ya), (xd, yd) in zip(start_corners_axes, bulk_corners_data):
        ax_left.annotate(
            "",
            xy=(xd, yd), xycoords="data",
            xytext=(xa, ya), textcoords="axes fraction",
            arrowprops=arrow_props,
            annotation_clip=False,
        )

    # ---- Bulk dashed square ----
    rect = plt.Rectangle(
        (bulk_x0, bulk_y0),
        Lx - 2 * n_edge,
        Ly - 2 * n_edge,
        fill=False,
        edgecolor="black",
        linewidth=2.5,
        linestyle="--",
        alpha=0.6,
    )
    ax_left.add_patch(rect)

    # ---- Bulk label ----
    ax_left.text(
        (bulk_x0 + bulk_x1) / 2,
        (bulk_y0 + bulk_y1) / 2,
        "Bulk",
        fontsize=25,
        ha="center",
        va="center",
        color="black",
    )

    # ---- Boundary label ----
    ax_left.text(
        (bulk_x0 + bulk_x1) / 2,
        bulk_y0 - 1.7,
        "Boundary",
        fontsize=24,
        ha="center",
        va="top",
    )

    # ---- Detuning ramps annotation (big arrowhead) ----
    ax_left.annotate(
        "Detuning ramps",
        xy=(bulk_x0 - 0.5, bulk_y0 - 0.5),
        xycoords="data",
        xytext=(0.5, -0.08),
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->",
            lw=2.8,
            color="black",
            mutation_scale=36,   # <<< BIG arrowhead here too
        ),
        fontsize=24,
        ha="center",
        va="top",
        annotation_clip=False,
    )

    # --------------------------------------------------
    # Right panels: excitation density (SEISMIC colormap)
    # --------------------------------------------------
    im_top = ax_top.imshow(data_uniform, cmap="hot", origin="upper")
    ax_top.set_axis_off()

    im_bot = ax_bottom.imshow(data_ramped, cmap="hot", origin="upper")
    ax_bottom.set_axis_off()

    # --------------------------------------------------
    # Common colorbar ABOVE the top-right panel
    # --------------------------------------------------
    pos = ax_top.get_position()

    cax = fig.add_axes([
        pos.x0,            # left aligned with top-right panel
        pos.y1 + 0.015,    # just above it
        pos.width,        # same width
        0.025,            # thin horizontal bar
    ])

    cbar = fig.colorbar(im_top, cax=cax, orientation="horizontal")
    # Move ticks and label to the TOP of the colorbar
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")

    # Ticks and colorbar label
    cbar.ax.tick_params(labelsize=21)
    cbar.set_label(r"$\langle n_{x,y} \rangle$", fontsize=23, labelpad=12)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    if save_folder is not None:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = save_folder / "detuning_profile_and_star_phase_comparison.pdf"
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        print(f"✅ Saved: {save_path.resolve()}")