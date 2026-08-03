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

## Scripts to plot the extended 2D phase diagram (PBC [bulk] vs. uniform (finite) lattice vs. boundary detuning ramps)
# LaTeX rendering for subplots
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Nimbus Roman"],
    "hatch.linewidth": 2.0, 
    "mathtext.fontset": "stix",
    "font.size": 12,
    "legend.frameon": False,
})

# Phase colors
PHASE_COLORS = {
    "square": (0.5, 0.0, 1.0),
    "star": "red",
    "striated": "blue",
    "checkerboard": "green",
}

# Opacity values
REGION_ALPHA = 0.30
MARKER_ALPHA = 1.00

# \delta values
delta_values = [3.7, 4.0, 4.3, 4.6, 4.9]

# Labeling phases for uniform lattice
uniform_phase_data = {}
for delta in delta_values:
    uniform_phase_data[(1.4, delta)] = "checkerboard"
    uniform_phase_data[(1.5, delta)] = "striated"
    uniform_phase_data[(1.6, delta)] = "square"
    uniform_phase_data[(1.7, delta)] = "square"
    uniform_phase_data[(1.8, delta)] = "square"

uniform_phase_data[(1.9, 3.7)] = "square"
uniform_phase_data[(1.9, 4.0)] = "star"
uniform_phase_data[(1.9, 4.3)] = "star"
uniform_phase_data[(1.9, 4.6)] = "square"
uniform_phase_data[(1.9, 4.9)] = "square"

# Labeling phases for ramped detuning cases
boundary_phase_data = {}
for delta in delta_values:
    boundary_phase_data[(1.4, delta)] = "checkerboard"
    boundary_phase_data[(1.5, delta)] = "striated"
    boundary_phase_data[(1.6, delta)] = "striated"
    boundary_phase_data[(1.7, delta)] = "star"
    boundary_phase_data[(1.8, delta)] = "star"
    boundary_phase_data[(1.9, delta)] = "star"

boundary_phase_data[(1.65, 3.7)] = "striated"
boundary_phase_data[(1.65, 4.0)] = "striated"
boundary_phase_data[(1.65, 4.3)] = "star"
boundary_phase_data[(1.65, 4.6)] = "star"
boundary_phase_data[(1.65, 4.9)] = "star"

# Labeling phases for the PBC / bulk cases
pbc_phase_data = {}

for delta in delta_values:
    pbc_phase_data[(1.4, delta)] = "checkerboard"
    pbc_phase_data[(1.5, delta)] = "striated"

    pbc_phase_data[(1.6, delta)] = "star"
    pbc_phase_data[(1.7, delta)] = "star"
    pbc_phase_data[(1.8, delta)] = "star"
    pbc_phase_data[(1.9, delta)] = "star"

# Helper functions for visualization
def plot_simulation_points(ax, phase_data, marker="o", marker_size=52):
    for (Rb, delta), phase in phase_data.items():
        ax.scatter(
            delta,
            Rb,
            s=marker_size,
            marker=marker,
            facecolor=PHASE_COLORS[phase],
            edgecolor="black",
            linewidth=0.6,
            alpha=MARKER_ALPHA,
            zorder=10,
        )


def fill_uniform_phase_regions(ax, x_min, x_max, y_min, y_max):
    Rb_cb_striated = 0.5 * (1.4 + 1.5)
    Rb_striated_square = 0.5 * (1.5 + 1.6)
    Rb_square_star = 0.5 * (1.8 + 1.9)

    delta_star_lower = 0.5 * (3.7 + 4.0)
    delta_star_upper = 0.5 * (4.3 + 4.6)

    hatch_pattern = "///"
    hatch_color = "black"

    # --------------------------------------------------------
    # Checkerboard and striated regions
    # --------------------------------------------------------
    ax.axhspan(
        y_min,
        Rb_cb_striated,
        color=PHASE_COLORS["checkerboard"],
        alpha=REGION_ALPHA,
        zorder=0,
    )

    ax.axhspan(
        Rb_cb_striated,
        Rb_striated_square,
        color=PHASE_COLORS["striated"],
        alpha=REGION_ALPHA,
        zorder=0,
    )

    # --------------------------------------------------------
    # Main square region:
    # translucent fill + opaque hatch overlay
    # --------------------------------------------------------
    ax.axhspan(
        Rb_striated_square,
        Rb_square_star,
        facecolor=PHASE_COLORS["square"],
        edgecolor="none",
        alpha=REGION_ALPHA,
        zorder=0,
    )

    ax.axhspan(
        Rb_striated_square,
        Rb_square_star,
        facecolor="none",
        edgecolor=hatch_color,
        hatch=hatch_pattern,
        linewidth=0.0,
        alpha=1.0,
        zorder=1,
    )

    # --------------------------------------------------------
    # Left square region above Rb_square_star
    # --------------------------------------------------------
    square_left_vertices = [
        (x_min, Rb_square_star),
        (delta_star_lower, Rb_square_star),
        (delta_star_lower, y_max),
        (x_min, y_max),
    ]

    square_left_fill = Polygon(
        square_left_vertices,
        closed=True,
        facecolor=PHASE_COLORS["square"],
        edgecolor="none",
        alpha=REGION_ALPHA,
        zorder=0,
    )

    square_left_hatch = Polygon(
        square_left_vertices,
        closed=True,
        facecolor="none",
        edgecolor=hatch_color,
        hatch=hatch_pattern,
        linewidth=0.0,
        alpha=1.0,
        zorder=1,
    )

    # --------------------------------------------------------
    # Right square region above Rb_square_star
    # --------------------------------------------------------
    square_right_vertices = [
        (delta_star_upper, Rb_square_star),
        (x_max, Rb_square_star),
        (x_max, y_max),
        (delta_star_upper, y_max),
    ]

    square_right_fill = Polygon(
        square_right_vertices,
        closed=True,
        facecolor=PHASE_COLORS["square"],
        edgecolor="none",
        alpha=REGION_ALPHA,
        zorder=0,
    )

    square_right_hatch = Polygon(
        square_right_vertices,
        closed=True,
        facecolor="none",
        edgecolor=hatch_color,
        hatch=hatch_pattern,
        linewidth=0.0,
        alpha=1.0,
        zorder=1,
    )

    # --------------------------------------------------------
    # Star region
    # --------------------------------------------------------
    star_region = Polygon(
        [
            (delta_star_lower, Rb_square_star),
            (delta_star_upper, Rb_square_star),
            (delta_star_upper, y_max),
            (delta_star_lower, y_max),
        ],
        closed=True,
        facecolor=PHASE_COLORS["star"],
        edgecolor="none",
        alpha=REGION_ALPHA,
        zorder=0,
    )

    ax.add_patch(square_left_fill)
    ax.add_patch(square_left_hatch)

    ax.add_patch(square_right_fill)
    ax.add_patch(square_right_hatch)

    ax.add_patch(star_region)

    # --------------------------------------------------------
    # Phase boundaries
    # --------------------------------------------------------
    kwargs = {
        "color": "black",
        "linewidth": 1.3,
        "zorder": 5,
    }

    ax.plot(
        [x_min, x_max],
        [Rb_cb_striated, Rb_cb_striated],
        **kwargs,
    )

    ax.plot(
        [x_min, x_max],
        [Rb_striated_square, Rb_striated_square],
        **kwargs,
    )

    ax.plot(
        [delta_star_lower, delta_star_upper],
        [Rb_square_star, Rb_square_star],
        **kwargs,
    )

    ax.plot(
        [delta_star_lower, delta_star_lower],
        [Rb_square_star, y_max],
        **kwargs,
    )

    ax.plot(
        [delta_star_upper, delta_star_upper],
        [Rb_square_star, y_max],
        **kwargs,
    )


def fill_boundary_ramp_phase_regions(ax, x_min, x_max, y_min, y_max):
    Rb_cb_striated = 0.5 * (1.4 + 1.5)
    Rb_striated_star_low_delta = 0.5 * (1.65 + 1.7)
    Rb_striated_star_high_delta = 0.5 * (1.6 + 1.65)
    delta_step = 0.5 * (4.0 + 4.3)

    checkerboard_polygon = Polygon(
        [(x_min, y_min), (x_max, y_min), (x_max, Rb_cb_striated), (x_min, Rb_cb_striated)],
        closed=True, facecolor=PHASE_COLORS["checkerboard"], edgecolor="none",
        alpha=REGION_ALPHA, zorder=0,
    )
    striated_polygon = Polygon(
        [(x_min, Rb_cb_striated), (x_max, Rb_cb_striated),
         (x_max, Rb_striated_star_high_delta),
         (delta_step, Rb_striated_star_high_delta),
         (delta_step, Rb_striated_star_low_delta),
         (x_min, Rb_striated_star_low_delta)],
        closed=True, facecolor=PHASE_COLORS["striated"], edgecolor="none",
        alpha=REGION_ALPHA, zorder=0,
    )
    star_polygon = Polygon(
        [(x_min, Rb_striated_star_low_delta),
         (delta_step, Rb_striated_star_low_delta),
         (delta_step, Rb_striated_star_high_delta),
         (x_max, Rb_striated_star_high_delta),
         (x_max, y_max), (x_min, y_max)],
        closed=True, facecolor=PHASE_COLORS["star"], edgecolor="none",
        alpha=REGION_ALPHA, zorder=0,
    )

    ax.add_patch(checkerboard_polygon)
    ax.add_patch(striated_polygon)
    ax.add_patch(star_polygon)

    kwargs = {"color": "black", "linewidth": 1.3, "zorder": 5}
    ax.plot([x_min, x_max], [Rb_cb_striated, Rb_cb_striated], **kwargs)
    ax.plot([x_min, delta_step], [Rb_striated_star_low_delta, Rb_striated_star_low_delta], **kwargs)
    ax.plot([delta_step, delta_step], [Rb_striated_star_high_delta, Rb_striated_star_low_delta], **kwargs)
    ax.plot([delta_step, x_max], [Rb_striated_star_high_delta, Rb_striated_star_high_delta], **kwargs)

    # ax.text(4.30, 1.395, "Checkerboard", ha="center", va="center", fontsize=10)
    # ax.text(4.65, 1.54, "Striated", ha="center", va="center", fontsize=10)
    # ax.text(4.30, 1.80, "Star", ha="center", va="center", fontsize=11)

def fill_pbc_phase_regions(
    ax,
    x_min,
    x_max,
    y_min,
    y_max,
):
    """
    Fill schematic phase regions for the PBC phase diagram.

    Boundaries
    ----------
    Checkerboard/striated:
        midpoint between Rb = 1.4 and 1.5 -> Rb = 1.45

    Striated/star:
        midpoint between Rb = 1.5 and 1.6 -> Rb = 1.55
    """

    Rb_cb_striated = 0.5 * (1.4 + 1.5)
    Rb_striated_star = 0.5 * (1.5 + 1.6)

    # Checkerboard region
    ax.axhspan(
        y_min,
        Rb_cb_striated,
        facecolor=PHASE_COLORS["checkerboard"],
        edgecolor="none",
        alpha=REGION_ALPHA,
        zorder=0,
    )

    # Striated region
    ax.axhspan(
        Rb_cb_striated,
        Rb_striated_star,
        facecolor=PHASE_COLORS["striated"],
        edgecolor="none",
        alpha=REGION_ALPHA,
        zorder=0,
    )

    # Star region
    ax.axhspan(
        Rb_striated_star,
        y_max,
        facecolor=PHASE_COLORS["star"],
        edgecolor="none",
        alpha=REGION_ALPHA,
        zorder=0,
    )

    # Phase-boundary lines
    boundary_kwargs = {
        "color": "black",
        "linewidth": 1.3,
        "zorder": 5,
    }

    ax.plot(
        [x_min, x_max],
        [Rb_cb_striated, Rb_cb_striated],
        **boundary_kwargs,
    )

    ax.plot(
        [x_min, x_max],
        [Rb_striated_star, Rb_striated_star],
        **boundary_kwargs,
    )

# Main driver function
def plot_pbc_uniform_boundary_phase_diagrams(
    save_folder=None,
    filename="pbc_uniform_boundary_phase_diagrams.pdf",
    dpi=600,
):
    """
    Plot schematic 2D phase diagrams for:

        (a) Periodic boundary conditions
        (b) Uniform finite lattice
        (c) Boundary-detuning ramp

    Horizontal axis: detuning
    Vertical axis: R_b
    """

    x_min = 3.55
    x_max = 5.05

    y_min = 1.35
    y_max = 1.95

    # Visualization
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(6.3, 2.1),
        dpi=dpi,
        sharex=False,
        sharey=True,
        constrained_layout=True,
    )

    # Define axes for subplots
    ax_pbc, ax_uniform, ax_boundary = axes

    # ========================================================
    # PBC phase diagram
    # ========================================================
    fill_pbc_phase_regions(
        ax=ax_pbc,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )

    plot_simulation_points(
        ax=ax_pbc,
        phase_data=pbc_phase_data,
        marker_size=40, 
    )

    ax_pbc.set_title(
        r"PBC (bulk)",
    )

    # ========================================================
    # Uniform finite-lattice phase diagram
    # ========================================================
    fill_uniform_phase_regions(
        ax=ax_uniform,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )

    plot_simulation_points(
        ax=ax_uniform,
        phase_data=uniform_phase_data,
        marker_size=40, 
    )

    ax_uniform.set_title(
        r"Uniform (finite) lattice",
    )

    # ========================================================
    # Boundary-detuning-ramp phase diagram
    # ========================================================
    fill_boundary_ramp_phase_regions(
        ax=ax_boundary,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )

    plot_simulation_points(
        ax=ax_boundary,
        phase_data=boundary_phase_data,
        marker_size=40, 
    )

    ax_boundary.set_title(
        r"Boundary detuning",
    )

    # ========================================================
    # Shared axis formatting
    # ========================================================
    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.set_xticks(delta_values)

        ax.set_yticks([
            1.4,
            1.5,
            1.6,
            1.65,
            1.7,
            1.8,
            1.9,
        ])

        ax.tick_params(
            axis="both",
            labelsize=10,
            direction="in",
            top=True,
            right=True,
        )

    # Separate x-axis labels
    ax_pbc.set_xlabel(
        r"$\delta_{\rm bulk} ~(= \delta_{\rm uniform})$",
    )

    ax_uniform.set_xlabel(
        r"$\delta_{\rm bulk} ~(= \delta_{\rm uniform})$",
    )

    ax_boundary.set_xlabel(
        r"$\delta_{\mathrm{bulk}}$",
    )

    ax_pbc.set_ylabel(
        r"$R_b$",
    )

    # ========================================================
    # Shared phase legend
    # ========================================================
    phase_handles = [
        Patch(
            facecolor=PHASE_COLORS["square"],
            edgecolor="black",
            hatch="///",
            label="Square",
        ),
        Patch(
            facecolor=PHASE_COLORS["star"],
            edgecolor="black",
            label="Star",
        ),
        Patch(
            facecolor=PHASE_COLORS["striated"],
            edgecolor="black",
            label="Striated",
        ),
        Patch(
            facecolor=PHASE_COLORS["checkerboard"],
            edgecolor="black",
            label="Checkerboard",
        ),
    ]

    # fig.legend(
    #     handles=phase_handles,
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 1.08),
    #     ncol=4,
    #     fontsize=11,
    #     frameon=False,
    # )

    # ========================================================
    # Save or show
    # ========================================================
    if save_folder is not None:
        save_folder = Path(save_folder)
        save_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_path = save_folder / filename

        output_format = save_path.suffix.lower().lstrip(".")

        if output_format == "":
            output_format = "png"
            save_path = save_path.with_suffix(".pdf")

        fig.savefig(
            save_path,
            format=output_format,
            dpi=dpi,
            bbox_inches="tight",
        )

        print(f"Saved figure to:\n{save_path.resolve()}")

    else:
        plt.show()

    return fig, axes

# Save path for figure
save_folder = Path("New_figures_PRX_revision_2D")
filename = "PBC_vs_uniform_detuning_2D_extended_phase_diagrams.png"

# Run driver
if __name__ == "__main__":
    plot_pbc_uniform_boundary_phase_diagrams(
        save_folder=save_folder,
        filename=filename,
        dpi=600,
    )

# ====================================================================================
# Updated helper functions 
# ====================================================================================
## New Fig. 2(c) -- representative density maps -- bulk star, finite lattice star, and finite lattice square
def plot_star_square_density_maps(
    density_star,
    square_density_values,
    bulk_star_npy_file,
    Lx=13,
    Ly=13,
    bulk_Lx=8,
    bulk_Ly=8,
    dpi=600,
    save_folder=None,
    filename="bulk_star_finite_star_square_density_maps.pdf",
):
    """
    Plot:
        (1) Bulk periodically repeating star phase (8x8 supercell)
        (2) Finite-lattice star phase
        (3) Finite-lattice square phase
    """

    # --------------------------------------------------------
    # Plot settings
    # --------------------------------------------------------
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Nimbus Roman"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "legend.frameon": False,
    })

    # --------------------------------------------------------
    # Load bulk-star density profile
    # --------------------------------------------------------
    data_bulk_star = np.asarray(
        np.load(bulk_star_npy_file),
        dtype=float,
    )

    if data_bulk_star.shape != (bulk_Ly, bulk_Lx):

        if data_bulk_star.size == bulk_Lx * bulk_Ly:
            data_bulk_star = data_bulk_star.reshape(
                bulk_Ly,
                bulk_Lx,
            )
        else:
            raise ValueError(
                f"Expected bulk array shape {(bulk_Ly, bulk_Lx)}, "
                f"received {data_bulk_star.shape}."
            )

    # --------------------------------------------------------
    # Convert finite-star dictionary to array
    # --------------------------------------------------------
    def dict_to_array(d):
        arr = np.zeros((Ly, Lx))

        for (yy, xx), value in d.items():
            arr[yy, xx] = value

        return arr

    data_star = dict_to_array(density_star)

    # --------------------------------------------------------
    # Convert finite-square list to array
    # --------------------------------------------------------
    square_density_values = np.asarray(
        square_density_values,
        dtype=float,
    )

    if square_density_values.size != Lx * Ly:
        raise ValueError(
            f"Expected {Lx*Ly} entries, got "
            f"{square_density_values.size}."
        )

    data_square = square_density_values.reshape(
        (Ly, Lx)
    )

    # --------------------------------------------------------
    # Transparent red colormap
    # --------------------------------------------------------
    redmap = np.zeros((256, 4))

    redmap[:, 0] = 1.0
    redmap[:, 3] = np.linspace(0.0, 0.8, 256)

    redmap = ListedColormap(redmap)

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(6.3, 2.1),
        dpi=dpi,
        constrained_layout=True,
    )

    ax_bulk, ax_star, ax_square = axes

    im_bulk = ax_bulk.imshow(
        data_bulk_star,
        cmap=redmap,
        origin="upper",
        vmin=0,
        vmax=1,
    )

    im_star = ax_star.imshow(
        data_star,
        cmap=redmap,
        origin="upper",
        vmin=0,
        vmax=1,
    )

    im_square = ax_square.imshow(
        data_square,
        cmap=redmap,
        origin="upper",
        vmin=0,
        vmax=1,
    )

    titles = [
        r"PBC (bulk): star",
        r"Finite lattice: star",
        r"Finite lattice: square",
    ]

    # Visualization
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])

    # --------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------
    cbar = fig.colorbar(
        im_square,
        ax=axes,
        orientation="vertical",
        fraction=0.03,
        pad=0.03,
    )

    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["0", "1"])
    cbar.set_label(
        r"$\langle n_{x,y}\rangle$",
        fontsize=12,
    )

    cbar.ax.tick_params(labelsize=12)

    # --------------------------------------------------------
    # Save or show
    # --------------------------------------------------------
    if save_folder is not None:
        save_folder = Path(save_folder)
        save_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_path = save_folder / filename
        fig.savefig(
            save_path,
            format="pdf",
            bbox_inches="tight",
        )

        print(f"Saved: {save_path.resolve()}")

    else:
        plt.show()

    return (
        data_bulk_star,
        data_star,
        data_square,
    )

## New Fig. 5(a): representative density maps -- uniform square phase + boundary-detuned .. 
# .. striated-like and star phases
def plot_square_star_striated_density_maps(
    square_density_values,
    star_density_values,
    striated_density_values,
    Lx=13,
    Ly=13,
    dpi=600,
    save_folder=None,
    filename="square_star_striated_density_maps.pdf",
):
    # LaTeX rendering for subplots
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Nimbus Roman"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "legend.frameon": False,
    })

    # --------------------------------------------------
    # Convert density dicts to arrays
    # --------------------------------------------------
    def dict_to_array(d):
        arr = np.zeros((Ly, Lx))
        for (yy, xx), v in d.items():
            arr[yy, xx] = v
        return arr

    ## Initialize data
    # Boundary-detuned star state (dict)
    data_star = dict_to_array(star_density_values)
    
    # Boundary-detuned striated state (dict)
    data_striated = dict_to_array(striated_density_values)

    # -----------------------------
    # Convert square density list to array
    # Assumes row-major ordering: y first, then x
    # -----------------------------
    square_density_values = np.asarray(square_density_values, dtype=float)
    if square_density_values.size != Lx * Ly:
        raise ValueError(
            f"Expected {Lx*Ly} density values, got {square_density_values.size}."
        )

    data_square = square_density_values.reshape((Ly, Lx))

    redmap = np.zeros((256, 4))
    redmap[:, 0] = 1.0
    redmap[:, 3] = np.linspace(0, 0.8, 256)
    redmap = ListedColormap(redmap)

    # Visualization
    fig, axes = plt.subplots(
        1, 3,
        figsize=(6.3, 2.1),
        dpi=dpi,
        constrained_layout=True,
    )

    maps = [
        (data_square, r"Uniform lattice: square"),
        (data_star, r"Boundary detuning: star"),
        (data_striated, r"Boundary detuning: striated-like"),
    ]

    ims = []
    for ax, (data, title) in zip(axes, maps):
        im = ax.imshow(data, cmap=redmap, origin="upper")
        im.set_clim(0, 1)
        ims.append(im)

        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(
        ims[-1],
        ax=axes[-1],
        orientation="vertical",
        fraction=0.046,
        pad=0.04,
    )
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["0", "1"])
    cbar.set_label(r"$\langle n_{x,y}\rangle$", fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    if save_folder is not None:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = save_folder / filename
        fig.savefig(save_path, format="png", bbox_inches="tight")
        print(f"✅ Saved: {save_path.resolve()}")
    else:
        plt.show()

    return data_square, data_star, data_striated

# =======================================================================================
# Helper routines for new Fig. 5(b) -- striated and (1, 1)-sublattice OPs
# =======================================================================================
def load_density_dict(filename):
    """
    Load density map stored as a .pkl dictionary:
        {(y, x): density}
    """
    with open(filename, "rb") as f:
        density_dict = pickle.load(f)

    return density_dict

# Dictionary data converted to Numpy array
def dict_to_array(density_dict, Lx=13, Ly=13):
    arr = np.zeros((Ly, Lx))
    for (yy, xx), v in density_dict.items():
        arr[yy, xx] = v
    return arr

# Compute order parameters from files
def compute_order_parameters_from_file_list(
    density_folder, 
    density_files,
    delta_bulk_values,
    Lx=13,
    Ly=13,
    n_edge=4,
):
    if len(density_files) != len(delta_bulk_values):
        raise ValueError("density_files and delta_bulk_values must have the same length.")

    O_striated = []
    O_11 = []

    for filename in density_files:
        filepath = Path(density_folder) / filename
        density_dict = load_density_dict(filepath)
        full_density = dict_to_array(density_dict, Lx=Lx, Ly=Ly)

        bulk_density = full_density[
            n_edge : Ly - n_edge,
            n_edge : Lx - n_edge,
        ]

        O_striated.append(striated_order_parameter(bulk_density))
        O_11.append(sublattice_11_order_parameter(bulk_density))

    return {
        "delta_bulk": np.asarray(delta_bulk_values, dtype=float),
        "O_striated": np.asarray(O_striated, dtype=float),
        "O_11": np.asarray(O_11, dtype=float),
    }

# Plot order parameters
def plot_striated_and_11_order_parameters_uniform_vs_ramped(
    density_folder,
    uniform_density_files,
    ramped_density_files,
    delta_bulk_values,
    Lx=13,
    Ly=13,
    n_edge=4,
    save_path=None,
):
    # LaTeX rendering for subplots
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Nimbus Roman"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "legend.frameon": False,
    })

    # Extract OP results
    results_uniform = compute_order_parameters_from_file_list(
        density_folder=density_folder,
        density_files=uniform_density_files,
        delta_bulk_values=delta_bulk_values,
        Lx=Lx,
        Ly=Ly,
        n_edge=n_edge,
    )

    results_ramped = compute_order_parameters_from_file_list(
        density_folder=density_folder,
        density_files=ramped_density_files,
        delta_bulk_values=delta_bulk_values,
        Lx=Lx,
        Ly=Ly,
        n_edge=n_edge,
    )

    delta = np.asarray(delta_bulk_values, dtype=float)

    # Visualization
    fig, axes = plt.subplots(
        1, 2,
        figsize=(6.8, 2.8),
        dpi=600,
        constrained_layout=True,
    )

    # -----------------------------
    # Left: O_striated
    # -----------------------------
    axes[0].plot(
        delta,
        results_uniform["O_striated"],
        marker="s",
        linewidth=1.6,
        markersize=4,
        label=r"Uniform lattice",
    )
    axes[0].plot(
        delta,
        results_ramped["O_striated"],
        marker="o",
        linewidth=1.6,
        markersize=4,
        label=r"Boundary detuning",
    )
    axes[0].set_xlabel(r"$\delta_{\mathrm{bulk}}$")
    axes[0].set_ylabel(r"$O_{\mathrm{striated}}$")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    # -----------------------------
    # Right: O_(1,1)
    # -----------------------------
    axes[1].plot(
        delta,
        results_uniform["O_11"],
        marker="s",
        linewidth=1.6,
        markersize=4,
        label=r"Uniform lattice",
    )
    axes[1].plot(
        delta,
        results_ramped["O_11"],
        marker="o",
        linewidth=1.6,
        markersize=4,
        label=r"Boundary detuning",
    )
    axes[1].set_xlabel(r"$\delta_{\mathrm{bulk}}$")
    axes[1].set_ylabel(r"$O_{(1,1)\mathrm{-sublattice}}$")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)

    # Add a main title
    fig.suptitle(r"$R_b = 1.6$", y=1.07) 

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="png", bbox_inches="tight")
        print(f"✅ Saved: {save_path.resolve()}")
    else:
        plt.show()

    return results_uniform, results_ramped

# ======================================================================================
# Helper routine for Fig. 5(c) -- suppression of (1, 1)-sublattice density fluctuations
# ======================================================================================
def compute_star_order_parameter_from_file_list(
    density_folder,
    density_files,
    delta_bulk_values=None,
    Lx=13,
    Ly=13,
    n_edge=4,
):
    density_folder = Path(density_folder)

    if delta_bulk_values is not None and len(density_files) != len(delta_bulk_values):
        raise ValueError("density_files and delta_bulk_values must have the same length.")

    O_star = []

    for filename in density_files:
        filepath = density_folder / filename
        density_dict = load_density_dict(filepath)

        full_density = dict_to_array(density_dict, Lx=Lx, Ly=Ly)

        bulk_density = full_density[
            n_edge : Ly - n_edge,
            n_edge : Lx - n_edge,
        ]

        O_star.append(compute_bulk_order_parameter(bulk_density))

    results = {
        "O_star": np.asarray(O_star, dtype=float),
    }

    if delta_bulk_values is not None:
        results["delta_bulk"] = np.asarray(delta_bulk_values, dtype=float)

    return results

# Visualization
def plot_star_order_parameter_omega0_suppression_test(
    density_folder,
    uniform_density_files,
    ramped_density_files,
    delta_bulk_values,
    Lx=13,
    Ly=13,
    n_edge=4,
    save_path=None,
):
    # LaTeX rendering 
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Nimbus Roman"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "legend.frameon": False,
    })

    # Compute star OPs
    results_uniform = compute_star_order_parameter_from_file_list(
        density_folder=density_folder,
        density_files=uniform_density_files,
        delta_bulk_values=delta_bulk_values,
        Lx=Lx,
        Ly=Ly,
        n_edge=n_edge,
    )

    results_ramped = compute_star_order_parameter_from_file_list(
        density_folder=density_folder,
        density_files=ramped_density_files,
        delta_bulk_values=delta_bulk_values,
        Lx=Lx,
        Ly=Ly,
        n_edge=n_edge,
    )

    delta = np.asarray(delta_bulk_values, dtype=float)

    # Visualization
    fig, ax = plt.subplots(figsize=(3.6, 2.8), dpi=600, constrained_layout=True)

    ax.plot(
        delta,
        results_uniform["O_star"],
        marker="s",
        linewidth=1.6,
        markersize=4,
        label=r"Uniform drive ($\Omega_{x, y} = 1$)",
    )

    ax.plot(
        delta,
        results_ramped["O_star"],
        marker="o",
        linewidth=1.6,
        markersize=4,
        label=r"$\Omega_{x, y} = 0 ~\mathrm{if} ~x \equiv y \equiv 1 ~(\mathrm{mod} ~2)$",
    )

    ax.set_xlabel(r"$\delta_{\mathrm{bulk}}$")
    ax.set_ylabel(r"$O_{\mathrm{star}}$")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="png", bbox_inches="tight")
        print(f"✅ Saved: {save_path.resolve()}")
    else:
        plt.show()

    return results_uniform, results_ramped

## New Fig. 3 -- ramped detuning profile -- schematic
def plot_ramped_detuning_profile(
    Lx=13,
    Ly=13,
    n_edge=4,
    delta_bulk=4.9,
    delta_boundary=1.8,
    alpha=0.05,
    cmap="viridis",
    dpi=600,
    save_folder=None,
    filename="ramped_detuning_profile.pdf",
):
    # LaTeX rendering for figure
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Nimbus Roman"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "legend.frameon": False,
    })

    # Build the ramped detuning profile
    delta_interface = delta_bulk - alpha
    detuning = np.zeros((Ly, Lx))

    for y in range(Ly):
        for x in range(Lx):
            d = min(x, Lx - 1 - x, y, Ly - 1 - y)
            if d < n_edge:
                t = d / (n_edge - 1) if n_edge > 1 else 1.0
                detuning[y, x] = delta_boundary + (delta_interface - delta_boundary) * t
            else:
                detuning[y, x] = delta_bulk

    detuning = gaussian_filter(detuning, sigma=0.8)

    # Visualization
    fig, ax = plt.subplots(figsize=(2.4, 2.2), dpi=dpi)

    im = ax.imshow(detuning, cmap=cmap, origin="upper")
    ax.set_axis_off()

    # Add a black dashed line along the bulk region boundary
    # Draw dashed square outlining the bulk region
    bulk_x0 = n_edge - 0.5
    bulk_y0 = n_edge - 0.5
    
    rect = plt.Rectangle(
        (bulk_x0, bulk_y0),
        Lx - 2 * n_edge,
        Ly - 2 * n_edge,
        fill=False,
        edgecolor="black",
        linewidth=1.0,
        linestyle="--",
        alpha=0.8,
    )
    
    ax.add_patch(rect)

    # Colorbar rendering
    pos = ax.get_position()

    cax = fig.add_axes([
        pos.x0,         # left aligned with image
        pos.y0 - 0.04,  # same placement as before
        pos.width,      # same width as image
        0.025,          # thin horizontal bar
    ])
    
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    
    hi, low = im.get_clim()
    cbar.set_ticks([low, hi])
    cbar.set_ticklabels(["bulk", "low"])
    
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("bottom")
    
    # Only addition
    cbar.set_label(r"$\delta$", fontsize=12, labelpad=0)

    # Save figure if path is specified
    if save_folder is not None:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = save_folder / filename
        fig.savefig(save_path, format="pdf", bbox_inches="tight")
        print(f"✅ Saved: {save_path.resolve()}")
    else:
        plt.show()

    return detuning

# ====================================================================
# Helper functions to compute star OP and build the 2D phase diagram(s)
# ====================================================================
def generate_bulk_data(
    Lx,
    Ly,
    folder,
    filename,
    n_edge=3,
):
    filepath = Path(folder) / filename

    with open(filepath, "rb") as f:
        density_dict = pickle.load(f)

    return get_bulk(
        density_dict,
        Lx=Lx,
        Ly=Ly,
        n_edge=n_edge,
    )

def get_bulk(data, Lx, Ly, n_edge):
    # Function to generate the bulk data

    # -----------------------
    # Map dictionary to array
    # -----------------------
    full_lattice_data = np.zeros((Ly, Lx))
    for (y, x), v in data.items():
        full_lattice_data[y, x] = v

    # Get bulk data
    bulk_data = full_lattice_data[
        n_edge : Ly - n_edge,
        n_edge : Lx - n_edge
    ]
    return bulk_data

## Function to generate the list of bulk data for a given Rb
def order_parameter_list(Lx, Ly, folder, filenames, n_edge=2):
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
    # LateX rendering for subplots
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Nimbus Roman"],
        "mathtext.fontset": "stix",
        "font.size": 25,
        "legend.frameon": False,
    })

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

    # ============================================
    # Common colorbar (normalized OP values)
    # ============================================
    vmin = min(np.min(OP_grid_left), np.min(OP_grid_right))
    vmax = max(np.max(OP_grid_left), np.max(OP_grid_right))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # ==================================================
    # LEFT SUBPLOT
    # ==================================================
    scL = axes[0].scatter(
        xL,
        yL,
        c=colorsL,
        cmap=cmap,
        norm=norm, 
        s=450,
        edgecolors="k",
        linewidths=0.8,
        zorder=5,
    )

    axes[0].set_xlim(delta_min, delta_max)
    axes[0].set_ylim(Rb_min, Rb_max)
    # Restrict tick params shown
    axes[0].set_xticks(delta_values_left)
    axes[0].set_yticks(Rb_values_left)
    
    axes[0].set_xticklabels(
        [fr"${d:.1f}$" for d in delta_values_left],
    )
    axes[0].set_yticklabels(
        [fr"${r:.1f}$" for r in Rb_values_left],
    )

    # Axis specifications
    axes[0].set_xlabel(r"$\delta_{\mathrm{bulk}} ~(= \delta_{\mathrm{uniform}})$")
    axes[0].set_ylabel(r"$R_b$")
    axes[0].tick_params(axis="both")
    axes[0].set_title("Uniform lattice")

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
        norm=norm, 
        s=450,
        edgecolors="k",
        linewidths=0.8,
        zorder=5,
    )

    axes[1].set_xlim(delta_min, delta_max)
    axes[1].set_ylim(Rb_min, Rb_max)
    # Restrict tick params shown
    axes[1].set_xticks(delta_values_left)
    axes[1].set_yticks(Rb_values_left)
    
    axes[1].set_xticklabels(
        [fr"${d:.1f}$" for d in delta_values_left],
    )
    axes[1].set_yticklabels(
        [fr"${r:.2f}$" if abs(r-1.65)<1e-8 else fr"${r:.1f}$"
         for r in Rb_values_left],
    )

    # Axis specifications
    axes[1].set_xlabel(r"$\delta_{\mathrm{bulk}}$")
    axes[1].tick_params(axis="both")

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

    axes[1].set_title("Boundary detuning ramps")

    # Common colorbar
    fig.subplots_adjust(right=0.88)

    cax = fig.add_axes([
        0.90,   # left
        0.15,   # bottom
        0.02,   # width
        0.72,   # height
    ])
    
    cbar = fig.colorbar(scR, cax=cax)
    # Only show 4 values on the colorbar
    cbar.locator = MaxNLocator(nbins=4)
    cbar.update_ticks()

    # Colorbar specifications
    cbar.set_label(r"$O_{\mathrm{star}}$")
    cbar.ax.tick_params()

    # # Figure layout
    # plt.tight_layout()

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

# =================================================================
# Helper function to compute star / square order parameters and ..
# .. compare as a function of \delta_{\rm boundary}
# =================================================================
def plot_star_square_order_parameters_vs_delta_boundary(
    results,
    figsize=(7.2, 5.2),
    save_folder="figures",
    filename="star_square_order_parameters_vs_delta_boundary.pdf",
    dpi=600,
    use_tex=True,
    star_color="tab:red",
    square_color="tab:blue",
):
    """
    Plot O_star and O_square versus delta_boundary using separate y-axes.

    Notes
    -----
    The values stored under results["delta_bulk"] are interpreted here as
    delta_boundary values.

    Parameters
    ----------
    results : dict
        Dictionary containing the NumPy arrays:
            "O_star"
            "O_square"
            "delta_bulk"
    figsize : tuple, optional
        Figure size in inches.
    save_folder : str or pathlib.Path, optional
        Folder, relative to the main working directory, in which the PDF is
        saved.
    filename : str, optional
        Output PDF filename.
    dpi : int, optional
        Resolution used when saving.
    use_tex : bool, optional
        Whether to use an external LaTeX installation for text rendering.
    star_color, square_color : str, optional
        Colors used for O_star and O_square, respectively.

    Returns
    -------
    fig, ax_star, ax_square
        Matplotlib figure and the two y-axis objects.
    """

    required_keys = {"O_star", "O_square", "delta_bulk"}
    missing_keys = required_keys.difference(results)

    if missing_keys:
        raise KeyError(
            f"results is missing the required keys: {sorted(missing_keys)}"
        )

    # The stored delta_bulk values are actually delta_boundary values.
    delta_boundary = np.asarray(results["delta_bulk"], dtype=float)
    O_star = np.asarray(results["O_star"], dtype=float)
    O_square = np.asarray(results["O_square"], dtype=float)

    if not (
        delta_boundary.ndim == O_star.ndim == O_square.ndim == 1
    ):
        raise ValueError("All input arrays must be one-dimensional.")

    if not (
        len(delta_boundary) == len(O_star) == len(O_square)
    ):
        raise ValueError(
            "delta_boundary, O_star, and O_square must have equal lengths."
        )

    if len(delta_boundary) == 0:
        raise ValueError("The input arrays must not be empty.")

    # Sort by boundary detuning before joining points with lines.
    sort_indices = np.argsort(delta_boundary)
    delta_boundary = delta_boundary[sort_indices]
    O_star = O_star[sort_indices]
    O_square = O_square[sort_indices]

    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    save_path = save_folder / filename

    # LaTeX rendering for figure
    rc_params = {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Nimbus Roman"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "legend.frameon": False,
    }

    with plt.rc_context(rc_params):
        fig, ax_star = plt.subplots(
            figsize=figsize,
            dpi=dpi,
        )
    
        ax_square = ax_star.twinx()

        # --------------------------------------------------
        # Star order parameter: left axis
        # --------------------------------------------------
        line_star, = ax_star.plot(
            delta_boundary,
            O_star,
            marker="o",
            markersize=5,
            markeredgewidth=1.2,
            linewidth=2.2,
            linestyle="-",
            color=star_color,
            label=r"$O_{\mathrm{star}}$",
            zorder=3,
        )

        # --------------------------------------------------
        # Square order parameter: right axis
        # --------------------------------------------------
        line_square, = ax_square.plot(
            delta_boundary,
            O_square,
            marker="s",
            markersize=5,
            markeredgewidth=1.2,
            linewidth=2.2,
            linestyle="-",
            color=square_color,
            label=r"$O_{\mathrm{square}}$",
            zorder=3,
        )

        # --------------------------------------------------
        # Axis labels
        # --------------------------------------------------
        ax_star.set_xlabel(r"$\delta_{\mathrm{boundary}}$")
        ax_star.set_ylabel(
            r"$O_{\mathrm{star}}$",
            color=star_color,
        )
        ax_square.set_ylabel(
            r"$O_{\mathrm{square}}$",
            color=square_color,
        )

        # Color-code the y axes.
        ax_star.tick_params(
            axis="y",
            colors=star_color,
            width=1.2,
            length=5,
        )
        ax_square.tick_params(
            axis="y",
            colors=square_color,
            width=1.2,
            length=5,
        )
        ax_star.tick_params(
            axis="x",
            width=1.2,
            length=5,
        )
        ax_star.spines["left"].set_color(star_color)
        ax_star.spines["left"].set_linewidth(1.5)

        ax_square.spines["right"].set_color(square_color)
        ax_square.spines["right"].set_linewidth(1.5)

        # Start both order-parameter axes at zero.
        star_upper = max(1.08 * np.nanmax(O_star), 1.0e-12)
        square_upper = max(1.08 * np.nanmax(O_square), 1.0e-12)

        # Set ylimits for final figure
        ax_star.set_ylim(
            -0.03 * np.nanmax(O_star),
            1.08 * np.nanmax(O_star),
        )
        
        ax_square.set_ylim(
            -0.03 * np.nanmax(O_square),
            1.08 * np.nanmax(O_square),
        )

        # Display the supplied detuning values explicitly.
        ax_star.set_xticks(delta_boundary)

        # Grid belongs to the primary axis only.
        ax_star.grid(
            True,
            which="major",
            axis="both",
            linestyle="--",
            linewidth=0.8,
            alpha=0.45,
            zorder=0,
        )
        ax_star.set_axisbelow(True)

        # Combined legend for both axes.
        # ax_star.legend(
        #     handles=[line_star, line_square],
        #     loc="upper center",
        #     bbox_to_anchor=(0.5, 1.15),
        #     ncol=2,
        #     frameon=False,
        # )

        # Figure layout
        fig.tight_layout()

        # Save the figure as a PDF 
        fig.savefig(
            save_path,
            format="pdf",
            dpi=dpi,
            bbox_inches="tight",
        )

        # Notify once figure is saved
        print(f"Saved figure to: {save_path.resolve()}")

        # plt.show()

    return fig, ax_star, ax_square
