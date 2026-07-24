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
from matplotlib.patches import Circle, Rectangle
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

## Helper functions (to extract wavefunction / MPS properties)
def get_per_site_excitation_densities(mps, N):
    """Compute <n_j> for all sites in the chain."""
    return [get_site_excitation_probability(mps, N, j) for j in range(N)]

# ---- Example usage with all MPSs ----
def compute_and_save_all_site_densities(all_mps, N, filename="site_densities.pkl"):
    """
    all_mps : dict
        Dictionary keyed by (R_b, delta), values are MPSs.
    N : int
        Number of sites in the chain.
    filename : str
        Output pickle filename.
    """
    all_site_densities = {}

    for (R_b, delta), mps in all_mps.items():
        site_probs = get_per_site_excitation_densities(mps, N)
        all_site_densities[(R_b, delta)] = site_probs

    # Save results for later analysis
    with open(filename, "wb") as f:
        pickle.dump(all_site_densities, f)

    return all_site_densities

def single_site_num_operator_MPO(N, j):
    """
    Construct MPO for number operator acting only on site j (0-indexed).
    """
    d = 2
    n_op = np.array([[0., 0.], [0., 1.]])
    I2   = np.eye(2)

    mpo = []
    for site in range(N):
        if site == j:
            W = n_op.reshape(1, d, 1, d)   # shape (1, d, 1, d)
        else:
            W = I2.reshape(1, d, 1, d)
        mpo.append(W)

    return mpo

def two_site_num_operator_MPO(N, j, l):
    """MPO for n_j n_l (with j,l zero-indexed)."""
    d = 2
    n_op = np.array([[0., 0.], [0., 1.]])
    I2   = np.eye(2)

    mpo = []
    for site in range(N):
        if site == j or site == l:
            W = n_op.reshape(1, d, 1, d)
        else:
            W = I2.reshape(1, d, 1, d)
        mpo.append(W)

    return mpo

def get_site_excitation_probability(mps, N, j):
    mpo = single_site_num_operator_MPO(N, j)
    Taux = np.ones((1, 1, 1))
    for l in range(N):
        Taux = ZipperLeft(Taux, mps[l].conj().T, mpo[l], mps[l])

    return Taux[0, 0, 0].real  # <n_j>

def overlap_mps(mps_1, mps_2, N):
    """
    Compute the overlap <mps_1 | mps_2> for an N-site chain.

    Parameters
    ----------
    mps_1, mps_2 : list of ndarrays
        MPS tensors of shape (D_left, d, D_right).
    N : int
        Number of sites.

    Returns
    -------
    overlap : float
        The scalar overlap <mps_1 | mps_2>.
    """
    d = mps_1[0].shape[1]  # local physical dimension
    # Identity MPO tensor (bond dim = 1)
    Id = np.zeros((1, d, 1, d), dtype=complex)
    for s in range(d):
        Id[0, s, 0, s] = 1.0

    # Initialize contraction
    Taux = np.ones((1, 1, 1), dtype=complex)

    # Sweep over sites
    for l in range(N):
        Taux = ZipperLeft(Taux, mps_1[l].conj().T, Id, mps_2[l])

    return Taux[0, 0, 0].real

def get_two_point_correlation(mps, N, j, l):
    mpo = two_site_num_operator_MPO(N, j, l)
    Taux = np.ones((1, 1, 1))
    for site in range(N):
        Taux = ZipperLeft(Taux, mps[site].conj().T, mpo[site], mps[site])

    return Taux[0, 0, 0].real

def get_correlation_matrix(mps, N):
    n_sites = np.array(get_per_site_excitation_densities(mps, N))
    corr_matrix = np.zeros((N, N))

    for j in range(N):
        for l in range(N):
            njnl = get_two_point_correlation(mps, N, j, l)
            corr_matrix[j, l] = njnl

    return corr_matrix

def correlation_vs_distance_bulk(mps, N, n_edge):
    """
    Compute averaged connected correlator C(r) vs distance r,
    restricted to the bulk region (excluding edges).

    Parameters
    ----------
    mps : MPS
        Optimized MPS.
    N : int
        Total chain length.
    n_edge : int
        Number of sites at each boundary to exclude (softened edges).

    Returns
    -------
    C_r : np.ndarray
        Connected correlation function in the bulk region.
    """
    # Define bulk region indices
    start_idx = n_edge
    end_idx = N - n_edge
    bulk_sites = list(range(start_idx, end_idx))
    N_bulk = len(bulk_sites)

    # Compute <n_j> for bulk sites
    n_sites_bulk = [get_site_excitation_probability(mps, N, j) for j in bulk_sites]

    C_r_bulk = []
    for r in range(N_bulk):
        vals = []
        for j in range(N_bulk - r):
            site_j = bulk_sites[j]
            site_l = bulk_sites[j + r]
            nj = n_sites_bulk[j]
            nl = n_sites_bulk[j + r]
            njnl = get_two_point_correlation(mps, N, site_j, site_l)
            vals.append(njnl - nj * nl)  # connected correlator
        if len(vals) > 0:
            C_r_bulk.append(np.mean(vals))
        else:
            C_r_bulk.append(np.nan)

    return np.array(C_r_bulk)

def compute_structure_factor_all_peaks_refined(
    N_fixed,
    optimized_mps,
    use_fluct=False,
    prominence=1,
    region="full",
    edge_cut=10,
    edge_extension=6,
    reference_bulk_size=85,
    oversample=32,
):
    """
    Compute structure factor peaks with discrete FFT + local continuous DFT refinement.

    Parameters
    ----------
    N_fixed : int
        Total chain length
    optimized_mps : dict
        Mapping Rb -> MPS
    use_fluct : bool
        Subtract mean density
    prominence : float
        Minimum prominence for peak detection
    region : str
        "full", "bulk", or "boundary"
    edge_cut : int
        Number of sites removed from edges for bulk
    oversample : int
        Local refinement resolution factor
    """
    peaks_dict = {}

    for Rb in sorted(optimized_mps.keys()):
        mps = optimized_mps[Rb]

        # Select region
        if region == "full":
            n_sites = np.array([get_site_excitation_probability(mps, N_fixed, j) for j in range(N_fixed)])
        elif region == "bulk":
            n_sites_full = np.array([get_site_excitation_probability(mps, N_fixed, j) for j in range(N_fixed)])
            start, end = edge_cut-1, N_fixed - edge_cut
            n_sites = n_sites_full[start:end]
        elif region == "boundary":
            n_sites_full = np.array([get_site_excitation_probability(mps, N_fixed, j) for j in range(N_fixed)])
            bulk_start = (N_fixed - reference_bulk_size) // 2
            bulk_end = bulk_start + reference_bulk_size
            left_added = n_sites_full[max(0, bulk_start - edge_extension):bulk_start]
            right_added = n_sites_full[bulk_end:min(N_fixed, bulk_end + edge_extension)]
            n_sites = np.concatenate([left_added, right_added])
        else:
            raise ValueError("region must be one of: 'full', 'bulk', or 'boundary'.")

        if use_fluct:
            n_sites = n_sites - np.mean(n_sites)

        N_region = len(n_sites)

        # ----- Step 1: Standard FFT -----
        n_fft = np.fft.fft(n_sites)
        S_fft = np.abs(n_fft)
        ks_fft = 2 * np.pi * np.arange(N_region) / N_region

        # Ignore k=0
        mask_nonzero = np.arange(N_region) != 0
        ks_fft = ks_fft[mask_nonzero]
        S_fft = S_fft[mask_nonzero]

        # Find integer-bin peaks
        peak_indices, _ = find_peaks(S_fft, prominence=prominence)
        if len(peak_indices) == 0:
            print(f"Rb={Rb:.4f}: no valid peaks found in {region} region.")
            peaks_dict[Rb] = {"k_peaks_2pi": [], "k_peaks": [], "S_peaks": []}
            continue

        k_peaks_refined = []
        S_peaks_refined = []

        # ----- Step 2: Local continuous DFT refinement -----
        j = np.arange(N_region)
        for idx in peak_indices:
            k_bin = ks_fft[idx]
            dk = 2 * np.pi / N_region
            k_local = np.linspace(k_bin - dk/2, k_bin + dk/2, oversample, endpoint=False)
            phase = np.exp(-1j * np.outer(k_local, j))
            n_k_local = phase.dot(n_sites)
            S_k_local = np.abs(n_k_local)

            # Refined peak
            peak_sub_idx = np.argmax(S_k_local)
            k_peak = k_local[peak_sub_idx]
            S_peak = S_k_local[peak_sub_idx]

            k_peaks_refined.append(k_peak)
            S_peaks_refined.append(S_peak)

        k_peaks_2pi = [k / (2*np.pi) for k in k_peaks_refined]

        peaks_dict[Rb] = {
            "k_peaks_2pi": k_peaks_2pi,
            "k_peaks": k_peaks_refined,
            "S_peaks": S_peaks_refined,
        }

        summary = ", ".join([f"k={kp/(2*np.pi):.4f}×2π (S={sp:.3f})"
                             for kp, sp in zip(k_peaks_refined, S_peaks_refined)])
        print(f"Rb={Rb:.4f} [{region}]: {summary}")

    return peaks_dict

## Helper functions to perform the Ornstein-Zernike fits and extract correlation properties
def fit_correlation_vs_distance_bulk(
    representative_points,
    mps_storage_by_n_edge,
    N_target,
    fit_range_ratio=0.6,
    min_fit_points=8,
    diagnostic_plots=True
):
    """
    Fit the bulk correlation function C(r) to the Ornstein–Zernike form
    for MPSs stored as mps_storage_by_n_edge[n_edge][Rb].
    The correlation function is computed only within the bulk region
    (excluding softened edges of size n_edge on both sides).

    Parameters
    ----------
    representative_points : dict
        Mapping from labels (e.g., "Point-1") → (Rb, n_edge).
    mps_storage_by_n_edge : dict
        From dmrg_sweep_n_edge(), structured as:
            mps_storage_by_n_edge[n_edge][Rb] = optimized MPS.
    N_target : int
        Total chain length.
    fit_range_ratio : float, optional
        Fraction of chain length to use for fitting (default 0.6).
    min_fit_points : int, optional
        Minimum number of points required for fitting (default 8).
    diagnostic_plots : bool, optional
        Whether to show diagnostic plots (default True).

    Returns
    -------
    results : dict
        results[(n_edge, label)] = {
            'xi': correlation length,
            'fit_params': parameters,
            'fit_errors': errors,
            'fit_quality': {r_squared, rmse},
            'convergence_info': status
        }
    """

    results = {}

    # Set up plotting grid
    if diagnostic_plots:
        n_points = len(representative_points)
        n_cols = min(2, n_points)
        n_rows = (n_points + n_cols - 1) // n_cols
        fig_main, axes_main = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), dpi=300)
        if n_points == 1:
            axes_main = [axes_main]
        elif n_rows > 1 and n_cols > 1:
            axes_main = axes_main.flatten()

    # Loop through representative points
    for idx, (point_key, (Rb, n_edge)) in enumerate(representative_points.items()):
        try:
            # Check that MPS exists
            if n_edge not in mps_storage_by_n_edge or Rb not in mps_storage_by_n_edge[n_edge]:
                print(f"Skipping {point_key}: missing MPS for n_edge={n_edge}, Rb={Rb}")
                results[(n_edge, point_key)] = {
                    'xi': np.nan,
                    'fit_params': [np.nan]*4,
                    'fit_errors': [np.nan]*4,
                    'fit_quality': {'r_squared': np.nan, 'rmse': np.nan},
                    'convergence_info': 'missing_data'
                }
                continue

            # Retrieve MPS and compute bulk correlation function
            mps = mps_storage_by_n_edge[n_edge][Rb]
            C_r_bulk = correlation_vs_distance_bulk(mps, N_target, n_edge)
            r_vals_bulk = np.arange(len(C_r_bulk))

            # Exclude r=0
            r_vals_bulk = r_vals_bulk[1:]
            C_r_bulk = C_r_bulk[1:]

            # Select fitting range
            max_fit_distance = int(fit_range_ratio * len(r_vals_bulk))
            max_fit_idx = max(min_fit_points, min(len(r_vals_bulk), max_fit_distance))
            r_fit = r_vals_bulk[:max_fit_idx]
            C_fit = C_r_bulk[:max_fit_idx]

            # Mask invalid values
            valid_mask = np.isfinite(C_fit) & (C_fit > 1e-8) & (C_fit <= 1.0)
            if np.sum(valid_mask) < min_fit_points:
                raise ValueError(f"Insufficient valid data points for {point_key}")

            r_fit = r_fit[valid_mask]
            C_fit = C_fit[valid_mask]

            # Initial parameter guesses
            A_guess = np.mean(C_fit[:3])
            xi_guess = estimate_initial_xi(r_fit, C_fit)
            k_guess = np.pi / 2
            phi0_guess = 0.0

            bounds_lower = [1e-3, 0.5, 0.1, -np.pi]
            bounds_upper = [2.0, N_target * 1.2, np.pi, np.pi]

            # Perform Ornstein–Zernike fit
            popt, pcov = curve_fit(
                ornstein_zernike_regularized,
                r_fit, C_fit,
                p0=[A_guess, xi_guess, k_guess, phi0_guess],
                bounds=(bounds_lower, bounds_upper),
                maxfev=10000
            )

            perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan]*4
            C_pred = ornstein_zernike_regularized(r_fit, *popt)

            # Compute quality metrics
            residuals = C_fit - C_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((C_fit - np.mean(C_fit))**2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean(residuals**2))
            xi_fit = popt[1]

            results[(n_edge, point_key)] = {
                'xi': xi_fit,
                'fit_params': popt,
                'fit_errors': perr,
                'fit_quality': {'r_squared': r_squared, 'rmse': rmse},
                'convergence_info': 'success'
            }

            # Plot results
            if diagnostic_plots and idx < len(axes_main):
                ax = axes_main[idx]
                ax.plot(r_vals_bulk, C_r_bulk, 'o', markersize=3, alpha=0.7, label='Data', color='blue')
                r_plot = np.linspace(min(r_vals_bulk), max(r_vals_bulk), 200)
                ax.plot(r_plot, ornstein_zernike_regularized(r_plot, *popt),
                        '-', linewidth=2, color='red',
                        label=fr'Fit: $\xi={xi_fit:.2f}$')
                ax.axvspan(min(r_fit), max(r_fit), alpha=0.2, color='gray', label='Fit region')
                ax.set_xlabel(r"$r_{\mathrm{bulk}}$", fontsize=19)
                ax.set_ylabel(r"$C(r)$", fontsize=19)
                ax.tick_params(axis='both', which='major', labelsize=17)
                ax.set_title(fr"{point_key}: $R_b={Rb:.2f},\ n_{{edge}}={n_edge}$" + f"\n$R^2={r_squared:.3f}$", fontsize=20)
                ax.legend(loc='best', fontsize=16)
                ax.grid(True, alpha=0.3)

        except Exception as e:
            print(f"Error fitting {point_key}: {e}")
            results[(n_edge, point_key)] = {
                'xi': np.nan,
                'fit_params': [np.nan]*4,
                'fit_errors': [np.nan]*4,
                'fit_quality': {'r_squared': np.nan, 'rmse': np.nan},
                'convergence_info': f'failed: {str(e)}'
            }

    if diagnostic_plots:
        for idx in range(len(representative_points), len(axes_main)):
            fig_main.delaxes(axes_main[idx])
        plt.tight_layout(pad=2.0)
        plt.show()

    return results

######################### More helper functions for the bulk correlation fitting function above #################################
def ornstein_zernike_regularized(r, A, xi, k, phi0):
    """
    Regularized Ornstein-Zernike correlation function to avoid divergence at r=0.
    """
    # Use r + 1 to avoid division by zero and ensure smooth behavior
    return A * np.exp(-r/xi) * np.cos(k*r + phi0) / np.sqrt(r + 1.0)


def estimate_initial_xi(r_vals, C_vals):
    """
    Better initial estimation of correlation length.
    """
    try:
        # Use logarithmic decay of envelope
        if len(C_vals) < 4:
            return 2.0

        # Simple exponential decay fit to first few points
        log_C = np.log(np.abs(C_vals) + 1e-8)
        valid_mask = (r_vals > 0) & np.isfinite(log_C)

        if np.sum(valid_mask) < 3:
            return 2.0

        r_valid = r_vals[valid_mask]
        log_C_valid = log_C[valid_mask]

        # Fit to linear decay
        coeffs = np.polyfit(r_valid[:min(6, len(r_valid))],
                           log_C_valid[:min(6, len(log_C_valid))], 1)
        xi_guess = -1.0 / coeffs[0] if coeffs[0] < -1e-6 else 2.0

        return max(0.5, min(xi_guess, 10.0))

    except:
        return 2.0  # Reasonable default

def estimate_parameter_errors(r_vals, C_vals, params, n_bootstrap=100):
    """
    Estimate parameter errors using bootstrap resampling.
    """
    try:
        n_samples = min(n_bootstrap, len(r_vals) // 2)
        param_samples = []

        for _ in range(n_samples):
            # Bootstrap resample
            indices = np.random.choice(len(r_vals), len(r_vals), replace=True)
            r_sample = r_vals[indices]
            C_sample = C_vals[indices]

            try:
                popt_sample, _ = curve_fit(
                    ornstein_zernike_regularized,
                    r_sample, C_sample,
                    p0=params,
                    bounds=([1e-3, 0.5, 0.1, -np.pi], [2.0, 100, np.pi, np.pi]),
                    maxfev=2000
                )
                param_samples.append(popt_sample)
            except:
                continue

        if len(param_samples) > 5:
            return np.std(param_samples, axis=0)
        else:
            return np.array([0.1 * abs(p) for p in params])  # Conservative estimate

    except:
        return np.array([0.1 * abs(p) for p in params])  # Fallback

## Helper functions to make the data dictionaries more user-friendly
def flatten_peaks_dict_variable_delta_boundary(structure_factor_comparison):
    """
    Flatten the nested peak dictionary for variable Δ_boundary cases.

    Input structure:
        structure_factor_comparison["linear_ramp"]["bulk"][n_edge][delta_boundary][Rb] = peaks_dict

    Output structure:
        flattened["linear_ramp"]["bulk"][n_edge][delta_boundary][Rb] = peaks_dict
        (same hierarchy, but inner levels flattened)
    """
    flattened = {}

    for bc_type, region_dict in structure_factor_comparison.items():
        flattened[bc_type] = {}
        for region, nedge_dict in region_dict.items():
            flattened[bc_type][region] = {}
            for n_edge, delta_dict in nedge_dict.items():
                flattened_delta_dict = {}
                for delta_boundary, rb_dict in delta_dict.items():
                    flattened_rb_dict = {}
                    for Rb_value, peaks in rb_dict.items():
                        # Flatten nested peak dictionaries if needed
                        if isinstance(peaks, dict) and len(peaks) == 1 and next(iter(peaks)) == Rb_value:
                            peaks = peaks[Rb_value]
                        flattened_rb_dict[Rb_value] = peaks
                    flattened_delta_dict[delta_boundary] = flattened_rb_dict
                flattened[bc_type][region][n_edge] = flattened_delta_dict
    return flattened

def flatten_peaks_dict_multiple_interfaces(structure_factor_comparison):
    """
    Flatten the nested peak dictionary for multiple interface configurations.

    Input structure:
        structure_factor_comparison[interface_config][bc_type]["bulk"][n_edge][delta_boundary][Rb] = peaks_dict

    Output structure:
        flattened[interface_config][bc_type]["bulk"][n_edge][delta_boundary][Rb] = peaks_dict
        (same hierarchy, but with inner peak dictionaries flattened)
    """
    flattened = {}

    for interface_config, bc_dict in structure_factor_comparison.items():
        flattened[interface_config] = {}
        for bc_type, region_dict in bc_dict.items():
            flattened[interface_config][bc_type] = {}
            for region, nedge_dict in region_dict.items():
                flattened[interface_config][bc_type][region] = {}
                for n_edge, delta_dict in nedge_dict.items():
                    flattened_delta_dict = {}
                    for delta_boundary, rb_dict in delta_dict.items():
                        flattened_rb_dict = {}
                        for Rb_value, peaks in rb_dict.items():
                            # Flatten nested peak dictionaries if redundant
                            if (
                                isinstance(peaks, dict)
                                and len(peaks) == 1
                                and next(iter(peaks)) == Rb_value
                            ):
                                peaks = peaks[Rb_value]
                            flattened_rb_dict[Rb_value] = peaks
                        flattened_delta_dict[delta_boundary] = flattened_rb_dict
                    flattened[interface_config][bc_type][region][n_edge] = flattened_delta_dict
    return flattened

def flatten_peaks_dict_uniform_chain(structure_factor_comparison):
    """
    Flatten the nested peak dictionary for both boundary types and regions.

    Input structure:
        structure_factor_comparison[bc_type][region][n_edge_used][Rb_value] = peaks_dict

    Output structure:
        flattened[bc_type][region][n_edge_used][Rb_value] = peaks_dict
        (same hierarchy, but inner one-level flattened)
    """
    flattened = {}

    for bc_type, region_dict in structure_factor_comparison.items():
        flattened[bc_type] = {}
        for region, nedge_dict in region_dict.items():
            flattened[bc_type][region] = {}
            for n_edge, rb_dict in nedge_dict.items():
                flattened_rb_dict = {}
                for Rb_outer, peaks in rb_dict.items():
                    # peaks may be nested (e.g., {Rb_value: peaks_dict}), flatten if needed
                    if isinstance(peaks, dict) and len(peaks) == 1 and next(iter(peaks)) == Rb_outer:
                        peaks = peaks[Rb_outer]
                    flattened_rb_dict[Rb_outer] = peaks
                flattened[bc_type][region][n_edge] = flattened_rb_dict
    return flattened

## Helper functions to analyze interface edge excitations
def compute_edge_excitation(mps_obj, N, left_idx, right_idx):
    """Return excitation density at the two bulk-edge sites."""
    exc = get_per_site_excitation_densities(mps_obj, N)
    return float(exc[left_idx]), float(exc[right_idx])


# =========================================================
# 1) NO DETUNING
# mps_storage_no_detuning[delta_boundary][n_edge][Rb]
# =========================================================
def compute_edge_exc_no_detuning(mps_storage_no_detuning,
                                 delta_boundary, n_edge):
    """Compute edge excitations for the no-detuning case."""
    results = {}

    if delta_boundary not in mps_storage_no_detuning:
        raise KeyError(f"delta_boundary={delta_boundary} missing")

    if n_edge not in mps_storage_no_detuning[delta_boundary]:
        raise KeyError(f"n_edge={n_edge} missing under delta_boundary={delta_boundary}")

    Rb_dict = mps_storage_no_detuning[delta_boundary][n_edge]

    print("\n=== Edge excitations: no-detuning case ===")

    for Rb in sorted(Rb_dict.keys(), key=float):
        mps = Rb_dict[Rb]
        left_exc, right_exc = compute_edge_excitation(mps, N, left_idx, right_idx)
        results[float(Rb)] = (left_exc, right_exc)

        print(f"Rb={float(Rb):.6f}  left={left_exc:.6e}  right={right_exc:.6e}")

    return results


# =========================================================
# 2) FINITE DETUNING
# mps_storage_detuned[(alpha, config_label, n_edge)][n_edge][Rb]
# =========================================================
def compute_edge_exc_detuned(mps_storage_detuned,
                             alpha, config_label, n_edge):
    """Compute edge excitations for the finite-detuning case."""
    results = {}

    outer_key = (alpha, config_label)

    if outer_key not in mps_storage_detuned:
        raise KeyError(f"Key {outer_key} missing in detuned dictionary")

    if n_edge not in mps_storage_detuned[outer_key]:
        raise KeyError(f"n_edge={n_edge} missing for outer key={outer_key}")

    Rb_dict = mps_storage_detuned[outer_key][n_edge]

    print("\n=== Edge excitations: detuned case ===")

    for Rb in sorted(Rb_dict.keys(), key=float):
        mps = Rb_dict[Rb]
        left_exc, right_exc = compute_edge_excitation(mps, N, left_idx, right_idx)
        results[float(Rb)] = (left_exc, right_exc)

        print(f"Rb={float(Rb):.6f}  left={left_exc:.6e}  right={right_exc:.6e}")

    return results

# =============================================
# Helper function to run k-deviation analysis
# =============================================
def compute_wavevector_deviation_from_1009(
    reference_peaks_1009,
    ramped_results_by_N,
    uniform_results_by_N,
    Rb_reference_values=None,
    N_values=None,
    scale_factor=2 ** (1 / 6),
    Rb_tolerance=1e-6,
    verbose=True,
):
    """
    Compare first wavevectors from ramped and uniform finite chains
    against the corresponding 1009-site uniform-chain reference.

    Expected finite-chain hierarchy
    --------------------------------
    results_by_N[N]["peaks"][Rb_target] = {
        "Rb_requested": ...,
        "Rb_stored": ...,
        "structure_factor_result": {
            Rb_stored: {
                "k_peaks_2pi": [...],
                "k_peaks": [...],
                "S_peaks": [...],
            }
        },
    }
    """

    # Rb values at which computations are to be run
    if Rb_reference_values is None:
        Rb_reference_values = [
            3.06,
            3.07,
            3.08,
            3.09,
            3.10,
            3.11,
            3.12,
            3.13,
        ]

    # Chain lengths considered
    if N_values is None:
        N_values = [
            121,
            133,
            145,
            157,
            169,
            181,
            193,
            205,
        ]

    # Rb reference values
    Rb_reference_values = np.asarray(
        Rb_reference_values,
        dtype=float,
    )

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------
    def find_closest_key(dictionary, target):
        if not isinstance(dictionary, dict) or len(dictionary) == 0:
            raise ValueError(
                "Cannot search for an Rb key in an empty dictionary."
            )

        return min(
            dictionary.keys(),
            key=lambda key: abs(float(key) - float(target)),
        )

    def extract_first_k_over_2pi(peak_entry):
        k_peaks = np.asarray(
            peak_entry.get("k_peaks_2pi", []),
            dtype=float,
        )

        if k_peaks.size == 0:
            return np.nan

        return float(k_peaks[0])

    def unwrap_peak_entry(result_entry):
        structure_factor_result = result_entry[
            "structure_factor_result"
        ]

        if not isinstance(structure_factor_result, dict):
            raise TypeError(
                "structure_factor_result must be a dictionary."
            )

        # Already an unwrapped peak dictionary
        if "k_peaks_2pi" in structure_factor_result:
            return structure_factor_result

        if len(structure_factor_result) == 0:
            return {}

        Rb_stored = result_entry.get(
            "Rb_stored",
            result_entry.get("Rb_requested", None),
        )

        if Rb_stored is not None:
            inner_key = find_closest_key(
                structure_factor_result,
                Rb_stored,
            )
            return structure_factor_result[inner_key]

        first_key = next(iter(structure_factor_result))
        return structure_factor_result[first_key]

    # --------------------------------------------------------
    # Output hierarchy
    # --------------------------------------------------------
    results = {
        "scale_factor": float(scale_factor),
        "Rb_reference": Rb_reference_values.copy(),
        "Rb_finite_target": (
            Rb_reference_values * scale_factor
        ),
        "N_values": list(N_values),
        "reference_1009": {
            "Rb": [],
            "Rb_stored": [],
            "Rb_difference": [],
            "k_first_over_2pi": [],
        },
        "by_N": {},
        "by_Rb": {},
    }

    # ========================================================
    # Extract 1009-site reference trajectory
    # ========================================================
    for Rb_reference in Rb_reference_values:
        reference_key = find_closest_key(
            reference_peaks_1009,
            Rb_reference,
        )

        Rb_reference_stored = float(reference_key)

        reference_difference = abs(
            Rb_reference_stored - float(Rb_reference)
        )

        reference_peak_entry = (
            reference_peaks_1009[reference_key]
        )

        k_reference = extract_first_k_over_2pi(
            reference_peak_entry
        )

        results["reference_1009"]["Rb"].append(
            float(Rb_reference)
        )
        results["reference_1009"]["Rb_stored"].append(
            Rb_reference_stored
        )
        results["reference_1009"]["Rb_difference"].append(
            reference_difference
        )
        results["reference_1009"][
            "k_first_over_2pi"
        ].append(
            k_reference
        )

        if (
            verbose
            and reference_difference > Rb_tolerance
        ):
            print(
                "Warning for 1009-site reference: "
                f"requested Rb={Rb_reference:.10f}, "
                f"using stored Rb={Rb_reference_stored:.10f}, "
                f"|difference|={reference_difference:.3e}."
            )

    for key in results["reference_1009"]:
        results["reference_1009"][key] = np.asarray(
            results["reference_1009"][key],
            dtype=float,
        )

    k_reference_array = results[
        "reference_1009"
    ]["k_first_over_2pi"]

    # ========================================================
    # Extract ramped and uniform finite-chain trajectories
    # ========================================================
    for N in N_values:
        if N not in ramped_results_by_N:
            raise KeyError(
                f"N={N} was not found in ramped_results_by_N."
            )

        if N not in uniform_results_by_N:
            raise KeyError(
                f"N={N} was not found in uniform_results_by_N."
            )

        ramped_data_N = ramped_results_by_N[N]
        uniform_data_N = uniform_results_by_N[N]

        if "peaks" not in ramped_data_N:
            raise KeyError(
                f"Ramped N={N} entry does not contain 'peaks'."
            )

        if "peaks" not in uniform_data_N:
            raise KeyError(
                f"Uniform N={N} entry does not contain 'peaks'."
            )

        ramped_peaks_N = ramped_data_N["peaks"]
        uniform_peaks_N = uniform_data_N["peaks"]

        N_result = {
            "N": N,
            "n_edge": ramped_data_N.get("n_edge", None),
            "bulk_size": ramped_data_N.get(
                "bulk_size",
                None,
            ),
            "scan_key": ramped_data_N.get(
                "scan_key",
                None,
            ),
            "uniform_delta": uniform_data_N.get(
                "delta",
                None,
            ),
            "Rb_reference": [],
            "Rb_finite_target": [],
            "k_reference_1009": [],

            "Rb_ramped_stored": [],
            "Rb_ramped_difference": [],
            "k_ramped": [],
            "ramped_absolute_deviation": [],

            "Rb_uniform_stored": [],
            "Rb_uniform_difference": [],
            "k_uniform": [],
            "uniform_absolute_deviation": [],
        }

        for index, Rb_reference in enumerate(
            Rb_reference_values
        ):
            Rb_finite_target = (
                float(Rb_reference) * scale_factor
            )

            # ------------------------------------------------
            # Ramped chain
            # ------------------------------------------------
            ramped_key = find_closest_key(
                ramped_peaks_N,
                Rb_finite_target,
            )

            ramped_result_entry = (
                ramped_peaks_N[ramped_key]
            )

            Rb_ramped_stored = float(
                ramped_result_entry.get(
                    "Rb_stored",
                    ramped_key,
                )
            )

            Rb_ramped_difference = abs(
                Rb_ramped_stored - Rb_finite_target
            )

            ramped_peak_entry = unwrap_peak_entry(
                ramped_result_entry
            )

            k_ramped = extract_first_k_over_2pi(
                ramped_peak_entry
            )

            # ------------------------------------------------
            # Uniform finite chain
            # ------------------------------------------------
            uniform_key = find_closest_key(
                uniform_peaks_N,
                Rb_finite_target,
            )

            uniform_result_entry = (
                uniform_peaks_N[uniform_key]
            )

            Rb_uniform_stored = float(
                uniform_result_entry.get(
                    "Rb_stored",
                    uniform_key,
                )
            )

            Rb_uniform_difference = abs(
                Rb_uniform_stored - Rb_finite_target
            )

            uniform_peak_entry = unwrap_peak_entry(
                uniform_result_entry
            )

            k_uniform = extract_first_k_over_2pi(
                uniform_peak_entry
            )

            # ------------------------------------------------
            # Deviations from 1009-site reference
            # ------------------------------------------------
            k_reference = k_reference_array[index]

            if (
                np.isfinite(k_reference)
                and np.isfinite(k_ramped)
            ):
                ramped_absolute_deviation = abs(
                    k_ramped - k_reference
                )
            else:
                ramped_absolute_deviation = np.nan

            if (
                np.isfinite(k_reference)
                and np.isfinite(k_uniform)
            ):
                uniform_absolute_deviation = abs(
                    k_uniform - k_reference
                )
            else:
                uniform_absolute_deviation = np.nan

            # ------------------------------------------------
            # Store
            # ------------------------------------------------
            N_result["Rb_reference"].append(
                float(Rb_reference)
            )
            N_result["Rb_finite_target"].append(
                Rb_finite_target
            )
            N_result["k_reference_1009"].append(
                k_reference
            )

            N_result["Rb_ramped_stored"].append(
                Rb_ramped_stored
            )
            N_result["Rb_ramped_difference"].append(
                Rb_ramped_difference
            )
            N_result["k_ramped"].append(
                k_ramped
            )
            N_result[
                "ramped_absolute_deviation"
            ].append(
                ramped_absolute_deviation
            )

            N_result["Rb_uniform_stored"].append(
                Rb_uniform_stored
            )
            N_result["Rb_uniform_difference"].append(
                Rb_uniform_difference
            )
            N_result["k_uniform"].append(
                k_uniform
            )
            N_result[
                "uniform_absolute_deviation"
            ].append(
                uniform_absolute_deviation
            )

            if (
                verbose
                and Rb_ramped_difference > Rb_tolerance
            ):
                print(
                    f"Warning for ramped N={N}: "
                    f"reference Rb={Rb_reference:.10f}, "
                    f"target Rb={Rb_finite_target:.10f}, "
                    f"stored Rb={Rb_ramped_stored:.10f}, "
                    f"|difference|={Rb_ramped_difference:.3e}."
                )

            if (
                verbose
                and Rb_uniform_difference > Rb_tolerance
            ):
                print(
                    f"Warning for uniform N={N}: "
                    f"reference Rb={Rb_reference:.10f}, "
                    f"target Rb={Rb_finite_target:.10f}, "
                    f"stored Rb={Rb_uniform_stored:.10f}, "
                    f"|difference|={Rb_uniform_difference:.3e}."
                )

        numeric_fields = [
            "Rb_reference",
            "Rb_finite_target",
            "k_reference_1009",
            "Rb_ramped_stored",
            "Rb_ramped_difference",
            "k_ramped",
            "ramped_absolute_deviation",
            "Rb_uniform_stored",
            "Rb_uniform_difference",
            "k_uniform",
            "uniform_absolute_deviation",
        ]

        for field in numeric_fields:
            N_result[field] = np.asarray(
                N_result[field],
                dtype=float,
            )

        # Backward-compatible alias for previous ramped plotting
        N_result["absolute_deviation"] = (
            N_result["ramped_absolute_deviation"]
        )

        results["by_N"][N] = N_result

    # ========================================================
    # Reorganize by reference Rb
    # ========================================================
    for index, Rb_reference in enumerate(
        Rb_reference_values
    ):
        Rb_float = float(Rb_reference)

        results["by_Rb"][Rb_float] = {
            "Rb_reference": Rb_float,
            "Rb_finite_target": float(
                Rb_reference * scale_factor
            ),
            "k_reference_1009": float(
                k_reference_array[index]
            ),
            "chains": {},
        }

        for N in N_values:
            N_result = results["by_N"][N]

            results["by_Rb"][Rb_float]["chains"][N] = {
                "n_edge": N_result["n_edge"],

                "Rb_ramped_stored": float(
                    N_result["Rb_ramped_stored"][index]
                ),
                "k_ramped": float(
                    N_result["k_ramped"][index]
                ),
                "ramped_absolute_deviation": float(
                    N_result[
                        "ramped_absolute_deviation"
                    ][index]
                ),

                "Rb_uniform_stored": float(
                    N_result["Rb_uniform_stored"][index]
                ),
                "k_uniform": float(
                    N_result["k_uniform"][index]
                ),
                "uniform_absolute_deviation": float(
                    N_result[
                        "uniform_absolute_deviation"
                    ][index]
                ),
            }

    # ========================================================
    # Summary
    # ========================================================
    if verbose:
        print("\n" + "#" * 115)
        print(
            "First-wavevector deviations from the "
            "1009-site uniform-chain reference"
        )
        print("#" * 115)

        for index, Rb_reference in enumerate(
            Rb_reference_values
        ):
            print(
                f"\nRb_reference={Rb_reference:.2f}, "
                f"Rb_finite_target="
                f"{Rb_reference * scale_factor:.10f}, "
                f"k_1009/(2pi)="
                f"{k_reference_array[index]:.10f}"
            )

            for N in N_values:
                N_result = results["by_N"][N]

                print(
                    f"  N={N:3d}, "
                    f"n_edge={N_result['n_edge']:2d}, "
                    f"k_ramped/(2pi)="
                    f"{N_result['k_ramped'][index]:.10f}, "
                    f"|Delta k|_ramped="
                    f"{N_result['ramped_absolute_deviation'][index]:.10e}, "
                    f"k_uniform/(2pi)="
                    f"{N_result['k_uniform'][index]:.10f}, "
                    f"|Delta k|_uniform="
                    f"{N_result['uniform_absolute_deviation'][index]:.10e}"
                )

        print("\n" + "#" * 115)

    return results
