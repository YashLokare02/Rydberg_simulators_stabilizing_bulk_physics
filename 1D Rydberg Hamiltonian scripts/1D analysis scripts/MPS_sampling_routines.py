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

## Helper functions to implement the MPS sampling routine introduced in: Phys. Rev. B 85, 165146 (2012)
# 1. Compute right environments
# --------------------------------------------------------
def compute_right_envs(mps):
    N = len(mps)
    R = [None] * N

    Dr = mps[-1].shape[2]
    R[-1] = np.eye(Dr, dtype=complex)

    for i in reversed(range(N - 1)):
        A = mps[i + 1]  # (Dl, d, Dr)
        Rnext = R[i + 1]
        Dl, d, Dr = A.shape

        Ri = np.zeros((Dl, Dl), dtype=complex)
        for s in range(d):
            M = A[:, s, :]
            Ri += M @ (Rnext @ M.conj().T)

        R[i] = Ri

    return R

# --------------------------------------------------------
# 2. Perfect sampling (returns configuration + exact prob)
# --------------------------------------------------------
def perfect_sample(mps, R, tol=1e-12, verbose=False):
    N = len(mps)
    config = []
    L = np.array([[1.0+0j]])
    config_prob = 1.0  # Store the exact probability

    for i in range(N):
        A = mps[i]
        Dl, d, Dr = A.shape
        weights = np.zeros(d, dtype=float)

        for s in range(d):
            M = A[:, s, :]
            v = L @ M
            val = v @ (R[i] @ v.conj().T)

            # Handle numerical issues
            if abs(val.imag) > tol:
                if verbose:
                    print(f"[i={i}] Warning: imaginary part {val.imag}")
                val = val.real
            else:
                val = val.real

            weights[s] = val

        if np.any(weights < -tol):
            if verbose:
                print(f"[i={i}] Warning: negative weights found; clipping.")
            weights = np.maximum(weights, 0)

        total = weights.sum()
        if total < tol:
            raise ValueError(f"At site {i}: all weights ~0.")

        probs = weights / total
        s_choice = np.random.choice(d, p=probs)
        config.append(s_choice)

        config_prob *= probs[s_choice]   # Update exact probability
        M = A[:, s_choice, :]
        L = (L @ M) / np.sqrt(probs[s_choice])

    return np.array(config), config_prob

# --------------------------------------------------------
# 3. Sample many configurations
# --------------------------------------------------------
def sample_many_configs(mps, n_samples, verbose=False):
    R = compute_right_envs(mps)
    configs, exact_probs = [], []

    for _ in range(n_samples):
        cfg, p_exact = perfect_sample(mps, R, verbose=verbose)
        configs.append(cfg)
        exact_probs.append(p_exact)

    return np.array(configs), np.array(exact_probs)

# --------------------------------------------------------
# 4. Extract dominant configs using exact probs
# --------------------------------------------------------
def dominant_configs(configs, exact_probs, top_k=5):
    config_strings = ["".join(map(str, s)) for s in configs]
    prob_dict = {}
    count_dict = Counter(config_strings)

    for cfg_str, p in zip(config_strings, exact_probs):
        prob_dict[cfg_str] = prob_dict.get(cfg_str, 0) + p

    sorted_cfgs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    total_samples = len(configs)

    for cfg_str, exact_sum in sorted_cfgs:
        cfg = np.array(list(map(int, cfg_str)))
        count = count_dict[cfg_str]
        results.append({
            "config": cfg,
            "exact_prob": exact_sum,
            "count": count,
            "empirical_prob": count / total_samples
        })

    return results

# --------------------------------------------------------
# 5. Compute restricted IPR using exact probabilities
# --------------------------------------------------------
def restricted_selection_exact(dominant_list):
    weights = np.array([item["exact_prob"] for item in dominant_list])
    weights /= weights.sum()
    return np.sum(weights ** 2)

################# Helper functions to estimate average excitation spacing using sampled configurations from the MPS #######################
# NN spacing
# NN spacing
def nn_spacings_from_config(cfg, bulk_min=0, bulk_max=None):
    if bulk_max is None:
        bulk_max = len(cfg)
    idx = np.where(cfg[bulk_min:bulk_max] == 1)[0]
    if idx.size <= 1:
        return np.array([], dtype=int)
    return np.diff(idx)

# Compute dominant spacing + bootstrap estimate
def dominant_spacing_from_samples(spacings, spacing_weights=None,
                                  n_boot=200, rng=None):
    """
    Compute the dominant spacing r* and its weight, optionally using
    weights for each spacing event (e.g. Born probabilities of configs).

    spacings: 1D array of integer spacings
    spacing_weights: 1D array of same length as spacings, or None
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(spacings) == 0:
        return np.nan, np.nan, np.nan, np.array([])

    spacings = np.array(spacings)

    if spacing_weights is None:
        # Unweighted case (old behavior)
        N = len(spacings)
        cnt = Counter(spacings)
        r_star = max(cnt, key=cnt.get)
        p_star = cnt[r_star] / N

        boot_vals = []
        for _ in range(n_boot):
            res = rng.choice(spacings, size=N, replace=True)
            cnt_b = Counter(res)
            boot_vals.append(cnt_b[r_star] / N)

        boot_vals = np.array(boot_vals)
        p_std = boot_vals.std()

        return r_star, p_star, p_std, boot_vals

    # ---- Weighted case (with Born probabilities) ----
    spacing_weights = np.array(spacing_weights, dtype=float)
    W_tot = spacing_weights.sum()

    # Weighted histogram over spacings
    weight_dict = {}
    for r, w in zip(spacings, spacing_weights):
        weight_dict[r] = weight_dict.get(r, 0.0) + w

    # Dominant spacing = argmax_r weight_dict[r]
    r_star = max(weight_dict, key=weight_dict.get)
    p_star = weight_dict[r_star] / W_tot  # fraction of total weight

    # Bootstrap: resample indices with probability \propto spacing_weights
    N = len(spacings)
    probs_idx = spacing_weights / W_tot

    boot_vals = []
    for _ in range(n_boot):
        idxs = rng.choice(N, size=N, replace=True, p=probs_idx)
        res = spacings[idxs]
        cnt_b = Counter(res)
        boot_vals.append(cnt_b[r_star] / N)

    boot_vals = np.array(boot_vals)
    p_std = boot_vals.std()

    return r_star, p_star, p_std, boot_vals

# Computing spacing distribution + mean + bootstrap + dominant spacing
def spacing_statistics_for_mps(mps, Rb_value,
                               bulk_min=0, bulk_max=None,
                               n_samples=20000, n_boot=200,
                               rng=None):

    if rng is None:
        rng = np.random.default_rng()

    # STEP 1 — sample configurations + exact Born probabilities
    configs, config_probs = sample_many_configs(mps, n_samples=n_samples)
    configs = np.array(configs)
    config_probs = np.array(config_probs, dtype=float)

    # normalize in case they don't sum to 1 exactly (numerical noise)
    Z = config_probs.sum()
    if Z <= 0:
        raise ValueError("Config probabilities sum to non-positive value.")
    config_probs /= Z

    # STEP 2 — collect spacings + per-spacing weights
    all_spacings = []
    all_weights = []
    no_pair_weight = 0.0

    for cfg, p_cfg in zip(configs, config_probs):
        s = nn_spacings_from_config(cfg, bulk_min, bulk_max)
        if s.size == 0:
            no_pair_weight += p_cfg
        else:
            all_spacings.append(s)
            # each spacing in this config gets weight p_cfg
            all_weights.append(np.full_like(s, p_cfg, dtype=float))

    if len(all_spacings) > 0:
        all_spacings = np.concatenate(all_spacings)
        all_weights = np.concatenate(all_weights)
    else:
        all_spacings = np.array([], dtype=int)
        all_weights = np.array([], dtype=float)

    frac_no_pairs = no_pair_weight  # probability mass of configs with <2 excitations

    # Weighted mean, median-like measure
    if all_spacings.size > 0:
        mean_spacing = np.average(all_spacings, weights=all_weights)
        # median in weighted sense is trickier; we can approximate
        sort_idx = np.argsort(all_spacings)
        s_sorted = all_spacings[sort_idx]
        w_sorted = all_weights[sort_idx]
        cdf = np.cumsum(w_sorted) / w_sorted.sum()
        median_spacing = s_sorted[np.searchsorted(cdf, 0.5)]
    else:
        mean_spacing = median_spacing = np.nan

    # STEP 3 — bootstrap error bars on mean (weighted bootstrap)
    boot_means = []
    if all_spacings.size > 0:
        W_tot = all_weights.sum()
        probs_idx = all_weights / W_tot

        N_eff = len(all_spacings)  # number of spacing events
        for _ in range(n_boot):
            idxs = rng.choice(len(all_spacings), size=N_eff, replace=True, p=probs_idx)
            bs_spacings = all_spacings[idxs]
            bs_weights = all_weights[idxs]
            m_bs = np.average(bs_spacings, weights=bs_weights)
            boot_means.append(m_bs)

        boot_means = np.array(boot_means)
        lower, upper = np.percentile(boot_means, [2.5, 97.5])
    else:
        boot_means = np.array([])
        lower, upper = np.nan, np.nan

    # STEP 4 — dominant spacing with weights
    r_star, p_star, p_star_std, boot_p = dominant_spacing_from_samples(
        all_spacings,
        spacing_weights=all_weights if all_spacings.size > 0 else None,
        n_boot=n_boot,
        rng=rng
    )

    return {
        "Rb": Rb_value,
        "spacings": all_spacings,
        "spacing_weights": all_weights,
        "mean": mean_spacing,
        "median": median_spacing,
        "frac_no_pairs": frac_no_pairs,
        "boot_means": boot_means,
        "mean_CI": (lower, upper),
        "dominant_spacing": r_star,
        "dominant_prob": p_star,
        "dominant_prob_std": p_star_std,
        "bootstrap_p_vals": boot_p,
    }

# Full workflow
def spacing_vs_Rb_workflow(mps_storage, delta_fixed,
                           Rb_list,
                           bulk_min=10,
                           bulk_max=None,
                           n_samples=20000,
                           n_boot=200,
                          phase='disordered'):

    results = {}

    for Rb in Rb_list:
        if phase == 'disordered':
            mps = mps_storage[(Rb, delta_fixed)]
        elif phase == 'floating':
            mps = mps_storage[(0.1, "minus_minus")][18][Rb] # corresponds to fine scan results for n_boundary = 18 and \alpha = 0.1 (121-
            ## site chain)

        stats = spacing_statistics_for_mps(
            mps,
            Rb_value=Rb,
            bulk_min=bulk_min,
            bulk_max=bulk_max,
            n_samples=n_samples,
            n_boot=n_boot
        )

        results[Rb] = stats

        print(f"Completed Rb = {Rb} → mean spacing = {stats['mean']:.3f}, "
              f"dominant r = {stats['dominant_spacing']} "
              f"(p = {stats['dominant_prob']:.3f})")

    return results