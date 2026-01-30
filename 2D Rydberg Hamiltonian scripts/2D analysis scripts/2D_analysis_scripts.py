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

## Helper functions to compute various order parameters to identify ordered states on 2D Rydberg lattices
# Get the bulk data
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

# Function to compute the bulk order parameter (corresponding to the star phase)
def compute_bulk_order_parameter(bulk):
    """
    Compute bulk anisotropy order parameter:
        sum_{x,y} (n_xy - n_yx)^2 / N_bulk
    Assumes bulk is square.
    """
    Ly, Lx = bulk.shape
    assert Lx == Ly, "Bulk must be square to compute x<->y order parameter"

    diff = bulk - bulk.T
    N_bulk = Lx * Ly
    return np.sum(diff**2) / N_bulk

# Functions to compute the square / striated and (1, 1)-sublattice order parameters
def F_tilde_unsym(n2d: np.ndarray, k1: float, k2: float) -> float:
    """
    Compute  \\tilde{F}(k1,k2) = | sum_{x,y} exp(i (k1 x + k2 y)) n[x,y] | / N

    Here n2d is assumed to be indexed as n2d[y, x].
    N = Lx * Ly.
    """
    n2d = np.asarray(n2d)
    if n2d.ndim != 2:
        raise ValueError("n2d must be a 2D array with shape (Ly, Lx).")

    Ly, Lx = n2d.shape
    N = Lx * Ly

    x = np.arange(Lx)
    y = np.arange(Ly)

    # exp(i k1 x) and exp(i k2 y)
    phase_x = np.exp(1j * k1 * x)          # (Lx,)
    phase_y = np.exp(1j * k2 * y)          # (Ly,)

    # phase[y,x] = exp(i(k1 x + k2 y))
    phase = np.outer(phase_y, phase_x)     # (Ly, Lx)

    amp = np.sum(n2d * phase)
    return np.abs(amp) / N


def F_tilde_sym(n2d: np.ndarray, k1: float, k2: float) -> float:
    """
    Symmetrized  \\widetilde{F}(k1,k2) = ( \\tilde{F}(k1,k2) + \\tilde{F}(k2,k1) ) / 2
    """
    return 0.5 * (F_tilde_unsym(n2d, k1, k2) + F_tilde_unsym(n2d, k2, k1))


def striated_order_parameter(n2d: np.ndarray) -> float:
    """
    O_str = \\widetilde{F}(pi,0) - \\widetilde{F}(pi/2, pi)
    """
    term1 = F_tilde_sym(n2d, np.pi, 0.0)
    term2 = F_tilde_sym(n2d, 0.5 * np.pi, np.pi)
    return term1 - term2

def sublattice_11_order_parameter(n2d: np.ndarray) -> float:
    """
    Compute the (1,1)-sublattice order parameter:

        O = (4/N) * sum_{x,y} n[x,y]
            with x mod 2 = 1 and y mod 2 = 1

    Parameters
    ----------
    n2d : np.ndarray
        2D array of excitation densities, indexed as n2d[y, x].

    Returns
    -------
    float
        Sublattice (1,1) order parameter.
    """
    n2d = np.asarray(n2d)
    if n2d.ndim != 2:
        raise ValueError("n2d must be a 2D array with shape (Ly, Lx).")

    Ly, Lx = n2d.shape
    N = Lx * Ly

    # indices
    x = np.arange(Lx)
    y = np.arange(Ly)

    # mask for x mod 2 = 1 and y mod 2 = 1
    mask_x = (x % 2 == 1)
    mask_y = (y % 2 == 1)

    # full 2D mask: mask[y, x]
    mask = np.outer(mask_y, mask_x)

    return 4.0 * np.sum(n2d[mask]) / N