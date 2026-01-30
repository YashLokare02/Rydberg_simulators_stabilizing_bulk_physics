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

## Helper functions
'''
    Function that takes an MPS 'M' as input (order of legs: left-bottom-right) and returns a copy of it that is
        transformed into left canonical form and normalized.
    Go through the explanation provided for more details.
'''
def LeftCanonical(M):
    Mcopy = M.copy() #create copy of M

    N = len(Mcopy) #nr of sites

    for l in range(N):
        # reshape
        Taux = Mcopy[l]
        Taux = np.reshape(Taux,(np.shape(Taux)[0]*np.shape(Taux)[1],np.shape(Taux)[2]))

        # SVD
        U,S,Vdag = np.linalg.svd(Taux,full_matrices=False)
        '''
            Note: full_matrices=False leads to a trivial truncation of the matrices (thin SVD).
        '''

        # update M[l]
        Mcopy[l] = np.reshape(U,(np.shape(Mcopy[l])[0],np.shape(Mcopy[l])[1],np.shape(U)[1]))

        # update M[l+1]
        SVdag = np.matmul(np.diag(S),Vdag)
        if l < N-1:
            Mcopy[l+1] = np.einsum('ij,jkl',SVdag,Mcopy[l+1])
        else:
            '''
                Note: in the last site (l=N-1), S*Vdag is a number that determines the normalization of the MPS.
                    We discard this number, which corresponds to normalizing the MPS.
            '''

    return Mcopy

# MPS to right-canonical form
def RightCanonical(M):

    Mcopy = M.copy()  # copy the MPS
    N = len(Mcopy)    # number of sites

    for l in reversed(range(N)):
        # reshape: merge left and physical legs
        Taux = Mcopy[l]
        Taux = np.reshape(Taux, (np.shape(Taux)[0], -1))  # (left, phys*right) → prepare for right-to-left sweep

        # SVD
        U, S, Vdag = np.linalg.svd(Taux, full_matrices=False)

        # update M[l] ← V† reshaped back
        Mcopy[l] = np.reshape(Vdag, (np.shape(S)[0], np.shape(Mcopy[l])[1], np.shape(Mcopy[l])[2]))

        # push U·S to the previous site
        if l > 0:
            US = np.matmul(U, np.diag(S))
            Mcopy[l-1] = np.einsum('ijk,kl->ijl', Mcopy[l-1], US)
        else:
            # site l = 0: drop norm factor
            pass

    return Mcopy

# Functions for tensor contraction
'''
    Function that makes the following contractions (numbers denote leg order):

         /--3--**--1--Mt--3--
         |             |
         |             2
         |             |
         |             *
         |             *
         |             |
         |             4                 /--3--
         |             |                 |
        Tl--2--**--1---O--3--     =     Tf--2--
         |             |                 |
         |             2                 \--1--
         |             |
         |             *
         |             *
         |             |
         |             2
         |             |
         \--1--**--3--Mb--1--
'''
def ZipperLeft(Tl,Mb,O,Mt):
    Taux = np.einsum('ijk,klm',Mb,Tl)
    Taux = np.einsum('ijkl,kjmn',Taux,O)
    Tf = np.einsum('ijkl,jlm',Taux,Mt)

    return Tf

'''
    Function that makes the following contractions (numbers denote leg order):

         --1--Mt--3--**--1--\
               |            |
               2            |
               |            |
               *            |
               *            |
               |            |
               4            |            --1--\
               |            |                 |
         --1---O--3--**--2--Tr     =     --2--Tf
               |            |                 |
               2            |            --3--/
               |            |
               *            |
               *            |
               |            |
               2            |
               |            |
         --3--Mb--1--**--3--/
'''
def ZipperRight(Tr,Mb,O,Mt):
    Taux = np.einsum('ijk,klm',Mt,Tr)
    Taux = np.einsum('ijkl,mnkj',Taux,O)
    Tf = np.einsum('ijkl,jlm',Taux,Mb)

    return Tf

def entanglement_entropy(S, eps=1e-15):
    """
    von Neumann entropy from Schmidt values S (singular values).
    Normalizes probabilities p_i = S_i^2 / sum(S^2) and removes tiny terms.
    """
    p = S**2
    norm = p.sum()
    if norm <= 0:
        return 0.0
    p = p / norm
    p = p[p > eps]
    return -np.sum(p * np.log(p))

## 1D Rydberg Hamiltonian MPO construction + additional helper functions
# -----------------------
# 1. Build single-site interaction MPO tensor
# -----------------------
def single_site_interaction_MPO_clean(A_list, b_list, R_b):
    """
    Build numeric single-site MPO tensor W_tensor of shape (D, d, D, d).
    Block structure repeated K times where D = 3*K.
    """
    A_list = np.asarray(A_list, dtype=float)
    b_list = np.asarray(b_list, dtype=float)
    K = len(A_list)
    D = 3 * K
    d = 2
    I2 = np.eye(2, dtype=float)
    n_op = np.array([[0., 0.], [0., 1.]], dtype=float)

    W = np.zeros((D, d, D, d), dtype=float)

    for k in range(K):
        base = 3 * k
        A_k = A_list[k]
        b_k = b_list[k]
        x_k = np.exp(-b_k)

        # block entries (2x2 each)
        W[base + 0, :, base + 0, :] = I2
        W[base + 1, :, base + 0, :] = n_op
        W[base + 1, :, base + 1, :] = x_k * I2
        W[base + 2, :, base + 1, :] = (A_k * x_k * (R_b**6)/2) * n_op
        W[base + 2, :, base + 2, :] = I2

    return W

# -----------------------
# 2. Build chain MPO from single-site tensor with OBC
# -----------------------
def build_interaction_chain_from_Wtensor_corrected(W_tensor, N):
    """
    Build chain MPO list of length N from W_tensor (D,d,D,d).
    Activates all channels at left and right boundaries.
    """
    D, d1, D2, d2 = W_tensor.shape
    assert d1 == d2 == 2
    assert D == D2
    assert D % 3 == 0
    K = D // 3

    start_cols = np.arange(0, D, 3)        # start columns of each channel
    end_rows   = start_cols + 2            # end rows of each channel

    H = [None] * N

    # Left boundary: sum of all end_rows rows
    H[0] = np.sum(W_tensor[end_rows, :, :, :], axis=0)[None, :, :, :]

    # Bulk sites: copy W_tensor
    for site in range(1, N-1):
        H[site] = W_tensor.copy()

    # Right boundary: sum of all start_cols columns
    H[-1] = np.sum(W_tensor[:, :, start_cols, :], axis=2)[:, :, None, :]

    return H

def onsite_MPO(
    N,
    Omega,
    delta_bulk,
    boundary_size=0,
    boundary_type=None,
    delta_boundary=None,
    alpha_left=0.0,
    alpha_right=0.0
):
    """
    Build the on-site MPO for a 1D Rydberg chain with site-dependent detuning delta[i],
    including optional uniform or linear boundary ramps.

    Parameters
    ----------
    N : int
        Number of sites.
    Omega : float
        Rabi frequency.
    delta_bulk : float
        Detuning in the bulk sites.
    boundary_size : int, optional
        Number of sites in each boundary region. Default is 0.
    boundary_type : {'uniform', 'linear', None}, optional
        Type of boundary detuning ramp.
        - 'uniform' : Constant detuning delta_boundary at boundary sites.
        - 'linear'  : Linear interpolation between
                      delta_boundary (edge) and delta_bulk ± alpha (interface).
        Default is None.
    delta_boundary : float, optional
        Detuning at the first and last sites (edge detuning).
        Required for 'uniform' or 'linear' boundaries.
    alpha_left : float, optional
        Detuning mismatch at the left boundary–bulk interface:
        interface detuning = delta_bulk + alpha_left.
    alpha_right : float, optional
        Detuning mismatch at the right boundary–bulk interface:
        interface detuning = delta_bulk + alpha_right.
    """

    import numpy as np

    d = 2
    D = 2
    I2 = np.eye(2)
    n_op = np.array([[0., 0.], [0., 1.]])
    sigma_x = np.array([[0., 1.], [1., 0.]])

    # Start with uniform bulk detuning
    delta = np.full(N, delta_bulk, dtype=float)

    if boundary_size > 0 and boundary_type is not None:
        btype = boundary_type.lower()

        # -------------------------------------
        # UNIFORM boundary case
        # -------------------------------------
        if btype == "uniform":
            if delta_boundary is None:
                raise ValueError("For 'uniform' boundaries, delta_boundary must be specified.")
            delta[:boundary_size] = delta_boundary
            delta[-boundary_size:] = delta_boundary

        # -------------------------------------
        # LINEAR boundary case
        # -------------------------------------
        elif btype == "linear":
            if delta_boundary is None:
                raise ValueError("For 'linear' boundaries, delta_boundary must be specified.")

            # Define left and right interface detunings
            delta_interface_left = delta_bulk + alpha_left
            delta_interface_right = delta_bulk + alpha_right

            # Linear interpolation from edge → interface
            left_ramp = np.linspace(delta_boundary, delta_interface_left, boundary_size)
            right_ramp = np.linspace(delta_interface_right, delta_boundary, boundary_size)

            delta[:boundary_size] = left_ramp
            delta[-boundary_size:] = right_ramp

        else:
            raise ValueError("boundary_type must be one of: 'uniform', 'linear', or None.")

    # -------------------------------------
    # Construct MPO tensors
    # -------------------------------------
    H = []
    for i in range(N):
        W = np.zeros((D, d, D, d))
        W[0, :, 0, :] = I2
        W[1, :, 1, :] = I2
        W[1, :, 0, :] = (Omega / 2) * sigma_x - delta[i] * n_op
        H.append(W)

    # Adjust MPO edges (open boundary conditions)
    H[0] = H[0][-1:, :, :, :]
    H[-1] = H[-1][:, :, 0:1, :]

    return H, delta  # return detuning profile for inspection

def num_operator_MPO(N):
    ## Function to build the number operator MPO

    # Define local operators
    d = 2
    D = 2
    I2 = np.eye(2)
    n_op = np.array([[0.,0.],[0.,1.]])

    # Build the MPO
    H = []
    for site in range(N):
        W = np.zeros((D, d, D, d))
        W[0, :, 0, :] = I2          # propagate identity
        W[1, :, 1, :] = I2          # propagate identity along bottom-right
        W[1, :, 0, :] = n_op  # local on-site term
        H.append(W)

    # Edge tensors
    Hl = W
    H[0] = Hl[-1:np.shape(Hl)[0],:,:,:]
    H[N-1] = Hl[:,:,0:1,:]

    return H

def add_mpos_clean(H1, H2):
    """
    Sum two MPOs (list of length N).
    Preserves correct boundary dimensions:
      left edge (1,d,D,d), bulk (D,d,D,d), right edge (D,d,1,d).
    """
    assert len(H1) == len(H2)
    N = len(H1)
    Hsum = []

    for i in range(N):
        D1l, d1a, D1r, d1b = H1[i].shape
        D2l, d2a, D2r, d2b = H2[i].shape
        assert d1a == d2a == d1b == d2b
        d = d1a

        # left edge
        if i == 0:
            W = np.zeros((1, d, D1r + D2r, d))
            W[0, :, :D1r, :] = H1[i][0, :, :, :]
            W[0, :, D1r:, :] = H2[i][0, :, :, :]
        # right edge
        elif i == N-1:
            W = np.zeros((D1l + D2l, d, 1, d))
            W[:D1l, :, 0, :] = H1[i][:, :, 0, :]
            W[D1l:, :, 0, :] = H2[i][:, :, 0, :]
        # bulk
        else:
            W = np.zeros((D1l + D2l, d, D1r + D2r, d))
            W[:D1l, :, :D1r, :] = H1[i]
            W[D1l:, :, D1r:, :] = H2[i]
        Hsum.append(W)

    return Hsum

def rydberg_full_mpo(N, Omega, delta, A_list, b_list, R_b):
    # Onsite MPO
    H_onsite = onsite_MPO(N, Omega, delta)

    # Interaction MPO
    W_tensor = single_site_interaction_MPO_clean(A_list, b_list, R_b)
    H_inter = build_interaction_chain_from_Wtensor_corrected(W_tensor, N)

    # Add them
    H_full = add_mpos_clean(H_onsite, H_inter)

    return H_fulle_MPO(N, Omega, delta)

    # Interaction MPO
    W_tensor = single_site_interaction_MPO_clean(A_list, b_list, R_b)
    H_inter = build_interaction_chain_from_Wtensor_corrected(W_tensor, N)

    # Add them
    H_full = add_mpos_clean(H_onsite, H_inter)

    return H_full

## Finite-system DMRG implementation + helper function(s)
# -----------------------------
# DMRG function with warm start
# -----------------------------
def fDMRG_1site_GS_OBC_warm_start(H, D, Nsweeps, clip_val=1e10, svd_cutoff=1e-12, M_init=None):
    """
    1-site DMRG for a chain with open boundary conditions.
    H : list of MPOs for each site
    D : bond dimension
    Nsweeps : number of DMRG sweeps
    clip_val : maximum absolute value to clip tensors (avoid NaN/Inf)
    svd_cutoff : cutoff for singular values to avoid small/NaN propagation
    M_init : optional initial MPS (list of tensors). If None, random MPS is used.

    Returns:
        E_list : list of energies during sweeps
        M : final optimized MPS
        S_mid_final : final middle-cut entanglement entropy
    """
    N = len(H)
    middle_cut = N // 2   # measure entropy at this cut

    # -----------------------
    # Initialization of MPS
    # -----------------------
    if M_init is None:
        # Random MPS
        M = []
        M.append(np.random.rand(1, np.shape(H[0])[3], D))
        for l in range(1, N-1):
            M.append(np.random.rand(D, np.shape(H[l])[3], D))
        M.append(np.random.rand(D, np.shape(H[N-1])[3], 1))
    else:
        # Use provided initial guess (deep copy to avoid overwriting)
        M = [np.copy(tensor) for tensor in M_init]

    # Canonicalize
    M = LeftCanonical(M)
    M = RightCanonical(M)

    # Initialize Hzip
    Hzip = [np.ones((1,1,1)) for _ in range(N+2)]
    for l in range(N-1, -1, -1):
        Hzip[l+1] = ZipperRight(Hzip[l+2], M[l].conj().T, H[l], M[l])
        Hzip[l+1] = np.clip(Hzip[l+1], -clip_val, clip_val)

    E_list = []
    S_mid_final = None

    for sweep in range(Nsweeps):
        # Right sweep
        for l in range(N):
            Taux = np.einsum('ijk,jlmn', Hzip[l], H[l])
            Taux = np.einsum('ijklm,nlo', Taux, Hzip[l+2])
            Taux = np.transpose(Taux, (0,2,5,1,3,4))
            Hmat = np.reshape(Taux, (Taux.shape[0]*Taux.shape[1]*Taux.shape[2],
                                      Taux.shape[3]*Taux.shape[4]*Taux.shape[5]))
            Hmat = np.nan_to_num(Hmat, nan=0.0, posinf=clip_val, neginf=-clip_val)
            scale = np.max(np.abs(Hmat))
            if scale > 0:
                Hmat /= scale

            if Hmat.shape[0] < 3000:
                val, vec = np.linalg.eigh(Hmat)
                val, vec = val[0:1], vec[:, 0:1]
            else:
                try:
                    val, vec = eigsh(Hmat, k=1, which='SA', maxiter=50000)
                except:
                    val, vec = np.linalg.eigh(Hmat)
                    val, vec = val[0:1], vec[:, 0:1]
            val *= scale
            E_list.append(val[0])

            # Update MPS tensor with truncated SVD
            Taux2 = vec.reshape(Taux.shape[0]*Taux.shape[1], Taux.shape[2])
            U, S, Vh = scipy.linalg.svd(Taux2, full_matrices=False)
            keep = S > svd_cutoff
            U, S, Vh = U[:, keep], S[keep], Vh[keep, :]
            M[l] = np.reshape(U, (Taux.shape[0], Taux.shape[1], len(S)))
            if l < N-1:
                M[l+1] = np.einsum('ij,jkl->ikl', np.diag(S) @ Vh, M[l+1])

            if l == middle_cut:
                S_mid_final = entanglement_entropy(S)

            Hzip[l+1] = ZipperLeft(Hzip[l], M[l].conj().T, H[l], M[l])
            Hzip[l+1] = np.clip(Hzip[l+1], -clip_val, clip_val)

        # Left sweep
        for l in range(N-1, -1, -1):
            Taux = np.einsum('ijk,jlmn', Hzip[l], H[l])
            Taux = np.einsum('ijklm,nlo', Taux, Hzip[l+2])
            Taux = np.transpose(Taux, (0,2,5,1,3,4))
            Hmat = np.reshape(Taux, (Taux.shape[0]*Taux.shape[1]*Taux.shape[2],
                                      Taux.shape[3]*Taux.shape[4]*Taux.shape[5]))
            Hmat = np.nan_to_num(Hmat, nan=0.0, posinf=clip_val, neginf=-clip_val)
            scale = np.max(np.abs(Hmat))
            if scale > 0:
                Hmat /= scale

            if Hmat.shape[0] < 3000:
                val, vec = np.linalg.eigh(Hmat)
                val, vec = val[0:1], vec[:,0:1]
            else:
                try:
                    val, vec = eigsh(Hmat, k=1, which='SA', maxiter=50000)
                except:
                    val, vec = np.linalg.eigh(Hmat)
                    val, vec = val[0:1], vec[:,0:1]
            val *= scale
            E_list.append(val[0])

            # Update MPS tensor
            Taux2 = vec.reshape(Taux.shape[0], Taux.shape[1]*Taux.shape[2])
            U, S, Vh = scipy.linalg.svd(Taux2, full_matrices=False)
            keep = S > svd_cutoff
            U, S, Vh = U[:, keep], S[keep], Vh[keep, :]
            M[l] = np.reshape(Vh, (Vh.shape[0], Taux.shape[1], Taux.shape[2]))
            if l > 0:
                M[l-1] = np.einsum('ijk,kl->ijl', M[l-1], U @ np.diag(S))

            if l == middle_cut:
                S_mid_final = entanglement_entropy(S)

            Hzip[l+1] = ZipperRight(Hzip[l+2], M[l].conj().T, H[l], M[l])
            Hzip[l+1] = np.clip(Hzip[l+1], -clip_val, clip_val)

    return E_list, M, S_mid_final

## Helper functions to run DMRG (final analysis)
def dmrg_sweep_n_edge(
    mps_initial,
    n_edge_list,
    delta_bulk,
    Rb_values,
    Omega,
    N,
    params,
    bond_dim=150,
    N_sweeps=10,
    boundary_type="uniform",
    delta_boundary=None,
    alpha_left=0.0,
    alpha_right=0.0,
):
    """
    Run DMRG sweeps for multiple n_edge values (softened boundary sizes)
    with flexible on-site detuning profiles (uniform or linear ramps).

    Parameters
    ----------
    mps_initial : MPS
        Initial guess for warm start.
    n_edge_list : list of ints
        List of n_edge values (number of boundary sites to modify).
    delta_bulk : float
        Bulk detuning.
    Rb_values : array-like
        List/array of Rb values to sweep over.
    Omega : float
        Rabi frequency.
    N : int
        Total chain length.
    params : dict
        Dictionary containing exponential expansion parameters { (n_exp, N): {"A":..., "b":...} }.
    bond_dim : int, optional
        Bond dimension for DMRG (default: 150).
    N_sweeps : int, optional
        Number of DMRG sweeps (default: 10).
    boundary_type : str, optional
        'uniform' or 'linear' for boundary detuning (default 'uniform').
    delta_boundary : float or None, optional
        Detuning at the edges. For 'uniform', this is the constant boundary detuning.
        For 'linear', this is the detuning at the first and last sites of the chain.
        If None, defaults to delta_bulk (no difference from bulk).
    alpha_left : float, optional
        Detuning mismatch (relative to delta_bulk) at the left boundary–bulk interface.
        interface detuning = delta_bulk + alpha_left.
    alpha_right : float, optional
        Detuning mismatch (relative to delta_bulk) at the right boundary–bulk interface.
        interface detuning = delta_bulk + alpha_right.

    Returns
    -------
    dmrg_results_by_n_edge : dict
        Results for each n_edge value.
    mps_storage_by_n_edge : dict
        MPS storage for each n_edge value.
    entropy_storage_by_n_edge : dict
        Entropy storage for each n_edge value.
    """
    dmrg_results_by_n_edge = {}
    mps_storage_by_n_edge = {}
    entropy_storage_by_n_edge = {}

    for n_edge in n_edge_list:
        print(f"\n==== Running sweep for n_edge = {n_edge} ====\n")

        dmrg_results_fixed_nedge = []
        mps_storage_fixed_nedge = {}
        entropy_storage_fixed_nedge = {}

        # Initialize warm-start MPS
        mps_prev = mps_initial

        for i, Rb_value in enumerate(Rb_values):
            # Select exponential expansion (based on Rb)
            if Rb_value <= 2.4:
                A_list, b_list = params[(20, N)]['A'], params[(20, N)]['b']
                n_exp = 20
            else:
                A_list, b_list = params[(24, N)]['A'], params[(24, N)]['b']
                n_exp = 24

            # Interaction MPO
            W_int = single_site_interaction_MPO_clean(A_list, b_list, Rb_value)
            H_int = build_interaction_chain_from_Wtensor_corrected(W_int, N)

            # Default delta_boundary = delta_bulk (if not specified)
            delta_boundary_use = delta_bulk if delta_boundary is None else delta_boundary

            # Onsite MPO (now consistent with new onsite_MPO)
            H_onsite, delta_profile = onsite_MPO(
                N,
                Omega,
                delta_bulk,
                boundary_size=n_edge,
                boundary_type=boundary_type,
                delta_boundary=delta_boundary_use,
                alpha_left=alpha_left,
                alpha_right=alpha_right
            )

            # Combine onsite and interaction MPOs
            H_full = add_mpos_clean(H_onsite, H_int)

            # Run 1-site DMRG with warm start
            energy_trace, mps_final, entropy_final = fDMRG_1site_GS_OBC_warm_start(
                H_full, bond_dim, N_sweeps, M_init=mps_prev
            )

            # Update warm-start MPS for next Rb value
            mps_prev = mps_final

            # Compute excitation observables
            site_excitations = get_per_site_excitation_densities(mps_final, N)
            num_MPO = num_operator_MPO(N)
            exc_prob = get_excitation_probability(mps_final, num_MPO, N)

            # Compute boundary excitations (useful diagnostics)
            if n_edge > 0:
                left_boundary_edge_exc = site_excitations[n_edge - 1]
                right_boundary_edge_exc = site_excitations[N - n_edge - 1]
                left_far_end_exc = site_excitations[0]
                right_far_end_exc = site_excitations[-1]
            else:
                left_boundary_edge_exc = right_boundary_edge_exc = None
                left_far_end_exc = right_far_end_exc = None

            # Store results for this Rb
            dmrg_results_fixed_nedge.append({
                "Rb": Rb_value,
                "delta_bulk": delta_bulk,
                "n_edge": n_edge,
                "boundary_type": boundary_type,
                "delta_boundary": delta_boundary_use,
                "alpha_left": alpha_left,
                "alpha_right": alpha_right,
                "energy": energy_trace[-1],
                "excitation_prob": exc_prob,
                "S_final": entropy_final,
                "bond_dim": bond_dim,
                "n_exp": n_exp,
                "N_sweeps": N_sweeps,
                "delta_profile": delta_profile,
            })

            # Store MPS and entropy
            mps_storage_fixed_nedge[Rb_value] = mps_final
            entropy_storage_fixed_nedge[Rb_value] = entropy_final

            # Print intermediate diagnostic info
            print(
                f"[{i+1:03d}/{len(Rb_values)}] R_b = {Rb_value:.3f}, "
                f"δ_bulk = {delta_bulk:.3f}, E0 = {energy_trace[-1]:.6f}, "
                f"ρ = {exc_prob:.6f}, S_final = {entropy_final:.6f}, "
                f"χ = {bond_dim}, n_exp = {n_exp}, sweeps = {N_sweeps}, "
                f"Left edge: {left_boundary_edge_exc}, Right edge: {right_boundary_edge_exc}, "
                f"Left far: {left_far_end_exc}, Right far: {right_far_end_exc}"
            )

        # Aggregate per n_edge
        dmrg_results_by_n_edge[n_edge] = dmrg_results_fixed_nedge
        mps_storage_by_n_edge[n_edge] = mps_storage_fixed_nedge
        entropy_storage_by_n_edge[n_edge] = entropy_storage_fixed_nedge

    return dmrg_results_by_n_edge, mps_storage_by_n_edge, entropy_storage_by_n_edge

def dmrg_sweep_multiple_N(
    N_values,
    n_edge_dict,
    mps_initial_dict,
    delta_bulk,
    Rb_values,
    Rb_start,
    Omega,
    params,
    bond_dim=150,
    N_sweeps=10,
    boundary_type="uniform",
    delta_boundary=None,
    alpha_left=0.0,
    alpha_right=0.0,
):
    """
    Run DMRG sweeps for multiple chain lengths N, each with its own n_edge list
    and initial MPS dictionary, keeping results, MPSs, and entropies separate.

    Parameters
    ----------
    N_values : list of ints
        Chain lengths (e.g. [97, 109, 121]).
    n_edge_dict : dict
        Mapping {N: [n_edge_1, n_edge_2, ...]}.
    mps_initial_dict : dict
        Mapping {N: mps_dict}, where mps_dict[(Rb_init, delta_bulk)] gives the warm-start MPS.
    delta_bulk : float
        Bulk detuning.
    Rb_values : array-like
        List of Rb values to sweep over.
    Omega : float
        Rabi frequency.
    params : dict
        Exponential expansion parameters {(n_exp, N): {"A":..., "b":...}}.
    bond_dim : int, optional
        Bond dimension for DMRG (default 150).
    N_sweeps : int, optional
        Number of DMRG sweeps (default 10).
    boundary_type : str, optional
        'uniform' or 'linear' (default 'uniform').
    delta_boundary : float or None, optional
        Edge detuning (defaults to delta_bulk if None).
    alpha_left, alpha_right : float, optional
        Interface detuning mismatches.

    Returns
    -------
    dmrg_results_all : dict
        {N: {n_edge: [list of result dicts]}}
    mps_storage_all : dict
        {N: {n_edge: {Rb: MPS}}}
    entropy_storage_all : dict
        {N: {n_edge: {Rb: float}}}
    """
    dmrg_results_all = {}
    mps_storage_all = {}
    entropy_storage_all = {}

    for N in N_values:
        print(f"\n###############################")
        print(f"### Running DMRG for N = {N} ###")
        print(f"###############################")

        if N not in mps_initial_dict:
            raise ValueError(f"No initial MPS dictionary provided for N = {N}")
        mps_dict = mps_initial_dict[N]

        dmrg_results_all[N] = {}
        mps_storage_all[N] = {}
        entropy_storage_all[N] = {}

        for n_edge in n_edge_dict[N]:
            print(f"\n==== Running sweep for N = {N}, n_edge = {n_edge} ====\n")

            dmrg_results_fixed_nedge = []
            mps_storage_fixed_nedge = {}
            entropy_storage_fixed_nedge = {}

            # Initial MPS for warm-start
            Rb_init = Rb_start
            mps_prev = mps_dict[(Rb_init, delta_bulk)]

            for i, Rb_value in enumerate(Rb_values):
                # Choose exponential expansion
                if Rb_value <= 2.4:
                    A_list, b_list = params[(20, N)]['A'], params[(20, N)]['b']
                    n_exp = 20
                else:
                    A_list, b_list = params[(24, N)]['A'], params[(24, N)]['b']
                    n_exp = 24

                # Interaction MPO
                W_int = single_site_interaction_MPO_clean(A_list, b_list, Rb_value)
                H_int = build_interaction_chain_from_Wtensor_corrected(W_int, N)

                # Onsite detuning MPO
                delta_boundary_use = delta_bulk if delta_boundary is None else delta_boundary
                H_onsite, delta_profile = onsite_MPO(
                    N,
                    Omega,
                    delta_bulk,
                    boundary_size=n_edge,
                    boundary_type=boundary_type,
                    delta_boundary=delta_boundary_use,
                    alpha_left=alpha_left,
                    alpha_right=alpha_right
                )

                # Full Hamiltonian
                H_full = add_mpos_clean(H_onsite, H_int)

                # Run 1-site DMRG with warm start
                energy_trace, mps_final, entropy_final = fDMRG_1site_GS_OBC_warm_start(
                    H_full, bond_dim, N_sweeps, M_init=mps_prev
                )

                # Update warm start
                mps_prev = mps_final

                # Observables
                site_excitations = get_per_site_excitation_densities(mps_final, N)
                num_MPO = num_operator_MPO(N)
                exc_prob = get_excitation_probability(mps_final, num_MPO, N)

                if n_edge > 0:
                    left_boundary_edge_exc = site_excitations[n_edge - 1]
                    right_boundary_edge_exc = site_excitations[N - n_edge - 1]
                    left_far_end_exc = site_excitations[0]
                    right_far_end_exc = site_excitations[-1]
                else:
                    left_boundary_edge_exc = right_boundary_edge_exc = None
                    left_far_end_exc = right_far_end_exc = None

                # Store results
                dmrg_results_fixed_nedge.append({
                    "Rb": Rb_value,
                    "delta_bulk": delta_bulk,
                    "n_edge": n_edge,
                    "boundary_type": boundary_type,
                    "delta_boundary": delta_boundary_use,
                    "alpha_left": alpha_left,
                    "alpha_right": alpha_right,
                    "energy": energy_trace[-1],
                    "excitation_prob": exc_prob,
                    "S_final": entropy_final,
                    "bond_dim": bond_dim,
                    "n_exp": n_exp,
                    "N_sweeps": N_sweeps,
                    "delta_profile": delta_profile,
                })

                mps_storage_fixed_nedge[Rb_value] = mps_final
                entropy_storage_fixed_nedge[Rb_value] = entropy_final

                print(
                    f"[{i+1:03d}/{len(Rb_values)}] N={N}, n_edge={n_edge}, R_b={Rb_value:.3f}, "
                    f"δ_bulk={delta_bulk:.3f}, E0={energy_trace[-1]:.6f}, "
                    f"ρ={exc_prob:.6f}, S_final={entropy_final:.6f}, χ={bond_dim}, "
                    f"n_exp={n_exp}, sweeps={N_sweeps}, "
                    f"Left edge={left_boundary_edge_exc}, Right edge={right_boundary_edge_exc}, "
                    f"Left far={left_far_end_exc}, Right far={right_far_end_exc}"
                )

            # Store for this n_edge
            dmrg_results_all[N][n_edge] = dmrg_results_fixed_nedge
            mps_storage_all[N][n_edge] = mps_storage_fixed_nedge
            entropy_storage_all[N][n_edge] = entropy_storage_fixed_nedge

    return dmrg_results_all, mps_storage_all, entropy_storage_all