## Importing relevant libraries
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from functools import reduce
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
from scipy.special import jv
from scipy.integrate import quad
from pathlib import Path

# Import
from matplotlib.colors import LogNorm

# Libaries for GL fitting procedure
import math
from numpy.polynomial.laguerre import laggauss
from scipy.optimize import least_squares, minimize
from scipy.optimize import curve_fit

## Note: The benchmarking functions below can be easily modified by the user to incorporate boundary detuning ramps (for testing purposes).

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

## GL-fitting procedure -- helper functions
# -----------------------------
# Utility: build GL-based init
# -----------------------------
def gauss_laguerre_init(K):
    """
    Returns initial positive A (amplitudes) and b (decays) from Gauss-Laguerre quadrature for 1/r^6.
    """
    t, w = laggauss(K)   # nodes and weights for weight e^{-t}
    # From derivation: b_i = t_i, A_i = (w_i * t_i^5 * exp (t_i)) / 5!
    A = (w * (t**5) * np.exp(t)) / math.factorial(5)
    b = t.copy()
    return A, b

# -----------------------------
# Model and residuals
# -----------------------------
def exp_sum(A, b, r):
    # A,b: arrays length K; r: (M,) array
    return np.sum((A[:,None] * np.exp(-b[:,None] * r[None,:])), axis=0)

def residuals_logparam(x, r, target):
    """
    x contains [alpha_0..alpha_{K-1}, beta_0..beta_{K-1}] where
    A_k = exp(alpha_k), b_k = exp(beta_k)
    """
    K = len(x)//2
    alpha = np.exp(x[:K])
    beta = np.exp(x[K:])
    pred = exp_sum(alpha, beta, r)
    target_max = target.max()
    return pred - target

# -----------------------------
# Positive exponential fitting
# -----------------------------
def fit_positive_exp_sum(K, r_fit, refine=True):
    target = 1.0 / (r_fit**6)
    A0, b0 = gauss_laguerre_init(K)
    alpha0 = np.log(A0)
    beta0 = np.log(b0)
    x0 = np.concatenate([alpha0, beta0])

    if not refine:
        return np.exp(alpha0), np.exp(beta0)

    fun = lambda x: residuals_logparam(x, r_fit, target)
    res = least_squares(fun, x0, method='trf', xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=10_000)
    Khalf = len(res.x)//2

    return np.exp(res.x[:Khalf]), np.exp(res.x[Khalf:]), res

def fit_positive_exp_sum_bfgs(K, r_fit, refine=True):
    target = 1.0 / (r_fit**6)
    A0, b0 = gauss_laguerre_init(K)
    alpha0 = np.log(A0)
    beta0  = np.log(b0)
    x0 = np.concatenate([alpha0, beta0])

    if not refine:
        return np.exp(alpha0), np.exp(beta0)

    # Scalar objective: sum of squared residuals
    def obj_fun(x):
        res = residuals_logparam(x, r_fit, target)
        return np.sum(res**2)

    res = minimize(obj_fun, x0, method='L-BFGS-B',
                   options={'gtol':1e-12, 'ftol':1e-12, 'maxiter':10_000, 'disp': True})

    Khalf = len(res.x)//2
    A_opt = np.exp(res.x[:Khalf])
    b_opt = np.exp(res.x[Khalf:])

    return A_opt, b_opt, res

def benchmark_errors_optimizer(K_list, Rmax_list, optimizer="ls"):
    """
    K_list: list of number of exponentials
    Rmax_list: list of maximum r values
    optimizer: "ls" for least_squares, "bfgs" for BFGS

    Returns:
        max_err_matrix: (len(K_list), len(Rmax_list)) array of max abs errors
        sum_err_matrix: (len(K_list), len(Rmax_list)) array of sum abs errors
        K_list: array of K values
        Rmax_list: array of Rmax values
        fit_params: dict keyed by (K, Rmax) with values {'A': A_opt, 'b': b_opt}
    """
    K_list = np.array(K_list)
    Rmax_list = np.array(Rmax_list)
    max_err_matrix = np.zeros((len(K_list), len(Rmax_list)))
    sum_err_matrix = np.zeros((len(K_list), len(Rmax_list)))

    # Dictionary to store fit parameters
    fit_params = {}

    for iK, K in enumerate(K_list):
        for iR, Rmax in enumerate(Rmax_list):
            print(f"\nValue of K: {K}")
            print(f"Value of Rmax: {Rmax}")

            r_fit = np.arange(1, Rmax)      # fit up to Rmax-1
            r_eval = np.linspace(1, Rmax+1)
            exact_eval = 1.0 / (r_eval**6)

            # Select optimizer
            if optimizer.lower() == "ls":
                A_opt, b_opt, _ = fit_positive_exp_sum(K, r_fit, refine=True)
            elif optimizer.lower() == "bfgs":
                A_opt, b_opt, _ = fit_positive_exp_sum_bfgs(K, r_fit)
            else:
                raise ValueError("optimizer must be 'ls' or 'bfgs'")

            approx_eval = exp_sum(A_opt, b_opt, r_eval)
            abs_err = np.abs(approx_eval - exact_eval)
            max_err_matrix[iK, iR] = abs_err.max()
            sum_err_matrix[iK, iR] = abs_err.sum()

            print(f"Max. abs. error: {max_err_matrix[iK, iR]:.3e}")
            print(f"Sum. abs. error: {sum_err_matrix[iK, iR]:.3e}")

            # Store fit parameters
            fit_params[(K, Rmax)] = {'A': A_opt, 'b': b_opt}

    return max_err_matrix, sum_err_matrix, K_list, Rmax_list, fit_params

## Building the interaction term MPO and benchmarking wrt exact interaction-only energies -- helper functions
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
        W[base + 2, :, base + 1, :] = (A_k * x_k * (R_b**6)) * n_op
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

# -----------------------
# 3. Build product-state MPS
# -----------------------
def product_mps_from_state(state_list):
    basis = {'g': np.array([1., 0.], dtype=float),
             'r': np.array([0., 1.], dtype=float)}
    return [basis[s].reshape(1, 2, 1) for s in state_list]

# -----------------------
# 4. Contract MPO with MPS (scalar output)
# -----------------------
def contract_product_mpo_mps(mpo, mps):
    env = np.array([[1.0 + 0j]])
    for A, W in zip(mps, mpo):
        temp = np.einsum('a i b, L i R j, a j b -> L R', A.conj(), W, A)
        env = np.tensordot(env, temp, axes=(1, 0))
    return float(np.squeeze(env).real)

# -----------------------
# 5. Full workflow
# -----------------------
def rydberg_interaction_mpo_workflow(N, state_list, A_list, b_list, R_b):
    """
    Build full Rydberg interaction MPO and compute expectation value for a product state.
    """
    # 1. Build numeric single-site MPO
    W_tensor = single_site_interaction_MPO_clean(A_list, b_list, R_b)

    # 2. Build chain MPO with OBC
    mpo_chain = build_interaction_chain_from_Wtensor_corrected(W_tensor, N)

    # 3. Build product-state MPS
    mps = product_mps_from_state(state_list)

    # 4. Compute expectation value
    E_mpo = contract_product_mpo_mps(mpo_chain, mps)

    return mpo_chain, mps, E_mpo

# -----------------------
# 6) direct pairwise evaluator for comparison
# -----------------------
def V_ij_from_fit(i, j, A_list, b_list, R_b):
    r = abs(i - j)
    return np.sum(A_list * (R_b**6) * np.exp(-b_list * r))

def direct_interaction_energy(state_list, A_list, b_list, R_b):
    n = [1 if s == 'r' else 0 for s in state_list]
    N = len(n)
    E = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if n[i] and n[j]:
                E += V_ij_from_fit(i, j, A_list, b_list, R_b)
    return E

# -----------------------
# Direct pairwise evaluation using the true Rydberg interaction
# -----------------------
def true_V_ij(i, j, R_b=1.0):
    """
    True van der Waals interaction between Rydberg excitations.
    C6: interaction strength (can be adjusted)
    """
    r = abs(i - j)
    return R_b**6 / r**6

def direct_interaction_energy_true(state_list, R_b=1.0):
    """
    Compute the total interaction energy for a given spin configuration
    using the exact van der Waals form.
    """
    n = [1 if s == 'r' else 0 for s in state_list]  # 1 for |r>, 0 for |g>
    N = len(n)
    E = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if n[i] and n[j]:
                E += true_V_ij(i, j, R_b)
    return E

## Construction of the full Rydberg Hamiltonian MPO and benchmarking wrt exact energies (all Hamiltonian terms included) -- helper functions
# Build full MPO for the Rydberg Hamiltonian
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

    return H_full

# Benchmarking functions
## Define local operators
# ---------- helpers: dense operators ----------
I2 = np.eye(2, dtype=complex)
sx = np.array([[0,1],[1,0]], dtype=complex)
n_op = np.array([[0,0],[0,1]], dtype=complex)

def kronN(ops):
    out = np.array([[1]], dtype=complex)
    for op in ops:
        out = np.kron(out, op)

    return out

# ---------- direct energies on product states ----------
def direct_interaction_energy_product(state_list, A_list, b_list, R_b, include_x_in_weight=True):
    """
    E = sum_k A_k * (R_b**6) * [x_k^include?] * sum_{i<j} n_i n_j e^{-b_k |i-j|}
    include_x_in_weight=True matches your current W[base+2, base+1] = A_k * x_k * R_b^6 * n
    """
    n = np.array([1 if s=='r' else 0 for s in state_list], dtype=float)
    E = 0.0
    for k, (A,b) in enumerate(zip(A_list, b_list)):
        x = np.exp(-b)
        pref = A * (R_b**6) * (x if include_x_in_weight else 1.0)
        # sum_{i<j} e^{-b |i-j|} n_i n_j
        total = 0.0
        for i in range(len(n)):
            if n[i]==0: continue
            for j in range(i+1, len(n)):
                if n[j]==0: continue
                total += np.exp(-b * (j-i))
        E += pref * total

    return E

def direct_onsite_energy_product(state_list, Omega, delta):
    """
    <prod| sum_i (Omega/2 * sigma_x - delta * n) |prod>
    In the computational basis, <sigma_x>=0 for |g> and |r>, so only detuning contributes.
    """
    n = sum(1 for s in state_list if s=='r')

    return -delta * n

# ---------- dense Hamiltonian (small N) ----------
def dense_full_hamiltonian(N, Omega, delta, A_list, b_list, R_b, include_x_in_weight=True):
    # on-site
    H = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        ops_x = [I2]*N
        ops_n = [I2]*N
        ops_x[i] = sx
        ops_n[i] = n_op
        H += (Omega/2) * kronN(ops_x)
        H += (-delta)   * kronN(ops_n)

    # interaction: sum_{i<j} sum_k pref * e^{-b |i-j|} n_i n_j
    for k, (A,b) in enumerate(zip(A_list, b_list)):
        x = np.exp(-b)
        pref = A * (R_b**6) * (x if include_x_in_weight else 1.0)
        for i in range(N):
            for j in range(i+1, N):
                ops_i = [I2]*N
                ops_j = [I2]*N
                ops_i[i] = n_op
                ops_j[j] = n_op
                H += pref * np.exp(-b*(j-i)) * kronN(ops_i) @ kronN(ops_j)

    return H

# ---------- MPO → dense (generic, OBC) ----------
def mpo_to_dense(mpo):
    """Convert MPO (list of tensors Dl,d,Dr,d) to a dense 2^N x 2^N matrix. For small N."""
    N = len(mpo)
    # Initialize as (Dl=1, phys_in=2, Dr=?, phys_out=2)
    T = mpo[0]  # (1,2,D1,2)
    # Contract site by site
    for l in range(1, N):
        # einsum over right bond of T with left bond of next
        T = np.einsum('L i R j, R k S l -> L i k S j l', T, mpo[l])
        # Merge physical legs in/out across sites to keep MPO structure tidy:
        L, i1, i2, S, j1, j2 = T.shape
        T = T.reshape(L, i1*i2, S, j1*j2)  # still a rank-4 MPO-like tensor
    # At the end, Dl=1 and Dr=1; reshape to matrix
    assert T.shape[0] == 1 and T.shape[2] == 1
    d_in, d_out = T.shape[1], T.shape[3]

    return T.reshape(d_in, d_out)

# ---------- benchmark runners ----------
def bench_interaction_product_states(N, A_list, b_list, R_b,
                                     include_x_in_weight=True,
                                     n_trials=20, rng=0):
    """
    Random product states: compare MPO energy vs direct interaction energy.
    Uses the improved interaction chain MPO and onsite MPO functions.
    """
    rs = np.random.RandomState(rng)

    # Build improved interaction MPO
    W = single_site_interaction_MPO_clean(A_list, b_list, R_b)
    H_inter = build_interaction_chain_from_Wtensor_corrected(W, N)

    max_abs_err = 0.0
    max_rel_err = 0.0

    for _ in range(n_trials):
        # random product in {g,r}
        state = rs.choice(['g','r'], size=N, p=[0.7,0.3]).tolist()
        mps = product_mps_from_state(state)

        # MPO contraction
        E_mpo = contract_product_mpo_mps(H_inter, mps)

        # Ground-truth direct energy
        E_dir = direct_interaction_energy_product(
            state, A_list, b_list, R_b,
            include_x_in_weight=include_x_in_weight
        )

        # Error metrics
        abs_err = abs(E_mpo - E_dir)
        rel_err = abs_err / (abs(E_dir) + 1e-15)
        max_abs_err = max(max_abs_err, abs_err)
        max_rel_err = max(max_rel_err, rel_err)

    return max_abs_err, max_rel_err

def bench_full_vs_dense(N, Omega, delta, A_list, b_list, R_b,
                        onsite_MPO, build_W_tensor, build_chain,
                        include_x_in_weight=True):
    """
    Build full MPO (onsite + interaction) and compare its dense matrix
    to a directly-built dense Hamiltonian. (Small N recommended.)
    """
    # MPO build
    H_onsite = onsite_MPO(N, Omega, delta)
    W = build_W_tensor(A_list, b_list, R_b)
    H_inter = build_chain(W, N)
    H_full = add_mpos_clean(H_onsite, H_inter)
    H_mpo_dense = mpo_to_dense(H_full)

    # Direct dense
    H_dense = dense_full_hamiltonian(N, Omega, delta, A_list, b_list, R_b,
                                     include_x_in_weight=include_x_in_weight)

    # Matrix error metrics
    diff = H_mpo_dense - H_dense
    frob_rel = np.linalg.norm(diff) / (np.linalg.norm(H_dense) + 1e-15)
    max_abs = np.max(np.abs(diff))

    return frob_rel, max_abs

def direct_full_energy_product_from_fit(state, A_list, b_list, R_b, Omega, delta):
    # number operators in product states
    n = [1 if s == 'r' else 0 for s in state]

    # onsite terms: only detuning contributes
    E_detune = -delta * sum(n)
    # (rabi contributes 0 for computational basis states)

    # interactions
    E_int = 0.0
    for k, A_k in enumerate(A_list):
        for i in range(len(n)):
            if n[i] == 0: continue
            for j in range(i+1, len(n)):
                if n[j] == 0: continue
                dist = abs(i-j)
                E_int += A_k * (R_b**6) * np.exp(-b_list[k] * dist)

    return E_detune + E_int

def direct_full_energy_product_from_true_potential(state, R_b, Omega, delta):
    # number operators in product states
    n = [1 if s == 'r' else 0 for s in state]

    # onsite terms: only detuning contributes
    E_detune = -delta * sum(n)
    # (rabi contributes 0 for computational basis states)

    # Interactions (true form of the potential)
    N = len(n)
    E_int = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if n[i] and n[j]:
                E_int += true_V_ij(i, j, R_b)

    return E_detune + E_int

def mpo_full_energy_product(state, A_list, b_list, R_b, Omega, delta):
    N = len(state)
    W_int = single_site_interaction_MPO_clean(A_list, b_list, R_b)
    H_int = build_interaction_chain_from_Wtensor_corrected(W_int, N)
    H_onsite = onsite_MPO(N, Omega, delta)
    H_full = add_mpos_clean(H_onsite, H_int)
    mps = product_mps_from_state(state)

    return contract_product_mpo_mps(H_full, mps)

def bench_full_chain(N, A_list, b_list, R_b, Omega, delta,
                     n_trials=20, rng=0):

    rs = np.random.RandomState(rng)
    max_abs_err, max_rel_err = 0.0, 0.0
    for _ in range(n_trials):
        # random computational basis product state
        state = rs.choice(['g','r'], size=N, p=[0.7,0.3]).tolist()
        E_mpo = mpo_full_energy_product(state, A_list, b_list, R_b, Omega, delta)
        E_dir = direct_full_energy_product_from_fit(state, A_list, b_list, R_b, Omega, delta)
        abs_err = abs(E_mpo - E_dir)
        rel_err = abs_err / (abs(E_dir) + 1e-15)
        max_abs_err = max(max_abs_err, abs_err)
        max_rel_err = max(max_rel_err, rel_err)
        print(f"state={state}, E_dir={E_dir:.6e}, E_mpo={E_mpo:.6e}, abs_err={abs_err:.2e}")

    return max_abs_err, max_rel_err

def bench_param_sweep(state, A_list, b_list, R_b_values, delta_values, Omega=1.0):
    """
    Benchmark MPO vs exact interaction for a fixed spin configuration.

    Parameters
    ----------
    state : list[str]
        Spin configuration, e.g. ['r','g','r','g',...]
    A_list, b_list : arrays
        Parameters for exponential fit (if used)
    R_b_values : array-like
        Values of R_b to sweep over
    delta_values : array-like
        Values of delta to sweep over
    Omega : float
        Rabi frequency (default 1.0)

    Returns
    -------
    results : dict
        Dictionary with keys ('E_dir', 'E_mpo', 'abs_err') each
        containing arrays of shape (len(R_b_values), len(delta_values)).
    """
    NR, ND = len(R_b_values), len(delta_values)
    E_dir = np.zeros((NR, ND))
    E_mpo = np.zeros((NR, ND))
    abs_err = np.zeros((NR, ND))

    for i, Rb in enumerate(R_b_values):
        for j, delta in enumerate(delta_values):
            E_d = direct_full_energy_product_from_true_potential(state, Rb, Omega, delta)
            E_m = mpo_full_energy_product(state, A_list, b_list, Rb, Omega, delta)
            E_dir[i, j] = E_d
            E_mpo[i, j] = E_m
            abs_err[i, j] = abs(E_m - E_d)
            # print(f"R_b={Rb}, delta={delta} | "
            #       f"E_dir={E_d}, E_mpo={E_m}, "
            #       f"abs_err={abs_err}")

    return {"E_dir": E_dir, "E_mpo": E_mpo, "abs_err": abs_err}

## Some visualization scripts
def plot_abs_error_heatmap(results_dict, R_b_values, delta_bulk_values,
                            delta_edge, cmap="inferno", savepath=None):
    """
    Plot absolute error heatmaps for softened-edge detuning sweeps.

    Parameters
    ----------
    results_dict : dict
        Keys are delta_bulk values; values are dicts containing "abs_err" arrays.
    R_b_values : array-like
        R_b sweep values.
    delta_bulk_values : array-like
        Bulk detuning values.
    delta_edge : float
        Fixed edge detuning value.
    cmap : str
        Colormap name (default "inferno").
    savepath : str or None
        Filepath to save the plot (default None = just show).
    """

    abs_err_matrix = np.zeros((len(R_b_values), len(delta_bulk_values)))

    # Collect abs_err data
    for j, delta_bulk in enumerate(delta_bulk_values):
        abs_err_matrix[:, j] = results_dict[delta_bulk]["abs_err"][:, 0]  # [:,0] because only one delta_array

    plt.figure(figsize=(8, 6), dpi=600)

    im = plt.imshow(abs_err_matrix,
                    extent=(delta_bulk_values[0], delta_bulk_values[-1],
                            R_b_values[0], R_b_values[-1]),
                    origin='lower',
                    aspect='auto',
                    norm=LogNorm(vmin=abs_err_matrix.min()+1e-20,
                                 vmax=abs_err_matrix.max()),
                    cmap=cmap)

    cbar = plt.colorbar(im)
    cbar.set_label(r'$|E_{\mathrm{MPO}} - E_{\mathrm{exact}}|$', fontsize=17)
    cbar.ax.tick_params(labelsize=17)

    plt.xlabel(r'$\delta_{\mathrm{bulk}}$', fontsize=19)
    plt.ylabel(r'$R_b$', fontsize=19)
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.title(rf"$N={len(R_b_values)}$-site Rydberg chain "
              rf"($\delta_{{\mathrm{{edge}}}}={delta_edge}$)", y=1.05, fontsize=19)
    plt.tight_layout(pad=1.0)

    if savepath:
        plt.savefig(savepath, bbox_inches='tight', dpi=600)
    else:
        plt.show()