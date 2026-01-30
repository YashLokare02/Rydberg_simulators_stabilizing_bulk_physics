import numpy as np
import sys
from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes
from pyblock2.algebra.io import MPOTools, MPSTools
import random
import shutil
import os


# ============================================================
# TOGGLE 
# ============================================================
USE_UNIFORM_OMEGA = True # If True, \Omega is uniform everywhere
USE_STAR_MODULATION = False   # True → ramp + star pinning, False → pure ramp
EPS_STAR = 0.09               # ignored if USE_STAR_MODULATION=False


# ============================================================
# DMRG helper
# ============================================================
def run_dmrg(driver, mpo, maxD):
    ket = driver.load_mps(tag="KET", nroots=1)
    #ket = driver.adjust_mps(ket, dot=2)
    bond_dims = [15] * 4 + [25] * 2 + [maxD] * 4
    noises = [1e-4] * 4 + [1e-5] * 4 + [0]
    thrds = [1e-10] * 8

    # Define sweep number and direction
    #sweep_start = 13
    #forward = False

    # Run DMRG
    return driver.dmrg(
        mpo, ket,
        n_sweeps=100,
        tol=1e-8,
        bond_dims=bond_dims,
        noises=noises,
        thrds=thrds,
        cutoff=0,
        iprint=2
    ), ket


# ============================================================
# Snake mapping
# ============================================================
def idx_2d_to_1d(y, x, Lx):
    base_index = y * Lx
    return base_index + x if y % 2 == 0 else base_index + (Lx - 1 - x)


def idx_1d_to_2d(idx, Lx):
    y = idx // Lx
    pos = idx % Lx
    x = pos if y % 2 == 0 else (Lx - 1 - pos)
    return (y, x)


def distance_to_boundary(idx, Lx, Ly):
    y, x = idx_1d_to_2d(idx, Lx)
    return min(x, Lx - 1 - x, y, Ly - 1 - y)


# ============================================================
# Correct ramp: interface ring is EXACTLY delta - alpha
# ============================================================
def ramp_detuning(d, n_edge, delta_boundary, delta_interface):
    """
    d = 0 ... n_edge-1
    d = 0         → physical boundary
    d = n_edge-1  → interface ring touching bulk
    """
    if n_edge <= 1:
        return delta_interface
    t = d / (n_edge - 1)
    return delta_boundary + (delta_interface - delta_boundary) * t


# ============================================================
# Star mask (4×2 unit cell, unbiased)
# ============================================================
def build_star_mask_4x2(Lx, Ly):
    """
    Pattern:
      even y: 1 0 0 0 1 0 0 0 ...
      odd  y: 0 0 1 0 0 0 1 0 ...
    """
    mask = np.full((Ly, Lx), -1, dtype=int)
    for y in range(Ly):
        for x in range(Lx):
            if (y % 2 == 0 and x % 4 == 0) or (y % 2 == 1 and x % 4 == 2):
                mask[y, x] = +1
    return mask


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # ---------------- lattice ----------------
    Lx = 13
    Ly = 13
    Ntot = Lx * Ly

    # ---------------- input params ----------------
    delta = float(sys.argv[1])
    Rb = float(sys.argv[2])
    maxD = int(float(sys.argv[3]))
    alpha = float(sys.argv[4])
    n_edge = int(float(sys.argv[5]))
    delta_boundary = float(sys.argv[6])

    delta_interface = delta - alpha

    print(
        f"Lx={Lx}, Ly={Ly}, delta={delta}, Rb={Rb}, "
        f"alpha={alpha}, n_edge={n_edge}, delta_boundary={delta_boundary}, "
        f"USE_STAR_MODULATION={USE_STAR_MODULATION}, "
        f"USE_UNIFORM_OMEGA={USE_UNIFORM_OMEGA}"
    )

    # ---------------- scratch ----------------
    print(f"running Lx={Lx}, Ly={Ly}, delta={delta}, Rb={Rb}, alpha={alpha}, "
          f"n_edge={n_edge}, delta_boundary={delta_boundary}")

    id = random.randint(0, 2000000)
    print(f"scratch id={id}")

    # Loading contents from the previous scratch file into the current one 
    scratch_path = f"/oscar/scratch/ylokare/tmp{id}"
    previous_scratch = "/users/ylokare/2D_DMRG_calculations/data/tmp748855"
    os.makedirs(scratch_path, exist_ok=True)
    
    # Copy all MPS/MPO scratch files into the new scratch folder
    print(f"Copying previous MPS/MPO data from {previous_scratch} to {scratch_path}")
    shutil.copytree(previous_scratch, scratch_path, dirs_exist_ok=True)
    
    # ---------------- driver ----------------
    driver = DMRGDriver(
        scratch=scratch_path,
        symm_type=SymmetryTypes.SAny | SymmetryTypes.SGB,
        n_threads=1,
        n_mkl_threads=1
    )
    driver.set_symmetry_groups()
    Q = driver.bw.SX

    # ---------------- ops ----------------
    site_basis = [[(Q(), 2)]] * Ntot
    ops = {
        "": np.eye(2),
        "X": np.array([[0, 1], [1, 0]]),
        "N": np.array([[1, 0], [0, 0]]),
    }
    site_ops = [ops] * Ntot

    driver.initialize_system(n_sites=Ntot, vacuum=Q(), target=Q(), hamil_init=False)
    driver.ghamil = driver.get_custom_hamiltonian(site_basis, site_ops)

    # ---------------- MPO ----------------
    b = driver.expr_builder()

    # Apply the \sigma_x term
    if USE_UNIFORM_OMEGA:
        b.add_term("X", list(range(Ntot)), 0.5)
    else:
        for idx in range(Ntot):
            y, x = idx_1d_to_2d(idx, Lx)

            # (1,1) sublattice → Omega = 0
            if (x % 2 == 1) and (y % 2 == 1):
                continue   # no X term on this site

            # all other sites → Omega = 1
            b.add_term("X", [idx], 0.5)

    if USE_STAR_MODULATION:
        star_mask = build_star_mask_4x2(Lx, Ly)

    for idx in range(Ntot):
        y, x = idx_1d_to_2d(idx, Lx)
        d = distance_to_boundary(idx, Lx, Ly)

        if d < n_edge:
            base = ramp_detuning(d, n_edge, delta_boundary, delta_interface)
            if USE_STAR_MODULATION:
                delta_local = base + EPS_STAR * star_mask[y, x]
            else:
                delta_local = base
        else:
            delta_local = delta

        b.add_term("N", [idx], -delta_local)

    # ---------------- interactions ----------------
    for i in range(Ntot):
        yi, xi = idx_1d_to_2d(i, Lx)
        for j in range(i + 1, Ntot):
            yj, xj = idx_1d_to_2d(j, Lx)
            r = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            b.add_term("NN", [i, j], (Rb**6) / (r**6))

    mpo = driver.get_mpo(
        b.finalize(),
        algo_type=MPOAlgorithmTypes.SVD | MPOAlgorithmTypes.Fast,
        cutoff=1e-16,
        iprint=2
    )

    # ---------------- run ----------------
    energies, ket = run_dmrg(driver, mpo, maxD)
    print("DMRG energy =", energies)

    # ---------------- densities ----------------
    rdm1 = driver.get_npdm(ket, pdm_type=1, npdm_expr="N")

    # Print local excitation densities
    print("\nLocal densities:")
    for i in range(Lx*Ly):
        print(rdm1[0][i])
    print("\ncorresponding 2D coords:")
    print([idx_1d_to_2d(i, Lx) for i in range(Ntot)])

    # ---------------- save ----------------
    import pickle
    import pandas as pd
    from pathlib import Path

    data_dict = {idx_1d_to_2d(i, Lx): float(rdm1[0][i]) for i in range(Ntot)}

    save_dir = Path("/users/ylokare/2D_DMRG_calculations/results/13_by_13_lattice/Local_densities/Final_calculations/Varying_delta_boundary/Convergence_calculations/")
    save_dir.mkdir(parents=True, exist_ok=True)

    fname_base = (
        f"density_Lx13_Ly13_Dmax600_delta4.75_Rb1.9_dbound2.5_"
        f"varying_delta_boundary_square_initial_guess"
    )

    # save pkl
    pkl_path = save_dir / f"{fname_base}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(data_dict, f)

    # save csv
    csv_path = save_dir / f"{fname_base}.csv"
    df = pd.DataFrame([(y, x, v) for (y, x), v in data_dict.items()],
                      columns=["y", "x", "density"])
    df.to_csv(csv_path, index=False)

    print("\nSaved density data:")
    print(f"PKL → {pkl_path}")
    print(f"CSV → {csv_path}")
