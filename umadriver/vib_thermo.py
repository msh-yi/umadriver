from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import math, numpy as np
from ase import Atoms
from ase.vibrations import Vibrations
import os
import glob
from typing import Optional

from .constants import (
    HARTREE_PER_EV,
    EV_PER_HARTREE,
    KCAL_PER_MOL_PER_EV,
    kB_J_K,
    h_J_s,
    c_cm_s,
    amu_kg,
    angstrom_m,
    kB_eV_K,
    eV_per_cm1,
    theta_per_cm1_K,
)
from .writer import ORCAWriter

# Fallback-safe Avogadro
try:
    from . import constants as _const_mod

    AVOGADRO = getattr(_const_mod, "AVOGADRO", 6.02214076e23)
except Exception:
    AVOGADRO = 6.02214076e23


def _free_space_mL_per_L(solv: Optional[str]) -> float:
    """
    Return accessible free space (mL per L) for a solute in bulk solvent.
    Based on Shakhnovich & Whitesides (J. Org. Chem. 1998, 63, 3821) and
    the GoodVibes implementation.

    Supported keys: 'none', 'H2O', 'toluene', 'DMF', 'AcOH', 'chloroform'.
    """
    if not solv:
        return 1000.0

    solvent_list = ["none", "H2O", "toluene", "DMF", "AcOH", "chloroform"]
    molarity = [1.0, 55.6, 9.4, 12.9, 17.4, 12.5]  # mol/L
    molecular_vol = [1.0, 27.944, 149.070, 77.442, 86.10, 97.0]  # Å^3

    try:
        i = solvent_list.index(solv)
    except ValueError:
        # Unknown solvent -> assume full liter is “free”
        return 1000.0

    if i == 0:  # 'none'
        return 1000.0

    solv_m = molarity[i]
    solv_volA3 = molecular_vol[i]
    # v_free (Å^3 per molecule) for accessible volume
    v_free = (
        8.0
        * ((1e27 / (solv_m * AVOGADRO)) ** (1.0 / 3.0) - solv_volA3 ** (1.0 / 3.0)) ** 3
    )
    # Convert to mL free space per liter of bulk solvent
    freespace_mL_per_L = v_free * solv_m * AVOGADRO * 1e-24
    return float(freespace_mL_per_L)


# ---------- Geometry classification / rotation ----------
def principal_moments_amuA2(atoms: Atoms) -> np.ndarray:
    return atoms.get_moments_of_inertia()


def rotational_constants_cm1_from_I(I_amuA2: np.ndarray) -> Tuple[float, float, float]:
    I_SI = I_amuA2 * amu_kg * (angstrom_m**2)
    B_cm = []
    for I in I_SI:
        B_cm.append(0.0 if I <= 0 else h_J_s / (8.0 * math.pi**2 * c_cm_s * I))
    return (B_cm[0], B_cm[1], B_cm[2])


def classify_geometry(I_amuA2: np.ndarray) -> str:
    """
    Return one of: 'atom', 'linear', 'nonlinear'
    - 'atom' if all principal moments are (near) zero
    - 'linear' if one moment is ~0 but not all (diatomics, etc.)
    - 'nonlinear' otherwise
    """
    I = np.array(I_amuA2, dtype=float)
    imax = float(np.max(np.abs(I)))
    # Treat as atom if all moments are essentially zero
    if imax < 1e-12:
        return "atom"
    if (float(np.min(np.abs(I))) / imax) < 1e-6:
        return "linear"
    return "nonlinear"


# ---------- Mass-weighted normal modes ----------
def compute_mass_weighted_modes(vib, atoms: Atoms) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (U, w2) where:
      U: (3N, 3N) mass-weighted eigenvectors (columns are modes)
      w2: (3N,) eigenvalues of the dynamical matrix (ascending)
    Uses the same Hessian as ASE Vibrations, so ordering aligns with ASE freqs.
    """
    vd = vib.get_vibrations()  # VibrationsData
    K4 = vd.get_hessian()  # (N,3,N,3) in eV/Å^2
    N = len(atoms)
    K = K4.reshape(3 * N, 3 * N)

    masses = np.repeat(atoms.get_masses(), 3)  # amu
    inv_sqrt_m = 1.0 / np.sqrt(masses)
    # dynamical matrix D = M^{-1/2} K M^{-1/2}
    D = (inv_sqrt_m[:, None]) * K * (inv_sqrt_m[None, :])
    D = 0.5 * (D + D.T)

    w2, U = np.linalg.eigh(D)  # ascending by eigenvalue
    return U, w2


def run_frequencies_and_write(
    writer: ORCAWriter,
    atoms: Atoms,
    *,
    delta: float,
    nfree: int,
    scale: float,
    scratch_dir: str | None = None,
    ts: bool = False,
) -> List[float]:
    """
    Prints:
      - 3 (atom), 5 (linear) or 6 (nonlinear) rigid-body zeros at indices 0..zero_first-1,
      - if ts=True, the single imaginary at index zero_first,
      - remaining modes in increasing |frequency| order,
    and clamps tiny post-facto negatives to +abs for readability.
    """
    import os, glob, shutil
    from ase.vibrations import Vibrations

    # thresholds (tune if you like)
    eps_zero_rigid_cm1 = 1.0  # classify "rigid" by |f| <= this
    eps_small_cm1 = 5.0  # clamp tiny negatives/positives to +|f| if |f| < this

    writer.write_energy_grad_banner()
    writer._coords_block6(atoms)

    # vib scratch
    base_tag = os.path.splitext(os.path.basename(writer.path))[0]
    if scratch_dir is None:
        scratch_dir = os.getcwd()
    os.makedirs(scratch_dir, exist_ok=True)
    vib_prefix = os.path.join(scratch_dir, f"vib_{base_tag}")
    for p in glob.glob(vib_prefix + "*"):
        try:
            os.remove(p)
        except IsADirectoryError:
            shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass

    vib = Vibrations(atoms, name=vib_prefix, delta=delta, nfree=nfree)
    vib.run()

    # Frequencies (cm^-1)
    freqs_raw = vib.get_frequencies()
    freqs_cm = []
    for f in np.asarray(freqs_raw):
        if np.iscomplexobj(f) and abs(f.imag) > 1e-8:
            freqs_cm.append(-float(abs(f.imag)))  # imaginary -> negative
        else:
            freqs_cm.append(float(np.real(f)))
    freqs_cm = np.array(freqs_cm, dtype=float)

    # Mass-weighted eigenvectors (same underlying Hessian)
    modes_mw, w2 = compute_mass_weighted_modes(vib, atoms)  # (3N,3N), (3N,)

    # Geometry class -> rigid-body count
    I = principal_moments_amuA2(atoms)
    geom = classify_geometry(I)  # "atom", "linear" or "nonlinear"
    if geom == "atom":
        zero_first = 3
    elif geom == "linear":
        zero_first = 5
    else:
        zero_first = 6

    nmode = len(freqs_cm)
    idx_all = np.arange(nmode)

    # Identify the most negative as "imag" if any
    idx_imag = int(np.argmin(freqs_cm)) if np.any(freqs_cm < 0.0) else None

    # Choose rigid-body indices as those with the smallest |f|, excluding the chosen imag
    idx_sorted_abs = np.argsort(np.abs(freqs_cm))
    rigid = [i for i in idx_sorted_abs if i != idx_imag][:zero_first]

    # Build permutation:
    #   rigid zeros, then (if TS) the imaginary, then the rest by increasing |f|
    perm = []
    perm.extend(rigid)
    if ts and idx_imag is not None:
        perm.append(idx_imag)
    placed = set(perm)
    rest = [i for i in idx_all if i not in placed]
    rest_sorted = sorted(rest, key=lambda i: abs(freqs_cm[i]))
    perm.extend(rest_sorted)

    # Apply permutation to freqs and modes
    freqs_cm = freqs_cm[perm]
    modes_mw = modes_mw[:, perm]

    # Post-facto cleanup for printing:
    #   - force the first zero_first to exactly 0.0
    #   - clamp tiny |f| < eps_small_cm1 to +abs(f) for the rest
    freqs_cm[:zero_first] = 0.0
    for i in range(zero_first, nmode):
        if abs(freqs_cm[i]) < eps_small_cm1:
            freqs_cm[i] = abs(freqs_cm[i])

    # Scale for printing
    freqs_print = [f * scale for f in freqs_cm]

    # Emit sections
    writer.write_vibrational_frequencies(freqs_print, scale=scale)
    writer.write_normal_modes_preamble()
    writer.write_normal_modes_matrix(modes_mw, zero_first=zero_first)

    # IR spectrum: positive modes only, index = column number
    vib_modes = [
        (i, freqs_print[i]) for i in range(zero_first, nmode) if freqs_print[i] > 0.0
    ]
    writer.write_ir_spectrum(vib_modes)

    vib.clean()

    # Keep the scratch dir for debugging; if completely empty, remove it.
    try:
        leftover = glob.glob(os.path.join(scratch_dir, "*"))
        if not leftover:
            shutil.rmtree(scratch_dir, ignore_errors=True)
    except Exception:
        pass

    # Optional CUDA memory hygiene:
    try:
        import gc, torch

        del vib
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return freqs_print


# ---------- Thermochemistry (RRHO / Quasi-RRHO) ----------
def rrho_thermo(
    atoms: Atoms,
    freqs_cm1: List[float],
    T: float,
    P_atm: float,
    sigma: int,
    E_el_Eh: float,
    *,
    qrrho: bool = True,
    cutoff_cm1: Optional[float] = None,
    qrrho_ref_cm1: float = 100.0,
    qrrho_alpha: float = 4.0,
    # NEW:
    conc_mol_L: Optional[float] = None,
    solv: Optional[str] = None,
    multiplicity: int = 1,
) -> Dict[str, float]:
    # Remove 3/5/6 zero modes
    I = principal_moments_amuA2(atoms)
    geom = classify_geometry(I)  # 'atom', 'linear', 'nonlinear'
    if geom == "atom":
        start = 3
    elif geom == "linear":
        start = 5
    else:
        start = 6
    vib_cm_all = [f for i, f in enumerate(freqs_cm1) if i >= start and f > 0.0]

    if cutoff_cm1 is None:
        cutoff_cm1 = 1.0 if qrrho else 35.0
    vib_cm = [f for f in vib_cm_all if f > cutoff_cm1]

    mass_amu = float(np.sum(atoms.get_masses()))
    mass_kg = mass_amu * amu_kg

    B_A, B_B, B_C = rotational_constants_cm1_from_I(I)
    theta = [theta_per_cm1_K * b for b in (abs(B_A), abs(B_B), abs(B_C))]
    theta = [t if t > 0 else 1e-30 for t in theta]

    vib_e = [f * eV_per_cm1 for f in vib_cm]
    ZPE_eV = 0.5 * sum(vib_e)

    # Rot/trans energies (unchanged)
    Erot_eV = (
        0.0 if geom == "atom" else ((1.0 if geom == "linear" else 1.5) * kB_eV_K * T)
    )
    Etrans_eV = 1.5 * kB_eV_K * T

    # ---------- Translational entropy: pressure OR concentration ----------
    # S/k = ln( (2π m kT)^{3/2} / (h^3 n) ) + 5/2  where n is number density [1/m^3].
    # - Gas phase:    n = P / (kT)
    # - Solution:     n = (conc [mol/L]) * 1000 [L/m^3] * Na / (free_space_fraction),
    #                 with free-space from Shakhnovich–Whitesides.
    lambda_factor = ((2.0 * math.pi * mass_kg * kB_J_K * T) ** 1.5) / (h_J_s**3)

    if conc_mol_L is not None:
        free_mL_per_L = _free_space_mL_per_L(solv)
        # free-space fraction in a liter:
        free_frac = max(free_mL_per_L / 1000.0, 1e-9)  # avoid zero
        number_density = conc_mol_L * 1000.0 * AVOGADRO / free_frac  # 1/m^3
    else:
        P_Pa = P_atm * 101325.0
        number_density = P_Pa / (kB_J_K * T)

    S_trans_over_k = math.log(lambda_factor / number_density) + 2.5
    TS_trans_eV = kB_eV_K * T * S_trans_over_k

    # ---------- Rotational entropy ----------
    if geom == "atom":
        S_rot_over_k = 0.0
    elif geom == "linear":
        theta_rot = theta[1]
        S_rot_over_k = math.log(T / (sigma * theta_rot)) + 1.0
    else:
        prod_theta = theta[0] * theta[1] * theta[2]
        S_rot_over_k = (
            math.log(math.sqrt(math.pi) * (T**1.5) / (sigma * math.sqrt(prod_theta)))
            + 1.5
        )
    TS_rot_eV = kB_eV_K * T * S_rot_over_k

    # ---------- Vibrational entropy + thermal vibrational energy ----------
    I_SI = I * amu_kg * (angstrom_m**2)
    I_av = float(np.mean(I_SI)) if np.any(I_SI > 0) else 1e-46

    S_vib_over_k = 0.0
    Evib_corr_eV = 0.0

    for f_cm in vib_cm:
        e_eV = f_cm * eV_per_cm1
        x = e_eV / (kB_eV_K * T) if T > 0 else float("inf")

        if T > 0:
            S_HO_over_k = (x / (math.expm1(x))) - math.log1p(-math.exp(-x))
            E_th_HO_eV = e_eV / (math.expm1(x))
        else:
            S_HO_over_k = 0.0
            E_th_HO_eV = 0.0

        if not qrrho:
            S_vib_over_k += S_HO_over_k
            Evib_corr_eV += E_th_HO_eV
            continue

        w = 1.0 / (1.0 + (qrrho_ref_cm1 / f_cm) ** qrrho_alpha)

        nu_Hz = c_cm_s * f_cm
        muK = h_J_s / (8.0 * math.pi**2 * nu_Hz)  # kg·m^2
        muEff = (muK * I_av) / (muK + I_av)

        S_FR_over_k = 0.5 + 0.5 * math.log(
            (8.0 * math.pi**2 * muEff * kB_J_K * T) / (h_J_s**2)
        )

        S_vib_over_k += w * S_HO_over_k + (1.0 - w) * S_FR_over_k
        Evib_corr_eV += w * E_th_HO_eV + (1.0 - w) * (0.5 * kB_eV_K * T)

    # ---------- Electronic entropy (optional; = 0 if multiplicity==1) ----------
    TS_el_eV = (
        kB_eV_K
        * T
        * (0.0 if multiplicity <= 1 else math.log(max(float(multiplicity), 1.0)))
    )

    E_el_eV = E_el_Eh * EV_PER_HARTREE

    U_total_eV = E_el_eV + ZPE_eV + Evib_corr_eV + Erot_eV + Etrans_eV
    Hcorr_eV = kB_eV_K * T
    H_total_eV = U_total_eV + Hcorr_eV
    TS_vib_eV = kB_eV_K * T * S_vib_over_k
    TS_tot_eV = TS_el_eV + TS_vib_eV + TS_rot_eV + TS_trans_eV
    G_total_eV = H_total_eV - TS_tot_eV
    G_minus_Eel_eV = G_total_eV - E_el_eV

    # ---- conversions (unchanged) ----
    ZPE_Eh = ZPE_eV * HARTREE_PER_EV
    Evib_corr_Eh = Evib_corr_eV * HARTREE_PER_EV
    Erot_Eh = Erot_eV * HARTREE_PER_EV
    Etrans_Eh = Etrans_eV * HARTREE_PER_EV
    U_total_Eh = U_total_eV * HARTREE_PER_EV
    H_total_Eh = H_total_eV * HARTREE_PER_EV
    Hcorr_Eh = Hcorr_eV * HARTREE_PER_EV
    TS_el_Eh = TS_el_eV * HARTREE_PER_EV
    TS_vib_Eh = TS_vib_eV * HARTREE_PER_EV
    TS_rot_Eh = TS_rot_eV * HARTREE_PER_EV
    TS_trans_Eh = TS_trans_eV * HARTREE_PER_EV
    G_total_Eh = G_total_eV * HARTREE_PER_EV
    G_minus_Eel_Eh = G_minus_Eel_eV * HARTREE_PER_EV

    # symmetry sweep (unchanged)
    rot_table = []
    for sn in range(1, 13):
        if geom == "atom":
            TS_rot_sn_Eh = 0.0
        elif geom == "linear":
            S_rot_sn = math.log(T / (sn * (theta[1]))) + 1.0
            TS_rot_sn_Eh = (kB_eV_K * T * S_rot_sn) * HARTREE_PER_EV
        else:
            prod_theta = theta[0] * theta[1] * theta[2]
            S_rot_sn = (
                math.log(math.sqrt(math.pi) * (T**1.5) / (sn * math.sqrt(prod_theta)))
                + 1.5
            )
            TS_rot_sn_Eh = (kB_eV_K * T * S_rot_sn) * HARTREE_PER_EV
        rot_table.append((sn, TS_rot_sn_Eh))

    return dict(
        mass_amu=float(np.sum(atoms.get_masses())),
        rotconsts_cm1=(B_A, B_B, B_C),
        ZPE_Eh=ZPE_Eh,
        Evib_corr_Eh=Evib_corr_Eh,
        Erot_Eh=Erot_Eh,
        Etrans_Eh=Etrans_Eh,
        U_total_Eh=U_total_Eh,
        H_total_Eh=H_total_Eh,
        Hcorr_Eh=Hcorr_Eh,
        TS_el_Eh=TS_el_Eh,
        TS_vib_Eh=TS_vib_Eh,
        TS_rot_Eh=TS_rot_Eh,
        TS_trans_Eh=TS_trans_Eh,
        G_total_Eh=G_total_Eh,
        G_minus_Eel_Eh=G_minus_Eel_Eh,
        rot_table_Eh=rot_table,
        # Echo config for writer:
        qrrho=qrrho,
        cutoff_cm1=float(cutoff_cm1),
        qrrho_ref_cm1=float(qrrho_ref_cm1) if qrrho else None,
        qrrho_alpha=float(qrrho_alpha) if qrrho else None,
        # NEW echoes:
        conc_mol_L=conc_mol_L,
        solv=(solv or "none"),
        multiplicity=int(multiplicity),
    )
