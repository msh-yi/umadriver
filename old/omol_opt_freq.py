#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Literal, List
import argparse, time, os, shutil, subprocess, logging, math
from functools import lru_cache
from datetime import datetime

import numpy as np
from ase import Atoms
from ase.io import read
from ase.optimize import LBFGS, BFGS, BFGSLineSearch, FIRE, QuasiNewton
from fairchem.core import FAIRChemCalculator, pretrained_mlip

# -------------------------
# Logging
# -------------------------
LOG = logging.getLogger("omol_opt")


def setup_logging(verbose: bool = False, debug: bool = False):
    level = logging.WARNING
    if verbose:
        level = logging.INFO
    if debug:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("fairchem").setLevel(
        logging.WARNING if not debug else logging.DEBUG
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# Thread hygiene
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# -------------------------
# Caching (VAST default)
# -------------------------
VAST_BASE = os.environ.get("UMA_CACHE_BASE", "/n/netscratch/jacobsen_lab/Lab/msak")
DEFAULT_FAIRCHEM_CACHE = os.path.join(VAST_BASE, "fairchem_cache")


def cache_has_files(path: str) -> bool:
    return os.path.isdir(path) and any(os.scandir(path))


# stay offline only if cache exists
if cache_has_files(DEFAULT_FAIRCHEM_CACHE):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# -------------------------
# Units & Gaussian-style criteria
# -------------------------
HARTREE_PER_EV = 1.0 / 27.211386245988
EV_PER_HARTREE = 27.211386245988
BOHR_PER_ANG = 1.0 / 0.529177210903
EV_A_to_HB = HARTREE_PER_EV / BOHR_PER_ANG  # eV/Å -> Hartree/Bohr
KCAL_PER_MOL_PER_EV = 23.060548867
EV_PER_KCAL_PER_MOL = 1.0 / KCAL_PER_MOL_PER_EV

# Physical constants (SI)
kB_J_K = 1.380649e-23
h_J_s = 6.62607015e-34
c_m_s = 2.99792458e8
c_cm_s = 2.99792458e10
NA = 6.02214076e23
amu_kg = 1.66053906660e-27
angstrom_m = 1e-10

# Conversions
eV_per_J = 1.0 / 1.602176634e-19
kB_eV_K = 8.617333262145e-5  # eV/K
eV_per_cm1 = 1.23984197386209e-4  # hc in eV·cm
theta_per_cm1_K = 1.438776877  # theta (K) = 1.438776877 * wavenumber(cm^-1)

BASE = dict(  # Gaussian defaults (au)
    GRMS=3.0e-4,  # Hartree/Bohr
    GMAX=4.5e-4,  # Hartree/Bohr
    DRMS=1.2e-3,  # Bohr
    DMAX=1.8e-3,  # Bohr
)

GAUSS_RMS_FORCE = {  # target RMS force (au)
    "Loose": 1.7e-3,
    "Normal": 3.0e-4,
    "Tight": 1.0e-5,
    "VeryTight": 1.0e-6,
}

OptimName = Literal["LBFGS", "BFGS", "BFGSLineSearch", "FIRE", "QuasiNewton"]
OPTIMIZERS = {
    "LBFGS": LBFGS,
    "BFGS": BFGS,
    "BFGSLineSearch": BFGSLineSearch,
    "FIRE": FIRE,
    "QuasiNewton": QuasiNewton,
}


@dataclass
class GaussCutoffs:
    grms: float
    gmax: float
    drms: float
    dmax: float


def gaussian_cutoffs(
    mode: Literal["Loose", "Normal", "Tight", "VeryTight"] = "Normal",
) -> GaussCutoffs:
    target = GAUSS_RMS_FORCE[mode]
    scale = target / BASE["GRMS"]
    return GaussCutoffs(
        BASE["GRMS"] * scale,
        BASE["GMAX"] * scale,
        BASE["DRMS"] * scale,
        BASE["DMAX"] * scale,
    )


def gaussian_converged(grms, gmax, drms, dmax, cuts: GaussCutoffs) -> bool:
    return (
        (grms < cuts.grms)
        and (gmax < cuts.gmax)
        and (drms < cuts.drms)
        and (dmax < cuts.dmax)
    )


# -------------------------
# Case-insensitive / punctuation-agnostic parsers
# -------------------------
def _normkey(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def device_type(s: str) -> str:
    key = _normkey(s)
    mapping = {"cuda": "cuda", "gpu": "cuda", "cpu": "cpu", "auto": "cuda"}
    if key in mapping:
        return mapping[key]
    raise argparse.ArgumentTypeError(f"Unknown device '{s}'. Valid: cpu, cuda (gpu).")


def mode_type(s: str) -> str:
    key = _normkey(s)
    mapping = {
        "loose": "Loose",
        "normal": "Normal",
        "tight": "Tight",
        "verytight": "VeryTight",
    }
    if key in mapping:
        return mapping[key]
    raise argparse.ArgumentTypeError(
        "Unknown mode. Valid: Loose, Normal, Tight, VeryTight."
    )


def optimizer_type(s: str) -> str:
    key = _normkey(s)
    mapping = {
        "lbfgs": "LBFGS",
        "bfgs": "BFGS",
        "bfgslinesearch": "BFGSLineSearch",
        "bfgsline": "BFGSLineSearch",
        "bfgsls": "BFGSLineSearch",
        "fire": "FIRE",
        "quasinewton": "QuasiNewton",
        "qn": "QuasiNewton",
    }
    if key in mapping:
        return mapping[key]
    raise argparse.ArgumentTypeError(
        "Unknown optimizer. Valid: LBFGS, BFGS, BFGSLineSearch, FIRE, QuasiNewton."
    )


# -------------------------
# Cache + calculator
# -------------------------
LOCAL_CACHE = os.path.join(os.environ.get("TMPDIR", "/scratch"), "fairchem_cache")


def stage_cache_to_local(src: str, dst: str) -> str:
    t0 = time.time()
    LOG.info("Cache staging: src=%s -> dst=%s", src, dst)
    try:
        if os.path.isdir(src):
            if not os.path.isdir(dst):
                os.makedirs(dst, exist_ok=True)
                shutil.copytree(src, dst, dirs_exist_ok=True)
                LOG.info("Staged cache to local scratch in %.2fs", time.time() - t0)
            else:
                LOG.info("Local scratch cache exists; reusing.")
        else:
            LOG.warning("Source cache (%s) missing; will rely on hub or target.", src)
        return dst if (os.path.isdir(dst) and os.listdir(dst)) else src
    except Exception as e:
        LOG.warning("Cache staging failed (%s). Falling back to src.", e)
        return src


@lru_cache(maxsize=None)
def _get_predict_cached(model: str, device_opt: Optional[str], cache_dir_key: str):
    cache_dir = cache_dir_key or None
    LOG.info(
        "UMA lookup: model=%s device=%s cache=%s",
        model,
        device_opt,
        cache_dir or "<default>",
    )
    t0 = time.time()
    try:
        pred = pretrained_mlip.get_predict_unit(
            model, device=device_opt, cache_dir=cache_dir
        )
    except TypeError:
        LOG.debug("fairchem.get_predict_unit lacks cache_dir; retrying without.")
        pred = pretrained_mlip.get_predict_unit(model, device=device_opt)
    LOG.info(
        "[UMA init] model=%s device=%s took %.2fs",
        model,
        device_opt or "auto",
        time.time() - t0,
    )
    return pred


def build_calculator(
    model="uma-m-1p1",
    device: Optional[str] = "cuda",
    cache_dir: Optional[str] = None,
    use_local_scratch=False,
):
    base = cache_dir or DEFAULT_FAIRCHEM_CACHE
    resolved = stage_cache_to_local(base, LOCAL_CACHE) if use_local_scratch else base
    key = os.path.abspath(resolved) if resolved else ""
    predictor = _get_predict_cached(model, device, key)
    return FAIRChemCalculator(predictor, task_name="omol")


# -------------------------
# ORCA-style writer (matches accepted formats)
# -------------------------
class ORCAWriter:
    def __init__(
        self,
        path: str,
        xyz_path: str,
        model: str,
        device: str,
        *,
        opt_banner: bool = True,
    ):
        self.path = path
        self.f = open(path, "w", buffering=1)
        # Top banner
        self._w("                                 *****************")
        self._w("                                 * O   R   C   A *")
        self._w("                                 *****************")
        self._w("")
        self._w(f"OMol/ASE; model={model}; device={device}")
        self._w(datetime.now().strftime("Start  : %a %b %d %H:%M:%S  %Y"))
        self._w(f"Input  : {xyz_path}")
        self._w("")
        if opt_banner:
            self._w("                       *****************************")
            self._w("                       * Geometry Optimization Run *")
            self._w("                       *****************************")
            self._w("")

    def _w(self, s=""):
        self.f.write(s + ("\n" if not s.endswith("\n") else ""))

    # Cycle banner: G at column 27; cycles start at 1 (caller enforces)
    def _cycle_banner(self, n: int):
        left = " " * 9
        stars = "*" * 61
        self._w(f"{left}{stars}")
        title = f"GEOMETRY OPTIMIZATION CYCLE   {n}"
        interior = 61 - 2
        prefix_spaces = 16
        pad = max(0, interior - prefix_spaces - len(title))
        self._w(f"{left}*{' ' * prefix_spaces}{title}{' ' * pad}*")
        self._w(f"{left}{stars}")

    # Coordinates with 16.10f (optimizer sections)
    def _coords_block10(self, atoms: Atoms):
        self._w("---------------------------------")
        self._w("CARTESIAN COORDINATES (ANGSTROEM)")
        self._w("---------------------------------")
        syms = atoms.get_chemical_symbols()
        pos = atoms.get_positions()
        for s, (x, y, z) in zip(syms, pos):
            self._w(f"  {s:<3s}{x:16.10f} {y:16.10f} {z:16.10f}")
        self._w("")

    # Coordinates with 6 decimals (freq banner sample)
    def _coords_block6(self, atoms: Atoms):
        self._w("---------------------------------")
        self._w("CARTESIAN COORDINATES (ANGSTROEM)")
        self._w("---------------------------------")
        syms = atoms.get_chemical_symbols()
        pos = atoms.get_positions()
        for s, (x, y, z) in zip(syms, pos):
            self._w(f"  {s:<3s}{x:12.6f} {y:11.6f} {z:12.6f}")
        self._w("")

    def _energy_box(self, energy_h: float):
        self._w("-------------------------   --------------------")
        self._w(f"FINAL SINGLE POINT ENERGY     {energy_h:.15f}")
        self._w("-------------------------   --------------------")
        self._w("")

    def _geom_conv_box(self, grms, gmax, drms, dmax, cuts):
        self._w("                                .--------------------.")
        self._w(
            "          ----------------------|Geometry convergence|-------------------------"
        )
        self._w(
            "          Item                value                   Tolerance       Converged"
        )
        self._w(
            "          ---------------------------------------------------------------------"
        )
        self._w(
            f"          RMS gradient        {grms:12.10f}            {cuts.grms:12.10f}      {'YES' if grms < cuts.grms else 'NO'}"
        )
        self._w(
            f"          MAX gradient        {gmax:12.10f}            {cuts.gmax:12.10f}      {'YES' if gmax < cuts.gmax else 'NO'}"
        )
        self._w(
            f"          RMS step            {drms:12.10f}            {cuts.drms:12.10f}      {'YES' if drms < cuts.drms else 'NO'}"
        )
        self._w(
            f"          MAX step            {dmax:12.10f}            {cuts.dmax:12.10f}      {'YES' if dmax < cuts.dmax else 'NO'}"
        )
        self._w("")
        self._w("          ........................................................")
        self._w("          Max(Bonds)      0.0000      Max(Angles)    0.00")
        self._w("          Max(Dihed)        0.00      Max(Improp)    0.00")
        self._w(
            "          ---------------------------------------------------------------------"
        )
        self._w("")

    # --- Frequency sections (match ORCA-ish sample) ---
    def write_energy_grad_banner(self):
        self._w("                     *******************************")
        self._w("                     * Energy+Gradient Calculation *")
        self._w("                     *******************************")
        self._w("")

    def write_vibrational_frequencies(self, freqs_cm1: List[float], scale: float):
        self._w("-----------------------")
        self._w("VIBRATIONAL FREQUENCIES")
        self._w("-----------------------")
        self._w("")
        self._w(f"Scaling factor for frequencies =  {scale:.9f} (already applied!)")
        self._w("")
        for i, f in enumerate(freqs_cm1):
            self._w(f"{i:4d}:{f:13.2f} cm**-1")
        self._w("")

    def write_normal_modes_preamble(self):
        self._w("------------")
        self._w("NORMAL MODES")
        self._w("------------")
        self._w("")
        self._w(
            "These modes are the cartesian displacements weighted by the diagonal matrix"
        )
        self._w("M(i,i)=1/sqrt(m[i]) where m[i] is the mass of the displaced atom")
        self._w("Thus, these vectors are normalized but *not* orthogonal")
        self._w("")

    def write_normal_modes_matrix(self, modes_mw: np.ndarray, zero_first: int = 0):
        """
        modes_mw: (3N, 3N) mass-weighted eigenvectors (columns are modes)
        zero_first: number of initial columns to zero (5 for linear, 6 for nonlinear)
        Prints in the exact ORCA-like format your viewer accepts.
        """
        V = modes_mw.copy()
        if zero_first > 0:
            V[:, :zero_first] = 0.0  # ensure pure '0.000000' (no -0.000000)

        nrows, ncols = V.shape
        block = 6

        for c0 in range(0, ncols, block):
            c1 = min(c0 + block, ncols)

            # Header line: 18-space indent, column indices left-aligned in width 11,
            # then exactly four trailing spaces (per your sample).
            indent = " " * 18
            pieces = [f"{j:<11d}" for j in range(c0, c1)]
            header = indent + "".join(pieces).rstrip() + "    "
            self._w(header)

            # Rows: row index right-aligned width 7, then 4 extra spaces, then each value as right-aligned width 11 with 6 decimals
            for r in range(nrows):
                line = (
                    f"{r:7d}"
                    + "    "
                    + "".join(f"{V[r, j]:11.6f}" for j in range(c0, c1))
                )
                self._w(line)

    def write_ir_spectrum(self, vib_modes: List[tuple[int, float]]):
        self._w("-----------")
        self._w("IR SPECTRUM")
        self._w("-----------")
        self._w("")
        self._w(
            " Mode   freq       eps      Int      T**2         TX        TY        TZ"
        )
        self._w("       cm**-1   L/(mol*cm) km/mol    a.u.")
        self._w(
            "----------------------------------------------------------------------------"
        )
        for idx, f in vib_modes:
            self._w(
                f"{idx:4d}: {f:8.2f}   {0.000000:0.6f}   {0.00:6.2f}  {0.000000:0.6f}  ({-0.000000: .6f} {-0.000000: .6f} {0.000000: .6f})"
            )
        self._w("")

    # High-level writers for optimization
    def write_cycle(
        self,
        cycle: int,
        atoms: Atoms,
        energy_h: float,
        grms: float,
        gmax: float,
        drms: float,
        dmax: float,
        cuts: GaussCutoffs,
    ):
        self._cycle_banner(cycle)
        self._coords_block10(atoms)
        self._energy_box(energy_h)
        self._geom_conv_box(grms, gmax, drms, dmax, cuts)

    def write_final_geom_and_energy(
        self, atoms: Atoms, energy_h: float, converged: bool
    ):
        if converged:
            self._w(
                "                    ***********************HURRAY********************"
            )
            self._w(
                "                    ***        THE OPTIMIZATION HAS CONVERGED     ***"
            )
            self._w(
                "                    *************************************************"
            )
            self._w("")
        self._coords_block10(atoms)
        self._energy_box(energy_h)

    def write_termination(self, wall_s: float):
        self._w("                             ****ORCA TERMINATED NORMALLY****")
        days = int(wall_s // 86400)
        rem = wall_s - days * 86400
        hours = int(rem // 3600)
        rem -= hours * 3600
        mins = int(rem // 60)
        rem -= mins * 60
        secs = int(rem)
        msec = int((rem - secs) * 1000)
        self._w(
            f"TOTAL RUN TIME: {days} days {hours} hours {mins} minutes {secs} seconds {msec} msec"
        )
        self.f.flush()

    # --- Thermochemistry block (now advertises RRHO vs qRRHO + knobs) ---
    def write_thermochemistry(
        self,
        T: float,
        P_atm: float,
        mass_amu: float,
        point_group: str,
        sigma: int,
        rotconsts_cm1: Tuple[float, float, float],
        use_qrrho: bool,
        cutoff_cm1: float,
        qrrho_ref_cm1: Optional[float],
        qrrho_alpha: Optional[float],
        # contributions in Eh and kcal/mol
        E_el_Eh: float,
        ZPE_Eh: float,
        Evib_corr_Eh: float,
        Erot_Eh: float,
        Etrans_Eh: float,
        Hcorr_Eh: float,  # kB*T (Eh)
        TS_el_Eh: float,
        TS_vib_Eh: float,
        TS_rot_Eh: float,
        TS_trans_Eh: float,
        G_total_Eh: float,
        H_total_Eh: float,
        U_total_Eh: float,
        G_minus_Eel_Eh: float,
        rot_entropy_table_Eh: List[Tuple[int, float]],
    ):
        self._w("--------------------------")
        self._w(f"THERMOCHEMISTRY AT {T:.2f}K")
        self._w("--------------------------")
        self._w("")
        self._w(f"Temperature         ... {T:.2f} K")
        self._w(f"Pressure            ... {P_atm:.2f} atm")
        self._w(f"Total Mass          ... {mass_amu:.2f} AMU")
        self._w("")
        self._w("Throughout the following assumptions are being made:")
        self._w("  (1) The electronic state is orbitally nondegenerate")
        self._w("  (2) There are no thermally accessible electronically excited states")
        self._w("  (3) Hindered rotations indicated by low frequency modes are not")
        self._w("      treated as such but are treated as vibrations and this may")
        yourline = "      cause some error"
        self._w(yourline)
        self._w("  (4) All equations used are the standard statistical mechanics")
        self._w("      equations for an ideal gas")
        self._w("  (5) All vibrations are strictly harmonic")
        self._w("")
        self._w("")

        self._w("------------")
        self._w("INNER ENERGY")
        self._w("------------")
        self._w("")
        self._w("The inner energy is: U= E(el) + E(ZPE) + E(vib) + E(rot) + E(trans)")
        self._w(
            "    E(el)   - is the total energy from the electronic structure calculation"
        )
        self._w("              = E(kin-el) + E(nuc-el) + E(el-el) + E(nuc-nuc)")
        self._w(
            "    E(ZPE)  - the the zero temperature vibrational energy from the frequency calculation"
        )
        self._w(
            "    E(vib)  - the the finite temperature correction to E(ZPE) due to population"
        )
        self._w("              of excited vibrational states")
        self._w("    E(rot)  - is the rotational thermal energy")
        self._w("    E(trans)- is the translational thermal energy")
        self._w("")

        def line(lbl, Eh, kcal):
            self._w(f"{lbl:<32} ... {Eh:12.8f} Eh  {kcal:9.2f} kcal/mol")

        line("Summary of contributions to the inner energy U:", 0.0, 0.0)
        line(
            "Electronic energy", E_el_Eh, E_el_Eh * EV_PER_HARTREE * KCAL_PER_MOL_PER_EV
        )
        line("Zero point energy", ZPE_Eh, ZPE_Eh * EV_PER_HARTREE * KCAL_PER_MOL_PER_EV)
        line(
            "Thermal vibrational correction",
            Evib_corr_Eh,
            Evib_corr_Eh * EV_PER_HARTREE * KCAL_PER_MOL_PER_EV,
        )
        line(
            "Thermal rotational correction",
            Erot_Eh,
            Erot_Eh * EV_PER_HARTREE * KCAL_PER_MOL_PER_EV,
        )
        line(
            "Thermal translational correction",
            Etrans_Eh,
            Etrans_Eh * EV_PER_HARTREE * KCAL_PER_MOL_PER_EV,
        )
        self._w(
            "-----------------------------------------------------------------------"
        )
        self._w(f"Total thermal energy                    {U_total_Eh:12.8f} Eh")
        self._w("")
        self._w("")
        self._w("Summary of corrections to the electronic energy:")
        self._w("(perhaps to be used in another calculation)")
        self._w(
            f"Total thermal correction                  {Evib_corr_Eh+Erot_Eh+Etrans_Eh:12.8f} Eh"
        )
        self._w(f"Non-thermal (ZPE) correction              {ZPE_Eh:12.8f} Eh")
        self._w(
            "-----------------------------------------------------------------------"
        )
        self._w(
            f"Total correction                          {ZPE_Eh+Evib_corr_Eh+Erot_Eh+Etrans_Eh:12.8f} Eh"
        )
        self._w("")
        self._w("")
        self._w("--------")
        self._w("ENTHALPY")
        self._w("--------")
        self._w("")
        self._w("The enthalpy is H = U + kB*T")
        self._w("                kB is Boltzmann's constant")
        self._w(f"Total free energy                 ...    {U_total_Eh:12.8f} Eh ")
        self._w(
            f"Thermal Enthalpy correction       ...      {Hcorr_Eh:10.8f} Eh      {Hcorr_Eh*EV_PER_HARTREE*KCAL_PER_MOL_PER_EV:6.2f} kcal/mol"
        )
        self._w(
            "-----------------------------------------------------------------------"
        )
        self._w(f"Total Enthalpy                    ...    {H_total_Eh:12.8f} Eh")
        self._w("")

        # --- Advertise method + knobs, then Herzberg + rot consts ---
        if use_qrrho:
            self._w("Vibrational entropy computed via Quasi-RRHO (Grimme).")
            self._w("Reference: Chem. Eur. J. 2012, 18, 9955.")
            self._w(f"QRRHORefFreq  ... {qrrho_ref_cm1:.1f} cm-1")
            self._w(f"Mix exponent α ... {qrrho_alpha:.1f}")
            self._w(
                f"CutOffFreq    ... {cutoff_cm1:.1f} cm-1 (modes below excluded from thermo)"
            )
        else:
            self._w("Vibrational entropy computed via RRHO (harmonic oscillator).")
            self._w(
                f"CutOffFreq    ... {cutoff_cm1:.1f} cm-1 (modes below excluded from thermo)"
            )
        self._w("")
        self._w("Note: Rotational entropy computed according to Herzberg ")
        self._w("Infrared and Raman Spectra, Chapter V,1, Van Nostrand Reinhold, 1945 ")
        pg = point_group or "C1"
        self._w(f"Point Group:  {pg}, Symmetry Number:  {sigma:3d}  ")
        self._w(
            f"Rotational constants in cm-1:   {rotconsts_cm1[0]:10.6f}   {rotconsts_cm1[1]:10.6f}   {rotconsts_cm1[2]:10.6f} "
        )
        self._w("")
        self._w("-------")
        self._w("ENTROPY")
        self._w("-------")
        self._w("")
        self._w("The entropy contributions are T*S = T*(S(el)+S(vib)+S(rot)+S(trans))")
        self._w("     S(el)   - electronic entropy")
        self._w("     S(vib)  - vibrational entropy")
        self._w("     S(rot)  - rotational entropy")
        self._w("     S(trans)- translational entropy")
        self._w("The entropies will be listed as multiplied by the temperature to get")
        self._w("units of energy")
        self._w("")

        def lineTS(lbl, TS_Eh):
            self._w(
                f"{lbl:<32} ...      {TS_Eh:10.8f} Eh  {TS_Eh*EV_PER_HARTREE*KCAL_PER_MOL_PER_EV:6.2f} kcal/mol"
            )

        lineTS("Electronic entropy", TS_el_Eh)
        lineTS("Vibrational entropy", TS_vib_Eh)
        lineTS("Rotational entropy", TS_rot_Eh)
        lineTS("Translational entropy", TS_trans_Eh)
        self._w(
            "-----------------------------------------------------------------------"
        )
        TS_tot = TS_el_Eh + TS_vib_Eh + TS_rot_Eh + TS_trans_Eh
        self._w(
            f"Final entropy term                ...      {TS_tot:10.8f} Eh  {TS_tot*EV_PER_HARTREE*KCAL_PER_MOL_PER_EV:6.2f} kcal/mol"
        )
        self._w("")
        self._w(
            "In case the symmetry of your molecule has not been determined correctly"
        )
        self._w(
            "or in case you have a reason to use a different symmetry number we print "
        )
        self._w("out the resulting rotational entropy values for sn=1,12 :")
        self._w(" --------------------------------------------------------")
        for sn, TSrot in rot_entropy_table_Eh:
            self._w(
                f"|  sn={sn:2d} | S(rot)=       {TSrot:0.8f} Eh  {TSrot*EV_PER_HARTREE*KCAL_PER_MOL_PER_EV:6.2f} kcal/mol|"
            )
        self._w(" --------------------------------------------------------")
        self._w("")
        self._w("")
        self._w("-------------------")
        self._w("GIBBS FREE ENERGY")
        self._w("-------------------")
        self._w("")
        self._w("The Gibbs free energy is G = H - T*S")
        self._w("")
        self._w(f"Total enthalpy                    ...    {H_total_Eh:12.8f} Eh ")
        self._w(
            f"Total entropy correction          ...     {-TS_tot:11.8f} Eh     {-TS_tot*EV_PER_HARTREE*KCAL_PER_MOL_PER_EV:6.2f} kcal/mol"
        )
        self._w(
            "-----------------------------------------------------------------------"
        )
        self._w(f"Final Gibbs free energy         ...    {G_total_Eh:12.8f} Eh")
        self._w("")
        self._w("For completeness - the Gibbs free energy minus the electronic energy")
        self._w(
            f"G-E(el)                           ...      {G_minus_Eel_Eh:10.8f} Eh      {G_minus_Eel_Eh*EV_PER_HARTREE*KCAL_PER_MOL_PER_EV:6.2f} kcal/mol"
        )
        self._w("")

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


# -------------------------
# Helpers
# -------------------------
def load_xyz(path: str) -> Atoms:
    LOG.info("Loading XYZ: %s", path)
    atoms = read(path)
    atoms.pbc = False
    LOG.info("Loaded natoms=%d", len(atoms))
    return atoms


def set_charge_mult(atoms: Atoms, charge: int = 0, multiplicity: int = 1):
    atoms.info.update({"charge": charge, "spin": multiplicity})


def force_metrics_HB(forces_evA: np.ndarray) -> Tuple[float, float]:
    mags = np.linalg.norm(forces_evA, axis=1) * EV_A_to_HB
    return float(np.sqrt((mags**2).mean())), float(mags.max())


def disp_metrics_bohr(
    prev_pos_A: Optional[np.ndarray], curr_pos_A: np.ndarray
) -> Tuple[float, float]:
    if prev_pos_A is None:
        return 0.0, 0.0
    mags = np.linalg.norm(curr_pos_A - prev_pos_A, axis=1) * BOHR_PER_ANG
    return float(np.sqrt((mags**2).mean())), float(mags.max())


def resolve_device(prefer: str = "cuda") -> str:
    if prefer == "cuda":
        try:
            import torch

            have = torch.cuda.is_available()
            LOG.info(
                "CUDA available: %s | CUDA_VISIBLE_DEVICES=%s",
                have,
                os.environ.get("CUDA_VISIBLE_DEVICES"),
            )
            if have:
                try:
                    out = subprocess.run(
                        ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=2
                    )
                    if out.returncode == 0:
                        LOG.info(
                            "nvidia-smi -L: %s", out.stdout.strip().replace("\n", " | ")
                        )
                except Exception:
                    pass
                return "cuda"
            LOG.warning("CUDA requested but unavailable; falling back to CPU.")
        except Exception as e:
            LOG.warning("CUDA check failed (%s); falling back to CPU.", e)
        return "cpu"
    return "cpu"


# -------------------------
# Vibrations helpers (mass-weighted modes)
# -------------------------
def principal_moments_amuA2(atoms: Atoms) -> np.ndarray:
    return atoms.get_moments_of_inertia()


def rotational_constants_cm1_from_I(I_amuA2: np.ndarray) -> Tuple[float, float, float]:
    I_SI = I_amuA2 * amu_kg * (angstrom_m**2)
    B_cm = []
    for I in I_SI:
        if I <= 0:
            B_cm.append(0.0)
        else:
            B_cm.append(h_J_s / (8.0 * math.pi**2 * c_cm_s * I))
    return (B_cm[0], B_cm[1], B_cm[2])


def classify_geometry(I_amuA2: np.ndarray) -> str:
    I = np.array(I_amuA2, dtype=float)
    imax = np.max(I)
    if imax == 0:
        return "nonlinear"
    if (np.min(I) / imax) < 1e-6:
        return "linear"
    return "nonlinear"


def compute_mass_weighted_modes(vib, atoms: Atoms) -> np.ndarray:
    """
    Return eigenvectors in mass-weighted Cartesian coordinates.
    Columns correspond to modes sorted by increasing eigenvalue (frequency^2).
    """
    vd = vib.get_vibrations()  # VibrationsData
    K4 = vd.get_hessian()  # shape (N, 3, N, 3) in eV/Å^2
    N = len(atoms)
    K = K4.reshape(3 * N, 3 * N)  # -> (3N, 3N)

    masses = np.repeat(atoms.get_masses(), 3)  # amu, length 3N
    inv_sqrt_m = 1.0 / np.sqrt(masses)
    # Dynamical matrix D = M^{-1/2} K M^{-1/2}
    D = (inv_sqrt_m[:, None]) * K * (inv_sqrt_m[None, :])
    D = 0.5 * (D + D.T)  # symmetrize for numerical hygiene

    w2, U = np.linalg.eigh(D)  # U columns are mass-weighted eigenvectors
    order = np.argsort(np.real(w2))
    return U[:, order]  # (3N, 3N)


# -------------------------
# Frequency driver (ASE Vibrations) + writing
# -------------------------
def run_frequencies_and_write(
    writer: ORCAWriter, atoms: Atoms, *, delta: float, nfree: int, scale: float
) -> List[float]:
    from ase.vibrations import Vibrations

    writer.write_energy_grad_banner()
    writer._coords_block6(atoms)

    vib = Vibrations(atoms, name="vib", delta=delta, nfree=nfree)
    vib.run()

    # --- Frequencies (handle complex → negative real for imaginary modes) ---
    freqs_raw = vib.get_frequencies()  # may be complex for imaginary modes
    freqs_cm = []
    for f in np.asarray(freqs_raw):
        if np.iscomplexobj(f) and abs(f.imag) > 1e-8:
            val = -float(abs(f.imag))  # print imaginary as negative cm^-1
        else:
            val = float(np.real(f))
        if abs(val) < 1e-2:  # tidy tiny ~0
            val = 0.0
        freqs_cm.append(val)

    # Apply printed scaling
    freqs = [f * scale for f in freqs_cm]

    writer.write_vibrational_frequencies(freqs, scale=scale)
    writer.write_normal_modes_preamble()

    # --- Normal modes matrix with your exact formatting + zeroed rot/trans ---
    I = principal_moments_amuA2(atoms)
    geom = classify_geometry(I)  # "linear" or "nonlinear"
    zero_first = 5 if geom == "linear" else 6

    modes_mw = compute_mass_weighted_modes(vib, atoms)  # (3N, 3N)
    writer.write_normal_modes_matrix(modes_mw, zero_first=zero_first)

    # --- IR SPECTRUM: skip the first zero_first modes, BUT keep indices ---
    # Viewer needs the mode numbers to be the original column indices (6, 7, 8, ...).
    vib_modes = [(i, freqs[i]) for i in range(zero_first, len(freqs)) if freqs[i] > 0.0]
    writer.write_ir_spectrum(vib_modes)

    vib.clean()
    return freqs


# -------------------------
# Thermochemistry (RRHO / Quasi-RRHO)
# -------------------------
def rrho_thermo(
    atoms: Atoms,
    freqs_cm1: List[float],
    T: float,
    P_atm: float,
    sigma: int,
    E_el_Eh: float,
    *,
    qrrho: bool = True,
    cutoff_cm1: Optional[float] = None,  # None -> set defaults below
    qrrho_ref_cm1: float = 100.0,
    qrrho_alpha: float = 4.0,
) -> Dict[str, float]:
    # Remove 5/6 zero modes
    I = principal_moments_amuA2(atoms)
    geom = classify_geometry(I)
    start = 5 if geom == "linear" else 6
    vib_cm_all = [f for i, f in enumerate(freqs_cm1) if i >= start and f > 0.0]

    # Defaults for cutoffs
    if cutoff_cm1 is None:
        cutoff_cm1 = 1.0 if qrrho else 35.0
    vib_cm = [f for f in vib_cm_all if f > cutoff_cm1]

    # Mass
    mass_amu = float(np.sum(atoms.get_masses()))
    mass_kg = mass_amu * amu_kg

    # Rot. consts & temperatures (K)
    B_A, B_B, B_C = rotational_constants_cm1_from_I(I)
    theta = [theta_per_cm1_K * b for b in (abs(B_A), abs(B_B), abs(B_C))]
    theta = [t if t > 0 else 1e-30 for t in theta]  # avoid log(0)

    # ZPE base (always HO 1/2 hν)
    vib_e = [f * eV_per_cm1 for f in vib_cm]
    ZPE_eV = 0.5 * sum(vib_e)

    # Rotational and translational energies (eV)
    if geom == "linear":
        Erot_eV = 1.0 * kB_eV_K * T
    else:
        Erot_eV = 1.5 * kB_eV_K * T
    Etrans_eV = 1.5 * kB_eV_K * T

    # Entropy: translational (Sackur–Tetrode)
    P_Pa = P_atm * 101325.0
    S_trans_over_k = (
        math.log(
            ((2 * math.pi * mass_kg * kB_J_K * T) ** 1.5)
            * (kB_J_K * T)
            / (h_J_s**3 * P_Pa)
        )
        + 2.5
    )
    TS_trans_eV = kB_eV_K * T * S_trans_over_k

    # Rotational entropy (Herzberg)
    if geom == "linear":
        theta_rot = theta[1]  # any non-zero
        S_rot_over_k = math.log(T / (sigma * theta_rot)) + 1.0
    else:
        prod_theta = theta[0] * theta[1] * theta[2]
        S_rot_over_k = (
            math.log(math.sqrt(math.pi) * (T**1.5) / (sigma * math.sqrt(prod_theta)))
            + 1.5
        )
    TS_rot_eV = kB_eV_K * T * S_rot_over_k

    # Vibrational entropy + thermal vibrational energy
    # qRRHO mixing (Grimme): weight w(ν) between HO and 1D FR
    I_SI = I * amu_kg * (angstrom_m**2)
    I_av = float(np.mean(I_SI)) if np.any(I_SI > 0) else 1e-46

    S_vib_over_k = 0.0
    Evib_corr_eV = 0.0

    for f_cm in vib_cm:
        e_eV = f_cm * eV_per_cm1
        x = e_eV / (kB_eV_K * T) if T > 0 else float("inf")

        # HO entropy, thermal piece
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

        # qRRHO weight
        w = 1.0 / (1.0 + (qrrho_ref_cm1 / f_cm) ** qrrho_alpha)

        # 1D free-rotor proxy for this mode
        nu_Hz = c_cm_s * f_cm
        muK = h_J_s / (8.0 * math.pi**2 * nu_Hz)  # kg·m^2
        muEff = (muK * I_av) / (muK + I_av)  # effective inertia

        # 1D FR entropy: S/k = 1/2 + ln( sqrt( 8π^2 I kT / h^2 ) )
        S_FR_over_k = 0.5 + 0.5 * math.log(
            (8.0 * math.pi**2 * muEff * kB_J_K * T) / (h_J_s**2)
        )

        # Entropy mixing
        S_vib_over_k += w * S_HO_over_k + (1.0 - w) * S_FR_over_k

        # Thermal energy mixing (Li et al., many codes): HO → 1/2 kT
        E_th_mix_eV = w * E_th_HO_eV + (1.0 - w) * (0.5 * kB_eV_K * T)
        Evib_corr_eV += E_th_mix_eV

    # Electronic entropy (assume 0)
    TS_el_eV = 0.0

    # Totals
    E_el_eV = E_el_Eh * EV_PER_HARTREE
    U_total_eV = E_el_eV + ZPE_eV + Evib_corr_eV + Erot_eV + Etrans_eV
    Hcorr_eV = kB_eV_K * T
    H_total_eV = U_total_eV + Hcorr_eV
    TS_vib_eV = kB_eV_K * T * S_vib_over_k
    TS_tot_eV = TS_el_eV + TS_vib_eV + TS_rot_eV + TS_trans_eV
    G_total_eV = H_total_eV - TS_tot_eV
    G_minus_Eel_eV = G_total_eV - E_el_eV

    # Symmetry-number sweep table for TS_rot (Eh)
    rot_table = []
    for sn in range(1, 13):
        if geom == "linear":
            S_rot_sn = math.log(T / (sn * (theta[1]))) + 1.0
        else:
            prod_theta = theta[0] * theta[1] * theta[2]
            S_rot_sn = (
                math.log(math.sqrt(math.pi) * (T**1.5) / (sn * math.sqrt(prod_theta)))
                + 1.5
            )
        TS_rot_sn_eV = kB_eV_K * T * S_rot_sn
        rot_table.append((sn, TS_rot_sn_eV * HARTREE_PER_EV))

    return dict(
        mass_amu=mass_amu,
        rotconsts_cm1=(B_A, B_B, B_C),
        ZPE_Eh=ZPE_eV * HARTREE_PER_EV,
        Evib_corr_Eh=Evib_corr_eV * HARTREE_PER_EV,
        Erot_Eh=Erot_eV * HARTREE_PER_EV,
        Etrans_Eh=Etrans_eV * HARTREE_PER_EV,
        U_total_Eh=U_total_eV * HARTREE_PER_EV,
        H_total_Eh=H_total_eV * HARTREE_PER_EV,
        Hcorr_Eh=Hcorr_eV * HARTREE_PER_EV,
        TS_el_Eh=TS_el_eV * HARTREE_PER_EV,
        TS_vib_Eh=TS_vib_eV * HARTREE_PER_EV,
        TS_rot_Eh=TS_rot_eV * HARTREE_PER_EV,
        TS_trans_Eh=TS_trans_eV * HARTREE_PER_EV,
        G_total_Eh=G_total_eV * HARTREE_PER_EV,
        G_minus_Eel_Eh=G_minus_Eel_eV * HARTREE_PER_EV,
        rot_table_Eh=rot_table,
        # Echo configuration for writer
        qrrho=qrrho,
        cutoff_cm1=float(cutoff_cm1),
        qrrho_ref_cm1=float(qrrho_ref_cm1) if qrrho else None,
        qrrho_alpha=float(qrrho_alpha) if qrrho else None,
    )


# -------------------------
# Optimizer
# -------------------------
def optimize_xyz(
    xyz_path: str,
    *,
    out_path: str,
    charge: int = 0,
    multiplicity: int = 1,
    model: str = "uma-m-1p1",
    device: str = "cuda",
    mode: Literal["Loose", "Normal", "Tight", "VeryTight"] = "Normal",
    maxcycles: int = 300,
    optimizer: OptimName = "LBFGS",
    maxstep: Optional[float] = None,
    damp: Optional[float] = None,
    cache_dir: Optional[str] = None,
    use_local_scratch: bool = False,
    # freq options
    do_freq: bool = False,
    freq_only: bool = False,
    freq_delta: float = 0.01,
    freq_nfree: int = 2,
    freq_scale: float = 1.0,
    # thermo options
    temp: float = 298.15,
    pressure_atm: float = 1.0,
    symmetry_number: int = 1,
    point_group: Optional[str] = None,
    thermo_scale: Optional[float] = None,
    # NEW: qRRHO controls
    qrrho: bool = True,
    cutoff_cm1: Optional[float] = None,
    qrrho_ref_cm1: float = 100.0,
    qrrho_alpha: float = 4.0,
) -> Dict:
    LOG.info("=== OMol optimize ===")
    LOG.info(
        "xyz=%s | out=%s | mode=%s | optimizer=%s | maxcycles=%d maxstep=%s",
        xyz_path,
        out_path,
        mode,
        optimizer,
        maxcycles,
        str(maxstep),
    )
    LOG.info(
        "cache_dir=%s | use_local_scratch=%s",
        cache_dir or DEFAULT_FAIRCHEM_CACHE,
        use_local_scratch,
    )

    t_start = time.time()
    atoms = load_xyz(xyz_path)
    set_charge_mult(atoms, charge, multiplicity)
    dev = resolve_device(device)
    LOG.info("Using device: %s  (natoms=%d)", dev, len(atoms))
    atoms.calc = build_calculator(
        model=model,
        device=dev,
        cache_dir=cache_dir,
        use_local_scratch=use_local_scratch,
    )

    # Writer
    writer = ORCAWriter(out_path, xyz_path, model, dev, opt_banner=not freq_only)

    # Frequencies-only path
    if freq_only:
        freqs = run_frequencies_and_write(
            writer, atoms, delta=freq_delta, nfree=freq_nfree, scale=freq_scale
        )
        used_scale = thermo_scale if (thermo_scale is not None) else freq_scale
        freqs_th = [f * used_scale for f in freqs]
        # Electronic energy at input geometry:
        E_h = atoms.get_potential_energy() * HARTREE_PER_EV
        th = rrho_thermo(
            atoms,
            freqs_th,
            temp,
            pressure_atm,
            symmetry_number,
            E_h,
            qrrho=qrrho,
            cutoff_cm1=cutoff_cm1,
            qrrho_ref_cm1=qrrho_ref_cm1,
            qrrho_alpha=qrrho_alpha,
        )
        writer.write_thermochemistry(
            T=temp,
            P_atm=pressure_atm,
            mass_amu=th["mass_amu"],
            point_group=point_group or "C1",
            sigma=symmetry_number,
            rotconsts_cm1=th["rotconsts_cm1"],
            use_qrrho=th["qrrho"],
            cutoff_cm1=th["cutoff_cm1"],
            qrrho_ref_cm1=th["qrrho_ref_cm1"],
            qrrho_alpha=th["qrrho_alpha"],
            E_el_Eh=E_h,
            ZPE_Eh=th["ZPE_Eh"],
            Evib_corr_Eh=th["Evib_corr_Eh"],
            Erot_Eh=th["Erot_Eh"],
            Etrans_Eh=th["Etrans_Eh"],
            Hcorr_Eh=th["Hcorr_Eh"],
            TS_el_Eh=0.0,
            TS_vib_Eh=th["TS_vib_Eh"],
            TS_rot_Eh=th["TS_rot_Eh"],
            TS_trans_Eh=th["TS_trans_Eh"],
            G_total_Eh=th["G_total_Eh"],
            H_total_Eh=th["H_total_Eh"],
            U_total_Eh=th["U_total_Eh"],
            G_minus_Eel_Eh=th["G_minus_Eel_Eh"],
            rot_entropy_table_Eh=th["rot_table_Eh"],
        )
        wall = time.time() - t_start
        writer.write_termination(wall)
        writer.close()
        return dict(
            converged=None,
            steps=0,
            energy_H=None,
            walltime_s=wall,
            model=model,
            device=dev,
            out=out_path,
            freq=True,
            freq_only=True,
        )

    # --- Optimization path ---
    Opt = OPTIMIZERS[optimizer]
    opt_kwargs: Dict = {}
    if maxstep is not None:
        opt_kwargs["maxstep"] = maxstep
    if optimizer == "FIRE" and damp is not None:
        opt_kwargs["damp"] = damp

    ase_fmax = 1e-12  # manage our own convergence
    dyn = Opt(atoms, trajectory=None, logfile=None, **opt_kwargs)

    cuts = gaussian_cutoffs(mode=mode)
    prev_pos = None
    converged = False
    steps = 0

    for cycle, _ in enumerate(dyn.irun(fmax=ase_fmax, steps=maxcycles), start=1):
        steps = cycle
        forces = atoms.get_forces()  # eV/Å
        grms, gmax = force_metrics_HB(forces)  # Hartree/Bohr
        curr_pos = atoms.get_positions()  # Å
        drms, dmax = disp_metrics_bohr(prev_pos, curr_pos)  # Bohr
        E_h = atoms.get_potential_energy() * HARTREE_PER_EV  # Hartree

        writer.write_cycle(cycle, atoms, E_h, grms, gmax, drms, dmax, cuts)

        if gaussian_converged(grms, gmax, drms, dmax, cuts):
            converged = True
            break

        prev_pos = curr_pos.copy()

    wall = time.time() - t_start
    writer.write_final_geom_and_energy(atoms, E_h, converged)

    # Frequencies + Thermochemistry as requested
    if do_freq:
        freqs = run_frequencies_and_write(
            writer, atoms, delta=freq_delta, nfree=freq_nfree, scale=freq_scale
        )

        used_scale = thermo_scale if (thermo_scale is not None) else freq_scale
        freqs_th = [f * used_scale for f in freqs]
        th = rrho_thermo(
            atoms,
            freqs_th,
            temp,
            pressure_atm,
            symmetry_number,
            E_h,
            qrrho=qrrho,
            cutoff_cm1=cutoff_cm1,
            qrrho_ref_cm1=qrrho_ref_cm1,
            qrrho_alpha=qrrho_alpha,
        )
        writer.write_thermochemistry(
            T=temp,
            P_atm=pressure_atm,
            mass_amu=th["mass_amu"],
            point_group=point_group or "C1",
            sigma=symmetry_number,
            rotconsts_cm1=th["rotconsts_cm1"],
            use_qrrho=th["qrrho"],
            cutoff_cm1=th["cutoff_cm1"],
            qrrho_ref_cm1=th["qrrho_ref_cm1"],
            qrrho_alpha=th["qrrho_alpha"],
            E_el_Eh=E_h,
            ZPE_Eh=th["ZPE_Eh"],
            Evib_corr_Eh=th["Evib_corr_Eh"],
            Erot_Eh=th["Erot_Eh"],
            Etrans_Eh=th["Etrans_Eh"],
            Hcorr_Eh=th["Hcorr_Eh"],
            TS_el_Eh=0.0,
            TS_vib_Eh=th["TS_vib_Eh"],
            TS_rot_Eh=th["TS_rot_Eh"],
            TS_trans_Eh=th["TS_trans_Eh"],
            G_total_Eh=th["G_total_Eh"],
            H_total_Eh=th["H_total_Eh"],
            U_total_Eh=th["U_total_Eh"],
            G_minus_Eel_Eh=th["G_minus_Eel_Eh"],
            rot_entropy_table_Eh=th["rot_table_Eh"],
        )

    # Termination & time
    writer.write_termination(wall)
    writer.close()

    res = dict(
        converged=converged,
        steps=steps,
        energy_H=float(E_h),
        walltime_s=wall,
        cutoffs=cuts.__dict__,
        mode=mode,
        optimizer=optimizer,
        model=model,
        device=dev,
        out=out_path,
        freq=do_freq,
        freq_only=False,
        T=temp,
        P_atm=pressure_atm,
        sigma=symmetry_number,
    )
    LOG.info(
        "Result: converged=%s steps=%d wall=%.1fs E(H)=%.6f",
        converged,
        steps,
        res["walltime_s"],
        res["energy_H"],
    )
    return res


# -------------------------
# CLI
# -------------------------
def main():
    p = argparse.ArgumentParser(
        description="OMol optimizer (UMA-M-1p1 + ASE) with ORCA-style output, optional frequencies & thermochemistry"
    )
    p.add_argument("--xyz", required=True, help="Input XYZ file")
    p.add_argument("--out", default="opt.out", help="ORCA-style output file")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--mult", type=int, default=1, help="Spin multiplicity (2S+1)")
    p.add_argument("--mode", type=mode_type, default=mode_type("Normal"))
    p.add_argument("--optimizer", type=optimizer_type, default=optimizer_type("LBFGS"))
    p.add_argument("--maxcycles", type=int, default=300)
    p.add_argument("--maxstep", type=float, default=None)
    p.add_argument("--damp", type=float, default=None, help="FIRE damping")
    p.add_argument("--model", default="uma-m-1p1")
    p.add_argument("--device", type=device_type, default=device_type("cuda"))
    p.add_argument(
        "--cache-dir",
        default=DEFAULT_FAIRCHEM_CACHE,
        help=f"Model cache dir (default: {DEFAULT_FAIRCHEM_CACHE})",
    )
    p.add_argument(
        "--use-local-scratch",
        action="store_true",
        help="Stage cache to $TMPDIR (node-local) for speed",
    )
    # frequencies
    p.add_argument(
        "--freq",
        action="store_true",
        help="Run a frequency calculation after optimization",
    )
    p.add_argument(
        "--freq-only",
        action="store_true",
        help="Skip optimization; run frequencies on input geometry",
    )
    p.add_argument(
        "--freq-delta",
        type=float,
        default=0.01,
        help="Finite-difference step in Å (default 0.01)",
    )
    p.add_argument(
        "--freq-nfree",
        type=int,
        default=2,
        help="nfree for Vibrations (2=central difference)",
    )
    p.add_argument(
        "--freq-scale",
        type=float,
        default=1.0,
        help="Printed scaling factor in VIBRATIONAL FREQUENCIES and applied to printed freqs",
    )
    # thermochemistry
    p.add_argument(
        "--temp",
        type=float,
        default=298.15,
        help="Thermochemistry temperature (K)",
    )
    p.add_argument(
        "--pressure-atm",
        type=float,
        default=1.00,
        help="Thermochemistry pressure (atm)",
    )
    p.add_argument(
        "--symmetry-number",
        type=int,
        default=1,
        help="Rotational symmetry number σ for thermochemistry",
    )
    p.add_argument(
        "--point-group",
        type=str,
        default=None,
        help="Point group label for printing (e.g., C1, C2v)",
    )
    p.add_argument(
        "--thermo-scale",
        type=float,
        default=None,
        help="Optional separate scaling factor for thermochemistry; default = freq-scale",
    )
    # qRRHO controls
    p.add_argument(
        "--qrrho", dest="qrrho", action="store_true", help="Use Quasi-RRHO (default)"
    )
    p.add_argument("--no-qrrho", dest="qrrho", action="store_false")
    p.set_defaults(qrrho=True)
    p.add_argument(
        "--cutoff-cm1",
        type=float,
        default=None,
        help="Thermo cutoff frequency (cm^-1); default 1 for qRRHO, 35 for RRHO",
    )
    p.add_argument(
        "--qrrho-ref-cm1",
        type=float,
        default=100.0,
        help="QRRHO reference frequency ω0 (cm^-1)",
    )
    p.add_argument(
        "--qrrho-alpha", type=float, default=4.0, help="QRRHO damping exponent α"
    )
    # logging
    p.add_argument("--verbose", action="store_true", help="Verbose logging (INFO)")
    p.add_argument("--debug", action="store_true", help="Debug logging (DEBUG)")
    args = p.parse_args()

    setup_logging(verbose=args.verbose, debug=args.debug)
    LOG.info("CLI args: %s", vars(args))
    LOG.info(
        "ENV: TMPDIR=%s | CUDA_VISIBLE_DEVICES=%s | HF_HUB_OFFLINE=%s",
        os.environ.get("TMPDIR"),
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        os.environ.get("HF_HUB_OFFLINE"),
    )

    res = optimize_xyz(
        args.xyz,
        out_path=args.out,
        charge=args.charge,
        multiplicity=args.mult,
        model=args.model,
        device=args.device,
        mode=args.mode,
        maxcycles=args.maxcycles,
        optimizer=args.optimizer,
        maxstep=args.maxstep,
        damp=args.damp,
        cache_dir=args.cache_dir,
        use_local_scratch=args.use_local_scratch,
        do_freq=args.freq,
        freq_only=args.freq_only,
        freq_delta=args.freq_delta,
        freq_nfree=args.freq_nfree,
        freq_scale=args.freq_scale,
        temp=args.temp,
        pressure_atm=args.pressure_atm,
        symmetry_number=args.symmetry_number,
        point_group=args.point_group,
        thermo_scale=args.thermo_scale,
        qrrho=args.qrrho,
        cutoff_cm1=args.cutoff_cm1,
        qrrho_ref_cm1=args.qrrho_ref_cm1,
        qrrho_alpha=args.qrrho_alpha,
    )
    print("\nRESULT:", res)


if __name__ == "__main__":
    main()
