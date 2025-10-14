from __future__ import annotations
import os

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

# Gaussian-style base thresholds (au)
BASE = dict(
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

# -------------------------
# Caching (VAST default)
# -------------------------
VAST_BASE = os.environ.get("UMA_CACHE_BASE", "/n/netscratch/jacobsen_lab/Lab/msak")
DEFAULT_FAIRCHEM_CACHE = os.path.join(VAST_BASE, "fairchem_cache")
LOCAL_SCRATCH_DEFAULT = os.path.join(
    os.environ.get("TMPDIR", "/scratch"), "fairchem_cache"
)
