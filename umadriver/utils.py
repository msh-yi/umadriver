from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple, Dict, Literal
import argparse, logging, os, time, shutil, subprocess, math, getpass

import numpy as np
from ase import Atoms
from ase.io import read

from .constants import (
    BASE, GAUSS_RMS_FORCE, EV_A_to_HB, BOHR_PER_ANG,
    DEFAULT_FAIRCHEM_CACHE, LOCAL_SCRATCH_DEFAULT
)

LOG = logging.getLogger("omol_opt")

# ---------- Logging / env ----------
def setup_logging(verbose: bool = False, debug: bool = False):
    level = logging.INFO
    if verbose:
        level = logging.INFO
    if debug:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("fairchem").setLevel(logging.WARNING if not debug else logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def initialize_env():
    # Threads hygiene
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Stay offline only if cache exists
    if cache_has_files(DEFAULT_FAIRCHEM_CACHE):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# ---------- Cache and calculator ----------
def cache_has_files(path: str, min_bytes: int = 1 << 20) -> bool:
    """True only if `path` holds a real, readable model blob.

    A non-empty directory is not enough. The HF layout is snapshots/ full of
    symlinks into blobs/, and scratch purges delete the blobs while leaving the
    symlink tree behind — so the cache looks populated but every checkpoint is a
    dangling link. Resolving one real file of nontrivial size is the honest check,
    and it decides whether we can safely go offline.
    """
    if not os.path.isdir(path):
        return False
    for root, _dirs, files in os.walk(path):
        for name in files:  # os.walk omits dangling symlinks from `files`
            try:
                if os.path.getsize(os.path.join(root, name)) >= min_bytes:
                    return True
            except OSError:
                continue
    return False

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
    from fairchem.core import pretrained_mlip
    cache_dir = cache_dir_key or None
    LOG.info("UMA lookup: model=%s device=%s cache=%s", model, device_opt, cache_dir or "<default>")
    t0 = time.time()
    try:
        pred = pretrained_mlip.get_predict_unit(model, device=device_opt, cache_dir=cache_dir)
    except TypeError:
        LOG.debug("fairchem.get_predict_unit lacks cache_dir; retrying without.")
        pred = pretrained_mlip.get_predict_unit(model, device=device_opt)
    LOG.info("[UMA init] model=%s device=%s took %.2fs", model, device_opt or "auto", time.time() - t0)
    return pred


def make_job_scratch(root: str, tag: str) -> str:
    """
    Create a unique per-job scratch directory under `root`.
    Examples:
      root=/n/netscratch/jacobsen_lab/Lab/msak
      tag="water" -> /n/.../msak/water-20250905_154455-12345
    """
    tstamp = time.strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    user = getpass.getuser()
    # keep the tag short & safe
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in tag)[:48]
    path = os.path.join(root, f"{safe}-{tstamp}-{pid}")
    os.makedirs(path, exist_ok=True)
    return path


def build_calculator(
    model="uma-m-1p1",
    device: Optional[str] = "cuda",
    cache_dir: Optional[str] = None,
    use_local_scratch=False,
):
    from fairchem.core import FAIRChemCalculator
    base = cache_dir or DEFAULT_FAIRCHEM_CACHE
    resolved = stage_cache_to_local(base, LOCAL_SCRATCH_DEFAULT) if use_local_scratch else base
    key = os.path.abspath(resolved) if resolved else ""
    predictor = _get_predict_cached(model, device, key)
    return FAIRChemCalculator(predictor, task_name="omol")

# ---------- Parsers for argparse ----------
def _normkey(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())

def device_type(s: str) -> str:
    key = _normkey(s)
    mapping = {"cuda": "cuda", "gpu": "cuda", "cpu": "cpu", "auto": "cuda"}
    if key in mapping: return mapping[key]
    raise ValueError(f"Unknown device '{s}'. Valid: cpu, cuda (gpu).")

def mode_type(s: str) -> str:
    key = _normkey(s)
    mapping = {"loose":"Loose","normal":"Normal","tight":"Tight","verytight":"VeryTight"}
    if key in mapping: return mapping[key]
    raise ValueError("Unknown mode. Valid: Loose, Normal, Tight, VeryTight.")

# utils.py

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
        "sella": "Sella",   # <— add this line
    }
    if key in mapping:
        return mapping[key]
    raise argparse.ArgumentTypeError(
        "Unknown optimizer. Valid: LBFGS, BFGS, BFGSLineSearch, FIRE, QuasiNewton, Sella."
    )


# ---------- Convergence helpers ----------
@dataclass
class GaussCutoffs:
    grms: float
    gmax: float
    drms: float
    dmax: float

def gaussian_cutoffs(mode: Literal["Loose","Normal","Tight","VeryTight"] = "Normal") -> GaussCutoffs:
    target = GAUSS_RMS_FORCE[mode]
    scale = target / BASE["GRMS"]
    return GaussCutoffs(BASE["GRMS"]*scale, BASE["GMAX"]*scale, BASE["DRMS"]*scale, BASE["DMAX"]*scale)

def gaussian_converged(grms, gmax, drms, dmax, cuts: GaussCutoffs) -> bool:
    return (grms < cuts.grms) and (gmax < cuts.gmax) and (drms < cuts.drms) and (dmax < cuts.dmax)

# ---------- Generic helpers ----------
def load_xyz(path: str) -> Atoms:
    LOG.info("Loading XYZ: %s", path)
    atoms = read(path)
    atoms.pbc = False
    LOG.info("Loaded natoms=%d", len(atoms))
    return atoms

def set_charge_mult(atoms: Atoms, charge: int = 0, multiplicity: int = 1):
    atoms.info.update({"charge": charge, "spin": multiplicity})

def force_metrics_HB(forces_evA: np.ndarray) -> tuple[float, float]:
    mags = np.linalg.norm(forces_evA, axis=1) * EV_A_to_HB
    return float(np.sqrt((mags**2).mean())), float(mags.max())


def project_out_rigid_body(positions_A: np.ndarray, forces_evA: np.ndarray) -> np.ndarray:
    """Remove the net-force and net-torque components of a force array.

    An isolated molecule's energy is invariant to translation and rotation, so
    those components of the gradient are numerical residue: they change nothing
    about the structure, and an optimizer working in internal coordinates cannot
    remove them even in principle.

    UMA leaves a small torque behind — on the test water it is 6.0e-5 Eh/Bohr,
    which is four times the Tight max-force cutoff, while the part of the gradient
    that actually deforms the molecule is down at 1.1e-8. Scoring convergence on
    the raw Cartesian forces therefore made `--opt-mode Tight` unreachable for that
    molecule no matter how many cycles it was given.
    """
    pos = np.asarray(positions_A, dtype=float)
    n = len(pos)
    if n < 2:
        return np.zeros_like(forces_evA)

    d = pos - pos.mean(axis=0)
    cols = []
    for k in range(3):
        v = np.zeros((n, 3))
        v[:, k] = 1.0
        cols.append(v.ravel())
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1.0
        cols.append(np.cross(e, d).ravel())

    B = np.array(cols).T
    U, s, _ = np.linalg.svd(B, full_matrices=False)
    U = U[:, s > 1e-8 * s[0]] if s[0] > 0 else U[:, :0]

    f = np.asarray(forces_evA, dtype=float).ravel()
    return (f - U @ (U.T @ f)).reshape(forces_evA.shape)


def internal_force_metrics_HB(
    positions_A: np.ndarray, forces_evA: np.ndarray
) -> tuple[float, float]:
    """``force_metrics_HB`` on the forces that actually deform the molecule."""
    return force_metrics_HB(project_out_rigid_body(positions_A, forces_evA))

def disp_metrics_bohr(prev_pos_A: Optional[np.ndarray], curr_pos_A: np.ndarray) -> tuple[float, float]:
    if prev_pos_A is None:
        return 0.0, 0.0
    mags = np.linalg.norm(curr_pos_A - prev_pos_A, axis=1) * BOHR_PER_ANG
    return float(np.sqrt((mags**2).mean())), float(mags.max())

@lru_cache(maxsize=None)
def resolve_device(prefer: str = "cuda") -> str:
    # Cached: device availability (and the nvidia-smi probe below) is constant for
    # the life of the process, so we only shell out / log once per `prefer` value
    # instead of on every calculator build.
    if prefer == "cuda":
        try:
            import torch
            have = torch.cuda.is_available()
            LOG.info("CUDA available: %s | CUDA_VISIBLE_DEVICES=%s", have, os.environ.get("CUDA_VISIBLE_DEVICES"))
            if have:
                try:
                    out = subprocess.run(["nvidia-smi","-L"], capture_output=True, text=True, timeout=2)
                    if out.returncode == 0:
                        LOG.info("nvidia-smi -L: %s", out.stdout.strip().replace("\n"," | "))
                except Exception:
                    pass
                return "cuda"
            LOG.warning("CUDA requested but unavailable; falling back to CPU.")
        except Exception as e:
            LOG.warning("CUDA check failed (%s); falling back to CPU.", e)
        return "cpu"
    return "cpu"
