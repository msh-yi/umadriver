#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Literal
import argparse, time, os, shutil, subprocess, logging
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

# Cache base (VAST)
VAST_BASE = os.environ.get("UMA_CACHE_BASE", "/n/netscratch/jacobsen_lab/Lab/msak")
DEFAULT_FAIRCHEM_CACHE = os.path.join(VAST_BASE, "fairchem_cache")


def cache_has_files(path: str) -> bool:
    return os.path.isdir(path) and any(os.scandir(path))


if cache_has_files(DEFAULT_FAIRCHEM_CACHE):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# -------------------------
# Units & Gaussian-style criteria
# -------------------------
HARTREE_PER_EV = 1.0 / 27.211386245988
BOHR_PER_ANG = 1.0 / 0.529177210903
EV_A_to_HB = HARTREE_PER_EV / BOHR_PER_ANG  # eV/Å -> Hartree/Bohr

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


# -------------------------
# Case-insensitive parsers
# -------------------------
def _normkey(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def device_type(s: str) -> str:
    key = _normkey(s)
    mapping = {
        "cuda": "cuda",
        "gpu": "cuda",
        "cpu": "cpu",
        "auto": "cuda",  # accept 'auto' but map to cuda (your default behavior)
    }
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
        "very_tight": "VeryTight",
        "very-tight": "VeryTight",
    }
    if key in mapping:
        return mapping[key]
    raise argparse.ArgumentTypeError(
        f"Unknown mode '{s}'. Valid: Loose, Normal, Tight, VeryTight."
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
        "quasi_newton": "QuasiNewton",
    }
    if key in mapping:
        return mapping[key]
    raise argparse.ArgumentTypeError(
        f"Unknown optimizer '{s}'. Valid: LBFGS, BFGS, BFGSLineSearch, FIRE, QuasiNewton."
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
# ORCA-style writer
# -------------------------
class ORCAWriter:
    def __init__(self, path: str, xyz_path: str, model: str, device: str):
        from datetime import datetime

        self.path = path
        self.f = open(path, "w", buffering=1)
        # Header exactly like your accepted file
        self._w("                                 *****************")
        self._w("                                 * O   R   C   A *")
        self._w("                                 *****************")
        self._w("")
        self._w(f"OMol/ASE; model={model}; device={device}")
        self._w(
            datetime.now().strftime("Start  : %a %b %d %H:%M:%S  %Y")
        )  # note two spaces before year
        self._w(f"Input  : {xyz_path}")
        self._w("")
        self._w("                       *****************************")
        self._w("                       * Geometry Optimization Run *")
        self._w("                       *****************************")
        self._w("")

    def _w(self, s=""):
        self.f.write(s + ("\n" if not s.endswith("\n") else ""))

    def _cycle_banner(self, n: int):
        left = " " * 9
        stars = "*" * 61  # matches your accepted file
        self._w(f"{left}{stars}")
        title = f"GEOMETRY OPTIMIZATION CYCLE   {n}"
        interior = 61 - 2  # width between the two stars
        prefix_spaces = 16  # puts 'G' at column 27 (9 + 1 + 16 + 1 = 27)
        pad = max(0, interior - prefix_spaces - len(title))
        self._w(f"{left}*{' ' * prefix_spaces}{title}{' ' * pad}*")
        self._w(f"{left}{stars}")

    def _coords_block(self, atoms: Atoms):
        self._w("---------------------------------")
        self._w("CARTESIAN COORDINATES (ANGSTROEM)")
        self._w("---------------------------------")
        syms = atoms.get_chemical_symbols()
        pos = atoms.get_positions()
        for s, (x, y, z) in zip(syms, pos):
            self._w(f"  {s:<3s}{x:16.10f} {y:16.10f} {z:16.10f}")
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

        # "Internals" summary line (viewer seems to expect the block; values can be placeholders)
        self._w("          ........................................................")
        # Use simple placeholders; if you later want real internals, we can wire those up.
        self._w("          Max(Bonds)      0.0000      Max(Angles)    0.00")
        self._w("          Max(Dihed)        0.00      Max(Improp)    0.00")
        self._w(
            "          ---------------------------------------------------------------------"
        )
        self._w("")

    def write_cycle(
        self,
        cycle: int,
        atoms: Atoms,
        energy_h: float,
        grms: float,
        gmax: float,
        drms: float,
        dmax: float,
        cuts,
        dE_h: float,
    ):
        self._cycle_banner(cycle)
        self._coords_block(atoms)
        self._energy_box(energy_h)  # exact FINAL SINGLE POINT ENERGY box per cycle
        self._geom_conv_box(grms, gmax, drms, dmax, cuts)

    def write_final(
        self, atoms: Atoms, energy_h: float, converged: bool, wall_s: float
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
        self._coords_block(atoms)
        self._energy_box(energy_h)
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


def gaussian_converged(grms, gmax, drms, dmax, cuts: GaussCutoffs) -> bool:
    return (
        (grms < cuts.grms)
        and (gmax < cuts.gmax)
        and (drms < cuts.drms)
        and (dmax < cuts.dmax)
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

    writer = ORCAWriter(out_path, xyz_path, model, dev)

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
    prev_E_h = atoms.get_potential_energy() * HARTREE_PER_EV
    steps = 0
    converged = False

    for cycle, _ in enumerate(dyn.irun(fmax=ase_fmax, steps=maxcycles), start=1):
        # you can keep steps = dyn.get_number_of_steps() if you still want it for logs,
        # but use 'cycle' for the banner and any human-facing reporting.
        forces = atoms.get_forces()
        grms, gmax = force_metrics_HB(forces)
        curr_pos = atoms.get_positions()
        drms, dmax = disp_metrics_bohr(prev_pos, curr_pos)
        E_h = atoms.get_potential_energy() * HARTREE_PER_EV
        dE_h = E_h - prev_E_h

        writer.write_cycle(cycle, atoms, E_h, grms, gmax, drms, dmax, cuts, dE_h)

        if gaussian_converged(grms, gmax, drms, dmax, cuts):
            converged = True
            break

        prev_pos = curr_pos.copy()
        prev_E_h = E_h

    wall = time.time() - t_start
    writer.write_final(atoms, E_h, converged, wall)
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
        description="OMol optimizer (UMA-M-1p1 + ASE) writing ORCA-style .out (case-insensitive args)"
    )
    p.add_argument("--xyz", required=True, help="Input XYZ file")
    p.add_argument("--out", default="opt.out", help="ORCA-style output file")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--mult", type=int, default=1, help="Spin multiplicity (2S+1)")
    p.add_argument(
        "--mode",
        type=mode_type,
        default=mode_type("Normal"),
        help="Loose|Normal|Tight|VeryTight (case-insensitive)",
    )
    p.add_argument(
        "--optimizer",
        type=optimizer_type,
        default=optimizer_type("LBFGS"),
        help="LBFGS|BFGS|BFGSLineSearch|FIRE|QuasiNewton (case-insensitive)",
    )
    p.add_argument("--maxcycles", type=int, default=300)
    p.add_argument("--maxstep", type=float, default=None)
    p.add_argument("--damp", type=float, default=None, help="FIRE damping")
    p.add_argument("--model", default="uma-m-1p1")
    p.add_argument(
        "--device",
        type=device_type,
        default=device_type("cuda"),
        help="cpu|cuda (also accepts 'gpu'/'auto')",
    )
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
    )
    print("\nRESULT:", res)


if __name__ == "__main__":
    main()
