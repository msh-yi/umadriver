# umadriver/ensemble.py
from __future__ import annotations
import os, csv, time, logging, glob, shutil, queue
from dataclasses import dataclass
from typing import Optional, List, Literal, Tuple, Dict
from pathlib import Path
import multiprocessing as mp

import numpy as np
from ase.io import read as ase_read, write as ase_write
from ase import Atoms

from .types import SellaOpts

try:
    import cctk  # optional; not required for this workflow
except Exception:
    cctk = None

from .constants import (
    HARTREE_PER_EV,
    EV_PER_HARTREE,
    KCAL_PER_MOL_PER_EV,
)
from .utils import make_job_scratch
from .optimize import (
    gaussian_cutoffs,
    gaussian_converged,
    force_metrics_HB,
    disp_metrics_bohr,
    resolve_device,
    build_calculator,
)
from .vib_thermo import (
    run_frequencies_and_write,  # supports ts=... and scratch_dir=...
    rrho_thermo,
)
from .writer import ORCAWriter

LOG = logging.getLogger("uma.ensemble")


# -----------------------
# Utilities
# -----------------------
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _gaussian_write_gjf(
    path: str,
    atoms: Atoms,
    charge: int,
    mult: int,
    route: str,
    mem: str = "16GB",
    nproc: str = "8",
):
    """
    Minimal Gaussian input writer.
    """
    chk = os.path.splitext(os.path.basename(path))[0] + ".chk"
    lines = []
    lines.append(f"%chk={chk}")
    lines.append(f"%mem={mem}")
    lines.append(f"%nprocshared={nproc}")
    lines.append(f"# {route}")
    lines.append("")  # blank
    lines.append(os.path.basename(path))
    lines.append("")  # blank
    lines.append(f"{charge} {mult}")
    syms = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    for s, (x, y, z) in zip(syms, pos):
        lines.append(f" {s:<2s}   {x: .8f}  {y: .8f}  {z: .8f}")
    lines.append("\n")  # Gaussian likes an ending blank line
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _sp_energy_Eh(atoms: Atoms) -> float:
    e_eV = float(atoms.get_potential_energy())
    return e_eV * HARTREE_PER_EV


# -----------------------
# Optimizers
# -----------------------
def _minimize_atoms_inplace(
    atoms: Atoms,
    optimizer: Literal[
        "LBFGS", "BFGS", "BFGSLineSearch", "FIRE", "QuasiNewton", "Sella"
    ],
    opt_mode: Literal["Loose", "Normal", "Tight", "VeryTight"],
    maxcycles: int,
    sella: Optional[SellaOpts] = None,
) -> Tuple[bool, int, float]:
    if optimizer == "Sella":
        from sella import Sella

        s = sella or SellaOpts()
        s.order = 0 if s.order is None else s.order
        LOG.info(
            "  [OPT] Sella (order=%d, internal=%s, eta=%.3g, gamma=%.3g, delta0=%.3g) — maxcycles=%d",
            s.order,
            s.internal,
            s.eta,
            s.gamma,
            s.delta0,
            maxcycles,
        )
        opt = Sella(
            atoms,
            order=s.order,
            internal=s.internal,
            eta=s.eta,
            gamma=s.gamma,
            delta0=s.delta0,
        )
        cuts = gaussian_cutoffs(mode=opt_mode)
        prev_pos = None
        converged, steps = False, 0
        t_start = time.perf_counter()
        for i in range(1, maxcycles + 1):
            steps = i
            opt.step()
            forces = atoms.get_forces()
            grms, gmax = force_metrics_HB(forces)
            curr_pos = atoms.get_positions()
            drms, dmax = disp_metrics_bohr(prev_pos, curr_pos)
            if gaussian_converged(grms, gmax, drms, dmax, cuts):
                converged = True
                break
            prev_pos = curr_pos.copy()
        wall = time.perf_counter() - t_start
        E_h = _sp_energy_Eh(atoms)
        LOG.info(
            "  [OPT] Done: conv=%s | steps=%d | E=%.8f Eh | wall=%.2fs",
            converged,
            steps,
            E_h,
            wall,
        )
        return converged, steps, E_h

    else:
        from ase.optimize import LBFGS, BFGS, BFGSLineSearch, FIRE, QuasiNewton

        OPT = {
            "LBFGS": LBFGS,
            "BFGS": BFGS,
            "BFGSLineSearch": BFGSLineSearch,
            "FIRE": FIRE,
            "QuasiNewton": QuasiNewton,
        }[optimizer]
        LOG.info(
            "  [OPT] %s (opt_mode=%s) — maxcycles=%d", optimizer, opt_mode, maxcycles
        )
        cuts = gaussian_cutoffs(mode=opt_mode)
        ase_fmax = 1e-12
        dyn = OPT(atoms, trajectory=None, logfile=None)
        prev_pos = None
        converged, steps = False, 0
        t_start = time.perf_counter()
        for i, _ in enumerate(dyn.irun(fmax=ase_fmax, steps=maxcycles), start=1):
            steps = i
            forces = atoms.get_forces()
            grms, gmax = force_metrics_HB(forces)
            curr_pos = atoms.get_positions()
            drms, dmax = disp_metrics_bohr(prev_pos, curr_pos)
            if gaussian_converged(grms, gmax, drms, dmax, cuts):
                converged = True
                break
            prev_pos = curr_pos.copy()
        wall = time.perf_counter() - t_start
        E_h = _sp_energy_Eh(atoms)
        LOG.info(
            "  [OPT] Done: conv=%s | steps=%d | E=%.8f Eh | wall=%.2fs",
            converged,
            steps,
            E_h,
            wall,
        )
        return converged, steps, E_h


def _ts_optimize_atoms_inplace(
    atoms: Atoms, maxcycles: int, sella: Optional[SellaOpts] = None
) -> Tuple[bool, int, float]:
    from sella import Sella

    s = sella or SellaOpts()
    s.order = 1
    LOG.info(
        "  [TS ] Sella (order=1, internal=%s, eta=%.3g, gamma=%.3g, delta0=%.3g) — maxcycles=%d",
        s.internal,
        s.eta,
        s.gamma,
        s.delta0,
        maxcycles,
    )
    opt = Sella(
        atoms,
        order=s.order,
        internal=s.internal,
        eta=s.eta,
        gamma=s.gamma,
        delta0=s.delta0,
    )
    cuts = gaussian_cutoffs(mode="Normal")
    prev_pos = None
    converged, steps = False, 0
    t_start = time.perf_counter()
    for i in range(1, maxcycles + 1):
        steps = i
        opt.step()
        forces = atoms.get_forces()
        grms, gmax = force_metrics_HB(forces)
        curr_pos = atoms.get_positions()
        drms, dmax = disp_metrics_bohr(prev_pos, curr_pos)
        if gaussian_converged(grms, gmax, drms, dmax, cuts):
            converged = True
            break
        prev_pos = curr_pos.copy()
    wall = time.perf_counter() - t_start
    E_h = _sp_energy_Eh(atoms)
    LOG.info(
        "  [TS ] Done: conv=%s | steps=%d | E=%.8f Eh | wall=%.2fs",
        converged,
        steps,
        E_h,
        wall,
    )
    return converged, steps, E_h


# -----------------------
# Worker processes
# -----------------------
def _bind_gpu_env(gpu_id: int, scratch_root: str):
    # Bind to a single GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Per-GPU scratch root for ASE vib cache etc.
    os.environ.setdefault(
        "UMA_SCRATCH_ROOT", os.path.join(scratch_root, f"_gpu{gpu_id}")
    )


def _rebuild_atoms(
    symbols: List[str], positions: np.ndarray, charge: int, mult: int
) -> Atoms:
    a = Atoms(symbols=symbols, positions=positions, pbc=False)
    a.info.update({"charge": charge, "spin": mult})
    return a


def run_conformer_workflow(
    xyz_path: str,
    out_dir: str,
    *,
    charge: int = 0,
    mult: int = 1,
    model: str = "uma-m-1p1",
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    use_local_scratch: bool = False,
    # what to do
    optimizer: Optional[
        Literal["LBFGS", "BFGS", "BFGSLineSearch", "FIRE", "QuasiNewton", "Sella"]
    ] = None,
    opt_mode: Literal["Loose", "Normal", "Tight", "VeryTight"] = "Normal",
    optts: bool = False,
    maxcycles: int = 300,
    # frequency / thermo
    do_freq: bool = False,
    freq_delta: float = 0.01,
    freq_nfree: int = 2,
    freq_scale: float = 1.0,
    temp: float = 298.15,
    pressure_atm: float = 1.0,
    symmetry_number: int = 1,
    point_group: Optional[str] = None,
    # qRRHO
    qrrho: bool = True,
    cutoff_cm1: Optional[float] = None,
    qrrho_ref_cm1: float = 100.0,
    qrrho_alpha: float = 4.0,
    # solvation
    solv: Optional[str] = None,
    gauss_mem: str = "16GB",
    gauss_nproc: str = "8",
    # Sella controls
    sella_internal: bool = True,
    sella_eta: float = 2e-2,
    sella_gamma: float = 1e-4,
    sella_delta0: float = 0.02,
    # IRC
    irc: bool = False,
    irc_dx: float = 0.1,
    # If provided by batch layer we’ll use it; otherwise build once here
    calc: Optional[object] = None,
) -> str:
    """
    Serial conformer workflow (ensemble-level parallelism happens in batch.py):
      (1) OPT/SP/TS — serial over conformers
      (2) Write Gaussian inputs (original order)
      (3) FREQ — serial over conformers (original order)
      (4) IRC — serial (original order)
    Returns energies.csv path.
    """
    t0 = time.perf_counter()
    _ensure_dir(out_dir)

    LOG.info("=== Ensemble workflow (serial) ===")
    LOG.info("Input: %s", xyz_path)
    LOG.info("Outdir: %s", out_dir)

    frames: List[Atoms] = ase_read(xyz_path, index=":")
    n = len(frames)
    if n == 0:
        raise RuntimeError("No frames found in input XYZ.")
    LOG.info("Loaded %d conformers", n)

    # Decide route
    if optts:
        route_kind = "TS"
    elif optimizer is None:
        route_kind = "SP"
    else:
        route_kind = "OPT"
    LOG.info("Route kind: %s", route_kind)

    # Scratch for vibrations etc.
    job_tag = os.path.splitext(os.path.basename(xyz_path))[0]
    scratch_root = os.environ.get("UMA_SCRATCH_ROOT", cache_dir or out_dir)
    job_scratch = make_job_scratch(scratch_root, f"ensemble-{job_tag}")
    _ensure_dir(job_scratch)
    LOG.info("Scratch: %s", job_scratch)

    # Build calculator once if not provided by caller
    if calc is None:
        dev = resolve_device(device)
        LOG.info(
            "Calculator: model=%s device=%s cache=%s",
            model,
            dev,
            cache_dir or "<default>",
        )
        calc = build_calculator(
            model=model,
            device=dev,
            cache_dir=cache_dir,
            use_local_scratch=use_local_scratch,
        )
    else:
        # Best-effort device label for banners
        dev = device

    # Sella options
    sella = SellaOpts(
        internal=sella_internal,
        order=(1 if optts else 0),
        eta=sella_eta,
        gamma=sella_gamma,
        delta0=sella_delta0,
    )

    # -----------------------
    # PHASE 1: OPT/SP/TS (serial; ORIGINAL ORDER)
    # -----------------------
    LOG.info("PHASE 1: Starting OPT/SP/TS for %d conformers", n)
    conformers: List[Dict] = []
    for idx, src in enumerate(frames):
        tag = f"conf_{idx:04d}"
        LOG.info("--- Conformer %d/%d | tag=%s ---", idx + 1, n, tag)

        a = src.copy()
        a.pbc = False
        a.info.update({"charge": charge, "spin": mult})
        a.calc = calc

        if route_kind == "TS":
            converged, steps, E_h = _ts_optimize_atoms_inplace(
                a, maxcycles=maxcycles, sella=sella
            )
        elif route_kind == "SP":
            t_sp = time.perf_counter()
            E_h = _sp_energy_Eh(a)
            steps, converged = 0, True
            LOG.info("  [SP ] E=%.8f Eh | wall=%.2fs", E_h, time.perf_counter() - t_sp)
        else:
            converged, steps, E_h = _minimize_atoms_inplace(
                a,
                optimizer=optimizer,
                opt_mode=opt_mode,
                maxcycles=maxcycles,
                sella=sella if optimizer == "Sella" else None,
            )

        conformers.append(
            dict(
                index=idx,
                tag=tag,
                atoms=a,
                route=route_kind,
                converged=bool(converged),
                steps=int(steps),
                energy_Eh=float(E_h),
                energy_kcal=float(E_h * EV_PER_HARTREE * KCAL_PER_MOL_PER_EV),
                gibbs_Eh=None,
                gibbs_kcal=None,
                n_imag=None,
                imag_ok=None,
            )
        )
        LOG.info("  Summary: E=%.8f Eh | converged=%s | steps=%d", E_h, converged, steps)
    LOG.info("PHASE 1: Completed.")

    # Rank for reporting (keep original order for subsequent steps)
    LOG.info("Ranking by electronic energy …")
    results_sorted = sorted(conformers, key=lambda r: r["energy_Eh"])
    e0 = results_sorted[0]["energy_Eh"]
    for r in results_sorted:
        r["rel_kcal"] = (r["energy_Eh"] - e0) * EV_PER_HARTREE * KCAL_PER_MOL_PER_EV

    ranked_xyz_path = os.path.join(out_dir, "optimized_ranked.xyz")
    ase_write(ranked_xyz_path, [r["atoms"] for r in results_sorted], format="xyz", parallel=False)
    LOG.info("Wrote ranked XYZ: %s", ranked_xyz_path)

    # -----------------------
    # PHASE 2: Gaussian inputs (ORIGINAL ORDER)
    # -----------------------
    if solv:
        gjf_dir = os.path.join(out_dir, "gaussian")
        _ensure_dir(gjf_dir)
        LOG.info("PHASE 2: Writing Gaussian inputs (original order) …")
        for r in conformers:
            a, tag = r["atoms"], r["tag"]
            if a is None:
                LOG.warning("  [GJF] %s skipped (no structure)", tag)
                continue
            route_gas = "M052X/6-31G*"
            route_smd = f"M052X/6-31G* scrf(SMD,solvent={solv})"
            _gaussian_write_gjf(os.path.join(gjf_dir, f"{tag}_gas.gjf"), a, charge, mult, route_gas, gauss_mem, gauss_nproc)
            _gaussian_write_gjf(os.path.join(gjf_dir, f"{tag}_smd.gjf"), a, charge, mult, route_smd, gauss_mem, gauss_nproc)
            LOG.info("  [GJF] %s done", tag)
        LOG.info("PHASE 2: Completed.")

    # -----------------------
    # PHASE 3: Frequencies (serial; ORIGINAL ORDER)
    # -----------------------
    if do_freq:
        freq_dir = os.path.join(out_dir, "freq_out")
        _ensure_dir(freq_dir)
        LOG.info("PHASE 3: Frequencies (original order) …")
        for r in conformers:
            if r["atoms"] is None:
                continue
            a, tag, route_kind, E_h = r["atoms"], r["tag"], r["route"], r["energy_Eh"]
            out_path = os.path.join(freq_dir, f"{tag}.out")
            LOG.info("  [FRQ] %s → %s", tag, out_path)
            writer = ORCAWriter(out_path, xyz_path, model, dev, opt_banner=False)
            try:
                freqs = run_frequencies_and_write(
                    writer,
                    a,
                    delta=freq_delta,
                    nfree=freq_nfree,
                    scale=freq_scale,
                    scratch_dir=os.path.join(job_scratch, tag),
                    ts=(route_kind == "TS"),
                )
                n_imag = sum(1 for f in freqs if f < 0.0)
                imag_ok = n_imag == (1 if route_kind == "TS" else 0)

                th = rrho_thermo(
                    a,
                    [f * freq_scale for f in freqs],
                    temp,
                    pressure_atm,
                    symmetry_number,
                    E_h,
                    qrrho=qrrho,
                    cutoff_cm1=cutoff_cm1,
                    qrrho_ref_cm1=qrrho_ref_cm1,
                    qrrho_alpha=qrrho_alpha,
                )
                G_Eh = th["G_total_Eh"]

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
                writer.write_termination(0.0)
                writer.close()

                r["gibbs_Eh"]   = float(G_Eh)
                r["gibbs_kcal"] = float(G_Eh * EV_PER_HARTREE * KCAL_PER_MOL_PER_EV)
                r["n_imag"]     = int(n_imag)
                r["imag_ok"]    = bool(imag_ok)

            except Exception as e:
                try: writer.close()
                except Exception: pass
                LOG.exception("  [FRQ] %s failed: %s", tag, e)
        LOG.info("PHASE 3: Completed.")

    # -----------------------
    # PHASE 4: IRC (serial; ORIGINAL ORDER)
    # -----------------------
    if irc:
        irc_dir = os.path.join(out_dir, "irc")
        _ensure_dir(irc_dir)
        LOG.info("PHASE 4: IRC (serial) …")
        from .irc import run_irc_trajectories
        for r in conformers:
            if r["atoms"] is None:
                continue
            a, tag, route_kind = r["atoms"], r["tag"], r["route"]
            run_this = False
            if optts:
                if do_freq:
                    run_this = (r.get("imag_ok", None) is True)
                else:
                    run_this = True
            else:
                run_this = True
            if not run_this:
                LOG.info("  [IRC] %s: skip (not TS with 1 imaginary)", tag)
                continue
            try:
                run_irc_trajectories(
                    a,
                    tag=tag,
                    out_dir=irc_dir,
                    dx=irc_dx,
                    sella_opts=SellaOpts(
                        internal=sella_internal,
                        order=1,
                        eta=sella_eta,
                        gamma=sella_gamma,
                        delta0=sella_delta0,
                    ),
                )
                LOG.info("  [IRC] %s done.", tag)
            except Exception as e:
                LOG.exception("  [IRC] %s failed: %s", tag, e)

    # -----------------------
    # CSV (ranked), with freq cols pulled from original rows
    # -----------------------
    csv_path = os.path.join(out_dir, "energies.csv")
    fieldnames = [
        "rank","index","tag","route","converged","steps",
        "energy_Eh","energy_kcal","rel_kcal","gibbs_Eh","gibbs_kcal","n_imag","imag_ok",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rank, r in enumerate(results_sorted, start=1):
            # find original row by tag to get any freq results
            match = next((x for x in conformers if x["tag"] == r["tag"]), r)
            w.writerow(dict(
                rank=rank,
                index=r["index"],
                tag=r["tag"],
                route=r["route"],
                converged=r["converged"],
                steps=r["steps"],
                energy_Eh=r["energy_Eh"],
                energy_kcal=r["energy_kcal"],
                rel_kcal=r["rel_kcal"],
                gibbs_Eh=match["gibbs_Eh"],
                gibbs_kcal=match["gibbs_kcal"],
                n_imag=match["n_imag"],
                imag_ok=match["imag_ok"],
            ))
    LOG.info("Wrote energies CSV: %s", csv_path)
    LOG.info("Ensemble workflow complete in %.2fs", time.perf_counter() - t0)
    return csv_path

