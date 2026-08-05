# umadriver/ensemble.py
from __future__ import annotations

import os
import csv
import time
import logging
from typing import Optional, List, Literal, Tuple, Dict, Any

import numpy as np
from ase import Atoms
from ase.io import read as ase_read, write as ase_write

from .types import SellaOpts
from .constants import (
    HARTREE_PER_EV,
    EV_PER_HARTREE,
    KCAL_PER_MOL_PER_EV,
)
from .utils import (
    make_job_scratch,
    gaussian_cutoffs,
    gaussian_converged,
    internal_force_metrics_HB,
    disp_metrics_bohr,
    resolve_device,
    build_calculator,
)
from .vib_thermo import (
    run_frequencies_and_write,
    rrho_thermo,
)
from .writer import ORCAWriter

LOG = logging.getLogger("uma.ensemble")

EH_TO_KCAL = EV_PER_HARTREE * KCAL_PER_MOL_PER_EV

# Column layout of the ranked ensemble energies.csv. Shared with the batch-level
# split-ensemble aggregator so both writers stay in lockstep.
ENERGIES_FIELDS = [
    "rank",
    "index",
    "tag",
    "route",
    "converged",
    "steps",
    "energy_Eh",
    "energy_kcal",
    "rel_kcal",
    "gibbs_Eh",
    "gibbs_kcal",
    "n_imag",
    "imag_ok",
    "solv_corr_kcal",
]


def rank_by_energy(
    rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    """Sort records by electronic energy (finite first) and assign ``rel_kcal``.

    Records are mutated in place (``rel_kcal`` added) and returned in ranked order
    alongside the reference energy ``e0`` (the lowest finite ``energy_Eh``), or
    ``None`` if no finite energy exists. Used by both the per-job workflow and the
    split-ensemble aggregator so ranking has a single source of truth.
    """

    def _sort_key(r: Dict[str, Any]) -> Tuple[int, float]:
        E = _safe_float(r.get("energy_Eh"))
        return (0, E) if np.isfinite(E) else (1, 0.0)

    rows_sorted = sorted(rows, key=_sort_key)

    e0 = None
    for r in rows_sorted:
        E = _safe_float(r.get("energy_Eh"))
        if np.isfinite(E):
            e0 = E
            break

    for r in rows_sorted:
        E = _safe_float(r.get("energy_Eh"))
        r["rel_kcal"] = (
            (E - e0) * EH_TO_KCAL if (e0 is not None and np.isfinite(E)) else float("nan")
        )
    return rows_sorted, e0


# -----------------------
# Utilities
# -----------------------
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _safe_float(x, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def _safe_bool(x, default: bool = False) -> bool:
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    return default


def _Eh_to_kcal(E_h: float) -> float:
    return float(E_h * EH_TO_KCAL)


def _load_attempted_conformers(per_conf_csv: str) -> Dict[str, Dict[str, str]]:
    """
    Load a map tag -> last-seen CSV row for conformers that have been attempted.
    """
    if (not per_conf_csv) or (not os.path.exists(per_conf_csv)):
        return {}

    rows_by_tag: Dict[str, Dict[str, str]] = {}
    try:
        with open(per_conf_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return {}
            for row in reader:
                tag = (row.get("tag") or "").strip()
                if tag:
                    rows_by_tag[tag] = row  # last occurrence wins
    except Exception as e:
        LOG.exception(
            "Failed to read per-conformer CSV for resume (%s): %s", per_conf_csv, e
        )
        return {}

    return rows_by_tag


def _sp_energy_Eh(atoms: Atoms) -> float:
    e_eV = float(atoms.get_potential_energy())
    return e_eV * HARTREE_PER_EV


def _log_unconverged(tag: str, cuts, grms, gmax, drms, dmax) -> None:
    """Name the criteria that ran out of cycles, and by how much.

    "conv=False" on its own says nothing about whether the run was one step away
    or could never have converged at all — the difference between raising
    --maxcycles and picking a different --opt-mode.
    """
    over = [
        f"{name}={value:.2e} (cutoff {cut:.2e}, {value / cut:.1f}x)"
        for name, value, cut in (
            ("GRMS", grms, cuts.grms),
            ("GMAX", gmax, cuts.gmax),
            ("DRMS", drms, cuts.drms),
            ("DMAX", dmax, cuts.dmax),
        )
        if value >= cut
    ]
    LOG.warning(
        "  [%s] Not converged after maxcycles; still outside: %s",
        tag,
        ", ".join(over) if over else "nothing (converged on the final step only)",
    )


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
            curr_pos = atoms.get_positions()
            grms, gmax = internal_force_metrics_HB(curr_pos, forces)
            drms, dmax = disp_metrics_bohr(prev_pos, curr_pos)
            # Require a real displacement measurement before declaring convergence:
            # on step 1 prev_pos is None -> drms/dmax are trivially 0.0.
            if prev_pos is not None and gaussian_converged(
                grms, gmax, drms, dmax, cuts
            ):
                converged = True
                break
            prev_pos = curr_pos.copy()

        if not converged and steps:
            _log_unconverged("OPT", cuts, grms, gmax, drms, dmax)

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

    from ase.optimize import LBFGS, BFGS, BFGSLineSearch, FIRE, QuasiNewton

    OPT = {
        "LBFGS": LBFGS,
        "BFGS": BFGS,
        "BFGSLineSearch": BFGSLineSearch,
        "FIRE": FIRE,
        "QuasiNewton": QuasiNewton,
    }[optimizer]

    LOG.info("  [OPT] %s (opt_mode=%s) — maxcycles=%d", optimizer, opt_mode, maxcycles)
    cuts = gaussian_cutoffs(mode=opt_mode)
    ase_fmax = 1e-12  # we do our own convergence checks
    dyn = OPT(atoms, trajectory=None, logfile=None)

    prev_pos = None
    converged, steps = False, 0
    t_start = time.perf_counter()

    for i, _ in enumerate(dyn.irun(fmax=ase_fmax, steps=maxcycles), start=1):
        steps = i
        forces = atoms.get_forces()
        curr_pos = atoms.get_positions()
        grms, gmax = internal_force_metrics_HB(curr_pos, forces)
        drms, dmax = disp_metrics_bohr(prev_pos, curr_pos)
        # Require a real displacement measurement before declaring convergence:
        # on step 1 prev_pos is None -> drms/dmax are trivially 0.0.
        if prev_pos is not None and gaussian_converged(grms, gmax, drms, dmax, cuts):
            converged = True
            break
        prev_pos = curr_pos.copy()

    if not converged and steps:
        _log_unconverged("OPT", cuts, grms, gmax, drms, dmax)

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

    try:
        for i in range(1, maxcycles + 1):
            steps = i
            opt.step()
            forces = atoms.get_forces()
            curr_pos = atoms.get_positions()
            grms, gmax = internal_force_metrics_HB(curr_pos, forces)
            drms, dmax = disp_metrics_bohr(prev_pos, curr_pos)
            # Require a real displacement measurement before declaring convergence:
            # on step 1 prev_pos is None -> drms/dmax are trivially 0.0.
            if prev_pos is not None and gaussian_converged(
                grms, gmax, drms, dmax, cuts
            ):
                converged = True
                break
            prev_pos = curr_pos.copy()

        if not converged and steps:
            _log_unconverged("TS ", cuts, grms, gmax, drms, dmax)

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

    except np.linalg.LinAlgError as e:
        LOG.exception(
            "  [TS ] Linear algebra failure in Sella; marking unconverged. %s", e
        )
    except Exception as e:
        LOG.exception(
            "  [TS ] Unexpected exception in TS optimization; marking unconverged. %s",
            e,
        )

    wall = time.perf_counter() - t_start
    try:
        E_h = _sp_energy_Eh(atoms)
    except Exception:
        E_h = float("nan")

    LOG.info(
        "  [TS ] Aborted: conv=False | steps=%d | E=%.8f Eh | wall=%.2fs",
        steps,
        E_h,
        wall,
    )
    return False, steps, E_h


# -----------------------
# Main workflow
# -----------------------
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
    freq_ts: Optional[
        bool
    ] = None,  # NEW: True=TS freq, False=GS freq, None=auto from route
    freq_delta: float = 0.01,
    freq_nfree: int = 2,
    freq_scale: float = 1.0,
    # >1 evaluates the finite-difference displacements in batches instead of one
    # forward pass at a time. Default 1 keeps ASE's Vibrations path.
    freq_batch_size: int = 1,
    freq_xtb_workers: Optional[int] = None,
    temp: float = 298.15,
    pressure_atm: float = 1.0,
    symmetry_number: int = 1,
    point_group: Optional[str] = None,
    # qRRHO
    qrrho: bool = True,
    cutoff_cm1: Optional[float] = None,
    qrrho_ref_cm1: float = 100.0,
    qrrho_alpha: float = 4.0,
    # concentration-aware thermo. `free_volume_solvent` only affects the
    # translational-entropy standard state, and only when conc_mol_L is set — it is
    # not a solvation energy. See _free_space_mL_per_L in vib_thermo.
    conc_mol_L: Optional[float] = None,
    free_volume_solvent: Optional[str] = None,
    # ALPB solvation: E_tot = E_UMA + (E_xtb,alpb - E_xtb,vac), applied to every
    # energy AND force, so geometries optimize in solvent.
    alpb: Optional[str] = None,
    alpb_method: str = "GFN2-xTB",
    alpb_concurrent: bool = True,
    # Sella controls
    sella_internal: bool = True,
    sella_eta: float = 2e-2,
    sella_gamma: float = 1e-4,
    sella_delta0: float = 0.02,
    # IRC
    irc: bool = False,
    irc_dx: float = 0.1,
    # Sella's IRC inner loop wants its own eta/gamma, not the saddle-search values
    # in SellaOpts; None means "use Sella's IRC defaults".
    irc_eta: Optional[float] = None,
    irc_gamma: Optional[float] = None,
    irc_steps: int = 200,
    # Optional calculator injection
    calc: Optional[object] = None,
    # Resume PHASE 1 by skipping tags present in per-conformer CSV
    resume_from_per_conformer_csv: bool = False,
) -> str:
    """
    Serial conformer workflow (ensemble-level parallelism happens in batch.py):
      (1) OPT/SP/TS — serial over conformers (original order)
      (2) Write Gaussian inputs (original order)
      (3) FREQ + thermo — serial over conformers (original order)
      (4) IRC — serial (original order)
    Returns energies.csv path.

    Resume behavior (if resume_from_per_conformer_csv=True):
      - If tag appears in energies_per_conformer_{job_tag}.csv, PHASE 1 is skipped (even if failed).
      - Skipped conformers attempt to reload per_struct_{job_tag}/{job_tag}_{tag}.xyz for downstream.
    """
    job_tag = os.path.splitext(os.path.basename(xyz_path))[0]
    t0 = time.perf_counter()

    _ensure_dir(out_dir)
    per_conf_dir = os.path.join(out_dir, f"per_struct_{job_tag}")
    _ensure_dir(per_conf_dir)

    per_conf_csv = os.path.join(out_dir, f"energies_per_conformer_{job_tag}.csv")
    attempted_rows = (
        _load_attempted_conformers(per_conf_csv)
        if resume_from_per_conformer_csv
        else {}
    )
    LOG.info(
        "[RESUME] resume=%s attempted_rows=%d per_conf_csv=%s",
        resume_from_per_conformer_csv,
        len(attempted_rows),
        per_conf_csv,
    )

    # Decide the route. Exactly one of TS / SP / OPT, and `optimizer` is made to
    # agree with it so the two can never describe different things.
    #
    # optts selects a first-order saddle search, which this package only
    # implements with Sella (order=1) — so optts implies Sella rather than
    # leaving `optimizer` to say something that will be ignored.
    if optts:
        if optimizer not in (None, "Sella"):
            raise ValueError(
                f"optts=True runs a saddle search, which is Sella-only; got "
                f"optimizer={optimizer!r}. Drop the optimizer, or set optts=False "
                f"to minimize with {optimizer!r} instead."
            )
        optimizer = "Sella"
        route_kind = "TS"
    elif optimizer is None:
        route_kind = "SP"
    else:
        route_kind = "OPT"

    # Scratch for vibrations etc.
    scratch_root = os.environ.get("UMA_SCRATCH_ROOT", cache_dir or out_dir)
    job_scratch = make_job_scratch(scratch_root, f"ensemble-{job_tag}")
    _ensure_dir(job_scratch)

    LOG.info("=== Ensemble workflow (serial) ===")
    LOG.info("Input: %s", xyz_path)
    LOG.info("Outdir: %s", out_dir)
    LOG.info("Route kind: %s", route_kind)
    LOG.info("Scratch: %s", job_scratch)

    frames: List[Atoms] = ase_read(xyz_path, index=":")
    n = len(frames)
    if n == 0:
        raise RuntimeError("No frames found in input XYZ.")
    LOG.info("Loaded %d conformers", n)

    # Build calculator once if not provided
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
        dev = device  # best-effort label

    # ALPB solvation. Wrapping the calculator here is the whole feature: every
    # downstream consumer — both optimizer loops, Sella, ase.vibrations, the IRC —
    # reads energies and forces through atoms.get_potential_energy()/get_forces(),
    # so optimization, TS search, frequencies, thermo and the IRC all become
    # solvated with no further changes.
    if alpb:
        from .solvation import make_solvated_calculator

        calc = make_solvated_calculator(
            calc,
            alpb,
            method=alpb_method,
            charge=charge,
            mult=mult,
            concurrent=alpb_concurrent,
        )

    # Sella options object reused for all conformers
    sella = SellaOpts(
        internal=sella_internal,
        order=(1 if optts else 0),
        eta=sella_eta,
        gamma=sella_gamma,
        delta0=sella_delta0,
    )

    # Prepare per-conformer CSV writer (append + flush each row)
    per_conf_fields = [
        "index",
        "tag",
        "route",
        "converged",
        "steps",
        "energy_Eh",
        "energy_kcal",
        # Which solvation the stored energy was computed under. Without this a
        # resumed run cannot tell a solvated energy from a gas-phase one and would
        # silently mix them.
        "alpb",
        "solv_corr_kcal",
    ]
    per_conf_write_header = (not os.path.exists(per_conf_csv)) or (
        os.path.getsize(per_conf_csv) == 0
    )
    per_conf_f = open(per_conf_csv, "a", newline="")
    per_conf_w = csv.DictWriter(per_conf_f, fieldnames=per_conf_fields)
    if per_conf_write_header:
        per_conf_w.writeheader()
        per_conf_f.flush()

    def _append_per_conf_row(
        idx: int,
        tag: str,
        route: str,
        converged: bool,
        steps: int,
        E_h: float,
        solv_corr_kcal: Optional[float] = None,
    ):
        try:
            per_conf_w.writerow(
                dict(
                    index=idx,
                    tag=tag,
                    route=route,
                    converged=bool(converged),
                    steps=int(steps),
                    energy_Eh=float(E_h),
                    energy_kcal=_Eh_to_kcal(float(E_h)),
                    alpb=(alpb or ""),
                    solv_corr_kcal=(
                        "" if solv_corr_kcal is None else f"{solv_corr_kcal:.6f}"
                    ),
                )
            )
            per_conf_f.flush()
        except Exception as e:
            LOG.exception(
                "  [DUMP] Failed to append per-conformer CSV row for %s: %s", tag, e
            )

    # -----------------------
    # PHASE 1: OPT/SP/TS (serial; ORIGINAL ORDER)
    # -----------------------
    LOG.info("PHASE 1: OPT/SP/TS for %d conformers (original order)", n)
    conformers: List[Dict[str, Any]] = []

    for idx, src in enumerate(frames):
        tag = f"conf_{idx:04d}"
        LOG.info("--- Conformer %d/%d | tag=%s ---", idx + 1, n, tag)

        # A stored energy is only reusable if it was computed under the same
        # solvation setting; otherwise a gas-phase and a solvated run would be
        # silently mixed in one ensemble.
        if resume_from_per_conformer_csv and (tag in attempted_rows):
            prev_alpb = (attempted_rows[tag].get("alpb") or "").strip()
            if prev_alpb != (alpb or ""):
                LOG.warning(
                    "  [RESUME] %s: stored energy used alpb=%r but this run uses "
                    "%r — recomputing.",
                    tag,
                    prev_alpb or None,
                    alpb,
                )
                attempted_rows.pop(tag)

        # Resume skip logic (skip PHASE 1 if tag already present)
        if resume_from_per_conformer_csv and (tag in attempted_rows):
            row = attempted_rows[tag]
            prev_route = (row.get("route") or route_kind).strip() or route_kind
            E_h = _safe_float(row.get("energy_Eh"))
            steps = _safe_int(row.get("steps"), default=0)
            converged = _safe_bool(row.get("converged"), default=False)

            LOG.info(
                "  [RESUME] skipping PHASE 1 (prev: route=%s converged=%s steps=%d E=%.8f Eh)",
                prev_route,
                converged,
                steps,
                E_h,
            )

            # Reload structure for downstream phases if available
            a = None
            conf_xyz_path = os.path.join(per_conf_dir, f"{job_tag}_{tag}.xyz")
            if os.path.exists(conf_xyz_path):
                try:
                    a = ase_read(conf_xyz_path)
                    a.pbc = False
                    a.info.update({"charge": charge, "spin": mult})
                    a.calc = calc
                except Exception as e:
                    LOG.exception(
                        "  [RESUME] %s: failed to load %s: %s", tag, conf_xyz_path, e
                    )
                    a = None
            else:
                LOG.warning(
                    "  [RESUME] %s: missing %s; downstream phases will skip.",
                    tag,
                    conf_xyz_path,
                )

            # Energy kcal: prefer Eh-derived; fall back to stored kcal if Eh is nan
            if np.isfinite(E_h):
                E_kcal = _Eh_to_kcal(E_h)
            else:
                E_kcal = _safe_float(row.get("energy_kcal"))

            conformers.append(
                dict(
                    index=idx,
                    tag=tag,
                    atoms=a,
                    route=prev_route,
                    converged=bool(converged),
                    steps=int(steps),
                    energy_Eh=float(E_h),
                    energy_kcal=float(E_kcal),
                    gibbs_Eh=None,
                    gibbs_kcal=None,
                    n_imag=None,
                    imag_ok=None,
                )
            )
            continue

        # Fresh run
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
                optimizer=optimizer,  # type: ignore[arg-type]
                opt_mode=opt_mode,
                maxcycles=maxcycles,
                sella=sella if optimizer == "Sella" else None,
            )

        rec = dict(
            index=idx,
            tag=tag,
            atoms=a,
            route=route_kind,
            converged=bool(converged),
            steps=int(steps),
            energy_Eh=float(E_h),
            energy_kcal=_Eh_to_kcal(float(E_h)),
            gibbs_Eh=None,
            gibbs_kcal=None,
            n_imag=None,
            imag_ok=None,
        )
        conformers.append(rec)

        LOG.info(
            "  Summary: E=%.8f Eh | converged=%s | steps=%d", E_h, converged, steps
        )

        # Write per-conformer XYZ
        try:
            conf_xyz_path = os.path.join(per_conf_dir, f"{job_tag}_{tag}.xyz")
            ase_write(conf_xyz_path, a, format="xyz", parallel=False)
        except Exception as e:
            LOG.exception(
                "  [DUMP] Failed to write per-conformer XYZ for %s: %s", tag, e
            )

        # The solvation term is already cached on the calculator from the last
        # evaluation — reading it costs no extra SCF.
        solv_corr_kcal = None
        if alpb:
            from .solvation import solvation_correction_eV

            d_eV = solvation_correction_eV(a.calc)
            if d_eV is not None:
                solv_corr_kcal = d_eV * KCAL_PER_MOL_PER_EV
                rec["solv_corr_kcal"] = solv_corr_kcal
                LOG.info("  Solvation correction: %+.3f kcal/mol", solv_corr_kcal)

        # Append per-conformer CSV row
        _append_per_conf_row(
            idx, tag, route_kind, converged, steps, E_h, solv_corr_kcal
        )

    # Close per-conf CSV handle
    try:
        per_conf_f.close()
    except Exception:
        pass

    LOG.info("PHASE 1: Completed.")

    # -----------------------
    # Ranking (by electronic energy)
    # -----------------------
    LOG.info("Ranking by electronic energy …")

    results_sorted, e0 = rank_by_energy(conformers)
    if e0 is None:
        raise RuntimeError(
            "No finite electronic energies available to rank conformers."
        )

    ranked_xyz_path = os.path.join(out_dir, "optimized_ranked.xyz")
    ranked_atoms = [r["atoms"] for r in results_sorted if r.get("atoms") is not None]
    if ranked_atoms:
        ase_write(ranked_xyz_path, ranked_atoms, format="xyz", parallel=False)
        LOG.info("Wrote ranked XYZ: %s", ranked_xyz_path)
    else:
        LOG.warning("No structures available to write ranked XYZ: %s", ranked_xyz_path)

    # -----------------------
    # PHASE 3: Frequencies + thermo (serial; ORIGINAL ORDER)
    # -----------------------
    if do_freq:
        freq_dir = os.path.join(out_dir, "freq_out")
        _ensure_dir(freq_dir)
        LOG.info("PHASE 3: Frequencies (original order) …")

        for r in conformers:
            a = r.get("atoms")
            if a is None:
                continue

            tag = r["tag"]
            route_used = r.get("route", route_kind)
            E_h = _safe_float(r.get("energy_Eh"))

            out_path = os.path.join(freq_dir, f"{tag}.out")
            LOG.info("  [FRQ] %s → %s", tag, out_path)

            writer = ORCAWriter(out_path, xyz_path, model, dev, opt_banner=False)
            try:
                ts_flag = freq_ts if freq_ts is not None else (route_used == "TS")

                freqs = run_frequencies_and_write(
                    writer,
                    a,
                    delta=freq_delta,
                    nfree=freq_nfree,
                    scale=freq_scale,
                    scratch_dir=os.path.join(job_scratch, tag),
                    ts=ts_flag,
                    batch_size=freq_batch_size,
                    xtb_workers=freq_xtb_workers,
                )

                n_imag = sum(1 for f in freqs if f < 0.0)
                imag_ok = n_imag == (1 if ts_flag else 0)

                th = rrho_thermo(
                    a,
                    freqs,  # already scaled by run_frequencies_and_write
                    temp,
                    pressure_atm,
                    symmetry_number,
                    E_h,
                    qrrho=qrrho,
                    cutoff_cm1=cutoff_cm1,
                    qrrho_ref_cm1=qrrho_ref_cm1,
                    qrrho_alpha=qrrho_alpha,
                    conc_mol_L=conc_mol_L,
                    solv=free_volume_solvent,
                    multiplicity=mult,
                )
                G_Eh = float(th["G_total_Eh"])

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
                    TS_el_Eh=th["TS_el_Eh"],
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

                r["gibbs_Eh"] = G_Eh
                r["gibbs_kcal"] = _Eh_to_kcal(G_Eh)
                r["n_imag"] = int(n_imag)
                r["imag_ok"] = bool(imag_ok)

            except Exception as e:
                try:
                    writer.close()
                except Exception:
                    pass
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
            a = r.get("atoms")
            if a is None:
                continue

            tag = r["tag"]

            # An IRC only means anything from a first-order saddle. When frequencies
            # were computed we know whether this is one, so trust that regardless of
            # how the geometry was produced — previously a non-TS route skipped the
            # check entirely and happily traced an IRC out of a minimum.
            n_imag = r.get("n_imag")
            if do_freq and n_imag is not None and int(n_imag) != 1:
                LOG.info(
                    "  [IRC] %s: skip (n_imag=%s, need exactly 1 imaginary mode)",
                    tag,
                    n_imag,
                )
                r["irc_ok"] = False
                continue
            if not do_freq:
                LOG.warning(
                    "  [IRC] %s: no frequencies were run — cannot confirm this is a "
                    "saddle point. Proceeding as requested.",
                    tag,
                )

            try:
                r["irc"] = run_irc_trajectories(
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
                    eta=irc_eta,
                    gamma=irc_gamma,
                    steps=irc_steps,
                )
                r["irc_ok"] = True
                LOG.info("  [IRC] %s done.", tag)
            except Exception as e:
                r["irc_ok"] = False
                LOG.exception("  [IRC] %s failed: %s", tag, e)

    # -----------------------
    # energies.csv (ranked)
    # -----------------------
    by_tag: Dict[str, Dict[str, Any]] = {c["tag"]: c for c in conformers}

    csv_path = os.path.join(out_dir, "energies.csv")
    fieldnames = ENERGIES_FIELDS
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rank, r in enumerate(results_sorted, start=1):
            tag = r["tag"]
            match = by_tag.get(tag, r)
            w.writerow(
                dict(
                    rank=rank,
                    index=r["index"],
                    tag=tag,
                    route=r.get("route", route_kind),
                    converged=r.get("converged"),
                    steps=r.get("steps"),
                    energy_Eh=r.get("energy_Eh"),
                    energy_kcal=r.get("energy_kcal"),
                    rel_kcal=r.get("rel_kcal"),
                    gibbs_Eh=match.get("gibbs_Eh"),
                    gibbs_kcal=match.get("gibbs_kcal"),
                    n_imag=match.get("n_imag"),
                    imag_ok=match.get("imag_ok"),
                    solv_corr_kcal=match.get("solv_corr_kcal"),
                )
            )

    LOG.info("Wrote energies CSV: %s", csv_path)
    LOG.info("Ensemble workflow complete in %.2fs", time.perf_counter() - t0)
    return csv_path
