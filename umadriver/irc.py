# umadriver/irc.py
from __future__ import annotations
import os
import csv
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.io import write as ase_write
from ase.io.trajectory import Trajectory

from .constants import HARTREE_PER_EV, EV_PER_HARTREE, KCAL_PER_MOL_PER_EV
from .types import SellaOpts  # reuse the small dataclass to pass hyperparams

LOG = logging.getLogger("uma.irc")

EH_TO_KCAL = EV_PER_HARTREE * KCAL_PER_MOL_PER_EV

# Do NOT fall back to Sella's own IRC defaults (eta=1e-4, gamma=0.1).
#
# `gamma` sets how tightly the initial Hessian diagonalization converges, and at
# 0.1 it does not resolve the negative mode: on an optimized TS it reports
# evals[0] = +41 instead of a negative eigenvalue. Sella's convergence test is
#     pes.converged(fmax)[0] and pes.H.evals[0] > 0
# and the force term is trivially true at a converged saddle (fmax there is ~2e-4,
# orders of magnitude under any sane threshold), so a positive evals[0] makes the
# IRC declare success before taking a single step. The result is one frame per
# leg, both sitting on the TS — a silent no-op that looks like a completed run.
#
# The saddle-search values in SellaOpts are tight enough to find the mode
# (measured on HCN <-> HNC: 27-38 frames per leg, endpoints 3.0 A apart, i.e. the
# two isomers), so those are the defaults here.
IRC_ETA_DEFAULT = 2e-2
IRC_GAMMA_DEFAULT = 1e-4


def _write_traj_from_coords(atoms_ref: Atoms, coords_list, path: str):
    """Write an ASE .traj from a sequence of Cartesian arrays (Å)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with Trajectory(path, "w") as traj:
        for q in coords_list:
            snap = atoms_ref.copy()
            snap.set_positions(np.asarray(q, dtype=float))
            traj.write(snap)


def _frames_to_atoms(atoms_ref: Atoms, coords_list) -> List[Atoms]:
    out = []
    for q in coords_list:
        snap = atoms_ref.copy()
        snap.set_positions(np.asarray(q, dtype=float))
        out.append(snap)
    return out


def _arc_lengths(coords_list: Sequence[np.ndarray]) -> List[float]:
    """Cumulative Cartesian path length (Å) from the first frame."""
    lengths = [0.0]
    for prev, curr in zip(coords_list, coords_list[1:]):
        lengths.append(lengths[-1] + float(np.linalg.norm(curr - prev)))
    return lengths


def _write_irc_csv(
    path: str,
    ts_energy_Eh: float,
    legs: Dict[str, Tuple[List[np.ndarray], List[float]]],
):
    """Energy profile for plotting: one row per frame, per direction."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["direction", "step", "arc_length_A", "energy_Eh", "rel_kcal"])
        w.writerow(["ts", 0, 0.0, ts_energy_Eh, 0.0])
        for direction, (coords, energies) in legs.items():
            arcs = _arc_lengths(coords)
            for i, (arc, E) in enumerate(zip(arcs, energies), start=1):
                rel = (E - ts_energy_Eh) * EH_TO_KCAL
                w.writerow([direction, i, f"{arc:.6f}", f"{E:.10f}", f"{rel:.6f}"])


def run_irc_trajectories(
    atoms: Atoms,
    tag: str,
    out_dir: str,
    *,
    dx: float = 0.1,
    sella_opts: Optional[SellaOpts] = None,
    eta: Optional[float] = None,
    gamma: Optional[float] = None,
    fmax: float = 0.05,
    steps: int = 200,
) -> Dict[str, Any]:
    """Trace the IRC forward and backward from a transition state.

    ``atoms`` is assumed to sit at a first-order saddle and must carry a
    calculator. Returns a dict describing both legs and the files written.

    Two Sella details drive the shape of this function:

    * ``IRC`` is an ASE ``Optimizer`` built from ``Atoms`` — not from a ``Sella``
      object — and its ``run()`` returns a convergence bool, not coordinates. We
      collect frames through an observer instead, which also gets us the energies
      for the profile.
    * The instance caches the TS geometry and Hessian on its first ``irun`` and
      restores them when the direction flips. That reset only happens if the *same*
      object runs both legs; building a second ``IRC`` would silently start the
      reverse leg from the forward endpoint.
    """
    from sella import IRC as SellaIRC

    if atoms.calc is None:
        raise ValueError("run_irc_trajectories requires atoms with a calculator")

    s = sella_opts or SellaOpts()
    irc_eta = float(eta) if eta is not None else (s.eta or IRC_ETA_DEFAULT)
    irc_gamma = float(gamma) if gamma is not None else (s.gamma or IRC_GAMMA_DEFAULT)

    os.makedirs(out_dir, exist_ok=True)
    x0 = atoms.get_positions().copy()
    ts_energy_Eh = float(atoms.get_potential_energy()) * HARTREE_PER_EV

    LOG.info(
        "  [IRC] dx=%.3f eta=%.3g gamma=%.3g fmax=%.3g steps=%d (internal=%s ignored: "
        "Sella's IRC has no internal-coordinate mode)",
        dx,
        irc_eta,
        irc_gamma,
        fmax,
        steps,
        s.internal,
    )

    # keep_going=True: an inner-loop stall raises IRCInnerLoopConvergenceFailure and
    # throws away the whole leg otherwise. A truncated path is still useful.
    irc = SellaIRC(
        atoms,
        dx=dx,
        eta=irc_eta,
        gamma=irc_gamma,
        keep_going=True,
        logfile=None,
    )

    frames: List[np.ndarray] = []
    energies: List[float] = []

    def _record():
        frames.append(atoms.get_positions().copy())
        energies.append(float(atoms.get_potential_energy()) * HARTREE_PER_EV)

    irc.attach(_record)

    result: Dict[str, Any] = {
        "tag": tag,
        "ts_energy_Eh": ts_energy_Eh,
        "dx": dx,
    }
    legs: Dict[str, Tuple[List[np.ndarray], List[float]]] = {}

    try:
        for direction, suffix in (("forward", "fwd"), ("reverse", "rev")):
            frames.clear()
            energies.clear()

            LOG.info("  [IRC] Running %s …", direction)
            try:
                converged = bool(irc.run(fmax=fmax, steps=steps, direction=direction))
            except Exception as e:
                LOG.exception("  [IRC] %s leg failed: %s", direction, e)
                converged = False

            coords = [f.copy() for f in frames]
            leg_energies = list(energies)
            legs[direction] = (coords, leg_energies)

            traj_path = os.path.join(out_dir, f"{tag}_irc_traj_{suffix}.traj")
            _write_traj_from_coords(atoms, coords, traj_path)

            if len(coords) <= 1:
                # Sella reports success without stepping when its convergence test
                # is satisfied at the saddle itself. That yields a one-frame "path"
                # that still looks like a completed run, so say so plainly.
                LOG.warning(
                    "  [IRC] %s %s leg produced %d frame(s) — the IRC did not move "
                    "off the saddle. This usually means the initial diagonalization "
                    "missed the negative mode; try a tighter --irc-gamma.",
                    tag,
                    direction,
                    len(coords),
                )

            result[f"traj_{suffix}"] = traj_path
            result[f"converged_{suffix}"] = converged
            result[f"nframes_{suffix}"] = len(coords)
            result[f"energy_end_{suffix}_Eh"] = (
                leg_energies[-1] if leg_energies else float("nan")
            )
            LOG.info(
                "  [IRC] %s: %d frames, converged=%s, ΔE=%+.2f kcal/mol → %s",
                direction,
                len(coords),
                converged,
                (leg_energies[-1] - ts_energy_Eh) * EH_TO_KCAL if leg_energies else float("nan"),
                traj_path,
            )
    finally:
        # The legs walk `atoms` away from the saddle; downstream phases still expect
        # the optimized TS geometry.
        atoms.set_positions(x0)

    # Stitched path: reverse endpoint → … → TS → … → forward endpoint
    ts_atoms = atoms.copy()
    path_atoms: List[Atoms] = []
    path_atoms += _frames_to_atoms(atoms, reversed(legs["reverse"][0]))
    path_atoms.append(ts_atoms)
    path_atoms += _frames_to_atoms(atoms, legs["forward"][0])

    path_xyz = os.path.join(out_dir, f"{tag}_irc_path.xyz")
    ase_write(path_xyz, path_atoms, format="xyz", parallel=False)
    result["path_xyz"] = path_xyz
    result["nframes_path"] = len(path_atoms)

    csv_path = os.path.join(out_dir, f"{tag}_irc.csv")
    _write_irc_csv(csv_path, ts_energy_Eh, legs)
    result["csv"] = csv_path

    LOG.info("  [IRC] Wrote %s (%d frames) and %s", path_xyz, len(path_atoms), csv_path)
    return result
