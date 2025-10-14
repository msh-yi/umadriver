# umadriver/irc.py
from __future__ import annotations
import os
import logging
from typing import Tuple
import numpy as np
from ase import Atoms
from ase.io.trajectory import Trajectory

from .types import SellaOpts  # reuse the small dataclass to pass hyperparams

LOG = logging.getLogger("uma.irc")


def _write_traj_from_coords(atoms_ref: Atoms, coords_list, path: str):
    """Write an ASE .traj from a sequence of Cartesian arrays (Å)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with Trajectory(path, "w") as traj:
        for q in coords_list:
            snap = atoms_ref.copy()
            snap.set_positions(np.asarray(q, dtype=float))
            traj.write(snap)


def run_irc_trajectories(
    atoms: Atoms,
    tag: str,
    out_dir: str,
    *,
    dx: float = 0.1,
    sella_opts: SellaOpts,
) -> Tuple[str, str]:
    """
    Build a Sella(order=1, ...) object from the current Atoms (assumed TS-ish),
    then run IRC forward and reverse with step size dx.
    Returns (traj_fwd_path, traj_rev_path).
    """
    from sella import Sella, IRC

    LOG.info(
        "  [IRC] Preparing Sella for IRC: order=1, internal=%s, eta=%.3g, gamma=%.3g, delta0=%.3g, dx=%.3f",
        sella_opts.internal,
        sella_opts.eta,
        sella_opts.gamma,
        sella_opts.delta0,
        dx,
    )

    # Sella for TS geometry (order=1)
    s = Sella(
        atoms,
        order=1,
        internal=sella_opts.internal,
        eta=sella_opts.eta,
        gamma=sella_opts.gamma,
        delta0=sella_opts.delta0,
    )

    irc = IRC(s, dx=dx)

    # Forward
    LOG.info("  [IRC] Running forward …")
    coords_fwd = irc.run(direction="forward")
    traj_fwd = os.path.join(out_dir, f"{tag}_irc_traj_fwd.traj")
    _write_traj_from_coords(atoms, coords_fwd, traj_fwd)
    LOG.info("  [IRC] Wrote %s (%d frames)", traj_fwd, len(coords_fwd))

    # Reverse
    LOG.info("  [IRC] Running reverse …")
    coords_rev = irc.run(direction="reverse")
    traj_rev = os.path.join(out_dir, f"{tag}_irc_traj_rev.traj")
    _write_traj_from_coords(atoms, coords_rev, traj_rev)
    LOG.info("  [IRC] Wrote %s (%d frames)", traj_rev, len(coords_rev))

    return traj_fwd, traj_rev
