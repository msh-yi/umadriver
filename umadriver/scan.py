# umadriver/scan.py
"""Relaxed scan along a bond distance.

Walk the distance between two atoms from one value to another in fixed steps. At
each step the bond is held with ``FixBondLength`` and everything else is
minimized, so the profile is a relaxed one — each point is the best structure
available at that value of the coordinate, not a rigid distortion of the input.

Each step starts from the previous *relaxed* geometry rather than from the input,
so the path stays continuous and the optimizer has a good guess to work from.

Atoms are numbered **1-based** everywhere a user sees them — the CLI, manifests,
error messages and the CSV — matching Gaussian input and the structure viewers
people pick atoms in. The conversion to ASE's 0-based indexing happens once, in
``parse_scan_spec``; ``ScanSpec.i``/``.j`` are 0-based and ``.i1``/``.j1`` are the
numbers to print. The chosen pair is logged with element symbols and the starting
distance, because an off-by-one silently scans a different bond and every number
downstream still looks perfectly reasonable.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.constraints import FixBondLengths
from ase.io import write as ase_write

from .constants import HARTREE_PER_EV, EV_PER_HARTREE, KCAL_PER_MOL_PER_EV

LOG = logging.getLogger("uma.scan")

EH_TO_KCAL = EV_PER_HARTREE * KCAL_PER_MOL_PER_EV

# Sella satisfies constraints to a tolerance rather than exactly (constraints_tol
# defaults to 1e-5), so a small mismatch between the requested and achieved
# distance is expected. Beyond this the point is not really on the coordinate you
# asked for, and the profile's x-axis is quietly wrong.
DRIFT_WARN_A = 1e-3

class FixBondLength64(FixBondLengths):
    """``FixBondLength`` that survives a calculator returning float32 forces.

    ASE enforces the constraint on forces with a RATTLE iteration whose
    convergence threshold is a hard-coded 1e-13 (``FixBondLengths.tolerance``).
    That is below float32 resolution, so with UMA — whose forces come back
    float32 — the loop can never converge: it burns all 500 iterations and raises
    ``RuntimeError('Did not converge')`` on *every single force evaluation*, which
    kills the scan on its first optimizer step.

    Doing the same arithmetic in float64 and writing the result back fixes it
    without touching ASE's semantics or loosening any tolerance. The result is
    still only float32-accurate, which is all the input forces were worth anyway.
    """

    def adjust_momenta(self, atoms, p):
        if p.dtype == np.float64:
            return super().adjust_momenta(atoms, p)
        buf = np.array(p, dtype=np.float64)
        super().adjust_momenta(atoms, buf)
        p[:] = buf


def fix_bond_length(i: int, j: int) -> FixBondLength64:
    """Hold the i-j distance at whatever it currently is (0-based indices)."""
    return FixBondLength64([(i, j)])


SCAN_FIELDS = [
    "point",
    "target_A",
    "actual_A",
    "energy_Eh",
    "energy_kcal",
    "rel_kcal",
    "converged",
    "steps",
]


@dataclass(frozen=True)
class ScanSpec:
    """Two atoms, a distance range, and how many points to put on it.

    ``i``/``j`` are 0-based for ASE. Use ``i1``/``j1`` in anything a user reads.
    """

    i: int
    j: int
    r_start: float
    r_end: float
    nsteps: int

    @property
    def i1(self) -> int:
        return self.i + 1

    @property
    def j1(self) -> int:
        return self.j + 1

    @property
    def targets(self) -> np.ndarray:
        return np.linspace(self.r_start, self.r_end, self.nsteps)

    @property
    def step_A(self) -> float:
        return abs(self.r_end - self.r_start) / (self.nsteps - 1)

    def describe(self) -> str:
        return (
            f"atoms {self.i1}-{self.j1}, {self.r_start:.3f} -> {self.r_end:.3f} A "
            f"in {self.nsteps} points ({self.step_A:.4f} A/step)"
        )


def parse_scan_spec(value: Any) -> ScanSpec:
    """Build a ScanSpec from CLI arguments or a manifest entry.

    Accepts either an ordered sequence — ``[i, j, r_start, r_end, nsteps]``, which
    is what the CLI produces — or a mapping, which reads better in YAML:

        scan: {i: 3, j: 7, from: 2.8, to: 1.4, steps: 15}

    Atoms are given 1-based; the returned spec is 0-based. This is the only place
    that conversion happens, so the CLI and manifests cannot disagree about it.
    """
    if isinstance(value, ScanSpec):
        return value

    if isinstance(value, dict):
        aliases = {
            "i": ("i", "atom1", "a1"),
            "j": ("j", "atom2", "a2"),
            "r_start": ("from", "r_start", "start"),
            "r_end": ("to", "r_end", "end"),
            "nsteps": ("steps", "nsteps", "npoints"),
        }
        known = {name for group in aliases.values() for name in group}
        unknown = set(value) - known
        if unknown:
            raise ValueError(
                f"scan: unknown key(s) {sorted(unknown)}; expected "
                "i/j/from/to/steps"
            )
        picked = {}
        for field, names in aliases.items():
            found = [n for n in names if n in value]
            if not found:
                raise ValueError(
                    f"scan: missing {names[0]!r} (need i, j, from, to, steps)"
                )
            if len(found) > 1:
                raise ValueError(f"scan: {found} are aliases; give only one")
            picked[field] = value[found[0]]
        raw = [picked[f] for f in ("i", "j", "r_start", "r_end", "nsteps")]
    else:
        raw = list(value)
        if len(raw) != 5:
            raise ValueError(
                f"scan needs 5 values (i j from to steps), got {len(raw)}: {raw}"
            )

    try:
        i1, j1 = int(raw[0]), int(raw[1])
        r_start, r_end = float(raw[2]), float(raw[3])
        nsteps = int(raw[4])
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"scan expects `i j from to steps` as two atom numbers, two distances "
            f"in angstroms, and a point count; got {list(raw)}"
        ) from e

    if i1 == j1:
        raise ValueError(f"scan: atoms i and j must differ (both are {i1})")
    if i1 < 1 or j1 < 1:
        raise ValueError(
            f"scan: atoms are numbered from 1, got {i1} and {j1}. "
            "(The first atom in the file is 1, not 0.)"
        )
    i, j = i1 - 1, j1 - 1
    if nsteps < 2:
        raise ValueError(f"scan: need at least 2 points to scan, got {nsteps}")
    if r_start <= 0 or r_end <= 0:
        raise ValueError(
            f"scan: distances must be positive, got {r_start} -> {r_end} A"
        )
    if r_start == r_end:
        raise ValueError(
            f"scan: `from` and `to` are both {r_start} A — nothing to scan"
        )

    return ScanSpec(i=i, j=j, r_start=r_start, r_end=r_end, nsteps=nsteps)


def _validate_against(atoms: Atoms, spec: ScanSpec) -> None:
    n = len(atoms)
    for label, num in (("i", spec.i1), ("j", spec.j1)):
        if num > n:
            raise ValueError(
                f"scan: atom {label}={num} is out of range for a {n}-atom "
                f"structure (atoms are numbered 1..{n})"
            )


def run_bond_scan(
    atoms: Atoms,
    spec: ScanSpec,
    *,
    relax: Callable[[Atoms], Tuple[bool, int, float]],
    out_dir: str,
    tag: str,
) -> List[Dict[str, Any]]:
    """Run the scan in place on ``atoms``; return one record per point.

    ``relax`` minimizes the atoms it is given and returns
    ``(converged, steps, energy_Eh)``. It is injected rather than imported so this
    module stays independent of the workflow, and so tests can drive it with a
    plain ASE optimizer.

    On return ``atoms`` holds the final scan point and its original constraints.
    """
    _validate_against(atoms, spec)
    os.makedirs(out_dir, exist_ok=True)

    sym = atoms.get_chemical_symbols()
    LOG.info(
        "  [SCAN] %s | %s%d-%s%d, starting from %.4f A",
        spec.describe(),
        sym[spec.i],
        spec.i1,
        sym[spec.j],
        spec.j1,
        atoms.get_distance(spec.i, spec.j),
    )

    original_constraints = list(atoms.constraints)
    records: List[Dict[str, Any]] = []
    frames: List[Atoms] = []

    try:
        for k, target in enumerate(spec.targets):
            # Clear before moving: FixBondLength would otherwise fight set_distance.
            atoms.set_constraint()
            atoms.set_distance(spec.i, spec.j, float(target), fix=0.5)
            # A fresh constraint each step — FixBondLengths captures the distance
            # it sees, so a reused one would pin every point at the first target.
            atoms.set_constraint(fix_bond_length(spec.i, spec.j))

            converged, steps, E_h = relax(atoms)
            actual = float(atoms.get_distance(spec.i, spec.j))
            drift = abs(actual - float(target))
            if drift > DRIFT_WARN_A:
                LOG.warning(
                    "  [SCAN] point %d: constraint drifted %.2e A (asked %.4f, got "
                    "%.4f) — this point is not on the requested coordinate",
                    k,
                    drift,
                    target,
                    actual,
                )

            records.append(
                {
                    "point": k,
                    "target_A": float(target),
                    "actual_A": actual,
                    "energy_Eh": float(E_h),
                    "energy_kcal": float(E_h) * EH_TO_KCAL,
                    "converged": bool(converged),
                    "steps": int(steps),
                }
            )

            frame = atoms.copy()
            frame.set_constraint()
            frame.info.update({"scan_r_A": actual, "scan_E_Eh": float(E_h)})
            frames.append(frame)

            LOG.info(
                "  [SCAN] %3d/%d  r=%.4f A  E=%.8f Eh  conv=%s  steps=%d",
                k + 1,
                spec.nsteps,
                actual,
                E_h,
                converged,
                steps,
            )
    finally:
        atoms.set_constraint(original_constraints)

    energies = np.array([r["energy_Eh"] for r in records], dtype=float)
    e_min = float(np.min(energies))
    for r in records:
        r["rel_kcal"] = (r["energy_Eh"] - e_min) * EH_TO_KCAL

    _write_outputs(records, frames, out_dir, tag)
    _log_profile(records, spec)
    return records


def _write_outputs(
    records: List[Dict[str, Any]], frames: List[Atoms], out_dir: str, tag: str
) -> None:
    csv_path = os.path.join(out_dir, f"{tag}_scan.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCAN_FIELDS)
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in SCAN_FIELDS})

    path_xyz = os.path.join(out_dir, f"{tag}_scan.xyz")
    ase_write(path_xyz, frames, format="extxyz")

    # The highest point is the reason most people run a scan: it is the starting
    # guess for a real saddle search. Emit it on its own so --optts can be pointed
    # straight at it.
    top = int(np.argmax([r["energy_Eh"] for r in records]))
    ase_write(os.path.join(out_dir, f"{tag}_scan_max.xyz"), frames[top], format="xyz")


def _log_profile(records: List[Dict[str, Any]], spec: ScanSpec) -> None:
    top = max(records, key=lambda r: r["energy_Eh"])
    unconverged = [r["point"] for r in records if not r["converged"]]

    if top["point"] in (0, len(records) - 1):
        LOG.warning(
            "  [SCAN] highest point is at the %s of the range (r=%.4f A). The "
            "barrier is probably outside the scanned window — widen it.",
            "start" if top["point"] == 0 else "end",
            top["actual_A"],
        )
    else:
        LOG.info(
            "  [SCAN] maximum at point %d: r=%.4f A, %.2f kcal/mol above the "
            "lowest point — written to *_scan_max.xyz as a TS guess for --optts",
            top["point"],
            top["actual_A"],
            top["rel_kcal"],
        )

    if unconverged:
        LOG.warning(
            "  [SCAN] %d/%d points did not converge: %s. Their energies are upper "
            "bounds, so the profile is unreliable there.",
            len(unconverged),
            len(records),
            unconverged,
        )
