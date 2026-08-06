# umadriver/scan.py
"""Relaxed scans along internal coordinates: distances, angles and dihedrals.

Walk one or more internal coordinates from one value to another. At every point
the scanned coordinates are held with ``FixInternals`` and everything else is
minimized, so the profile is a genuine slice through the PES rather than a rigid
distortion. Each point starts from the previous *relaxed* geometry, which keeps
the path continuous and gives the optimizer a good guess.

Modelled on xtb's ``$scan`` block (https://xtb-docs.readthedocs.io/en/latest/scan.html),
including its two ways of combining several coordinates:

``sequential`` (default)
    Scan the coordinates one after another. Coordinate 2 starts from where
    coordinate 1 finished, and coordinate 1 stays at its end value throughout.
    Costs ``sum(nsteps)`` points.

``concerted``
    Advance every coordinate together, so point *p* holds coordinate *k* at its
    *p*-th target. All coordinates must therefore have the same step count.
    Costs ``nsteps`` points and traces one path through the several coordinates.

``grid``
    Every combination of the two coordinates — a proper 2D surface, at
    ``n1 * n2`` optimizations. Limited to exactly two coordinates: a third turns
    a manageable 12x12 into 1728 points. xtb does not offer this at all.

    The grid is walked in boustrophedon order (each row traversed opposite to the
    last) so consecutive points are always neighbours. A plain row-major sweep
    would jump the first coordinate from its last value back to its first at every
    row break, throwing away the starting guess exactly where the geometry has
    drifted furthest.

Atoms are numbered **1-based** everywhere a user sees them — CLI, manifests,
errors, logs and the output CSV — matching xtb, Gaussian and the viewers people
pick atoms in. Conversion to ASE's 0-based indexing happens once, in
``parse_scan_spec``, so the CLI and manifests cannot disagree about it.
"""

from __future__ import annotations

import csv
import logging
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.constraints import FixInternals
from ase.io import write as ase_write

from .constants import EV_PER_HARTREE, KCAL_PER_MOL_PER_EV

LOG = logging.getLogger("uma.scan")

EH_TO_KCAL = EV_PER_HARTREE * KCAL_PER_MOL_PER_EV

# How many atoms each coordinate type takes, and what its value means.
COORD_KINDS: Dict[str, Tuple[int, str]] = {
    "distance": (2, "A"),
    "angle": (3, "deg"),
    "dihedral": (4, "deg"),
}

# Aliases accepted in manifests, so `bond:`/`torsion:` do not silently fail.
KIND_ALIASES = {
    "distance": "distance",
    "bond": "distance",
    "r": "distance",
    "angle": "angle",
    "bend": "angle",
    "a": "angle",
    "dihedral": "dihedral",
    "torsion": "dihedral",
    "t": "dihedral",
}

SCAN_MODES = ("sequential", "concerted", "grid")

# A grid over three coordinates is 12x12x12 = 1728 optimizations before anyone
# notices. Two is the most that stays defensible.
GRID_MAX_COORDS = 2

# Constraints are satisfied to a tolerance, not exactly (Sella's constraints_tol
# is 1e-5, and ASE's FixInternals iterates to epsilon=1e-7). Past these the point
# is not really on the coordinate that was asked for, and the profile's x-axis is
# quietly wrong — so every point records what it actually reached.
DRIFT_WARN = {"A": 1e-3, "deg": 1e-1}


@dataclass(frozen=True)
class ScanCoord:
    """One internal coordinate and the range to walk it over.

    ``indices`` are 0-based for ASE; ``atoms1`` is the 1-based tuple to print.
    """

    kind: str
    indices: Tuple[int, ...]
    start: float
    end: float
    nsteps: int

    @property
    def atoms1(self) -> Tuple[int, ...]:
        return tuple(i + 1 for i in self.indices)

    @property
    def unit(self) -> str:
        return COORD_KINDS[self.kind][1]

    @property
    def targets(self) -> np.ndarray:
        return np.linspace(self.start, self.end, self.nsteps)

    @property
    def step(self) -> float:
        return abs(self.end - self.start) / (self.nsteps - 1)

    @property
    def label(self) -> str:
        """Short, stable column name: ``d_1_2``, ``a_2_1_3``, ``t_8_5_1_4``."""
        prefix = {"distance": "d", "angle": "a", "dihedral": "t"}[self.kind]
        return prefix + "_" + "_".join(str(n) for n in self.atoms1)

    def describe(self) -> str:
        atoms = "-".join(str(n) for n in self.atoms1)
        return (
            f"{self.kind} {atoms}: {self.start:g} -> {self.end:g} {self.unit} "
            f"in {self.nsteps} points ({self.step:g} {self.unit}/step)"
        )

    def measure(self, atoms: Atoms) -> float:
        if self.kind == "distance":
            return float(atoms.get_distance(*self.indices))
        if self.kind == "angle":
            return float(atoms.get_angle(*self.indices))
        return float(atoms.get_dihedral(*self.indices))


@dataclass(frozen=True)
class ScanSpec:
    """One or more coordinates, and how to combine them."""

    coords: Tuple[ScanCoord, ...]
    mode: str = "sequential"

    @property
    def npoints(self) -> int:
        return len(self.schedule())

    def describe(self) -> str:
        if self.mode == "grid":
            dims = " x ".join(str(c.nsteps) for c in self.coords)
            head = f"{len(self.coords)} coordinates, mode=grid, {dims} = {self.npoints} points"
        else:
            head = (
                f"{len(self.coords)} coordinate(s), mode={self.mode}, "
                f"{self.npoints} points"
            )
        return head + "".join("\n           " + c.describe() for c in self.coords)

    def schedule(self) -> List[Tuple[float, ...]]:
        """The value every coordinate is held at, for each point in order.

        This is the whole difference between the modes, in one place.
        """
        if self.mode == "grid":
            outer, inner = self.coords
            points = []
            for row, a in enumerate(outer.targets):
                # Boustrophedon: reverse every other row so the step from the end
                # of one row to the start of the next is a single inner-coordinate
                # increment rather than a jump back across the whole range.
                values = inner.targets if row % 2 == 0 else inner.targets[::-1]
                points.extend((float(a), float(b)) for b in values)
        elif self.mode == "concerted":
            points = [
                tuple(float(c.targets[p]) for c in self.coords)
                for p in range(self.coords[0].nsteps)
            ]
        else:
            # Sequential: walk one coordinate at a time. Coordinates not yet
            # scanned sit at their start value; ones already scanned stay at their
            # end value — the same "each completed scan leaves its constraint at
            # its final value" rule xtb uses.
            points = []
            held = [float(c.start) for c in self.coords]
            for k, coord in enumerate(self.coords):
                for value in coord.targets:
                    held[k] = float(value)
                    points.append(tuple(held))
                held[k] = float(coord.end)

        # Drop a point that repeats the one before it. In a sequential scan the
        # handover is always such a repeat: the next coordinate's first target is
        # the value it was already being held at, so without this every extra
        # coordinate costs one duplicate optimization of an identical geometry.
        deduped = [points[0]]
        for p in points[1:]:
            if p != deduped[-1]:
                deduped.append(p)
        return deduped


def _coerce_kind(name: str) -> str:
    key = str(name).strip().lower()
    if key not in KIND_ALIASES:
        raise ValueError(
            f"scan: unknown coordinate type {name!r}; expected one of "
            f"{sorted(set(KIND_ALIASES.values()))}"
        )
    return KIND_ALIASES[key]


def _build_coord(kind: str, atom_numbers: Sequence, start, end, nsteps) -> ScanCoord:
    """Validate one coordinate. ``atom_numbers`` are 1-based."""
    natoms, unit = COORD_KINDS[kind]
    try:
        nums = [int(n) for n in atom_numbers]
        start_f, end_f = float(start), float(end)
        steps_i = int(nsteps)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"scan: a {kind} needs {natoms} atom numbers, a start and end value in "
            f"{unit}, and a point count; got atoms={list(atom_numbers)}, "
            f"{start!r} -> {end!r} in {nsteps!r}"
        ) from e

    if len(nums) != natoms:
        raise ValueError(
            f"scan: a {kind} takes {natoms} atoms, got {len(nums)}: {nums}"
        )
    if any(n < 1 for n in nums):
        raise ValueError(
            f"scan: atoms are numbered from 1, got {nums}. "
            "(The first atom in the file is 1, not 0.)"
        )
    if len(set(nums)) != len(nums):
        raise ValueError(f"scan: a {kind} needs {natoms} distinct atoms, got {nums}")
    if steps_i < 2:
        raise ValueError(f"scan: need at least 2 points to scan, got {steps_i}")
    if start_f == end_f:
        raise ValueError(
            f"scan: `from` and `to` are both {start_f:g} {unit} — nothing to scan"
        )
    if kind == "distance" and (start_f <= 0 or end_f <= 0):
        raise ValueError(
            f"scan: distances must be positive, got {start_f:g} -> {end_f:g} A"
        )

    return ScanCoord(
        kind=kind,
        indices=tuple(n - 1 for n in nums),
        start=start_f,
        end=end_f,
        nsteps=steps_i,
    )


def _coord_from_flat(values: Sequence) -> ScanCoord:
    """``[i, j, from, to, steps]`` and its 6- and 7-value cousins.

    The length says which coordinate it is: 2 atoms is a distance, 3 an angle,
    4 a dihedral. That is what lets the CLI take all three from one flag.
    """
    by_length = {2 + 3: "distance", 3 + 3: "angle", 4 + 3: "dihedral"}
    kind = by_length.get(len(values))
    if kind is None:
        raise ValueError(
            f"scan takes 5, 6 or 7 values — `i j from to steps` for a distance, "
            f"`i j k from to steps` for an angle, `i j k l from to steps` for a "
            f"dihedral. Got {len(values)}: {list(values)}"
        )
    natoms = COORD_KINDS[kind][0]
    return _build_coord(kind, values[:natoms], *values[natoms:])


def _coord_from_mapping(entry: Dict[str, Any]) -> ScanCoord:
    """``{distance: [1, 2], from: 1.4, to: 2.6, steps: 13}``."""
    kinds = [k for k in entry if str(k).lower() in KIND_ALIASES]
    if len(kinds) != 1:
        raise ValueError(
            f"scan: each coordinate needs exactly one of "
            f"{sorted(set(KIND_ALIASES.values()))}; got {sorted(entry)}"
        )
    kind_key = kinds[0]
    kind = _coerce_kind(kind_key)

    atom_numbers = entry[kind_key]
    if isinstance(atom_numbers, (int, str)):
        raise ValueError(
            f"scan: {kind_key!r} takes a list of atom numbers, e.g. "
            f"{kind_key}: [1, 2]"
        )

    def _pick(*names):
        found = [n for n in names if n in entry]
        if not found:
            raise ValueError(f"scan: missing {names[0]!r} for the {kind} coordinate")
        if len(found) > 1:
            raise ValueError(f"scan: {found} are aliases; give only one")
        return entry[found[0]]

    known = {kind_key, "from", "r_start", "start", "to", "r_end", "end",
             "steps", "nsteps", "npoints"}
    unknown = set(entry) - known
    if unknown:
        raise ValueError(
            f"scan: unknown key(s) {sorted(unknown)} in a coordinate; expected "
            "a coordinate type plus from/to/steps"
        )

    return _build_coord(
        kind,
        atom_numbers,
        _pick("from", "r_start", "start"),
        _pick("to", "r_end", "end"),
        _pick("steps", "nsteps", "npoints"),
    )


def _is_flat(value: Sequence) -> bool:
    return all(not isinstance(v, (list, tuple, dict)) for v in value)


def parse_scan_spec(value: Any) -> ScanSpec:
    """Build a ScanSpec from CLI arguments or a manifest entry.

    Accepted, in rough order of how often they get written:

        [1, 2, 1.4, 2.6, 13]                       one distance (the CLI's form)
        [1, 2, 3, 100, 140, 9]                     one angle
        [[1, 2, 1.4, 2.6, 13], [2, 1, 3, ...]]     several coordinates
        {distance: [1, 2], from: 1.4, to: 2.6, steps: 13}
        {mode: concerted, coords: [ ... ]}

    Atoms are given 1-based; the returned spec is 0-based.
    """
    if isinstance(value, ScanSpec):
        return value

    mode = "sequential"
    entries: List[Any]

    if isinstance(value, dict) and ("coords" in value or "mode" in value):
        unknown = set(value) - {"coords", "mode"}
        if unknown:
            raise ValueError(
                f"scan: unknown key(s) {sorted(unknown)} beside coords/mode"
            )
        if "coords" not in value:
            raise ValueError("scan: `mode` given without `coords`")
        mode = str(value.get("mode", mode)).strip().lower()
        raw = value["coords"]
        entries = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    elif isinstance(value, dict):
        entries = [value]
    elif isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("scan: no coordinates given")
        entries = [value] if _is_flat(value) else list(value)
    else:
        raise ValueError(f"scan: expected a list or mapping, got {type(value).__name__}")

    if mode not in SCAN_MODES:
        raise ValueError(f"scan: mode must be one of {SCAN_MODES}, got {mode!r}")

    coords = tuple(
        _coord_from_mapping(e) if isinstance(e, dict) else _coord_from_flat(list(e))
        for e in entries
    )
    if not coords:
        raise ValueError("scan: no coordinates given")

    if mode == "concerted":
        counts = {c.nsteps for c in coords}
        if len(counts) > 1:
            raise ValueError(
                "scan: a concerted scan advances every coordinate together, so "
                f"they must all have the same number of points; got "
                f"{[c.nsteps for c in coords]}. Use mode=sequential for "
                "different counts."
            )

    if mode == "grid" and len(coords) != GRID_MAX_COORDS:
        total = int(np.prod([c.nsteps for c in coords]))
        raise ValueError(
            f"scan: a grid takes exactly {GRID_MAX_COORDS} coordinates, got "
            f"{len(coords)} (which would be {total} optimizations). Use "
            "mode=concerted to move them together along one path, or "
            "mode=sequential to scan them one after another."
        )

    seen = [c.indices for c in coords]
    if len(set(seen)) != len(seen):
        raise ValueError("scan: the same coordinate is listed more than once")

    return ScanSpec(coords=coords, mode=mode)


# ------------------------------------------------------------------ geometry
def _fragment_beyond(atoms: Atoms, anchor: int, moving: int) -> Optional[List[int]]:
    """Atoms on ``moving``'s side once the anchor-moving bond is cut.

    Returns None when the two stay connected — a ring, where there is no "side"
    to move and the caller should fall back to ASE's default.
    """
    from ase.neighborlist import build_neighbor_list

    try:
        nl = build_neighbor_list(atoms, self_interaction=False, bothways=True)
        cm = nl.get_connectivity_matrix(sparse=False)
    except Exception:  # pragma: no cover - connectivity is best-effort
        return None

    cm = np.array(cm)
    cm[anchor, moving] = 0
    cm[moving, anchor] = 0

    reached = {moving}
    queue = deque([moving])
    while queue:
        cur = queue.popleft()
        for nxt in np.nonzero(cm[cur])[0]:
            n = int(nxt)
            if n not in reached:
                reached.add(n)
                queue.append(n)

    if anchor in reached:
        return None
    return sorted(reached)


def _apply_value(atoms: Atoms, coord: ScanCoord, value: float) -> None:
    """Move the geometry so ``coord`` reads ``value``.

    Whole fragments are moved rather than single atoms. ASE's defaults move only
    the last atom of the coordinate, which for an angle or dihedral distorts the
    molecule instead of rotating a group — the relaxation then has to undo the
    damage, and often cannot.
    """
    idx = coord.indices
    if coord.kind == "distance":
        i, j = idx
        frag = _fragment_beyond(atoms, i, j)
        if frag is None:
            atoms.set_distance(i, j, value, fix=0.5)
        else:
            atoms.set_distance(i, j, value, fix=0, indices=frag)
    elif coord.kind == "angle":
        i, j, k = idx
        frag = _fragment_beyond(atoms, j, k)
        atoms.set_angle(i, j, k, value, indices=frag)
    else:
        i, j, k, l = idx
        frag = _fragment_beyond(atoms, j, k)
        atoms.set_dihedral(i, j, k, l, value, indices=frag)


def _constraint(spec: ScanSpec, values: Sequence[float]) -> FixInternals:
    """Hold every scanned coordinate at once.

    FixInternals rather than FixBondLength deliberately: it covers all three
    coordinate types, holds several simultaneously (which a concerted scan
    needs), and — unlike ``FixBondLengths`` — copes with the float32 forces this
    model returns. FixBondLengths runs a RATTLE iteration to a hard-coded 1e-13
    tolerance that float32 can never reach, so it exhausts maxiter and raises on
    *every* force evaluation. Do not "simplify" this back to FixBondLength.
    """
    kwargs: Dict[str, List] = {"bonds": [], "angles_deg": [], "dihedrals_deg": []}
    key = {
        "distance": "bonds",
        "angle": "angles_deg",
        "dihedral": "dihedrals_deg",
    }
    for coord, value in zip(spec.coords, values):
        kwargs[key[coord.kind]].append([float(value), list(coord.indices)])
    return FixInternals(**{k: v for k, v in kwargs.items() if v})


def _validate_against(atoms: Atoms, spec: ScanSpec) -> None:
    n = len(atoms)
    for coord in spec.coords:
        for num in coord.atoms1:
            if num > n:
                raise ValueError(
                    f"scan: atom {num} is out of range for a {n}-atom structure "
                    f"(atoms are numbered 1..{n})"
                )


# ------------------------------------------------------------------ the scan
def run_scan(
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
    plain ASE optimizer and no model.

    On return ``atoms`` holds the final scan point and its original constraints.
    """
    _validate_against(atoms, spec)
    os.makedirs(out_dir, exist_ok=True)

    sym = atoms.get_chemical_symbols()
    LOG.info("  [SCAN] %s", spec.describe())
    if spec.mode == "grid":
        LOG.info(
            "  [SCAN] a grid is %d full constrained optimizations, run one after "
            "another (each point starts from the previous, so they cannot be "
            "parallelized within a job). Separate structures still fan out across "
            "GPUs as usual.",
            spec.npoints,
        )
    for coord in spec.coords:
        LOG.info(
            "  [SCAN] %s %s | currently %.4f %s",
            coord.kind,
            "-".join(f"{sym[i]}{i + 1}" for i in coord.indices),
            coord.measure(atoms),
            coord.unit,
        )

    original_constraints = list(atoms.constraints)
    records: List[Dict[str, Any]] = []
    frames: List[Atoms] = []

    try:
        for k, values in enumerate(spec.schedule()):
            # Clear before moving: the constraints would otherwise fight the
            # set_distance/set_angle/set_dihedral calls below.
            atoms.set_constraint()
            for coord, value in zip(spec.coords, values):
                _apply_value(atoms, coord, float(value))
            # A fresh constraint each point, built from this point's targets.
            atoms.set_constraint(_constraint(spec, values))

            converged, steps, E_h = relax(atoms)

            row: Dict[str, Any] = {"point": k}
            reached = []
            for coord, value in zip(spec.coords, values):
                actual = coord.measure(atoms)
                row[f"{coord.label}_target"] = float(value)
                row[f"{coord.label}_actual"] = actual
                reached.append(f"{coord.label}={actual:.4f}")

                drift = abs(actual - float(value))
                if coord.kind == "dihedral":  # 359.9 deg and -0.1 deg are 0.2 apart
                    drift = min(drift, 360.0 - drift)
                if drift > DRIFT_WARN[coord.unit]:
                    LOG.warning(
                        "  [SCAN] point %d: %s drifted %.2e %s (asked %.4f, got "
                        "%.4f) — this point is not on the requested coordinate",
                        k,
                        coord.label,
                        drift,
                        coord.unit,
                        value,
                        actual,
                    )

            row.update(
                energy_Eh=float(E_h),
                energy_kcal=float(E_h) * EH_TO_KCAL,
                converged=bool(converged),
                steps=int(steps),
            )
            records.append(row)

            frame = atoms.copy()
            frame.set_constraint()
            frame.info.update({"scan_E_Eh": float(E_h)})
            for coord in spec.coords:
                frame.info[coord.label] = row[f"{coord.label}_actual"]
            frames.append(frame)

            LOG.info(
                "  [SCAN] %3d/%d  %s  E=%.8f Eh  conv=%s  steps=%d",
                k + 1,
                spec.npoints,
                " ".join(reached),
                E_h,
                converged,
                steps,
            )
    finally:
        atoms.set_constraint(original_constraints)

    e_min = min(r["energy_Eh"] for r in records)
    for r in records:
        r["rel_kcal"] = (r["energy_Eh"] - e_min) * EH_TO_KCAL

    _write_outputs(records, frames, spec, out_dir, tag)
    _log_profile(records, spec)
    return records


def scan_fields(spec: ScanSpec) -> List[str]:
    """CSV column order: the coordinates, then the energetics."""
    cols = ["point"]
    for coord in spec.coords:
        cols += [f"{coord.label}_target", f"{coord.label}_actual"]
    return cols + ["energy_Eh", "energy_kcal", "rel_kcal", "converged", "steps"]


def _write_outputs(
    records: List[Dict[str, Any]],
    frames: List[Atoms],
    spec: ScanSpec,
    out_dir: str,
    tag: str,
) -> None:
    fields = scan_fields(spec)
    with open(os.path.join(out_dir, f"{tag}_scan.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in fields})

    ase_write(os.path.join(out_dir, f"{tag}_scan.xyz"), frames, format="extxyz")

    # The highest point is the reason most people run a scan: it is the starting
    # guess for a real saddle search. Emit it on its own so --optts can be pointed
    # straight at it.
    top = int(np.argmax([r["energy_Eh"] for r in records]))
    ase_write(os.path.join(out_dir, f"{tag}_scan_max.xyz"), frames[top], format="xyz")


def _log_profile(records: List[Dict[str, Any]], spec: ScanSpec) -> None:
    top = max(records, key=lambda r: r["energy_Eh"])
    unconverged = [r["point"] for r in records if not r["converged"]]
    where = " ".join(f"{c.label}={top[f'{c.label}_actual']:.4f}" for c in spec.coords)

    if top["point"] in (0, len(records) - 1):
        LOG.warning(
            "  [SCAN] highest point is at the %s of the range (%s). The barrier is "
            "probably outside the scanned window — widen it.",
            "start" if top["point"] == 0 else "end",
            where,
        )
    else:
        LOG.info(
            "  [SCAN] maximum at point %d (%s), %.2f kcal/mol above the lowest "
            "point — written to *_scan_max.xyz as a TS guess for --optts",
            top["point"],
            where,
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
