"""Relaxed scans over internal coordinates.

Two layers, as elsewhere in this suite:

1. Parsing and the scan mechanics, with no model. ``run_scan`` takes its minimizer
   as an argument, so a plain ASE optimizer on EMT exercises the whole loop — the
   schedule, the constraint handling, the fragment moves, the chaining and the
   output files — in a second.
2. The real thing with UMA, which is the only way to check that a scan through a
   coordinate produces a physically sensible profile.
"""

from __future__ import annotations

import csv
import os

import numpy as np
import pytest
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.io import read as ase_read
from ase.optimize import LBFGS

from umadriver.batch import _shard_groups
from umadriver.scan import merge_scan_shards, parse_scan_spec, run_scan


# ---------------------------------------------------------------- parsing
def test_atoms_are_one_based():
    """The user-facing convention, matching xtb and Gaussian. Getting this wrong
    scans a different coordinate and every number downstream still looks
    reasonable, so it is pinned explicitly."""
    (coord,) = parse_scan_spec([1, 2, 0.9, 1.6, 8]).coords

    assert coord.indices == (0, 1), "should convert to ASE's 0-based indexing"
    assert coord.atoms1 == (1, 2), "should report back in the user's numbering"


@pytest.mark.parametrize(
    "values,kind,natoms",
    [
        ([1, 2, 0.9, 1.6, 8], "distance", 2),
        ([2, 1, 3, 100.0, 140.0, 8], "angle", 3),
        ([8, 5, 1, 4, 60.0, 420.0, 8], "dihedral", 8 and 4),
    ],
)
def test_the_value_count_picks_the_coordinate(values, kind, natoms):
    """5/6/7 values is what lets one --scan flag take all three coordinate types."""
    (coord,) = parse_scan_spec(values).coords

    assert coord.kind == kind
    assert len(coord.indices) == natoms


def test_mapping_and_sequence_forms_agree():
    seq = parse_scan_spec([3, 7, 2.8, 1.4, 15])
    mapping = parse_scan_spec({"distance": [3, 7], "from": 2.8, "to": 1.4, "steps": 15})

    assert seq == mapping


def test_coordinate_aliases():
    """`bond` and `torsion` are what people type; failing on them would be rude."""
    assert parse_scan_spec({"bond": [1, 2], "from": 1.0, "to": 2.0, "steps": 3}) == (
        parse_scan_spec({"distance": [1, 2], "from": 1.0, "to": 2.0, "steps": 3})
    )
    assert parse_scan_spec(
        {"torsion": [1, 2, 3, 4], "from": 0, "to": 90, "steps": 3}
    ) == parse_scan_spec(
        {"dihedral": [1, 2, 3, 4], "from": 0, "to": 90, "steps": 3}
    )


def test_units_and_labels():
    spec = parse_scan_spec([[1, 2, 1.0, 2.0, 3], [2, 1, 3, 100, 120, 3]])
    d, a = spec.coords

    assert (d.unit, a.unit) == ("A", "deg")
    assert (d.label, a.label) == ("d_1_2", "a_2_1_3")


def test_targets_span_the_requested_range():
    (coord,) = parse_scan_spec([1, 2, 1.0, 2.0, 5]).coords

    np.testing.assert_allclose(coord.targets, [1.0, 1.25, 1.5, 1.75, 2.0])
    assert coord.step == pytest.approx(0.25)


def test_a_decreasing_range_is_allowed():
    """Scanning inward (association) is as valid as scanning outward."""
    (coord,) = parse_scan_spec([1, 2, 3.0, 1.4, 5]).coords

    assert coord.targets[0] > coord.targets[-1]
    assert coord.step == pytest.approx(0.4)


# ------------------------------------------------- combining several coords
def test_sequential_scans_one_coordinate_at_a_time():
    """xtb's default: finish coordinate 1, then walk coordinate 2 from where it
    left off, with coordinate 1 held at its end value."""
    spec = parse_scan_spec([[1, 2, 1.0, 2.0, 3], [2, 1, 3, 100, 120, 3]])

    assert spec.mode == "sequential"
    assert spec.schedule() == [
        (1.0, 100.0),
        (1.5, 100.0),
        (2.0, 100.0),
        (2.0, 110.0),
        (2.0, 120.0),
    ]


def test_the_sequential_handover_is_not_computed_twice():
    """The last point of coordinate 1 and the first of coordinate 2 are the same
    geometry. Without dedup, every extra coordinate costs one wasted
    optimization — 3 + 3 points would run 6 rather than 5."""
    spec = parse_scan_spec([[1, 2, 1.0, 2.0, 3], [2, 1, 3, 100, 120, 3]])

    assert spec.npoints == 5
    assert len(set(spec.schedule())) == 5


def test_concerted_advances_everything_together():
    spec = parse_scan_spec(
        {"mode": "concerted", "coords": [[1, 2, 1.0, 2.0, 3], [2, 1, 3, 100, 120, 3]]}
    )

    assert spec.npoints == 3, "one path, not a grid"
    assert spec.schedule() == [(1.0, 100.0), (1.5, 110.0), (2.0, 120.0)]


def test_concerted_requires_matching_step_counts():
    with pytest.raises(ValueError, match="same number of points"):
        parse_scan_spec(
            {
                "mode": "concerted",
                "coords": [[1, 2, 1.0, 2.0, 3], [2, 1, 3, 100, 120, 5]],
            }
        )


def test_grid_covers_every_combination_exactly_once():
    spec = parse_scan_spec(
        {"mode": "grid", "coords": [[1, 2, 1.0, 1.4, 3], [1, 3, 2.0, 2.4, 3]]}
    )
    points = spec.schedule()

    assert spec.npoints == 9
    assert len(set(points)) == 9
    assert set(points) == {
        (a, b) for a in (1.0, 1.2, 1.4) for b in (2.0, 2.2, 2.4)
    }


def test_grid_is_walked_without_jumping():
    """Boustrophedon order: each row traversed opposite to the last, so every
    step is to a neighbouring grid point. A plain row-major sweep would jump the
    inner coordinate across its whole range at every row break — discarding the
    starting guess exactly where the geometry has drifted furthest."""
    spec = parse_scan_spec(
        {"mode": "grid", "coords": [[1, 2, 1.0, 1.4, 3], [1, 3, 2.0, 2.4, 3]]}
    )
    points = spec.schedule()

    for prev, curr in zip(points, points[1:]):
        changed = [i for i in (0, 1) if abs(curr[i] - prev[i]) > 1e-12]
        assert len(changed) == 1, f"{prev} -> {curr} moved both coordinates"
        assert abs(curr[changed[0]] - prev[changed[0]]) == pytest.approx(0.2), (
            f"{prev} -> {curr} is not a single step"
        )


@pytest.mark.parametrize("ncoords", [1, 3])
def test_grid_takes_exactly_two_coordinates(ncoords):
    """A third coordinate turns 12x12 into 1728 optimizations."""
    coords = [[1, i + 2, 1.0, 1.4, 3] for i in range(ncoords)]
    with pytest.raises(ValueError, match="exactly 2 coordinates"):
        parse_scan_spec({"mode": "grid", "coords": coords})


def test_grid_allows_different_step_counts():
    """Unlike concerted, the two axes are independent."""
    spec = parse_scan_spec(
        {"mode": "grid", "coords": [[1, 2, 1.0, 1.4, 3], [1, 3, 2.0, 2.4, 4]]}
    )

    assert spec.npoints == 12


def test_a_single_coordinate_is_the_same_either_way():
    seq = parse_scan_spec({"mode": "sequential", "coords": [[1, 2, 1.0, 2.0, 4]]})
    con = parse_scan_spec({"mode": "concerted", "coords": [[1, 2, 1.0, 2.0, 4]]})

    assert seq.schedule() == con.schedule()


@pytest.mark.parametrize(
    "value,message",
    [
        ([0, 2, 0.9, 1.6, 8], "numbered from 1"),
        ([1, 1, 0.9, 1.6, 8], "distinct atoms"),
        ([1, 2, 0.9, 1.6, 1], "at least 2 points"),
        ([1, 2, 1.5, 1.5, 8], "nothing to scan"),
        ([1, 2, -0.9, 1.6, 8], "must be positive"),
        ([1, 2, "x", 1.6, 8], "atom numbers"),
        ([1, 2, 0.9, 1.6], "5, 6 or 7 values"),
        ([1, 2, 3, 4, 5, 0.9, 1.6, 8], "5, 6 or 7 values"),
        ({"distance": [1, 2], "from": 0.9, "to": 1.6}, "missing"),
        ({"distance": [1, 2], "from": 0.9, "to": 1.6, "steps": 8, "no": 1}, "unknown"),
        ({"distance": 1, "from": 0.9, "to": 1.6, "steps": 8}, "list of atom numbers"),
        ({"from": 0.9, "to": 1.6, "steps": 8}, "exactly one of"),
        ({"mode": "diagonal", "coords": [[1, 2, 1.0, 2.0, 3]]}, "mode must be one of"),
        ({"mode": "concerted"}, "without `coords`"),
        ([[1, 2, 1.0, 2.0, 3], [1, 2, 1.0, 2.0, 3]], "more than once"),
        ([], "no coordinates"),
    ],
)
def test_bad_specs_are_rejected(value, message):
    with pytest.raises(ValueError, match=message):
        parse_scan_spec(value)


def test_out_of_range_atom_is_caught_against_the_structure(tmp_path):
    """Range depends on the molecule, so it cannot be checked at parse time."""
    atoms = molecule("H2O")
    atoms.calc = EMT()

    with pytest.raises(ValueError, match=r"numbered 1\.\.3"):
        run_scan(
            atoms,
            parse_scan_spec([1, 4, 0.9, 1.2, 3]),
            relax=lambda a: (True, 0, 0.0),
            out_dir=str(tmp_path),
            tag="conf_0000",
        )


# ---------------------------------------------------------------- mechanics
def _lbfgs_relax(atoms):
    """Stand-in for the workflow's minimizer."""
    LBFGS(atoms, logfile=None).run(fmax=0.05, steps=50)
    return True, 12, float(atoms.get_potential_energy())


def _scan(tmp_path, spec_values=(1, 2, 1.0, 1.6, 7), name="C2H6", **kw):
    atoms = molecule(name)
    atoms.calc = EMT()
    spec = parse_scan_spec(spec_values)
    records = run_scan(
        atoms,
        spec,
        relax=kw.pop("relax", _lbfgs_relax),
        out_dir=str(tmp_path / "scan"),
        tag="conf_0000",
        **kw,
    )
    return atoms, records, spec


@pytest.mark.parametrize(
    "values,tol",
    [
        ((1, 2, 1.0, 1.6, 5), 1e-6),
        ((3, 1, 2, 100.0, 125.0, 5), 1e-4),
        ((3, 1, 2, 6, 60.0, 200.0, 5), 1e-3),
    ],
    ids=["distance", "angle", "dihedral"],
)
def test_every_point_lands_on_its_target(tmp_path, values, tol):
    """All three coordinate types, held to their requested value."""
    _, records, spec = _scan(tmp_path, values)
    (coord,) = spec.coords

    assert len(records) == 5
    for r in records:
        target = r[f"{coord.label}_target"]
        actual = r[f"{coord.label}_actual"]
        assert actual == pytest.approx(target, abs=tol), (
            f"point {r['point']} asked for {target:.4f} {coord.unit} and got "
            f"{actual:.4f} {coord.unit}"
        )


def test_a_concerted_scan_holds_both_coordinates_at_once(tmp_path):
    _, records, spec = _scan(
        tmp_path,
        {"mode": "concerted", "coords": [[1, 2, 1.5, 1.7, 4], [3, 1, 2, 108, 118, 4]]},
    )

    assert len(records) == 4
    d, a = spec.coords
    for r in records:
        assert r[f"{d.label}_actual"] == pytest.approx(r[f"{d.label}_target"], abs=1e-5)
        assert r[f"{a.label}_actual"] == pytest.approx(r[f"{a.label}_target"], abs=1e-3)


def test_a_grid_runs_every_point(tmp_path):
    _, records, spec = _scan(
        tmp_path,
        {"mode": "grid", "coords": [[1, 2, 1.4, 1.6, 3], [1, 3, 1.05, 1.15, 3]]},
    )
    a, b = spec.coords

    assert len(records) == 9
    seen = {
        (round(r[f"{a.label}_target"], 6), round(r[f"{b.label}_target"], 6))
        for r in records
    }
    assert len(seen) == 9, "a grid point was visited twice or skipped"
    for r in records:
        assert r[f"{a.label}_actual"] == pytest.approx(r[f"{a.label}_target"], abs=1e-5)
        assert r[f"{b.label}_actual"] == pytest.approx(r[f"{b.label}_target"], abs=1e-5)


def test_the_scan_relaxes_rather_than_just_distorting(tmp_path):
    """A scan that skipped the minimization would still hit every target — so
    check the rest of the molecule actually moved."""
    moved = []

    def relax(atoms):
        before = atoms.get_positions().copy()
        out = _lbfgs_relax(atoms)
        moved.append(float(np.abs(atoms.get_positions() - before).max()))
        return out

    _scan(tmp_path, relax=relax)

    assert max(moved) > 0.01, "no relaxation happened between points"


def test_the_coordinate_is_constrained_while_relaxing(tmp_path):
    """The whole point: `relax` must be handed atoms that cannot change the
    coordinate. Without the constraint this is an ordinary optimization from a
    distorted start, and every point collapses to the same minimum.

    Asserted by effect rather than by constraint class: what matters is that a
    full minimization cannot move it, not which ASE object arranges that.
    """
    held = []

    def relax(atoms):
        before = atoms.get_distance(0, 1)
        out = _lbfgs_relax(atoms)
        held.append((before, atoms.get_distance(0, 1)))
        return out

    _scan(tmp_path, relax=relax)

    assert held, "relax was never called"
    for before, after in held:
        assert after == pytest.approx(before, abs=1e-6), (
            f"minimization moved the constrained bond {before:.4f} -> {after:.4f} A"
        )
    # ...and the points really were at different values, so this is not vacuously
    # satisfied by nothing ever moving.
    assert len({round(b, 3) for b, _ in held}) == len(held)


def test_a_dihedral_rotates_a_whole_group(tmp_path):
    """ASE's set_dihedral moves only the fourth atom unless told otherwise, which
    tears a methyl off instead of rotating it. The scan works out the fragment on
    the far side of the central bond and moves all of it."""
    seen = {}

    def relax(atoms):
        if "first" not in seen:
            seen["first"] = atoms.get_positions().copy()
        return True, 0, float(atoms.get_potential_energy())

    atoms = molecule("C2H6")
    atoms.calc = EMT()
    start = atoms.get_positions().copy()
    run_scan(
        atoms,
        parse_scan_spec([3, 1, 2, 6, 60.0, 120.0, 3]),
        relax=relax,
        out_dir=str(tmp_path / "scan"),
        tag="conf_0000",
    )

    # atoms 5,6,7 (0-based) are the three H on the rotating carbon: all should move
    moved = np.abs(seen["first"] - start).max(axis=1) > 1e-6
    assert moved[[5, 6, 7]].all(), (
        "only part of the rotating group moved — the molecule was distorted, not "
        f"rotated (moved: {np.where(moved)[0].tolist()})"
    )


def test_each_point_continues_from_the_previous_one(tmp_path):
    """Chaining is what keeps the path continuous. If every point restarted from
    the input, consecutive geometries would differ by far more."""
    starts = []

    def relax(atoms):
        starts.append(atoms.get_positions().copy())
        return _lbfgs_relax(atoms)

    _scan(tmp_path, (1, 2, 1.5, 1.7, 5), relax=relax)

    for prev, curr in zip(starts, starts[1:]):
        assert np.abs(curr - prev).max() < 0.3, (
            "point did not start from the previous geometry"
        )


def test_original_constraints_are_restored(tmp_path):
    atoms = molecule("C2H6")
    atoms.calc = EMT()
    atoms.set_constraint(FixAtoms(indices=[0]))

    run_scan(
        atoms,
        parse_scan_spec([1, 2, 1.5, 1.6, 3]),
        relax=_lbfgs_relax,
        out_dir=str(tmp_path / "scan"),
        tag="conf_0000",
    )

    assert [type(c).__name__ for c in atoms.constraints] == ["FixAtoms"]


def test_atoms_are_left_at_the_final_point(tmp_path):
    atoms, records, spec = _scan(tmp_path)
    (coord,) = spec.coords

    assert atoms.get_distance(0, 1) == pytest.approx(
        records[-1][f"{coord.label}_actual"], abs=1e-9
    )


def test_the_constraint_survives_float32_forces():
    """UMA returns float32 forces, and that is not a detail.

    ASE's FixBondLengths — the obvious choice for a bond scan — enforces itself
    with a RATTLE iteration to a hard-coded 1e-13 tolerance, which float32 can
    never reach: it exhausts maxiter and raises on *every* force evaluation, so a
    scan dies on its first optimizer step. EMT returns float64 and hides this
    completely. FixInternals iterates to 1e-7 instead and is fine, which is why
    the scan uses it for distances as well as angles and dihedrals.
    """
    from ase.constraints import FixBondLengths

    from umadriver.scan import _constraint

    # The trap, demonstrated. Whether it bites depends on the system — H2O and
    # CH4 do, C2H6 does not — which is exactly what makes it dangerous: it can
    # pass a smoke test and then kill a real run.
    water = molecule("H2O")
    water.calc = EMT()
    with pytest.raises(RuntimeError, match="Did not converge"):
        FixBondLengths([(0, 1)]).adjust_forces(
            water, np.asarray(water.get_forces(), dtype=np.float32)
        )

    atoms = molecule("C2H6")
    atoms.calc = EMT()
    f64 = atoms.get_forces()
    f32 = np.asarray(f64, dtype=np.float32)

    # what the scan actually builds, over all three coordinate types
    spec = parse_scan_spec(
        [
            [1, 2, 1.5, 1.6, 2],
            [3, 1, 2, 108.0, 110.0, 2],
            [3, 1, 2, 6, 60.0, 70.0, 2],
        ]
    )
    values = (
        atoms.get_distance(0, 1),
        atoms.get_angle(2, 0, 1),
        atoms.get_dihedral(2, 0, 1, 5),
    )

    g32 = f32.copy()
    _constraint(spec, values).adjust_forces(atoms, g32)  # must not raise
    g64 = f64.copy()
    _constraint(spec, values).adjust_forces(atoms, g64)

    assert g32.dtype == np.float32, "must write back in the caller's dtype"
    # and float32 gives the right answer, not merely a non-crashing one
    assert np.abs(g32 - g64).max() < 1e-6


# ---------------------------------------------------------------- outputs
def test_csv_has_a_column_pair_per_coordinate(tmp_path):
    _, records, spec = _scan(
        tmp_path, [[1, 2, 1.0, 1.6, 3], [3, 1, 2, 100.0, 120.0, 3]]
    )

    with open(tmp_path / "scan" / "conf_0000_scan.csv") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == len(records) == 5  # 3 + 3, minus the shared handover
    for coord in spec.coords:
        assert f"{coord.label}_target" in rows[0]
        assert f"{coord.label}_actual" in rows[0]
    # rel_kcal is measured from the lowest point, so some point must be 0
    assert min(abs(float(r["rel_kcal"])) for r in rows) == pytest.approx(0.0, abs=1e-9)


def test_trajectory_carries_every_frame(tmp_path):
    _, records, spec = _scan(tmp_path)
    (coord,) = spec.coords

    frames = ase_read(str(tmp_path / "scan" / "conf_0000_scan.xyz"), index=":")

    assert len(frames) == len(records)
    for frame, r in zip(frames, records):
        assert frame.get_distance(0, 1) == pytest.approx(
            r[f"{coord.label}_actual"], abs=1e-6
        )


def test_max_file_is_the_highest_point(tmp_path):
    """It is the TS guess users feed to --optts, so it has to be the right frame."""
    _, records, spec = _scan(tmp_path)
    (coord,) = spec.coords

    top = max(records, key=lambda r: r["energy_Eh"])
    guess = ase_read(str(tmp_path / "scan" / "conf_0000_scan_max.xyz"))

    assert guess.get_distance(0, 1) == pytest.approx(
        top[f"{coord.label}_actual"], abs=1e-6
    )


def test_unconverged_points_are_reported_not_hidden(tmp_path, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="uma.scan"):
        _scan(tmp_path, relax=lambda a: (False, 3, float(a.get_potential_energy())))

    assert any("did not converge" in r.message for r in caplog.records)


# ---------------------------------------------------------------- sharding
# A relax whose energy depends only on the constrained coordinates, so a sharded run
# is directly comparable to an unsharded one: any difference is the sharding itself
# and never the optimizer's path. Energies are negative on purpose — a string sort of
# "-0.01" and "-0.5" puts them in the opposite order to a numeric one, which is what
# catches an argmax over values that were never coerced back off the CSV.
def _flat_pes(atoms):
    d1 = atoms.get_distance(0, 1)
    d2 = atoms.get_distance(0, 2)
    return True, 1, -((d1 - 1.0) ** 2) - 2.0 * (d2 - 1.0) ** 2


GRID = {
    "mode": "grid",
    "coords": [
        {"distance": [1, 2], "from": 0.90, "to": 1.20, "steps": 4},
        {"distance": [1, 3], "from": 0.90, "to": 1.20, "steps": 4},
    ],
}


def _water():
    atoms = molecule("H2O")
    atoms.calc = EMT()
    return atoms


def _run_shards(tmp_path, spec, groups, traversal="rowmajor", relax=_flat_pes):
    """Run each shard in its own directory, as separate workers would."""
    dirs = []
    for i, group in enumerate(groups):
        d = str(tmp_path / f"shard{i:02d}")
        run_scan(
            _water(),
            spec,
            relax=relax,
            out_dir=d,
            tag="conf_0000",
            shard={"traversal": traversal, "indices": list(group)},
        )
        dirs.append(d)
    return dirs


def test_sharded_grid_reproduces_the_unsharded_scan(tmp_path):
    """The whole contract in one test: sharding changes who computes a point,
    nothing about the point."""
    spec = parse_scan_spec(GRID)
    reference = run_scan(
        _water(), spec, relax=_flat_pes, out_dir=str(tmp_path / "ref"), tag="conf_0000"
    )

    groups = _shard_groups(spec, "rowmajor", 4)
    merged = merge_scan_shards(
        _run_shards(tmp_path, spec, groups),
        spec,
        "rowmajor",
        str(tmp_path / "merged"),
        "conf_0000",
    )

    assert merged is not None
    assert len(merged) == len(reference) == 16

    # Order differs (rowmajor vs boustrophedon), so compare on physical identity:
    # a point *is* its target values, and those must agree bit for bit.
    def key(r):
        return (r["d_1_2_target"], r["d_1_3_target"])

    for want, got in zip(sorted(reference, key=key), sorted(merged, key=key)):
        # Targets are bit-identical: every shard computes them from the same full
        # spec, which is what lets the merged CSV be pivoted on these columns.
        assert got["d_1_2_target"] == want["d_1_2_target"]
        assert got["d_1_3_target"] == want["d_1_3_target"]
        # Energies are not, and cannot be: set_distance lands within float noise of
        # the target, and the noise depends on the geometry it started from — which
        # sharding changes by design. A real mismatch here would be O(0.01), not
        # O(1e-16).
        assert got["energy_Eh"] == pytest.approx(want["energy_Eh"], abs=1e-12)
        assert got["rel_kcal"] == pytest.approx(want["rel_kcal"], abs=1e-9)

    with open(tmp_path / "merged" / "conf_0000_scan.csv") as f:
        merged_header = next(csv.reader(f))
    with open(tmp_path / "ref" / "conf_0000_scan.csv") as f:
        assert next(csv.reader(f)) == merged_header


def test_a_shard_keeps_global_point_numbers(tmp_path):
    """Merging is a concatenation, which only works if indices are global."""
    spec = parse_scan_spec(GRID)
    full = spec.schedule("rowmajor")

    (d,) = _run_shards(tmp_path, spec, [[4, 5, 6, 7]])
    with open(os.path.join(d, "conf_0000_scan.csv")) as f:
        rows = list(csv.DictReader(f))

    assert [int(r["point"]) for r in rows] == [4, 5, 6, 7]
    for k, row in zip([4, 5, 6, 7], rows):
        assert float(row["d_1_2_target"]) == full[k][0]
        assert float(row["d_1_3_target"]) == full[k][1]


def test_out_of_range_shard_is_rejected(tmp_path):
    spec = parse_scan_spec(GRID)
    with pytest.raises(ValueError, match="16 points"):
        run_scan(
            _water(),
            spec,
            relax=_flat_pes,
            out_dir=str(tmp_path / "bad"),
            tag="conf_0000",
            shard={"traversal": "rowmajor", "indices": [15, 16]},
        )


def test_merged_max_is_the_global_max(tmp_path):
    """_scan_max.xyz feeds --optts, so picking it per-shard or off unconverted
    strings would hand the user the wrong geometry."""
    spec = parse_scan_spec(GRID)
    groups = _shard_groups(spec, "rowmajor", 4)
    merged = merge_scan_shards(
        _run_shards(tmp_path, spec, groups),
        spec,
        "rowmajor",
        str(tmp_path / "merged"),
        "conf_0000",
    )

    top = max(merged, key=lambda r: r["energy_Eh"])
    assert (top["d_1_2_target"], top["d_1_3_target"]) == (1.0, 1.0)  # the PES peak

    guess = ase_read(str(tmp_path / "merged" / "conf_0000_scan_max.xyz"))
    assert guess.get_distance(0, 1) == pytest.approx(1.0, abs=1e-6)
    assert guess.get_distance(0, 2) == pytest.approx(1.0, abs=1e-6)


def test_merge_rereferences_rel_kcal_to_the_whole_surface(tmp_path):
    """Each shard's rel_kcal is measured from its own lowest point."""
    spec = parse_scan_spec(GRID)
    groups = _shard_groups(spec, "rowmajor", 4)
    dirs = _run_shards(tmp_path, spec, groups)

    # every shard called its own minimum zero
    for d in dirs:
        with open(os.path.join(d, "conf_0000_scan.csv")) as f:
            assert min(float(r["rel_kcal"]) for r in csv.DictReader(f)) == 0.0

    merged = merge_scan_shards(
        dirs, spec, "rowmajor", str(tmp_path / "merged"), "conf_0000"
    )

    lowest = min(merged, key=lambda r: r["energy_Eh"])
    assert lowest["rel_kcal"] == 0.0
    assert sum(1 for r in merged if r["rel_kcal"] == 0.0) == 1


def test_merge_preserves_a_failed_point(tmp_path):
    """csv.DictReader hands back the string 'False', and bool('False') is True."""
    spec = parse_scan_spec(GRID)
    groups = _shard_groups(spec, "rowmajor", 4)

    dirs = []
    for i, group in enumerate(groups):
        d = str(tmp_path / f"shard{i:02d}")
        run_scan(
            _water(),
            spec,
            relax=_flat_pes if i else (lambda a: (False,) + _flat_pes(a)[1:]),
            out_dir=d,
            tag="conf_0000",
            shard={"traversal": "rowmajor", "indices": list(group)},
        )
        dirs.append(d)

    merged = merge_scan_shards(
        dirs, spec, "rowmajor", str(tmp_path / "merged"), "conf_0000"
    )

    assert [r["converged"] for r in merged[:4]] == [False] * 4
    assert all(r["converged"] for r in merged[4:])


def test_an_incomplete_merge_writes_nothing_it_could_be_mistaken_for(tmp_path):
    """A gap in a profile is not a shorter profile. Refusing to write is also what
    lets the next --resume rerun only the shards that failed."""
    spec = parse_scan_spec(GRID)
    groups = _shard_groups(spec, "rowmajor", 4)
    dirs = _run_shards(tmp_path, spec, groups)

    merged = merge_scan_shards(
        dirs[:2] + dirs[3:], spec, "rowmajor", str(tmp_path / "merged"), "conf_0000"
    )

    assert merged is None
    out = tmp_path / "merged"
    assert not (out / "conf_0000_scan.csv").exists()
    assert not (out / "conf_0000_scan_max.xyz").exists()
    assert (out / "conf_0000_scan.partial.csv").exists()


def test_shard_groups_partition_the_scan(tmp_path):
    spec = parse_scan_spec(GRID)
    for n in (2, 3, 4):
        groups = _shard_groups(spec, "rowmajor", n)
        flat = [k for g in groups for k in g]
        assert sorted(flat) == list(range(16))
        assert len(flat) == len(set(flat))


def test_a_sharded_grid_is_cut_on_row_boundaries(tmp_path):
    """Each shard owns whole rows, so no shard starts a row in the middle."""
    spec = parse_scan_spec(GRID)
    full = spec.schedule("rowmajor")

    for group in _shard_groups(spec, "rowmajor", 2):
        rows = {full[k][0] for k in group}
        # every point of every row this shard touches belongs to this shard
        assert sum(1 for values in full if values[0] in rows) == len(group)


# ---------------------------------------------------------------- with UMA
def test_water_bond_scan_has_a_minimum_in_the_middle(
    tmp_path, h2o_xyz, uma_calc, energies
):
    """The physics check. Stretching or squeezing an O-H bond away from
    equilibrium must cost energy, so a scan straddling ~0.96 A dips in the middle.
    """
    from umadriver.ensemble import run_conformer_workflow

    out = str(tmp_path / "scan")
    csv_path = run_conformer_workflow(
        h2o_xyz,
        out_dir=out,
        scan=[1, 2, 0.85, 1.25, 9],
        opt_mode="Normal",
        maxcycles=200,
        calc=uma_calc,
    )

    with open(os.path.join(out, "scan", "conf_0000_scan.csv")) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 9
    for r in rows:
        assert float(r["d_1_2_actual"]) == pytest.approx(float(r["d_1_2_target"]), abs=1e-3)

    e = [float(r["energy_Eh"]) for r in rows]
    lowest = int(np.argmin(e))
    assert 0 < lowest < 8, f"minimum landed at the edge (point {lowest}) of the scan"
    r_min = float(rows[lowest]["d_1_2_actual"])
    assert 0.9 < r_min < 1.05, f"O-H equilibrium looks wrong: {r_min:.3f} A"

    row = energies(csv_path)[0]
    assert row["route"] == "SCAN"
    # the summary row describes the last point, which is the saved geometry
    assert float(row["energy_Eh"]) == pytest.approx(e[-1], abs=1e-10)


def test_water_angle_scan_has_a_minimum_in_the_middle(tmp_path, h2o_xyz, uma_calc):
    """Angles go through the same machinery as distances but a different ASE call,
    so the physics is worth checking independently. Water bends at ~104.5 deg."""
    from umadriver.ensemble import run_conformer_workflow

    out = str(tmp_path / "angle")
    run_conformer_workflow(
        h2o_xyz,
        out_dir=out,
        scan={"angle": [2, 1, 3], "from": 85.0, "to": 125.0, "steps": 9},
        maxcycles=200,
        calc=uma_calc,
    )

    with open(os.path.join(out, "scan", "conf_0000_scan.csv")) as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        assert float(r["a_2_1_3_actual"]) == pytest.approx(
            float(r["a_2_1_3_target"]), abs=1e-1
        )

    e = [float(r["energy_Eh"]) for r in rows]
    best = float(rows[int(np.argmin(e))]["a_2_1_3_actual"])
    assert 95.0 < best < 115.0, f"H-O-H equilibrium angle looks wrong: {best:.1f} deg"


def test_a_two_coordinate_scan_runs_end_to_end(tmp_path, h2o_xyz, uma_calc, energies):
    """Distance and angle together, sequentially — the multi-coordinate path
    through the real workflow."""
    from umadriver.ensemble import run_conformer_workflow

    out = str(tmp_path / "multi")
    csv_path = run_conformer_workflow(
        h2o_xyz,
        out_dir=out,
        scan={
            "coords": [
                {"distance": [1, 2], "from": 0.95, "to": 1.15, "steps": 3},
                {"angle": [2, 1, 3], "from": 100.0, "to": 115.0, "steps": 3},
            ]
        },
        maxcycles=200,
        calc=uma_calc,
    )

    with open(os.path.join(out, "scan", "conf_0000_scan.csv")) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 5, "3 + 3 points sharing one handover geometry"
    # coordinate 1 is held at its end value while coordinate 2 is scanned
    for r in rows[2:]:
        assert float(r["d_1_2_actual"]) == pytest.approx(1.15, abs=1e-3)
    assert float(rows[-1]["a_2_1_3_actual"]) == pytest.approx(115.0, abs=1e-1)
    assert energies(csv_path)[0]["route"] == "SCAN"


def test_water_two_bond_grid_is_symmetric(tmp_path, h2o_xyz, uma_calc, energies):
    """A 3x3 grid over both O-H bonds — the full-surface mode.

    Water's two O-H bonds are equivalent, so E(r1, r2) must equal E(r2, r1). That
    symmetry is a much stronger check than "the numbers look plausible": it would
    break if the grid mislabelled its axes, applied a constraint to the wrong
    atoms, or let one coordinate drift while the other was being scanned.
    """
    from umadriver.ensemble import run_conformer_workflow

    out = str(tmp_path / "grid")
    csv_path = run_conformer_workflow(
        h2o_xyz,
        out_dir=out,
        scan={
            "mode": "grid",
            "coords": [
                {"distance": [1, 2], "from": 0.90, "to": 1.10, "steps": 3},
                {"distance": [1, 3], "from": 0.90, "to": 1.10, "steps": 3},
            ],
        },
        maxcycles=200,
        calc=uma_calc,
    )

    with open(os.path.join(out, "scan", "conf_0000_scan.csv")) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 9, "3 x 3 grid"

    surface = {
        (round(float(r["d_1_2_actual"]), 3), round(float(r["d_1_3_actual"]), 3)): float(
            r["energy_Eh"]
        )
        for r in rows
    }
    assert len(surface) == 9

    for (r1, r2), e in surface.items():
        mirrored = surface.get((r2, r1))
        assert mirrored is not None, f"grid is not square at ({r1}, {r2})"
        assert e == pytest.approx(mirrored, abs=1e-5), (
            f"E({r1}, {r2}) = {e:.6f} but E({r2}, {r1}) = {mirrored:.6f}; water's "
            "two O-H bonds are equivalent so the surface must be symmetric"
        )

    # The grid samples 0.90/1.00/1.10 A, and 1.00 is the value nearest water's
    # ~0.96 A equilibrium, so the surface bottoms out in the middle — away from
    # every edge, which is what makes this a real minimum rather than a corner.
    best = min(surface, key=surface.get)
    assert best == (1.0, 1.0), f"minimum of the grid landed at {best}"
    assert energies(csv_path)[0]["route"] == "SCAN"


def test_scan_max_is_written_for_reoptimization(tmp_path, h2o_xyz, uma_calc):
    """The advertised workflow: scan, then hand the maximum to --optts."""
    from umadriver.ensemble import run_conformer_workflow

    out = str(tmp_path / "scan")
    run_conformer_workflow(
        h2o_xyz,
        out_dir=out,
        scan=[1, 2, 0.95, 1.6, 5],
        maxcycles=200,
        calc=uma_calc,
    )

    guess = os.path.join(out, "scan", "conf_0000_scan_max.xyz")
    assert os.path.isfile(guess)
    assert len(ase_read(guess)) == 3


def test_scan_and_optts_are_rejected_together(tmp_path, h2o_xyz, uma_calc):
    from umadriver.ensemble import run_conformer_workflow

    with pytest.raises(ValueError, match="different routes"):
        run_conformer_workflow(
            h2o_xyz,
            out_dir=str(tmp_path / "bad"),
            scan=[1, 2, 0.9, 1.2, 3],
            optts=True,
            calc=uma_calc,
        )


def test_a_sharded_grid_agrees_with_itself_end_to_end(
    tmp_path, h2o_xyz, energies, requires_model
):
    """The whole two-phase path through the scheduler, with the real model.

    Water's two O-H bonds are equivalent, so the surface must be symmetric under
    swapping them. Here the two halves of that symmetry are computed by *different
    shards*, seeded independently — so an asymmetry means a shard relaxed into a
    different basin, which is the one failure mode no unit test can see.

    Runs on CPU so the scheduler takes its serial loop instead of spawning workers,
    which is fragile under pytest; sharding is requested explicitly because `auto`
    correctly declines to shard when there are no GPUs.
    """
    from umadriver.batch import BatchCommon, run_batch_from_glob

    out_root = str(tmp_path / "runs")
    run_batch_from_glob(
        [h2o_xyz],
        BatchCommon(
            out_root=out_root, device="cpu", resume=False, scan_shards=2
        ),
        scan={
            "mode": "grid",
            "coords": [
                {"distance": [1, 2], "from": 0.90, "to": 1.20, "steps": 4},
                {"distance": [1, 3], "from": 0.90, "to": 1.20, "steps": 4},
            ],
        },
        opt_mode="Loose",
        maxcycles=30,
    )

    ens = os.path.join(out_root, "h2o.ensemble")
    assert os.path.isdir(os.path.join(ens, "shard00"))
    assert os.path.isdir(os.path.join(ens, "shard01"))

    # The merged profile lands where an unsharded run would have put it, so every
    # documented path keeps working.
    merged_csv = os.path.join(ens, "scan", "conf_0000_scan.csv")
    assert os.path.isfile(merged_csv), "shards were never merged"
    assert os.path.isfile(os.path.join(ens, "scan", "conf_0000_scan_max.xyz"))

    with open(merged_csv) as f:
        rows = list(csv.DictReader(f))
    assert [int(r["point"]) for r in rows] == list(range(16))
    assert all(r["converged"] == "True" for r in rows)

    E = {
        (round(float(r["d_1_2_target"]), 3), round(float(r["d_1_3_target"]), 3)): float(
            r["energy_Eh"]
        )
        for r in rows
    }
    for (r1, r2), e in E.items():
        assert e == pytest.approx(E[(r2, r1)], abs=1e-5), (
            f"E({r1},{r2}) and E({r2},{r1}) disagree — a shard found a different "
            f"basin than its mirror image"
        )

    (row,) = energies(os.path.join(ens, "energies.csv"))
    assert row["route"] == "SCAN"
    assert row["converged"] == "True"
