"""Relaxed bond scans.

Two layers, as elsewhere in this suite:

1. Parsing and the scan mechanics, with no model. ``run_bond_scan`` takes its
   minimizer as an argument, so a plain ASE optimizer on EMT exercises the whole
   loop — the displacement schedule, the constraint handling, the chaining and the
   output files — in a second.
2. The real thing on water with UMA, which is the only way to check that a scan
   through a bond produces a physically sensible profile.
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

from umadriver.scan import ScanSpec, parse_scan_spec, run_bond_scan


# ---------------------------------------------------------------- parsing
def test_atoms_are_one_based():
    """The user-facing convention. Getting this wrong scans a different bond and
    every number downstream still looks reasonable, so it is pinned explicitly."""
    spec = parse_scan_spec([1, 2, 0.9, 1.6, 8])

    assert (spec.i, spec.j) == (0, 1), "should convert to ASE's 0-based indexing"
    assert (spec.i1, spec.j1) == (1, 2), "should report back in the user's numbering"


def test_mapping_and_sequence_forms_agree():
    seq = parse_scan_spec([3, 7, 2.8, 1.4, 15])
    mapping = parse_scan_spec({"i": 3, "j": 7, "from": 2.8, "to": 1.4, "steps": 15})

    assert seq == mapping


def test_targets_span_the_requested_range():
    spec = parse_scan_spec([1, 2, 1.0, 2.0, 5])

    np.testing.assert_allclose(spec.targets, [1.0, 1.25, 1.5, 1.75, 2.0])
    assert spec.step_A == pytest.approx(0.25)


def test_a_decreasing_range_is_allowed():
    """Scanning inward (association) is as valid as scanning outward."""
    spec = parse_scan_spec([1, 2, 3.0, 1.4, 5])

    assert spec.targets[0] > spec.targets[-1]
    assert spec.step_A == pytest.approx(0.4)


@pytest.mark.parametrize(
    "value,message",
    [
        ([0, 2, 0.9, 1.6, 8], "numbered from 1"),
        ([1, 1, 0.9, 1.6, 8], "must differ"),
        ([1, 2, 0.9, 1.6, 1], "at least 2 points"),
        ([1, 2, 1.5, 1.5, 8], "nothing to scan"),
        ([1, 2, -0.9, 1.6, 8], "must be positive"),
        ([1, 2, "x", 1.6, 8], "two atom numbers"),
        ([1, 2, 0.9, 1.6], "5 values"),
        ({"i": 1, "j": 2, "from": 0.9, "to": 1.6}, "missing"),
        ({"i": 1, "j": 2, "from": 0.9, "to": 1.6, "steps": 8, "nope": 1}, "unknown"),
    ],
)
def test_bad_specs_are_rejected(value, message):
    with pytest.raises(ValueError, match=message):
        parse_scan_spec(value)


def test_out_of_range_atom_is_caught_against_the_structure(tmp_path):
    """Range depends on the molecule, so it cannot be checked at parse time."""
    atoms = molecule("H2O")
    atoms.calc = EMT()

    with pytest.raises(ValueError, match="numbered 1..3"):
        run_bond_scan(
            atoms,
            parse_scan_spec([1, 4, 0.9, 1.2, 3]),
            relax=lambda a: (True, 0, 0.0),
            out_dir=str(tmp_path),
            tag="conf_0000",
        )


# ---------------------------------------------------------------- mechanics
def _lbfgs_relax(atoms):
    """Stand-in for the workflow's minimizer. LBFGS honours FixBondLength to
    ~1e-13 A, so any drift the scan reports is the scan's own doing."""
    opt = LBFGS(atoms, logfile=None)
    opt.run(fmax=0.05, steps=50)
    return True, 12, float(atoms.get_potential_energy())


def _scan(tmp_path, spec_values=(1, 2, 1.0, 1.6, 7), name="C2H6", **kw):
    atoms = molecule(name)
    atoms.calc = EMT()
    records = run_bond_scan(
        atoms,
        parse_scan_spec(list(spec_values)),
        relax=kw.pop("relax", _lbfgs_relax),
        out_dir=str(tmp_path / "scan"),
        tag="conf_0000",
        **kw,
    )
    return atoms, records


def test_every_point_lands_on_its_target(tmp_path):
    _, records = _scan(tmp_path)

    assert len(records) == 7
    for r in records:
        assert r["actual_A"] == pytest.approx(r["target_A"], abs=1e-6), (
            f"point {r['point']} asked for {r['target_A']:.4f} A and got "
            f"{r['actual_A']:.4f} A"
        )


def test_the_scan_relaxes_rather_than_just_stretching(tmp_path):
    """A scan that only moved the two atoms and skipped the minimization would
    still hit every target — so check the rest of the molecule actually moved."""
    seen = {}

    def relax(atoms):
        before = atoms.get_positions().copy()
        out = _lbfgs_relax(atoms)
        seen.setdefault("moved", []).append(
            float(np.abs(atoms.get_positions() - before).max())
        )
        return out

    _scan(tmp_path, relax=relax)

    assert max(seen["moved"]) > 0.01, "no relaxation happened between points"


def test_the_bond_is_constrained_while_relaxing(tmp_path):
    """The whole point: `relax` must be handed atoms that cannot change the bond.
    Without the constraint this is an ordinary optimization from a stretched
    start, and every point collapses to the same minimum.

    Asserted by effect rather than by constraint class: what matters is that a
    full minimization cannot move the bond, not which ASE object arranges that.
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
        assert after == pytest.approx(before, abs=1e-9), (
            f"minimization moved the constrained bond {before:.4f} -> {after:.4f} A"
        )
    # ...and the points really were at different distances, so the check is not
    # vacuously satisfied by nothing ever moving.
    assert len({round(b, 3) for b, _ in held}) == len(held)


def test_each_point_continues_from_the_previous_one(tmp_path):
    """Chaining is what keeps the path continuous. If every point restarted from
    the input, consecutive geometries would differ by far more."""
    starts = []

    def relax(atoms):
        starts.append(atoms.get_positions().copy())
        return _lbfgs_relax(atoms)

    _scan(tmp_path, spec_values=(1, 2, 1.5, 1.7, 5), relax=relax)

    # Ignore the two scanned atoms: set_distance moves those by construction.
    for prev, curr in zip(starts, starts[1:]):
        rest = np.abs(curr - prev)[2:]
        assert rest.max() < 0.2, "point did not start from the previous geometry"


def test_original_constraints_are_restored(tmp_path):
    atoms = molecule("C2H6")
    atoms.calc = EMT()
    atoms.set_constraint(FixAtoms(indices=[0]))

    run_bond_scan(
        atoms,
        parse_scan_spec([1, 2, 1.5, 1.6, 3]),
        relax=_lbfgs_relax,
        out_dir=str(tmp_path / "scan"),
        tag="conf_0000",
    )

    assert [type(c).__name__ for c in atoms.constraints] == ["FixAtoms"]


def test_atoms_are_left_at_the_final_point(tmp_path):
    atoms, records = _scan(tmp_path)

    assert atoms.get_distance(0, 1) == pytest.approx(records[-1]["actual_A"], abs=1e-9)


def test_constraint_survives_float32_forces():
    """UMA returns float32 forces. ASE's stock FixBondLength runs a RATTLE
    iteration to a hard-coded 1e-13 tolerance, which float32 can never reach — it
    exhausts maxiter and raises on *every* force evaluation, so a scan dies on its
    first optimizer step. EMT returns float64 and hides this completely.
    """
    from ase.constraints import FixBondLengths

    from umadriver.scan import fix_bond_length

    atoms = molecule("H2O")
    atoms.calc = EMT()
    f32 = np.asarray(atoms.get_forces(), dtype=np.float32)

    with pytest.raises(RuntimeError, match="Did not converge"):
        FixBondLengths([(0, 1)]).adjust_forces(atoms, f32.copy())

    ours = f32.copy()
    fix_bond_length(0, 1).adjust_forces(atoms, ours)  # must not raise

    assert ours.dtype == np.float32, "must write back in the caller's dtype"
    # and it actually did something: the component along the bond is projected out
    assert not np.allclose(ours, f32)


def test_float64_forces_still_take_the_normal_path():
    from umadriver.scan import fix_bond_length

    atoms = molecule("H2O")
    atoms.calc = EMT()
    f64 = atoms.get_forces()

    ours, stock = f64.copy(), f64.copy()
    fix_bond_length(0, 1).adjust_forces(atoms, ours)
    from ase.constraints import FixBondLengths

    FixBondLengths([(0, 1)]).adjust_forces(atoms, stock)

    np.testing.assert_allclose(ours, stock, atol=0)


# ---------------------------------------------------------------- outputs
def test_csv_has_one_row_per_point(tmp_path):
    _, records = _scan(tmp_path)

    with open(tmp_path / "scan" / "conf_0000_scan.csv") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == len(records)
    assert float(rows[0]["target_A"]) == pytest.approx(1.0)
    assert float(rows[-1]["target_A"]) == pytest.approx(1.6)
    # rel_kcal is measured from the lowest point, so some point must be 0
    assert min(abs(float(r["rel_kcal"])) for r in rows) == pytest.approx(0.0, abs=1e-9)


def test_trajectory_carries_every_frame(tmp_path):
    _, records = _scan(tmp_path)

    frames = ase_read(str(tmp_path / "scan" / "conf_0000_scan.xyz"), index=":")

    assert len(frames) == len(records)
    for frame, r in zip(frames, records):
        assert frame.get_distance(0, 1) == pytest.approx(r["actual_A"], abs=1e-6)


def test_max_file_is_the_highest_point(tmp_path):
    """It is the TS guess users feed to --optts, so it has to be the right frame."""
    _, records = _scan(tmp_path)

    top = max(records, key=lambda r: r["energy_Eh"])
    guess = ase_read(str(tmp_path / "scan" / "conf_0000_scan_max.xyz"))

    assert guess.get_distance(0, 1) == pytest.approx(top["actual_A"], abs=1e-6)


def test_unconverged_points_are_reported_not_hidden(tmp_path, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="uma.scan"):
        _scan(tmp_path, relax=lambda a: (False, 3, float(a.get_potential_energy())))

    assert any("did not converge" in r.message for r in caplog.records)


# ---------------------------------------------------------------- with UMA
def test_water_bond_scan_has_a_minimum_in_the_middle(tmp_path, h2o_xyz, uma_calc, energies):
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
        assert float(r["actual_A"]) == pytest.approx(float(r["target_A"]), abs=1e-3)

    e = [float(r["energy_Eh"]) for r in rows]
    lowest = int(np.argmin(e))
    assert 0 < lowest < 8, f"minimum landed at the edge (point {lowest}) of the scan"
    r_min = float(rows[lowest]["actual_A"])
    assert 0.9 < r_min < 1.05, f"O-H equilibrium looks wrong: {r_min:.3f} A"

    row = energies(csv_path)[0]
    assert row["route"] == "SCAN"
    # the summary row describes the last point, which is the saved geometry
    assert float(row["energy_Eh"]) == pytest.approx(e[-1], abs=1e-10)


def test_scan_max_can_be_reoptimized_as_a_ts(tmp_path, h2o_xyz, uma_calc):
    """The advertised workflow: scan, then hand the maximum to --optts. This only
    checks the handoff — that the file exists, is readable, and is a structure the
    workflow accepts — not that water has a TS."""
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
