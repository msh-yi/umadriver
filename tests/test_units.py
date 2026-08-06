"""Pure-function tests: no calculator, no GPU, milliseconds.

These cover the plumbing that sits either side of the model — ranking, XYZ
splitting, manifest merging, split-ensemble aggregation, convergence cutoffs.
"""

from __future__ import annotations

import csv
import math
import os

import pytest

from umadriver.ensemble import ENERGIES_FIELDS, rank_by_energy
from umadriver.batch import (
    BatchCommon,
    _aggregate_split_ensembles,
    _expand_xyz_inputs,
    _job_out_dir,
    _split_xyz_into_structures,
)
from umadriver.utils import gaussian_converged, gaussian_cutoffs


# ---------------------------------------------------------------- ranking
def test_rank_by_energy_orders_ascending_and_sets_rel_kcal():
    rows = [
        {"tag": "b", "energy_Eh": -10.0},
        {"tag": "a", "energy_Eh": -10.5},
        {"tag": "c", "energy_Eh": -9.0},
    ]
    ranked, e0 = rank_by_energy(rows)

    assert [r["tag"] for r in ranked] == ["a", "b", "c"]
    assert e0 == pytest.approx(-10.5)
    assert ranked[0]["rel_kcal"] == pytest.approx(0.0)
    # 0.5 Eh above the reference, in kcal/mol
    assert ranked[1]["rel_kcal"] == pytest.approx(0.5 * 627.5, rel=1e-3)
    assert ranked[2]["rel_kcal"] > ranked[1]["rel_kcal"]


def test_rank_by_energy_sorts_nonfinite_last_without_crashing():
    rows = [
        {"tag": "nan", "energy_Eh": float("nan")},
        {"tag": "good", "energy_Eh": -5.0},
        {"tag": "junk", "energy_Eh": "not-a-number"},
    ]
    ranked, e0 = rank_by_energy(rows)

    assert ranked[0]["tag"] == "good"
    assert e0 == pytest.approx(-5.0)
    assert {r["tag"] for r in ranked[1:]} == {"nan", "junk"}
    for r in ranked[1:]:
        assert math.isnan(r["rel_kcal"])


def test_rank_by_energy_all_nonfinite_returns_none_reference():
    rows = [{"tag": "x", "energy_Eh": float("nan")}]
    ranked, e0 = rank_by_energy(rows)
    assert e0 is None
    assert math.isnan(ranked[0]["rel_kcal"])


# ---------------------------------------------------------------- xyz splitting
def _write(path, text):
    with open(path, "w") as f:
        f.write(text)
    return str(path)


ONE_FRAME = "3\nH2O\nO 0.0 0.0 0.0\nH 0.0 0.8 0.6\nH 0.0 -0.8 0.6\n"


def test_split_single_structure(tmp_path):
    p = _write(tmp_path / "one.xyz", ONE_FRAME)
    structs = _split_xyz_into_structures(p)

    assert len(structs) == 1
    assert structs[0][1] == "conf0000"
    assert structs[0][0].splitlines()[0].strip() == "3"


def test_split_three_structures_labels_and_content(tmp_path):
    p = _write(tmp_path / "three.xyz", ONE_FRAME * 3)
    structs = _split_xyz_into_structures(p)

    assert [lbl for _, lbl in structs] == ["conf0000", "conf0001", "conf0002"]
    for content, _ in structs:
        lines = content.splitlines()
        assert lines[0].strip() == "3"
        assert len(lines) == 5  # count + comment + 3 atoms


def test_split_tolerates_trailing_blank_lines(tmp_path):
    p = _write(tmp_path / "trailing.xyz", ONE_FRAME * 2 + "\n\n  \n")
    structs = _split_xyz_into_structures(p)
    assert len(structs) == 2


def test_split_roundtrips_through_ase(tmp_path):
    """Every split chunk must be a readable XYZ — the splitter is hand-rolled."""
    ase_read = pytest.importorskip("ase.io").read

    p = _write(tmp_path / "three.xyz", ONE_FRAME * 3)
    for i, (content, label) in enumerate(_split_xyz_into_structures(p)):
        member = _write(tmp_path / f"{label}.xyz", content)
        atoms = ase_read(member)
        assert len(atoms) == 3


# ---------------------------------------------------------------- job paths
def test_job_out_dir_default_and_explicit():
    assert _job_out_dir("runs", "/a/b/mol.xyz", None) == os.path.join(
        "runs", "mol.ensemble"
    )
    assert _job_out_dir("runs", "/a/b/mol.xyz", "/explicit/dir") == "/explicit/dir"


def test_expand_xyz_inputs_globs(tmp_path):
    for n in ("a.xyz", "b.xyz"):
        _write(tmp_path / n, ONE_FRAME)
    got = _expand_xyz_inputs([str(tmp_path / "*.xyz")])
    assert [os.path.basename(p) for p in got] == ["a.xyz", "b.xyz"]


def test_expand_xyz_inputs_passes_through_plain_paths():
    assert _expand_xyz_inputs(["/a/b.xyz"]) == ["/a/b.xyz"]


# ---------------------------------------------------------------- aggregation
def _member(ens_dir, label, energy, tag="conf_0000"):
    """Write a one-row member energies.csv the way a split job would."""
    d = os.path.join(ens_dir, label)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "energies.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ENERGIES_FIELDS)
        w.writeheader()
        w.writerow(
            {
                "rank": 1,
                "index": 0,
                "tag": tag,
                "route": "OPT",
                "converged": True,
                "steps": 5,
                "energy_Eh": energy,
                "energy_kcal": energy * 627.5,
                "rel_kcal": 0.0,
                "gibbs_Eh": "",
                "gibbs_kcal": "",
                "n_imag": "",
                "imag_ok": "",
            }
        )
    with open(os.path.join(d, "optimized_ranked.xyz"), "w") as f:
        f.write(ONE_FRAME)
    return {"out_dir": d, "_original_xyz": "/src/mol.xyz"}


def test_aggregate_split_ensembles_writes_ranked_csv(tmp_path):
    """Regression test for the missing ``import csv`` in batch.py.

    Before that fix this raised NameError, which the caller swallowed — so the
    top-level ensemble CSV silently never appeared.
    """
    ens = str(tmp_path / "mol.ensemble")
    jobs = [
        _member(ens, "conf0000", -10.0),
        _member(ens, "conf0001", -10.5),
        _member(ens, "conf0002", -9.0),
    ]

    _aggregate_split_ensembles(jobs)

    out_csv = os.path.join(ens, "energies.csv")
    assert os.path.isfile(out_csv), "aggregate energies.csv was not written"

    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3
    # ranked ascending by energy, and relabelled with the split label
    assert [r["tag"] for r in rows] == ["conf0001", "conf0000", "conf0002"]
    assert [r["rank"] for r in rows] == ["1", "2", "3"]
    assert [r["index"] for r in rows] == ["1", "0", "2"]
    assert float(rows[0]["rel_kcal"]) == pytest.approx(0.0)
    assert float(rows[1]["rel_kcal"]) > 0.0


def test_aggregate_split_ensembles_concatenates_xyz_in_ranked_order(tmp_path):
    ens = str(tmp_path / "mol.ensemble")
    jobs = [_member(ens, "conf0000", -10.0), _member(ens, "conf0001", -10.5)]

    _aggregate_split_ensembles(jobs)

    out_xyz = os.path.join(ens, "optimized_ranked.xyz")
    assert os.path.isfile(out_xyz)
    # two 5-line frames concatenated
    assert len([l for l in open(out_xyz) if l.strip()]) == 10


def test_aggregate_ignores_jobs_without_original_xyz(tmp_path):
    """Non-split jobs already own their ensemble CSV; they must not be regrouped."""
    ens = str(tmp_path / "mol.ensemble")
    job = _member(ens, "conf0000", -10.0)
    job.pop("_original_xyz")

    _aggregate_split_ensembles([job])

    assert not os.path.exists(os.path.join(ens, "energies.csv"))


def test_aggregate_skips_missing_member_csv(tmp_path):
    ens = str(tmp_path / "mol.ensemble")
    jobs = [_member(ens, "conf0000", -10.0)]
    jobs.append({"out_dir": os.path.join(ens, "conf0001"), "_original_xyz": "/src/mol.xyz"})

    _aggregate_split_ensembles(jobs)  # must not raise

    with open(os.path.join(ens, "energies.csv"), newline="") as f:
        assert len(list(csv.DictReader(f))) == 1


# ---------------------------------------------------------------- convergence
def test_gaussian_cutoffs_tighten_monotonically():
    modes = ["Loose", "Normal", "Tight", "VeryTight"]
    grms = [gaussian_cutoffs(m).grms for m in modes]
    assert grms == sorted(grms, reverse=True)


def test_gaussian_converged_requires_all_four_criteria():
    cuts = gaussian_cutoffs("Normal")
    tiny = cuts.grms / 10, cuts.gmax / 10, cuts.drms / 10, cuts.dmax / 10

    assert gaussian_converged(*tiny, cuts)
    # any single criterion over the line blocks convergence
    for i in range(4):
        vals = list(tiny)
        vals[i] *= 1000
        assert not gaussian_converged(*vals, cuts)


# ---------------------------------------------------------------- manifest merge
def test_batch_common_fields_are_stripped_from_workflow_overrides():
    """_BATCH_KEYS must cover every BatchCommon field, or scheduler config leaks
    into run_conformer_workflow() as an unexpected kwarg."""
    from umadriver.batch import _BATCH_KEYS

    assert set(BatchCommon.__dataclass_fields__) <= _BATCH_KEYS


# ------------------------------------------------------------------ CLI routes
def _run_cli(argv, monkeypatch):
    """Parse argv through the real CLI, capturing what it would dispatch."""
    import sys

    import umadriver.driver as drv

    captured = {}

    def _fake_batch(inputs, common, **overrides):
        captured["overrides"] = overrides
        return []

    monkeypatch.setattr(drv, "run_batch_from_glob", _fake_batch)
    monkeypatch.setattr(drv, "initialize_env", lambda: None)
    monkeypatch.setattr(sys, "argv", ["umadriver", *argv])
    drv.main()
    return captured["overrides"]


def test_optts_on_the_cli_names_sella(monkeypatch):
    """--optts must dispatch an optimizer, not None. Passing None left the route
    and the optimizer disagreeing, which is where `optimizer: null` + `optts:
    true` got its reputation."""
    overrides = _run_cli(["mol.xyz", "--optts"], monkeypatch)

    assert overrides["optts"] is True
    assert overrides["optimizer"] == "Sella"


def test_sp_and_optts_conflict(monkeypatch):
    """--sp means "do not move this geometry" and --optts moves it. Letting one
    win silently is how a frequency job optimizes away the structure it was
    given."""
    with pytest.raises(SystemExit):
        _run_cli(["mol.xyz", "--sp", "--optts"], monkeypatch)


def test_optts_with_another_optimizer_is_rejected(monkeypatch):
    with pytest.raises(SystemExit):
        _run_cli(["mol.xyz", "--optts", "--opt", "LBFGS"], monkeypatch)


def test_freq_ts_reaches_the_workflow(monkeypatch):
    """--sp --freq --freq-ts is the documented freq-only-on-a-saddle recipe."""
    overrides = _run_cli(["ts.xyz", "--sp", "--freq", "--freq-ts"], monkeypatch)

    assert overrides["optimizer"] is None
    assert overrides["optts"] is False
    assert overrides["do_freq"] is True
    assert overrides["freq_ts"] is True


def test_freq_ts_defaults_to_letting_the_route_decide(monkeypatch):
    """Unset must stay None: False would tell a TS job to expect zero imaginary
    modes and mark every real saddle as imag_ok=False."""
    overrides = _run_cli(["mol.xyz", "--freq"], monkeypatch)

    assert overrides["freq_ts"] is None


def test_scan_reaches_the_workflow_verbatim(monkeypatch):
    """The CLI hands the raw values through; parse_scan_spec does the 1-based
    conversion once, in one place, for both CLI and manifests."""
    overrides = _run_cli(["mol.xyz", "--scan", "1", "2", "0.9", "1.6", "8"], monkeypatch)

    assert overrides["scan"] == {
        "mode": "sequential",
        "coords": [["1", "2", "0.9", "1.6", "8"]],
    }


def test_repeated_scan_flags_become_several_coordinates(monkeypatch):
    """One --scan per coordinate, combined by --scan-mode."""
    overrides = _run_cli(
        [
            "mol.xyz",
            "--scan", "1", "2", "0.9", "1.6", "8",
            "--scan", "2", "1", "3", "100", "140", "8",
            "--scan-mode", "concerted",
        ],
        monkeypatch,
    )

    assert overrides["scan"]["mode"] == "concerted"
    assert len(overrides["scan"]["coords"]) == 2

    from umadriver.scan import parse_scan_spec

    spec = parse_scan_spec(overrides["scan"])
    assert [c.kind for c in spec.coords] == ["distance", "angle"]
    assert spec.npoints == 8, "concerted traces one path, it does not build a grid"


def test_scan_mode_without_scan_is_rejected(monkeypatch):
    """Silently ignoring it would let someone think a concerted scan ran."""
    with pytest.raises(SystemExit):
        _run_cli(["mol.xyz", "--opt", "Sella", "--scan-mode", "concerted"], monkeypatch)


def test_concerted_with_mismatched_steps_fails_at_argv(monkeypatch):
    with pytest.raises(SystemExit):
        _run_cli(
            [
                "mol.xyz",
                "--scan", "1", "2", "0.9", "1.6", "8",
                "--scan", "2", "1", "3", "100", "140", "5",
                "--scan-mode", "concerted",
            ],
            monkeypatch,
        )


def test_scan_defaults_to_none(monkeypatch):
    assert _run_cli(["mol.xyz", "--opt", "Sella"], monkeypatch)["scan"] is None


def test_scan_and_optts_conflict(monkeypatch):
    with pytest.raises(SystemExit):
        _run_cli(["mol.xyz", "--scan", "1", "2", "0.9", "1.6", "8", "--optts"], monkeypatch)


def test_scan_and_sp_conflict(monkeypatch):
    """A scan relaxes at every point, so --sp ("don't move it") cannot hold."""
    with pytest.raises(SystemExit):
        _run_cli(["mol.xyz", "--scan", "1", "2", "0.9", "1.6", "8", "--sp"], monkeypatch)


def test_a_bad_scan_spec_fails_before_the_batch_starts(monkeypatch):
    """Atom 0 does not exist in 1-based numbering. Catching it at argv time beats
    discovering it inside a worker after the model has loaded."""
    with pytest.raises(SystemExit):
        _run_cli(["mol.xyz", "--scan", "0", "2", "0.9", "1.6", "8"], monkeypatch)
