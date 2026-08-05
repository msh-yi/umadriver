"""TS optimization with Sella.

The small fixture is the HCN <-> HNC isomerization saddle: 3 atoms, neutral,
closed shell, one unambiguous imaginary mode, and frequencies cost only 18
gradient calls.
"""

from __future__ import annotations

import os
import re

import pytest

from umadriver.ensemble import run_conformer_workflow

FREQ_LINE = re.compile(r"^\s*(\d+):\s+(-?\d+\.\d+)\s+cm\*\*-1", re.M)


def parse_frequencies(out_path: str) -> list[float]:
    """Pull the frequency column out of the ORCA-style output."""
    txt = open(out_path).read()
    if "VIBRATIONAL FREQUENCIES" not in txt:
        # The frequency job failed and the section was never written. Say that,
        # instead of an IndexError from splitting on a missing marker.
        raise AssertionError(
            f"{out_path} has no VIBRATIONAL FREQUENCIES section — the frequency "
            "calculation did not complete (check the log for a swallowed error)."
        )
    block = txt.split("VIBRATIONAL FREQUENCIES", 1)[1].split("NORMAL MODES", 1)[0]
    return [float(m.group(2)) for m in FREQ_LINE.finditer(block)]


def test_ts_optimization_finds_one_imaginary_mode(
    tmp_path, hcn_ts_xyz, uma_calc, energies
):
    out = str(tmp_path / "ts")
    csv_path = run_conformer_workflow(
        hcn_ts_xyz,
        out_dir=out,
        charge=0,
        mult=1,
        optts=True,
        maxcycles=300,
        do_freq=True,
        calc=uma_calc,
    )

    r = energies(csv_path)[0]
    assert r["route"] == "TS"
    assert r["converged"] == "True", "TS optimization did not converge"
    assert int(r["n_imag"]) == 1, f"expected 1 imaginary mode, got {r['n_imag']}"
    assert r["imag_ok"] == "True"


def test_ts_imaginary_mode_is_substantial(tmp_path, hcn_ts_xyz, uma_calc):
    """A near-zero 'imaginary' mode is numerical noise, not a reaction coordinate."""
    out = str(tmp_path / "ts")
    run_conformer_workflow(
        hcn_ts_xyz,
        out_dir=out,
        optts=True,
        maxcycles=300,
        do_freq=True,
        calc=uma_calc,
    )

    freqs = parse_frequencies(os.path.join(out, "freq_out", "conf_0000.out"))
    negative = [f for f in freqs if f < 0.0]

    assert len(negative) == 1, f"expected exactly one negative frequency, got {negative}"
    assert abs(negative[0]) > 100.0, (
        f"imaginary mode {negative[0]:.1f} cm^-1 is too small to be a real "
        "reaction coordinate"
    )


def test_ts_route_ignores_opt_flag(tmp_path, hcn_ts_xyz, uma_calc, energies):
    """--optts alone selects the TS route; --opt is not required."""
    csv_path = run_conformer_workflow(
        hcn_ts_xyz,
        out_dir=str(tmp_path / "ts_noopt"),
        optimizer=None,
        optts=True,
        maxcycles=300,
        do_freq=False,
        calc=uma_calc,
    )
    assert energies(csv_path)[0]["route"] == "TS"


@pytest.mark.big
def test_catalyst_ts_completes(tmp_path, catalyst_ts_xyz, uma_calc, energies):
    """170-atom catalyst TS.

    The fixture is an already-converged saddle, so this asserts a *clean* TS rather
    than merely a completed run — a regression in the TS route, or a fixture swap,
    shows up as n_imag != 1 instead of passing silently.
    """
    out = str(tmp_path / "cat")
    csv_path = run_conformer_workflow(
        catalyst_ts_xyz,
        out_dir=out,
        charge=0,
        mult=1,
        optts=True,
        maxcycles=300,
        do_freq=True,
        calc=uma_calc,
    )

    rows = energies(csv_path)
    assert len(rows) == 1
    r = rows[0]

    assert r["route"] == "TS"
    assert r["converged"] == "True"
    assert float(r["energy_Eh"]) < 0.0
    assert int(r["n_imag"]) == 1, f"expected a clean saddle, got n_imag={r['n_imag']}"
    assert r["imag_ok"] == "True"

    freqs = parse_frequencies(os.path.join(out, "freq_out", "conf_0000.out"))
    assert len(freqs) == 3 * 170
    negative = [f for f in freqs if f < 0.0]
    assert len(negative) == 1
    assert abs(negative[0]) > 100.0
