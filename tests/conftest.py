"""Shared fixtures for the umadriver test suite.

The suite runs against the *real* UMA model — there are no mock calculators. That
means every test outside ``test_units.py`` needs a GPU and a populated fairchem
cache; when either is missing the tests skip rather than fail.

The one thing that keeps this affordable is the session-scoped ``uma_calc``
fixture: ``run_conformer_workflow`` accepts a ``calc=`` injection, so the model is
loaded once for the whole session instead of once per test.
"""

from __future__ import annotations

import os

# Must happen before sella / fairchem are imported anywhere.
#
# The CLI pins JAX to CPU in driver._early_parse_threads (which runs at import
# time, ahead of the sella import) so that JAX does not reserve GPU memory UMA
# needs. Sella's internal-coordinate machinery runs on JAX, so a test session that
# skips this pinning exercises a configuration the driver never produces — and on
# this cluster it hard-crashes, because the installed jaxlib is built against a
# newer cuDNN than the runtime provides.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import csv
import shutil
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"


# --------------------------------------------------------------------------
# CLI options / markers
# --------------------------------------------------------------------------
def pytest_addoption(parser):
    parser.addoption(
        "--runbig",
        action="store_true",
        default=False,
        help="Also run the slow 170-atom catalyst cases.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runbig"):
        return
    skip_big = pytest.mark.skip(reason="needs --runbig")
    for item in items:
        if "big" in item.keywords:
            item.add_marker(skip_big)


# --------------------------------------------------------------------------
# Environment hygiene
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def isolated_scratch(tmp_path, monkeypatch):
    """Keep vibration/job scratch inside the test's tmp_path.

    The production default scratch root is the fairchem cache dir, which is why it
    accumulates ``ensemble-*`` directories. Tests must never write there.
    """
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("UMA_SCRATCH_ROOT", str(scratch))
    yield scratch


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------
def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _cache_populated() -> bool:
    """Honest check: resolves a real blob, so a symlink tree left behind by a
    scratch purge does not read as a usable cache."""
    from umadriver.constants import DEFAULT_FAIRCHEM_CACHE
    from umadriver.utils import cache_has_files

    return cache_has_files(DEFAULT_FAIRCHEM_CACHE)


def _model_unavailable_reason() -> str | None:
    if not _cuda_available():
        return "no CUDA device available"
    if not _cache_populated():
        return (
            "no usable UMA checkpoint in the fairchem cache "
            "(run `hf auth login`, then let a run re-download it)"
        )
    return None


@pytest.fixture
def requires_model():
    """For tests that let the workflow build its own calculator.

    Those never touch `uma_calc`, and the batch scheduler turns a model-load
    failure into a job status rather than an exception — so without this guard
    they fail with a confusing missing-output assertion instead of skipping.
    """
    reason = _model_unavailable_reason()
    if reason:
        pytest.skip(reason)


@pytest.fixture(scope="session")
def uma_calc():
    """A single UMA calculator shared by every test in the session.

    Loading uma-m-1p1 costs ~30-60 s; doing it per test would dominate the suite.
    """
    reason = _model_unavailable_reason()
    if reason:
        pytest.skip(reason)

    from umadriver.utils import build_calculator, resolve_device

    return build_calculator(
        model="uma-m-1p1",
        device=resolve_device("cuda"),
        cache_dir=None,
        use_local_scratch=False,
    )


# --------------------------------------------------------------------------
# Input fixtures
# --------------------------------------------------------------------------
def _staged(tmp_path: Path, name: str) -> str:
    """Copy a data fixture into tmp_path.

    Jobs are named after their input file's stem, and several tests run the same
    input into different out_dirs; staging keeps the repo copy pristine.
    """
    dst = tmp_path / name
    shutil.copy(DATA / name, dst)
    return str(dst)


@pytest.fixture
def h2o_xyz(tmp_path):
    return _staged(tmp_path, "h2o.xyz")


@pytest.fixture
def h2o_ens_xyz(tmp_path):
    return _staged(tmp_path, "h2o_ens.xyz")


@pytest.fixture
def hcn_ts_xyz(tmp_path):
    return _staged(tmp_path, "hcn_ts.xyz")


@pytest.fixture
def catalyst_ts_xyz(tmp_path):
    """170-atom catalyst TS guess — the --runbig fixture."""
    return _staged(tmp_path, "pudovik_test_TS.xyz")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def read_energies(csv_path) -> list[dict]:
    """Read an energies.csv into a list of dicts, ranked order preserved."""
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def energies():
    return read_energies
