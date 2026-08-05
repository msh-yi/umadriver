"""The example manifests in examples/.

They are documentation, which means they rot silently: a renamed keyword argument
leaves a manifest that looks authoritative and dies at runtime with an unexpected
kwarg, and a new user cannot tell whether they mistyped something or the example
is stale. These tests parse each file through the same code path the CLI uses and
check every key against the real signature. No GPU needed.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from umadriver.batch import BatchCommon, _load_manifest  # noqa: E402
from umadriver.ensemble import run_conformer_workflow  # noqa: E402

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
MANIFESTS = sorted(EXAMPLES.glob("*.yaml"))

WORKFLOW_KEYS = set(inspect.signature(run_conformer_workflow).parameters)
BATCH_KEYS = set(BatchCommon.__dataclass_fields__)


def test_there_are_examples_to_check():
    """Guards the glob: an empty match would make every test below vacuous."""
    assert MANIFESTS, f"no manifests found in {EXAMPLES}"


@pytest.mark.parametrize("path", MANIFESTS, ids=lambda p: p.name)
def test_manifest_parses(path):
    cfg = _load_manifest(str(path))
    assert isinstance(cfg, dict), "top level must be a mapping"
    assert isinstance(cfg.get("jobs"), list) and cfg["jobs"], "needs a non-empty jobs list"


@pytest.mark.parametrize("path", MANIFESTS, ids=lambda p: p.name)
def test_every_key_is_real(path):
    """A typo here is invisible until a job dies mid-batch."""
    cfg = _load_manifest(str(path))

    for key in cfg.get("common", {}) or {}:
        assert key in BATCH_KEYS | WORKFLOW_KEYS, f"common: unknown key {key!r}"

    for i, job in enumerate(cfg["jobs"]):
        assert "xyz" in job, f"job {i} has no xyz"
        flat = set(job) - {"xyz", "out_dir", "overrides"}
        for key in flat | set(job.get("overrides", {}) or {}):
            assert key in BATCH_KEYS | WORKFLOW_KEYS, f"job {i}: unknown key {key!r}"


@pytest.mark.parametrize("path", MANIFESTS, ids=lambda p: p.name)
def test_inputs_exist(path):
    """Examples must run as written — a new user's first command should work."""
    root = EXAMPLES.parent
    for job in _load_manifest(str(path))["jobs"]:
        assert (root / job["xyz"]).is_file(), f"missing input: {job['xyz']}"


@pytest.mark.parametrize("path", MANIFESTS, ids=lambda p: p.name)
def test_freq_only_jobs_do_not_secretly_optimize(path):
    """`optts: true` runs a saddle search whatever `optimizer` says, so a job that
    means "frequencies on this exact geometry" must not set it. This is the
    documented trap in example 05; the examples themselves have to be clean."""
    for i, job in enumerate(_load_manifest(str(path))["jobs"]):
        settings = {**job, **(job.get("overrides", {}) or {})}
        if not settings.get("do_freq") or "optimizer" not in settings:
            continue
        if settings["optimizer"] is None and settings.get("freq_ts") is not None:
            assert not settings.get("optts"), (
                f"job {i} in {path.name} sets freq_ts (frequency-only intent) "
                "alongside optts=true, which re-optimizes the geometry"
            )
