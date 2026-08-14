"""Scheduling tests: how jobs are expanded and distributed.

These need no calculator — they exercise job-list construction, which is where the
multi-GPU behaviour is actually decided. The end-to-end runs that consume these job
lists live in test_multistructure.py.
"""

from __future__ import annotations

import os

import pytest

from umadriver.batch import (
    BatchCommon,
    _BATCH_KEYS,
    _expand_jobs_with_splitting,
    _worker_slots,
    _bind_gpu_env,
    run_batch_from_glob,
    run_batch_from_manifest,
)

ONE_FRAME = "3\nH2O\nO 0.0 0.0 0.0\nH 0.0 0.8 0.6\nH 0.0 -0.8 0.6\n"


def _xyz(tmp_path, name, n_frames):
    p = tmp_path / name
    p.write_text(ONE_FRAME * n_frames)
    return str(p)


def _capture_jobs(monkeypatch):
    """Intercept the job list the scheduler would run, without running anything."""
    seen = {}

    def fake_run(jobs_in, merged):
        seen["jobs"] = jobs_in
        seen["merged"] = merged
        return []

    monkeypatch.setattr("umadriver.batch._run_parallel_jobs", fake_run)
    return seen


# ------------------------------------------------------------------ splitting
def test_split_expands_one_job_per_structure(tmp_path):
    xyz = _xyz(tmp_path, "confs.xyz", 3)
    out_dir = str(tmp_path / "ens")
    jobs = [{"xyz": xyz, "out_dir": out_dir, "overrides": {"charge": -1}}]

    expanded = _expand_jobs_with_splitting(jobs, split_multi=True)

    assert len(expanded) == 3
    assert [os.path.basename(j["out_dir"]) for j in expanded] == [
        "conf0000",
        "conf0001",
        "conf0002",
    ]
    for j in expanded:
        # members sit *under* the job's out_dir, which is what lets the aggregator
        # group them back together
        assert os.path.dirname(j["out_dir"]) == out_dir
        assert j["_original_xyz"] == xyz
        assert j["_cleanup_xyz"] == j["xyz"]
        assert j["overrides"] == {"charge": -1}


def test_split_writes_temp_files_under_job_out_dir(tmp_path):
    """Temp XYZs must not land relative to cwd — manifests use absolute out_dirs."""
    xyz = _xyz(tmp_path, "confs.xyz", 2)
    out_dir = str(tmp_path / "deep" / "ens")
    jobs = [{"xyz": xyz, "out_dir": out_dir, "overrides": {}}]

    expanded = _expand_jobs_with_splitting(jobs, split_multi=True)

    for j in expanded:
        assert j["xyz"].startswith(os.path.join(out_dir, ".split"))
        assert os.path.isfile(j["xyz"])
        assert len(open(j["xyz"]).read().strip().splitlines()) == 5


def test_split_leaves_single_structure_jobs_untouched(tmp_path):
    xyz = _xyz(tmp_path, "one.xyz", 1)
    jobs = [{"xyz": xyz, "out_dir": str(tmp_path / "ens"), "overrides": {}}]

    expanded = _expand_jobs_with_splitting(jobs, split_multi=True)

    assert len(expanded) == 1
    assert expanded[0]["xyz"] == xyz
    assert "_original_xyz" not in expanded[0]


def test_split_disabled_is_a_passthrough(tmp_path):
    xyz = _xyz(tmp_path, "confs.xyz", 3)
    jobs = [{"xyz": xyz, "out_dir": str(tmp_path / "ens"), "overrides": {}}]

    assert _expand_jobs_with_splitting(jobs, split_multi=False) == jobs


def test_split_pools_structures_across_multiple_jobs(tmp_path):
    """The whole point: every structure from every input on one queue."""
    jobs = [
        {"xyz": _xyz(tmp_path, "a.xyz", 3), "out_dir": str(tmp_path / "a"), "overrides": {}},
        {"xyz": _xyz(tmp_path, "b.xyz", 2), "out_dir": str(tmp_path / "b"), "overrides": {}},
    ]

    expanded = _expand_jobs_with_splitting(jobs, split_multi=True)

    assert len(expanded) == 5
    assert len({j["out_dir"] for j in expanded}) == 5


def test_split_overrides_are_copied_not_shared(tmp_path):
    """Members must not alias one dict — the worker mutates its copy."""
    xyz = _xyz(tmp_path, "confs.xyz", 2)
    jobs = [{"xyz": xyz, "out_dir": str(tmp_path / "ens"), "overrides": {"charge": 0}}]

    expanded = _expand_jobs_with_splitting(jobs, split_multi=True)
    expanded[0]["overrides"]["charge"] = 99

    assert expanded[1]["overrides"]["charge"] == 0


def test_split_preserves_a_completed_unsplit_run(tmp_path):
    """A job finished under the old whole-file layout must not be redone.

    Members resume off <out_dir>/<label>/energies.csv, which never existed for such
    a run, so splitting it would silently recompute everything.
    """
    xyz = _xyz(tmp_path, "confs.xyz", 3)
    out_dir = tmp_path / "ens"
    out_dir.mkdir()
    (out_dir / "energies.csv").write_text("rank,tag\n1,conf_0000\n")

    jobs = [{"xyz": xyz, "out_dir": str(out_dir), "overrides": {}}]
    expanded = _expand_jobs_with_splitting(jobs, split_multi=True, resume=True)

    assert len(expanded) == 1
    assert expanded[0]["xyz"] == xyz


def test_split_proceeds_for_a_completed_unsplit_run_when_resume_is_off(tmp_path):
    xyz = _xyz(tmp_path, "confs.xyz", 3)
    out_dir = tmp_path / "ens"
    out_dir.mkdir()
    (out_dir / "energies.csv").write_text("rank,tag\n1,conf_0000\n")

    jobs = [{"xyz": xyz, "out_dir": str(out_dir), "overrides": {}}]
    expanded = _expand_jobs_with_splitting(jobs, split_multi=True, resume=False)

    assert len(expanded) == 3


def test_split_continues_a_partially_finished_split_run(tmp_path):
    """Once member dirs exist the run is already in the new layout — the aggregate
    CSV next to them must not be mistaken for an old whole-file run, or the
    unfinished members would never be picked up."""
    xyz = _xyz(tmp_path, "confs.xyz", 3)
    out_dir = tmp_path / "ens"
    (out_dir / "conf0000").mkdir(parents=True)
    (out_dir / "conf0000" / "energies.csv").write_text("rank,tag\n1,conf0000\n")
    (out_dir / "energies.csv").write_text("rank,tag\n1,conf0000\n")

    jobs = [{"xyz": xyz, "out_dir": str(out_dir), "overrides": {}}]
    expanded = _expand_jobs_with_splitting(jobs, split_multi=True, resume=True)

    assert len(expanded) == 3


def test_split_falls_back_to_whole_job_on_unreadable_input(tmp_path):
    missing = str(tmp_path / "nope.xyz")
    jobs = [{"xyz": missing, "out_dir": str(tmp_path / "ens"), "overrides": {}}]

    expanded = _expand_jobs_with_splitting(jobs, split_multi=True)

    assert len(expanded) == 1
    assert expanded[0]["xyz"] == missing


# ------------------------------------------------------------------ manifest wiring
def test_manifest_path_splits_multi_conformer_jobs(tmp_path, monkeypatch):
    """Regression: manifest jobs used to run whole-file on a single GPU.

    The real manifests point at multi-conformer files, so this was the difference
    between one busy GPU and all of them.
    """
    xyz = _xyz(tmp_path, "confs_for_opt.xyz", 4)
    out_dir = str(tmp_path / "uma_opt")
    manifest = tmp_path / "jobs.yaml"
    manifest.write_text(
        "jobs:\n"
        f"- xyz: {xyz}\n"
        f"  out_dir: {out_dir}\n"
        "  overrides:\n"
        "    charge: 0\n"
        "    optts: true\n"
    )

    seen = _capture_jobs(monkeypatch)
    run_batch_from_manifest(str(manifest), BatchCommon(out_root=str(tmp_path)))

    jobs = seen["jobs"]
    assert len(jobs) == 4, "manifest job was not split into per-structure jobs"
    assert [os.path.basename(j["out_dir"]) for j in jobs] == [
        f"conf{i:04d}" for i in range(4)
    ]
    # per-job overrides survive the split
    assert all(j["overrides"]["optts"] is True for j in jobs)


def test_manifest_rejects_null_optimizer_with_optts(tmp_path, monkeypatch):
    """`optimizer: null` + `optts: true` reads as "don't optimize, it's a TS".

    It is not: optts selects the saddle search, so the job re-optimizes and throws
    away the geometry it was handed. Nothing downstream shows this — the run
    succeeds and the numbers look fine — so it has to fail at manifest load,
    before any GPU time is spent.
    """
    xyz = _xyz(tmp_path, "confs_for_freq.xyz", 2)
    manifest = tmp_path / "jobs.yaml"
    manifest.write_text(
        "jobs:\n"
        f"- xyz: {xyz}\n"
        f"  out_dir: {tmp_path}/freq\n"
        "  overrides:\n"
        "    optimizer: null\n"
        "    optts: true\n"
        "    do_freq: true\n"
    )

    _capture_jobs(monkeypatch)
    with pytest.raises(RuntimeError, match="ambiguous"):
        run_batch_from_manifest(str(manifest), BatchCommon(out_root=str(tmp_path)))


def test_manifest_rejects_the_flattened_spelling_too(tmp_path, monkeypatch):
    """Same pairing without an `overrides:` block must not slip through."""
    xyz = _xyz(tmp_path, "confs.xyz", 1)
    manifest = tmp_path / "jobs.yaml"
    manifest.write_text(
        f"jobs:\n- xyz: {xyz}\n  optimizer: null\n  optts: true\n  do_freq: true\n"
    )

    _capture_jobs(monkeypatch)
    with pytest.raises(RuntimeError, match="ambiguous"):
        run_batch_from_manifest(str(manifest), BatchCommon(out_root=str(tmp_path)))


def test_manifest_allows_optts_without_naming_an_optimizer(tmp_path, monkeypatch):
    """Only an explicitly written null is ambiguous. `optts: true` on its own is
    the ordinary way to ask for a TS search and must keep working."""
    xyz = _xyz(tmp_path, "ts.xyz", 1)
    manifest = tmp_path / "jobs.yaml"
    manifest.write_text(f"jobs:\n- xyz: {xyz}\n  out_dir: {tmp_path}/ts\n  optts: true\n")

    seen = _capture_jobs(monkeypatch)
    run_batch_from_manifest(str(manifest), BatchCommon(out_root=str(tmp_path)))

    assert len(seen["jobs"]) == 1


def test_manifest_allows_freq_only_on_a_saddle(tmp_path, monkeypatch):
    """The spelling the error message recommends has to actually be accepted."""
    xyz = _xyz(tmp_path, "confs_for_freq.xyz", 2)
    manifest = tmp_path / "jobs.yaml"
    manifest.write_text(
        "jobs:\n"
        f"- xyz: {xyz}\n"
        f"  out_dir: {tmp_path}/freq\n"
        "  overrides:\n"
        "    optimizer: null\n"
        "    optts: false\n"
        "    freq_ts: true\n"
        "    do_freq: true\n"
    )

    seen = _capture_jobs(monkeypatch)
    run_batch_from_manifest(str(manifest), BatchCommon(out_root=str(tmp_path)))

    assert len(seen["jobs"]) == 2
    assert all(j["overrides"]["freq_ts"] is True for j in seen["jobs"])


def test_manifest_respects_split_disabled(tmp_path, monkeypatch):
    xyz = _xyz(tmp_path, "confs.xyz", 3)
    manifest = tmp_path / "jobs.yaml"
    manifest.write_text(f"jobs:\n- xyz: {xyz}\n  out_dir: {tmp_path}/ens\n")

    seen = _capture_jobs(monkeypatch)
    run_batch_from_manifest(
        str(manifest),
        BatchCommon(out_root=str(tmp_path), split_multi_structure=False),
    )

    assert len(seen["jobs"]) == 1


def test_glob_path_still_splits(tmp_path, monkeypatch):
    xyz = _xyz(tmp_path, "confs.xyz", 3)

    seen = _capture_jobs(monkeypatch)
    run_batch_from_glob([xyz], BatchCommon(out_root=str(tmp_path)))

    jobs = seen["jobs"]
    assert len(jobs) == 3
    # glob path keeps the <out_root>/<stem>.ensemble/<label> layout
    assert os.path.dirname(jobs[0]["out_dir"]) == os.path.join(
        str(tmp_path), "confs.ensemble"
    )


def test_batch_level_keys_never_reach_workflow_overrides(tmp_path, monkeypatch):
    """split_multi_structure in a per-job `overrides:` used to reach
    run_conformer_workflow as an unexpected kwarg."""
    xyz = _xyz(tmp_path, "one.xyz", 1)
    manifest = tmp_path / "jobs.yaml"
    manifest.write_text(
        "jobs:\n"
        f"- xyz: {xyz}\n"
        f"  out_dir: {tmp_path}/ens\n"
        "  overrides:\n"
        "    split_multi_structure: false\n"
        "    workers_per_gpu: 4\n"
        "    charge: -1\n"
    )

    seen = _capture_jobs(monkeypatch)
    run_batch_from_manifest(str(manifest), BatchCommon(out_root=str(tmp_path)))

    ov = seen["jobs"][0]["overrides"]
    assert ov["charge"] == -1
    for k in _BATCH_KEYS:
        assert k not in ov, f"batch key {k!r} leaked into workflow overrides"


# ------------------------------------------------------------------ scan sharding
GRID = {
    "mode": "grid",
    "coords": [
        {"distance": [1, 2], "from": 0.90, "to": 1.60, "steps": 8},
        {"distance": [1, 3], "from": 0.90, "to": 1.60, "steps": 8},
    ],
}


def _scan_job(tmp_path, scan=None, out="ens", **overrides):
    return {
        "xyz": _xyz(tmp_path, "one.xyz", 1),
        "out_dir": str(tmp_path / out),
        "overrides": {"scan": scan if scan is not None else GRID, **overrides},
    }


def _plan(jobs, monkeypatch=None, gpus=(), **merged):
    from umadriver.batch import _plan_scan_shards

    if monkeypatch is not None:
        monkeypatch.setattr("umadriver.batch._discover_gpus", lambda d: list(gpus))
    merged.setdefault("device", "cpu")
    merged.setdefault("scan_shards", "auto")
    return _plan_scan_shards(jobs, merged)


def test_a_grid_becomes_a_spine_plus_one_shard_per_row(tmp_path, monkeypatch):
    jobs, plans = _plan(
        [_scan_job(tmp_path)], monkeypatch, gpus=["0", "1"], device="cuda"
    )

    assert len(plans) == 1
    plan = plans[0]
    # The job itself is replaced by its spine; the shards come later, once the
    # spine has produced the geometries that seed them.
    assert len(jobs) == 1
    assert os.path.basename(jobs[0]["out_dir"]) == "spine"

    # 8x8: one shard per row, and the spine walks each row's first point.
    assert [len(g) for g in plan["groups"]] == [8] * 8
    assert sorted(k for g in plan["groups"] for k in g) == list(range(64))
    assert jobs[0]["overrides"]["scan_shard"]["indices"] == [g[0] for g in plan["groups"]]


def test_the_partition_does_not_depend_on_the_machine(tmp_path, monkeypatch):
    """Otherwise a rerun under a different GPU count would resume half of one
    partition into half of another and merge a surface that never existed."""
    job = _scan_job(tmp_path)
    _, first = _plan([job], monkeypatch, gpus=["0", "1"], device="cuda")
    _, second = _plan([job], monkeypatch, gpus=list("01234567"), device="cuda")

    assert first[0]["groups"] == second[0]["groups"]
    assert len(first[0]["groups"]) == 8


def test_a_changed_scan_refuses_to_reuse_the_old_partition(tmp_path, caplog):
    import logging

    job = _scan_job(tmp_path)
    _plan([job], scan_shards=4)  # writes .shards.json

    wider = dict(job)
    wider["overrides"] = {
        "scan": {
            "mode": "grid",
            "coords": [
                {"distance": [1, 2], "from": 0.90, "to": 2.00, "steps": 12},
                {"distance": [1, 3], "from": 0.90, "to": 1.60, "steps": 8},
            ],
        }
    }
    with caplog.at_level(logging.WARNING, logger="uma.batch"):
        jobs, plans = _plan([wider], scan_shards=4)

    assert plans == []
    assert jobs == [wider]  # runs whole rather than mixing two surfaces
    assert any("different scan" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "kw",
    [
        {"scan_shards": "off"},
        {"scan_shards": 1},
        {"scan_shards": 0},
    ],
)
def test_sharding_can_be_turned_off(tmp_path, kw):
    jobs, plans = _plan([_scan_job(tmp_path)], **kw)
    assert plans == []
    assert len(jobs) == 1 and "spine" not in jobs[0]["out_dir"]


def test_auto_does_not_shard_without_gpus(tmp_path):
    """Sharding the serial CPU path is strictly worse: same points, extra cold
    starts, no parallelism."""
    jobs, plans = _plan([_scan_job(tmp_path)], scan_shards="auto", device="cpu")
    assert plans == []


def test_a_short_scan_is_not_worth_sharding(tmp_path):
    short = {"mode": "grid", "coords": [
        {"distance": [1, 2], "from": 0.9, "to": 1.1, "steps": 3},
        {"distance": [1, 3], "from": 0.9, "to": 1.1, "steps": 3},
    ]}
    jobs, plans = _plan([_scan_job(tmp_path, scan=short)], scan_shards=3)
    assert plans == []  # 9 points, below 2x SHARD_MIN_POINTS


def test_non_scan_and_freq_jobs_are_left_alone(tmp_path):
    plain = {"xyz": _xyz(tmp_path, "a.xyz", 1), "out_dir": str(tmp_path / "a"),
             "overrides": {"charge": 0}}
    freq = _scan_job(tmp_path, out="b", do_freq=True)

    jobs, plans = _plan([plain, freq], scan_shards=4)

    assert plans == []
    assert jobs == [plain, freq]


def test_a_finished_unsharded_scan_is_not_redone(tmp_path):
    """Mirrors _already_done_unsplit: without this the merge would overwrite the
    completed scan CSV with its own."""
    job = _scan_job(tmp_path)
    os.makedirs(job["out_dir"], exist_ok=True)
    open(os.path.join(job["out_dir"], "energies.csv"), "w").close()

    _, plans = _plan([job], scan_shards=4, resume=True)
    assert plans == []

    _, redo = _plan([job], scan_shards=4, resume=False)
    assert len(redo) == 1


def test_shards_never_share_a_cleanup_file(tmp_path):
    """The worker deletes _cleanup_xyz on success and on error alike, so an
    inherited one would have shard 0 delete the input its siblings still need."""
    from umadriver.batch import _build_shard_jobs
    from umadriver.scan import parse_scan_spec, run_scan
    from ase.build import molecule
    from ase.calculators.emt import EMT

    job = _scan_job(tmp_path)
    job["_cleanup_xyz"] = job["xyz"]
    job["_original_xyz"] = str(tmp_path / "parent.xyz")
    jobs, plans = _plan([job], scan_shards=4)

    # stand in for the spine having run
    atoms = molecule("H2O")
    atoms.calc = EMT()
    run_scan(
        atoms,
        parse_scan_spec(GRID),
        relax=lambda a: (True, 1, float(a.get_potential_energy())),
        out_dir=os.path.join(plans[0]["spine_dir"], "scan"),
        tag="conf_0000",
        shard=jobs[0]["overrides"]["scan_shard"],
    )

    shards = _build_shard_jobs(plans)

    assert len(shards) == 4
    assert all("_cleanup_xyz" not in s for s in shards)
    assert len({s["out_dir"] for s in shards}) == 4
    # each shard is seeded from its own geometry, not the original input
    assert all(s["xyz"].endswith(f"seg{i:02d}.xyz") for i, s in enumerate(shards))
    # and none of them looks like an ensemble member of <parent>
    assert all("_original_xyz" not in s for s in shards)


# ------------------------------------------------------------------ worker slots
@pytest.mark.parametrize(
    "gpu_ids,per_gpu,expected",
    [
        ([0], 1, [0]),
        ([0, 1], 1, [0, 1]),
        ([0, 1], 2, [0, 1, 0, 1]),
        ([0, 1, 2], 3, [0, 1, 2, 0, 1, 2, 0, 1, 2]),
        ([0, 1], 0, [0, 1]),  # degenerate input clamps to 1
    ],
)
def test_worker_slots(gpu_ids, per_gpu, expected):
    assert _worker_slots(gpu_ids, per_gpu) == expected


def test_worker_slots_interleave_so_few_jobs_span_gpus():
    """Grouped slots ([0,0,1,1]) would put the first two jobs on the same GPU."""
    slots = _worker_slots([0, 1, 2, 3], 2)
    assert slots[:4] == [0, 1, 2, 3]


MIG_UUIDS = (
    "MIG-69ef5d14-d33c-5ce4-87c5-39b1afbab08c,"
    "MIG-726bfd38-6b8a-5ee8-b3fc-a58b1732b274,"
    "MIG-8ea4d39a-51e8-5533-807c-c5dd5fdd4736,"
    "MIG-a419df06-5031-532e-8439-2c54739c5dd0"
)


def test_visible_devices_keeps_mig_uuids(monkeypatch):
    """MIG slices are addressable only by UUID.

    Coercing these to int throws, falls back to torch.cuda.device_count(), and
    hands every worker an integer index — and on a MIG card *every* integer index
    resolves to the first slice. The result is N workers silently sharing one
    slice while the log claims N GPUs.
    """
    from umadriver.batch import _parse_visible_devices_env

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", MIG_UUIDS)
    toks = _parse_visible_devices_env()

    assert toks == MIG_UUIDS.split(",")
    assert len(set(toks)) == 4, "workers would collide on one slice"


def test_discover_gpus_returns_distinct_mig_tokens(monkeypatch):
    from umadriver.batch import _discover_gpus

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", MIG_UUIDS)
    gpus = _discover_gpus("cuda")

    assert len(gpus) == 4
    assert all(g.startswith("MIG-") for g in gpus)


def test_bind_gpu_env_writes_the_mig_token_back(monkeypatch):
    """The bound value must be the UUID, not an index."""
    uuid = MIG_UUIDS.split(",")[2]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", MIG_UUIDS)
    monkeypatch.setenv("UMA_SCRATCH_ROOT", "/tmp/whatever")

    _bind_gpu_env(uuid, "/tmp/out", n_workers=1)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == uuid


def test_worker_slots_spread_across_distinct_mig_slices(monkeypatch):
    from umadriver.batch import _discover_gpus

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", MIG_UUIDS)
    slots = _worker_slots(_discover_gpus("cuda"), 2)

    assert len(slots) == 8
    assert len(set(slots)) == 4, "8 workers must cover all 4 slices, not pile up"
    assert len(set(slots[:4])) == 4, "the first 4 workers must land on 4 distinct slices"


def test_scratch_shard_name_stays_short_for_mig_uuids(monkeypatch, tmp_path):
    """UUID tokens would otherwise produce absurd directory names."""
    monkeypatch.delenv("UMA_SCRATCH_ROOT", raising=False)
    uuid = MIG_UUIDS.split(",")[0]

    _bind_gpu_env(uuid, str(tmp_path), n_workers=1)

    shard = os.environ["UMA_SCRATCH_ROOT"]
    assert "MIG-" not in shard
    assert len(os.path.basename(shard)) < 32


def test_bind_gpu_env_divides_thread_budget(monkeypatch):
    """Workers inherit the parent's full-CPU thread setting; each must take a share."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "24")
    monkeypatch.setenv("OMP_NUM_THREADS", "24")
    monkeypatch.setenv("MKL_NUM_THREADS", "24")
    monkeypatch.setenv("UMA_SCRATCH_ROOT", "/tmp/whatever")

    _bind_gpu_env(2, "/tmp/out", n_workers=4)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    assert os.environ["OMP_NUM_THREADS"] == "6"
    assert os.environ["MKL_NUM_THREADS"] == "6"


def test_bind_gpu_env_leaves_threads_alone_for_single_worker(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "24")
    monkeypatch.setenv("OMP_NUM_THREADS", "24")
    monkeypatch.setenv("UMA_SCRATCH_ROOT", "/tmp/whatever")

    _bind_gpu_env(0, "/tmp/out", n_workers=1)

    assert os.environ["OMP_NUM_THREADS"] == "24"


def test_bind_gpu_env_never_drops_below_one_thread(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("UMA_SCRATCH_ROOT", "/tmp/whatever")

    _bind_gpu_env(0, "/tmp/out", n_workers=8)

    assert os.environ["OMP_NUM_THREADS"] == "1"


# ------------------------------------------------------------------ CLI
@pytest.mark.parametrize(
    "argv,expected",
    [
        (["batch", "--manifest", "x.yaml"], ["--manifest", "x.yaml"]),
        (["batch", "a.xyz", "b.xyz"], ["a.xyz", "b.xyz"]),
        (["mol.xyz", "--optts"], ["mol.xyz", "--optts"]),
        (["--manifest", "x.yaml"], ["--manifest", "x.yaml"]),
        ([], []),
    ],
)
def test_strip_batch_token(argv, expected):
    from umadriver.driver import _strip_batch_token

    assert _strip_batch_token(argv) == expected


def test_batch_token_does_not_eat_a_real_input_named_batch_xyz():
    from umadriver.driver import _strip_batch_token

    assert _strip_batch_token(["batch.xyz"]) == ["batch.xyz"]
