from __future__ import annotations

import os
import sys
import csv
import glob
import hashlib
import logging
import json
import multiprocessing as mp
import queue
from typing import Any, Dict, Iterable, List, Optional, Tuple
from dataclasses import dataclass

LOG = logging.getLogger("uma.batch")

try:
    import yaml  # optional

    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False

from ase.io import read as ase_read, write as ase_write

from .ensemble import (
    run_conformer_workflow,
    _ensure_dir,
    rank_by_energy,
    ENERGIES_FIELDS,
)
from .scan import merge_scan_shards, parse_scan_spec


# =========================
# Config
# =========================
@dataclass
class BatchCommon:
    model: str = "uma-m-1p1"
    device: str = (
        "cuda"  # "cuda" | "cpu" | "auto" (auto treated like cuda if GPUs exist)
    )
    cache_dir: Optional[str] = None
    use_local_scratch: bool = False
    out_root: str = "runs"
    resume: bool = True  # skip job if energies.csv exists
    # Explode multi-structure XYZs into one job per structure so conformers from
    # every input pool into the shared queue and spread across GPUs.
    split_multi_structure: bool = True
    # Worker processes per GPU. >1 hides single-structure inference latency on
    # large cards; costs VRAM linearly.
    workers_per_gpu: int = 1
    # Split a big relaxed scan into pieces that run on separate GPUs. "auto" shards
    # a grid one row at a time and anything else by worker slot; "off"/1 keeps the
    # scan whole. See _plan_scan_shards.
    scan_shards: Any = "auto"


# =========================
# Helpers
# =========================
def _load_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        text = f.read()
    if _HAVE_YAML and (path.endswith(".yaml") or path.endswith(".yml")):
        return yaml.safe_load(text)
    return json.loads(text)


def _job_out_dir(out_root: str, xyz_path: str, explicit: Optional[str]) -> str:
    """
    Default: <out_root>/<stem>.ensemble unless explicit out_dir is provided.
    """
    if explicit:
        return explicit
    base = os.path.splitext(os.path.basename(xyz_path))[0]
    return os.path.join(out_root, base + ".ensemble")


def _expand_xyz_inputs(xyz_list: Iterable[str]) -> List[str]:
    out: List[str] = []
    for s in xyz_list:
        # If user quoted globs, expand here; if shell expanded already, we just get filenames.
        if any(ch in s for ch in "*?[]"):
            out.extend(sorted(glob.glob(s)))
        else:
            out.append(s)
    return out


def _split_xyz_into_structures(xyz_path: str) -> List[tuple]:
    """
    Read an XYZ file and split into individual structures.
    Returns list of (structure_content, structure_label) tuples.
    Each structure_content is the full XYZ text for one geometry.
    """
    structures = []

    with open(xyz_path, "r") as f:
        lines = f.readlines()

    i = 0
    struct_idx = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # First line of XYZ: atom count
        try:
            natoms = int(line.split()[0])
        except (ValueError, IndexError):
            i += 1
            continue

        # Capture: natoms line + comment line + natoms coordinate lines
        if i + natoms + 1 < len(lines):
            structure_block = lines[i : i + natoms + 2]
            structures.append(("".join(structure_block), f"conf{struct_idx:04d}"))
            struct_idx += 1
            i += natoms + 2
        else:
            break

    # Fallback: if parsing failed, treat whole file as single structure
    if len(structures) == 0:
        with open(xyz_path, "r") as f:
            structures.append((f.read(), "conf0000"))

    return structures


def _already_done_unsplit(out_dir: str) -> bool:
    """True if out_dir holds a finished run from before splitting was applied here.

    Splitting moves each structure into ``<out_dir>/<label>/``, so a job that
    previously completed whole-file has no member directories and would be redone
    from scratch — the members' resume checks look at paths that never existed.
    An ensemble CSV with no member dirs next to it is exactly that case.
    """
    if not os.path.isfile(os.path.join(out_dir, "energies.csv")):
        return False
    return not os.path.isdir(os.path.join(out_dir, "conf0000"))


def _expand_jobs_with_splitting(
    jobs: List[Dict[str, Any]], split_multi: bool, resume: bool = False
) -> List[Dict[str, Any]]:
    """Explode each multi-structure job into one member job per structure.

    The scheduler's unit of work is a whole job, and ``run_conformer_workflow``
    walks a job's conformers serially. So an unsplit N-conformer file occupies a
    single GPU for the whole run while the others idle. Splitting puts every
    structure on the shared queue instead, where free workers pick them up.

    Members land in ``<job out_dir>/<label>/``, which is what lets
    ``_aggregate_split_ensembles`` recompile them (it groups by the parent of
    ``out_dir``). Temp single-structure XYZs go under the job's own out_dir rather
    than a batch-global ``.tmp``, so manifests with absolute ``out_dir:`` stay
    self-contained instead of scattering files relative to cwd.

    Used by both the manifest and glob paths — previously only the glob path split,
    which is why manifest runs never fanned out.
    """
    if not split_multi:
        return jobs

    expanded: List[Dict[str, Any]] = []
    for job in jobs:
        xyz = job["xyz"]
        out_dir = job["out_dir"]

        try:
            structures = _split_xyz_into_structures(xyz)
        except Exception as e:
            LOG.exception("Could not split %s (%s); running as a single job.", xyz, e)
            expanded.append(job)
            continue

        if len(structures) <= 1:
            LOG.info("Job %s: single structure", xyz)
            expanded.append(job)
            continue

        if resume and _already_done_unsplit(out_dir):
            LOG.info(
                "Job %s: %s already holds a completed whole-file run; keeping it "
                "unsplit so --resume can skip it (use --no-resume to redo it split).",
                xyz,
                out_dir,
            )
            expanded.append(job)
            continue

        LOG.info("Splitting %s into %d structures", xyz, len(structures))
        split_dir = os.path.join(out_dir, ".split")
        _ensure_dir(split_dir)
        base = os.path.splitext(os.path.basename(xyz))[0]

        for content, label in structures:
            temp_xyz = os.path.join(split_dir, f"{base}_{label}.xyz")
            with open(temp_xyz, "w") as f:
                f.write(content)

            member = dict(job)
            member["xyz"] = temp_xyz
            member["out_dir"] = os.path.join(out_dir, label)
            member["overrides"] = (job.get("overrides") or {}).copy()
            member["_cleanup_xyz"] = temp_xyz
            member["_original_xyz"] = xyz
            expanded.append(member)

    return expanded


# =========================
# Spreading one scan over several GPUs
# =========================
# Below this, a shard spends more time cold-starting than it saves. A scan point
# normally costs ~3 optimizer steps because it begins from the previous relaxed
# point; the first point of a shard costs ~5 even seeded.
SHARD_MIN_POINTS = 8


def _already_done_unsharded(out_dir: str) -> bool:
    """True if out_dir holds a scan that finished before sharding was applied here.

    Exactly the ``_already_done_unsplit`` problem: sharding moves the work into
    ``<out_dir>/shardNN/``, so a scan that previously completed whole has no shard
    directories, every shard would recompute from scratch, and the merge would
    overwrite the finished ``scan/*_scan.csv`` with its own.
    """
    if not os.path.isfile(os.path.join(out_dir, "energies.csv")):
        return False
    return not os.path.isdir(os.path.join(out_dir, "shard00"))


def _scan_fingerprint(spec, traversal: str) -> str:
    """Identifies the surface, so a resumed run can tell it is the same one."""
    body = repr(
        (
            spec.mode,
            traversal,
            [(c.kind, c.indices, c.start, c.end, c.nsteps) for c in spec.coords],
        )
    )
    return hashlib.sha1(body.encode()).hexdigest()[:16]


def _contiguous_groups(items: List[int], n: int) -> List[List[int]]:
    """Split into n near-equal contiguous groups."""
    n = max(1, min(n, len(items)))
    q, r = divmod(len(items), n)
    out, i = [], 0
    for g in range(n):
        size = q + (1 if g < r else 0)
        out.append(items[i : i + size])
        i += size
    return out


def _shard_groups(spec, traversal: str, n_shards: int) -> List[List[int]]:
    """Point indices for each shard, as contiguous runs of the schedule.

    A grid is cut on row boundaries: a row is already a self-contained left-to-right
    walk under ``rowmajor``, so a shard that owns whole rows never starts a row in
    the middle.
    """
    full = spec.schedule(traversal)
    if spec.mode == "grid":
        rows: List[List[int]] = []
        for k, values in enumerate(full):
            if not rows or values[0] != full[rows[-1][0]][0]:
                rows.append([])
            rows[-1].append(k)
        return [
            [k for r in rg for k in rows[r]]
            for rg in _contiguous_groups(list(range(len(rows))), n_shards)
        ]
    return _contiguous_groups(list(range(len(full))), n_shards)


def _resolve_shard_count(scan_shards: Any, spec, n_slots: int) -> int:
    """How many pieces to cut this scan into. 1 means do not shard."""
    s = str(scan_shards).strip().lower() if scan_shards is not None else "off"

    if s in ("off", "no", "none", "false"):
        want = 1
    elif s == "auto":
        # A grid's row count is a property of the scan, not of the machine, so the
        # partition comes out the same on 2 GPUs as on 8 — which is what makes it
        # safe to reuse on a resumed run. The shared queue balances however many
        # rows there are across however many slots exist.
        if n_slots < 1:
            want = 1
        elif spec.mode == "grid":
            want = spec.coords[0].nsteps
        else:
            want = n_slots
    elif s.isdigit():
        want = int(s)
    else:
        raise ValueError(
            f"scan_shards must be 'auto', 'off' or a number, got {scan_shards!r}"
        )

    if want <= 1:
        return 1
    return max(1, min(want, spec.npoints // SHARD_MIN_POINTS))


def _plan_one_scan(
    job: Dict[str, Any], scan_shards: Any, n_slots: int, resume: bool
) -> Optional[Dict[str, Any]]:
    """Work out how to shard one job's scan, or None to leave the job alone."""
    overrides = job.get("overrides") or {}
    out_dir = job["out_dir"]
    if not overrides.get("scan"):
        return None

    if overrides.get("do_freq"):
        LOG.info(
            "Job %s: not sharding — frequencies would be computed on each shard's "
            "own last constrained point, which is meaningless N times over.",
            job["xyz"],
        )
        return None

    try:
        spec = parse_scan_spec(overrides["scan"])
    except Exception as e:
        LOG.info("Job %s: not sharding (%s); the worker will report it.", job["xyz"], e)
        return None

    try:
        if len(_split_xyz_into_structures(job["xyz"])) > 1:
            LOG.info(
                "Job %s: not sharding a multi-structure input. Leave "
                "split_multi_structure on and each structure shards on its own.",
                job["xyz"],
            )
            return None
    except Exception:
        return None

    if resume and _already_done_unsharded(out_dir):
        LOG.info(
            "Job %s: %s already holds a completed unsharded scan; leaving it whole "
            "so --resume can skip it (use --no-resume to redo it sharded).",
            job["xyz"],
            out_dir,
        )
        return None

    traversal = "rowmajor" if spec.mode == "grid" else "boustrophedon"
    n_shards = _resolve_shard_count(scan_shards, spec, n_slots)
    if n_shards <= 1:
        return None

    scan_dir = os.path.join(out_dir, "scan")
    state_path = os.path.join(scan_dir, ".shards.json")
    fingerprint = _scan_fingerprint(spec, traversal)
    groups: Optional[List[List[int]]] = None

    if resume and os.path.isfile(state_path):
        try:
            with open(state_path) as f:
                prev = json.load(f)
        except Exception:
            prev = None
        if prev and prev.get("fingerprint") != fingerprint:
            # The stored shard dirs hold points from a different surface, and their
            # energies.csv files would let resume skip them — merging stale rows into
            # the new scan with no error anywhere. Refuse rather than guess.
            LOG.warning(
                "Job %s: %s was sharded for a different scan. Not sharding this "
                "one — rerun with --no-resume, or point it at a fresh out_dir.",
                job["xyz"],
                out_dir,
            )
            return None
        if prev:
            groups = [[int(k) for k in g] for g in prev["groups"]]
            LOG.info("Job %s: reusing the stored %d-shard partition", job["xyz"], len(groups))

    if groups is None:
        groups = _shard_groups(spec, traversal, n_shards)
        _ensure_dir(scan_dir)
        with open(state_path, "w") as f:
            json.dump(
                {
                    "fingerprint": fingerprint,
                    "traversal": traversal,
                    "npoints": spec.npoints,
                    "groups": groups,
                },
                f,
            )

    LOG.info(
        "Job %s: %d scan points over %d shards (%s), seeded by a %d-point spine",
        job["xyz"],
        spec.npoints,
        len(groups),
        traversal,
        len(groups),
    )

    # The spine walks the first point of every shard, in order and chained, so each
    # shard can start from a geometry that is already relaxed at its own first point
    # instead of jumping there cold from the input and risking a different basin.
    spine = dict(job)
    spine["out_dir"] = os.path.join(out_dir, "spine")
    spine["overrides"] = {
        **overrides,
        "scan_shard": {"traversal": traversal, "indices": [g[0] for g in groups]},
    }
    # It must not look like an ensemble member: _aggregate_split_ensembles groups on
    # the parent of out_dir, which for the spine is the job's own out_dir.
    spine.pop("_original_xyz", None)

    return {
        "job": job,
        "spec": spec,
        "traversal": traversal,
        "groups": groups,
        "parent": out_dir,
        "spine_dir": spine["out_dir"],
        "spine_job": spine,
    }


def _plan_scan_shards(
    jobs: List[Dict[str, Any]], merged: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Replace each shardable scan job with its spine job; return (jobs, plans)."""
    scan_shards = merged.get("scan_shards", "auto")
    resume = bool(merged.get("resume", True))
    slots = _worker_slots(
        _discover_gpus(merged.get("device", "cuda")),
        int(merged.get("workers_per_gpu", 1) or 1),
    )

    jobs_out: List[Dict[str, Any]] = []
    plans: List[Dict[str, Any]] = []
    for job in jobs:
        plan = _plan_one_scan(
            job, job.get("scan_shards", scan_shards), len(slots), resume
        )
        if plan is None:
            jobs_out.append(job)
        else:
            jobs_out.append(plan["spine_job"])
            plans.append(plan)
    return jobs_out, plans


def _build_shard_jobs(plans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """After the spines have run, emit one job per shard seeded from its geometry."""
    shard_jobs: List[Dict[str, Any]] = []
    for plan in plans:
        spine_scan = os.path.join(plan["spine_dir"], "scan")
        trajectories = sorted(glob.glob(os.path.join(spine_scan, "*_scan.xyz")))
        if not trajectories:
            LOG.error(
                "Spine produced nothing in %s; leaving %s unsharded this run.",
                spine_scan,
                plan["parent"],
            )
            continue

        traj = trajectories[0]
        plan["tag"] = os.path.basename(traj)[: -len("_scan.xyz")]
        frames = ase_read(traj, index=":")
        if len(frames) != len(plan["groups"]):
            LOG.error(
                "Spine wrote %d geometries but there are %d shards (%s); leaving "
                "%s unsharded this run.",
                len(frames),
                len(plan["groups"]),
                traj,
                plan["parent"],
            )
            continue

        seed_dir = os.path.join(plan["parent"], "scan", ".seed")
        _ensure_dir(seed_dir)
        plan["shard_dirs"] = []
        for i, (group, frame) in enumerate(zip(plan["groups"], frames)):
            seed = os.path.join(seed_dir, f"seg{i:02d}.xyz")
            ase_write(seed, frame, format="xyz")

            child = dict(plan["job"])
            child["xyz"] = seed
            child["out_dir"] = os.path.join(plan["parent"], f"shard{i:02d}")
            child["overrides"] = {
                **(plan["job"].get("overrides") or {}),
                "scan_shard": {"traversal": plan["traversal"], "indices": group},
            }
            # The parent's temp XYZ belongs to the spine, which has already used it.
            # Inheriting it here would have shard 0 delete the file out from under
            # its siblings. Seeds are left in place so a rerun can reuse them.
            child.pop("_cleanup_xyz", None)
            child.pop("_original_xyz", None)
            shard_jobs.append(child)
            plan["shard_dirs"].append(os.path.join(child["out_dir"], "scan"))

    return shard_jobs


def _aggregate_scan_shards(plans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge each scan's shards back into the files an unsharded run would write.

    Returns one synthesized job per merged scan, standing in for the job that was
    replaced by the spine and shards, so ``_aggregate_split_ensembles`` sees an
    ordinary member at ``<parent>`` and needs no knowledge of shards.
    """
    parents: List[Dict[str, Any]] = []
    for plan in plans:
        if "shard_dirs" not in plan:
            continue
        records = merge_scan_shards(
            plan["shard_dirs"],
            plan["spec"],
            plan["traversal"],
            os.path.join(plan["parent"], "scan"),
            plan["tag"],
        )
        if records is None:
            # Deliberately no <parent>/energies.csv: its absence is what makes the
            # next --resume rerun the shards that failed.
            continue

        last = records[-1]
        row = {
            "rank": 1,
            "index": 0,
            "tag": plan["tag"],
            "route": "SCAN",
            "converged": all(r["converged"] for r in records),
            "steps": sum(r["steps"] for r in records),
            "energy_Eh": last["energy_Eh"],
            "energy_kcal": last["energy_kcal"],
            "rel_kcal": 0.0,
        }
        out_csv = os.path.join(plan["parent"], "energies.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ENERGIES_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerow({k: row.get(k) for k in ENERGIES_FIELDS})

        # The last shard holds the final point, which is what the row above reports.
        tail = os.path.join(os.path.dirname(plan["shard_dirs"][-1]), "optimized_ranked.xyz")
        if os.path.isfile(tail):
            with open(tail) as src, open(
                os.path.join(plan["parent"], "optimized_ranked.xyz"), "w"
            ) as dst:
                dst.write(src.read())

        parent = dict(plan["job"])
        parent.pop("_cleanup_xyz", None)
        parents.append(parent)

    return parents


def _parse_visible_devices_env() -> Optional[List[str]]:
    """Device tokens from CUDA_VISIBLE_DEVICES, kept verbatim.

    Deliberately NOT parsed as integers. On a MIG-partitioned card the tokens are
    UUIDs (``MIG-69ef5d14-...``), and those are the only way to address a specific
    slice — an integer index silently resolves to the *first* slice no matter what
    number you use, so int-coercing here would collapse every worker onto one
    slice while reporting that several were in use.
    """
    s = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not s:
        return None
    toks = [t.strip() for t in s.split(",") if t.strip()]
    return toks or None


def _discover_gpus(device: str) -> List[str]:
    """
    Returns the device tokens visible to this process (integer indices as strings,
    or MIG UUIDs). If device is cpu -> [].
    """
    if device.lower() == "cpu":
        return []

    env_ids = _parse_visible_devices_env()
    if env_ids is not None:
        return env_ids

    try:
        import torch

        n = torch.cuda.device_count()
        return [str(i) for i in range(n)]
    except Exception:
        return []


def _worker_slots(gpu_ids: List[str], workers_per_gpu: int) -> List[str]:
    """One entry per worker process, holding the device token it binds to.

    Interleaved rather than grouped (``[0,1,0,1]`` not ``[0,0,1,1]``) so that a run
    with fewer jobs than slots still spreads across distinct GPUs.
    """
    n = max(1, int(workers_per_gpu or 1))
    return [g for _ in range(n) for g in gpu_ids]


_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _bind_gpu_env(gpu_id: str, out_root: str, n_workers: int = 1):
    """
    Bind this child process to exactly one GPU and give it its share of the CPUs.

    ``gpu_id`` is a device token — an integer index or a MIG UUID — and is written
    back verbatim, which is what actually pins a worker to a specific MIG slice.

    ``n_workers`` is the total number of worker processes in this run, not the
    number per GPU.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # The parent sets OMP_NUM_THREADS & friends to the *full* CPU allocation
    # (driver._early_parse_threads), and every spawned worker inherits that value —
    # so N workers each try to use all the cores. Overwrite with this worker's share.
    # Must be assignment, not setdefault: the vars are already set by inheritance.
    if n_workers > 1:
        total = int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or (os.cpu_count() or 1)
        share = str(max(1, total // n_workers))
        for var in _THREAD_VARS:
            os.environ[var] = share

    # If user did not specify UMA_SCRATCH_ROOT, shard by GPU to reduce contention.
    if "UMA_SCRATCH_ROOT" not in os.environ:
        # MIG tokens are long UUIDs; keep the directory name short but distinct.
        short = str(gpu_id).replace("MIG-", "")[:8]
        os.environ["UMA_SCRATCH_ROOT"] = os.path.join(out_root, f"_gpu{short}_scratch")


# =========================
# Worker
# =========================
def _worker_loop(
    gpu_id: str,
    task_q: mp.Queue,
    result_q: mp.Queue,
    common: Dict[str, Any],
):
    """
    Each worker:
      - pins to a single GPU
      - processes a stream of jobs (each job is one ensemble XYZ)
      - run_conformer_workflow constructs its own calculator internally (unless injected)
    """
    from .utils import setup_logging

    setup_logging(verbose=True, debug=False)
    _bind_gpu_env(gpu_id, common["out_root"], common.get("n_workers", 1))
    LOG.info("[GPU %s] worker start", gpu_id)

    while True:
        try:
            job = task_q.get(timeout=2.0)
        except queue.Empty:
            continue

        if job is None:
            LOG.info("[GPU %s] worker exit", gpu_id)
            break

        xyz = job["xyz"]
        out_dir = job["out_dir"]
        overrides = (job.get("overrides", {}) or {}).copy()
        resume = bool(job.get("resume", True))
        cleanup_xyz = job.get("_cleanup_xyz")

        # Avoid keyword collisions: allow this to be set either in overrides or in common
        resume_from_per_conf = overrides.pop(
            "resume_from_per_conformer_csv",
            common.get("resume_from_per_conformer_csv", False),
        )

        # Batch-level keys are stripped at job-construction time (see _BATCH_KEYS),
        # so `overrides` here is workflow kwargs only.

        try:
            energies_csv = os.path.join(out_dir, "energies.csv")
            if resume and os.path.isfile(energies_csv):
                LOG.info("[GPU %s] SKIP (resume): %s", gpu_id, out_dir)
                result_q.put({"xyz": xyz, "out_dir": out_dir, "status": "skipped"})
                if cleanup_xyz and os.path.exists(cleanup_xyz):
                    try:
                        os.remove(cleanup_xyz)
                    except Exception:
                        pass
                continue

            _ensure_dir(out_dir)
            LOG.info("[GPU %s] RUN: %s → %s", gpu_id, xyz, out_dir)

            run_conformer_workflow(
                xyz,
                out_dir,
                model=common["model"],
                device="cuda",
                cache_dir=common.get("cache_dir"),
                use_local_scratch=common.get("use_local_scratch", False),
                resume_from_per_conformer_csv=resume_from_per_conf,
                **overrides,
            )
            result_q.put({"xyz": xyz, "out_dir": out_dir, "status": "ok"})

            # Cleanup temp file after successful run
            if cleanup_xyz and os.path.exists(cleanup_xyz):
                try:
                    os.remove(cleanup_xyz)
                except Exception:
                    pass

        except Exception as e:
            LOG.exception("[GPU %s] Job failed: %s", gpu_id, e)
            result_q.put({"xyz": xyz, "out_dir": out_dir, "status": f"error: {e}"})
            # Cleanup temp file even on error
            if cleanup_xyz and os.path.exists(cleanup_xyz):
                try:
                    os.remove(cleanup_xyz)
                except Exception:
                    pass


# =========================
# Public API
# =========================
# Batch-level configuration keys that must never be forwarded to
# run_conformer_workflow() as per-job workflow overrides.
_BATCH_KEYS = set(BatchCommon.__dataclass_fields__.keys())


def _reject_ambiguous_ts_route(xyz: str, job_keys: Dict[str, Any]) -> None:
    """Refuse a job that writes ``optimizer: null`` next to ``optts: true``.

    That pairing reads as "no optimization, and it's a transition state", which is
    what people mean by it. It is not what it does: optts selects the saddle
    search and the geometry gets re-optimized, quietly discarding the structure
    the job was pointed at. Nothing downstream reveals this — the run succeeds and
    the numbers look reasonable.

    Only an *explicitly written* null counts, so `--optts` on the command line and
    a plain `optts: true` in a manifest are unaffected.
    """
    if not job_keys.get("optts"):
        return
    if "optimizer" not in job_keys or job_keys["optimizer"] is not None:
        return
    raise RuntimeError(
        f"{xyz}: `optimizer: null` with `optts: true` is ambiguous.\n"
        "  optts selects a saddle search, so this job WOULD re-optimize the "
        "geometry despite the null optimizer.\n"
        "  - to optimize to a TS:            drop `optimizer: null`\n"
        "  - for frequencies on this exact\n"
        "    geometry, left untouched:       `optts: false` + `freq_ts: true`"
    )


def run_batch_from_manifest(manifest_path: str, common: BatchCommon, **cli_overrides):
    cfg = _load_manifest(manifest_path)
    try:
        common_cfg = cfg.get("common", {}) or {}
    except AttributeError:
        print(
            "Manifest parse error. Expected top-level mapping with 'jobs:' and optional 'common:'."
        )
        sys.exit(1)

    jobs_cfg = cfg.get("jobs", [])
    if not isinstance(jobs_cfg, list):
        raise RuntimeError("Manifest 'jobs' must be a list.")

    # Merge CLI overrides -> manifest common -> BatchCommon (for scheduler-level config).
    merged = {**common.__dict__, **common_cfg, **cli_overrides}

    model = merged["model"]
    device = merged.get("device", "cuda")
    cache_dir = merged.get("cache_dir")
    use_local_scratch = merged.get("use_local_scratch", False)
    out_root = merged.get("out_root", "runs")
    resume = merged.get("resume", True)
    split_multi = merged.get("split_multi_structure", True)

    # Workflow-level overrides that apply to every job, before per-job specialization.
    # Batch-level keys (model/device/out_root/...) are stripped so they never reach
    # run_conformer_workflow(). Precedence (low -> high):
    #   CLI overrides < manifest common: < per-job flattened keys < per-job overrides:
    cli_workflow = {k: v for k, v in cli_overrides.items() if k not in _BATCH_KEYS}
    common_workflow = {k: v for k, v in common_cfg.items() if k not in _BATCH_KEYS}

    _ensure_dir(out_root)
    LOG.info(
        "Batch(manifest): model=%s device=%s cache=%s out_root=%s resume=%s split_multi=%s",
        model,
        device,
        cache_dir or "<default>",
        out_root,
        resume,
        split_multi,
    )

    # Prepare job list (resolve out_dir now)
    jobs: List[Dict[str, Any]] = []
    for j in jobs_cfg:
        xyz = j["xyz"]
        out_dir = _job_out_dir(out_root, xyz, j.get("out_dir"))
        # Flattened per-job keys: anything that isn't a structural key.
        flattened = {
            k: v
            for k, v in j.items()
            if k not in ("xyz", "out_dir", "overrides") and k not in _BATCH_KEYS
        }
        # Per-job `overrides:` is filtered too — otherwise a batch-level key written
        # there (e.g. split_multi_structure) reaches run_conformer_workflow as an
        # unexpected kwarg and kills the job.
        job_overrides = {
            k: v
            for k, v in (j.get("overrides", {}) or {}).items()
            if k not in _BATCH_KEYS
        }
        overrides = {
            **cli_workflow,
            **common_workflow,
            **flattened,
            **job_overrides,
        }
        _reject_ambiguous_ts_route(xyz, {**flattened, **job_overrides})
        job = {"xyz": xyz, "out_dir": out_dir, "overrides": overrides}
        # scan_shards is batch-level, so the filters above strip it — but a manifest
        # with one big grid next to five small scans is exactly when it is wanted per
        # job. Carry it through rather than dropping it silently.
        per_job_shards = j.get("scan_shards", (j.get("overrides") or {}).get("scan_shards"))
        if per_job_shards is not None:
            job["scan_shards"] = per_job_shards
        jobs.append(job)

    jobs = _expand_jobs_with_splitting(jobs, split_multi, resume=resume)
    return _run_parallel_jobs(jobs, merged)


def run_batch_from_glob(xyz_glob: List[str], common: BatchCommon, **overrides):
    xyz_paths = _expand_xyz_inputs(xyz_glob)
    if not xyz_paths:
        raise RuntimeError("No inputs matched.")

    merged = {**common.__dict__, **overrides}

    model = merged["model"]
    device = merged.get("device", "cuda")
    cache_dir = merged.get("cache_dir")
    out_root = merged.get("out_root", "runs")
    resume = merged.get("resume", True)
    split_multi = merged.get("split_multi_structure", True)

    # Batch-level config must not ride along as run_conformer_workflow kwargs.
    overrides = {k: v for k, v in overrides.items() if k not in _BATCH_KEYS}

    _ensure_dir(out_root)
    LOG.info(
        "Batch(glob): model=%s device=%s cache=%s out_root=%s resume=%s split_multi=%s",
        model,
        device,
        cache_dir or "<default>",
        out_root,
        resume,
        split_multi,
    )

    jobs: List[Dict[str, Any]] = [
        {
            "xyz": xyz,
            "out_dir": _job_out_dir(out_root, xyz, None),
            "overrides": overrides.copy(),
        }
        for xyz in xyz_paths
    ]

    jobs = _expand_jobs_with_splitting(jobs, split_multi, resume=resume)
    return _run_parallel_jobs(jobs, merged)


# =========================
# Split-ensemble aggregation
# =========================
def _aggregate_split_ensembles(jobs_in: List[Dict[str, Any]]) -> None:
    """Compile per-structure split jobs back into a single ranked ensemble.

    When ``split_multi_structure`` explodes ``mol.xyz`` into per-structure jobs under
    ``<root>/mol.ensemble/<label>/``, each writes its own single-row ``energies.csv``.
    After the parallel/serial phase we read those back, rank across all structures,
    and write ``<root>/mol.ensemble/energies.csv`` (+ ``optimized_ranked.xyz``).

    Runs even when members were skipped via resume (their CSVs already exist), so the
    aggregate is always regenerated. Non-split jobs (no ``_original_xyz``) are ignored
    — they already own their ensemble CSV directly.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for j in jobs_in:
        if not j.get("_original_xyz"):
            continue
        ens_dir = os.path.dirname(j["out_dir"])
        groups.setdefault(ens_dir, []).append(j)

    for ens_dir, members in groups.items():
        rows: List[Dict[str, Any]] = []
        xyz_by_label: Dict[str, str] = {}
        for j in members:
            member_dir = j["out_dir"]
            label = os.path.basename(member_dir)  # e.g. "conf0007"
            member_csv = os.path.join(member_dir, "energies.csv")
            if not os.path.isfile(member_csv):
                continue
            try:
                with open(member_csv, "r", newline="") as f:
                    for row in csv.DictReader(f):
                        # Each split job holds a single structure; relabel its row with
                        # the split label so tags/indices stay unique across the ensemble.
                        row["tag"] = label
                        digits = "".join(ch for ch in label if ch.isdigit())
                        if digits:
                            row["index"] = int(digits)
                        rows.append(row)
            except Exception as e:
                LOG.exception("Aggregation: failed reading %s: %s", member_csv, e)
                continue
            ranked_xyz = os.path.join(member_dir, "optimized_ranked.xyz")
            if os.path.isfile(ranked_xyz):
                xyz_by_label[label] = ranked_xyz

        if not rows:
            continue

        ranked, _e0 = rank_by_energy(rows)

        _ensure_dir(ens_dir)
        out_csv = os.path.join(ens_dir, "energies.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ENERGIES_FIELDS, extrasaction="ignore")
            w.writeheader()
            for rank, r in enumerate(ranked, start=1):
                r["rank"] = rank
                w.writerow({k: r.get(k) for k in ENERGIES_FIELDS})
        LOG.info(
            "Aggregated %d structures into ranked ensemble: %s", len(ranked), out_csv
        )

        # Concatenate per-structure optimized geometries in ranked order.
        out_xyz = os.path.join(ens_dir, "optimized_ranked.xyz")
        try:
            chunks: List[str] = []
            for r in ranked:
                p = xyz_by_label.get(r.get("tag"))
                if not p:
                    continue
                with open(p, "r") as xf:
                    txt = xf.read()
                if txt and not txt.endswith("\n"):
                    txt += "\n"
                chunks.append(txt)
            if chunks:
                with open(out_xyz, "w") as xf:
                    xf.write("".join(chunks))
        except Exception as e:
            LOG.exception("Aggregation: failed writing %s: %s", out_xyz, e)


# =========================
# Scheduler
# =========================
def _run_parallel_jobs(jobs_in: List[Dict[str, Any]], merged_common: Dict[str, Any]):
    """Run all jobs, then compile split and sharded jobs back into ranked ensembles.

    A sharded scan takes two passes: the spine jobs go through the queue alongside
    every ordinary job, and only once they have produced their seed geometries can
    the shard jobs be built and dispatched. Both passes use the same worker pool and
    the same shared queue, so a batch of several scans still balances as one pool.
    """
    jobs_in, plans = _plan_scan_shards(jobs_in, merged_common)
    summary = _run_parallel_jobs_impl(jobs_in, merged_common)

    if plans:
        try:
            shard_jobs = _build_shard_jobs(plans)
        except Exception as e:
            LOG.exception("Building scan shards failed: %s", e)
            shard_jobs = []
        if shard_jobs:
            summary = summary + _run_parallel_jobs_impl(shard_jobs, merged_common)
        try:
            # Spine jobs are scaffolding; the merged parents stand in for the jobs
            # they replaced so ensemble aggregation below sees ordinary members.
            spines = {p["spine_dir"] for p in plans}
            jobs_in = [
                j for j in jobs_in if j["out_dir"] not in spines
            ] + _aggregate_scan_shards(plans)
        except Exception as e:
            LOG.exception("Scan shard aggregation failed: %s", e)

    try:
        _aggregate_split_ensembles(jobs_in)
    except Exception as e:
        LOG.exception("Split-ensemble aggregation failed: %s", e)
    return summary


def _run_parallel_jobs_impl(
    jobs_in: List[Dict[str, Any]], merged_common: Dict[str, Any]
):
    """
    Fan out jobs across GPUs (``workers_per_gpu`` workers each, default 1).
    CPU fallback = serial loop.
    """
    out_root = merged_common.get("out_root", "runs")
    resume = merged_common.get("resume", True)
    device = merged_common.get("device", "cuda")
    cache_dir = merged_common.get("cache_dir")
    model = merged_common.get("model", "uma-m-1p1")
    use_local_scratch = merged_common.get("use_local_scratch", False)
    workers_per_gpu = max(1, int(merged_common.get("workers_per_gpu", 1) or 1))
    resume_from_per_common = merged_common.get("resume_from_per_conformer_csv", False)

    # Materialize job list with resume flag; skip already-done jobs
    jobs: List[Dict[str, Any]] = []
    skipped_summary: List[Dict[str, str]] = []
    for j in jobs_in:
        xyz = j["xyz"]
        out_dir = j["out_dir"]
        overrides = (j.get("overrides", {}) or {}).copy()
        cleanup_xyz = j.get("_cleanup_xyz")

        if resume and os.path.isfile(os.path.join(out_dir, "energies.csv")):
            LOG.info("Skipping (resume): %s", out_dir)
            display_xyz = j.get("_original_xyz", xyz)
            skipped_summary.append(
                {"xyz": display_xyz, "out_dir": out_dir, "status": "skipped"}
            )
            # Clean up temp file if it exists
            if cleanup_xyz and os.path.exists(cleanup_xyz):
                try:
                    os.remove(cleanup_xyz)
                except Exception:
                    pass
            continue

        jobs.append(
            {
                "xyz": xyz,
                "out_dir": out_dir,
                "overrides": overrides,
                "resume": resume,
                "_cleanup_xyz": cleanup_xyz,
                "_original_xyz": j.get("_original_xyz", xyz),
            }
        )

    if not jobs:
        return skipped_summary

    gpu_ids = _discover_gpus(device)
    if not gpu_ids:
        # Serial fallback
        LOG.info("No GPUs detected or device=cpu — running serial.")
        summary = skipped_summary.copy()
        for i, job in enumerate(jobs, start=1):
            xyz = job["xyz"]
            out_dir = job["out_dir"]
            overrides = (job.get("overrides", {}) or {}).copy()
            cleanup_xyz = job.get("_cleanup_xyz")
            display_xyz = job.get("_original_xyz", xyz)

            # same collision-avoidance as GPU path
            resume_from_per_conf = overrides.pop(
                "resume_from_per_conformer_csv", resume_from_per_common
            )

            LOG.info("=== Job %d/%d: %s → %s ===", i, len(jobs), xyz, out_dir)
            try:
                _ensure_dir(out_dir)
                run_conformer_workflow(
                    xyz,
                    out_dir,
                    model=model,
                    device="cpu",
                    cache_dir=cache_dir,
                    use_local_scratch=use_local_scratch,
                    resume_from_per_conformer_csv=resume_from_per_conf,
                    **overrides,
                )
                summary.append({"xyz": display_xyz, "out_dir": out_dir, "status": "ok"})

                # Cleanup temp file
                if cleanup_xyz and os.path.exists(cleanup_xyz):
                    try:
                        os.remove(cleanup_xyz)
                    except Exception:
                        pass

            except Exception as e:
                LOG.exception("Job failed: %s", e)
                summary.append(
                    {"xyz": display_xyz, "out_dir": out_dir, "status": f"error: {e}"}
                )
                # Cleanup temp file even on error
                if cleanup_xyz and os.path.exists(cleanup_xyz):
                    try:
                        os.remove(cleanup_xyz)
                    except Exception:
                        pass
        return summary

    # Parallel: `workers_per_gpu` workers per GPU, all pulling from one shared queue
    # (dynamic work-stealing, so uneven job costs self-balance).
    slots = _worker_slots(gpu_ids, workers_per_gpu)
    LOG.info(
        "Detected GPUs: %s | workers_per_gpu=%d | %d worker(s)",
        gpu_ids,
        workers_per_gpu,
        len(slots),
    )
    mp.set_start_method("spawn", force=True)

    task_q: mp.Queue = mp.Queue()
    result_q: mp.Queue = mp.Queue()

    # Enqueue all jobs, then one sentinel per worker
    for j in jobs:
        task_q.put(j)
    for _ in slots:
        task_q.put(None)

    worker_common = {
        "model": model,
        "device": "cuda",
        "cache_dir": cache_dir,
        "use_local_scratch": use_local_scratch,
        "out_root": out_root,
        "resume_from_per_conformer_csv": resume_from_per_common,
        "n_workers": len(slots),
    }

    procs: List[mp.Process] = []
    for gid in slots:
        p = mp.Process(target=_worker_loop, args=(gid, task_q, result_q, worker_common))
        p.daemon = True
        p.start()
        procs.append(p)

    # Collect results. Track outstanding jobs by out_dir so that if a worker dies
    # (segfault / OOM-kill) without posting a result we can detect it and bail out
    # instead of blocking forever on result_q.get().
    summary: List[Dict[str, str]] = skipped_summary.copy()
    pending: Dict[str, Dict[str, Any]] = {j["out_dir"]: j for j in jobs}

    def _record(res: Dict[str, str]):
        summary.append(res)
        pending.pop(res.get("out_dir"), None)

    while pending:
        try:
            _record(result_q.get(timeout=5.0))
        except queue.Empty:
            if any(p.is_alive() for p in procs):
                continue  # workers still churning; keep waiting
            # All workers have exited — drain any final results, then give up.
            while True:
                try:
                    _record(result_q.get_nowait())
                except queue.Empty:
                    break
            if pending:
                LOG.error(
                    "All workers exited with %d job(s) unfinished (likely crash/OOM).",
                    len(pending),
                )
                for out_dir, j in pending.items():
                    summary.append(
                        {
                            "xyz": j.get("_original_xyz", j.get("xyz")),
                            "out_dir": out_dir,
                            "status": "error: worker died before completion",
                        }
                    )
            break

    for p in procs:
        p.join()

    return summary
