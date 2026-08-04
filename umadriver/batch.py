from __future__ import annotations

import os
import sys
import glob
import logging
import json
import multiprocessing as mp
import queue
from typing import Any, Dict, List, Optional, Iterable
from dataclasses import dataclass

LOG = logging.getLogger("uma.batch")

try:
    import yaml  # optional

    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False

from .ensemble import (
    run_conformer_workflow,
    _ensure_dir,
    rank_by_energy,
    ENERGIES_FIELDS,
)


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


def _parse_visible_devices_env() -> Optional[List[int]]:
    s = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not s:
        return None
    try:
        devs = [int(x) for x in s.split(",") if x.strip() != ""]
        return devs if devs else None
    except Exception:
        return None


def _discover_gpus(device: str) -> List[int]:
    """
    Returns a list of GPU ids visible to this process.
    If device is cpu -> [].
    """
    if device.lower() == "cpu":
        return []

    env_ids = _parse_visible_devices_env()
    if env_ids is not None:
        return env_ids

    try:
        import torch

        n = torch.cuda.device_count()
        return list(range(n)) if n > 0 else []
    except Exception:
        return []


def _bind_gpu_env(gpu_id: int, out_root: str):
    """
    Bind this child process to exactly one GPU.
    Also shard scratch if UMA_SCRATCH_ROOT is not already set.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # If user did not specify UMA_SCRATCH_ROOT, shard by GPU to reduce contention.
    if "UMA_SCRATCH_ROOT" not in os.environ:
        os.environ["UMA_SCRATCH_ROOT"] = os.path.join(out_root, f"_gpu{gpu_id}_scratch")


# =========================
# Worker
# =========================
def _worker_loop(
    gpu_id: int,
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
    _bind_gpu_env(gpu_id, common["out_root"])
    LOG.info("[GPU %d] worker start", gpu_id)

    while True:
        try:
            job = task_q.get(timeout=2.0)
        except queue.Empty:
            continue

        if job is None:
            LOG.info("[GPU %d] worker exit", gpu_id)
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

        # Remove batch-level parameters that shouldn't be passed to run_conformer_workflow
        overrides.pop("split_multi_structure", None)

        try:
            energies_csv = os.path.join(out_dir, "energies.csv")
            if resume and os.path.isfile(energies_csv):
                LOG.info("[GPU %d] SKIP (resume): %s", gpu_id, out_dir)
                result_q.put({"xyz": xyz, "out_dir": out_dir, "status": "skipped"})
                if cleanup_xyz and os.path.exists(cleanup_xyz):
                    try:
                        os.remove(cleanup_xyz)
                    except Exception:
                        pass
                continue

            _ensure_dir(out_dir)
            LOG.info("[GPU %d] RUN: %s → %s", gpu_id, xyz, out_dir)

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
            LOG.exception("[GPU %d] Job failed: %s", gpu_id, e)
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

    # Workflow-level overrides that apply to every job, before per-job specialization.
    # Batch-level keys (model/device/out_root/...) are stripped so they never reach
    # run_conformer_workflow(). Precedence (low -> high):
    #   CLI overrides < manifest common: < per-job flattened keys < per-job overrides:
    cli_workflow = {k: v for k, v in cli_overrides.items() if k not in _BATCH_KEYS}
    common_workflow = {k: v for k, v in common_cfg.items() if k not in _BATCH_KEYS}

    _ensure_dir(out_root)
    LOG.info(
        "Batch(manifest): model=%s device=%s cache=%s out_root=%s resume=%s",
        model,
        device,
        cache_dir or "<default>",
        out_root,
        resume,
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
        job_overrides = (j.get("overrides", {}) or {})
        overrides = {
            **cli_workflow,
            **common_workflow,
            **flattened,
            **job_overrides,
        }
        jobs.append({"xyz": xyz, "out_dir": out_dir, "overrides": overrides})

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

    jobs: List[Dict[str, Any]] = []
    for xyz in xyz_paths:
        base_name = os.path.splitext(os.path.basename(xyz))[0]

        if split_multi:
            structures = _split_xyz_into_structures(xyz)

            if len(structures) == 1:
                # Single structure - keep original behavior
                out_dir = _job_out_dir(out_root, xyz, None)
                jobs.append(
                    {"xyz": xyz, "out_dir": out_dir, "overrides": overrides.copy()}
                )
                LOG.info(f"Added job: {xyz} (single structure)")
            else:
                # Multiple structures - create one job per structure
                LOG.info(f"Splitting {xyz} into {len(structures)} structures")
                for content, label in structures:
                    # Create temporary single-structure XYZ file
                    temp_dir = os.path.join(out_root, ".tmp")
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_xyz = os.path.join(temp_dir, f"{base_name}_{label}.xyz")
                    with open(temp_xyz, "w") as f:
                        f.write(content)

                    out_dir = os.path.join(out_root, f"{base_name}.ensemble", label)
                    jobs.append(
                        {
                            "xyz": temp_xyz,
                            "out_dir": out_dir,
                            "overrides": overrides.copy(),
                            "_cleanup_xyz": temp_xyz,
                            "_original_xyz": xyz,
                        }
                    )
        else:
            # Original behavior: treat whole file as one job
            out_dir = _job_out_dir(out_root, xyz, None)
            jobs.append({"xyz": xyz, "out_dir": out_dir, "overrides": overrides.copy()})

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
    """Run all jobs, then compile any split multi-structure jobs into ranked ensembles."""
    summary = _run_parallel_jobs_impl(jobs_in, merged_common)
    try:
        _aggregate_split_ensembles(jobs_in)
    except Exception as e:
        LOG.exception("Split-ensemble aggregation failed: %s", e)
    return summary


def _run_parallel_jobs_impl(
    jobs_in: List[Dict[str, Any]], merged_common: Dict[str, Any]
):
    """
    Fan out jobs across GPUs (one worker per GPU). CPU fallback = serial loop.
    """
    out_root = merged_common.get("out_root", "runs")
    resume = merged_common.get("resume", True)
    device = merged_common.get("device", "cuda")
    cache_dir = merged_common.get("cache_dir")
    model = merged_common.get("model", "uma-m-1p1")
    use_local_scratch = merged_common.get("use_local_scratch", False)
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

            # Remove batch-level parameters that shouldn't be passed to run_conformer_workflow
            overrides.pop("split_multi_structure", None)

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

    # Parallel: one worker per GPU
    LOG.info("Detected GPUs: %s", gpu_ids)
    mp.set_start_method("spawn", force=True)

    task_q: mp.Queue = mp.Queue()
    result_q: mp.Queue = mp.Queue()

    # Enqueue all jobs, then one sentinel per worker
    for j in jobs:
        task_q.put(j)
    for _ in gpu_ids:
        task_q.put(None)

    worker_common = {
        "model": model,
        "device": "cuda",
        "cache_dir": cache_dir,
        "use_local_scratch": use_local_scratch,
        "out_root": out_root,
        "resume_from_per_conformer_csv": resume_from_per_common,
    }

    procs: List[mp.Process] = []
    for gid in gpu_ids:
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
