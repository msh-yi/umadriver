# umadriver/batch.py
from __future__ import annotations
import os, glob, logging, json, multiprocessing as mp, queue
from typing import Any, Dict, List, Optional, Iterable
from dataclasses import dataclass

LOG = logging.getLogger("uma.batch")

try:
    import yaml  # optional
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False

from .ensemble import run_conformer_workflow, _ensure_dir


# =========================
# Config
# =========================
@dataclass
class BatchCommon:
    model: str = "uma-m-1p1"
    device: str = "cuda"        # "cuda" | "cpu"
    cache_dir: Optional[str] = None
    use_local_scratch: bool = False
    out_root: str = "runs"
    resume: bool = True         # skip job if energies.csv exists


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
    if explicit:
        return explicit
    base = os.path.splitext(os.path.basename(xyz_path))[0]
    return os.path.join(out_root, base)


def _expand_xyz_inputs(xyz_list: Iterable[str]) -> List[str]:
    out: List[str] = []
    for s in xyz_list:
        if any(ch in s for ch in "*?[]"):
            out.extend(sorted(glob.glob(s)))
        else:
            out.append(s)
    return out


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
    # Bind this child process to exactly one GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Optional: per-GPU scratch segregation to reduce cache contention.
    # If UMA_SCRATCH_ROOT is already set by the user, keep it; otherwise shard by GPU.
    if "UMA_SCRATCH_ROOT" not in os.environ:
        os.environ["UMA_SCRATCH_ROOT"] = os.path.join(out_root, f"_gpu{gpu_id}_scratch")


# =========================
# Worker
# =========================
def _worker_loop(gpu_id: int,
                 task_q: mp.Queue,
                 result_q: mp.Queue,
                 common: Dict[str, Any]):
    """
    Each worker:
      - pins to a single GPU
      - processes a stream of jobs (each job is one ensemble XYZ)
      - builds nothing global in the parent; run_conformer_workflow handles calc internally
    """
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
        overrides = job.get("overrides", {})
        resume = job["resume"]

        try:
            energies_csv = os.path.join(out_dir, "energies.csv")
            if resume and os.path.isfile(energies_csv):
                LOG.info("[GPU %d] SKIP (resume): %s", gpu_id, out_dir)
                result_q.put({"xyz": xyz, "out_dir": out_dir, "status": "skipped"})
                continue

            _ensure_dir(out_dir)
            LOG.info("[GPU %d] RUN: %s → %s", gpu_id, xyz, out_dir)
            # NOTE: do NOT pass a calculator object; each job constructs its own inside ensemble.py
            run_conformer_workflow(
                xyz,
                out_dir,
                model=common["model"],
                device="cuda",  # child sees only one GPU
                cache_dir=common["cache_dir"],
                use_local_scratch=common["use_local_scratch"],
                **overrides,
            )
            result_q.put({"xyz": xyz, "out_dir": out_dir, "status": "ok"})
        except Exception as e:
            LOG.exception("[GPU %d] Job failed: %s", gpu_id, e)
            result_q.put({"xyz": xyz, "out_dir": out_dir, "status": f"error: {e}"})


# =========================
# Public API
# =========================
def run_batch_from_manifest(manifest_path: str, common: BatchCommon, **cli_overrides):
    cfg = _load_manifest(manifest_path)
    common_cfg = cfg.get("common", {})
    jobs_cfg = cfg.get("jobs", [])

    # Merge CLI overrides -> manifest common -> BatchCommon
    merged = {**common.__dict__, **common_cfg, **cli_overrides}
    model = merged["model"]
    device = merged["device"]
    cache_dir = merged.get("cache_dir")
    use_local_scratch = merged.get("use_local_scratch", False)
    out_root = merged.get("out_root", "runs")
    resume = merged.get("resume", True)

    _ensure_dir(out_root)
    LOG.info("Batch(manifest): model=%s device=%s cache=%s out_root=%s resume=%s",
             model, device, cache_dir or "<default>", out_root, resume)

    # Prepare job list (resolve out_dir now; also filter resumed ones later)
    jobs: List[Dict[str, Any]] = []
    for j in jobs_cfg:
        xyz = j["xyz"]
        out_dir = _job_out_dir(out_root, xyz, j.get("out_dir"))
        overrides = j.get("overrides", {})
        jobs.append({"xyz": xyz, "out_dir": out_dir, "overrides": overrides})

    return _run_parallel_jobs(jobs, merged)


def run_batch_from_glob(xyz_glob: List[str], common: BatchCommon, **overrides):
    # Expand globs
    xyz_paths = _expand_xyz_inputs(xyz_glob)
    if not xyz_paths:
        raise RuntimeError("No inputs matched.")

    merged = {**common.__dict__, **overrides}
    model = merged["model"]
    device = merged["device"]
    cache_dir = merged.get("cache_dir")
    out_root = merged.get("out_root", "runs")
    resume = merged.get("resume", True)

    _ensure_dir(out_root)
    LOG.info("Batch(glob): model=%s device=%s cache=%s out_root=%s resume=%s",
             model, device, cache_dir or "<default>", out_root, resume)

    jobs: List[Dict[str, Any]] = []
    for xyz in xyz_paths:
        out_dir = _job_out_dir(out_root, xyz, None)
        jobs.append({"xyz": xyz, "out_dir": out_dir, "overrides": {}})

    return _run_parallel_jobs(jobs, merged)


# =========================
# Scheduler
# =========================
def _run_parallel_jobs(jobs_in: List[Dict[str, Any]], merged_common: Dict[str, Any]):
    """
    Fan out jobs across GPUs (one worker per GPU). CPU fallback = serial loop.
    """
    out_root = merged_common.get("out_root", "runs")
    resume = merged_common.get("resume", True)
    device = merged_common.get("device", "cuda")
    cache_dir = merged_common.get("cache_dir")
    model = merged_common.get("model", "uma-m-1p1")
    use_local_scratch = merged_common.get("use_local_scratch", False)

    # Materialize job list with resume flag
    jobs = []
    skipped_summary = []
    for j in jobs_in:
        xyz = j["xyz"]
        out_dir = j["out_dir"]
        overrides = j.get("overrides", {})
        if resume and os.path.isfile(os.path.join(out_dir, "energies.csv")):
            LOG.info("Skipping (resume): %s", out_dir)
            skipped_summary.append({"xyz": xyz, "out_dir": out_dir, "status": "skipped"})
            continue
        jobs.append({"xyz": xyz, "out_dir": out_dir, "overrides": overrides, "resume": resume})

    if not jobs:
        return skipped_summary

    gpu_ids = _discover_gpus(device)
    if not gpu_ids:
        # Serial fallback
        LOG.info("No GPUs detected or device=cpu — running serial.")
        summary = skipped_summary.copy()
        for i, job in enumerate(jobs, start=1):
            xyz, out_dir, overrides = job["xyz"], job["out_dir"], job["overrides"]
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
                    **overrides,
                )
                summary.append({"xyz": xyz, "out_dir": out_dir, "status": "ok"})
            except Exception as e:
                LOG.exception("Job failed: %s", e)
                summary.append({"xyz": xyz, "out_dir": out_dir, "status": f"error: {e}"})
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

    # Start workers
    procs: List[mp.Process] = []
    worker_common = {
        "model": model,
        "device": "cuda",
        "cache_dir": cache_dir,
        "use_local_scratch": use_local_scratch,
        "out_root": out_root,
    }
    for gid in gpu_ids:
        p = mp.Process(target=_worker_loop, args=(gid, task_q, result_q, worker_common))
        p.daemon = True
        p.start()
        procs.append(p)

    # Collect results
    summary: List[Dict[str, str]] = skipped_summary.copy()
    remaining = len(jobs)
    while remaining > 0:
        res = result_q.get()
        summary.append(res)
        remaining -= 1

    # Join
    for p in procs:
        p.join()

    return summary
