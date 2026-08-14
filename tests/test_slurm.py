"""Generating a SLURM batch script.

All pure text and arithmetic — no cluster, no GPU, milliseconds. The script this
produces is meant to be read and submitted by hand, so the things worth pinning
are that it stays valid bash, that it reruns the command that was actually
typed, and that it refuses requests gpu_test will not honour.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from umadriver.slurm import (
    SlurmOpts,
    check_limits,
    job_name_from,
    render_script,
    strip_slurm_flags,
    write_script,
)


# ---------------------------------------------------------------- the command
def test_the_script_reruns_the_command_you_typed():
    argv = ["mol.xyz", "--optts", "--freq", "--temp", "218.15", "--submit"]

    assert strip_slurm_flags(argv) == [
        "mol.xyz",
        "--optts",
        "--freq",
        "--temp",
        "218.15",
    ]


@pytest.mark.parametrize(
    "argv",
    [
        ["a.xyz", "--submit", "--slurm-gpus", "4", "--opt", "Sella"],
        ["a.xyz", "--submit", "--slurm-gpus=4", "--opt", "Sella"],
    ],
)
def test_slurm_flags_are_dropped_in_both_spellings(argv):
    """--flag value and --flag=value; leaving either behind puts an unknown
    argument in the script and the job dies on the node."""
    assert strip_slurm_flags(argv) == ["a.xyz", "--opt", "Sella"]


def test_a_value_that_looks_like_a_flag_is_still_consumed():
    assert strip_slurm_flags(["--slurm-mem", "160g", "x.xyz"]) == ["x.xyz"]


def test_arguments_are_quoted():
    """Paths with spaces would otherwise split into two arguments on the node."""
    script = render_script(
        ["umadriver", "/some dir/mol.xyz"], SlurmOpts(), conda_env="ccml"
    )

    assert "'/some dir/mol.xyz'" in script


# ---------------------------------------------------------------- the preamble
def test_the_script_is_valid_bash(tmp_path):
    path = write_script(
        ["umadriver", "mol.xyz", "--optts"],
        SlurmOpts(job_name="uma_mol"),
        directory=str(tmp_path),
    )

    assert subprocess.run(["bash", "-n", path]).returncode == 0
    assert os.access(path, os.X_OK)


def test_the_preamble_sets_up_the_caches_and_environment():
    script = render_script(
        ["umadriver", "mol.xyz"],
        SlurmOpts(),
        cache_base="/n/netscratch/lab/me",
        conda_env="ccml",
    )

    # The model is several GB; pointing these at home is slow and blows the quota.
    for var in (
        "UMA_CACHE_BASE",
        "FAIRCHEM_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_HOME",
        "CUDA_CACHE_PATH",
    ):
        assert f"export {var}=" in script
    assert "/n/netscratch/lab/me" in script

    assert "module load cuda/12.9" in script
    assert "module load Mambaforge" in script
    # conda activate needs conda.sh sourced, and needs -u off while it runs.
    assert "conda.sh" in script
    assert "conda activate ccml" in script
    assert script.index("set +u") < script.index("conda activate ccml")


def test_threads_are_divided_between_the_workers():
    """One worker per GPU, so each gets cpus/gpus. Handing every worker the whole
    core count oversubscribes the node by a factor of the GPU count."""
    script = render_script(["umadriver", "m.xyz"], SlurmOpts(gpus=6, cpus=16))

    assert "export OMP_NUM_THREADS=2" in script


def test_the_headers_match_what_was_asked_for():
    script = render_script(
        ["umadriver", "m.xyz"],
        SlurmOpts(
            partition="gpu", gpus=4, cpus=32, mem="200g", time="1-00:00:00",
            job_name="big",
        ),
    )

    for line in (
        "#SBATCH -p gpu",
        "#SBATCH --gres=gpu:4",
        "#SBATCH -c 32",
        "#SBATCH --mem=200g",
        "#SBATCH -t 1-00:00:00",
        "#SBATCH -J big",
    ):
        assert line in script


# ---------------------------------------------------------------- gpu_test limits
def test_the_defaults_are_accepted():
    check_limits(SlurmOpts())


@pytest.mark.parametrize(
    "opts,message",
    [
        (SlurmOpts(time="24:00:00"), "12 h limit"),
        (SlurmOpts(time="1-00:00:00"), "12 h limit"),
        (SlurmOpts(gpus=12), "exceeds the limit of 8"),
        (SlurmOpts(gpus=6, cpus=48), "fewer than 8 per MIG slice"),
        (SlurmOpts(gpus=6, mem="400g"), "under 64 GB per MIG slice"),
        (SlurmOpts(gpus=2, mem="160g"), "under 64 GB per MIG slice"),
    ],
)
def test_gpu_test_limits_are_enforced(opts, message):
    """These do not fail at submission — the job either never schedules or dies
    on the node, hours later."""
    with pytest.raises(ValueError, match=message):
        check_limits(opts)


def test_other_partitions_are_the_users_business():
    """gpu allows 3 days and whole A100s; this module only knows gpu_test."""
    check_limits(SlurmOpts(partition="gpu", time="3-00:00:00", gpus=4, cpus=64))


def test_an_unreadable_memory_request_says_so():
    with pytest.raises(ValueError, match="cannot read"):
        check_limits(SlurmOpts(mem="lots"))


# ---------------------------------------------------------------- naming
@pytest.mark.parametrize(
    "inputs,manifest,expected",
    [
        (["mol.xyz"], None, "uma_mol"),
        (["/a/b/cat_intC.xyz"], None, "uma_cat_intC"),
        ([], "jobs.yaml", "uma_jobs"),
        (["mol.xyz"], "jobs.yaml", "uma_jobs"),
        ([], None, "umadriver"),
    ],
)
def test_job_name_follows_the_input(inputs, manifest, expected):
    assert job_name_from(inputs, manifest) == expected


def test_the_script_is_named_after_the_job(tmp_path):
    path = write_script(
        ["umadriver", "m.xyz"], SlurmOpts(job_name="uma_m"), directory=str(tmp_path)
    )

    assert os.path.basename(path) == "uma_m.sbatch"


def test_an_over_limit_request_writes_nothing(tmp_path):
    with pytest.raises(ValueError):
        write_script(
            ["umadriver", "m.xyz"],
            SlurmOpts(time="48:00:00", job_name="uma_m"),
            directory=str(tmp_path),
        )

    assert list(tmp_path.glob("*.sbatch")) == []


def test_submit_needs_something_to_run(monkeypatch, capsys):
    """Otherwise the script is written happily and fails on the node, after the
    job has already waited in the queue."""
    import sys

    from umadriver.driver import main

    monkeypatch.setattr(sys, "argv", ["umadriver", "--submit"])
    with pytest.raises(SystemExit):
        main()

    assert "needs something to run" in capsys.readouterr().err
