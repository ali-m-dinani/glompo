from types import ModuleType, SimpleNamespace
import sys

import pytest

from reaxkit_parallel import configure_task, engine_environment, resource_plan


def test_serial_and_deferred_parallel_checks():
    assert resource_plan(1, {})["job_workers"] == 1
    assert not resource_plan(4, {}, 1, check_only=True)["allocation_checked"]
    with pytest.raises(ValueError, match="allocation"):
        resource_plan(4, {})


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "4"])
def test_invalid_workers(workers):
    with pytest.raises(ValueError, match="positive integer"):
        resource_plan(workers, {})


def test_allocation_and_affinity():
    environment = dict(SLURM_JOB_ID="123", SLURM_CPUS_PER_TASK="4")
    assert resource_plan(4, environment, 4)["allocation_checked"]
    with pytest.raises(ValueError, match="CPUS_PER_TASK"):
        resource_plan(5, environment, 8)
    with pytest.raises(ValueError, match="affinity"):
        resource_plan(4, environment, 2)
    for name in ("SLURM_NTASKS", "SLURM_JOB_NUM_NODES", "SLURM_NNODES"):
        with pytest.raises(ValueError, match="one Slurm node"):
            resource_plan(4, dict(environment, **{name: "2"}), 8)


def test_native_parallel_levels(monkeypatch):
    module = ModuleType("glompo.interfaces.params")
    module.ParallelLevels = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "glompo.interfaces.params", module)
    task = SimpleNamespace()
    configure_task(task, 4)
    assert task.par_levels == dict(jobs=4, processes=1, threads=1)


def test_engine_environment(monkeypatch):
    import os
    names = ("NSCM", "OMP_NUM_THREADS", "SCM_USE_THREADS", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
    for name in names:
        monkeypatch.setenv(name, "8")
    monkeypatch.setenv("OMP_DYNAMIC", "TRUE")
    monkeypatch.setenv("SCM_DISABLE_MPI", "1")
    engine_environment()
    assert all(os.environ[name] == "1" for name in names)
    assert "SCM_DISABLE_MPI" not in os.environ
    assert os.environ["OMP_DYNAMIC"] == "FALSE"
