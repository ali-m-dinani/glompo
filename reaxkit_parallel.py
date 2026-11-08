"""Resource checks for geometry-level ParAMS parallelism."""

import os


def resource_plan(job_workers, environment, affinity=None, check_only=False):
    if type(job_workers) is not int or job_workers < 1:
        raise ValueError("job_workers must be a positive integer")
    for name in ("SLURM_NTASKS", "SLURM_JOB_NUM_NODES", "SLURM_NNODES"):
        if int(environment.get(name, "1")) != 1:
            raise ValueError("GloMPO geometry parallelism requires one Slurm node and one task")
    allocated = bool(environment.get("SLURM_JOB_ID"))
    if job_workers > 1 and not allocated and not check_only:
        raise ValueError("Parallel execution requires a Slurm allocation")
    if allocated:
        cpus = int(environment.get("SLURM_CPUS_PER_TASK", "1"))
        if cpus < job_workers:
            raise ValueError("job_workers exceeds SLURM_CPUS_PER_TASK")
    if affinity is not None and (allocated or not check_only) and affinity < job_workers:
        raise ValueError("job_workers exceeds process CPU affinity")
    return dict(job_workers=job_workers, optimizer_workers=1, candidate_workers=1,
                engine_processes=1, engine_threads=1,
                allocation_checked=allocated, affinity_cpus=affinity)


def configure_task(task, workers):
    from glompo.interfaces.params import ParallelLevels
    task.par_levels = ParallelLevels(optimizations=1, parametervectors=1,
                                    jobs=workers, processes=1, threads=1)


def engine_environment():
    os.environ.pop("SCM_DISABLE_MPI", None)
    for name in ("NSCM", "OMP_NUM_THREADS", "SCM_USE_THREADS", "OPENBLAS_NUM_THREADS",
                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["OMP_DYNAMIC"] = "FALSE"
