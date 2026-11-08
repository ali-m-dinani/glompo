"""Small non-ReaxFF benchmark for inspecting native GloMPO output."""

import argparse
from pathlib import Path

from glompo.core.manager import GloMPOManager
from glompo.convergence import MaxFuncCalls, MaxSeconds, NOptConverged
from glompo.generators import SinglePointGenerator
from glompo.opt_selectors import CycleSelector
from glompo.optimizers.cmawrapper import CMAOptimizer
from glompo.reaxkit_io import atomic_json, export_native
from glompo import __version__


def sphere(values):
    return sum(float(value) ** 2 for value in values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    bounds = [(-5, 5)] * 3
    selector = CycleSelector((CMAOptimizer, dict(workers=1, popsize=4, maxiter=3, verbose=False), dict(sigma0=0.5)))
    manager = GloMPOManager.new_manager(
        task=sphere, bounds=bounds, opt_selector=selector,
        working_dir=args.output, max_jobs=1, backend="threads",
        x0_generator=SinglePointGenerator(bounds, [1.0] * 3),
        convergence_checker=MaxFuncCalls(12) | MaxSeconds(session_max=30) | NOptConverged(1),
        summary_files=3, split_printstreams=False,
    )
    result = manager.start_manager()
    print(result)
    if not manager.converged:
        raise RuntimeError("Sample did not converge normally; inspect native logs")
    export_native(args.output)
    atomic_json(args.output / "glompo-profile.json", dict(
        profile="sphere", engine="analytic benchmark (no ReaxFF)",
        objective="sum_squared_coordinates", dataset_scope="synthetic 3D sphere benchmark",
        optimizer_version=__version__, parameter_space="physical"))


if __name__ == "__main__":
    main()
