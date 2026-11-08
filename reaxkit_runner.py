"""Run the existing AMS/ParAMS GloMPO objective through ReaxKit's file contract."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile


def finite(value):
    result = float(str(value).replace("D", "E").replace("d", "e"))
    if not math.isfinite(result):
        raise ValueError("Nonfinite parameter or setting")
    return result


def load_request(path):
    request = json.loads(Path(path).read_text())
    if request.get("contract") != "reaxkit-external-runner-v1" or request.get("optimizer") != "glompo":
        raise ValueError("Expected a glompo external-runner request")
    if request.get("seed") is not None:
        raise ValueError("Use seed: null; multi-optimizer RNG forwarding is not verified")
    settings = request["settings"]
    allowed = {"max_function_calls", "max_seconds", "population_size", "max_iterations", "sigma0"}
    if set(settings) - allowed:
        raise ValueError("Unsupported GloMPO runner settings")
    for key in ("max_function_calls", "population_size", "max_iterations"):
        if type(settings.get(key)) is not int or settings[key] < (2 if key == "population_size" else 1):
            raise ValueError(f"Invalid {key}")
    if finite(settings["max_seconds"]) <= 0 or not 0 < finite(settings["sigma0"]) <= 1:
        raise ValueError("Positive timeout and sigma0 in (0,1] required")
    for key in ("ffield", "params", "geo", "trainset", "control"):
        if not request["inputs"].get(key) or not Path(request["inputs"][key]).is_file():
            raise ValueError(f"AMS/ParAMS profile requires input file: {key}")
    order = request.get("params_bounds_order")
    if order not in ("min_max", "max_min"):
        raise ValueError("Explicit params_bounds_order is required")
    rows, locations, bounds = [], set(), []
    for line in Path(request["inputs"]["params"]).read_text().splitlines():
        fields = re.split(r"[!#]", line, maxsplit=1)[0].split()
        if not fields:
            continue
        if len(fields) != 6:
            raise ValueError("params requires six columns")
        location = tuple(map(int, fields[:3]))
        if location in locations or min(location) < 0:
            raise ValueError("Duplicate/invalid params address")
        locations.add(location)
        step, first, second = map(finite, fields[3:])
        lower, upper = (first, second) if order == "min_max" else (second, first)
        if lower >= upper:
            raise ValueError("Parameter bounds must be strictly increasing")
        bounds.append((lower, upper))
        rows.append(" ".join(map(str, location)) + f" {step:.17g} {lower:.17g} {upper:.17g}\n")
    if not rows:
        raise ValueError("No active parameters")
    return request, rows, bounds


def convert_inputs(request, rows, expected_bounds, directory, objective_class):
    directory = Path(directory)
    directory.mkdir()
    for key, name in {"ffield": "ffield", "geo": "geo", "trainset": "trainset.in", "control": "control"}.items():
        shutil.copyfile(request["inputs"][key], directory / name)
    (directory / "params").write_text("".join(rows))
    task = objective_class.from_classic_files(directory, strict_inputs=True, scm_conversion=True)
    actual_bounds = [tuple(map(float, pair)) for pair in task.par_eng.active.range]
    if task.n_parms != len(rows) or sorted(actual_bounds) != sorted(expected_bounds):
        raise ValueError("ParAMS active parameter count/bounds differ from requested params; aborting")
    return task


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_config", nargs="?", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--export-native", type=Path)
    args = parser.parse_args(argv)
    if args.export_native:
        if args.run_config or args.check_only:
            parser.error("--export-native cannot be combined with other actions")
        from glompo.reaxkit_io import export_native
        print(f"Exported {export_native(args.export_native)} evaluations")
        return 0
    if not args.run_config:
        parser.error("run_config is required")
    request, rows, expected_bounds = load_request(args.run_config)
    try:
        from glompo.interfaces.params import ReaxFFError
    except (ImportError, AttributeError) as error:
        raise RuntimeError("A compatible licensed AMS/ParAMS Python installation is required; load AMS and verify glompo.interfaces.params imports before submission") from error
    if args.check_only:
        with tempfile.TemporaryDirectory(prefix="glompo-conversion-") as temporary:
            task = convert_inputs(request, rows, expected_bounds, Path(temporary) / "inputs", ReaxFFError)
            initial = list(map(float, task.convert_parms_real2scaled(task.par_eng.active.x)))
            print(f"Classic conversion passed for {task.n_parms} parameters; scaled initial values: {initial}")
        print("Engine execution remains unverified; no evaluations performed")
        return 0
    from glompo import __version__
    from glompo.convergence import MaxFuncCalls, MaxSeconds, NOptConverged
    from glompo.core.manager import GloMPOManager
    from glompo.generators import SinglePointGenerator
    from glompo.opt_selectors import CycleSelector
    from glompo.optimizers.cmawrapper import CMAOptimizer
    from glompo.reaxkit_io import atomic_json, export_native

    if int(os.environ.get("SLURM_NTASKS", "1")) != 1:
        raise ValueError("This initial serial profile requires one Slurm task")
    root = Path(request["native_directory"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    inputs = root / "ams-inputs"
    if inputs.exists() or (root / "manager").exists():
        raise ValueError("Run already prepared/started; use a fresh campaign")
    task = convert_inputs(request, rows, expected_bounds, inputs, ReaxFFError)
    converted = root / "converted_to_params"
    converted.mkdir()
    task.dat_set.store(str(converted / "training_set.yaml"))
    task.job_col.store(str(converted / "job_collection.yaml"))
    task.par_eng.yaml_store(str(converted / "parameter_interface.yaml"))
    settings = request["settings"]
    initial = task.convert_parms_real2scaled(task.par_eng.active.x)
    selector = CycleSelector((CMAOptimizer,
                              dict(workers=1, popsize=settings["population_size"],
                                   maxiter=settings["max_iterations"], verbose=False),
                              dict(sigma0=settings["sigma0"], preserve_initial=True)))
    directory = root / "manager"
    directory.mkdir()
    manager = GloMPOManager.new_manager(
        task=task, bounds=task.bounds, opt_selector=selector, working_dir=directory,
        max_jobs=1, backend="threads", x0_generator=SinglePointGenerator(task.bounds, initial),
        convergence_checker=MaxFuncCalls(settings["max_function_calls"]) | MaxSeconds(session_max=settings["max_seconds"]) | NOptConverged(1),
        summary_files=3, split_printstreams=False,
    )
    atomic_json(directory / "glompo-profile.json", dict(
        profile="ams_params", engine="SCM AMS/ParAMS ReaxFF", objective="ParAMS SSE (native dataset sigma/weights)",
        dataset_scope="ParAMS-converted trainset; scientific equivalence unverified",
        optimizer_version=__version__, parameter_space="scaled_0_1",
        initial_candidate_policy="exact_first_evaluation",
        initial_parameters_scaled=list(map(float, initial))))
    result = manager.start_manager()
    export_native(directory)
    if not manager.converged or result.x is None or not math.isfinite(float(result.fx)):
        raise RuntimeError("GloMPO did not return a finite converged result; inspect native logs")
    best = directory / "best"
    best.mkdir()
    task.set_parameters(result.x, "scaled")
    task.save(best)
    atomic_json(best / "candidate.json", dict(score=float(result.fx), parameters_scaled=list(map(float, result.x)),
                                             origin=result.origin, restart="parameter_only"))
    print(f"Completed GloMPO: {result.fx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
