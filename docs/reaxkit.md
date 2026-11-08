# ReaxKit file integration

This fork keeps GloMPO and its runner under the existing GPL-3.0 license.
ReaxKit launches a separate interpreter and consumes files; it does not import
GloMPO. AMS and ParAMS are separately licensed and are not distributed here.

## Local benchmark (no ReaxFF license required)

Use a dedicated Python 3.11 or 3.12 environment:

```bash
python -m pip install -r requirements-reaxkit.txt
python -m pip install -e .
python -m pip check
python examples/reaxkit_sample.py local_runs/sphere-smoke
python -m pytest tests/test_reaxkit_runner.py -q
```

The sample uses the actual GloMPO manager and CMA optimizer, three-dimensional
sphere objective, one optimizer worker, three CMA iterations and a nominal
12-evaluation budget. Budgets are checked at manager polling boundaries and
may overshoot. This is not a force-field fit or a cluster validation.

Observed Windows/Python 3.12 sample: 12 evaluations, initial score 3, final
score 1.1022127822092154. The stochastic result is not promised reproducible.
Native outputs are `glompo_log.h5`, `glompo_manager_log.yml`, and
`trajectories.png`. The sample also writes `glompo-profile.json`,
`glompo_history.jsonl` and `glompo_history.meta.json`.

The exporter reads numerical HDF5 table columns only, never checkpoint pickle
attributes. It exports every optimizer's call_id, fx, x and native row identity.
SHA256 hashes bind the JSON export to the native HDF5. Do not export while
the manager is writing. Re-export after an interrupted job has terminated:

```bash
python reaxkit_runner.py --export-native /path/to/native/manager
```

## AMS/ParAMS profile

`reaxkit_runner.py run-config.json --check-only` validates the contract and performs
strict classic-file conversion on temporary copies, including count/bounds and
initial scaled-vector checks, without running an engine. This profile uses upstream
`ReaxFFError.from_classic_files`, **not** AMS MCFF, OldReaxFF iopt optimization,
JAX or LAMMPS. A compatible licensed SCM Python stack must be available.
The upstream interface imports version-specific ParAMS internals; successful
installation of GloMPO does not establish compatibility with AMS 2023.103.
Do not install an unrelated package named `scm` from PyPI to satisfy imports.

The first profile supports one CMA optimizer with one worker in one Slurm task.
It retains the native ParAMS SSE objective, scaled [0,1] parameter space,
and supplied control-file geometry settings. Six-column params bounds are
converted according to the explicit input bounds order. Strict conversion
rejects missing jobs, changed bounds/counts, and out-of-bounds starting values;
it does not silently drop data or recenter a supplied force field.

Use `seed: null`; RNG forwarding and exact restart are not implemented.
Settings are max_function_calls, max_seconds, population_size, max_iterations,
and sigma0. max_seconds is a manager convergence check, not a hard timeout for
an individual engine call. Slurm walltime remains the hard job limit.

Successful runs write `native/manager/` history and `best/ffield` with
`best/candidate.json`. That field is a parameter-only restart artifact, not
an exact manager restart. Labeled parity data are not exported. Scientific
equivalence to the original trainset and other optimizers needs manual review.
No AMS/ParAMS calculation has yet been validated for this integration.

### AMS 2023.103 import compatibility

Roar's bundled Python 3.8 successfully imported the GloMPO manager, CMA wrapper
and history exporter with its existing NumPy 1.21.2, PyYAML 5.4.1, PyTables 3.7.0,
CMA 3.2.2 and Matplotlib 3.5.1. This is import evidence only, not an engine test.
Run from the checkout with `amspython`; do not install the modern pinned
requirements into AMS's shared environment.

The optional `GlompoParamsWrapper` requires a ParAMS optimizer API absent in
this installation. Its absence no longer prevents importing the direct
`ReaxFFError` interface used by the ReaxKit runner. Instantiating that optional
wrapper still raises an explicit error; unrelated dependency failures propagate.

### Verified cobalt conversion and boundary handling

The user-run AMS 2023.103 converter produced 12 active parameters, 144 energy
targets and 147 jobs. Inspection confirmed the requested parameter bounds,
target expressions/references, and Weight/Sigma-squared equal to inverse
accuracy-squared. Jobs comprise 130 geometry optimizations and 17 single points;
this dataset is not an all-single-point smoke test.

The ReaxKit runner opts into `scm_conversion=True`: explicit `use_weights=True`,
the control-file path passed as `normal_run_settings`, and default ReaxParams
construction. It stages six-column params in minimum/maximum order, matching
the successfully converted input, regardless of the declared original order.
It calls the installed SCM library; no SCM script is copied or redistributed.

Strict mode accepts finite starting values on either bound without modifying
them (including cobalt p_ovun2=-2.5 at its upper bound). Invalid/nonfinite ranges,
out-of-bounds values and missing jobs still fail. Existing non-strict callers
retain their previous midpoint behavior. No clipping or recentering is applied
to the ReaxKit starting field. Actual runs preserve converted YAML files in
`native/converted_to_params` for review. Conversion and engine compatibility
of the updated runner must still be checked on Roar; local tests mock SCM APIs.
