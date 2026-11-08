import ast
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pytest


def conversion_function(tmp_path, value, bounds=(-25.0, -2.5), missing_jobs=None):
    source = Path(__file__).parents[1] / "glompo/interfaces/params.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "setup_reax_from_classic")
    for name in ("ffield", "geo", "trainset.in", "control", "params"):
        (tmp_path / name).write_text("fixture\n")
    parameter = SimpleNamespace(name="Co:p_ovun2", value=value, range=bounds)
    calls = []
    engine = SimpleNamespace(active=[parameter], read_paramsfile=lambda path: calls.append(("params", path)))
    dataset = SimpleNamespace(check_consistency=lambda jobs: missing_jobs or [], forces=lambda: False)

    def trainset(path, **kwargs):
        calls.append(("trainset", kwargs))
        return dataset

    def geo(path, *args, **kwargs):
        calls.append(("geo", args, kwargs))
        return "jobs"

    def parameters(path, **kwargs):
        calls.append(("ffield", kwargs))
        return engine

    namespace = dict(Path=Path, np=np, warnings=warnings, PARAMS_VERSION_INFO=(2023, 103),
                     trainset_to_params=trainset, geo_to_params=geo, ReaxParams=parameters,
                     reaxff_control_to_settings=lambda path: "legacy settings")
    future = ast.parse("from __future__ import annotations").body[0]
    exec(compile(ast.Module(body=[future, function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["setup_reax_from_classic"], parameter, calls


@pytest.mark.parametrize("value", [-25.0, -2.5, -10.0])
def test_inclusive_bounds_and_scm_conversion(tmp_path, value):
    convert, parameter, calls = conversion_function(tmp_path, value)
    convert(tmp_path, strict=True, scm_conversion=True)
    assert parameter.value == value
    assert parameter.range == (-25.0, -2.5)
    assert ("trainset", {"use_weights": True}) in calls
    assert ("geo", (), {"normal_run_settings": str(tmp_path / "control")}) in calls
    assert ("ffield", {}) in calls


@pytest.mark.parametrize("value,bounds", [
    (-25.1, (-25, -2.5)), (-2.4, (-25, -2.5)), (float("nan"), (-25, -2.5)),
    (-2.5, (-2.5, -2.5)), (-2.5, (-25, float("inf"))), (-2.5, (-2, -25)),
])
def test_strict_invalid_values_are_not_recentered(tmp_path, value, bounds):
    convert, parameter, _ = conversion_function(tmp_path, value, bounds)
    with pytest.raises(ValueError, match="invalid/out of bounds"):
        convert(tmp_path, strict=True, scm_conversion=True)
    assert parameter.value == value or np.isnan(parameter.value)


def test_missing_jobs_not_dropped(tmp_path):
    convert, _, _ = conversion_function(tmp_path, -2.5, missing_jobs=[4])
    with pytest.raises(ValueError, match="unavailable jobs"):
        convert(tmp_path, strict=True, scm_conversion=True)


def test_missing_control_not_defaulted(tmp_path):
    convert, _, _ = conversion_function(tmp_path, -2.5)
    (tmp_path / "control").unlink()
    with pytest.raises(ValueError, match="requires control"):
        convert(tmp_path, strict=True, scm_conversion=True)


def test_legacy_default_behavior_unchanged(tmp_path):
    convert, parameter, calls = conversion_function(tmp_path, -2.5)
    with pytest.warns(UserWarning, match="midpoint"):
        convert(tmp_path)
    assert parameter.value == (-25 - 2.5) / 2
    assert ("trainset", {}) in calls
    assert ("geo", ("legacy settings",), {}) in calls
