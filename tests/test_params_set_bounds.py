import ast
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pytest


def objective_stub():
    source = Path(__file__).parents[1] / "glompo/interfaces/params.py"
    tree = ast.parse(source.read_text())
    base = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == "BaseParamsError")
    method = next(node for node in base.body if isinstance(node, ast.FunctionDef)
                  and node.name == "set_parameters")
    future = ast.parse("from __future__ import annotations").body[0]
    namespace = dict(np=np, warnings=warnings)
    exec(compile(ast.Module(body=[future, method], type_ignores=[]), str(source), "exec"), namespace)
    engine = SimpleNamespace(range=[(100, 200), (-25, -2.5)], x=None,
                             active=SimpleNamespace(range=[(-25, -2.5)], x=None))
    objective = SimpleNamespace(par_eng=engine,
        convert_parms_scaled2real=lambda values: [-25 + float(value) * 22.5 for value in values])
    return namespace["set_parameters"], objective


@pytest.mark.parametrize("value", [-25.0, -2.5, -10.0])
def test_active_ranges_and_inclusive_endpoints(value):
    setter, objective = objective_stub()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        setter(objective, [value], "real")
    assert not caught
    assert objective.par_eng.active.x == [value]


def test_full_ranges_and_scaled_endpoint():
    setter, objective = objective_stub()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        setter(objective, [100, -2.5], "real", full=True)
        setter(objective, [1.0], "scaled")
    assert not caught
    assert objective.par_eng.x == [100, -2.5]
    assert objective.par_eng.active.x == [-2.5]


@pytest.mark.parametrize("value", [-26, -2.4, float("nan"), float("inf")])
def test_invalid_active_values_still_warn(value):
    setter, objective = objective_stub()
    with pytest.warns(UserWarning, match="outside their bounds"):
        setter(objective, [value], "real")


def test_length_mismatch_rejected():
    setter, objective = objective_stub()
    with pytest.raises(ValueError, match="length"):
        setter(objective, [-2.5, -2.5], "real")
