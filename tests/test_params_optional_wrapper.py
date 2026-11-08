import ast
import builtins
from pathlib import Path

import pytest


def wrapper_namespace(monkeypatch, missing=None):
    source = Path(__file__).parents[1] / "glompo/interfaces/params.py"
    tree = ast.parse(source.read_text())
    optional = next(node for node in tree.body if isinstance(node, ast.Try)
                    and isinstance(node.body[0], ast.ImportFrom)
                    and node.body[0].module == "scm.params.optimizers.base")
    wrapper = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "GlompoParamsWrapper")
    original = builtins.__import__

    class NativeBase:
        pass

    class NativeModule:
        BaseOptimizer = NativeBase
        MinimizeResult = dict

    def import_module(name, *args, **kwargs):
        if name == "scm.params.optimizers.base":
            if missing:
                raise ModuleNotFoundError("missing dependency", name=missing)
            return NativeModule
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    future = ast.parse("from __future__ import annotations").body[0]
    namespace = {}
    exec(compile(ast.Module(body=[future, optional, wrapper], type_ignores=[]), str(source), "exec"), namespace)
    return namespace, NativeBase


def test_absent_optimizer_api_disables_only_wrapper(monkeypatch):
    namespace, _ = wrapper_namespace(monkeypatch, "scm.params.optimizers")
    assert namespace["BaseOptimizer"] is object
    with pytest.raises(ImportError, match="direct ReaxFFError"):
        namespace["GlompoParamsWrapper"](None)


def test_existing_api_keeps_native_base(monkeypatch):
    namespace, native = wrapper_namespace(monkeypatch)
    assert issubclass(namespace["GlompoParamsWrapper"], native)


def test_unrelated_import_failure_is_not_hidden(monkeypatch):
    with pytest.raises(ModuleNotFoundError):
        wrapper_namespace(monkeypatch, "unrelated_dependency")
