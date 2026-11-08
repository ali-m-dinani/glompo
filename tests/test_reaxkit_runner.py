import json

import pytest

from reaxkit_runner import load_request, convert_inputs
from glompo.reaxkit_io import export_native


def request_file(tmp_path):
    inputs = {}
    for name in ("ffield", "params", "geo", "trainset", "control"):
        path = tmp_path / name
        path.write_text("2 1 1 0.1 1.7 2.6\n" if name == "params" else "fixture\n")
        inputs[name] = str(path)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(dict(
        contract="reaxkit-external-runner-v1", optimizer="glompo", seed=None,
        params_bounds_order="min_max", inputs=inputs,
        settings=dict(max_function_calls=12, max_seconds=30, population_size=4,
                      max_iterations=3, sigma0=0.1))))
    return path


def test_native_bounds_order(tmp_path):
    request = request_file(tmp_path)
    _, rows, bounds = load_request(request)
    assert list(map(float, rows[0].split()[-2:])) == [1.7, 2.6]
    assert bounds == [(1.7, 2.6)]


def test_max_min_input_becomes_scm_min_max(tmp_path):
    path = request_file(tmp_path)
    data = json.loads(path.read_text())
    data["params_bounds_order"] = "max_min"
    path.write_text(json.dumps(data))
    (tmp_path / "params").write_text("2 1 1 0.1 2.6 1.7\n")
    _, rows, bounds = load_request(path)
    assert list(map(float, rows[0].split()[-2:])) == [1.7, 2.6]
    assert bounds == [(1.7, 2.6)]


@pytest.mark.parametrize("converted_bounds", [[(1.7, 2.6)], [(1.0, 2.0)]])
def test_runner_conversion_uses_copies_and_checks_bounds(tmp_path, converted_bounds):
    from types import SimpleNamespace
    request, rows, bounds = load_request(request_file(tmp_path))
    original = (tmp_path / "params").read_bytes()

    class Objective:
        @staticmethod
        def from_classic_files(directory, **kwargs):
            assert kwargs == dict(strict_inputs=True, scm_conversion=True)
            assert (directory / "params").read_text() == "".join(rows)
            assert (directory / "ffield").read_text() == "fixture\n"
            return SimpleNamespace(n_parms=1, par_eng=SimpleNamespace(
                active=SimpleNamespace(range=converted_bounds)))

    if converted_bounds == bounds:
        assert convert_inputs(request, rows, bounds, tmp_path / "copy", Objective).n_parms == 1
    else:
        with pytest.raises(ValueError, match="count/bounds differ"):
            convert_inputs(request, rows, bounds, tmp_path / "copy", Objective)
    assert (tmp_path / "params").read_bytes() == original


def test_seed_and_missing_control_rejected(tmp_path):
    path = request_file(tmp_path)
    data = json.loads(path.read_text())
    data["seed"] = 42
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="seed"):
        load_request(path)
    data["seed"] = None
    path.write_text(json.dumps(data))
    (tmp_path / "control").unlink()
    with pytest.raises(ValueError, match="control"):
        load_request(path)


def test_hdf5_export(tmp_path):
    import tables
    import numpy as np
    with tables.open_file(str(tmp_path / "glompo_log.h5"), "w") as archive:
        group = archive.create_group("/", "optimizer_1")
        archive.create_table(group, "iter_hist", obj=np.array(
            [(1, [0.5, 0.2], 3.0), (2, [0.1, 0.2], float("nan"))],
            dtype=[("call_id", "i8"), ("x", "f8", (2,)), ("fx", "f8")]))
    assert export_native(tmp_path) == 2
    records = [json.loads(line) for line in (tmp_path / "glompo_history.jsonl").read_text().splitlines()]
    assert records[1]["score"] is None
    assert records[1]["raw_score"] == "nan"
    assert records[0]["parameters"] == [0.5, 0.2]


def test_runner_creates_profile_before_real_manager_starts(tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    import reaxkit_runner

    path = request_file(tmp_path)
    request = json.loads(path.read_text())
    root = tmp_path / "native"
    request["native_directory"] = str(root)
    path.write_text(json.dumps(request))
    monkeypatch.setenv("SLURM_NTASKS", "1")

    class Collection:
        def store(self, destination):
            from pathlib import Path
            Path(destination).write_text("synthetic fixture\n")

    class Objective:
        bounds = [(0.0, 1.0)] * 3
        dat_set = Collection()
        job_col = Collection()
        par_eng = SimpleNamespace(active=SimpleNamespace(x=[0.4, 1.0, 0.984375]),
                                  yaml_store=Collection().store, write=Collection().store)

        def __call__(self, values):
            profile = json.loads((root / "manager/glompo-profile.json").read_text())
            assert profile["profile"] == "ams_params"
            return sum(float(value) ** 2 for value in values)

        def convert_parms_real2scaled(self, values):
            return values

        def convert_parms_scaled2real(self, values):
            return values

        def set_parameters(self, values, space):
            assert space == "scaled"

        def save(self, directory):
            (directory / "ffield").write_text("synthetic best-field fixture\n")

    def convert(request, rows, bounds, directory, objective_class):
        directory.mkdir()
        return Objective()

    fake_params = ModuleType("glompo.interfaces.params")
    fake_params.ReaxFFError = Objective
    fake_params.ParallelLevels = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setattr(reaxkit_runner, "engine_environment", lambda: None)
    monkeypatch.setitem(sys.modules, "glompo.interfaces.params", fake_params)
    monkeypatch.setattr(reaxkit_runner, "convert_inputs", convert)
    assert not root.exists()
    assert reaxkit_runner.main([str(path)]) == 0
    directory = root / "manager"
    assert (directory / "glompo_log.h5").is_file()
    metadata = json.loads((directory / "glompo_history.meta.json").read_text())
    assert metadata["count"] > 0
    first = json.loads((directory / "glompo_history.jsonl").read_text().splitlines()[0])
    assert first["parameters"] == [0.4, 1.0, 0.984375]
    assert first["score"] == sum(value ** 2 for value in first["parameters"])
    saved_profile = json.loads((directory / "glompo-profile.json").read_text())
    assert saved_profile["initial_candidate_policy"] == "exact_first_evaluation"
    assert saved_profile["initial_parameters_scaled"] == first["parameters"]
    assert (directory / "best/ffield").is_file()
    profile = (directory / "glompo-profile.json").read_bytes()
    with pytest.raises(ValueError, match="already prepared/started"):
        reaxkit_runner.main([str(path)])
    assert (directory / "glompo-profile.json").read_bytes() == profile
