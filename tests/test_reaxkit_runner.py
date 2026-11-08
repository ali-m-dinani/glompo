import json

import pytest

from reaxkit_runner import load_request
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
    assert list(map(float, rows[0].split()[-2:])) == [2.6, 1.7]
    assert bounds == [(1.7, 2.6)]


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
