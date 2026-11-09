from types import SimpleNamespace

import pytest

from reaxkit_runner import convert_inputs


@pytest.mark.parametrize("separator", ["", "\n", "  \n\n"])
def test_conversion_removes_only_interstructure_blank_records(tmp_path, separator):
    first = "XTLGRF 200\nDESCRP first\n\nHETATM fixed columns unchanged\nEND\n"
    second = "XTLGRF 200\nDESCRP second\nHETATM other fixed columns\nEND\n"
    source = tmp_path / "geo"
    source.write_text(separator + first + separator + second + separator)
    original = source.read_bytes()
    inputs = {"geo": str(source)}
    for name in ("ffield", "control", "trainset"):
        path = tmp_path / name
        path.write_text("unchanged\n")
        inputs[name] = str(path)
    bounds = [(1.7, 2.6)]

    class Objective:
        @staticmethod
        def from_classic_files(directory, **kwargs):
            assert (directory / "geo").read_text() == first + second
            assert kwargs == dict(strict_inputs=True, scm_conversion=True)
            return SimpleNamespace(n_parms=1, par_eng=SimpleNamespace(
                active=SimpleNamespace(range=bounds)))

    convert_inputs({"inputs": inputs}, ["2 1 1 0.1 1.7 2.6\n"], bounds,
                   tmp_path / "copy", Objective)
    assert source.read_bytes() == original
