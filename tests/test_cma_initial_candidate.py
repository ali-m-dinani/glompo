import numpy as np
import pytest

from glompo.optimizers.cmawrapper import CMAOptimizer


@pytest.mark.parametrize("initial", [[0.5, 1.0, 0.984375], [0.0, 0.5, 0.015625]])
def test_exact_first_population_member(initial):
    evaluations = []

    def objective(vector):
        evaluations.append(list(vector))
        return float(np.sum((np.asarray(vector) - initial) ** 2))

    optimizer = CMAOptimizer(workers=1, popsize=4, maxiter=1, verbose=False, seed=7)
    result = optimizer.minimize(objective, initial, [(0, 1)] * 3,
                                sigma0=0.1, preserve_initial=True)
    assert evaluations[0] == initial
    assert len(evaluations) == 4
    assert result.fx == 0
    assert list(result.x) == initial
    assert list(optimizer.incumbent["x"]) == initial
    assert optimizer.incumbent["fx"] == 0


def test_default_incumbent_uses_evaluated_not_requested_vector():
    evaluated = []

    def objective(vector):
        evaluated.append(list(vector))
        return sum(float(value) ** 2 for value in vector)

    optimizer = CMAOptimizer(workers=1, popsize=4, maxiter=1, verbose=False, seed=7)
    optimizer.minimize(objective, [0.5, 1.0, 0.984375], [(0, 1)] * 3, sigma0=0.1)
    assert list(optimizer.incumbent["x"]) == evaluated[0]
    assert optimizer.incumbent["fx"] == objective(evaluated[0])


@pytest.mark.parametrize("initial", [[0.5, 1.01, 0.9], [0.5, float("nan"), 0.9]])
def test_exact_candidate_rejects_invalid_values(initial):
    optimizer = CMAOptimizer(workers=1, popsize=4, maxiter=1, verbose=False)
    with pytest.raises(ValueError, match="Exact initial candidate"):
        optimizer.minimize(lambda vector: 0, initial, [(0, 1)] * 3,
                           sigma0=0.1, preserve_initial=True)
