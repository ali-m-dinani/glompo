"""Check exact first-candidate handling with the current interpreter's CMA."""

from glompo.optimizers.cmawrapper import CMAOptimizer


def main():
    initial = [0.5, 1.0, 0.984375]
    evaluations = []

    def objective(vector):
        evaluations.append(list(map(float, vector)))
        return sum((float(value) - target) ** 2 for value, target in zip(vector, initial))

    optimizer = CMAOptimizer(workers=1, popsize=4, maxiter=1, verbose=False, seed=7)
    result = optimizer.minimize(objective, initial, [(0, 1)] * 3,
                                sigma0=0.1, preserve_initial=True)
    if evaluations[0] != initial or len(evaluations) != 4 or result.fx != 0:
        raise RuntimeError("Exact initial-candidate regression failed")
    print("Exact initial-candidate check passed: four analytic evaluations; no AMS calls")


if __name__ == "__main__":
    main()
