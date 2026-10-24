"""The MIP checked against exhaustive enumeration.

For small m, every decision (subset, threshold) can be enumerated. The model
optimum must coincide with the enumerated optimum on every instance. It was
this test that once exposed a flipped sign in an activation constraint.
"""
import numpy as np
import pytest
from opera.model import Instance, solve, brute_force


def _aleatoria(rng, n=14, m=5):
    M = (rng.random((n, m)) < 0.5).astype(int)
    y = (rng.random(n) < 0.4).astype(int)
    custos = np.array([[float(rng.integers(1, 100)), float(rng.integers(0, 10)),
                        -float(rng.integers(1, 500)), -float(rng.integers(1, 50))]])
    return Instance(M, y, custos)


@pytest.mark.parametrize("semente", range(25))
def test_otimo_bate_com_forca_bruta(semente):
    inst = _aleatoria(np.random.default_rng(semente))
    sol = solve(inst, backend="highs")
    melhor, _, _ = brute_force(inst)
    assert sol.U_star == pytest.approx(melhor, abs=1e-6)


def test_caso_esparso():
    """M quase vazio: exercita q_ik = 0 e tbar_k = max(1, qbar_k)."""
    M = np.zeros((8, 4), dtype=int); M[0, 0] = M[3, 2] = 1
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0])
    inst = Instance(M, y, np.array([[50.0, 1.0, -200.0, -5.0]]))
    sol = solve(inst, backend="highs")
    melhor, _, _ = brute_force(inst)
    assert sol.U_star == pytest.approx(melhor, abs=1e-6)


def test_utilidade_da_decisao_otima_confere():
    """U* must equal exactly the utility of (J*, t*) recomputed by hand."""
    inst = _aleatoria(np.random.default_rng(99))
    sol = solve(inst, backend="highs")
    assert sol.U_star == pytest.approx(
        inst.utility_of(sol.selected[0], sol.thresholds[0]), abs=1e-6)
