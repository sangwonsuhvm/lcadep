"""Checks of lcadep: run with python -m pytest (or python tests/test_lcadep.py) from the repository root."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lcadep as ld  # noqa: E402


def test_correlation_valid_and_values():
    rng = np.random.default_rng(0)
    for _ in range(200):
        lam = rng.uniform(0, 1, 6)
        sign = rng.choice([-1, 1], 6)
        R = ld.tier2_correlation(lam, sign)
        assert np.linalg.eigvalsh(R).min() > -1e-12
        assert math.isclose(R[0, 1], sign[0] * sign[1] * math.sqrt(lam[0] * lam[1]))


def test_marginals_unchanged_and_correlation():
    dr = ld.Draws(400_000, seed=3)
    z = ld.tier2_normals(dr, "p", {"a": 0.9, "b": 0.4}, {"a": 1, "b": -1})
    assert abs(z["a"].std() - 1) < 0.01 and abs(z["b"].std() - 1) < 0.01
    assert abs(np.corrcoef(z["a"], z["b"])[0, 1] + math.sqrt(0.36)) < 0.01
    x = ld.sample_exchange(2.0, 0.3, z["a"], 0.2, dr.normal(("p", "a", "pedigree")))
    assert abs(np.log(x).std() - math.sqrt(0.3 ** 2 + 0.2 ** 2)) < 0.01
    assert abs(np.median(x) - 2.0) < 0.01


def test_sample_once():
    dr = ld.Draws(1000, seed=5)
    a = dr.normal(("gas supply", "L"))
    dr.normal("something else")
    assert np.array_equal(a, dr.normal(("gas supply", "L")))
    assert np.array_equal(a, ld.Draws(1000, seed=5).normal(("gas supply", "L")))  # order-independent
    assert abs(np.corrcoef(a, dr.normal(("gas supply", "e_u")))[0, 1]) < 0.1


def test_dirichlet():
    s = ld.dirichlet_shares([0.6, 0.3, 0.1], 50_000, seed=1, cv_largest=0.1)
    assert np.allclose(s.sum(axis=1), 1)
    assert abs(s[:, 0].std() / s[:, 0].mean() - 0.1) < 0.005


def test_screen_misses_no_pair():
    rng = np.random.default_rng(2)
    for _ in range(200):
        n = 12
        a, sig = rng.lognormal(0, 1.5, n), rng.uniform(0.05, 1, n)
        names = [f"x{i}" for i in range(n)]
        _, pairs = ld.screen_pairs(a, sig, names, threshold=0.05)
        v = (a * sig) ** 2
        brute = {frozenset((names[i], names[j])) for i in range(n) for j in range(i + 1, n)
                 if 2 * math.sqrt(v[i] * v[j]) / v.sum() >= 0.05}
        assert brute == {frozenset((p.i, p.j)) for p in pairs}


def test_first_order_sd():
    a, s = np.array([1.0, 2.0]), np.array([0.5, 0.5])
    assert math.isclose(ld.first_order_sd(a, s), math.sqrt(0.25 + 1.0))
    assert math.isclose(ld.first_order_sd(a, s, [[1, 1], [1, 1]]), 1.5)


def test_default_table():
    lam = ld.default_lambda("Natural gas, combined cycle", "within")
    assert set(lam) == {"heat_input", "co2", "so2", "nox"}
    assert all(0 <= v[0] <= 1 and v[1] in (-1, 1) for v in lam.values())


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
