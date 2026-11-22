"""Checks of lcadep.fit: run with python -m pytest from the repository root."""
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lcadep as ld  # noqa: E402
from lcadep import fit  # noqa: E402

LAM = np.array([0.9, 0.5, 0.2, 0.6])
SIGN = np.array([1, 1, -1, 1])


def simulate(n, lam=LAM, sign=SIGN, seed=0):
    rng = np.random.default_rng(seed)
    eta = rng.standard_normal((n, 1))
    return sign * np.sqrt(lam) * eta + np.sqrt(1 - lam) * rng.standard_normal((n, len(lam)))


def test_normal_scores():
    z = fit.normal_scores([3.0, 1.0, 2.0, 2.0])
    inv = fit.NormalDist().inv_cdf
    assert np.allclose(z, [inv(3.5 / 4), inv(0.5 / 4), inv(2.0 / 4), inv(2.0 / 4)])
    x = np.random.default_rng(1).lognormal(size=500)
    assert np.allclose(fit.normal_scores(x), fit.normal_scores(np.log(x)))  # rank-based


def test_one_factor_exact():
    load = SIGN * np.sqrt(LAM)
    R = np.outer(load, load)
    np.fill_diagonal(R, 1.0)
    assert np.allclose(fit.one_factor(R), load, atol=1e-5)
    with pytest.raises(ValueError):
        fit.one_factor(np.eye(2))


def test_recovers_lambda_and_sign():
    z = simulate(20_000)
    x = np.exp(0.3 * z)                       # any monotone transformation keeps the scores
    res = fit.fit_lambda(x, np.arange(len(x)), list("abcd"), n_boot=0)
    assert np.allclose([res.lam[k] for k in "abcd"], LAM, atol=0.03)
    assert [res.sign[k] for k in "abcd"] == SIGN.tolist()
    assert res.lam_between is None and res.lam_within is None   # one row per facility
    assert res.max_residual() < 0.03
    # the fitted values feed the Tier 2 sampler
    R = ld.tier2_correlation([res.lam[k] for k in "abcd"], [res.sign[k] for k in "abcd"])
    assert abs(R[0, 2] + math.sqrt(LAM[0] * LAM[2])) < 0.03


def test_between_within_and_interval():
    rng = np.random.default_rng(3)
    lam_b, lam_w = np.array([0.8, 0.6, 0.3]), np.array([0.2, 0.7, 0.5])
    n_fac, n_year = 600, 6
    mean = 0.4 * simulate(n_fac, lam_b, np.ones(3), seed=4)
    y = np.repeat(mean, n_year, axis=0) + 0.2 * simulate(n_fac * n_year, lam_w, np.ones(3), seed=5)
    fac = np.repeat(np.arange(n_fac), n_year)
    perm = rng.permutation(len(y))            # row order must not matter
    res = fit.fit_lambda(y[perm], fac[perm], ["a", "b", "c"], log=False, n_boot=60, seed=1)
    assert np.allclose([res.lam_within[k] for k in "abc"], lam_w, atol=0.05)
    assert np.allclose([res.lam_between[k] for k in "abc"], lam_b, atol=0.12)
    assert res.facilities == n_fac and res.facility_years == n_fac * n_year
    assert all(res.ci90[k][0] <= res.lam[k] <= res.ci90[k][1] for k in "abc")
    assert all(0.7 < res.share_between[k] < 0.9 for k in "abc")   # 0.16 / (0.16 + 0.04) = 0.8


def test_dropped_rows_and_errors():
    x = np.exp(simulate(300))
    x[0, 0], x[1, 1], x[2, 2] = np.nan, -1.0, 0.0
    res = fit.fit_lambda(x, np.arange(300), list("abcd"), n_boot=0)
    assert res.dropped_rows == 3 and res.facility_years == 297
    with pytest.raises(ValueError):
        fit.fit_lambda(x[:, :2], np.arange(300), ["a", "b"], n_boot=0)
    x[:, 3] = 2.0
    with pytest.raises(ValueError, match="no variation in: d"):
        fit.fit_lambda(x, np.arange(300), list("abcd"), n_boot=0)


def test_holdout_beats_independence():
    x = np.exp(simulate(2000))
    h = fit.holdout(x, np.repeat(np.arange(400), 5), splits=20, seed=2)
    assert h["mae_rho_default_median"] < 0.06 < h["mae_rho_independence_median"]
    assert h["rel_error_var_default_median"] < h["rel_error_var_independence_median"]


def test_rows_match_default_table_and_cli(tmp_path, capsys):
    with open(ld.DEFAULTS_CSV, newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == fit.COLUMNS
    data = tmp_path / "data.csv"
    x = np.exp(simulate(500))
    with open(data, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["farm", "year", "a", "b", "c", "d"])
        for k, row in enumerate(x):
            w.writerow([f"farm{k // 5}", 2019 + k % 5, *row])
    out = tmp_path / "rows.csv"
    fit.main([str(data), "--facility", "farm", "--year", "year", "--exchanges", "a", "b", "c", "d",
              "--technology", "Test farm", "--boot", "20", "--splits", "10", "--out", str(out)])
    assert "hold-out" in capsys.readouterr().out
    assert len(ld.load_defaults(out)) == 12                       # 3 value sets x 4 exchanges
    lam = ld.default_lambda("Test farm", "pooled", path=out)
    assert set(lam) == set("abcd") and lam["c"][1] == -1


def test_edge_cases():
    with pytest.raises(ValueError):
        fit.normal_scores([1.0, np.nan, 2.0])
    x = np.exp(simulate(60)[:, :3])
    # labels keep their type; missing labels are refused
    res = fit.fit_lambda(x[:6], [1, "1", 2, "2", 3, "3"], list("abc"), n_boot=0)
    assert res.facilities == 6
    with pytest.raises(ValueError, match="facility label"):
        fit.fit_lambda(x[:6], [1, 2, 3, float("nan"), 5, 6], list("abc"), n_boot=0)
    # a bootstrap over three facilities redraws replicates without variation
    res = fit.fit_lambda(x[:3], range(3), list("abc"), n_boot=20, seed=1)
    assert all(0 <= lo <= hi <= 1 for lo, hi in res.ci90.values())
    # no within-facility variation: the within fit is left out
    res = fit.fit_lambda(np.repeat(x[:20], 3, axis=0), np.repeat(np.arange(20), 3), list("abc"), n_boot=0)
    assert res.lam_within is None and res.lam_between is not None
    with pytest.warns(RuntimeWarning, match="not converged"):
        load = np.array([0.99, 0.1, 0.1])
        R = np.outer(load, load)
        np.fill_diagonal(R, 1.0)
        fit.one_factor(R)


def test_cli_blank_years(tmp_path):
    data, out = tmp_path / "d.csv", tmp_path / "rows.csv"
    with open(data, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["farm", "year", "a", "b", "c"])
        for k, row in enumerate(np.exp(simulate(90)[:, :3])):
            w.writerow([f"f{k}", "", *row])
    fit.main([str(data), "--facility", "farm", "--year", "year", "--exchanges", "a", "b", "c",
              "--boot", "0", "--splits", "0", "--out", str(out)])
    assert {r["data_years"] for r in ld.load_defaults(out)} == {""}
