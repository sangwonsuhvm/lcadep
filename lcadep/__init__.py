"""lcadep: default dependence rules for Monte Carlo uncertainty analysis in life cycle assessment.

The package implements the three tiers of the default dependence rule described in an article in
preparation (Suh and co-authors; see README.md):

  Tier 1  structural rules
          Draws            one set of standard-normal draws per key, so that a dataset, factor or
                           background process used in several places, or by several compared
                           alternatives, is sampled once per run
          dirichlet_shares shares that must sum to one
          (balances)       exchanges tied by a balance are computed from their driver in the
                           user's model; no function is needed
  Tier 2  empirical defaults from a one-factor model
          default_lambda   look up the shared-variance fraction lambda and its sign
          tier2_normals    correlated standard normals for the exchanges of one process
          sample_exchange  lognormal exchange with lambda acting on the basic uncertainty (Eq. 3)
          tier2_correlation  the implied correlation matrix, R = l l' + diag(1 - lambda)
  Tier 3  screening and bounds
          screen_pairs     first-order screen of input pairs (Eq. 4)
          first_order_sd   first-order standard deviation for a given correlation matrix

Only numpy is required. The worked example in examples/worked_case_ngcc.py applies every step.
"""
from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

__version__ = "0.1.0"
DEFAULTS_CSV = Path(__file__).with_name("default_lambda.csv")


# ---------------------------------------------------------------- Tier 1: shared draws
def _key_int(key) -> int:
    return int.from_bytes(hashlib.sha256(repr(key).encode()).digest()[:8], "little")


class Draws:
    """Standard-normal draws, one vector of length `runs` per key.

    The same key always returns the same draws, whatever the order of the calls. Keying the draws
    of a dataset, factor or background process by one name implements the Tier 1 rule 'sample
    once per run': every use of that input, including its use by compared alternatives, reads the
    same values. Distinct keys give independent draws.
    """

    def __init__(self, runs: int, seed: int = 0):
        self.runs = int(runs)
        self.seed = int(seed)
        self._store: dict = {}

    def normal(self, key) -> np.ndarray:
        if key not in self._store:
            ss = np.random.SeedSequence(self.seed, spawn_key=(_key_int(key),))
            self._store[key] = np.random.default_rng(ss).standard_normal(self.runs)
        return self._store[key]

    def keys(self):
        return list(self._store)


def lognormal(median, sigma, z):
    """Lognormal values with the given median and log-scale standard deviation from normals z."""
    return np.asarray(median, float) * np.exp(np.asarray(sigma, float) * z)


def dirichlet_shares(shares, runs: int, seed: int = 0, alpha0: float | None = None,
                     cv_largest: float | None = None) -> np.ndarray:
    """Tier 1: shares that sum to one in every run, from a Dirichlet distribution with mean `shares`.

    The concentration alpha0 is given directly or set so that the largest share has the
    coefficient of variation `cv_largest`, from Var(p_k) = p_k (1 - p_k) / (alpha0 + 1).
    Returns an array of shape (runs, len(shares)).
    """
    p = np.asarray(shares, float)
    p = p / p.sum()
    if alpha0 is None:
        if cv_largest is None:
            raise ValueError("give alpha0 or cv_largest")
        k = int(np.argmax(p))
        alpha0 = p[k] * (1 - p[k]) / (cv_largest * p[k]) ** 2 - 1
        if alpha0 <= 0:
            raise ValueError("cv_largest is too large for a Dirichlet distribution with these shares")
    return np.random.default_rng(seed).dirichlet(alpha0 * p, size=runs)


# ---------------------------------------------------------------- Tier 2: one-factor defaults
def tier2_correlation(lam, sign=None) -> np.ndarray:
    """Correlation matrix implied by shared-variance fractions lam and signs: l l' + diag(1 - lam).

    It is positive semidefinite for any lam in [0, 1], because it is a rank-one matrix plus a
    non-negative diagonal; rho_ij = s_i s_j sqrt(lam_i lam_j) off the diagonal.
    """
    lam = np.asarray(lam, float)
    if np.any((lam < 0) | (lam > 1)):
        raise ValueError("lambda must lie in [0, 1]")
    s = np.ones_like(lam) if sign is None else np.sign(np.asarray(sign, float))
    load = s * np.sqrt(lam)
    R = np.outer(load, load)
    np.fill_diagonal(R, 1.0)
    return R


def tier2_normals(draws: Draws, process: str, lam: dict, sign: dict | None = None) -> dict:
    """Correlated standard normals z_i = s_i sqrt(lam_i) eta + sqrt(1 - lam_i) xi_i for one process.

    eta, the common driver, is drawn once per process (key (process, 'common driver')); xi_i is
    each exchange's own variation (key (process, exchange)). Every z_i is standard normal, so the
    marginal distribution of each exchange is unchanged.
    """
    eta = draws.normal((process, "common driver"))
    out = {}
    for ex, val in lam.items():
        if not 0 <= val <= 1:
            raise ValueError(f"lambda of {ex} must lie in [0, 1]")
        s = 1.0 if sign is None else float(np.sign(sign.get(ex, 1.0)) or 1.0)
        out[ex] = s * math.sqrt(val) * eta + math.sqrt(1 - val) * draws.normal((process, ex))
    return out


def sample_exchange(median, sigma_basic, z_basic, sigma_pedigree: float = 0.0, z_pedigree=None):
    """Eq. 3: ln x = ln median + sigma_b z_b + sigma_p z_p.

    z_b comes from tier2_normals (or is an independent normal when the exchange has no default);
    lambda therefore acts on the basic uncertainty only, and the pedigree addition, with its own
    independent normals z_p, is not shared.
    """
    x = math.log(median) + sigma_basic * np.asarray(z_basic, float)
    if sigma_pedigree:
        if z_pedigree is None:
            raise ValueError("z_pedigree is needed when sigma_pedigree > 0")
        x = x + sigma_pedigree * np.asarray(z_pedigree, float)
    return np.exp(x)


def load_defaults(path: Path | str = DEFAULTS_CSV) -> list[dict]:
    """Rows of the default table, with numeric fields converted."""
    num = {"lambda", "ci90_low", "ci90_high", "sign", "facilities", "facility_years"}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in num:
            if r.get(k) not in (None, ""):
                r[k] = float(r[k])
    return rows


def default_lambda(technology: str, value_set: str = "between", path: Path | str = DEFAULTS_CSV) -> dict:
    """{exchange: (lambda, sign)} for one technology and value set ('between', 'within' or 'pooled').

    'between' suits a dataset that represents a technology average, 'within' a time-explicit
    dataset for one facility. Raises KeyError when the table has no such entry.
    """
    rows = [r for r in load_defaults(path) if r["technology"] == technology and r["value_set"] == value_set]
    if not rows:
        raise KeyError(f"no defaults for {technology!r}, value set {value_set!r}")
    return {r["exchange"]: (r["lambda"], r["sign"]) for r in rows}


# ---------------------------------------------------------------- Tier 3: screen and bounds
@dataclass
class Pair:
    i: str
    j: str
    weight: float
    covered: bool


def screen_pairs(a, sigma, names, threshold: float = 0.05, keep: float | None = None, covered=()):
    """First-order screen of input pairs (Eq. 4).

    a       first-order sensitivity of the result to the log of each input, dh/d ln x_i, at the
            deterministic point; for h > 0, a_i = e_i h with e_i the relative sensitivity
            coefficient. For a comparison, use the sensitivities of the difference.
    sigma   log-scale standard deviation of each input.
    keep    None (default): keep every input i with 2 sqrt(s_i s_max) >= threshold, where s_i is its
            share of the first-order variance and s_max the largest share. Because
            w_ij = 2 sqrt(s_i s_j), no pair outside the kept set can reach the threshold.
            A number instead keeps inputs, largest first, until they explain that share of the
            first-order variance; this cheaper rule can miss a pair of a large and a small input.
    covered pairs (i, j) already handled by Tier 1 or Tier 2.

    The weight w_ij = 2 |a_i a_j| sigma_i sigma_j / sum_k a_k^2 sigma_k^2 is the largest share by
    which the pair can change the first-order variance (reached at rho = +1 or -1). Returns the
    kept inputs and the pairs with w_ij >= threshold, largest first.
    """
    a = np.asarray(a, float)
    sigma = np.asarray(sigma, float)
    v = (a * sigma) ** 2
    total = v.sum()
    s = v / total
    if keep is None:
        kept = sorted(np.nonzero(2 * np.sqrt(s * s.max()) >= threshold)[0].tolist())
    else:
        order = np.argsort(-v)
        n_keep = int(np.searchsorted(np.cumsum(v[order]) / total, keep - 1e-12) + 1)
        kept = sorted(order[:n_keep].tolist())
    cov = {frozenset(p) for p in covered}
    pairs = []
    for i, j in combinations(kept, 2):
        w = 2 * abs(a[i] * a[j]) * sigma[i] * sigma[j] / total
        if w >= threshold:
            pairs.append(Pair(names[i], names[j], float(w), frozenset((names[i], names[j])) in cov))
    return [names[k] for k in kept], sorted(pairs, key=lambda p: -p.weight)


def first_order_sd(a, sigma, rho=None) -> float:
    """First-order standard deviation of the result, sqrt(b' R b) with b_i = a_i sigma_i."""
    b = np.asarray(a, float) * np.asarray(sigma, float)
    R = np.eye(len(b)) if rho is None else np.asarray(rho, float)
    return float(math.sqrt(b @ R @ b))
