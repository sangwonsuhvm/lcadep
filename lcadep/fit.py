"""Estimate default dependence parameters (lambda and its sign) from facility-level data.

The steps are those used for the default table of the article (US power plants and oil and gas
production), so that new data, such as farm-level data, can go through the same steps:

  1. Take one column per exchange, each per unit of output, with one row per facility-year.
  2. Take logarithms and convert each column to normal scores (a Gaussian copula).
  3. Fit a one-factor model to the correlation matrix of the scores by minimum residuals
     (MINRES). lambda_i is the squared loading and delta_i the sign of the loading.
  4. Repeat the fit for facility means ('between', for datasets that represent a technology
     average) and for deviations from facility means ('within', for a time-explicit dataset of
     one facility). 'pooled' uses all facility-years.
  5. Get 90% intervals of the pooled lambda from a bootstrap over facilities.
  6. Test the defaults on facilities not used for fitting, against independence (`holdout`).

Functions
  fit_lambda   steps 2-5 for one technology; returns a FitResult
  holdout      step 6
  table_rows   rows in the layout of default_lambda.csv
  write_rows   write such rows to a CSV file
  read_table   read a CSV file with one row per facility-year

Command line (see README.md, 'Fitting lambda from your own data'):
  python -m lcadep.fit data.csv --facility farm_id --year year --exchanges feed n2o ch4 \\
      --sector "Agriculture" --technology "Dairy farm" --region NL --source "..." --out rows.csv

Only numpy is required.
"""
from __future__ import annotations

import argparse
import csv
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

import numpy as np

COLUMNS = ["sector", "technology", "region", "data_years", "value_set", "exchange",
           "exchange_description", "lambda", "ci90_low", "ci90_high", "sign", "facilities",
           "facility_years", "source", "note"]
BOUND = 0.999  # loadings are bounded to |l| <= 0.999, so lambda <= 0.998
_INV = np.frompyfunc(NormalDist().inv_cdf, 1, 1)


# ---------------------------------------------------------------- building blocks
def normal_scores(x) -> np.ndarray:
    """Normal scores of a vector: the standard-normal quantile of (rank - 0.5) / n, ties averaged."""
    x = np.asarray(x, float)
    if not np.isfinite(x).all():
        raise ValueError("normal scores need finite values")
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    start = np.r_[True, xs[1:] != xs[:-1]]
    group = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    last = np.r_[first[1:], n] - 1
    rank = np.empty(n)
    rank[order] = ((first + last) / 2 + 1)[group]
    return _INV((rank - 0.5) / n).astype(float)


def score_correlation(x) -> np.ndarray:
    """Correlation matrix of the normal scores of the columns of x."""
    x = np.asarray(x, float)
    return np.corrcoef(np.column_stack([normal_scores(c) for c in x.T]), rowvar=False)


def one_factor(R, iters: int = 500) -> np.ndarray:
    """MINRES loadings of a one-factor model for a correlation matrix R.

    The communalities are iterated on the diagonal (principal-axis iteration) until they change
    by less than 1e-8; a warning is issued when `iters` iterations do not reach that. Loadings
    are bounded to |l| <= 0.999 and oriented so that their sum is not negative. When a loading
    is at the bound (a Heywood case), the other loadings are not re-optimized; treat such a fit
    with care. A one-factor model needs at least three variables to be identified.
    """
    R = np.asarray(R, float)
    p = R.shape[0]
    if p < 3:
        raise ValueError("a one-factor model needs at least three exchanges; with two, only the "
                         "product lambda_1 lambda_2 = rho^2 is identified")
    h = np.clip(np.max(np.abs(R - np.eye(p)), axis=1), 0.05, 0.99)
    for _ in range(iters):
        Rr = R.copy()
        np.fill_diagonal(Rr, h)
        w, v = np.linalg.eigh(Rr)
        load = np.clip(v[:, -1] * np.sqrt(max(w[-1], 1e-9)), -BOUND, BOUND)
        h_new = load ** 2
        if np.max(np.abs(h_new - h)) < 1e-8:
            break
        h = h_new
    else:
        warnings.warn(f"one-factor fit not converged after {iters} iterations", RuntimeWarning, stacklevel=2)
    return -load if load.sum() < 0 else load


def _offdiag(M):
    i, j = np.triu_indices(M.shape[0], 1)
    return M[i, j]


def _groups(facility):
    """Facility labels in order of first appearance, and the row indices of each."""
    idx: dict = {}
    for k, f in enumerate(facility):
        idx.setdefault(f, []).append(k)
    return list(idx), [np.array(v) for v in idx.values()]


def _rng(seed):
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def complete_rows(x, log: bool = True) -> np.ndarray:
    """Boolean mask of the rows that enter the fit: all values finite and, with log, positive."""
    x = np.asarray(x, float)
    if x.ndim != 2:
        raise ValueError("x must have one row per facility-year and one column per exchange")
    ok = np.isfinite(x).all(axis=1)
    if log:
        ok &= (np.where(np.isfinite(x), x, 1.0) > 0).all(axis=1)
    return ok


def _clean(x, facility, log):
    ok = complete_rows(x, log)
    x = np.asarray(x, float)
    facility = list(facility)
    if len(facility) != len(x):
        raise ValueError("facility must have one label per row of x")
    if any(f is None or f == "" or f != f for f in facility):
        raise ValueError("every row needs a facility label")
    fac = np.empty(len(facility), dtype=object)   # labels keep their type: 1 and "1" differ
    fac[:] = facility
    y = np.log(x[ok]) if log else x[ok]
    return y, fac[ok], int((~ok).sum())


def _varies(a, ref=None) -> bool:
    """True when every column varies by more than rounding error relative to `ref`."""
    ref = np.abs(a).max(axis=0) if ref is None else ref
    return bool((np.ptp(a, axis=0) > 1e-10 * ref).all())


# ---------------------------------------------------------------- the fit
@dataclass
class FitResult:
    """Result of fit_lambda. Dictionaries are keyed by exchange name; pairs by 'a-b'."""
    names: list
    lam: dict                 # pooled lambda
    sign: dict                # sign of the pooled loading
    ci90: dict                # (low, high) of the pooled lambda, bootstrap over facilities
    lam_between: dict | None  # None when there are too few facilities
    sign_between: dict | None
    lam_within: dict | None   # None when too few facilities have enough years
    sign_within: dict | None
    observed_rho: dict        # correlations of the normal scores, pooled
    implied_rho: dict         # delta_i delta_j sqrt(lambda_i lambda_j), pooled
    share_between: dict       # share of the log variance that lies between facilities
    sd_log: dict              # standard deviation of the logs, pooled
    facilities: int
    facility_years: int
    facilities_within: int    # facilities used for the within fit
    facility_years_within: int
    dropped_rows: int         # rows left out for missing or non-positive values
    at_bound: list = field(default_factory=list)  # exchanges whose pooled loading is at the bound

    def max_residual(self) -> float:
        """Largest absolute difference between an observed and an implied pooled correlation."""
        return max(abs(self.observed_rho[k] - self.implied_rho[k]) for k in self.observed_rho)

    def summary(self) -> str:
        w = max(len(n) for n in self.names)
        out = [f"{self.facilities} facilities, {self.facility_years} facility-years"
               + (f" ({self.dropped_rows} rows dropped)" if self.dropped_rows else ""),
               f"{'exchange':<{w}}  pooled lambda [90% interval] sign   between  within"]
        for n in self.names:
            lo, hi = self.ci90[n]
            b = "   n/a " if self.lam_between is None else f"{self.lam_between[n]:7.3f}"
            wi = "   n/a " if self.lam_within is None else f"{self.lam_within[n]:7.3f}"
            ci = "[  n/a       ]" if np.isnan(lo) else f"[{lo:.3f}, {hi:.3f}]"
            out.append(f"{n:<{w}}  {self.lam[n]:13.3f} {ci}  {self.sign[n]:+d}   {b} {wi}")
        out.append(f"largest residual correlation (pooled): {self.max_residual():.3f}")
        if self.at_bound:
            out.append("loading at the bound for: " + ", ".join(self.at_bound))
        return "\n".join(out)


def fit_lambda(x, facility, names, log: bool = True, n_boot: int = 300, seed=0,
               min_facilities: int = 20, min_years_within: int = 3) -> FitResult:
    """Fit lambda and its sign for the exchanges of one technology.

    x          array, one row per facility-year and one column per exchange, each per unit of
               output (per MWh, per kg of milk). Values must be positive; rows with a missing or
               non-positive value are left out and counted in `dropped_rows`.
    facility   facility label of each row. With one row per facility, only the pooled fit is made
               (it then equals a between-facility fit).
    names      exchange names, one per column.
    log        False when x already holds logarithms.
    n_boot     bootstrap replicates over facilities for the 90% interval (0 for none).
    seed       integer or numpy Generator.
    min_facilities     smallest number of facilities for the between fit.
    min_years_within   a facility enters the within fit when it has at least this many rows
                       (one row per year is assumed).

    Pooled and between fits use normal scores; the within fit uses the correlation of the log
    deviations from facility means, without normal scores, as in the article. A between or
    within fit is left out (None) when its data have no variation in some exchange.
    """
    names = list(names)
    y, fac, dropped = _clean(x, facility, log)
    if y.shape[1] != len(names):
        raise ValueError("names must have one entry per column of x")
    labels, rows = _groups(fac)
    if len(labels) < 3:
        raise ValueError("too few facilities with complete data")
    constant = [n for n, c in zip(names, y.T) if np.ptp(c) == 0]
    if constant:
        raise ValueError(f"no variation in: {', '.join(constant)}; leave these exchanges out")
    rng = _rng(seed)
    p = len(names)
    R = score_correlation(y)
    load = one_factor(R)

    boot = []
    tries = 0
    while len(boot) < n_boot:
        yb = y[np.concatenate([rows[k] for k in rng.choice(len(labels), len(labels), replace=True)])]
        if _varies(yb):
            boot.append(one_factor(score_correlation(yb)) ** 2)
        elif (tries := tries + 1) > 10 * n_boot:   # replicates without variation are redrawn
            raise ValueError("too few facilities for a bootstrap; use n_boot=0")
    boot = np.array(boot)

    means = np.array([y[r].mean(axis=0) for r in rows])
    dev = y.copy()
    for r, m in zip(rows, means):
        dev[r] = y[r] - m
    repeated = any(len(r) > 1 for r in rows)
    lb = None
    if repeated and len(labels) >= min_facilities and _varies(means):
        lb = one_factor(score_correlation(means))
    multi = [r for r in rows if len(r) >= min_years_within]
    lw = None
    if len(multi) >= min_facilities and _varies(dev[np.concatenate(multi)], y.std(axis=0)):
        lw = one_factor(np.corrcoef(dev[np.concatenate(multi)], rowvar=False))

    def d(v, cast=float):
        return dict(zip(names, map(cast, v)))

    def sgn(v):
        return d(np.where(v < 0, -1, 1), int)

    iu = list(zip(*np.triu_indices(p, 1)))
    var_all = y.var(axis=0, ddof=1)
    return FitResult(
        names=names, lam=d(load ** 2), sign=sgn(load),
        ci90={n: ((float(np.percentile(boot[:, k], 5)), float(np.percentile(boot[:, k], 95)))
                  if n_boot else (float("nan"), float("nan"))) for k, n in enumerate(names)},
        lam_between=None if lb is None else d(lb ** 2), sign_between=None if lb is None else sgn(lb),
        lam_within=None if lw is None else d(lw ** 2), sign_within=None if lw is None else sgn(lw),
        observed_rho={f"{names[i]}-{names[j]}": float(R[i, j]) for i, j in iu},
        implied_rho={f"{names[i]}-{names[j]}": float(load[i] * load[j]) for i, j in iu},
        share_between=d(1 - dev.var(axis=0, ddof=1) / var_all), sd_log=d(np.sqrt(var_all)),
        facilities=len(labels), facility_years=len(y),
        facilities_within=len(multi), facility_years_within=int(sum(len(r) for r in multi)),
        dropped_rows=dropped, at_bound=[n for n, v in zip(names, load) if abs(v) >= BOUND - 1e-12])


# ---------------------------------------------------------------- hold-out test
def _transfer(y_fit, y_test):
    load = one_factor(score_correlation(y_fit))
    R = score_correlation(y_test)
    pred, obs = _offdiag(np.outer(load, load)), _offdiag(R)
    return (np.mean(np.abs(pred - obs)), np.mean(np.abs(obs)),
            np.mean(np.abs((2 + 2 * pred) / (2 + 2 * obs) - 1)), np.mean(np.abs(2 / (2 + 2 * obs) - 1)))


def holdout(x, facility, log: bool = True, splits: int = 200, seed=0) -> dict:
    """Test of the defaults on facilities not used for fitting.

    The facilities are split at random into two halves `splits` times. lambda is fitted on one
    half and the implied correlations are compared with those observed in the other half.
    Returns the median absolute error in the pairwise correlations with the fitted defaults and
    under independence (and the 90th percentile with the defaults), and the median relative
    error in the variance of the sum of two standardized exchanges.
    """
    y, fac, _ = _clean(x, facility, log)
    labels, rows = _groups(fac)
    if len(labels) < 20:
        raise ValueError("the hold-out test needs at least 20 facilities")
    rng = _rng(seed)
    res = []
    for _ in range(splits):
        a = set(rng.choice(len(labels), len(labels) // 2, replace=False).tolist())
        fit = np.concatenate([rows[k] for k in range(len(labels)) if k in a])
        test = np.concatenate([rows[k] for k in range(len(labels)) if k not in a])
        res.append(_transfer(y[fit], y[test]))
    res = np.array(res)
    return {"splits": splits,
            "mae_rho_default_median": float(np.median(res[:, 0])),
            "mae_rho_independence_median": float(np.median(res[:, 1])),
            "mae_rho_default_p90": float(np.percentile(res[:, 0], 90)),
            "rel_error_var_default_median": float(np.median(res[:, 2])),
            "rel_error_var_independence_median": float(np.median(res[:, 3]))}


# ---------------------------------------------------------------- default-table rows and files
def table_rows(result: FitResult, sector: str, technology: str, region: str, data_years: str,
               source: str, descriptions: dict | None = None, note: str = "") -> list[dict]:
    """Rows in the layout of default_lambda.csv: pooled (with intervals), between and within."""
    descriptions = descriptions or {}
    sets = [("pooled", result.lam, result.sign, result.facilities, result.facility_years),
            ("between", result.lam_between, result.sign_between, result.facilities, result.facility_years),
            ("within", result.lam_within, result.sign_within, result.facilities_within,
             result.facility_years_within)]
    out = []
    for value_set, lam, sign, n_fac, n_rows in sets:
        if lam is None:
            continue
        for n in result.names:
            lo, hi = result.ci90[n] if value_set == "pooled" else (float("nan"), float("nan"))
            out.append({
                "sector": sector, "technology": technology, "region": region, "data_years": data_years,
                "value_set": value_set, "exchange": n, "exchange_description": descriptions.get(n, ""),
                "lambda": f"{lam[n]:.3f}", "ci90_low": "" if np.isnan(lo) else f"{lo:.3f}",
                "ci90_high": "" if np.isnan(hi) else f"{hi:.3f}", "sign": sign[n],
                "facilities": n_fac, "facility_years": n_rows, "source": source, "note": note})
    return out


def write_rows(rows: list[dict], path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def read_table(path, facility: str, exchanges: list, year: str | None = None):
    """Read a CSV file with one row per facility-year.

    Returns x (float array; empty or non-numeric cells become NaN), the facility labels and the
    years (None when no year column is given).
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    missing = [c for c in [facility, *exchanges, *([year] if year else [])] if c not in rows[0]]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    def num(s):
        try:
            return float(s)
        except (TypeError, ValueError):
            return float("nan")

    x = np.array([[num(r[c]) for c in exchanges] for r in rows])
    return x, np.array([r[facility] for r in rows]), ([r[year] for r in rows] if year else None)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m lcadep.fit",
        description="Fit lambda and its sign from a CSV file with one row per facility-year.")
    ap.add_argument("data", type=Path, help="CSV file; exchange columns per unit of output")
    ap.add_argument("--facility", required=True, help="column with the facility identifier")
    ap.add_argument("--exchanges", required=True, nargs="+", help="exchange columns (three or more)")
    ap.add_argument("--year", help="column with the data year (used for the data_years field)")
    ap.add_argument("--sector", default="")
    ap.add_argument("--technology", default="")
    ap.add_argument("--region", default="")
    ap.add_argument("--source", default="")
    ap.add_argument("--boot", type=int, default=300, help="bootstrap replicates (default 300)")
    ap.add_argument("--splits", type=int, default=200, help="hold-out splits (default 200; 0 for none)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="write rows in the layout of default_lambda.csv")
    a = ap.parse_args(argv)
    x, fac, years = read_table(a.data, a.facility, a.exchanges, a.year)
    rng = np.random.default_rng(a.seed)
    res = fit_lambda(x, fac, a.exchanges, n_boot=a.boot, seed=rng)
    print(res.summary())
    if a.splits and res.facilities >= 20:
        h = holdout(x, fac, splits=a.splits, seed=rng)
        print(f"hold-out, median absolute error in correlations: {h['mae_rho_default_median']:.3f} with "
              f"the fitted defaults, {h['mae_rho_independence_median']:.3f} under independence")
    if a.out:
        span = ""
        if years:
            ys = sorted({y for y, keep in zip(years, complete_rows(x)) if keep and y})
            span = "" if not ys else ys[0] if len(ys) == 1 else f"{ys[0]}-{ys[-1]}"
        write_rows(table_rows(res, a.sector, a.technology, a.region, span, a.source), a.out)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
