"""Fit lambda from facility-level data: an example with simulated data.

The script simulates 400 facilities observed for five years, with four exchanges per unit of
output whose dependence follows the one-factor model with known lambda. It writes the data to a
CSV file, fits lambda with lcadep.fit as one would for real data, and writes rows in the layout
of the default table. The data are simulated; they describe no real facility.

Usage: python examples/fit_simulated_facilities.py <out_dir>

For your own data, the same steps run from the command line:
    python -m lcadep.fit <out_dir>/simulated_facilities.csv --facility facility --year year \
        --exchanges feed_input ch4 n2o nh3 --out rows.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lcadep import fit  # noqa: E402

NAMES = ["feed_input", "ch4", "n2o", "nh3"]
MEDIAN = np.array([1.2, 0.03, 0.002, 0.008])            # per unit of output
SD_BETWEEN = np.array([0.15, 0.20, 0.40, 0.35])         # log-scale spread of facility means
SD_WITHIN = np.array([0.05, 0.08, 0.20, 0.15])          # log-scale spread around facility means
LAM_BETWEEN = np.array([0.80, 0.60, 0.30, 0.10])
SIGN_BETWEEN = np.array([1, 1, 1, -1])
LAM_WITHIN = np.array([0.70, 0.50, 0.10, 0.05])


def one_factor_normals(rng, n, lam, sign):
    eta = rng.standard_normal((n, 1))
    return sign * np.sqrt(lam) * eta + np.sqrt(1 - lam) * rng.standard_normal((n, len(lam)))


def simulate(facilities=400, years=range(2019, 2024), seed=1):
    rng = np.random.default_rng(seed)
    mean = SD_BETWEEN * one_factor_normals(rng, facilities, LAM_BETWEEN, SIGN_BETWEEN)
    rows = []
    for k in range(facilities):
        dev = SD_WITHIN * one_factor_normals(rng, len(years), LAM_WITHIN, np.ones(4))
        for y, d in zip(years, dev):
            rows.append([f"F{k + 1:03d}", y, *(MEDIAN * np.exp(mean[k] + d))])
    return rows


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = out / "simulated_facilities.csv"
    with open(data, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["facility", "year", *NAMES])
        w.writerows(simulate())

    x, facility, years = fit.read_table(data, "facility", NAMES, year="year")
    rng = np.random.default_rng(0)
    res = fit.fit_lambda(x, facility, NAMES, seed=rng)
    print(res.summary())
    print("\nlambda used in the simulation")
    print("  between:", dict(zip(NAMES, LAM_BETWEEN.tolist())))
    print("  within: ", dict(zip(NAMES, LAM_WITHIN.tolist())))
    h = fit.holdout(x, facility, seed=rng)
    print(f"\nhold-out, median absolute error in correlations: {h['mae_rho_default_median']:.3f} with the "
          f"fitted defaults, {h['mae_rho_independence_median']:.3f} under independence")
    rows = fit.table_rows(res, "Simulated", "Simulated facilities", "none", "2019-2023",
                          "simulated data (examples/fit_simulated_facilities.py)", note="simulated; not a default")
    fit.write_rows(rows, out / "fitted_rows.csv")
    print(f"wrote {data} and {out / 'fitted_rows.csv'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fit_example_out")
