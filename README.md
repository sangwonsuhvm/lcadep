# lcadep: default dependence rules for LCA Monte Carlo

`lcadep` applies a three-tier default rule for dependence between uncertain inputs in Monte Carlo
uncertainty analysis for life cycle assessment (LCA) and other industrial ecology models. It is a
small Python package that needs only numpy. Its default values are in `lcadep/default_lambda.csv`,
and `lcadep.fit` estimates such values from facility-level data.

The rule and the values are described in an article in preparation by Sangwon Suh and co-authors
(working title *Default rules for dependence between uncertain flows in industrial ecology:
empirical defaults and an application to life cycle assessment*).

Status: version 0.2.0, a research release that accompanies the manuscript. Check the values and
the code before using them in published work, and report problems as GitHub issues.

## Install

```bash
pip install git+https://github.com/sangwonsuhvm/lcadep
```

Or clone the repository and run `pip install -e .[test,examples]`.

## What the rule does

| Tier | When it applies | What it does | Function |
| --- | --- | --- | --- |
| 1 | Dependence follows from how the data are built or used: a shared dataset, background or factor; a balance; shares that sum to one; the same factor in several years | It shares one draw, computes the exchange from its driver, or samples Dirichlet shares. No correlation data are needed | `Draws`, `dirichlet_shares` |
| 2 | Exchanges of one process vary together because of a common driver, such as efficiency | It uses one shared-variance fraction λ per exchange; ρij = sign·√(λiλj) | `default_lambda`, `tier2_normals`, `sample_exchange` |
| 3 | Pairs that remain | A first-order screen finds the pairs that can matter. Those pairs get data or elicitation, or else bounds | `screen_pairs`, `first_order_sd` |

## Step by step

The worked case in `examples/worked_case_ngcc.py` follows these six steps. It compares the
global warming impact of electricity from two gas-fired plants that burn the same Marcellus gas.

**Step 1. Set up the model and its inputs.**
- For each uncertain input, record its median and its log-scale standard deviation σ (the
  lognormal "basic" uncertainty).
- Record any pedigree addition separately.
- Compute the first-order sensitivity of the result to the log of each input, a_i = ∂h/∂ln x_i.
  For h > 0 this equals the relative sensitivity coefficient times h. For a comparison, compute it
  for the difference between the alternatives.

**Step 2. Apply Tier 1.**
- Give every shared dataset, background process or characterization factor one key in a `Draws`
  object. Use the same key wherever the input is used, including in both alternatives of a
  comparison:

  ```python
  import lcadep as ld
  dr = ld.Draws(runs=100_000, seed=1)
  L = ld.lognormal(0.008, 0.65, dr.normal(("gas supply", "L")))   # same draws for every user
  ```

- Compute exchanges that a balance or a reporting formula links to a driver from that driver.
  Do not give them their own uncertainty. An example is SO₂ that a monitoring rule computes from
  heat input.
- Sample shares that must sum to one with `dirichlet_shares`.

**Step 3. Apply Tier 2.**
- Look up λ and its sign for the exchanges of each process that has a default:

  ```python
  lam = ld.default_lambda("Natural gas, combined cycle", value_set="within")
  ```

- Use `between` for a dataset that represents a technology average. Use `within` for a
  time-explicit dataset of one facility.
- Draw correlated normals once per process and sample each exchange with Eq. 3. λ acts on the
  basic uncertainty only; pedigree additions are drawn independently:

  ```python
  z = ld.tier2_normals(dr, "plant A", {k: v[0] for k, v in lam.items()}, {k: v[1] for k, v in lam.items()})
  gas = ld.sample_exchange(6.33, 0.046, z["heat_input"], sigma_pedigree=0.0)
  ```

**Step 4. Screen the remaining pairs (Tier 3).**
- Run the screen and list the pairs that Tiers 1 and 2 already cover:

  ```python
  kept, pairs = ld.screen_pairs(a, sigma, names, threshold=0.05, covered=[("q_A", "c_A")])
  ```

- Pairs with a weight of 5% or more that no tier covers are *flagged*.
- By default the screen keeps every input that could form such a pair with the largest input, so
  it misses no flagged pair. Retaining the inputs that explain 95% of the first-order variance
  can drop a small input paired with a large one: in the worked case it would miss two pairs with
  weights of 18%.

**Step 5. Deal with flagged pairs.**
- Use data or structured elicitation where they exist.
- Otherwise, report the result at ρ = 0 and at the worst-case sign: ρ = +1 or −1, whichever
  widens the result or the difference.

**Step 6. Run and report.**
- Run the Monte Carlo simulation. Compute all alternatives of a comparison in the same runs.
- Report:
  - the Tier 1 rules applied;
  - the source and value set of each λ;
  - the flagged pairs and how each was treated;
  - the bounding results.

## Fitting λ from your own data

`lcadep.fit` estimates λ and its sign by the steps used for the default table, so that data for
other sectors, such as farm-level data, go through the same steps as the power plants.

**What you need.** A table with one row per facility-year:
- a column that identifies the facility (plant, farm, site);
- optionally a column with the data year;
- one column per exchange, each **per unit of output** (per MWh, per kg of milk). Values must be
  positive, because the fit works on logarithms. Rows with a missing or non-positive value are
  left out and counted.

At least three exchanges are needed: with two, only the product λ₁λ₂ is identified. The between
fit needs at least 20 facilities, and the within fit at least 20 facilities with three or more
rows (years) each. With one row per facility, only the pooled fit is made.

**From the command line.**

```bash
python -m lcadep.fit farms.csv --facility farm_id --year year --exchanges feed_input ch4 n2o nh3 \
    --sector "Agriculture" --technology "Dairy farm" --region NL --source "your data source" --out rows.csv
```

This prints λ, its 90% interval and its sign for each exchange, the largest residual correlation
and a hold-out test, and writes `rows.csv` in the layout of `default_lambda.csv`.

**From Python.**

```python
import numpy as np
from lcadep import fit

x, facility, years = fit.read_table("farms.csv", "farm_id", ["feed_input", "ch4", "n2o", "nh3"], year="year")
res = fit.fit_lambda(x, facility, ["feed_input", "ch4", "n2o", "nh3"], seed=1)
print(res.summary())
check = fit.holdout(x, facility, seed=1)
rows = fit.table_rows(res, "Agriculture", "Dairy farm", "NL", "2019-2023", "your data source")
```

`fit_lambda` also takes numpy arrays directly, so the data can come from pandas or any other tool.

**What the fit does.**

| Step | What happens | Where |
| --- | --- | --- |
| 1 | Logarithms of each exchange per unit of output | `fit_lambda` |
| 2 | Normal scores of each column, which corresponds to a Gaussian copula | `normal_scores` |
| 3 | One-factor model fitted to the correlation matrix of the scores by minimum residuals; λ is the squared loading and the sign is that of the loading | `one_factor` |
| 4 | The same fit for the normal scores of facility means (`between`), and a one-factor fit to the correlation of the log deviations from facility means (`within`) | `fit_lambda` |
| 5 | 90% intervals of the pooled λ from a bootstrap over facilities (300 replicates by default) | `fit_lambda` |
| 6 | Hold-out test: fit on a random half of the facilities, compare the implied correlations with those observed in the other half, against independence (200 splits by default) | `holdout` |

**How to read the result.**
- A large residual correlation means that one common driver does not describe a pair. In the
  power plants this happened for SO₂ and NOx from coal, which share emission controls. Treat such
  a pair under Tier 3, or report it.
- A loading at the bound (λ = 0.998) signals an exchange that is nearly a function of the common
  driver. The other loadings are then not re-optimized. Check whether it is computed from another exchange; if so, it belongs under Tier 1.
- Exchanges that a reporting rule or an emission factor computes from another exchange also
  belong under Tier 1. Leave them out of the fit.
- A wide interval means that λ is poorly determined. This happens with few facilities and when
  the exchanges are nearly independent, where a one-factor fit can assign a large λ to one
  exchange by chance. Do not use such values as defaults.
- The sign is relative: only the product of two signs matters.

`examples/fit_simulated_facilities.py` runs all of this on simulated data with known λ.

With the same data and seeds, `lcadep.fit` returns the power-plant values of the default table
(pooled, between and within λ, intervals and hold-out errors) to within 10⁻¹⁴.

## Files

| File | Content |
| --- | --- |
| `lcadep/__init__.py` | The functions above |
| `lcadep/default_lambda.csv` | The versioned default table. It holds λ, 90% intervals (pooled fits), signs and sample sizes for two sets of facilities: US fossil power plants (eGRID 2019–2024) and US onshore oil and gas production (GHGRP Subpart W, 2015–2023) |
| `examples/worked_case_ngcc.py` | The worked case of the article. It writes the step log, the results tables and the figure |
| `examples/worked_case_inputs.json` | Plant data and within-plant spreads, derived from eGRID |
| `lcadep/fit.py` | Estimation of λ and its sign from facility-level data, the hold-out test and the command line |
| `examples/fit_simulated_facilities.py` | The fit applied to simulated data with known λ |
| `tests/test_lcadep.py` | Tests of validity, marginals, sample-once behaviour and the screen |
| `tests/test_fit.py` | Tests of the fit: recovery of known λ and signs, between and within fits, the hold-out test and the table layout |
| `tools/build_default_table.py` | Rebuilds the default table from the pilot results of the article (inputs not included) |

## Run

```bash
python -m pytest
python examples/worked_case_ngcc.py worked_case_out/
python examples/fit_simulated_facilities.py fit_example_out/
```

The figure of the worked case needs matplotlib.

## Contributing default values

Values for further sectors and regions are welcome. Each row of the table states:
- the sector and technology;
- the region and data years;
- the value set (`between` for facility means, `within` for year-to-year deviations, `pooled` for all
  facility-years);
- λ and its sign, with a 90% interval where available;
- the number of facilities and facility-years;
- the source.

Estimate the values with `lcadep.fit` (see 'Fitting λ from your own data'), which writes rows in this
layout. Please open an issue that describes the data before submitting a pull request.

## Licence and citation

MIT licence; see `LICENSE`. Citation metadata are in `CITATION.cff`. A Zenodo DOI will be added at
the first archived release.
