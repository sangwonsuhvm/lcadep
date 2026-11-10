"""Worked case: applying the default dependence rule to a comparison of two power plants.

Question. Which of two natural gas combined-cycle (NGCC) plants in Pennsylvania, both supplied with
Marcellus gas, has the lower global warming impact per kWh of electricity, and how certain is that?
Hamilton Patriot (eGRID ORISPL 58426) and Hamilton Liberty (58420) are both 870 MW plants. In 2024
their heat rates were 6,662 and 6,824 Btu/kWh, which differ by 2.4%.

Model (g CO2e per kWh, global warming impact characterized with GWP100), for plant k:
    h_k = c_k + q_k (e_u + kappa L G)
  c_k    CO2 from combustion at plant k, g/kWh                  plant exchange (eGRID 2024)
  q_k    natural gas input at plant k, MJ (net)/kWh             plant exchange (eGRID 2024)
  e_u    CO2 from onshore gas production in the Appalachian basin, g/MJ (net) of gas
                                                                background, shared by both plants
                                                                (US EPA GHGRP Subpart W; processing and
                                                                transmission are not included)
  L      methane loss rate of the Marcellus supply chain        background, shared by both plants
  G      GWP100 of fossil methane, kg CO2e/kg CH4               characterization factor, shared
  kappa  methane in the gas, g CH4/MJ (fixed; IPCC 2006 net calorific value of natural gas, gas
         treated as methane)
All inputs come from public sources (eGRID, GHGRP Subpart W, Barkley et al. 2017, Alvarez et al. 2018,
IPCC 2006 and AR6); the example contains no ecoinvent data.

Steps (Section 4 of the paper):
  1  inputs, medians, log-scale SDs, and first-order sensitivities of the result
  2  Tier 1: shared inputs keyed once, so that both plants read the same draws
  3  Tier 2: within-plant defaults for heat input and CO2 (a time-explicit dataset of one plant)
  4  Tier 3: first-order screen of the difference between the plants
  5  data check and bounds for the flagged pairs that no tier covers
  6  Monte Carlo under each configuration and report

Usage: python worked_case_ngcc.py [out_dir]     (default: ./worked_case_out)
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import lcadep as ld  # noqa: E402

RUNS = 200_000
SEED = 20261109
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "worked_case_out"
OUT.mkdir(parents=True, exist_ok=True)
INP = json.loads((HERE / "worked_case_inputs.json").read_text())

BTU_MJ = 1.055056e-3            # MJ per Btu
NCV_GCV = 0.9                   # net/gross calorific value of natural gas (IPCC 2006, Vol. 2, Ch. 1)
KAPPA = 1e3 / 48.0              # g CH4 per MJ (net): NCV of natural gas 48.0 MJ/kg (IPCC 2006, Vol. 2,
                                # Table 1.2), gas treated as methane
ALVAREZ = 0.023 / 13.16         # national loss rate per Tg CH4 of segment emissions (Alvarez et al. 2018)

# ---------------------------------------------------------------- step 1: inputs
SHARED = {  # name: (median, sigma of ln, source)
    "e_u": (INP["appalachian_production_co2"]["g_per_mj_hhv"] / NCV_GCV,
            INP["appalachian_production_co2"]["sd_ln_yearly"],
            "GHGRP Subpart W, Appalachian basin, gas-dominated onshore production, 2015-2023: sales-weighted CO2 "
            "per MJ sold, converted to net calorific value; sigma: log-scale SD of the yearly basin values"),
    "L": (0.0036 + ALVAREZ * (0.72 + 1.8), math.log(9.63 / 0.75) / (2 * 1.96),
          "Marcellus production and gathering 0.36% (Barkley et al. 2017) plus processing and transmission "
          "and storage at national rates (Alvarez et al. 2018); sigma from the regional range of Sherwin et al. 2024"),
    "G": (29.8, math.log((29.8 + 11) / 29.8) / 1.645, "IPCC AR6, Table 7.15, 29.8 +/- 11 (5-95%)"),
}
SIG_PLANT = INP["pooled_within_plant_sd_ln"]   # default basic uncertainty of a plant-specific exchange
PLANT = {}
for short, name in (("P", "Hamilton Patriot"), ("B", "Hamilton Liberty")):
    p = INP["plants"][name]
    PLANT[short] = {"name": name, "q": p["heat_rate_btu_per_kwh_2024"] * BTU_MJ * NCV_GCV,
                    "c": p["co2_g_per_kwh_2024"],
                    "own_sd": (p["own_sd_ln_heat_input_2019_2024"], p["own_sd_ln_co2_2019_2024"])}


def h(q, c, e_u, L, G):
    return c + q * (e_u + KAPPA * L * G)


med = {k: v[0] for k, v in SHARED.items()}
up = med["e_u"] + KAPPA * med["L"] * med["G"]          # g CO2e per MJ of gas, upstream
h0 = {k: h(p["q"], p["c"], **med) for k, p in PLANT.items()}
names = ["q_P", "c_P", "q_B", "c_B", "e_u", "L", "G"]
dq = PLANT["B"]["q"] - PLANT["P"]["q"]
# first-order sensitivities of the difference D = h_B - h_P to the log of each input
a_diff = np.array([-PLANT["P"]["q"] * up, -PLANT["P"]["c"], PLANT["B"]["q"] * up, PLANT["B"]["c"],
                   dq * med["e_u"], dq * KAPPA * med["L"] * med["G"], dq * KAPPA * med["L"] * med["G"]])
sig = np.array([SIG_PLANT["heat_input"], SIG_PLANT["co2"], SIG_PLANT["heat_input"], SIG_PLANT["co2"],
                SHARED["e_u"][1], SHARED["L"][1], SHARED["G"][1]])

# ---------------------------------------------------------------- steps 2-3: Tier 1 and Tier 2
LAM = ld.default_lambda("Natural gas, combined cycle", "within")       # {exchange: (lambda, sign)}
lam = {"heat_input": LAM["heat_input"][0], "co2": LAM["co2"][0]}
sgn = {"heat_input": LAM["heat_input"][1], "co2": LAM["co2"][1]}

# ---------------------------------------------------------------- step 4: Tier 3 screen
covered = [("q_P", "c_P"), ("q_B", "c_B")]                             # Tier 2 within each plant
kept, pairs = ld.screen_pairs(a_diff, sig, names, threshold=0.05, covered=covered)
kept95, pairs95 = ld.screen_pairs(a_diff, sig, names, threshold=0.05, keep=0.95, covered=covered)
_, pairs_fig = ld.screen_pairs(a_diff, sig, names, threshold=0.02, covered=covered)   # for Fig. 4a


# ---------------------------------------------------------------- steps 5-6: Monte Carlo
def simulate(shared: bool, tier2: bool, rho_plants: float = 0.0, sd=None, seed=SEED):
    """One configuration. shared: Tier 1 (both plants read the same background draws).
    tier2: within-plant defaults for q and c. rho_plants: correlation of the two plants' common
    drivers (Tier 3 bound). sd: per-plant (sd_q, sd_c), default the pooled within-plant SDs."""
    dr = ld.Draws(RUNS, seed)
    out = {}
    for k, p in PLANT.items():
        bkey = "gas supply" if shared else f"gas supply for plant {k}"
        e_u = ld.lognormal(med["e_u"], SHARED["e_u"][1], dr.normal((bkey, "e_u")))
        L = ld.lognormal(med["L"], SHARED["L"][1], dr.normal((bkey, "L")))
        G = ld.lognormal(med["G"], SHARED["G"][1], dr.normal(("characterization" if shared else f"cf {k}", "G")))
        s_q, s_c = sd[k] if sd else (SIG_PLANT["heat_input"], SIG_PLANT["co2"])
        if tier2:
            z = ld.tier2_normals(dr, f"plant {k}", lam, sgn)
            if rho_plants and k == "B":   # bound: driver of plant B tied to that of plant P
                eta_p, eta_b = dr.normal(("plant P", "common driver")), dr.normal(("plant B", "common driver"))
                eta = rho_plants * eta_p + math.sqrt(max(0.0, 1 - rho_plants ** 2)) * eta_b
                z = {ex: sgn[ex] * math.sqrt(lam[ex]) * eta + math.sqrt(1 - lam[ex]) * dr.normal(("plant B", ex))
                     for ex in lam}
            zq, zc = z["heat_input"], z["co2"]
        else:
            zq, zc = dr.normal((f"plant {k}", "heat_input")), dr.normal((f"plant {k}", "co2"))
        q = ld.sample_exchange(p["q"], s_q, zq)
        c = ld.sample_exchange(p["c"], s_c, zc)
        out[k] = h(q, c, e_u, L, G)
    D = out["B"] - out["P"]
    q = lambda x: np.percentile(x, [2.5, 50, 97.5])  # noqa: E731
    return {"P": q(out["P"]), "B": q(out["B"]), "D": q(D), "p_patriot_lower": float((D > 0).mean()),
            "sd_D": float(D.std())}


CONFIGS = [
    ("Independence", dict(shared=False, tier2=False)),
    ("Tier 1", dict(shared=True, tier2=False)),
    ("Tiers 1 and 2 (default rule)", dict(shared=True, tier2=True)),
    ("Tier 3 bound, plants rho = -1", dict(shared=True, tier2=True, rho_plants=-1.0)),
    ("Tier 3 bound, plants rho = +1", dict(shared=True, tier2=True, rho_plants=1.0)),
]
res = {lab: simulate(**kw) for lab, kw in CONFIGS}
own = {k: p["own_sd"] for k, p in PLANT.items()}
sens = {"Independence, own SDs": simulate(False, False, sd=own),
        "Tiers 1 and 2, own SDs": simulate(True, True, sd=own)}

# ---------------------------------------------------------------- report
log = []
log.append(f"Step 1. Deterministic result: Patriot {h0['P']:.1f}, Liberty {h0['B']:.1f} g CO2e/kWh "
           f"(difference {h0['B'] - h0['P']:.1f}, {100 * (h0['B'] / h0['P'] - 1):.1f}%).")
log.append("Step 2. Tier 1: e_u and L keyed 'gas supply', G keyed 'characterization', shared by both plants.")
log.append(f"Step 3. Tier 2 (within-plant, NGCC): lambda heat input {lam['heat_input']:.3f}, CO2 {lam['co2']:.3f}, "
           f"rho = {math.sqrt(lam['heat_input'] * lam['co2']):.3f}.")
log.append(f"Step 4. Screen of the difference: inputs kept {kept} (a 95%-variance retention would keep only {kept95}).")
for p in pairs:
    log.append(f"         {p.i}-{p.j}: w = {100 * p.weight:.1f}%  {'covered (Tier 2)' if p.covered else 'not covered'}")
pc = INP["pa_pairs_heat_rate_deviation_corr"]
log.append(f"Step 5. Data check: yearly heat-rate deviations of {pc['pairs']} pairs of Pennsylvania NGCC plants "
           f"correlate at {pc['mean']:.2f} on average (10th-90th percentile {pc['p10']:.2f} to {pc['p90']:.2f}); "
           "bounds at rho = -1 and +1 between the plants' drivers are reported.")
for lab, r in {**res, **sens}.items():
    log.append(f"Step 6. {lab}: Patriot {r['P'][1]:.0f} ({r['P'][0]:.0f}-{r['P'][2]:.0f}); Liberty {r['B'][1]:.0f} "
               f"({r['B'][0]:.0f}-{r['B'][2]:.0f}); difference {r['D'][1]:.1f} ({r['D'][0]:.1f} to {r['D'][2]:.1f}), "
               f"95% width {r['D'][2] - r['D'][0]:.1f}; P(Patriot lower) = {r['p_patriot_lower']:.3f}")
(OUT / "worked_case_log.txt").write_text("\n".join(log) + "\n")
print("\n".join(log))

with open(OUT / "worked_case_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["configuration", "patriot_p2.5", "patriot_median", "patriot_p97.5", "liberty_p2.5", "liberty_median",
                "liberty_p97.5", "difference_p2.5", "difference_median", "difference_p97.5", "difference_95_width",
                "p_patriot_lower"])
    for lab, r in {**res, **sens}.items():
        w.writerow([lab, *np.round(r["P"], 2), *np.round(r["B"], 2), *np.round(r["D"], 2),
                    round(r["D"][2] - r["D"][0], 2), round(r["p_patriot_lower"], 4)])
with open(OUT / "worked_case_screen.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["input_i", "input_j", "weight", "covered_by_tier1_or_2"])
    for p in pairs:
        w.writerow([p.i, p.j, round(p.weight, 4), p.covered])
json.dump({"results": {k: {kk: (vv.tolist() if hasattr(vv, "tolist") else vv) for kk, vv in v.items()}
                       for k, v in {**res, **sens}.items()},
           "screen": [p.__dict__ for p in pairs], "kept": kept, "deterministic": h0,
           "lambda_within": lam, "runs": RUNS, "seed": SEED}, open(OUT / "worked_case_results.json", "w"), indent=1)

# ---------------------------------------------------------------- figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit(0)
MM = 1 / 25.4
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "pdf.fonttype": 42})
BLUE, DARK, MID, PALE, ORANGE = "#2b6cb0", "#374151", "#6b7280", "#e5e7eb", "#c05621"
LAB = {"q_P": "Gas input, Patriot", "c_P": "CO$_2$, Patriot", "q_B": "Gas input, Liberty", "c_B": "CO$_2$, Liberty",
       "e_u": "Production CO$_2$", "L": "Methane loss rate", "G": "GWP100 of methane"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(174 * MM, 70 * MM), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.9})
for i, p in enumerate(pairs_fig):
    ax1.barh(i, 100 * p.weight, color=(BLUE if p.covered else ORANGE) if p.weight >= 0.05 else PALE, height=0.62)
    ax1.text(100 * p.weight + 0.8, i, f"{100 * p.weight:.0f}%", va="center", fontsize=8, color=DARK)
ax1.set_yticks(range(len(pairs_fig)), [f"{LAB[p.i]} – {LAB[p.j]}" for p in pairs_fig])
ax1.invert_yaxis()
ax1.axvline(5, color=MID, lw=0.6, ls=(0, (3, 2)))
ax1.text(6.0, -0.75, "5% threshold", fontsize=8, color=MID, va="bottom")
ax1.set_xlabel("Screen weight w$_{ij}$ for the difference (%)")
ax1.tick_params(axis="y", length=0)
ax1.spines["left"].set_visible(False)
from matplotlib.patches import Patch
ax1.legend([Patch(color=BLUE), Patch(color=ORANGE), Patch(color=PALE)],
           ["Covered by Tier 2", "Not covered (Tier 3)", "Below the threshold"],
           frameon=False, fontsize=8, loc="center right", bbox_to_anchor=(1.04, 0.47), handlelength=0.9,
           handletextpad=0.5)
ax1.text(-0.02, 1.04, "a", transform=ax1.transAxes, fontsize=10, fontweight="bold", ha="right")

order = [lab for lab, _ in CONFIGS]
short = {"Independence": "Independence", "Tier 1": "Tier 1 only", "Tiers 1 and 2 (default rule)": "Tiers 1 and 2",
         "Tier 3 bound, plants rho = -1": "Bound, ρ = −1", "Tier 3 bound, plants rho = +1": "Bound, ρ = +1"}
for i, lab in enumerate(order):
    r = res[lab]
    col = BLUE if lab.startswith("Tiers 1 and 2") else (MID if lab.startswith("Tier 3") else DARK)
    ax2.plot([r["D"][0], r["D"][2]], [i, i], color=col, lw=2.4, solid_capstyle="butt")
    ax2.plot(r["D"][1], i, "o", ms=4, mfc="white", mec=col, mew=1.0)
    ax2.text(122, i, f"{r['p_patriot_lower']:.2f}", va="center", fontsize=8, color=DARK)
ax2.text(122, -0.55, "P(Patriot\nlower)", fontsize=8, color=DARK, va="bottom")
ax2.axvline(0, color=MID, lw=0.6)
ax2.set_yticks(range(len(order)), [short[o] for o in order])
ax2.invert_yaxis()
ax2.set_xlim(-100, 120)
ax2.set_xlabel("Liberty minus Patriot, global warming impact (g CO$_2$e/kWh)")
ax2.tick_params(axis="y", length=0)
ax2.spines["left"].set_visible(False)
ax2.grid(axis="x", color=PALE, lw=0.5)
ax2.set_axisbelow(True)
ax2.text(-0.02, 1.04, "b", transform=ax2.transAxes, fontsize=10, fontweight="bold", ha="right")
for ext, kw in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
    fig.savefig(OUT / f"Fig4.{ext}", bbox_inches="tight", **kw)
print(OUT)
