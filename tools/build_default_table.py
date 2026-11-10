"""Write default_lambda.csv, the versioned default table, from the pilot results.

Usage: python build_default_table.py <pilot_results.json> <ghgrp_w_results.json>

Each row gives one exchange of one technology and value set:
  pooled   all facility-years, with a 90% interval from a bootstrap over facilities
  between  facility means (for datasets that represent a technology average)
  within   deviations from facility means (for a time-explicit dataset of one facility)
Intervals are available for the pooled fit only. Signs are those of the fitted loadings.
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "lcadep"
POWER = {"heat_input": "heat input (fuel input) per MWh net", "co2": "CO2 per MWh net",
         "so2": "SO2 per MWh net", "nox": "NOx per MWh net"}
OILGAS = {"co2_comb": "CO2 from combustion equipment per GJ sold", "co2_proc": "other CO2 per GJ sold",
          "ch4_equip": "CH4 from pneumatic devices and equipment leaks per GJ sold",
          "ch4_vent": "other vented CH4 per GJ sold"}
NOTE_GAS_SO2 = "largely reporting-method dependence; compute SO2 from heat input under Tier 1 instead"


def rows_for(res, counts, sector, exchanges, source, years, sign_key="loadings", within_key="loadings_within"):
    out = []
    for tech, r in res.items():
        c = counts.get(tech, {})
        for vs, key in (("pooled", "lambda"), ("between", "lambda_between"), ("within", "lambda_within")):
            for ex, desc in exchanges.items():
                lam = min(float(r[key][ex]), 0.999)
                lo, hi = (r["lambda_ci90"][ex] if vs == "pooled" else ("", ""))
                load = r[within_key][ex] if (vs == "within" and within_key in r) else r[sign_key][ex]
                note = NOTE_GAS_SO2 if (ex == "so2" and tech.startswith("Natural gas")) else ""
                if lam >= 0.998:
                    note = (note + "; " if note else "") + "at the estimation bound"
                out.append({"sector": sector, "technology": tech, "region": "US", "data_years": years,
                            "value_set": vs, "exchange": ex, "exchange_description": desc,
                            "lambda": f"{lam:.3f}", "ci90_low": f"{min(lo, 0.999):.3f}" if lo != "" else "",
                            "ci90_high": f"{min(hi, 0.999):.3f}" if hi != "" else "",
                            "sign": "1" if load >= 0 else "-1",
                            "facilities": c.get("plants", c.get("facilities", "")),
                            "facility_years": c.get("plant_years", c.get("facility_years", "")),
                            "source": source, "note": note})
    return out


def main(pilot, ghgrp):
    p = json.loads(Path(pilot).read_text())
    g = json.loads(Path(ghgrp).read_text())
    rows = rows_for(p["results"], p["counts"], "Electricity, fossil", POWER,
                    "US EPA eGRID 2019-2023; eGRID2024 (Cornerstone Sustainability Data Initiative 2026)", "2019-2024")
    rows += rows_for(g["results"], g["counts"], "Oil and gas production, onshore", OILGAS,
                     "US EPA GHGRP Subpart W, Envirofacts (retrieved 8 Nov 2026)", "2015-2023")
    out = HERE / "default_lambda.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
