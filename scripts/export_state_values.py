"""Write the value used for every Nigerian state, ingredient and property to one CSV.

Usage:  python scripts/export_state_values.py [output.csv]
Default output: data/derived/state_values.csv

The app does not read this file. It is a snapshot for review: for each of the
36 states and the FCT, every ingredient value and price the engine would use,
whether it was measured there, calculated, or estimated from other states,
with its 95% range and how it was obtained. Re-run it after changing the data.
"""

import csv
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from ai_engine import properties as P  # noqa: E402
from ai_engine.dataset import LOCATION_STATUS_LABELS, Dataset  # noqa: E402
from ai_engine.locations import NIGERIA_STATES  # noqa: E402

DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "derived", "state_values.csv")
HEADER = ["state", "ingredient_id", "ingredient", "property", "label", "unit", "value", "low", "high", "type",
          "confidence", "how_obtained", "n_records", "based_on_states", "method"]
TYPE = {"measured": "measured", "calculated": "calculated", "predicted": "estimated"}


def _rows(ds, state):
    view = ds.view(state)
    for iid, ing in view.ingredients.items():
        for key in P.PROPERTIES:
            v = ing.values.get(key)
            if v is None:
                continue
            p = v.prediction or {}
            yield {
                "state": state, "ingredient_id": iid, "ingredient": ing.name, "property": key,
                "label": P.get(key).label, "unit": P.get(key).unit, "value": round(v.value, 4),
                "low": round(v.low, 4), "high": round(v.high, 4), "type": TYPE[v.kind], "confidence": v.confidence,
                "how_obtained": LOCATION_STATUS_LABELS.get(v.location_status, ""), "n_records": v.n,
                "based_on_states": "; ".join(p.get("training_states") or v.locations),
                "method": p.get("method") or ("; ".join(sorted({r.method for r in v.records if r.method}))),
            }
        pr = ing.price
        if pr is not None:
            p = pr.prediction or {}
            yield {
                "state": state, "ingredient_id": iid, "ingredient": ing.name, "property": "price",
                "label": "Price", "unit": "NGN/kg", "value": round(pr.value), "low": round(pr.low),
                "high": round(pr.high), "type": TYPE[pr.kind], "confidence": pr.confidence,
                "how_obtained": LOCATION_STATUS_LABELS.get(pr.location_status, ""), "n_records": pr.n,
                "based_on_states": (
                    "; ".join(pr.states) if not p else
                    f"{state} price level; {ing.name} prices from {', '.join(p['ingredient_states'])}"
                    if p["case"] == "missing_ingredient" else
                    "; ".join(f"{s} {w:.0%}" for s, w in p["weights"].items())),
                "method": p.get("method", f"average of {pr.n} verified lot(s)"),
            }


def main(out_path):
    ds = Dataset(DATA)
    if ds.issues:
        print("Fix the data issues first (flask --app run.py validate-data):")
        for issue in ds.issues:
            print("  -", issue)
        raise SystemExit(1)
    rows = [r for st in NIGERIA_STATES for r in _rows(ds, st)]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)

    counts = {}
    for r in rows:
        counts[r["type"], r["confidence"]] = counts.get((r["type"], r["confidence"]), 0) + 1
    print(f"Wrote {len(rows)} rows for {len(NIGERIA_STATES)} states to {os.path.relpath(out_path, ROOT)}")
    for (kind, conf), n in sorted(counts.items()):
        print(f"  {kind:10s} {conf:10s} {n:5d}")
    print("Never measured in any state (no estimate possible):")
    for iid, ing in ds.ingredients.items():
        missing = [k for k in P.PROPERTIES if not ds.records.get((iid, k))]
        print(f"  {ing.name}: {len(missing)} of {len(P.PROPERTIES)} properties ({', '.join(missing)})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else OUT)
