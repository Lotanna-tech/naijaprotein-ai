"""Import the Hack the Fork Nigeria 2026 plant-protein dataset into data/.

Usage:  python scripts/import_htf_dataset.py [path/to/raw.csv]

The script is idempotent: rows previously imported from this source (S008)
are replaced, so it can be re-run after the raw file changes.

Mapping decisions (the raw file does not state units or basis for all columns):
- Proximate columns (protein, fat, ash, crude fibre) are read as g/100 g on an
  as-is basis, because moisture is reported alongside them. They are converted
  to dry basis with the same record's moisture.
- water/oil absorption capacity are read as g per 100 g (values 148-274), and
  converted to g/g.
- emulsifying capacity is read as a percentage and stored as
  emulsifying_capacity_pct, separate from emulsifying capacity in ml oil/g.
- verification_status maps to status: Sourced -> sourced, Estimated ->
  estimated (estimate by the data provider), Assumed -> assumed.
- Composition and functional values are located where the lot was GROWN
  ("Nigeria / <production_origin_state>"), because soil, climate and variety
  act there. Prices and stock are located where the lot is SOLD
  ("Nigeria / <acquisition_location> / <market>").
- Records whose variety is a defatted flour of soybean or Bambara groundnut
  are a different ingredient form and are not imported.
- Placeholder rows (S000) for properties and prices that this dataset now
  covers are removed.
"""

import csv
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw", "hack-the-fork-nigeria-2026-plant-proteins.csv")
SOURCE_ID = "S008"
SOURCE_ROW = [
    SOURCE_ID,
    "Hack the Fork Nigeria 2026 - plant proteins dataset (hack-the-fork-nigeria-2026-plant-proteins.csv), "
    "provided to the team. Lot-level market samples from Akwa Ibom, Benue and Kano.",
    "2026",
    "",
    "Record-level reference codes (e.g. IITA-SOY-2023) are kept in each value's notes; full citations were not "
    "provided. Units of functional columns and the proximate basis are not stated in the file; see "
    "scripts/import_htf_dataset.py for how they were read.",
]

INGREDIENT_IDS = {
    "Soybean": "soybean",
    "Cowpea": "cowpea",
    "Bambara Groundnut": "bambara_groundnut",
    "African Yam Bean": "african_yam_bean",
    "Defatted Groundnut": "defatted_groundnut",
}
STATUS = {"Sourced": "sourced", "Estimated": "estimated", "Assumed": "assumed"}

# raw column -> (property, unit, basis)
COLUMNS = {
    "moisture_pct": ("moisture", "g/100 g", "as_is"),
    "protein_pct": ("protein", "g/100 g", "as_is"),
    "fat_pct": ("fat", "g/100 g", "as_is"),
    "ash_pct": ("ash", "g/100 g", "as_is"),
    "crude_fiber_pct": ("fibre", "g/100 g", "as_is"),
    "water_absorption_capacity": ("water_absorption", "g/100 g", ""),
    "oil_absorption_capacity": ("oil_absorption", "g/100 g", ""),
    "protein_solubility": ("protein_solubility", "%", ""),
    "least_gelation_concentration": ("least_gelation", "% w/v", ""),
    "emulsifying_capacity": ("emulsifying_capacity_pct", "%", ""),
}
# Placeholders superseded by this dataset.
REPLACED_PROPERTIES = {p for p, _, _ in COLUMNS.values()} | {"carbohydrate"}

METHOD_PREFIXES = ("AOAC", "Kjeldahl")


def out_of_scope(row):
    """Defatted soy / Bambara flours are not the full-fat seed flours in scope."""
    return row["ingredient"] in ("Soybean", "Bambara Groundnut") and "defatted" in row["variety"].lower()


def processing_of(row):
    first = row["measurement_method"].split("/")[0].strip()
    return "" if first.startswith(METHOD_PREFIXES) else first


def location_of(row):
    """Where the lot is sold (prices, stock)."""
    return f"Nigeria / {row['acquisition_location'].strip()} / {row['market_location'].strip()}"


def origin_of(row):
    """Where the lot was grown (composition, functional properties)."""
    return f"Nigeria / {row['production_origin_state'].strip()}"


def read(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f)), None


def rewrite(path, rows, header):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def header_of(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return next(csv.reader(f))


def main(raw_path):
    raw, _ = read(raw_path)
    imported, skipped = [], []
    for r in raw:
        if r["ingredient"] not in INGREDIENT_IDS:
            skipped.append((r["record_id"], f"ingredient '{r['ingredient']}' outside scope"))
        elif out_of_scope(r):
            skipped.append((r["record_id"], f"{r['variety']}: defatted form of {r['ingredient']}, not in scope"))
        elif r["verification_status"] not in STATUS:
            skipped.append((r["record_id"], f"unknown verification_status '{r['verification_status']}'"))
        else:
            imported.append(r)

    # sources.csv
    src_path = os.path.join(DATA, "sources.csv")
    sources, _ = read(src_path)
    sources = [s for s in sources if s["source_id"] != SOURCE_ID]
    sources.append(dict(zip(header_of(src_path), SOURCE_ROW)))
    rewrite(src_path, sources, header_of(src_path))

    # ingredient_properties.csv
    prop_path = os.path.join(DATA, "ingredient_properties.csv")
    header = header_of(prop_path)
    props, _ = read(prop_path)
    covered = {INGREDIENT_IDS[r["ingredient"]] for r in imported}
    props = [p for p in props
             if p["source_id"] != SOURCE_ID
             and not (p["source_id"] == "S000" and p["ingredient_id"] in covered
                      and p["property"] in REPLACED_PROPERTIES)]
    new_props = []
    for r in imported:
        iid = INGREDIENT_IDS[r["ingredient"]]
        note = (f"{r['record_id']}, batch {r['batch_id']}; bought at {r['market_location']} "
                f"({r['acquisition_location']}); dataset references: {r['source_id']}")
        for col, (prop, unit, basis) in COLUMNS.items():
            if r[col].strip() == "":
                continue
            new_props.append({
                "ingredient_id": iid, "property": prop, "value": r[col], "unit": unit, "basis": basis,
                "method": r["measurement_method"], "processing": processing_of(r), "source_id": SOURCE_ID,
                "status": STATUS[r["verification_status"]], "notes": note, "sample": r["record_id"],
                "location": origin_of(r), "variety": r["variety"],
            })
    rewrite(prop_path, new_props + props, header)

    # ingredient_prices.csv
    price_path = os.path.join(DATA, "ingredient_prices.csv")
    price_header = header_of(price_path)
    for col in ("supplier", "availability_kg", "max_blend_share_pct", "lot_id"):
        if col not in price_header:
            price_header.append(col)
    prices, _ = read(price_path)
    prices = [p for p in prices
              if p["source_id"] != SOURCE_ID and not (p["source_id"] == "S000" and p["ingredient_id"] in covered)]
    new_prices = [{
        "ingredient_id": INGREDIENT_IDS[r["ingredient"]], "price_ngn_per_kg": r["price_per_kg_ngn"],
        "market": r["market_location"], "price_date": r["price_date"], "source_id": SOURCE_ID,
        "status": STATUS[r["verification_status"]],
        "notes": f"{r['record_id']}; {r['variety']}; dataset references: {r['source_id']}",
        "location": location_of(r), "supplier": r["supplier"], "availability_kg": r["availability_kg"],
        "max_blend_share_pct": r["max_blend_share_pct"], "lot_id": r["record_id"],
    } for r in imported]
    rewrite(price_path, new_prices + prices, price_header)

    print(f"Imported {len(imported)} records ({len(new_props)} values, {len(new_prices)} prices).")
    for rid, why in skipped:
        print(f"Skipped {rid}: {why}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else RAW)
