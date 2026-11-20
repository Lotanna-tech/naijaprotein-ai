"""Test fixtures.

The datasets built here are SYNTHETIC and exist only inside the tests, to
check calculations against hand-worked answers. They are never written to the
application's data/ folder.
"""

import csv
import os

import pytest

PROP_HEADER = ["ingredient_id", "property", "value", "unit", "basis", "method", "processing", "source_id",
               "status", "notes", "sample", "location", "variety"]
PRICE_HEADER = ["ingredient_id", "price_ngn_per_kg", "market", "price_date", "source_id", "status", "notes",
                "location", "supplier", "availability_kg", "max_blend_share_pct", "lot_id"]
INGREDIENTS = [
    ("cowpea", "Cowpea"), ("bambara_groundnut", "Bambara Groundnut"), ("defatted_groundnut", "Defatted Groundnut"),
    ("african_yam_bean", "African Yam Bean"), ("soybean", "Soybean"),
]


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def prop(iid, key, value, state, unit="g/100 g", basis="dry", status="sourced", sample="", source="T1"):
    return {"ingredient_id": iid, "property": key, "value": value, "unit": unit, "basis": basis,
            "source_id": source, "status": status, "sample": sample, "location": f"Nigeria / {state}"}


def price(iid, value, state, stock="", max_share="", status="sourced"):
    return {"ingredient_id": iid, "price_ngn_per_kg": value, "source_id": "T1", "status": status,
            "location": f"Nigeria / {state} / Test market", "availability_kg": stock,
            "max_blend_share_pct": max_share, "price_date": "2026-10-01"}


def build(tmp_path, props, prices=(), experiments=()):
    d = str(tmp_path)
    _write(os.path.join(d, "sources.csv"), ["source_id", "citation", "year", "doi_or_url", "notes"],
           [{"source_id": "T1", "citation": "Synthetic test source", "year": "2026"}])
    _write(os.path.join(d, "ingredients.csv"),
           ["id", "name", "scientific_name", "category", "form", "description", "processing",
            "default_min_inclusion", "default_max_inclusion", "image_url"],
           [{"id": i, "name": n, "default_min_inclusion": 0, "default_max_inclusion": 1} for i, n in INGREDIENTS])
    _write(os.path.join(d, "ingredient_properties.csv"), PROP_HEADER, props)
    _write(os.path.join(d, "ingredient_prices.csv"), PRICE_HEADER, prices)
    _write(os.path.join(d, "formulation_experiments.csv"),
           ["experiment_id", "source_id", "property", "value", "unit", "method", "location"]
           + [f"share_{i}" for i, _ in INGREDIENTS] + ["notes"], experiments)
    _write(os.path.join(d, "product_targets.csv"), ["product", "property", "min", "max", "source_id", "notes"], [])
    return d


@pytest.fixture
def simple_data(tmp_path):
    """Two ingredients measured in Kano with known values, prices and stock."""
    props = []
    for iid, protein, fat, moist in (("soybean", 40.0, 20.0, 10.0), ("cowpea", 25.0, 2.0, 10.0)):
        props += [prop(iid, "protein", protein, "Kano"), prop(iid, "fat", fat, "Kano"),
                  prop(iid, "moisture", moist, "Kano", basis="as_is"),
                  prop(iid, "water_absorption", 2.0 if iid == "soybean" else 1.5, "Kano", unit="g/g", basis="")]
    prices = [price("soybean", 1000, "Kano", stock=1000, max_share=80), price("cowpea", 1200, "Kano", stock=1000)]
    return build(tmp_path, props, prices)
