"""Tests for the web application and the real dataset rules."""

import csv
import os

import pytest

from ai_engine.dataset import Dataset
from ai_engine.locations import parse_location
from app import create_app
from app.extensions import db

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SCOPE = {"cowpea", "bambara_groundnut", "defatted_groundnut", "african_yam_bean", "soybean"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    import app.config as cfg
    monkeypatch.setattr(cfg.Config, "SQLALCHEMY_DATABASE_URI", f"sqlite:///{tmp_path / 'test.db'}")
    app = create_app()
    app.config["TESTING"] = True
    with app.app_context():
        db.create_all()
    c = app.test_client()
    c.post("/register", data=dict(company_name="T", full_name="T", email="t@example.com", password="password123"))
    c.post("/login", data=dict(email="t@example.com", password="password123"))
    return c


# ---------------------------------------------------------------- real dataset
def test_real_dataset_has_only_scope_nigeria_and_no_placeholders():
    with open(os.path.join(ROOT, "data", "ingredient_properties.csv")) as f:
        rows = list(csv.DictReader(f))
    assert {r["ingredient_id"] for r in rows} <= SCOPE
    assert all(parse_location(r["location"])[0] for r in rows), "every value must be located in a Nigerian state"
    assert not any(r["source_id"] == "S000" for r in rows), "placeholder values must not be in the dataset"
    ds = Dataset(os.path.join(ROOT, "data"), SCOPE)
    assert set(ds.ingredients) == SCOPE
    assert ds.issues == []
    for (iid, key), recs in ds.records.items():
        assert all(r.status in ("measured", "sourced") for r in recs)


# ---------------------------------------------------------------- pages and API
def test_pages_load(client):
    for path in ["/", "/ingredients/", "/ingredients/?location=Kano", "/ingredients/compare?product=sausage",
                 "/ingredients/soybean?location=Lagos", "/formulations/", "/formulations/simulate"]:
        assert client.get(path).status_code == 200, path


def test_no_standalone_prediction_page(client):
    for path in ["/predict", "/prediction", "/api/predict", "/formulations/predict"]:
        assert client.get(path).status_code == 404


def test_location_must_be_nigerian(client):
    assert client.get("/ingredients/?location=Sudan").status_code == 400
    r = client.post("/api/simulate", json={"product": "beef", "location": "Ghana", "recipe": {"soybean": 100}})
    assert r.status_code == 400


def test_simulate_requires_product_and_valid_recipe(client):
    assert client.post("/api/simulate", json={"recipe": {"soybean": 100}}).status_code == 400
    r = client.post("/api/simulate", json={"product": "beef", "recipe": {"soybean": 60}})
    assert r.status_code == 400 and "100%" in r.json["error"]


def test_simulate_and_optimise(client):
    r = client.post("/api/simulate", json={"product": "beef", "location": "Kano",
                                           "recipe": {"soybean": 40, "defatted_groundnut": 60}, "batch_kg": 100})
    assert r.status_code == 200
    blend = r.json["result"]["blend"]
    assert 0 <= blend["score"]["low"] <= blend["score"]["value"] <= blend["score"]["high"] <= 100
    assert r.json["result"]["certainty"]["level"] in ("high", "medium", "low")
    r = client.post("/api/simulate", json={"product": "beef", "recipe": {"soybean": 50, "water": 50}})
    assert r.status_code == 400 and "Water and oil" in r.json["error"]
    r = client.post("/api/optimise", json={"product": "sausage", "location": "Kano", "batch_kg": 100,
                                           "ingredient_ids": sorted(SCOPE), "baseline": {"soybean": 50, "cowpea": 50}})
    assert r.status_code == 200 and r.json["result"]["feasible"]
    res = r.json["result"]
    assert [x["rank"] for x in res["ranking"]] == list(range(1, len(SCOPE) + 1))
    assert abs(sum(res["blend"]["shares"].values()) - 1) < 1e-4
    assert all(c["share"] <= c["max_share"] + 1e-6 for c in res["blend"]["components"])
    assert res["comparison"]["score_gain"] >= 0 and res["lab_plan"]["runs"]
    r = client.post("/api/optimise", json={"product": "sausage", "location": "Kano", "batch_kg": 100,
                                           "ingredient_ids": sorted(SCOPE), "budget": 1000})
    assert r.status_code == 200 and not r.json["result"]["feasible"]
    assert "budget" in r.json["explanation"]["reasons"][0]


def test_state_without_prices_uses_estimated_prices(client):
    r = client.post("/api/optimise", json={"product": "beef", "location": "Lagos", "batch_kg": 100,
                                           "ingredient_ids": sorted(SCOPE)})
    assert r.status_code == 200 and r.json["result"]["feasible"]
    used = r.json["result"]["blend"]["components"]
    assert used and all(c["price"]["kind"] == "predicted" for c in used)
    assert r.json["result"]["blend"]["cost"]["estimated_prices"]
