"""Tests for the calculation engine: data rules, location model, simulation,
optimisation and prediction. Expected values are worked out by hand."""

import pytest

from ai_engine import locations
from ai_engine.blend import BlendError, simulate
from ai_engine.dataset import Dataset
from ai_engine.location_model import fit
from ai_engine.optimiser import optimise
from ai_engine.predictor import Predictor
from tests.conftest import build, price, prop


# ---------------------------------------------------------------- scope
def test_nigeria_has_36_states_and_fct_only():
    assert len(locations.NIGERIA_STATES) == 37
    assert "FCT" in locations.NIGERIA_STATES
    assert locations.parse_location("Nigeria / Kano / Dawanau") == ("Kano", "Dawanau")
    assert locations.parse_location("Sudan / Gezira")[0] is None
    assert locations.normalise_state("abuja") == "FCT"


def test_non_nigerian_and_unverified_records_are_not_used(tmp_path):
    d = build(tmp_path, [
        prop("soybean", "protein", 40, "Kano"),
        {**prop("soybean", "protein", 99, "Kano"), "location": "Sudan / Gezira"},
        prop("soybean", "protein", 70, "Kano", status="assumed"),
        prop("soybean", "protein", 60, "Kano", status="estimated"),
    ])
    ds = Dataset(d)
    v = ds.get("soybean").values["protein"]
    assert v.value == 40 and v.n == 1
    assert v.unverified == 2
    assert any("Nigerian" in e for e in ds.excluded)


def test_as_is_values_use_same_sample_moisture(tmp_path):
    d = build(tmp_path, [
        prop("cowpea", "moisture", 10, "Kano", basis="as_is", sample="A"),
        prop("cowpea", "protein", 18, "Kano", basis="as_is", sample="A"),
        prop("cowpea", "protein", 30, "Kano", basis="as_is", sample="B"),  # no moisture for B
    ])
    ds = Dataset(d)
    v = ds.get("cowpea").values["protein"]
    assert v.n == 1 and v.value == pytest.approx(20.0)  # 18 * 100 / 90
    assert any("sample 'B'" in i for i in ds.issues)


# ---------------------------------------------------------------- location model
def test_location_model_shrinks_and_validates():
    groups = {"Kano": [25.0, 25.4], "Benue": [25.8, 26.0], "Oyo": [25.2], "Lagos": [25.5, 25.7]}
    m = fit("protein", groups)
    assert m.validated
    assert m.validation["relative_error"] < 0.05
    # a state with one record is shrunk towards the national mean
    assert abs(m.state_effects["Oyo"]) <= abs(25.2 - m.mu) + 1e-9
    est = m.predict_unseen()
    assert est["low"] < m.mu < est["high"]


def test_location_model_refuses_with_few_or_diverse_states():
    assert not fit("p", {"Kano": [25.0], "Benue": [26.0]}).validated
    diverse = fit("p", {"Kano": [10.0], "Benue": [30.0], "Oyo": [20.0]})
    assert not diverse.validated and diverse.predict_unseen() is None


def test_state_view_uses_state_data_or_validated_estimate(tmp_path):
    props = [prop("cowpea", "protein", v, s) for s, v in (("Kano", 25.0), ("Benue", 25.5), ("Oyo", 25.2))]
    props += [prop("soybean", "protein", 40.0, "Kano"), prop("soybean", "protein", 46.0, "Benue")]
    ds = Dataset(build(tmp_path, props))
    kano = ds.view("Kano")
    assert kano.get("cowpea").values["protein"].value == 25.0
    assert kano.get("cowpea").values["protein"].location_status == "selected"
    lagos = ds.view("Lagos")
    est = lagos.get("cowpea").values["protein"]
    assert est.status == "predicted" and est.location_status == "location_model"
    assert "protein" not in lagos.get("soybean").values  # only 2 states: not validated
    assert ("soybean", "protein") in lagos.location_refusals
    with pytest.raises(ValueError):
        ds.view("Sudan")


# ---------------------------------------------------------------- simulation
def test_mass_balance_with_water_and_oil(simple_data):
    ds = Dataset(simple_data).view("Kano")
    r = simulate(ds, Predictor(ds), {"soybean": 50, "water": 40, "oil": 10}, "beef", batch_kg=10)
    # soybean as-is: protein 40 * 0.9 = 36, fat 20 * 0.9 = 18, moisture 10
    assert r["mix"]["protein"]["value"] == pytest.approx(0.5 * 36)
    assert r["mix"]["fat"]["value"] == pytest.approx(0.5 * 18 + 0.1 * 100)
    assert r["mix"]["moisture"]["value"] == pytest.approx(0.5 * 10 + 0.4 * 100)
    assert r["dry_protein_blend"]["protein"] == pytest.approx(40)
    # cost excludes unpriced water and oil, and says so
    assert r["cost"]["value"] == pytest.approx(500) and "Added water" in r["cost"]["excludes"]
    assert r["components"][0]["kg"] == pytest.approx(5)


def test_recipe_must_sum_to_100(simple_data):
    ds = Dataset(simple_data)
    with pytest.raises(BlendError):
        simulate(ds, Predictor(ds), {"soybean": 50, "water": 40}, "beef")
    with pytest.raises(ValueError):
        simulate(ds, Predictor(ds), {"soybean": 100}, "pizza")


def test_functional_properties_are_not_averaged(simple_data):
    ds = Dataset(simple_data)
    r = simulate(ds, Predictor(ds), {"soybean": 50, "cowpea": 50}, "beef")
    f = r["functional"]["water_absorption"]
    assert f["available"] is False and len(f["ingredient_values"]) == 2


def test_targets_are_checked(simple_data):
    ds = Dataset(simple_data)
    r = simulate(ds, Predictor(ds), {"soybean": 50, "water": 50}, "beef",
                 targets={"protein": {"min": 20}, "fat": {"max": 5}, "water_absorption": {"min": 1}})
    status = {t["property"]: t["status"] for t in r["targets"]}
    assert status == {"protein": "not_met", "fat": "not_met", "water_absorption": "not_checked"}


# ---------------------------------------------------------------- optimisation
def test_optimiser_meets_constraints_at_lowest_cost(simple_data):
    ds = Dataset(simple_data).view("Kano")
    r = optimise(ds, ["soybean", "cowpea"], 100, targets={"protein": {"min": 20}},
                 water={"min": 0, "max": 50}, water_price=0)
    assert r["feasible"]
    x = r["recipe"]
    protein = x.get("soybean", 0) * 36 + x.get("cowpea", 0) * 22.5
    assert protein >= 20 - 1e-4  # recipe fractions are rounded to 6 decimals
    # soybean is cheaper per unit of protein; its share is capped at 80% of the protein blend
    prot_total = x["soybean"] + x.get("cowpea", 0)
    assert x["soybean"] / prot_total <= 0.8 + 1e-4
    assert sum(x.values()) == pytest.approx(1)


def test_optimiser_reports_infeasible_honestly(simple_data):
    ds = Dataset(simple_data).view("Kano")
    r = optimise(ds, ["soybean", "cowpea"], 100, targets={"protein": {"min": 60}})
    assert not r["feasible"] and "recipe" not in r
    assert "highest protein" in r["diagnosis"]["reasons"][0]


def test_optimiser_respects_stock(simple_data):
    ds = Dataset(simple_data).view("Kano")
    r = optimise(ds, ["soybean", "cowpea"], 5000, targets={"protein": {"min": 20}})
    assert not r["feasible"]
    assert "stock" in r["diagnosis"]["reasons"][0]


def test_optimiser_refuses_functional_constraints(simple_data):
    ds = Dataset(simple_data)
    with pytest.raises(ValueError):
        optimise(ds, ["soybean"], 100, targets={"water_absorption": {"min": 1}})


def test_water_holding_rule_limits_water(simple_data):
    ds = Dataset(simple_data).view("Kano")
    # Both ingredients are needed: soybean alone may not exceed 80% of the protein blend.
    r = optimise(ds, ["soybean", "cowpea"], 100, targets={"protein": {"min": 5}}, water={"min": 0, "max": 90},
                 water_price=0, limit_water_to_holding=True)
    assert r["feasible"]
    x = r["recipe"]
    # water absorption: soybean 2.0 g/g, cowpea 1.5 g/g
    assert x["water"] <= 2.0 * x["soybean"] + 1.5 * x.get("cowpea", 0) + 1e-4
    assert x["water"] > 0  # free water is used up to the holding limit


def test_no_price_means_no_optimisation(tmp_path):
    ds = Dataset(build(tmp_path, [prop("soybean", "protein", 40, "Kano"),
                                  prop("soybean", "moisture", 10, "Kano", basis="as_is")]))
    r = optimise(ds, ["soybean"], 100, targets={"protein": {"min": 20}})
    assert r["status"] == "no_ingredients"


# ---------------------------------------------------------------- prediction
def _composition_samples(n, iid="soybean", with_target=True):
    rows = []
    for k in range(n):
        p, f = 38 + k * 0.8, 18 + (k % 4)
        rows += [prop(iid, "protein", p, "Kano", sample=f"s{k}"), prop(iid, "fat", f, "Kano", sample=f"s{k}")]
        if with_target:
            rows.append(prop(iid, "phytate", 900 + 20 * (p - 40) - 8 * (f - 20), "Kano", unit="mg/100 g",
                             sample=f"s{k}"))
    return rows


def test_prediction_never_overwrites_recorded_values(tmp_path):
    ds = Dataset(build(tmp_path, _composition_samples(10)))
    pr = Predictor(ds)
    assert pr.estimate_ingredient(ds, "soybean", "phytate")["estimated"] is False
    filled = pr.fill_missing(ds)
    assert filled.get("soybean").values["phytate"].status == "sourced"


def test_prediction_only_with_enough_same_ingredient_data(tmp_path):
    ds = Dataset(build(tmp_path, _composition_samples(10) + [prop("cowpea", "protein", 25, "Kano")]))
    pr = Predictor(ds)
    view = ds.view("Kano")
    del view.get("soybean").values["phytate"]
    est = pr.estimate_ingredient(view, "soybean", "phytate")
    assert est["estimated"] and est["n_training"] == 10 and est["q2_loo"] >= 0.5
    # a soybean model is never applied to cowpea
    assert pr.estimate_ingredient(view, "cowpea", "phytate")["estimated"] is False


def test_prediction_refused_with_too_little_data(tmp_path):
    ds = Dataset(build(tmp_path, _composition_samples(4)))
    view = ds.view("Kano")
    del view.get("soybean").values["phytate"]
    est = Predictor(ds).estimate_ingredient(view, "soybean", "phytate")
    assert est["estimated"] is False and "needed" in est["reason"]
