"""Tests for the calculation engine: data rules, location model and prediction.
Expected values are worked out by hand."""

import pytest

from ai_engine import locations
from ai_engine.dataset import Dataset
from ai_engine.location_model import fit
from ai_engine.predictor import Predictor
from tests.conftest import build, prop


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
    assert est.confidence == "checked"
    # only 2 states: still estimated, but as a rough (indicative) estimate with a wider range
    rough = lagos.get("soybean").values["protein"]
    assert rough.status == "predicted" and rough.confidence == "indicative"
    assert rough.low < 43.0 < rough.high and (rough.high - rough.low) > (est.high - est.low)
    with pytest.raises(ValueError):
        ds.view("Sudan")


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
