import json
from functools import wraps

from flask import Blueprint, abort, jsonify, request
from flask_login import current_user

from ai_engine import products as PR
from ai_engine import properties as P
from ai_engine.blend import BlendError, OIL, WATER, simulate
from ai_engine.explain import explain_optimisation, explain_simulation
from ai_engine.optimiser import OptimiseError, optimise
from app.extensions import db
from app.models.formulation import Formulation
from app.services.formulation_service import get_view

api = Blueprint("api", __name__, url_prefix="/api")

MAX_BATCH_KG = 1_000_000


def api_login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({"error": "Please log in first."}), 401
        return view(*args, **kwargs)
    return wrapper


class InputError(ValueError):
    pass


def _number(data, key, label=None, required=False, minimum=None, maximum=None):
    label = label or key
    value = data.get(key)
    if value in (None, ""):
        if required:
            raise InputError(f"{label} is required.")
        return None
    if isinstance(value, bool):
        raise InputError(f"{label} must be a number.")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise InputError(f"{label} must be a number.")
    if minimum is not None and value < minimum:
        raise InputError(f"{label} must be at least {minimum:g}.")
    if maximum is not None and value > maximum:
        raise InputError(f"{label} must be at most {maximum:g}.")
    return value


def _json_body():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise InputError("Request body must be a JSON object.")
    return data


def _product(data):
    pid = data.get("product")
    if pid not in PR.PRODUCTS:
        raise InputError("Choose the type of meat product: " + ", ".join(p.name for p in PR.PRODUCTS.values()) + ".")
    return pid


def _location(data):
    loc = data.get("location")
    if loc in (None, ""):
        return None
    if not isinstance(loc, str) or len(loc) > 60:
        raise InputError("Location must be a Nigerian state or the FCT.")
    return loc.strip()


def _view(location):
    try:
        return get_view(location)
    except ValueError as err:
        raise InputError(str(err))


def _targets(data):
    raw = data.get("targets") or {}
    if not isinstance(raw, dict):
        raise InputError("Targets must be an object keyed by property.")
    out = {}
    for key, t in raw.items():
        if key not in P.PROPERTIES or not isinstance(t, dict):
            raise InputError(f"Unknown target '{key}'.")
        label = P.get(key).label
        mn = _number(t, "min", f"Minimum {label.lower()}", minimum=0)
        mx = _number(t, "max", f"Maximum {label.lower()}", minimum=0)
        if mn is None and mx is None:
            continue
        if mn is not None and mx is not None and mn > mx:
            raise InputError(f"The minimum for {label.lower()} is above the maximum.")
        out[key] = {"min": mn, "max": mx}
    return out


def _range(data, key, label):
    raw = data.get(key)
    if raw in (None, False):
        return None
    if not isinstance(raw, dict):
        raise InputError(f"{label} limits must be an object with min and max.")
    return {"min": _number(raw, "min", f"Minimum {label.lower()}", minimum=0, maximum=100),
            "max": _number(raw, "max", f"Maximum {label.lower()}", minimum=0, maximum=100)}


def _prices(data):
    return (_number(data, "water_price", "Water price", minimum=0),
            _number(data, "oil_price", "Oil price", minimum=0))


def _save(kind, name, req, result, explanation, protein, cost, feasible):
    name = (name or "").strip()[:150] or f"{kind.capitalize()} run"
    f = Formulation(user_id=current_user.id, name=name, kind=kind, request_json=json.dumps(req),
                    result_json=json.dumps({"result": result, "explanation": explanation}),
                    feasible=feasible, protein=protein, cost_per_kg=cost)
    db.session.add(f)
    db.session.commit()
    return f.id


@api.errorhandler(InputError)
@api.errorhandler(BlendError)
@api.errorhandler(OptimiseError)
def _bad_request(err):
    return jsonify({"error": str(err)}), 400


@api.post("/simulate")
@api_login_required
def simulate_blend():
    data = _json_body()
    product = _product(data)
    recipe = data.get("recipe")
    if not isinstance(recipe, dict):
        raise InputError("Set how much of each ingredient you want to use.")
    batch_kg = _number(data, "batch_kg", "Batch size", minimum=0.001, maximum=MAX_BATCH_KG)
    targets = _targets(data)
    water_price, oil_price = _prices(data)
    location = _location(data)
    dataset, predictor, _ = _view(location)
    result = simulate(dataset, predictor, recipe, product, batch_kg, targets, water_price, oil_price)
    explanation = explain_simulation(dataset, result)
    body = {"result": result, "explanation": explanation}
    if data.get("save"):
        prot = result["mix"]["protein"]
        req = {"product": product, "location": location, "recipe": recipe, "batch_kg": batch_kg,
               "targets": targets, "water_price": water_price, "oil_price": oil_price}
        body["saved_id"] = _save("simulation", data.get("name"), req, result, explanation, prot.get("value"),
                                 result["cost"].get("value"), True)
    return jsonify(body)


@api.post("/optimise")
@api_login_required
def optimise_blend():
    data = _json_body()
    product = _product(data)
    ids = data.get("ingredient_ids")
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise InputError("Choose the protein ingredients you want to use.")
    raw_limits = data.get("limits") or {}
    if not isinstance(raw_limits, dict):
        raise InputError("Ingredient limits must be an object keyed by ingredient.")
    limits = {iid: {"min": _number(lim, "min", "Minimum share", minimum=0, maximum=100),
                    "max": _number(lim, "max", "Maximum share", minimum=0, maximum=100)}
              for iid, lim in raw_limits.items() if isinstance(lim, dict)}
    targets = _targets(data)
    lp_targets = {k: v for k, v in targets.items() if P.get(k).optimisable}
    water_price, oil_price = _prices(data)
    req = {
        "ingredient_ids": list(dict.fromkeys(ids)),
        "batch_kg": _number(data, "batch_kg", "Batch size", required=True, minimum=0.001, maximum=MAX_BATCH_KG),
        "targets": lp_targets, "limits": limits,
        "water": _range(data, "water", "Added water"), "oil": _range(data, "oil", "Edible oil"),
        "budget": _number(data, "budget", "Budget", minimum=0.01),
        "water_price": water_price, "oil_price": oil_price,
        "limit_water_to_holding": bool(data.get("limit_water_to_holding")),
    }
    location = _location(data)
    dataset, predictor, national = _view(location)
    result = optimise(dataset, **req)
    result["location"] = location
    profile = None
    if result["feasible"]:
        profile = simulate(dataset, predictor, {k: v * 100 for k, v in result["recipe"].items()}, product,
                           req["batch_kg"], targets, water_price, oil_price)
    explanation = explain_optimisation(dataset, predictor, {**req, "product": product}, result, profile, national)
    body = {"result": result, "profile": profile, "explanation": explanation}
    if data.get("save"):
        body["saved_id"] = _save("optimisation", data.get("name"),
                                 {**req, "targets": targets, "product": product, "location": location},
                                 {**result, "profile": profile}, explanation,
                                 profile["mix"]["protein"].get("value") if profile else None,
                                 result.get("cost_per_kg"), result["feasible"])
    return jsonify(body)


@api.get("/formulations")
@api_login_required
def list_formulations():
    items = current_user.formulations.order_by(Formulation.created_at.desc()).all()
    return jsonify({"formulations": [f.to_dict() for f in items]})


@api.get("/formulations/<int:formulation_id>")
@api_login_required
def get_formulation(formulation_id):
    f = db.session.get(Formulation, formulation_id)
    if f is None or f.user_id != current_user.id:
        abort(404)
    return jsonify(f.to_dict(full=True))


@api.delete("/formulations/<int:formulation_id>")
@api_login_required
def delete_formulation(formulation_id):
    f = db.session.get(Formulation, formulation_id)
    if f is None or f.user_id != current_user.id:
        abort(404)
    db.session.delete(f)
    db.session.commit()
    return jsonify({"deleted": formulation_id})
