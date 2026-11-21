import json
from functools import wraps

from flask import Blueprint, abort, jsonify, request
from flask_login import current_user

from ai_engine import application as AP
from ai_engine import products as PR
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


def _recipe(data, key, required):
    """{ingredient id: % of the protein blend}, validated for type only (the engine checks the rest)."""
    recipe = data.get(key)
    if recipe in (None, {}) and not required:
        return None
    if not isinstance(recipe, dict) or not all(isinstance(k, str) for k in recipe):
        raise InputError("Set how much of each plant protein you want to use, as % of the protein blend.")
    return recipe


def _importance(data):
    """Optional {property: high/medium/low/off}; the engine checks the property names and levels."""
    raw = data.get("importance")
    if raw in (None, {}):
        return None
    if not isinstance(raw, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in raw.items()):
        raise InputError("Importance settings must be an object: {property: high, medium, low or off}.")
    return raw


def _blend_summary(result):
    """(protein, cost per kg) of the blend in a result, for the saved-runs list."""
    blend = result.get("blend") or {}
    protein = (blend.get("protein") or {}).get("value")
    return protein, (blend.get("cost") or {}).get("value")


def _save(kind, name, req, result, explanation, protein, cost, feasible):
    name = (name or "").strip()[:150] or f"{kind.capitalize()} run"
    f = Formulation(user_id=current_user.id, name=name, kind=kind, request_json=json.dumps(req),
                    result_json=json.dumps({"result": result, "explanation": explanation}),
                    feasible=feasible, protein=protein, cost_per_kg=cost)
    db.session.add(f)
    db.session.commit()
    return f.id


@api.errorhandler(InputError)
@api.errorhandler(AP.ApplicationError)
def _bad_request(err):
    return jsonify({"error": str(err)}), 400


@api.post("/simulate")
@api_login_required
def simulate_blend():
    """What a stated blend of plant proteins does for the product, with its uncertainty."""
    data = _json_body()
    product = _product(data)
    recipe = _recipe(data, "recipe", required=True)
    batch_kg = _number(data, "batch_kg", "Batch size", minimum=0.001, maximum=MAX_BATCH_KG)
    importance = _importance(data)
    location = _location(data)
    dataset, _, _ = _view(location)
    result, explanation = AP.simulate(dataset, product, recipe, batch_kg, importance)
    body = {"result": result, "explanation": explanation}
    if data.get("save"):
        req = {"product": product, "location": location, "recipe": recipe, "batch_kg": batch_kg,
               "importance": importance}
        protein, cost = _blend_summary(result)
        body["saved_id"] = _save("simulation", data.get("name"), req, result, explanation, protein, cost, True)
    return jsonify(body)


@api.post("/optimise")
@api_login_required
def optimise_blend():
    """Rank the chosen plant proteins for the product and find the best blend.

    Optional `baseline`: a simulated blend ({id: % of the protein blend}) to compare with.
    """
    data = _json_body()
    product = _product(data)
    ids = data.get("ingredient_ids")
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise InputError("Choose the plant proteins you want to consider.")
    batch_kg = _number(data, "batch_kg", "Batch size", required=True, minimum=0.001, maximum=MAX_BATCH_KG)
    budget = _number(data, "budget", "Budget", minimum=0.01)
    baseline = _recipe(data, "baseline", required=False)
    importance = _importance(data)
    location = _location(data)
    dataset, _, _ = _view(location)
    result, explanation = AP.optimise(dataset, product, ids, batch_kg, budget, baseline, importance)
    body = {"result": result, "explanation": explanation}
    if data.get("save"):
        req = {"product": product, "location": location, "ingredient_ids": list(dict.fromkeys(ids)),
               "batch_kg": batch_kg, "budget": budget, "baseline": baseline, "importance": importance}
        protein, cost = _blend_summary(result)
        body["saved_id"] = _save("optimisation", data.get("name"), req, result, explanation, protein, cost,
                                 result["feasible"])
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
