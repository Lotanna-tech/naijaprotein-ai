from flask import Blueprint, abort, jsonify, render_template, request

from ai_engine import products as PR
from ai_engine import properties as P
from ai_engine.dataset import LOCATION_STATUS_LABELS
from ai_engine.locations import NIGERIA_STATES, STATE_LABELS
from app.services.formulation_service import get_engine, get_view

# JSON API
ingredients = Blueprint("ingredients", __name__, url_prefix="/api/ingredients")

# HTML pages (Ingredient Intelligence)
catalogue = Blueprint("catalogue", __name__, url_prefix="/ingredients")


def _state_view():
    """The view for ?location=<state>, or 400 for anything that is not a Nigerian state."""
    location = request.args.get("location", "").strip() or None
    try:
        view, predictor, _ = get_view(location)
    except ValueError as err:
        abort(400, description=str(err))
    return view, predictor, view.location


def _product_arg():
    pid = request.args.get("product", "").strip()
    return PR.PRODUCTS.get(pid)


def _float_arg(name):
    value = request.args.get(name, "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        abort(400, description=f"'{name}' must be a number.")


def _common(dataset, location):
    base, _ = get_engine()
    return {
        "states": NIGERIA_STATES, "state_labels": STATE_LABELS, "location": location,
        "coverage": base.coverage(), "price_states": base.states_with_prices(),
        "summary": dataset.summary(), "location_labels": LOCATION_STATUS_LABELS,
        "products": PR.PRODUCTS,
    }


def _unavailable_reason(dataset, iid, key):
    refusals = {**getattr(dataset, "prediction_refusals", {})}
    loc = getattr(dataset, "location_refusals", {}).get((iid, key))
    if loc:
        return f"No measurements in {dataset.location}, and it cannot be estimated from other states: {loc}"
    return refusals.get((iid, key), "")


@ingredients.get("")
@ingredients.get("/")
def list_ingredients():
    dataset, _, _ = _state_view()
    items = dataset.list(q=request.args.get("q", "").strip() or None, min_protein=_float_arg("min_protein"))
    return jsonify({"ingredients": [i.summary() for i in items], "dataset": dataset.summary()})


@ingredients.get("/meta")
def meta():
    dataset, predictor = get_engine()
    return jsonify({
        "properties": [{"key": p.key, "label": p.label, "unit": p.unit, "group": p.group, "plain": p.plain,
                        "blend_rule": p.blend_rule, "optimisable": p.optimisable} for p in P.PROPERTIES.values()],
        "groups": P.GROUP_LABELS, "states": list(NIGERIA_STATES),
        "products": [PR.to_dict(p) for p in PR.PRODUCTS.values()],
        "product_targets": {pid: PR.default_targets(dataset, pid) for pid in PR.PRODUCTS},
        "dataset": dataset.summary(), "blend_models": predictor.status(),
        "sources": [s.to_dict() for s in dataset.sources.values()],
    })


@ingredients.get("/<ingredient_id>")
def get_ingredient(ingredient_id):
    dataset, _, location = _state_view()
    ing = dataset.get(ingredient_id)
    if ing is None:
        abort(404, description="Ingredient not found.")
    base, _ = get_engine()
    data = ing.to_dict(dataset.sources)
    data["location"] = location
    data["unavailable"] = [{"property": p.key, "reason": _unavailable_reason(dataset, ingredient_id, p.key)}
                           for p in P.PROPERTIES.values() if p.key not in ing.values]
    data["variation"] = {key: base.location_model(ingredient_id, key).summary()
                         for (iid, key) in base.records if iid == ingredient_id}
    return jsonify(data)


# ---------------------------------------------------------------- pages
@catalogue.get("/")
def index():
    dataset, _, location = _state_view()
    items = dataset.list(q=request.args.get("q", "").strip() or None, min_protein=_float_arg("min_protein"))
    return render_template("ingredients.html", items=items, args=request.args, **_common(dataset, location))


@catalogue.get("/compare")
def compare():
    dataset, _, location = _state_view()
    ids = [i for i in request.args.getlist("ids") if i in dataset.ingredients] or list(dataset.ingredients)
    items = [dataset.get(i) for i in ids]
    product = _product_arg()
    groups = P.by_group()
    # For the chosen product, which ingredient is highest / lowest on each relevant property.
    leaders = {}
    if product:
        for rel in product.relevant:
            vals = [(i.id, i.values[rel.property].value) for i in items if rel.property in i.values]
            if len(vals) > 1:
                pick = max if rel.direction == PR.HIGHER else min
                leaders[rel.property] = pick(vals, key=lambda v: v[1])[0]
    return render_template("compare.html", items=items, all_items=list(dataset.ingredients.values()),
                           selected=set(ids), groups=groups, group_labels=P.GROUP_LABELS, product=product,
                           leaders=leaders, **_common(dataset, location))


@catalogue.get("/<ingredient_id>")
def profile(ingredient_id):
    dataset, _, location = _state_view()
    ing = dataset.get(ingredient_id)
    if ing is None:
        abort(404)
    base, _ = get_engine()
    variation = {}
    for key in P.PROPERTIES:
        if (ingredient_id, key) in base.records:
            variation[key] = base.location_model(ingredient_id, key).summary()
    unverified = {key: recs for (iid, key), recs in base.unverified.items() if iid == ingredient_id}
    reasons = {p.key: _unavailable_reason(dataset, ingredient_id, p.key) for p in P.PROPERTIES.values()
               if p.key not in ing.values}
    return render_template("ingredient_profile.html", ing=ing, groups=P.by_group(), group_labels=P.GROUP_LABELS,
                           group_notes=P.GROUP_NOTES, sources=dataset.sources, variation=variation,
                           unverified=unverified, reasons=reasons,
                           unverified_prices=base.unverified_prices.get(ingredient_id, []),
                           **_common(dataset, location))
