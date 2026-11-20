from flask import Blueprint, abort, render_template
from flask_login import current_user, login_required

from ai_engine import products as PR
from ai_engine import properties as P
from ai_engine.locations import NIGERIA_STATES, STATE_LABELS
from app.extensions import db
from app.models.formulation import Formulation
from app.services.formulation_service import get_engine

formulations = Blueprint(
    "formulations",
    __name__,
    url_prefix="/formulations"
)


def _page_config():
    """Everything the simulation and optimisation forms need, as plain data."""
    dataset, _ = get_engine()
    nutrition = [p for p in P.PROPERTIES.values() if p.group == "proximate"]
    functional = [p for p in P.PROPERTIES.values() if p.group == "functional"]
    return {
        "products": [PR.to_dict(p) for p in PR.PRODUCTS.values()],
        "product_targets": {pid: PR.default_targets(dataset, pid) for pid in PR.PRODUCTS},
        "states": [{"id": s, "label": STATE_LABELS[s],
                    "ingredients": sorted(dataset.coverage()[s]),
                    "prices": s in dataset.states_with_prices()} for s in NIGERIA_STATES],
        "ingredients": [{"id": i.id, "name": i.name, "form": i.form} for i in dataset.ingredients.values()],
        "nutrition": [{"key": p.key, "label": p.label, "unit": p.unit, "plain": p.plain} for p in nutrition],
        "functional": [{"key": p.key, "label": p.label, "unit": p.unit, "plain": p.plain} for p in functional],
        "blend_models": {m["property"]: m["usable"] for m in get_engine()[1].status()},
    }


@formulations.route("/")
@login_required
def index():
    """Formulation Optimisation page."""
    return render_template("formulation.html", config=_page_config())


@formulations.route("/simulate")
@login_required
def simulate():
    """Blend Simulation page."""
    return render_template("simulate.html", config=_page_config())


@formulations.route("/history")
@login_required
def history():
    items = current_user.formulations.order_by(Formulation.created_at.desc()).all()
    return render_template("history.html", items=items)


@formulations.route("/<int:formulation_id>")
@login_required
def detail(formulation_id):
    f = db.session.get(Formulation, formulation_id)
    if f is None or f.user_id != current_user.id:
        abort(404)
    return render_template("formulation_detail.html", f=f, data=f.result_data, req=f.request_data)
