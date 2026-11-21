from flask import Blueprint, abort, render_template
from flask_login import current_user, login_required

from ai_engine import application as AP
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
    coverage = dataset.coverage()
    price_states = dataset.states_with_prices()
    return {
        "products": [{**PR.to_dict(p), "criteria": AP.preview(dataset, p)} for p in PR.PRODUCTS.values()],
        "states": [{"id": s, "label": STATE_LABELS[s], "ingredients": sorted(coverage[s]),
                    "prices": s in price_states} for s in NIGERIA_STATES],
        "ingredients": [{"id": i.id, "name": i.name, "form": i.form} for i in dataset.ingredients.values()],
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
    dataset, _ = get_engine()
    names = {i.id: i.name for i in dataset.ingredients.values()}
    labels = {k: p.label for k, p in P.PROPERTIES.items()}
    product = PR.PRODUCTS.get(f.request_data.get("product"))
    return render_template("formulation_detail.html", f=f, data=f.result_data, req=f.request_data, names=names,
                           labels=labels, product_name=product.name if product else None)
