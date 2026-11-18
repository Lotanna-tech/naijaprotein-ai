
from flask import Blueprint, render_template
from flask_login import login_required, current_user

formulations = Blueprint(
    "formulations",
    __name__,
    url_prefix="/formulations"
)


@formulations.route("/")
@login_required
def index():
    return render_template("formulation.html")
