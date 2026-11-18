from flask import Blueprint, render_template, abort
from flask_login import login_required, current_user
from app.models.user import User

admin = Blueprint("admin", __name__, url_prefix="/admin")


@admin.route("/")
@login_required
def dashboard():

    if not current_user.is_admin:
        abort(403)

    users = User.query.order_by(User.id.desc()).all()

    total_users = User.query.count()
    total_admins = User.query.filter_by(is_admin=True).count()

    return render_template(
        "admin.html",
        users=users,
        total_users=total_users,
        total_admins=total_admins
    )