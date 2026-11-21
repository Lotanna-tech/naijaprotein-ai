from urllib.parse import urlparse

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from app.extensions import db
from app.models.user import User

auth = Blueprint("auth", __name__)


def _safe_next(target):
    """The page to return to after logging in, only if it is a path on this site."""
    if not target or "\\" in target or not target.startswith("/") or target.startswith("//"):
        return None
    parsed = urlparse(target)
    return None if parsed.scheme or parsed.netloc else target


def _apply_admin_role(user):
    """Users whose email is listed in ADMIN_EMAILS get the admin role (never removed here)."""
    if user.email in current_app.config.get("ADMIN_EMAILS", ()) and not user.is_admin:
        user.is_admin = True
        return True
    return False


@auth.route("/register", methods=["GET", "POST"])
def register():

    # Already logged in? Go to dashboard.
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    next_url = _safe_next(request.args.get("next"))

    if request.method == "POST":

        company_name = request.form.get("company_name", "").strip()
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not all([company_name, full_name, email, password]):
            flash("Please fill in all fields.", "error")
            return redirect(url_for("auth.register", next=next_url))

        if len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
            return redirect(url_for("auth.register", next=next_url))

        existing_user = User.query.filter_by(email=email).first()

        if existing_user:
            flash("An account with this email already exists.", "error")
            return redirect(url_for("auth.register", next=next_url))

        new_user = User(
            company_name=company_name,
            full_name=full_name,
            email=email
        )

        new_user.set_password(password)
        _apply_admin_role(new_user)

        db.session.add(new_user)
        db.session.commit()

        flash("Account created. Please log in.", "success")
        return redirect(url_for("auth.login", next=next_url))

    return render_template("register.html")


@auth.route("/login", methods=["GET", "POST"])
def login():

    # Already logged in? No need to log in again.
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            if _apply_admin_role(user):
                db.session.commit()
            login_user(user)
            return redirect(_safe_next(request.args.get("next")) or url_for("dashboard"))

        flash("Invalid email or password.", "error")

    return render_template("login.html")


@auth.route("/logout")
@login_required
def logout():

    logout_user()

    flash("You have been logged out.", "success")
    return redirect(url_for("auth.login"))
