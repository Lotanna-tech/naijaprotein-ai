from flask import Flask, render_template
from werkzeug.middleware.proxy_fix import ProxyFix

from app.config import Config
from app.extensions import db, login_manager, migrate
from app.routes.admin import admin


LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}


class ProxyPrefixMiddleware:
    """Mount the app under a URL prefix for requests coming through a proxy.

    Some proxies (such as code-server's /proxy/<port>/) strip the prefix and
    do not send X-Forwarded-Prefix, so ProxyFix alone cannot know about it.
    For those requests this sets SCRIPT_NAME so url_for() and static URLs
    include the prefix. Direct requests to localhost are left untouched.
    """

    def __init__(self, wsgi_app, prefix):
        self.wsgi_app = wsgi_app
        prefix = (prefix or "").strip("/")
        self.prefix = f"/{prefix}" if prefix else ""

    def _is_proxied(self, environ):
        host = environ.get("HTTP_HOST", "").rsplit(":", 1)[0]
        return host not in LOCAL_HOSTS or "HTTP_X_FORWARDED_FOR" in environ

    def __call__(self, environ, start_response):
        # SCRIPT_NAME already set (e.g. by ProxyFix from X-Forwarded-Prefix).
        if self.prefix and not environ.get("SCRIPT_NAME") and self._is_proxied(environ):
            environ["SCRIPT_NAME"] = self.prefix
            # If the proxy did not strip the prefix, remove it from the path.
            path = environ.get("PATH_INFO", "")
            if path == self.prefix or path.startswith(self.prefix + "/"):
                environ["PATH_INFO"] = path[len(self.prefix):] or "/"
        return self.wsgi_app(environ, start_response)


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # ProxyFix runs first and honours X-Forwarded-For/Proto/Host/Prefix.
    # ProxyPrefixMiddleware is the fallback when no X-Forwarded-Prefix is sent.
    app.wsgi_app = ProxyFix(
        ProxyPrefixMiddleware(app.wsgi_app, app.config.get("URL_PREFIX")),
        x_for=1,
        x_proto=1,
        x_host=1,
        x_prefix=1,
    )

    migrate.init_app(app, db)

    db.init_app(app)

    login_manager.init_app(app)

    from app.models.user import User

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    from app.routes.auth import auth
    from app.routes.ingredients import ingredients, catalogue
    from app.routes.formulations import formulations
    from app.routes.api import api

    app.register_blueprint(auth)
    app.register_blueprint(ingredients)
    app.register_blueprint(catalogue)
    app.register_blueprint(formulations)
    app.register_blueprint(api)
    app.register_blueprint(admin)

    from app.services.formulation_service import get_engine

    @app.context_processor
    def inject_dataset_status():
        # Lets every page show the "sample data" banner while values are unverified.
        try:
            return {"dataset_summary": get_engine()[0].summary()}
        except Exception:  # never break page rendering over the banner
            app.logger.exception("Could not load dataset summary")
            return {"dataset_summary": None}

    @app.route("/")
    def dashboard():
        return render_template("dashboard.html")

    @app.cli.command("validate-data")
    def validate_data():
        """Check the CSV dataset for unit, source and status problems."""
        from ai_engine.dataset import Dataset
        ds = Dataset(app.config["DATA_DIR"], app.config.get("INGREDIENT_SCOPE"))
        s = ds.summary()
        print(f"{s['ingredients']} ingredients, {s['sources']} sources, {s['experiments']} blend experiments")
        print(f"{s['verified_records']} verified values used; {s['unverified_records']} unverified values kept "
              f"but not used; {s['verified_price_lots']} verified price lots")
        print("States with data:", ", ".join(s["states_with_data"]) or "none")
        print("States with prices:", ", ".join(s["states_with_prices"]) or "none")
        for note in ds.excluded:
            print("  excluded:", note)
        if ds.issues:
            print(f"{len(ds.issues)} issue(s):")
            for issue in ds.issues:
                print("  -", issue)
            raise SystemExit(1)
        print("No issues found.")

    return app
