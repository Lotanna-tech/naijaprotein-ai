from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

from app.config import Config
from app.extensions import db, login_manager, migrate
from app.routes.admin import admin


LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}

ERROR_PAGES = {
    400: ("Something is not right", "That request could not be handled. Please check what you entered."),
    403: ("No access", "You do not have permission to see this page."),
    404: ("Page not found", "We could not find that page."),
    405: ("Not allowed", "That action is not available here."),
    500: ("Something went wrong", "Please try again in a moment."),
}


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

    @app.errorhandler(HTTPException)
    def handle_http_error(err):
        """Friendly error pages; JSON for the API."""
        custom = err.description != type(err).description
        if request.path.startswith("/api/"):
            return jsonify({"error": err.description if custom else ERROR_PAGES.get(err.code, ERROR_PAGES[500])[1]}), \
                err.code
        title, message = ERROR_PAGES.get(err.code, ERROR_PAGES[500])
        if custom and err.code == 400:
            message = err.description
        return render_template("error.html", title=title, message=message), err.code

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
        pm = ds.price_model().summary()
        if pm["usable"]:
            pct = lambda v: "not checked" if v is None else f"{v:.0%}"
            print(f"Price model: {pm['n_lots']} verified lots; typical error {pct(pm['unsurveyed_state_error'])} for "
                  f"a state without a price survey, {pct(pm['missing_ingredient_error'])} for an ingredient "
                  "missing in a surveyed state")
        else:
            print("Price model: not usable -", pm["reason"])
        coverage = ds.state_coverage()
        complete = [st for st, c in coverage.items() if c and all(e["complete"] for e in c.values())]
        print(f"States where every ingredient has protein, moisture and a price (measured or estimated): "
              f"{len(complete)} of {len(coverage)}")
        for st, c in coverage.items():
            gaps = {i: e["missing"] for i, e in c.items() if not e["complete"]}
            if gaps:
                print(f"  {st}: " + "; ".join(f"{i} has no {', '.join(m)}" for i, m in gaps.items()))
        for note in ds.excluded:
            print("  excluded:", note)
        if ds.issues:
            print(f"{len(ds.issues)} issue(s):")
            for issue in ds.issues:
                print("  -", issue)
            raise SystemExit(1)
        print("No issues found.")

    @app.cli.command("evaluate-ranking")
    def evaluate_ranking():
        """Backtest the ranking for states without data, on states that have their own measurements."""
        from statistics import mean
        from ai_engine.application import backtest
        from ai_engine.dataset import Dataset
        ds = Dataset(app.config["DATA_DIR"], app.config.get("INGREDIENT_SCOPE"))
        rows = backtest(ds)
        if not rows:
            print("No state has its own measurements of every scored property for 3 or more ingredients.")
            return
        print("Each state's own measurements are hidden, its values are estimated from the other states, and the "
              "ranking is compared with the one from its own data.")
        for r in rows:
            print(f"  {r['product']:8s} {r['state']:10s} {r['n_ingredients']} ingredients  "
                  f"top pick {'same' if r['top_agrees'] else 'DIFFERENT'} ({r['own_top']})  "
                  f"Kendall tau {r['kendall_tau']:.2f}  estimated chance the true top ranks first "
                  f"{r['p_first_of_true_top']:.0%}")
        print(f"Same top pick in {sum(r['top_agrees'] for r in rows)} of {len(rows)} tests; mean Kendall tau "
              f"{mean(r['kendall_tau'] for r in rows):.2f}.")

    return app
