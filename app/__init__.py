from flask import Flask, render_template
from app.config import Config
from app.extensions import db, login_manager, migrate
from app.routes.admin import admin
def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    migrate.init_app(app, db)



    db.init_app(app)

    login_manager.init_app(app)

    from app.models.user import User

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    from app.routes.auth import auth
    from app.routes.ingredients import ingredients
    from app.routes.formulations import formulations

    app.register_blueprint(auth)
    app.register_blueprint(ingredients)
    app.register_blueprint(formulations)
    app.register_blueprint(admin)

    @app.route("/")
    def dashboard():
        return render_template("dashboard.html")

    return app