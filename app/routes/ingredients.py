from flask import Blueprint

ingredients = Blueprint("ingredients", __name__, url_prefix="/api/ingredients")