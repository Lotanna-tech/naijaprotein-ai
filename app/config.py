import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


class Config:
    # Folder holding the shared ingredient dataset CSV files.
    DATA_DIR = os.environ.get("DATA_DIR") or os.path.join(BASE_DIR, "data")

    # Initial scope: five Nigerian protein raw materials, four plant-based meat
    # product categories (ai_engine/products.py), Nigeria only (ai_engine/locations.py).
    # Ingredients outside this list are rejected by the dataset loader.
    INGREDIENT_SCOPE = ("cowpea", "bambara_groundnut", "defatted_groundnut", "african_yam_bean", "soybean")

    SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-secret-key"
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL") or "sqlite:///naijaprotein.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Path prefix used by the reverse proxy (e.g. code-server's /proxy/5000/).
    # Applied only to requests that arrive through the proxy; direct access
    # on 127.0.0.1/localhost is unaffected. Set URL_PREFIX="" to disable.
    URL_PREFIX = os.environ.get("URL_PREFIX", "/proxy/5000")

    ADMIN_EMAILS = {
    "latechsurge@gmail.com",
    "bernardtech.co1@gmail.com",
    "ebukaakpuoye@gmail.com",
    "nkananglawrence@gmail.com",
}