import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-secret-key"
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL") or "sqlite:///naijaprotein.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    ADMIN_EMAILS = {
    "latechsurge@gmail.com",
    "bernardtech.co1@gmail.com",
    "ebukaakpuoye@gmail.com",
    "nkananglawrence@gmail.com",
}