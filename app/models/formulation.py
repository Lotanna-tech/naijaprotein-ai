import json
from datetime import datetime, timezone

from app.extensions import db


class Formulation(db.Model):
    """A saved optimisation or simulation run, with its inputs and results."""

    __tablename__ = "formulations"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    name = db.Column(db.String(150), nullable=False)
    kind = db.Column(db.String(20), nullable=False)  # "optimisation" or "simulation"
    request_json = db.Column(db.Text, nullable=False)
    result_json = db.Column(db.Text, nullable=False)
    feasible = db.Column(db.Boolean, nullable=False, default=True)
    protein = db.Column(db.Float)
    cost_per_kg = db.Column(db.Float)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    user = db.relationship("User", backref=db.backref("formulations", lazy="dynamic"))

    @property
    def request_data(self):
        return json.loads(self.request_json)

    @property
    def result_data(self):
        return json.loads(self.result_json)

    def to_dict(self, full=False):
        d = {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "feasible": self.feasible,
            "protein": self.protein,
            "cost_per_kg": self.cost_per_kg,
            "created_at": self.created_at.isoformat(),
        }
        if full:
            d["request"] = self.request_data
            d["result"] = self.result_data
        return d
