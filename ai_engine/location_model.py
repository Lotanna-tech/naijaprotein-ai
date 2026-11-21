"""Location variation of one ingredient property across Nigerian states.

Model (fitted separately for each ingredient and each property):

    y_{r,l} = mu + u_l + e_r          u_l ~ N(0, tau^2),  e_r ~ N(0, sigma^2)

y_{r,l} is a verified record r from state l. mu is the national level of the
ingredient (alpha_i + beta_0 in the project notation, since each ingredient
is fitted separately), u_l is the state effect and e_r the variation between
samples within a state (variety, lot, laboratory). The variance components
are estimated by the method of moments (one-way random-effects ANOVA):

    sigma^2 = pooled within-state variance (states with 2+ records)
    tau^2   = max(0, var(state means) - mean(sigma^2 / n_l))

State effects are shrunk towards mu (empirical Bayes / partial pooling):

    u_l_hat = B_l * (ybar_l - mu),   B_l = tau^2 / (tau^2 + sigma^2 / n_l)

so a state with one record is not over-trusted. Environmental covariates
(soil, rainfall, temperature) and variety/processing terms are not included
because no such data is held; they can be added when it is.

States WITHOUT records get mu, the value expected for a new state. Its
uncertainty is the predictive standard deviation of a new state's mean,
sqrt(tau^2 + Var(mu)). The estimate is checked by leave-one-state-out
validation: each state with data is held out in turn, mu is refitted on the
others and compared with the held-out state's mean.

- "checked": at least MIN_STATES states and a relative RMSE no larger than
  MAX_RELATIVE_ERROR (`predict_unseen`).
- "indicative" (`estimate_unseen`): fewer states, or larger differences
  between states. The value is still mu, but its standard deviation is the
  largest of the model's own predictive SD, the leave-one-state-out error and
  the typical relative error of the same property in other ingredients
  (borrowing strength across ingredients when this one has few states). The
  interval is therefore wide when the evidence is thin.

Measured values for a state are always used as they are; the model never
replaces them.
"""

import math
from dataclasses import dataclass, field
from statistics import mean

import numpy as np

MIN_STATES = 3
MAX_RELATIVE_ERROR = 0.10
Z95 = 1.96

CHECKED = "checked"
INDICATIVE = "indicative"

UNCERTAINTY_LABELS = {
    "model": "spread between the states with data",
    "loso": "error when each state with data is predicted from the others",
    "other_ingredients": "typical state-to-state error of this property in the other ingredients",
    "largest_seen": "largest state-to-state error seen for any property (this one has not been checked "
                    "between states)",
    "resolution": "measurement step of the test (values are reported in fixed steps, so identical results "
                  "do not mean identical samples)",
}


def _fit_components(groups):
    """groups: {state: [values]} -> (mu, tau2, sigma2 or None, var_mu)."""
    states = list(groups)
    n = np.array([len(groups[s]) for s in states], dtype=float)
    ybar = np.array([mean(groups[s]) for s in states])
    within = [(len(v) - 1, np.var(v, ddof=1)) for v in groups.values() if len(v) > 1]
    sigma2 = (sum(df * var for df, var in within) / sum(df for df, _ in within)) if within else None
    if len(states) > 1:
        between = float(np.var(ybar, ddof=1))
        tau2 = max(0.0, between - (float(np.mean(sigma2 / n)) if sigma2 is not None else 0.0))
    else:
        tau2 = 0.0
    if sigma2 is not None and (tau2 > 0 or sigma2 > 0):
        w = 1.0 / (tau2 + sigma2 / n)
    else:
        w = np.ones_like(n)
    mu = float(np.sum(w * ybar) / np.sum(w))
    var_mu = float(1.0 / np.sum(w)) if sigma2 is not None else float(np.var(ybar, ddof=1) / len(ybar)) if len(ybar) > 1 else 0.0
    return mu, tau2, sigma2, var_mu


def _leave_one_state_out(groups):
    """Predict each state's mean from the other states: (rmse, relative rmse)."""
    errors = []
    for s in groups:
        rest = {k: v for k, v in groups.items() if k != s}
        errors.append(mean(groups[s]) - _fit_components(rest)[0])
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    mu = _fit_components(groups)[0]
    return rmse, (rmse / abs(mu) if mu else float("inf"))


@dataclass
class LocationModel:
    key: str
    groups: dict                      # state -> [values]
    mu: float = None
    tau2: float = None
    sigma2: float = None
    var_mu: float = None
    state_effects: dict = field(default_factory=dict)   # state -> shrunk effect
    validation: dict = field(default_factory=dict)

    @property
    def validated(self):
        return bool(self.validation.get("validated"))

    def summary(self, prior_rel_sd=None, prior_basis="other_ingredients", floor_sd=0.0):
        icc = None
        if self.tau2 is not None and self.sigma2 is not None and (self.tau2 + self.sigma2) > 0:
            icc = self.tau2 / (self.tau2 + self.sigma2)
        unseen = self.estimate_unseen(prior_rel_sd, prior_basis, floor_sd)
        return {
            "national_mean": None if self.mu is None else round(self.mu, 4),
            "between_state_sd": None if self.tau2 is None else round(self.tau2 ** 0.5, 4),
            "within_state_sd": None if self.sigma2 is None else round(self.sigma2 ** 0.5, 4),
            "share_of_variation_from_location": None if icc is None else round(icc, 3),
            "states": {s: {"n": len(v), "mean": round(mean(v), 4),
                           "state_effect": round(self.state_effects.get(s, 0.0), 4)}
                       for s, v in sorted(self.groups.items())},
            "validation": self.validation,
            "unseen_state": None if unseen is None else {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in unseen.items()},
            "covariates": "State effects only. No soil, climate or variety covariates are included because "
                          "that data is not held.",
        }

    def predict_unseen(self):
        """Estimate for a state without records, or None if the model is not validated."""
        if not self.validated:
            return None
        sd = max((self.tau2 + self.var_mu) ** 0.5, self.validation["loso_rmse"])
        return {"value": self.mu, "low": self.mu - Z95 * sd, "high": self.mu + Z95 * sd}

    def estimate_unseen(self, prior_rel_sd=None, prior_basis="other_ingredients", floor_sd=0.0):
        """Estimate for a state without records, graded 'checked' or 'indicative'.

        prior_rel_sd: relative error borrowed from other ingredients or properties
        (prior_basis names which, see UNCERTAINTY_LABELS). floor_sd: smallest credible
        standard deviation, e.g. half the step in which a test reports its results.
        Returns None only when there are no records, or when nothing indicates how much
        the property varies between states (one state, one record, nothing to learn from).
        """
        if self.mu is None:
            return None
        candidates = {"model": math.sqrt(max(self.tau2 + self.var_mu, 0.0))}
        if self.validation.get("loso_rmse") is not None:
            candidates["loso"] = self.validation["loso_rmse"]
        # A checked model needs no help from other ingredients; it is validated on its own states.
        if prior_rel_sd and not self.validated:
            candidates[prior_basis] = prior_rel_sd * abs(self.mu)
        if floor_sd and floor_sd > 0:
            candidates["resolution"] = floor_sd
        basis, sd = max(candidates.items(), key=lambda kv: kv[1])
        if sd <= 0:
            return None
        return {"value": self.mu, "low": max(0.0, self.mu - Z95 * sd), "high": self.mu + Z95 * sd, "sd": sd,
                "relative_sd": sd / abs(self.mu) if self.mu else None,
                "confidence": CHECKED if self.validated else INDICATIVE,
                "uncertainty_from": basis, "reason": self.validation.get("reason", "")}


def fit(key, groups):
    """Fit the location model for one ingredient property. groups: {state: [values]}."""
    model = LocationModel(key, groups)
    if not groups:
        model.validation = {"validated": False, "n_states": 0, "reason": "No verified records."}
        return model
    model.mu, model.tau2, model.sigma2, model.var_mu = _fit_components(groups)
    for s, v in groups.items():
        n = len(v)
        if model.sigma2 is None:
            b = 1.0
        else:
            denom = model.tau2 + model.sigma2 / n
            b = model.tau2 / denom if denom > 0 else 0.0
        model.state_effects[s] = b * (mean(v) - model.mu)

    n_states = len(groups)
    if n_states < MIN_STATES:
        model.validation = {
            "validated": False, "n_states": n_states,
            "reason": f"Verified data from {n_states} state(s); at least {MIN_STATES} are needed to check how well "
                      "the national value represents a state without data.",
        }
        if n_states >= 2:
            rmse, rel = _leave_one_state_out(groups)
            model.validation.update({"loso_rmse": round(rmse, 4), "relative_error": round(rel, 4)})
        return model
    rmse, rel = _leave_one_state_out(groups)
    ok = rel <= MAX_RELATIVE_ERROR
    model.validation = {
        "validated": ok, "n_states": n_states, "loso_rmse": round(rmse, 4), "relative_error": round(rel, 4),
        "max_relative_error": MAX_RELATIVE_ERROR,
        "reason": ("" if ok else
                   f"Leaving out one state at a time, the national value missed the state's value by "
                   f"{rel:.0%} on average (limit {MAX_RELATIVE_ERROR:.0%}); state differences are too large "
                   "for a checked estimate."),
    }
    return model
