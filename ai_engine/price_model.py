"""Prices for a state, or an ingredient in a state, without a verified price lot.

Fitted on verified price lots only (status measured or sourced).

1. Ingredient and state price levels, by the Country-Product-Dummy (CPD)
   model that the World Bank's International Comparison Program uses to
   compare price levels between places from an incomplete price table:

       ln p_l = alpha_i + lambda_s + e_l      (lot l of ingredient i sold in state s)

   alpha_i is the ingredient's price level in an average surveyed state and
   lambda_s how much more or less everything costs in state s (the lambda of
   the surveyed states sum to 0). Least squares on log prices.

2. A state without a price survey gets its price level from the surveyed
   states by inverse-distance weighting (Shepard), using the distance between
   state capitals:

       lambda_s = sum_m w_m lambda_m / sum_m w_m,      w_m = 1 / d(s, m)^2

The estimate is exp(alpha_i + lambda_s), a median (geometric-mean) price.

The uncertainty is measured on the data, not assumed:
- missing ingredient in a surveyed state: each surveyed (ingredient, state)
  pair is left out in turn, the model refitted and the pair predicted;
- state without a survey: each surveyed state is left out in turn and its
  prices predicted from the other states.
The root-mean-square log error is the standard deviation of the log price;
the 95% interval is p * exp(-1.96 sd) to p * exp(1.96 sd). The estimate is
"checked" when the error is at most 10% over at least 3 held-out cases,
otherwise "indicative".

Stock is never estimated: it belongs to a lot, not to a place.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from ai_engine.locations import distance_km
from ai_engine.location_model import CHECKED, INDICATIVE, MAX_RELATIVE_ERROR, Z95

IDW_POWER = 2
MIN_TESTS_FOR_CHECKED = 3

METHOD = ("Price levels by ingredient and by state (Country-Product-Dummy model, as used by the World Bank's "
          "International Comparison Program), fitted on verified price lots")


def _cpd(lots):
    """Least-squares CPD fit. lots: [(iid, state, price)]. Returns (alpha, lam, resid_sd) or None."""
    ings = sorted({i for i, _, _ in lots})
    states = sorted({s for _, s, _ in lots})
    X = np.zeros((len(lots), len(ings) + len(states) - 1))
    y = np.array([math.log(p) for _, _, p in lots])
    for r, (i, s, _) in enumerate(lots):
        X[r, ings.index(i)] = 1.0
        k = states.index(s)
        if k:
            X[r, len(ings) + k - 1] = 1.0
    # Ingredients and states must be linked by shared lots, otherwise levels are not identified.
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    lam = np.concatenate([[0.0], beta[len(ings):]])
    shift = float(lam.mean())
    lam = lam - shift
    alpha = beta[:len(ings)] + shift
    resid = y - X @ beta
    df = len(lots) - X.shape[1]
    resid_sd = float(np.sqrt(resid @ resid / df)) if df > 0 else None
    return dict(zip(ings, map(float, alpha))), dict(zip(states, map(float, lam))), resid_sd


def _idw(lam, state):
    """Inverse-distance-weighted state price level and the weights used."""
    w = {m: 1.0 / distance_km(state, m) ** IDW_POWER for m in lam}
    total = sum(w.values())
    weights = {m: w[m] / total for m in sorted(w, key=w.get, reverse=True)}
    return sum(weights[m] * lam[m] for m in weights), weights


def _cell_means(lots):
    cells = {}
    for i, s, p in lots:
        cells.setdefault((i, s), []).append(math.log(p))
    return {k: float(np.mean(v)) for k, v in cells.items()}


def _rmse(errors):
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else None


@dataclass
class PriceModel:
    lots: list                                     # [(iid, state, price, date)]
    alpha: dict = field(default_factory=dict)      # ingredient -> log price level
    lam: dict = field(default_factory=dict)        # surveyed state -> log state level
    resid_sd: float = None
    cell_rmse: float = None                        # error for a missing ingredient in a surveyed state
    cell_tests: int = 0
    state_rmse: float = None                       # error for a state without a survey
    state_tests: int = 0
    usable: bool = False
    reason: str = ""

    @property
    def period(self):
        dates = sorted(d for *_, d in self.lots if d)
        return (dates[0], dates[-1]) if dates else ("", "")

    def summary(self):
        return {
            "usable": self.usable, "reason": self.reason, "n_lots": len(self.lots),
            "surveyed_states": sorted(self.lam),
            "ingredient_levels_ngn": {i: round(math.exp(a), 2) for i, a in self.alpha.items()},
            "state_levels_pct": {s: round((math.exp(v) - 1) * 100, 1) for s, v in self.lam.items()},
            "residual_sd_log": None if self.resid_sd is None else round(self.resid_sd, 4),
            "missing_ingredient_error": None if self.cell_rmse is None else round(math.exp(self.cell_rmse) - 1, 4),
            "missing_ingredient_tests": self.cell_tests,
            "unsurveyed_state_error": None if self.state_rmse is None else round(math.exp(self.state_rmse) - 1, 4),
            "unsurveyed_state_tests": self.state_tests, "period": list(self.period),
        }

    def refusal(self, iid, state):
        if not self.usable:
            return self.reason
        if iid not in self.alpha:
            return "This ingredient has no verified price lot in any state."
        if state not in self.lam and self.state_rmse is None:
            return ("Verified prices from at least 2 states are needed to estimate prices for a state without a "
                    "price survey.")
        return "Not enough verified lots to check a price estimate."

    def estimate(self, iid, state):
        """Estimated price of an ingredient in a state, or None (see `refusal`)."""
        if not self.usable or iid not in self.alpha:
            return None
        ingredient_states = sorted({s for i, s, _, _ in self.lots if i == iid})
        if state in self.lam:
            level, weights, case = self.lam[state], {state: 1.0}, "missing_ingredient"
            sd, tests = self.cell_rmse, self.cell_tests
            if sd is None:  # fall back on the wider, state-level error
                sd, tests = self.state_rmse, self.state_tests
        else:
            if self.state_rmse is None:
                return None
            level, weights = _idw(self.lam, state)
            case, sd, tests = "unsurveyed_state", self.state_rmse, self.state_tests
        if sd is None:
            return None
        sd = max(sd, self.resid_sd or 0.0)
        value = math.exp(self.alpha[iid] + level)
        rel = math.exp(sd) - 1
        checked = rel <= MAX_RELATIVE_ERROR and tests >= MIN_TESTS_FOR_CHECKED
        return {
            "value": value, "low": value * math.exp(-Z95 * sd), "high": value * math.exp(Z95 * sd),
            "sd_log": round(sd, 4), "relative_error": round(rel, 4), "n_tests": tests,
            "confidence": CHECKED if checked else INDICATIVE, "case": case,
            "state_level_pct": round((math.exp(level) - 1) * 100, 1),
            "weights": {s: round(w, 3) for s, w in weights.items()},
            "ingredient_states": ingredient_states,
            "training_states": sorted(self.lam), "n_training": len(self.lots),
            "period": list(self.period),
            "method": METHOD + (" from " + ", ".join(sorted(self.lam)) + "; state level from nearby surveyed "
                                "states by inverse-distance weighting" if case == "unsurveyed_state" else ""),
        }


def fit(lots):
    """Fit the price model. lots: [(iid, state, price, date)] from verified lots only."""
    model = PriceModel(list(lots))
    if not lots:
        model.reason = "No verified price lots."
        return model
    simple = [(i, s, p) for i, s, p, _ in lots]
    fitted = _cpd(simple)
    if fitted is None:
        model.reason = "The surveyed states do not share enough ingredients to compare their price levels."
        return model
    model.alpha, model.lam, model.resid_sd = fitted
    model.usable = True
    cells = _cell_means(simple)

    # Missing ingredient in a surveyed state: leave each (ingredient, state) pair out.
    errors = []
    for (i, s), obs in cells.items():
        rest = [l for l in simple if (l[0], l[1]) != (i, s)]
        if not any(l[0] == i for l in rest) or not any(l[1] == s for l in rest):
            continue
        refit = _cpd(rest)
        if refit is None:
            continue
        a, lam, _ = refit
        errors.append(obs - (a[i] + lam[s]))
    model.cell_rmse, model.cell_tests = _rmse(errors), len(errors)

    # State without a survey: leave each surveyed state out.
    errors = []
    if len(model.lam) >= 2:
        for s in model.lam:
            rest = [l for l in simple if l[1] != s]
            refit = _cpd(rest)
            if refit is None:
                continue
            a, lam, _ = refit
            level, _ = _idw(lam, s)
            errors += [obs - (a[i] + level) for (i, st), obs in cells.items() if st == s and i in a]
    model.state_rmse, model.state_tests = _rmse(errors), len(errors)
    return model
