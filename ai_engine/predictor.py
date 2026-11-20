"""Property prediction: internal supporting logic, not a user-facing feature.

Two entry points, both called by other parts of the system:

1. `fill_missing(view)` - ingredient level. For every property an ingredient
   has no recorded value for, try to estimate it from measured samples of the
   SAME ingredient (a model trained on one ingredient is never applied to
   another). Recorded values are never replaced. Estimates are added with
   status "predicted" and carry the model details. If the data does not
   support an estimate, nothing is added and the property stays unavailable.

2. `estimate(key, shares)` - blend level, for properties that do not mix
   linearly. The blend simulator and the explanations call it for functional
   properties of a blend. It returns an estimate only when there is enough
   relevant measured blend data, otherwise a refusal with the reason.

Blend-level model:

A Scheffe mixture model fitted on measured blends from
formulation_experiments.csv. The linear form is y = sum(b_i x_i); the quadratic
form adds b_ij x_i x_j terms that capture interactions between ingredients.
The model with the lower leave-one-out error is kept.

Sufficiency rules (all must hold):
- at least MIN_EXPERIMENTS measured blends for the property;
- at least SAMPLES_PER_PARAMETER measured blends per model coefficient;
- leave-one-out Q^2 of at least MIN_Q2 (the model explains most of the
  variation it has not seen);
- every ingredient in the requested blend appears in at least
  MIN_PER_INGREDIENT training blends, and its share lies inside the range
  covered by the training data (no extrapolation).
"""

from dataclasses import replace
from itertools import combinations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from ai_engine import properties as P

MIN_EXPERIMENTS = 8
SAMPLES_PER_PARAMETER = 2
MIN_Q2 = 0.5
MIN_PER_INGREDIENT = 3
SHARE_TOLERANCE = 0.05

# Ingredient-level prediction (missing values).
FEATURE_KEYS = ("protein", "fat", "fibre", "ash", "moisture", "carbohydrate")
TRUSTED_STATUSES = ("measured", "sourced")
MIN_SAMPLES = 8          # independent measured samples of the same ingredient
FEATURE_TOLERANCE = 0.10  # allowed extrapolation, as a fraction of the training range


class _Model:
    def __init__(self, key, experiments):
        self.key = key
        self.experiments = experiments
        self.ingredients = sorted({i for e in experiments for i in e.shares})
        self.usable = False
        self.reason = ""
        self._fit()

    def _features(self, shares_list, quadratic):
        rows = []
        for shares in shares_list:
            x = [shares.get(i, 0.0) for i in self.ingredients]
            if quadratic:
                x += [shares.get(a, 0.0) * shares.get(b, 0.0)
                      for a, b in combinations(self.ingredients, 2)]
            rows.append(x)
        return np.array(rows, dtype=float)

    def _fit(self):
        n = len(self.experiments)
        if n < MIN_EXPERIMENTS:
            self.reason = (f"Only {n} measured blend(s) for this property; at least "
                           f"{MIN_EXPERIMENTS} are needed before estimating.")
            return
        y = np.array([e.value for e in self.experiments])
        shares = [e.shares for e in self.experiments]
        best = None
        for quadratic in (False, True):
            X = self._features(shares, quadratic)
            n_params = X.shape[1]
            if n < SAMPLES_PER_PARAMETER * n_params:
                continue
            reg = LinearRegression(fit_intercept=False)
            pred = cross_val_predict(reg, X, y, cv=LeaveOneOut())
            press = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            q2 = 1 - press / ss_tot if ss_tot > 0 else 0.0
            rmse = float(np.sqrt(press / n))
            if best is None or rmse < best["rmse"]:
                best = {"quadratic": quadratic, "q2": q2, "rmse": rmse,
                        "n_params": n_params, "model": reg.fit(X, y)}
        if best is None:
            self.reason = (f"{n} measured blends cover {len(self.ingredients)} ingredients, "
                           "which is too few samples per model coefficient.")
            return
        if best["q2"] < MIN_Q2:
            self.reason = (f"Model cross-validation Q\u00b2 is {best['q2']:.2f} (minimum {MIN_Q2}); "
                           "the measured data does not support a reliable estimate.")
            return
        self.__dict__.update(best)
        self.usable = True
        self.share_ranges = {
            i: (min(e.shares.get(i, 0.0) for e in self.experiments),
                max(e.shares.get(i, 0.0) for e in self.experiments))
            for i in self.ingredients
        }
        self.counts = {i: sum(1 for e in self.experiments if e.shares.get(i, 0) > 0)
                       for i in self.ingredients}
        self.source_ids = sorted({e.source_id for e in self.experiments})
        self.locations = sorted({e.location for e in self.experiments if e.location})

    def predict(self, shares):
        for iid, s in shares.items():
            if s <= 0:
                continue
            if iid not in self.ingredients or self.counts.get(iid, 0) < MIN_PER_INGREDIENT:
                return None, f"No or too few measured blends contain '{iid}'."
        for iid in self.ingredients:
            lo, hi = self.share_ranges[iid]
            s = shares.get(iid, 0.0)
            if s < lo - SHARE_TOLERANCE or s > hi + SHARE_TOLERANCE:
                return None, (f"Share of '{iid}' ({s:.0%}) is outside the measured range "
                              f"{lo:.0%}-{hi:.0%}; the model would be extrapolating.")
        X = self._features([shares], self.quadratic)
        return float(self.model.predict(X)[0]), ""


def _training_samples(dataset, iid):
    """Measured / sourced samples of one ingredient: {sample key: {property: value}}.

    A sample is one material analysed in one study: same source, sample label,
    location and processing. Placeholder (assumed) values are never used.
    """
    samples = {}
    for (rid, key), recs in dataset.records.items():
        if rid != iid:
            continue
        for r in recs:
            if r.status not in TRUSTED_STATUSES:
                continue
            sk = (r.source_id, r.sample, r.location, r.processing)
            samples.setdefault(sk, {})[key] = r.value
    return samples


class Predictor:
    def __init__(self, dataset):
        self.dataset = dataset
        self._models = {}

    # ------------------------------------------------------ ingredient level
    def estimate_ingredient(self, view, iid, key):
        """Estimate a missing property of one ingredient, or explain why not."""
        ing = view.get(iid)
        pdef = P.get(key)
        base = {"ingredient": iid, "property": key, "label": pdef.label, "unit": pdef.unit}
        if key in ing.values:
            return {**base, "estimated": False, "reason": "A recorded value exists; it is used instead."}
        query = {k: ing.values[k].value for k in FEATURE_KEYS
                 if k != key and k in ing.values and ing.values[k].status in TRUSTED_STATUSES}
        if not query:
            return {**base, "estimated": False,
                    "reason": "No measured or sourced composition is available for this ingredient to base "
                              "an estimate on."}
        samples = [v for v in _training_samples(self.dataset, iid).values() if key in v]
        features = sorted(query)
        # Drop the least-covered features until enough complete samples remain.
        while features:
            complete = [smp for smp in samples if all(f in smp for f in features)]
            if len(complete) >= max(MIN_SAMPLES, SAMPLES_PER_PARAMETER * (len(features) + 1)):
                break
            coverage = {f: sum(1 for smp in samples if f in smp) for f in features}
            features.remove(min(coverage, key=coverage.get))
        if not features:
            return {**base, "estimated": False,
                    "reason": f"Only {len(samples)} measured sample(s) of this ingredient report "
                              f"{pdef.label.lower()}; at least {MIN_SAMPLES} with matching composition data "
                              "are needed before estimating."}
        X = np.array([[smp[f] for f in features] for smp in complete])
        y = np.array([smp[key] for smp in complete])
        pred = cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut())
        press = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        q2 = 1 - press / ss_tot if ss_tot > 0 else 0.0
        if q2 < MIN_Q2:
            return {**base, "estimated": False,
                    "reason": f"Cross-validation Q\u00b2 is {q2:.2f} (minimum {MIN_Q2}) on {len(y)} samples; "
                              "the data does not support a reliable estimate."}
        for j, f in enumerate(features):
            lo, hi = X[:, j].min(), X[:, j].max()
            pad = FEATURE_TOLERANCE * max(hi - lo, 1e-9)
            if not lo - pad <= query[f] <= hi + pad:
                return {**base, "estimated": False,
                        "reason": f"{P.get(f).label} of this ingredient ({query[f]:.2f}) is outside the range of "
                                  f"the training samples ({lo:.2f}-{hi:.2f}); the model would be extrapolating."}
        model = LinearRegression().fit(X, y)
        value = float(model.predict(np.array([[query[f] for f in features]]))[0])
        rmse = float(np.sqrt(press / len(y)))
        return {
            **base, "estimated": True, "value": round(value, 4),
            "low": round(value - 1.96 * rmse, 4), "high": round(value + 1.96 * rmse, 4),
            "method": "linear regression on " + ", ".join(features),
            "features": features, "n_training": len(y), "q2_loo": round(q2, 3), "rmse_loo": round(rmse, 4),
            "training_locations": sorted({loc for (_, _, loc, _), smp in _training_samples(self.dataset, iid).items()
                                          if key in smp and loc}),
            "note": "Predicted from measured samples of the same ingredient. Not a measurement.",
        }

    def fill_missing(self, view):
        """Return a view where missing ingredient values are estimated when the data allows.

        Recorded values are left untouched. Refusals are kept in
        view.prediction_log so explanations can report them.
        """
        from ai_engine.dataset import AggregatedValue
        filled = {}
        log = []
        refusals = {}
        for iid, ing in view.ingredients.items():
            values = dict(ing.values)
            for key in P.PROPERTIES:
                if key in values:
                    continue
                est = self.estimate_ingredient(view, iid, key)
                if est["estimated"]:
                    values[key] = AggregatedValue(
                        key, est["value"], est["low"], est["high"], 0, "predicted",
                        P.DRY_BASIS if P.get(key).basis_convertible else "", [], "predicted", est)
                    log.append(est)
                else:
                    refusals[(iid, key)] = est["reason"]
            filled[iid] = replace(ing, values=values)
        out = __import__("copy").copy(view)
        out.ingredients = filled
        out.prediction_log = log
        out.prediction_refusals = refusals
        return out

    def _model(self, key):
        if key not in self._models:
            exps = [e for e in self.dataset.experiments if e.property == key]
            self._models[key] = _Model(key, exps)
        return self._models[key]

    def estimate(self, key, shares):
        """Estimate a non-linear property for a blend, or decline with a reason."""
        pdef = P.get(key)
        base = {"property": key, "label": pdef.label, "unit": pdef.unit}
        model = self._model(key)
        if not model.usable:
            return {**base, "estimated": False, "reason": model.reason}
        value, reason = model.predict({k: v for k, v in shares.items() if v > 1e-9})
        if value is None:
            return {**base, "estimated": False, "reason": reason}
        half_width = 1.96 * model.rmse
        return {
            **base,
            "estimated": True,
            "value": round(value, 3),
            "low": round(value - half_width, 3),
            "high": round(value + half_width, 3),
            "method": "Scheffe {} mixture model".format("quadratic" if model.quadratic else "linear"),
            "n_training": len(model.experiments),
            "q2_loo": round(model.q2, 3),
            "rmse_loo": round(model.rmse, 3),
            "source_ids": model.source_ids,
            "training_locations": model.locations,
            "note": "Predicted value. Confirm by laboratory measurement before use.",
        }

    def status(self):
        """Which functional properties currently have a usable model."""
        out = []
        for pdef in P.PROPERTIES.values():
            if pdef.blend_rule != P.NONLINEAR:
                continue
            m = self._model(pdef.key)
            out.append({
                "property": pdef.key,
                "label": pdef.label,
                "n_experiments": len(m.experiments),
                "usable": m.usable,
                "reason": m.reason if not m.usable else "",
                "q2_loo": round(m.q2, 3) if m.usable else None,
            })
        return out
