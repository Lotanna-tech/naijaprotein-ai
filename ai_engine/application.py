"""Application fit: which plant proteins suit a product, in what proportions, and how sure we are.

This is the decision core behind the Simulate and Optimise pages (Hack the Fork Nigeria 2026,
Challenge 1: rank Nigerian protein ingredients for a defined alternative-protein application).

1. Application profile (products.py). Each product lists the properties that matter, the
   direction that helps and an importance level (high 3, medium 2, low 1 points). The weights are
   the points normalised over the properties that every selected ingredient has data for. Only
   protein content and functional properties are scored; fat, carbohydrate etc. are not.

2. Score. Each value is put on a 0-1 scale between the lowest and the highest verified value
   seen for any of the five ingredients in Nigeria (reversed when lower is better), so the score
   compares the candidates: 100 = best of the five on everything that matters, 0 = worst.

       score_i = 100 * sum_k w_k * n_ik

3. Uncertainty: Monte Carlo, in the manner of SMAA-2 (Lahdelma, Hokkanen & Salminen 1998,
   Eur. J. Oper. Res. 106: 137-143). N_SIM times, every ingredient value is drawn from its spread
   (lot-to-lot spread of the records, plus the 95% range when the value is estimated for the
   state) and the weights are drawn from a Dirichlet distribution centred on the profile weights,
   because the importance levels are judgement too (about +-30%). Each draw ranks the
   ingredients. Results: score with a 90% range, the probability of each rank, and how often the
   shortlist stays the same.

4. Blend. A blend's property is the share-weighted average of its ingredients' properties (no
   interaction assumed; real blends can do better or worse, which the lab plan tests). Its score
   in each draw is therefore sum_i s_i * score_i.

5. Optimisation, robust to the uncertainty: the best blend maximises the conditional value at risk
   (CVaR) of its score, the average of its worst 10% of simulated outcomes (Rockafellar & Uryasev
   2000, J. Risk 2(3): 21-41). This is a linear program, solved with SciPy/HiGHS. Shares stay
   within the supplier maximum share for the state. Among blends within COST_TOLERANCE points of
   the best CVaR, the cheapest is taken.

6. Lab plan. Up to 3 shortlisted ingredients go into a simplex-centroid mixture design (Scheffe
   1963, J. R. Stat. Soc. B 25: 235-263) plus the optimised blend. Those measurements are the
   training data the blend model in predictor.py needs to learn how the ingredients interact.
"""

import math
from itertools import combinations
from statistics import mean, median, stdev

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from ai_engine import products as PR
from ai_engine import properties as P
from ai_engine.location_model import INDICATIVE

ENGINE = "application-fit-1"
N_SIM = 2000
SEED = 2026
WEIGHT_CONCENTRATION = 30.0
CVAR_ALPHA = 0.10
COST_TOLERANCE = 1.0          # score points the optimiser may give up to save cost
SHORTLIST_SIZE = 3
LAB_BATCH_G = 200
MIN_LAB_SHARE = 0.10          # an ingredient this large in the best blend goes on the shortlist
DEFAULT_REL_SPREAD = 0.05
SHARE_SUM_TOLERANCE = 0.05    # percentage points
DISPLAY_KEYS = ("protein", "water_absorption", "oil_absorption")
# Same property in two units: score only one of them.
ALTERNATIVE_UNITS = {"emulsifying_capacity_pct": "emulsifying_capacity"}

LEVEL_LABELS = {"high": "High", "medium": "Medium", "low": "Low"}

IMPORTANCE_LEVELS = ("high", "medium", "low", "off")   # "off": not scored

# Shown under "Good to know" on every result. Plain words: the method is in the README.
ASSUMPTIONS = [
    "How much each property matters is general food-science guidance for this product. You can change it on the "
    "form.",
    "Scores compare the five proteins with each other: 100 means best of the five on everything that matters.",
    "A blend is scored from its proteins' scores. How proteins work together can only be measured in the lab.",
    "Values are for the flours in our data. Processing such as dehulling, roasting or oil removal changes them.",
]
HOW_RANKING_IS_CHECKED = (
    f"We ran the ranking {N_SIM:,} times, each time varying the ingredient values within their measured range and "
    "the importance of each property a little. High: the top pick wins at least 80% of the runs. Medium: at least "
    "55%. Low: below that.")
HOW_BLEND_IS_CHECKED = (
    f"We scored the blend {N_SIM:,} times, each time varying the ingredient values within their measured range and "
    "the importance of each property a little. High: 90% of the scores fall within 12 points. Medium: within 24. "
    "Low: wider.")


class ApplicationError(ValueError):
    """Input that cannot be evaluated; the message is shown to the user."""


# ---------------------------------------------------------------- helpers
def _r(x, nd=2):
    return None if x is None else round(float(x), nd)


def _fmt(key, value):
    unit = P.get(key).unit
    nd = 2 if unit in ("g/g", "g/mL", "ml oil/g") else 1
    return f"{value:.{nd}f}{'' if unit == '%' else ' '}{unit}"


def _name(label):
    """Property label for use in a sentence: 'Emulsifying capacity (%)' -> 'emulsifying capacity'."""
    return label.split(" (")[0].lower()


def _count(n, singular, plural=None):
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def _share_of_sims(p, n):
    if p >= 1:
        return f"in all {n:,} test runs"
    if p <= 0:
        return f"in none of the {n:,} test runs"
    if p >= 0.995:
        return f"in more than 99% of {n:,} test runs"
    if p < 0.005:
        return f"in fewer than 1% of {n:,} test runs"
    return f"in {p:.0%} of {n:,} test runs"


def _join(names):
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _whole(x):
    """Round half up, as the page does, so a number reads the same in sentences and tables."""
    return int(math.floor(float(x) + 0.5))


def _pct(x, nd=0):
    return f"{_whole(x * 100)}%" if nd == 0 else f"{x * 100:.{nd}f}%"


def _share_label(x):
    """'50%' for round shares, '66.7%' otherwise, so a blend's parts add up to 100."""
    return _pct(x) if abs(x * 100 - round(x * 100)) < 0.05 else _pct(x, 1)


def _naira(x):
    return f"\u20a6{x:,.0f}"


def _has_records(dataset, key):
    return any(k == key and recs for (_, k), recs in dataset.records.items())


def _rel_spread(dataset, key):
    """Typical lot-to-lot relative spread of a property: median over ingredients with 2+ verified records."""
    rels = []
    for iid in dataset.ingredients:
        vals = [r.value for r in dataset.records.get((iid, key), [])]
        if len(vals) >= 2 and mean(vals) > 0:
            rels.append(stdev(vals) / mean(vals))
    return median(rels) if rels else DEFAULT_REL_SPREAD


def _distribution(v, rel_spread):
    """(mean, sd) of the value that a lot bought in this state is likely to have."""
    lot_sd = rel_spread * abs(v.value)
    if v.status == "predicted":
        est_sd = (v.high - v.low) / (2 * 1.96)
        return v.value, math.sqrt(est_sd ** 2 + lot_sd ** 2)
    vals = [r.value for r in v.records]
    return v.value, max(stdev(vals) if len(vals) >= 2 else 0.0, lot_sd)


def _quality(v):
    if v.status != "predicted":
        return v.kind  # measured / calculated
    return "rough" if v.confidence == INDICATIVE else "checked"


def _price(ing):
    p = ing.price
    if p is None:
        return None
    return {"value": _r(p.value), "low": _r(p.low), "high": _r(p.high), "kind": p.kind, "confidence": p.confidence,
            "location_status": p.location_status, "max_share": p.max_share,
            "note": (p.prediction or {}).get("note", "")}


# ---------------------------------------------------------------- criteria
def scorable(key):
    """Properties the fit score may use: protein content and functional properties (not fat, fibre etc.)."""
    pdef = P.PROPERTIES.get(key)
    return pdef is not None and (key == "protein" or pdef.group == "functional")


def _defaults(product):
    """Default importance of every scorable property for the product ('off' if the product does not list it)."""
    out = {key: "off" for key in P.PROPERTIES if scorable(key)}
    out.update({r.property: r.importance for r in product.relevant})
    return out


def _relevances(product, overrides):
    """The product's properties plus any other scorable property the user switched on."""
    rels = list(product.relevant)
    known = {r.property for r in rels}
    for key in overrides:
        if key not in known:
            pdef = P.get(key)
            rels.append(PR.Relevance(key, PR.LOWER if key == "least_gelation" else PR.HIGHER,
                                     pdef.plain or pdef.label, "off"))
    return rels


def importance_overrides(product, raw):
    """Validate the user's {property: high/medium/low/off}; return only the changes from the defaults."""
    if raw in (None, {}):
        return {}
    if not isinstance(raw, dict):
        raise ApplicationError("Importance settings must be an object: {property: high, medium, low or off}.")
    defaults = _defaults(product)
    out = {}
    for key, level in raw.items():
        if key not in defaults:
            raise ApplicationError(f"'{key}' cannot be weighed: only protein and the functional properties can.")
        if level not in IMPORTANCE_LEVELS:
            raise ApplicationError(f"The importance of {_name(P.get(key).label)} must be high, medium, low or off.")
        if level != defaults[key]:
            out[key] = level
    return out


def _importance_changes(product, overrides):
    defaults = _defaults(product)
    return [{"property": k, "label": P.get(k).label, "from": defaults[k], "to": v} for k, v in overrides.items()]


def _assumptions(changes):
    if not changes:
        return ASSUMPTIONS
    listed = _join(f"{_name(c['label'])} ({c['from']} \u2192 {c['to']})" for c in changes)
    return [f"You changed how much {listed} matter{'s' if len(changes) == 1 else ''}. The other properties use "
            "general food-science guidance."] + ASSUMPTIONS[1:]


def criteria(dataset, product, ids, overrides=None):
    """(scored, not_scored) properties of the product for the chosen ingredients.

    overrides: {property: high/medium/low/off} replacing the default importance (see importance_overrides).
    """
    overrides = overrides or {}
    scored, missing = [], []
    rels = _relevances(product, overrides)
    keys = [r.property for r in rels]
    for rel in rels:
        key = rel.property
        pdef = P.PROPERTIES.get(key)
        if pdef is None:
            continue
        level = overrides.get(key, rel.importance)
        info = {"property": key, "label": pdef.label, "unit": pdef.unit, "direction": rel.direction,
                "importance": level, "default_importance": rel.importance, "why": rel.why, "plain": pdef.plain}
        if level == "off":
            missing.append({**info, "reason": "Turned off in your importance settings.", "turned_off": True})
            continue
        recs = [r.value for (_, k), rs in dataset.records.items() if k == key for r in rs]
        lacking = [dataset.get(i).name for i in ids if key not in dataset.get(i).values]
        if not recs:
            alt = ALTERNATIVE_UNITS.get(key) or next((a for a, b in ALTERNATIVE_UNITS.items() if b == key), None)
            if alt in keys and _has_records(dataset, alt):
                continue  # the same property is scored in its other unit
            missing.append({**info, "reason": "Not measured for any of the five ingredients yet."})
        elif lacking:
            missing.append({**info, "reason": "No value for " + ", ".join(lacking) + ", so it cannot be compared."})
        elif max(recs) - min(recs) <= 0:
            missing.append({**info, "reason": "Every ingredient has the same value, so it does not separate them."})
        else:
            scored.append({**info, "lo": min(recs), "hi": max(recs)})
    scored_keys = {c["property"] for c in scored}
    scored = [c for c in scored if not (c["property"] in ALTERNATIVE_UNITS
                                        and ALTERNATIVE_UNITS[c["property"]] in scored_keys)]
    total = sum(PR.IMPORTANCE_POINTS[c["importance"]] for c in scored)
    for c in scored:
        c["weight"] = PR.IMPORTANCE_POINTS[c["importance"]] / total
    return scored, missing


def preview(dataset, product):
    """What the ranking weighs for a product by default, for the editable table on the input forms.

    scored: the product's properties with data; optional: other scorable properties with data, off by
    default but available to switch on; not_scored: properties that matter but have no data.
    """
    scored, missing = criteria(dataset, product, [])
    listed = {r.property for r in product.relevant}
    extra = {k: "low" for k in P.PROPERTIES if scorable(k) and k not in listed and _has_records(dataset, k)}
    optional, _ = criteria(dataset, product, [], extra) if extra else ([], [])
    optional = [c for c in optional if c["property"] in extra]
    fields = ("property", "label", "importance", "why", "direction")
    return {"scored": [{k: c[k] for k in fields} | {"weight": round(c["weight"], 4)} for c in scored],
            "optional": [{k: c[k] for k in fields} | {"importance": "off"} for c in optional],
            "not_scored": [{k: c[k] for k in ("property", "label", "importance", "reason")} for c in missing]}


# ---------------------------------------------------------------- simulation
class Simulation:
    """Monte Carlo draws of every ingredient value and of the weights, and the resulting scores."""

    def __init__(self, dataset, ids, crits, n=N_SIM, seed=SEED):
        rng = np.random.default_rng(seed)
        self.ids, self.crits, self.n = list(ids), crits, n
        keys = [c["property"] for c in crits]
        extra = [k for k in DISPLAY_KEYS if k not in keys and all(k in dataset.get(i).values for i in ids)]
        self.keys = keys + extra
        spread = {k: _rel_spread(dataset, k) for k in self.keys}
        dist = [[_distribution(dataset.get(i).values[k], spread[k]) for k in self.keys] for i in self.ids]
        self.mu = np.array([[m for m, _ in row] for row in dist])
        self.sd = np.array([[s for _, s in row] for row in dist])
        self.x = rng.normal(self.mu, self.sd, size=(n,) + self.mu.shape).clip(min=0)   # (n, I, K_all)

        K = len(crits)
        lo = np.array([c["lo"] for c in crits])
        hi = np.array([c["hi"] for c in crits])
        lower = np.array([c["direction"] == PR.LOWER for c in crits])
        self.w0 = np.array([c["weight"] for c in crits])

        def scale(values):
            z = ((values - lo) / (hi - lo)).clip(0, 1)
            return np.where(lower, 1 - z, z)

        norm = scale(self.x[:, :, :K])
        self.weights = rng.dirichlet(WEIGHT_CONCENTRATION * self.w0, size=n) if K > 1 else np.ones((n, 1))
        self.scores = 100 * np.einsum("nik,nk->ni", norm, self.weights)          # (n, I)
        self.norm0 = scale(self.mu[:, :K])                                        # at the central values
        self.contrib0 = 100 * self.norm0 * self.w0                                # (I, K) points
        self.score0 = self.contrib0.sum(axis=1)

    def index(self, key):
        return self.keys.index(key)

    def blend_scores(self, s):
        return self.scores @ s

    def ranks(self):
        order = np.argsort(-self.scores, axis=1)
        ranks = np.empty_like(order)
        ranks[np.arange(self.n)[:, None], order] = np.arange(1, len(self.ids) + 1)
        return order, ranks


def _cvar(values, alpha=CVAR_ALPHA):
    k = max(1, int(round(alpha * len(values))))
    return float(np.sort(values)[:k].mean())


# ---------------------------------------------------------------- ranking
def _ranking(dataset, sim):
    order, ranks = sim.ranks()
    I = len(sim.ids)
    det = list(np.argsort(-sim.score0))
    k_short = min(SHORTLIST_SIZE, I)
    best_value = {}
    for k, c in enumerate(sim.crits):
        pick = np.argmin if c["direction"] == PR.LOWER else np.argmax
        best_value[k] = int(pick(sim.mu[:, k]))
    rows = []
    for pos, a in enumerate(det):
        ing = dataset.get(sim.ids[a])
        contributions = []
        for k, c in enumerate(sim.crits):
            v = ing.values[c["property"]]
            contributions.append({
                "property": c["property"], "label": c["label"], "unit": c["unit"], "importance": c["importance"],
                "points": _r(sim.contrib0[a, k], 1), "max_points": _r(100 * sim.w0[k], 1),
                "value": _r(sim.mu[a, k], 3), "relative": _r(sim.norm0[a, k], 3),
                "best_of_selected": bool(best_value[k] == a and I > 1), "quality": _quality(v),
            })
        quality = {}
        for c in contributions:
            quality[c["quality"]] = quality.get(c["quality"], 0) + 1
        rows.append({
            "id": ing.id, "name": ing.name, "rank": pos + 1,
            "score": _r(sim.score0[a], 1), "low": _r(np.percentile(sim.scores[:, a], 5), 1),
            "high": _r(np.percentile(sim.scores[:, a], 95), 1),
            "p_first": _r((ranks[:, a] == 1).mean(), 4),
            "p_shortlist": _r((ranks[:, a] <= k_short).mean(), 4),
            "expected_rank": _r(ranks[:, a].mean(), 2),
            "rank_probabilities": [_r((ranks[:, a] == r).mean(), 4) for r in range(1, I + 1)],
            "contributions": contributions, "quality": quality, "price": _price(ing),
            "reason": _ingredient_reason(ing.name, contributions, sim.score0[a]),
        })
    return rows


def _ingredient_reason(name, contributions, score):
    """'Strong on protein (52.7 g/100 g) and oil absorption (2.08 g/g); weaker on emulsifying capacity (64.7%).'"""
    strong = [c for c in sorted(contributions, key=lambda c: -c["points"]) if c["relative"] >= 0.5][:2]
    weak = [c for c in sorted(contributions, key=lambda c: c["relative"])
            if c["importance"] in ("high", "medium") and c["relative"] < 0.5 and c not in strong][:1]
    say = lambda cs: _join(f"{_name(c['label'])} ({_fmt(c['property'], c['value'])})" for c in cs)
    if strong and weak:
        return f"Strong on {say(strong)}; weaker on {say(weak)}."
    if strong:
        return f"Strong on {say(strong)}."
    if weak:
        return f"No stand-out strengths for this product; weakest on {say(weak)}."
    return "Average on everything that matters for this product."


def _sensitivity(sim, ranking):
    """Does doubling or ignoring one property's importance change the top pick?"""
    if len(sim.ids) < 2:
        return []
    top = int(np.argmax(sim.score0))
    out = []
    for k, c in enumerate(sim.crits):
        for factor, text in ((2.0, f"If {_name(c['label'])} mattered twice as much"),
                             (0.0, f"If {_name(c['label'])} were ignored")):
            w = sim.w0.copy()
            w[k] *= factor
            if w.sum() <= 0:
                continue
            alt = int(np.argmax(sim.norm0 @ (w / w.sum())))
            if alt != top:
                out.append(f"{text}, {next(r['name'] for r in ranking if r['id'] == sim.ids[alt])} would come first.")
    return out


def _data_counts(dataset, ids, crits):
    counts = {"measured": 0, "calculated": 0, "checked": 0, "rough": 0}
    rough = []
    for i in ids:
        ing = dataset.get(i)
        for c in crits:
            q = _quality(ing.values[c["property"]])
            counts[q] += 1
            if q == "rough":
                rough.append(f"{ing.name} {_name(c['label'])}")
    return counts, rough


def _price_counts(dataset, ids):
    out = {"measured": 0, "estimated": 0, "missing": 0, "estimated_names": []}
    for i in ids:
        p = dataset.get(i).price
        kind = "missing" if p is None else "estimated" if p.kind == "predicted" else "measured"
        out[kind] += 1
        if kind == "estimated":
            out["estimated_names"].append(dataset.get(i).name)
    return out


def _price_sentence(prices):
    if not prices["estimated"]:
        return None
    if not prices["measured"]:
        return "Prices here are estimates (no market prices were collected in this state). They do not change the ranking."
    names = prices["estimated_names"]
    return (f"The price of {_join(names)} is an estimate (no market price for {'it' if len(names) == 1 else 'them'} "
            "here). Prices do not change the ranking.")


def _downgrade(level):
    return {"high": "medium", "medium": "low", "low": "low"}[level]


def _data_sentence(dataset, counts):
    total = sum(counts.values())
    measured = counts["measured"] + counts["calculated"]
    if not dataset.location:
        return "Uses all the data we have from across Nigeria."
    where = dataset.location
    if measured == total:
        return f"All the values used were measured in {where}."
    if measured == 0:
        return f"Nothing has been measured in {where} yet, so all values are estimated from other states."
    return f"{measured} of {total} values were measured in {where}; the rest are estimated from other states."


def _missing_sentence(missing):
    important = [m for m in missing if m["importance"] in ("high", "medium") and not m.get("turned_off")]
    if not important:
        return None
    return (f"{_join(_name(m['label']).capitalize() if i == 0 else _name(m['label']) for i, m in enumerate(important))}"
            f" also {'matters' if len(important) == 1 else 'matter'}, but no protein has been tested for "
            f"{'it' if len(important) == 1 else 'them'} yet. A lab test could change the result.")


def ranking_certainty(dataset, sim, ranking, missing, changes=()):
    """Certainty card for a ranking. changes: importance levels the user changed."""
    counts, rough = _data_counts(dataset, sim.ids, sim.crits)
    total = sum(counts.values())
    rough_share = counts["rough"] / total if total else 0.0
    top = ranking[0]
    points = []
    if len(ranking) == 1:
        level = "medium"
        headline = f"Only {top['name']} was chosen, so there is nothing to compare it with."
    else:
        p = top["p_first"]
        level = "high" if p >= 0.80 else "medium" if p >= 0.55 else "low"
        headline = f"{top['name']} came out best {_share_of_sims(p, sim.n)}."
        second = ranking[1]
        points.append(f"Next best: {second['name']}"
                      + (f" (best in {_pct(second['p_first'])} of the runs)." if second["p_first"] >= 0.005 else "."))
    top_rough = top["quality"].get("rough", 0) / max(1, sum(top["quality"].values()))
    where = f" for {dataset.location}" if dataset.location else ""
    if rough_share >= 0.5 or top_rough >= 0.5:
        level = _downgrade(level)
        who = "the values" if rough_share >= 0.5 else f"{top['name']}'s values"
        points.append(f"Most of {who}{where} are rough estimates, so we lowered the certainty. Testing a local "
                      "sample would raise it.")
    points.append(_data_sentence(dataset, counts))
    prices = _price_counts(dataset, sim.ids)
    if _price_sentence(prices):
        points.append(_price_sentence(prices))
    if len(ranking) > 1:
        points += (_sensitivity(sim, ranking)
                   or ["Making any one property count twice as much, or not at all, does not change the top pick."])
    miss = _missing_sentence(missing)
    if miss:
        points.append(miss)
    if changes:
        points.append("Uses your importance settings.")
    return {"level": level, "label": LEVEL_LABELS[level], "headline": headline, "points": points,
            "how": HOW_RANKING_IS_CHECKED, "assumptions": _assumptions(changes),
            "data": counts, "rough_values": rough, "prices": prices, "simulations": sim.n}


# ---------------------------------------------------------------- blends
def evaluate_blend(dataset, sim, shares, batch_kg=None):
    """Score, protein, water and oil binding and cost of a blend {ingredient: share 0..1}."""
    s = np.array([shares.get(i, 0.0) for i in sim.ids])
    scores = sim.blend_scores(s)
    props = np.einsum("nik,i->nk", sim.x, s)
    out = {"shares": {i: _r(v, 6) for i, v in zip(sim.ids, s) if v > 1e-9},
           "score": {"value": _r(sim.score0 @ s, 1), "low": _r(np.percentile(scores, 5), 1),
                     "high": _r(np.percentile(scores, 95), 1), "worst_10pct": _r(_cvar(scores), 1)},
           "properties": {}}
    for k in sim.keys:
        j = sim.index(k)
        out["properties"][k] = {"label": P.get(k).label, "unit": P.get(k).unit, "value": _r(sim.mu[:, j] @ s, 3),
                                "low": _r(np.percentile(props[:, j], 5), 3),
                                "high": _r(np.percentile(props[:, j], 95), 3)}
    out["protein"] = out["properties"].get("protein")
    binding = {}
    for key, noun in (("water_absorption", "water"), ("oil_absorption", "oil")):
        if key in out["properties"]:
            p = out["properties"][key]
            binding[noun] = {"per_kg": p["value"], "low": p["low"], "high": p["high"],
                             "batch_kg": _r(p["value"] * batch_kg, 1) if batch_kg else None}
    out["binding"] = binding

    components, cost, low, high, estimated, missing_price = [], 0.0, 0.0, 0.0, [], []
    criteria_points = 100 * sim.norm0 * sim.w0
    for a, iid in enumerate(sim.ids):
        if s[a] <= 1e-9:
            continue
        ing = dataset.get(iid)
        price = _price(ing)
        components.append({"id": iid, "name": ing.name, "share": _r(s[a], 6),
                           "kg": _r(s[a] * batch_kg, 3) if batch_kg else None,
                           "points": _r(s[a] * sim.score0[a], 1), "price": price})
        if price is None:
            missing_price.append(ing.name)
            continue
        cost += s[a] * price["value"]
        low += s[a] * price["low"]
        high += s[a] * price["high"]
        if price["kind"] == "predicted":
            estimated.append(ing.name)
    out["components"] = sorted(components, key=lambda c: -c["share"])
    out["criteria_points"] = [{"property": c["property"], "label": c["label"],
                               "points": _r(criteria_points[:, k] @ s, 1), "max_points": _r(100 * sim.w0[k], 1)}
                              for k, c in enumerate(sim.crits)]
    out["cost"] = ({"available": False, "reason": "No price for " + ", ".join(missing_price) + "."}
                   if missing_price else
                   {"available": True, "value": _r(cost), "low": _r(low), "high": _r(high),
                    "batch": _r(cost * batch_kg) if batch_kg else None, "estimated_prices": estimated,
                    "per_kg_protein": _r(cost / (out["protein"]["value"] / 100))
                    if out["protein"] and out["protein"]["value"] else None})
    out["_scores"] = scores  # removed before returning to the caller
    return out


def _binding_text(product, blend):
    b = blend.get("binding", {})
    if not b:
        return None
    parts = []
    if "water" in b and "oil" in b:
        parts.append(f"1 kg of this blend can absorb about {b['water']['per_kg']:.1f} kg of water, or about "
                     f"{b['oil']['per_kg']:.1f} kg of oil (measured separately, so not both at once).")
    elif "water" in b:
        parts.append(f"1 kg of this blend can absorb about {b['water']['per_kg']:.1f} kg of water.")
    else:
        parts.append(f"1 kg of this blend can absorb about {b['oil']['per_kg']:.1f} kg of oil.")
    if "oil" in b:
        parts.append("Because the product is cooked, oil the blend holds stays in it instead of cooking out.")
    return " ".join(parts)


def _blend_certainty(dataset, sim, blend, missing, changes=()):
    used = list(blend["shares"])
    counts, rough = _data_counts(dataset, used, sim.crits)
    total = sum(counts.values())
    width = blend["score"]["high"] - blend["score"]["low"]
    level = "high" if width <= 12 else "medium" if width <= 24 else "low"
    points = []
    if total and counts["rough"] / total >= 0.5:
        level = _downgrade(level)
        where = f" for {dataset.location}" if dataset.location else ""
        points.append(f"Most of the values{where} are rough estimates, so we lowered the certainty. Testing local "
                      "samples would raise it.")
    points.append(_data_sentence(dataset, counts))
    if blend["cost"].get("estimated_prices"):
        points.append(f"Prices for {_join(blend['cost']['estimated_prices'])} are estimates (no market prices were "
                      f"collected in this state), so the cost could be between {_naira(blend['cost']['low'])} and "
                      f"{_naira(blend['cost']['high'])} per kg.")
    miss = _missing_sentence(missing)
    if miss:
        points.append(miss)
    if changes:
        points.append("Uses your importance settings.")
    low, high = _whole(blend["score"]["low"]), _whole(blend["score"]["high"])
    headline = {"high": f"The score is reliable: most likely between {low} and {high}.",
                "medium": f"The score could move a little: most likely between {low} and {high}.",
                "low": f"The score is uncertain: it could be anywhere from {low} to {high}."}[level]
    return {"level": level, "label": LEVEL_LABELS[level], "headline": headline, "points": points,
            "how": HOW_BLEND_IS_CHECKED, "assumptions": _assumptions(changes), "data": counts,
            "rough_values": rough, "simulations": sim.n}


# ---------------------------------------------------------------- optimisation
def robust_blend(sim, upper, prices=None, budget_per_kg=None, alpha=CVAR_ALPHA, tolerance=COST_TOLERANCE):
    """Shares maximising the CVaR of the blend score; then the cheapest within `tolerance` points of it.

    Variables: shares s (I), threshold t, shortfalls u (n).
        max  t - 1/(alpha n) sum_j u_j
        s.t. u_j >= t - S_j s,  u_j >= 0,  sum s = 1,  0 <= s_i <= upper_i  [,  price . s <= budget]
    Returns (shares or None, info).
    """
    S = sim.scores
    n, I = S.shape
    c = np.concatenate([np.zeros(I), [-1.0], np.full(n, 1.0 / (alpha * n))])
    A = sparse.hstack([sparse.csr_matrix(-S), sparse.csr_matrix(np.ones((n, 1))), -sparse.identity(n)]).tocsr()
    b = np.zeros(n)
    have_prices = prices is not None and all(p is not None for p in prices)
    if budget_per_kg is not None and have_prices:
        A = sparse.vstack([A, sparse.csr_matrix(np.concatenate([prices, [0.0], np.zeros(n)]))]).tocsr()
        b = np.append(b, budget_per_kg)
    A_eq = sparse.csr_matrix(np.concatenate([np.ones(I), [0.0], np.zeros(n)]))
    bounds = [(0.0, u) for u in upper] + [(None, None)] + [(0.0, None)] * n
    res = linprog(c, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=[1.0], bounds=bounds, method="highs")
    if res.status != 0:
        return None, {"status": int(res.status), "message": res.message}
    best_cvar = float(-res.fun)
    info = {"status": 0, "cvar": best_cvar, "cost_step": False}
    x = res.x
    if have_prices:
        cvar_row = sparse.csr_matrix(np.concatenate([np.zeros(I), [-1.0], np.full(n, 1.0 / (alpha * n))]))
        res2 = linprog(np.concatenate([prices, [0.0], np.zeros(n)]), A_ub=sparse.vstack([A, cvar_row]).tocsr(),
                       b_ub=np.append(b, -(best_cvar - tolerance)), A_eq=A_eq, b_eq=[1.0], bounds=bounds,
                       method="highs")
        if res2.status == 0:
            x, info["cost_step"] = res2.x, True
    shares = np.clip(x[:I], 0, None)
    shares[shares < 1e-6] = 0.0
    return shares / shares.sum(), info


def _cheapest(prices, upper):
    res = linprog(prices, A_eq=np.ones((1, len(prices))), b_eq=[1.0], bounds=[(0.0, u) for u in upper],
                  method="highs")
    return float(res.fun) if res.status == 0 else None


# ---------------------------------------------------------------- lab plan
def _combinations_10pct(m):
    return math.comb(10 + m - 1, m - 1)


def _same_blend(a, b):
    return all(abs(a.get(i, 0) - b.get(i, 0)) < 0.01 for i in set(a) | set(b))


def lab_plan(dataset, product, ranking, blend, missing, selected, overrides=None):
    """Shortlist and a small mixture design for the laboratory.

    Augmented simplex-centroid design on the shortlist (pure ingredients, equal two- and three-way
    blends, and axial check blends), plus the optimised blend, plus repeats of the centre blend until
    there are enough runs for the quadratic blend model in predictor.py (at least MIN_EXPERIMENTS
    blends and SAMPLES_PER_PARAMETER per coefficient).
    """
    from ai_engine.predictor import MIN_EXPERIMENTS, SAMPLES_PER_PARAMETER
    order = [r["id"] for r in ranking]
    big = sorted((i for i, s in blend["shares"].items() if s >= MIN_LAB_SHARE), key=lambda i: -blend["shares"][i])
    short = [i for i in order if i in big[:SHORTLIST_SIZE]]
    short += [i for i in order if i not in short][:max(0, SHORTLIST_SIZE - len(short))]
    short = [i for i in order if i in short]
    names = {i: dataset.get(i).name for i in selected}
    m = len(short)

    def label(shares):
        parts = sorted(shares.items(), key=lambda kv: -kv[1])
        return f"{names[parts[0][0]]} only" if len(parts) == 1 else " + ".join(f"{_share_label(s)} {names[i]}"
                                                                               for i, s in parts)

    runs = []
    for size in range(1, m + 1):                                   # simplex-centroid points
        for subset in combinations(short, size):
            runs.append({"type": "design", "shares": {i: 1.0 / size for i in subset}})
    if m >= 2:                                                     # axial check blends
        for i in short:
            runs.append({"type": "check", "shares": {j: ((1 + m) / (2 * m) if j == i else 1 / (2 * m))
                                                     for j in short}})
    opt = {i: s for i, s in blend["shares"].items() if s > 1e-9}
    if not any(_same_blend(r["shares"], opt) for r in runs):
        runs.append({"type": "optimised", "shares": opt})
    if m >= 2:
        needed = max(MIN_EXPERIMENTS, SAMPLES_PER_PARAMETER * (m + math.comb(m, 2)))
        centre = {i: 1.0 / m for i in short}
        while len(runs) < needed:
            runs.append({"type": "repeat", "shares": dict(centre)})
    notes = {"design": "", "check": "", "optimised": "recommended blend", "repeat": "repeat, to check the test"}
    for n, r in enumerate(runs, start=1):
        r["run"] = n
        r["label"] = label(r["shares"]) + (f" ({notes[r['type']]})" if notes[r["type"]] else "")
        r["grams"] = {i: round(r["shares"].get(i, 0.0) * LAB_BATCH_G, 1)
                      for i in short + [j for j in r["shares"] if j not in short]}
        r["shares"] = {i: round(s, 4) for i, s in r["shares"].items()}
    overrides = overrides or {}
    rels = _relevances(product, overrides)
    level = {r.property: overrides.get(r.property, r.importance) for r in rels}
    measure = [r for r in rels if r.property in P.PROPERTIES and level[r.property] != "off"]
    shown = {r.property for r in measure}
    measure = [r for r in measure if not (r.property in ALTERNATIVE_UNITS.values()
                                          and any(a in shown for a, b in ALTERNATIVE_UNITS.items()
                                                  if b == r.property))]
    measure.sort(key=lambda r: -PR.IMPORTANCE_POINTS[level[r.property]])
    no_data = {x["property"] for x in missing if not x.get("turned_off")}
    every = _combinations_10pct(len(selected))
    text = (f"Make these {len(runs)} small batches ({LAB_BATCH_G} g each) and test every one. That is {len(runs)} "
            f"tests instead of {every:,} for every mix of your {len(selected)} proteins in 10% steps, and it shows "
            "how the proteins work together.")
    return {"shortlist": [{"id": i, "name": names[i], "rank": order.index(i) + 1} for i in short],
            "runs": runs, "batch_g": LAB_BATCH_G,
            "measure": [{"property": r.property, "label": P.get(r.property).label, "importance": level[r.property],
                         "no_data_yet": r.property in no_data} for r in measure],
            "every_blend_count": every, "text": text}


# ---------------------------------------------------------------- entry points
def _product(product_id):
    try:
        return PR.get(product_id)
    except ValueError as err:
        raise ApplicationError(str(err))


def normalise_recipe(dataset, recipe):
    """{ingredient: % of the protein blend} -> shares summing to 1."""
    if not isinstance(recipe, dict) or not recipe:
        raise ApplicationError("Choose at least one plant protein.")
    clean = {}
    for iid, pct in recipe.items():
        if iid in ("water", "oil"):
            raise ApplicationError("Water and oil are no longer part of the blend: the result shows how much water "
                                   "and oil the blend can bind.")
        if iid not in dataset.ingredients:
            raise ApplicationError(f"Unknown ingredient '{iid}'.")
        try:
            v = float(pct)
        except (TypeError, ValueError):
            raise ApplicationError(f"The amount of {dataset.get(iid).name} must be a number.")
        if not 0 <= v <= 100:
            raise ApplicationError(f"The amount of {dataset.get(iid).name} must be between 0 and 100%.")
        if v > 0:
            clean[iid] = v
    if not clean:
        raise ApplicationError("Choose at least one plant protein with an amount above 0%.")
    total = sum(clean.values())
    if abs(total - 100) > SHARE_SUM_TOLERANCE:
        raise ApplicationError(f"The amounts must add up to 100% of the protein blend (currently {total:.1f}%).")
    return {k: v / total for k, v in clean.items()}


def _prepare(dataset, product, ids, overrides=None):
    keys = [r.property for r in product.relevant if _has_records(dataset, r.property)]
    usable, excluded = [], []
    for i in ids:
        if any(k in dataset.get(i).values for k in keys):
            usable.append(i)
        else:
            excluded.append({"id": i, "name": dataset.get(i).name,
                             "reason": "no data on any property that matters for this product."})
    if not usable:
        raise ApplicationError("None of the chosen ingredients has data on the properties that matter for "
                               f"{product.name.lower()}.")
    crits, missing = criteria(dataset, product, usable, overrides)
    if not crits:
        if any(m.get("turned_off") for m in missing):
            raise ApplicationError("Every property that has data is turned off. Set at least one to high, medium "
                                   "or low.")
        raise ApplicationError("The chosen ingredients do not share data on any property that matters for "
                               f"{product.name.lower()}, so they cannot be compared.")
    return usable, excluded, crits, missing


def quick_ranking(dataset, product):
    """{ingredient: {"rank", "of", "score"}} for a product at the central values, for overview pages."""
    try:
        ids, _, crits, _ = _prepare(dataset, product, list(dataset.ingredients))
    except ApplicationError:
        return {}
    sim = Simulation(dataset, ids, crits, n=1)
    order = list(np.argsort(-sim.score0))
    return {ids[a]: {"rank": pos + 1, "of": len(ids), "score": _r(sim.score0[a], 0)} for pos, a in enumerate(order)}


def _criteria_out(crits):
    return [{k: c[k] for k in ("property", "label", "unit", "direction", "importance", "default_importance", "why",
                               "plain")}
            | {"weight": _r(c["weight"], 4), "range": [_r(c["lo"], 3), _r(c["hi"], 3)]} for c in crits]


def _finish(blend):
    blend.pop("_scores", None)
    return blend


def simulate(dataset, product_id, recipe, batch_kg=None, importance=None):
    """What a stated blend does for the product, and how sure we are. Returns (result, explanation).

    importance: optional {property: high/medium/low/off} replacing the product's default importance.
    """
    product = _product(product_id)
    shares = normalise_recipe(dataset, recipe)
    overrides = importance_overrides(product, importance)
    changes = _importance_changes(product, overrides)
    ids, excluded, crits, missing = _prepare(dataset, product, list(shares), overrides)
    if excluded:
        raise ApplicationError("There is no usable data on " + _join(e["name"] for e in excluded)
                               + " for this product.")
    sim = Simulation(dataset, ids, crits)
    blend = evaluate_blend(dataset, sim, shares, batch_kg)
    ranking = _ranking(dataset, sim)
    certainty = _blend_certainty(dataset, sim, blend, missing, changes)
    where = f" in {dataset.location}" if dataset.location else ""
    headline = f"Your blend scores {_whole(blend['score']['value'])} out of 100 for {product.name.lower()}{where}."
    if len(blend["components"]) > 1:
        top = max(blend["components"], key=lambda c: c["points"])
        headline += f" {top['name']} adds the most."
    if blend["protein"]:
        headline += f" Protein: {blend['protein']['value']:.0f} g per 100 g (dry)."
    result = {
        "engine": ENGINE, "kind": "simulation", "product": PR.to_dict(product), "location": dataset.location,
        "batch_kg": batch_kg, "criteria": _criteria_out(crits), "not_scored": missing,
        "importance_changes": changes, "blend": _finish(blend), "ranking": ranking, "certainty": certainty,
    }
    explanation = {"headline": headline, "binding": _binding_text(product, blend)}
    return result, explanation


def _blend_reasons(blend, ranking, cost_step):
    """Why the recommended blend has these shares, in a few plain sentences."""
    capped = [c for c in blend["components"] if c["at_limit"]]
    rest = [c for c in blend["components"] if not c["at_limit"]]
    out = []
    if capped:
        out.append(f"{_join(c['name'] for c in capped)} {'is' if len(capped) == 1 else 'are'} used up to the "
                   f"supplier maximum ({_join(_pct(c['max_share']) for c in capped)}); more would score higher.")
    if capped and rest:
        out.append(f"{_join(c['name'] for c in rest)} {'makes' if len(rest) == 1 else 'make'} up the remaining "
                   f"{_pct(sum(c['share'] for c in rest))}.")
    elif len(rest) > 1:
        minor = rest[1:]
        one = len(minor) == 1
        out.append(f"{_join(c['name'] for c in minor)} {'is' if one else 'are'} added because "
                   + (f"{'it costs' if one else 'they cost'} less and the score stays about the same." if cost_step
                      else f"{'it makes' if one else 'they make'} the blend more reliable across the test runs."))
    unused = [r["name"] for r in ranking if r["id"] not in blend["shares"]]
    if unused:
        out.append(f"{_join(unused)} {'is' if len(unused) == 1 else 'are'} not used: "
                   f"{'it' if len(unused) == 1 else 'they'} would lower the score or raise the cost.")
    out.append("This blend stays strongest across the test runs"
               + (", and it is the cheapest of the blends that do about as well." if cost_step else "."))
    return out


def optimise(dataset, product_id, ingredient_ids, batch_kg, budget=None, baseline=None, importance=None):
    """Rank the chosen proteins for the product and find the best blend. Returns (result, explanation).

    importance: optional {property: high/medium/low/off} replacing the product's default importance.
    """
    product = _product(product_id)
    overrides = importance_overrides(product, importance)
    changes = _importance_changes(product, overrides)
    if not ingredient_ids:
        raise ApplicationError("Choose at least one plant protein.")
    ids = []
    for i in dict.fromkeys(ingredient_ids):
        if i not in dataset.ingredients:
            raise ApplicationError(f"Unknown ingredient '{i}'.")
        ids.append(i)
    if batch_kg is None or batch_kg <= 0:
        raise ApplicationError("Batch size must be a positive number of kilograms.")
    base_shares = None
    if baseline:
        base_shares = normalise_recipe(dataset, baseline)
        if not set(base_shares) <= set(ids):
            raise ApplicationError("The blend to compare uses proteins that are not selected.")
    ids, excluded, crits, missing = _prepare(dataset, product, ids, overrides)
    sim = Simulation(dataset, ids, crits)
    ranking = _ranking(dataset, sim)
    certainty = ranking_certainty(dataset, sim, ranking, missing, changes)

    notes = []
    upper, limit_source = [], {}
    for i in ids:
        p = dataset.get(i).price
        u = p.max_share if p is not None and p.max_share is not None else 1.0
        upper.append(u)
        limit_source[i] = ("supplier maximum in this state" if p is not None and p.kind != "predicted"
                           and p.max_share is not None else
                           "usual supplier maximum" if p is not None and p.max_share is not None else "")
    if sum(upper) < 1 - 1e-9:
        if len(ids) > 1:
            notes.append("The supplier maximums of the chosen proteins add up to less than 100%, so they were not "
                         "used.")
        upper = [1.0] * len(ids)
        limit_source = {i: "" for i in ids}
    prices = [dataset.get(i).price.value if dataset.get(i).price else None for i in ids]
    have_prices = all(p is not None for p in prices)
    budget_per_kg = budget / batch_kg if budget else None
    if budget_per_kg is not None and not have_prices:
        notes.append("Your budget was not used: some proteins have no price here.")
    shares, info = robust_blend(sim, upper, prices if have_prices else None, budget_per_kg)

    base = {"engine": ENGINE, "kind": "optimisation", "product": PR.to_dict(product), "location": dataset.location,
            "batch_kg": batch_kg, "budget": budget, "criteria": _criteria_out(crits), "not_scored": missing,
            "importance_changes": changes, "ranking": ranking, "certainty": certainty,
            "excluded_ingredients": excluded, "notes": notes}
    if shares is None:
        cheapest = _cheapest(prices, upper) if have_prices else None
        reason = (f"The cheapest possible blend of these proteins costs {_naira(cheapest)} per kg, "
                  f"{_naira(cheapest * batch_kg)} for {batch_kg:,.0f} kg. That is more than your budget of "
                  f"{_naira(budget)}." if cheapest is not None and budget else
                  "No blend could be found for these choices.")
        result = {**base, "feasible": False, "reason": reason}
        explanation = {"headline": "No blend fits your budget." if budget else "No blend could be found.",
                       "reasons": [reason],
                       "suggestions": ["Raise the budget or make a smaller batch."] if budget else []}
        return result, explanation

    share_map = {i: float(s) for i, s in zip(ids, shares)}
    blend = evaluate_blend(dataset, sim, share_map, batch_kg)
    for comp in blend["components"]:
        a = ids.index(comp["id"])
        comp["max_share"] = _r(upper[a], 4)
        comp["at_limit"] = bool(upper[a] < 1 and share_map[comp["id"]] >= upper[a] - 1e-6)
        comp["limit_source"] = limit_source[comp["id"]]
    opt_scores = blend["_scores"]
    comparison = None
    if base_shares:
        theirs = evaluate_blend(dataset, sim, base_shares, batch_kg)
        gain = blend["score"]["value"] - theirs["score"]["value"]
        p_better = float((opt_scores > theirs["_scores"] + 1e-9).mean())
        mine, yours = _whole(blend["score"]["value"]), _whole(theirs["score"]["value"])
        text = (f"The recommended blend scores {mine}, your blend {yours} ({'+' if mine >= yours else ''}"
                f"{mine - yours} points). It does better {_share_of_sims(p_better, sim.n)}.")
        comparison = {"baseline": _finish(theirs), "score_gain": _r(gain, 1), "p_better": _r(p_better, 4)}
        if theirs["cost"].get("available") and blend["cost"].get("available"):
            diff = blend["cost"]["value"] - theirs["cost"]["value"]
            comparison["cost_change"] = _r(diff)
            text += (" It costs about the same." if abs(diff) < 1 else
                     f" It costs {_naira(abs(diff))} per kg {'less' if diff < 0 else 'more'}.")
        comparison["text"] = text
    plan = lab_plan(dataset, product, ranking, blend, missing, ids, overrides) if len(ids) > 1 else None

    top = ranking[0]
    where = f" in {dataset.location}" if dataset.location else ""
    cost = f", about {_naira(blend['cost']['value'])} per kg" if blend["cost"].get("available") else ""
    if len(ids) == 1:
        headline = f"Only {top['name']} was chosen for {product.name.lower()}{where}, so the blend is {top['name']} alone{cost}."
        reasons = []
    else:
        recipe_text = _join(f"{_pct(c['share'])} {c['name']}" for c in blend["components"])
        headline = (f"For {product.name.lower()}{where}, {top['name']} is the best of your {len(ids)} proteins. "
                    f"Recommended blend: {recipe_text}{cost}.")
        reasons = _blend_reasons(blend, ranking, info.get("cost_step"))
    result = {**base, "feasible": True, "blend": _finish(blend), "comparison": comparison, "lab_plan": plan,
              "robust_score": blend["score"]["worst_10pct"], "best_robust_score": _r(info["cvar"], 1)}
    explanation = {"headline": headline, "blend": reasons, "binding": _binding_text(product, blend),
                   "comparison": comparison["text"] if comparison else None}
    return result, explanation


# ---------------------------------------------------------------- evaluation
def backtest(dataset, min_ingredients=3):
    """Check rankings for states without data: rank each well-measured state's ingredients from its own
    measurements, then again with that state's data removed (so every value is estimated from other
    states), and compare. Returns one row per product and state."""
    from scipy.stats import kendalltau
    from ai_engine.locations import NIGERIA_STATES
    rows = []
    for product in PR.PRODUCTS.values():
        for state in NIGERIA_STATES:
            full = dataset.view(state)
            keys = [r.property for r in product.relevant if _has_records(dataset, r.property)]
            ids = [i for i, ing in full.ingredients.items()
                   if all(k in ing.values and ing.values[k].location_status == "selected" for k in keys)]
            if len(ids) < min_ingredients:
                continue
            crits, _ = criteria(dataset, product, ids)
            own = _ranking(full, Simulation(full, ids, crits))
            blind_ds = dataset.without_state(state).view(state)
            blind_crits = [{**c} for c in crits]  # same scales, so the scores are comparable
            blind = _ranking(blind_ds, Simulation(blind_ds, ids, blind_crits))
            own_order = [r["id"] for r in own]
            blind_order = [r["id"] for r in blind]
            tau = kendalltau([own_order.index(i) for i in ids], [blind_order.index(i) for i in ids])[0]
            p_true_top = next(r["p_first"] for r in blind if r["id"] == own_order[0])
            rows.append({"product": product.id, "state": state, "n_ingredients": len(ids),
                         "own_top": dataset.get(own_order[0]).name, "estimated_top": dataset.get(blind_order[0]).name,
                         "top_agrees": own_order[0] == blind_order[0], "kendall_tau": _r(tau, 3),
                         "p_first_of_true_top": p_true_top,
                         "own_order": [dataset.get(i).name for i in own_order],
                         "estimated_order": [dataset.get(i).name for i in blind_order]})
    return rows
