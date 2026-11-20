"""Formulation Optimisation by linear programming (SciPy linprog, HiGHS).

Unknowns: x_c, the mass fraction of each component in the final mix:
protein ingredients, and optionally added water and edible oil.

Objective: minimise cost per kg of mix, sum(price_c * x_c). Water and oil are
costed only if the user gives their price.

Constraints (all linear):
    sum(x_c) = 1
    nutrition targets on the final mix, by mass balance on an as-is basis:
        sum(N_cj * x_c) >= min_j  /  <= max_j
    share of each protein ingredient WITHIN the protein blend:
        min_i * S <= x_i <= max_i * S,  S = sum of protein ingredient fractions
    stock: x_i * batch_kg <= reported stock in the selected state
    water and oil: min <= x_water, x_oil <= max
    budget: batch_kg * cost <= budget
    optional: x_water <= sum(WHC_i * x_i)  - added water no more than the protein
        ingredients are each reported to hold. This ASSUMES no interaction
        between ingredients and is only applied if the user asks for it.

Functional targets are not linear in the proportions and are not forced into
the linear program. They are checked afterwards on the optimal mix, only where
a validated estimate exists (see Blend Simulation).

If no mix satisfies every constraint, the result is "infeasible" with a
diagnosis of which constraints conflict. No formulation is invented.
"""

import numpy as np
from scipy.optimize import linprog

from ai_engine import properties as P
from ai_engine.blend import OIL, SUPPORTING, WATER, component_composition

EPS = 1e-7


class OptimiseError(ValueError):
    pass


def _solve(c, rows, bounds, sense=1):
    A_ub = [r["a"] for r in rows] or None
    b_ub = [r["b"] for r in rows] or None
    n = len(c)
    return linprog(sense * np.asarray(c, dtype=float),
                   A_ub=np.array(A_ub, dtype=float) if A_ub else None,
                   b_ub=np.array(b_ub, dtype=float) if b_ub else None,
                   A_eq=np.ones((1, n)), b_eq=np.array([1.0]), bounds=bounds, method="highs")


def _name(dataset, cid):
    return SUPPORTING[cid]["name"] if cid in SUPPORTING else dataset.get(cid).name


def _percent(v, name, allow_none=True):
    if v is None:
        if allow_none:
            return None
        raise OptimiseError(f"{name} is required.")
    v = float(v)
    if not 0 <= v <= 100:
        raise OptimiseError(f"{name} must be between 0 and 100%.")
    return v / 100


def optimise(dataset, ingredient_ids, batch_kg, targets=None, limits=None, water=None, oil=None,
             budget=None, water_price=None, oil_price=None, limit_water_to_holding=False):
    """Find the lowest-cost mix that satisfies every constraint.

    targets: {property: {"min": v, "max": v}} on the final mix (property units per 100 g of mix).
    limits:  {ingredient: {"min": %, "max": %}} share of the protein blend; a blank max uses the
             supply-lot limit for the location, then the ingredient default.
    water / oil: {"min": %, "max": %} of the final mix; None means not used.
    """
    targets = targets or {}
    limits = limits or {}
    if not ingredient_ids:
        raise OptimiseError("Choose at least one protein ingredient.")
    if batch_kg is None or batch_kg <= 0:
        raise OptimiseError("Batch size must be a positive number of kilograms.")
    if budget is not None and budget <= 0:
        raise OptimiseError("Budget must be a positive amount.")
    for key, t in targets.items():
        pdef = P.PROPERTIES.get(key)
        if pdef is None:
            raise OptimiseError(f"Unknown property '{key}'.")
        if not pdef.optimisable:
            raise OptimiseError(f"{pdef.label} cannot be optimised directly: it does not add up in a simple "
                                "way when ingredients are mixed. Set it as a target in Blend Simulation instead.")
        if t.get("min") is not None and t.get("max") is not None and t["min"] > t["max"]:
            raise OptimiseError(f"The minimum for {pdef.label} is above its maximum.")

    # ---- components that can enter the solver
    needed = {"protein"} | set(targets)
    used, excluded, composition = [], [], {}
    for iid in dict.fromkeys(ingredient_ids):
        ing = dataset.get(iid)
        if ing is None:
            raise OptimiseError(f"Unknown ingredient '{iid}'.")
        comp, problems = {}, []
        for key in needed:
            value, info = component_composition(dataset, iid, key)
            if value is None:
                problems.append(info)
            else:
                comp[key] = value
        if not ing.price:
            where = f" in {dataset.location}" if dataset.location else ""
            problems.append("no verified price" + where)
        if problems:
            excluded.append({"id": iid, "name": ing.name, "reason": "; ".join(problems) + "."})
        else:
            used.append(iid)
            composition[iid] = comp
    if not used:
        return {"feasible": False, "status": "no_ingredients", "batch_kg": batch_kg,
                "excluded_ingredients": excluded,
                "message": "None of the chosen ingredients has the verified data needed here.",
                "diagnosis": {"reasons": [f"{e['name']}: {e['reason']}" for e in excluded], "suggestions": [
                    "Choose a state with price and composition data, or choose other ingredients."]}}

    support = []
    for cid, spec in ((WATER, water), (OIL, oil)):
        if spec is None:
            continue
        lo = _percent(spec.get("min"), f"Minimum {SUPPORTING[cid]['name'].lower()}") or 0.0
        hi = _percent(spec.get("max"), f"Maximum {SUPPORTING[cid]['name'].lower()}")
        hi = 1.0 if hi is None else hi
        if lo > hi:
            raise OptimiseError(f"Minimum {SUPPORTING[cid]['name'].lower()} is above the maximum.")
        if hi > 0:
            support.append((cid, lo, hi))
            composition[cid] = {k: component_composition(dataset, cid, k)[0] for k in needed}

    comps = used + [c for c, _, _ in support]
    idx = {c: k for k, c in enumerate(comps)}
    n = len(comps)
    bounds = [(0.0, 1.0)] * len(used) + [(lo, hi) for _, lo, hi in support]

    prices = []
    unpriced = []
    for c in comps:
        if c == WATER:
            prices.append(water_price or 0.0)
            if water_price is None:
                unpriced.append("added water")
        elif c == OIL:
            prices.append(oil_price or 0.0)
            if oil_price is None:
                unpriced.append("edible oil")
        else:
            prices.append(dataset.get(c).price.value)

    rows = []

    def add(rid, label, group, a, b, kind, limit=None, prop=None):
        rows.append({"id": rid, "label": label, "group": group, "a": a, "b": b, "kind": kind,
                     "limit": limit, "property": prop})

    # nutrition targets
    for key, t in targets.items():
        coef = [composition[c][key] for c in comps]
        label = P.get(key).label
        unit = P.get(key).unit.replace("100 g", "100 g mix")
        if t.get("min") is not None:
            add(f"{key}_min", f"{label} at least {t['min']:g} {unit}", "nutrition",
                [-v for v in coef], -t["min"], "min", t["min"], key)
        if t.get("max") is not None:
            add(f"{key}_max", f"{label} at most {t['max']:g} {unit}", "nutrition", coef, t["max"], "max", t["max"], key)

    # share limits within the protein blend, and stock
    limit_source = {}
    for iid in used:
        ing = dataset.get(iid)
        lim = limits.get(iid, {})
        lo = _percent(lim.get("min"), f"Minimum share of {ing.name}")
        hi = _percent(lim.get("max"), f"Maximum share of {ing.name}")
        lo = ing.default_min_inclusion if lo is None else lo
        if hi is not None:
            src = "your limit"
        elif ing.price and ing.price.max_share is not None:
            hi, src = ing.price.max_share, "supplier limit for this location"
        else:
            hi, src = ing.default_max_inclusion, "no limit set"
        if lo > hi:
            raise OptimiseError(f"Minimum share of {ing.name} is above its maximum.")
        limit_source[iid] = {"min": lo, "max": hi, "source": src}
        if hi < 1:
            a = [0.0] * n
            for j in used:
                a[idx[j]] -= hi
            a[idx[iid]] += 1
            add(f"{iid}_max", f"{ing.name} at most {hi:.0%} of the protein ingredients", "share", a, 0.0, "max", hi)
        if lo > 0:
            a = [0.0] * n
            for j in used:
                a[idx[j]] += lo
            a[idx[iid]] -= 1
            add(f"{iid}_min", f"{ing.name} at least {lo:.0%} of the protein ingredients", "share", a, 0.0, "min", lo)
        stock = ing.price.availability_kg if ing.price else None
        if stock is not None:
            a = [0.0] * n
            a[idx[iid]] = 1.0
            add(f"{iid}_stock", f"{ing.name} within the {stock:,.0f} kg reported in stock", "stock",
                a, stock / batch_kg, "max", stock)

    if budget is not None:
        add("budget", f"Batch cost at most \u20a6{budget:,.0f}", "budget", list(prices), budget / batch_kg, "max", budget)

    holding_key = None
    if limit_water_to_holding and WATER in idx:
        for key in ("water_holding_capacity", "water_absorption"):
            if all(key in dataset.get(i).values for i in used):
                holding_key = key
                break
        if holding_key:
            a = [0.0] * n
            a[idx[WATER]] = 1.0
            for i in used:
                a[idx[i]] -= dataset.get(i).values[holding_key].value
            add("water_holding", f"Added water no more than the ingredients hold ({P.get(holding_key).label.lower()}, "
                "assuming no interaction)", "water_holding", a, 0.0, "max")

    base = {"batch_kg": batch_kg, "excluded_ingredients": excluded, "limits": limit_source,
            "solver": "SciPy linprog (HiGHS)", "unpriced_components": unpriced,
            "water_holding_rule": ({"applied": holding_key is not None, "property": holding_key}
                                   if limit_water_to_holding else None)}

    res = _solve(prices, rows, bounds)
    if res.status == 2:
        return {**base, "feasible": False, "status": "infeasible",
                "message": "No mix of the chosen ingredients meets every requirement.",
                "diagnosis": _diagnose(dataset, comps, prices, rows, bounds, used, limit_source, batch_kg)}
    if res.status != 0:
        return {**base, "feasible": False, "status": "solver_error",
                "message": f"The calculation stopped without a result: {res.message}",
                "diagnosis": {"reasons": [res.message], "suggestions": []}}

    x = np.clip(res.x, 0, None)
    x = x / x.sum()
    marginals = list(res.ineqlin.marginals) if rows else []
    protein_total = sum(x[idx[i]] for i in used)
    constraint_status = []
    for r, m in zip(rows, marginals):
        lhs = float(np.dot(r["a"], x))
        slack = r["b"] - lhs
        binding = abs(slack) < 1e-6 * max(1.0, abs(r["b"]))
        entry = {"id": r["id"], "label": r["label"], "group": r["group"], "binding": bool(binding),
                 "satisfied": bool(slack >= -1e-6), "cost_effect_per_kg": round(abs(float(m)), 4)}
        if r["group"] == "nutrition":
            entry["achieved"] = round(-lhs if r["kind"] == "min" else lhs, 4)
            entry["limit"] = r["limit"]
        constraint_status.append(entry)

    components = []
    for c in comps:
        share = float(x[idx[c]])
        price = prices[idx[c]]
        item = {"id": c, "name": _name(dataset, c), "kind": "supporting" if c in SUPPORTING else "protein",
                "share": round(share, 6), "kg": round(share * batch_kg, 3),
                "price_ngn_per_kg": round(price, 2) if (c not in SUPPORTING or price) else None,
                "cost_ngn": round(share * batch_kg * price, 2)}
        if c not in SUPPORTING:
            lim = limit_source[c]
            ps = share / protein_total if protein_total > 0 else 0
            item.update({"share_of_protein_blend": round(ps, 6), "min_share": lim["min"], "max_share": lim["max"],
                         "limit_source": lim["source"],
                         "at_max": bool(lim["max"] < 1 and abs(ps - lim["max"]) < 1e-5),
                         "at_min": bool(lim["min"] > 0 and abs(ps - lim["min"]) < 1e-5),
                         "protein_as_is": round(composition[c]["protein"], 3),
                         "cost_per_protein_point": round(price / composition[c]["protein"], 3)
                         if composition[c]["protein"] > 0 else None})
        else:
            lo, hi = next((lo, hi) for cid, lo, hi in support if cid == c)
            item.update({"min_share": lo, "max_share": hi, "at_max": bool(hi < 1 and abs(share - hi) < 1e-6),
                         "at_min": bool(lo > 0 and abs(share - lo) < 1e-6)})
        components.append(item)
    cost_per_kg = float(np.dot(prices, x))
    return {
        **base, "feasible": True, "status": "optimal", "message": "Lowest-cost mix found.",
        "components": sorted(components, key=lambda b: (b["kind"] != "protein", -b["share"])),
        "recipe": {c["id"]: c["share"] for c in components if c["share"] > 1e-9},
        "cost_per_kg": round(cost_per_kg, 2), "batch_cost": round(cost_per_kg * batch_kg, 2),
        "constraints": constraint_status,
    }


def _diagnose(dataset, comps, prices, rows, bounds, used, limit_source, batch_kg):
    """Explain, in plain words, why no mix satisfies every requirement."""
    reasons, suggestions = [], []
    max_sum = sum(limit_source[i]["max"] for i in used)
    if max_sum < 1 - EPS:
        reasons.append(f"The maximum shares of the chosen protein ingredients add up to only {max_sum:.0%}, so they "
                       "cannot make up a full protein blend.")
        suggestions.append("Raise some maximum shares or add another ingredient.")
    structural = [r for r in rows if r["group"] in ("share", "stock")]
    if _solve([0.0] * len(comps), structural, bounds).status == 2 and not reasons:
        stock_rows = [r for r in rows if r["group"] == "stock"]
        if stock_rows:
            reasons.append(f"There is not enough reported stock in this state for a {batch_kg:,.0f} kg batch "
                           "within the share limits.")
            suggestions.append("Reduce the batch size or choose another state.")
    if not reasons:
        for r in rows:
            if r["group"] not in ("nutrition", "budget", "water_holding"):
                continue
            c = [-v for v in r["a"]] if r["kind"] == "min" else list(r["a"])
            # Best achievable value of this quantity under the structural limits alone.
            best = _solve(c, structural, bounds, sense=-1 if r["kind"] == "min" else 1)
            if best.status != 0:
                continue
            if r["kind"] == "min":
                top = float(-best.fun)
                if top < r["limit"] - EPS:
                    reasons.append(f"The highest {P.get(r['property']).label.lower()} possible with these ingredients "
                                   f"and limits is {top:.2f}, below your minimum of {r['limit']:g}.")
                    suggestions.append(f"Lower the minimum to {top:.2f} or less, reduce added water, or allow more "
                                       "of the richest ingredient.")
            elif r["group"] == "budget":
                cheapest = _solve(prices, [o for o in rows if o is not r], bounds)
                if cheapest.status == 0:
                    need = float(cheapest.fun) * batch_kg
                    reasons.append(f"The cheapest mix that meets the other requirements costs \u20a6{need:,.0f}, more "
                                   f"than your budget of \u20a6{r['limit']:,.0f}.")
                    suggestions.append(f"Increase the budget to at least \u20a6{need:,.0f} or reduce the batch size.")
            elif r["group"] == "nutrition":
                low = float(best.fun)
                if low > r["limit"] + EPS:
                    reasons.append(f"The lowest {P.get(r['property']).label.lower()} possible is {low:.2f}, above "
                                   f"your maximum of {r['limit']:g}.")
                    suggestions.append(f"Raise the maximum to at least {low:.2f}.")
    if not reasons:
        for r in rows:
            others = [o for o in rows if o is not r]
            if _solve([0.0] * len(comps), others, bounds).status == 0:
                reasons.append(f"'{r['label']}' cannot be met together with your other requirements.")
                suggestions.append(f"Relaxing '{r['label']}' would make a mix possible.")
        if not reasons:
            reasons.append("Several requirements conflict with each other.")
            suggestions.append("Relax the nutrition targets, limits or budget.")
    return {"reasons": reasons, "suggestions": suggestions}
