"""Blend Simulation: properties of a user-defined recipe.

A recipe is a list of components, each a percentage of the total mix:
- protein ingredients (the five raw materials, used as flours);
- supporting components: added water and edible oil.

Nutritional composition is a mass balance on a consistent basis. Ingredient
values are stored on a dry-matter basis; each is converted to the ingredient
"as it is" using that ingredient's moisture, then

    N_j = sum_c  w_c * N_cj       (per 100 g of the final mix)

where w_c is the mass fraction of component c. Added water is 100 g water per
100 g; refined edible oil is taken as 100 g fat per 100 g and nothing else.
The dry protein blend (protein ingredients only, dry basis) is also reported.

Functional properties are NOT averaged. A value for the mix is given only if
the prediction logic has a validated model built from measured blends;
otherwise the ingredients' own values are shown for reference.
"""

from ai_engine import properties as P
from ai_engine import products as PR
from ai_engine.dataset import STATUS_RANK

# WHO/FAO/UNU (2007) amino acid scoring pattern for adults, g per 100 g protein.
AA_REFERENCE = {"leucine": 5.9, "lysine": 4.5, "sulphur_amino_acids": 2.2, "threonine": 2.3, "tryptophan": 0.6}
AA_REFERENCE_SOURCE = "WHO/FAO/UNU (2007) Protein and amino acid requirements in human nutrition, TRS 935"

WATER, OIL = "water", "oil"
SUPPORTING = {
    WATER: {"name": "Added water", "composition": {"moisture": 100.0},
            "note": "Water: 100 g water per 100 g."},
    OIL: {"name": "Edible oil", "composition": {"fat": 100.0},
          "note": "Refined vegetable oil: taken as 100 g fat per 100 g, with no protein, fibre, ash or water."},
}
# Properties computed for the final mix (mass balance).
MIX_KEYS = [p.key for p in P.PROPERTIES.values() if p.blend_rule == P.LINEAR]
SHARE_SUM_TOLERANCE = 0.05  # percentage points


class BlendError(ValueError):
    pass


def _num(value, name, lo=None, hi=None):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise BlendError(f"{name} must be a number.")
    if lo is not None and v < lo:
        raise BlendError(f"{name} cannot be less than {lo:g}.")
    if hi is not None and v > hi:
        raise BlendError(f"{name} cannot be more than {hi:g}.")
    return v


def normalise_recipe(dataset, recipe):
    """Validate a recipe {component: percent of total mix}; return fractions summing to 1."""
    if not isinstance(recipe, dict) or not recipe:
        raise BlendError("Choose at least one protein ingredient.")
    clean = {}
    for cid, pct in recipe.items():
        if cid not in dataset.ingredients and cid not in SUPPORTING:
            raise BlendError(f"Unknown ingredient '{cid}'.")
        v = _num(pct, f"The amount of {cid}", 0, 100)
        if v > 0:
            clean[cid] = v
    if not any(c in dataset.ingredients for c in clean):
        raise BlendError("Choose at least one protein ingredient with an amount above 0%.")
    total = sum(clean.values())
    if abs(total - 100) > SHARE_SUM_TOLERANCE:
        raise BlendError(f"The amounts must add up to 100% of the mix (currently {total:.1f}%).")
    return {k: v / total for k, v in clean.items()}


def protein_shares(recipe):
    """Shares of each protein ingredient within the protein blend (sum 1)."""
    prot = {k: v for k, v in recipe.items() if k not in SUPPORTING}
    total = sum(prot.values())
    return {k: v / total for k, v in prot.items()}


def as_is_value(ing, key, dataset=None):
    """(value per 100 g of the ingredient as it is, record) or (None, reason)."""
    v = ing.values.get(key)
    if v is None:
        return None, f"no {P.get(key).label.lower()} value for {ing.name}" + _missing_reason(dataset, ing.id, key)
    pdef = P.get(key)
    if not pdef.basis_convertible:
        return v.value, v
    m = ing.values.get("moisture")
    if m is None:
        return None, (f"no moisture value for {ing.name}" + _missing_reason(dataset, ing.id, "moisture")
                      + ", so its dry-matter values cannot be converted")
    return v.value * (100 - m.value) / 100, v


def _missing_reason(dataset, iid, key):
    """Why a property has no value for an ingredient at the selected location."""
    if dataset is None:
        return ""
    if not dataset.records.get((iid, key)):
        return " in any Nigerian state"
    if dataset.location:
        loc = getattr(dataset, "location_refusals", {}).get((iid, key))
        if loc:
            return (f" in {dataset.location}: there are no measurements there, and other states' data cannot "
                    f"stand in for it ({loc})")
    why = getattr(dataset, "prediction_refusals", {}).get((iid, key))
    return f" ({why})" if why else ""


def component_composition(dataset, cid, key):
    """Per-100 g value of a recipe component, with provenance, or (None, reason)."""
    if cid in SUPPORTING:
        return SUPPORTING[cid]["composition"].get(key, 0.0), {"kind": "definition", "status": "definition",
                                                              "location_status": "", "note": SUPPORTING[cid]["note"]}
    ing = dataset.get(cid)
    value, rec = as_is_value(ing, key, dataset)
    if value is None:
        return None, rec
    moisture = ing.values.get("moisture")
    statuses = [rec.status] + ([moisture.status] if P.get(key).basis_convertible and moisture else [])
    return value, {"kind": "predicted" if "predicted" in statuses else "measured",
                   "status": min(statuses, key=STATUS_RANK.get), "location_status": rec.location_status,
                   "dry_value": rec.value if P.get(key).basis_convertible else None}


def mix_property(dataset, recipe, key):
    """Mass-balance value of a property for the final mix (per 100 g of mix)."""
    pdef = P.get(key)
    base = {"property": key, "label": pdef.label, "unit": pdef.unit.replace("100 g", "100 g mix"),
            "group": pdef.group, "plain": pdef.plain}
    inputs, contributions, missing = [], {}, []
    for cid, w in recipe.items():
        value, info = component_composition(dataset, cid, key)
        if value is None:
            missing.append(info)
            continue
        contributions[cid] = w * value
        name = SUPPORTING[cid]["name"] if cid in SUPPORTING else dataset.get(cid).name
        inputs.append({"id": cid, "name": name, "value_as_is": round(value, 4), **info})
    if missing:
        return {**base, "available": False, "kind": "unavailable",
                "reason": "Cannot calculate: " + "; ".join(missing) + "."}
    value = sum(contributions.values())
    kinds = {i["kind"] for i in inputs}
    return {
        **base, "available": True, "kind": "calculated", "value": round(value, 4),
        "contributions": {k: round(v, 4) for k, v in contributions.items()},
        "inputs": inputs,
        "uses_predicted": [i["name"] for i in inputs if i["kind"] == "predicted"],
        "method": "mass balance per 100 g of mix" + (" (some inputs estimated)" if "predicted" in kinds else ""),
    }


def dry_blend_property(dataset, shares, key):
    """Dry-matter value of the protein blend alone (weighted average of dry values)."""
    pdef = P.get(key)
    if not pdef.basis_convertible:
        return None
    vals = {}
    for iid, s in shares.items():
        v = dataset.get(iid).values.get(key)
        if v is None:
            return None
        vals[iid] = v
    return round(sum(shares[i] * v.value for i, v in vals.items()), 4)


def amino_acid_profile(dataset, shares):
    """Protein-weighted amino acids of the protein blend, and the amino acid score."""
    out = {}
    for key, pdef in P.PROPERTIES.items():
        if pdef.blend_rule != P.PROTEIN_WEIGHTED:
            continue
        ings = [dataset.get(i) for i in shares]
        if any(key not in i.values or "protein" not in i.values for i in ings):
            out[key] = {"property": key, "label": pdef.label, "available": False,
                        "reason": "Not documented for every selected ingredient."}
            continue
        tp = sum(shares[i.id] * i.values["protein"].value for i in ings)
        value = sum(shares[i.id] * i.values["protein"].value * i.values[key].value for i in ings) / tp
        out[key] = {"property": key, "label": pdef.label, "unit": pdef.unit, "available": True,
                    "kind": "calculated", "value": round(value, 3)}
    get = lambda k: out[k]["value"] if out.get(k, {}).get("available") else None
    observed = {k: get(k) for k in ("leucine", "lysine", "threonine", "tryptophan")}
    met, cys = get("methionine"), get("cysteine")
    observed["sulphur_amino_acids"] = met + cys if met is not None and cys is not None else None
    ratios = {k: observed[k] / AA_REFERENCE[k] for k in AA_REFERENCE if observed[k] is not None}
    score = None
    if ratios:
        lim = min(ratios, key=ratios.get)
        score = {"score": round(min(ratios[lim], 1.0) * 100, 1), "limiting_amino_acid": lim,
                 "reference": AA_REFERENCE_SOURCE,
                 "note": "Not corrected for digestibility."}
    return out, score


def functional_profile(dataset, predictor, shares, product):
    """Functional properties: blend estimate if validated, else the ingredients' own values."""
    out = {}
    for key, pdef in P.PROPERTIES.items():
        if pdef.blend_rule != P.NONLINEAR:
            continue
        rel = product.relevance(key) if product else None
        est = predictor.estimate(key, shares)
        ingredient_values = []
        for iid in shares:
            v = dataset.get(iid).values.get(key)
            if v is not None:
                ingredient_values.append({"id": iid, "name": dataset.get(iid).name, "value": round(v.value, 3),
                                          "kind": v.kind, "location_status": v.location_status})
        entry = {"property": key, "label": pdef.label, "unit": pdef.unit, "plain": pdef.plain,
                 "relevant": rel is not None, "direction": rel.direction if rel else None,
                 "why": rel.why if rel else "", "ingredient_values": ingredient_values,
                 "missing_for": [dataset.get(i).name for i in shares if key not in dataset.get(i).values]}
        if est.get("estimated"):
            entry.update({"available": True, "kind": "predicted", "value": est["value"], "low": est["low"],
                          "high": est["high"], "model": est})
        else:
            entry.update({"available": False, "kind": "unavailable",
                          "reason": "No value for the mix: these properties change when ingredients are combined, "
                                    "and there is not yet enough measured data on blends to estimate them.",
                          "technical_reason": est.get("reason", "")})
        out[key] = entry
    return out


def holding_checks(dataset, recipe):
    """Indicative comparison of added water / oil with what the ingredients absorb.

    Compares added water (or oil) per gram of protein ingredient with the
    lowest and highest absorption capacity among the selected ingredients.
    It does not predict the mix's capacity, which can differ.
    """
    checks = []
    prot = {k: v for k, v in recipe.items() if k not in SUPPORTING}
    base = sum(prot.values())
    for comp, keys, noun in ((WATER, ("water_holding_capacity", "water_absorption"), "water"),
                             (OIL, ("oil_holding_capacity", "oil_absorption"), "oil")):
        added = recipe.get(comp, 0.0)
        if added <= 0 or base <= 0:
            continue
        ratio = added / base
        for key in keys:
            vals = [(dataset.get(i).name, dataset.get(i).values[key].value) for i in prot
                    if key in dataset.get(i).values]
            if len(vals) < len(prot):
                continue
            lo, hi = min(v for _, v in vals), max(v for _, v in vals)
            label = P.get(key).label.lower()
            if ratio > hi:
                level, text = "warning", (
                    f"You are adding {ratio:.2f} g of {noun} per g of protein ingredients, more than any of them is "
                    f"reported to take up ({label}: {lo:.2f}-{hi:.2f} g/g). Some {noun} may stay free and separate.")
            elif ratio < lo:
                level, text = "info", (
                    f"Added {noun} ({ratio:.2f} g per g of protein ingredients) is below the {label} of every "
                    f"selected ingredient ({lo:.2f}-{hi:.2f} g/g).")
            else:
                level, text = "info", (
                    f"Added {noun} ({ratio:.2f} g per g of protein ingredients) is within the range of the "
                    f"ingredients' {label} ({lo:.2f}-{hi:.2f} g/g).")
            checks.append({"component": comp, "property": key, "ratio": round(ratio, 3), "low": lo, "high": hi,
                           "level": level, "text": text + " This is a rough guide, not a prediction for the mix."})
            break
    return checks


def check_targets(targets, mix, functional):
    """Compare user targets with calculated / predicted values."""
    out = []
    for key, t in (targets or {}).items():
        lo, hi = t.get("min"), t.get("max")
        entry = mix.get(key) or functional.get(key)
        label = P.get(key).label
        if entry is None or not entry.get("available"):
            out.append({"property": key, "label": label, "min": lo, "max": hi, "status": "not_checked",
                        "text": f"{label}: cannot be checked, no value for the mix."})
            continue
        v = entry["value"]
        unit = entry.get("unit", "")
        if lo is not None and v < lo - 1e-9:
            status, text = "not_met", f"{label} is {v:.2f} {unit}, {lo - v:.2f} below your minimum of {lo:g}."
        elif hi is not None and v > hi + 1e-9:
            status, text = "not_met", f"{label} is {v:.2f} {unit}, {v - hi:.2f} above your maximum of {hi:g}."
        else:
            status, text = "met", f"{label} is {v:.2f} {unit}, within your target."
        out.append({"property": key, "label": label, "min": lo, "max": hi, "value": v, "status": status,
                    "kind": entry.get("kind"), "text": text})
    return out


def cost(dataset, recipe, batch_kg=None, water_price=None, oil_price=None):
    """Cost per kg of mix. Protein ingredients need a verified price for the location."""
    prices, missing, inputs = {}, [], []
    for cid, w in recipe.items():
        if cid == WATER:
            p = water_price
        elif cid == OIL:
            p = oil_price
        else:
            ing = dataset.get(cid)
            p = ing.price.value if ing.price else None
            if p is not None:
                inputs.append({"id": cid, "name": ing.name, "price": round(p, 2), "status": ing.price.status,
                               "date": ing.price.latest_date, "markets": ing.price.markets,
                               "location_status": ing.price.location_status})
        if p is None:
            missing.append(SUPPORTING[cid]["name"] if cid in SUPPORTING else dataset.get(cid).name)
            continue
        prices[cid] = p
    unpriced_protein = [m for m in missing if m not in (SUPPORTING[WATER]["name"], SUPPORTING[OIL]["name"])]
    if unpriced_protein:
        where = f" in {dataset.location}" if dataset.location else ""
        return {"available": False, "reason": "No verified price" + where + " for " + ", ".join(unpriced_protein) + ".",
                "inputs": inputs}
    per_kg = sum(recipe[c] * prices[c] for c in prices)
    out = {"available": True, "kind": "calculated", "value": round(per_kg, 2), "unit": "NGN/kg of mix",
           "contributions": {c: round(recipe[c] * prices[c], 2) for c in prices}, "inputs": inputs,
           "excludes": missing}
    if batch_kg:
        out["batch_cost"] = round(per_kg * batch_kg, 2)
    return out


def availability(dataset, recipe, batch_kg):
    """Compare the kg needed with the stock reported in the selected state."""
    if not batch_kg:
        return []
    out = []
    for cid, w in recipe.items():
        if cid in SUPPORTING:
            continue
        ing = dataset.get(cid)
        need = w * batch_kg
        stock = ing.price.availability_kg if ing.price else None
        if stock is None:
            out.append({"id": cid, "name": ing.name, "needed_kg": round(need, 2), "stock_kg": None,
                        "status": "unknown", "text": f"{ing.name}: no stock information for this location."})
        else:
            ok = need <= stock
            out.append({"id": cid, "name": ing.name, "needed_kg": round(need, 2), "stock_kg": stock,
                        "status": "ok" if ok else "short",
                        "text": f"{ing.name}: {need:,.1f} kg needed, {stock:,.0f} kg reported in stock"
                                + ("" if ok else " - not enough") + "."})
    return out


def simulate(dataset, predictor, recipe, product_id, batch_kg=None, targets=None,
             water_price=None, oil_price=None):
    product = PR.get(product_id)
    recipe = normalise_recipe(dataset, recipe)
    shares = protein_shares(recipe)
    mix = {key: mix_property(dataset, recipe, key) for key in MIX_KEYS}
    dry = {key: dry_blend_property(dataset, shares, key) for key in MIX_KEYS}
    amino, score = amino_acid_profile(dataset, shares)
    functional = functional_profile(dataset, predictor, shares, product)
    blend_cost = cost(dataset, recipe, batch_kg, water_price, oil_price)

    components = []
    for cid, w in sorted(recipe.items(), key=lambda kv: (kv[0] in SUPPORTING, -kv[1])):
        name = SUPPORTING[cid]["name"] if cid in SUPPORTING else dataset.get(cid).name
        components.append({"id": cid, "name": name, "kind": "supporting" if cid in SUPPORTING else "protein",
                           "share": round(w, 6), "share_of_protein_blend": round(shares.get(cid, 0), 6) or None,
                           "kg": round(w * batch_kg, 3) if batch_kg else None})

    contributions = []
    for cid, w in recipe.items():
        if cid in SUPPORTING:
            continue
        row = {"id": cid, "name": dataset.get(cid).name, "share": round(w, 6)}
        for key in ("protein", "fat", "fibre"):
            m = mix[key]
            if m.get("available") and m["value"] > 0:
                row[key] = m["contributions"][cid]
                row[f"{key}_pct"] = round(100 * m["contributions"][cid] / m["value"], 1)
        if blend_cost.get("available") and blend_cost["value"] > 0 and cid in blend_cost["contributions"]:
            row["cost"] = blend_cost["contributions"][cid]
            row["cost_pct"] = round(100 * blend_cost["contributions"][cid] / blend_cost["value"], 1)
        contributions.append(row)

    result = {
        "product": PR.to_dict(product), "location": dataset.location,
        "components": components, "recipe": {k: round(v, 6) for k, v in recipe.items()},
        "protein_shares": {k: round(v, 6) for k, v in shares.items()},
        "mix": mix, "dry_protein_blend": dry, "amino_acids": amino, "amino_acid_score": score,
        "functional": functional, "holding_checks": holding_checks(dataset, recipe),
        "targets": check_targets(targets, mix, functional), "cost": blend_cost,
        "availability": availability(dataset, recipe, batch_kg), "contributions": contributions,
        "supporting_notes": [SUPPORTING[c]["note"] for c in recipe if c in SUPPORTING],
        "disclaimer": PR.DISCLAIMER,
    }
    if batch_kg:
        result["batch_kg"] = batch_kg
    return result
