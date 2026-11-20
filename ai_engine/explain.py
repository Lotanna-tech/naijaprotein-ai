"""Explainable Recommendations.

Turns the numbers produced by Blend Simulation and Formulation Optimisation
into plain-language explanations. Every sentence is generated from those
numbers and from the dataset; no language model is involved, and nothing is
stated that the calculations did not produce.
"""

from ai_engine import properties as P
from ai_engine import products as PR
from ai_engine.blend import OIL, SUPPORTING, WATER
from ai_engine.dataset import LOCATION_STATUS_LABELS
from ai_engine.optimiser import optimise


def _naira(v):
    return f"\u20a6{v:,.0f}" if abs(v) >= 100 else f"\u20a6{v:,.2f}"


def _pct(v, nd=0):
    return f"{v * 100:.{nd}f}%"


def confidence_level(kinds):
    """'high' if every input value is measured, 'medium' if some are estimated, 'unknown' if none."""
    if not kinds:
        return "unknown"
    return "medium" if "predicted" in kinds else "high"


def evidence(dataset, ingredient_ids, keys):
    """Every ingredient value the result relies on, with its source, state and type."""
    items = []
    for iid in ingredient_ids:
        ing = dataset.get(iid)
        for key in keys:
            v = ing.values.get(key)
            if v is None:
                continue
            items.append({
                "ingredient": ing.name, "quantity": P.get(key).label, "value": round(v.value, 3),
                "unit": P.get(key).unit, "range": [round(v.low, 3), round(v.high, 3)], "n_records": v.n,
                "kind": v.kind, "status": v.status, "basis": v.basis, "location_status": v.location_status,
                "location_note": LOCATION_STATUS_LABELS.get(v.location_status, ""), "states": v.locations,
                "prediction": v.prediction,
                "sources": [dataset.sources[s].citation for s in v.source_ids if s in dataset.sources]
                or ([v.prediction["method"]] if v.prediction else []),
            })
        if ing.price:
            items.append({
                "ingredient": ing.name, "quantity": "Price", "value": round(ing.price.value, 2), "unit": "NGN/kg",
                "range": [round(ing.price.low, 2), round(ing.price.high, 2)], "n_records": ing.price.n,
                "kind": "measured", "status": ing.price.status, "basis": "", "date": ing.price.latest_date,
                "location_status": ing.price.location_status,
                "location_note": LOCATION_STATUS_LABELS.get(ing.price.location_status, ""), "states": ing.price.states,
                "sources": [dataset.sources[s].citation for s in ing.price.source_ids if s in dataset.sources],
            })
    return items


def _data_notes(dataset, items):
    notes = []
    loc = dataset.location
    if loc:
        here = sum(1 for e in items if e["location_status"] == "selected")
        est = [e for e in items if e["location_status"] == "location_model"]
        notes.append(f"{here} of {len(items)} values used were measured in {loc}.")
        if est:
            notes.append("Not measured in this state, so estimated from other Nigerian states: "
                         + "; ".join(f"{e['ingredient']} {e['quantity'].lower()}" for e in est) + ".")
    else:
        notes.append("Values combine all verified Nigerian data. Choose a state to use that state's data.")
    for e in items:
        if e["location_status"] == "predicted":
            notes.append(f"{e['ingredient']} {e['quantity'].lower()} is estimated from its composition, "
                         "not measured.")
    return notes


def _product_notes(dataset, product, ingredient_ids):
    """For each property that matters for the product, which selected ingredient is highest/lowest."""
    notes = []
    for rel in product.relevant:
        vals = [(dataset.get(i).name, dataset.get(i).values[rel.property])
                for i in ingredient_ids if rel.property in dataset.get(i).values]
        if len(vals) < 1:
            continue
        label = P.get(rel.property).label.lower()
        unit = P.get(rel.property).unit
        best = (max if rel.direction == PR.HIGHER else min)(vals, key=lambda nv: nv[1].value)
        listing = ", ".join(f"{n} {v.value:.2f}" for n, v in sorted(vals, key=lambda nv: -nv[1].value))
        missing = [dataset.get(i).name for i in ingredient_ids if rel.property not in dataset.get(i).values]
        text = (f"{P.get(rel.property).label}: {rel.why} Among the chosen ingredients, {best[0]} is "
                f"{'highest' if rel.direction == PR.HIGHER else 'lowest'} ({listing} {unit}).")
        if missing:
            text += " No value for " + ", ".join(missing) + "."
        notes.append({"property": rel.property, "text": text})
    return notes


def _support_effects(profile):
    """How added water and oil change the mix, from the calculated values."""
    out = []
    recipe = profile["recipe"]
    w, o = recipe.get(WATER, 0.0), recipe.get(OIL, 0.0)
    prot = profile["mix"]["protein"]
    if (w or o) and prot.get("available"):
        base = 1 - w - o
        out.append(f"Added water and oil make up {_pct(w + o, 1)} of the mix. Without them the protein ingredients "
                   f"alone would give {prot['value'] / base:.1f} g protein per 100 g; in the full mix it is "
                   f"{prot['value']:.1f} g per 100 g.")
    if o:
        fat = profile["mix"]["fat"]
        if fat.get("available"):
            out.append(f"Edible oil adds {o * 100:.1f} g fat per 100 g of mix, of a total {fat['value']:.1f} g.")
    if w:
        moist = profile["mix"]["moisture"]
        if moist.get("available"):
            out.append(f"Moisture of the mix is {moist['value']:.1f} g per 100 g, of which {w * 100:.1f} g is added water.")
    out += [c["text"] for c in profile.get("holding_checks", [])]
    return out


def _limitations(profile, dataset):
    out = []
    func = [f for f in profile["functional"].values() if f["relevant"] and not f["available"]]
    if func:
        out.append("The mix's " + ", ".join(f["label"].lower() for f in func) + " cannot be calculated yet. These "
                   "properties change when ingredients are combined, and there is no measured data on blends to "
                   "estimate them. The ingredients' own values are shown for guidance.")
    if not profile["cost"].get("available"):
        out.append(profile["cost"]["reason"])
    elif profile["cost"].get("excludes"):
        out.append("Cost does not include " + ", ".join(x.lower() for x in profile["cost"]["excludes"])
                   + " because no price was entered.")
    unavailable = [m["label"] for m in profile["mix"].values() if not m.get("available")
                   and m["property"] in ("protein", "fat", "moisture", "fibre", "ash", "carbohydrate")]
    if unavailable:
        out.append("Not calculated because data is missing: " + ", ".join(unavailable).lower() + ".")
    out.append(PR.DISCLAIMER)
    return out


def explain_simulation(dataset, profile):
    product = PR.get(profile["product"]["id"])
    ids = [c["id"] for c in profile["components"] if c["kind"] == "protein"]
    lines = []
    prot = profile["mix"]["protein"]
    if prot.get("available"):
        top = max(prot["contributions"].items(), key=lambda kv: kv[1])
        lines.append(f"This {product.name.lower()} mix has {prot['value']:.1f} g protein per 100 g. "
                     f"{_name(dataset, top[0])} supplies the most ({top[1]:.1f} g).")
    else:
        lines.append(prot["reason"])
    c = profile["cost"]
    if c.get("available"):
        top = max(c["contributions"].items(), key=lambda kv: kv[1])
        lines.append(f"It costs {_naira(c['value'])} per kg; {_name(dataset, top[0])} is "
                     f"{top[1] / c['value']:.0%} of that.")
    keys = sorted({"protein", "moisture", "fat"} | {k for k, m in profile["mix"].items() if m.get("available")})
    items = evidence(dataset, ids, keys)
    return {
        "headline": " ".join(lines),
        "confidence": confidence_level({e["kind"] for e in items}),
        "targets": profile["targets"],
        "product": _product_notes(dataset, product, ids),
        "water_oil": _support_effects(profile),
        "availability": [a["text"] for a in profile["availability"]],
        "data": _data_notes(dataset, items),
        "limitations": _limitations(profile, dataset),
        "contributions": profile["contributions"],
        "evidence": items,
    }


def _name(dataset, cid):
    return SUPPORTING[cid]["name"] if cid in SUPPORTING else dataset.get(cid).name


def _location_comparison(dataset, reference, request, result):
    """Compare the optimum in the selected state with the optimum on all Nigerian data."""
    if not dataset.location or reference is None:
        return None
    ref = optimise(reference, **request)
    diffs = []
    keys = ["protein", "moisture"] + list(request.get("targets") or {})
    involved = set(result.get("recipe", {})) | set(ref.get("recipe", {})) | {
        e["id"] for e in result.get("excluded_ingredients", [])}
    for iid in [i for i in request["ingredient_ids"] if i in involved]:
        a, b = dataset.get(iid), reference.get(iid)
        for k in dict.fromkeys(keys):
            va, vb = a.values.get(k), b.values.get(k)
            if va and vb and abs(va.value - vb.value) > 1e-6:
                diffs.append(f"{a.name} {P.get(k).label.lower()} {va.value:.2f} here vs {vb.value:.2f} nationally")
            elif vb and not va:
                diffs.append(f"{a.name} {P.get(k).label.lower()} has no data here")
        if a.price and b.price and abs(a.price.value - b.price.value) > 1e-6:
            diffs.append(f"{a.name} costs {_naira(a.price.value)}/kg here vs {_naira(b.price.value)}/kg nationally")
        elif b.price and not a.price:
            diffs.append(f"{a.name} has no verified price here")
    out = {"location": dataset.location, "differences": diffs}
    if len(diffs) > 6:
        diffs = diffs[:6] + [f"and {len(out['differences']) - 6} more differences (see technical details)"]
    if not diffs:
        out["text"] = f"The data for {dataset.location} matches the national data for these ingredients, so the mix is the same."
    elif result.get("feasible") and ref["feasible"]:
        changes = []
        for cid in set(result["recipe"]) | set(ref["recipe"]):
            sa, sb = result["recipe"].get(cid, 0), ref["recipe"].get(cid, 0)
            if abs(sa - sb) > 0.005:
                changes.append(f"{_name(dataset, cid)} {_pct(sa, 1)} here vs {_pct(sb, 1)} nationally")
        if changes:
            out["text"] = (f"{dataset.location}'s data changes the mix: " + "; ".join(changes) + ". Cost "
                           f"{_naira(result['cost_per_kg'])}/kg here vs {_naira(ref['cost_per_kg'])}/kg nationally. "
                           "Because: " + "; ".join(diffs) + ".")
        else:
            out["text"] = (f"{dataset.location}'s data differs (" + "; ".join(diffs) + ") but not enough to change "
                           "the best proportions.")
    else:
        out["text"] = (f"With {dataset.location}'s data a mix is {'possible' if result.get('feasible') else 'not possible'}; "
                       f"with national data it is {'possible' if ref['feasible'] else 'not possible'}. "
                       "Because: " + "; ".join(diffs) + ".")
    return out


def explain_optimisation(dataset, predictor, request, result, profile=None, reference=None):
    """Explanation for an optimiser result; `profile` is the simulation of the optimal mix."""
    product = PR.get(request["product"]) if request.get("product") else None
    req = {k: v for k, v in request.items() if k != "product"}
    location = _location_comparison(dataset, reference, req, result)
    if not result.get("feasible"):
        diag = result.get("diagnosis", {})
        return {"headline": result.get("message", "No mix found."),
                "summary": "No mix was produced because the requirements cannot all be met with these ingredients. "
                           "This is a real result: no combination works.",
                "reasons": diag.get("reasons", []), "suggestions": diag.get("suggestions", []),
                "excluded": result.get("excluded_ingredients", []), "location": location}

    comps = result["components"]
    used = [c for c in comps if c["share"] > 1e-6]
    protein_used = [c for c in used if c["kind"] == "protein"]
    unused = [c for c in comps if c["share"] <= 1e-6 and c["kind"] == "protein"]
    headline = ("Lowest-cost mix: " + ", ".join(f"{_pct(c['share'], 1)} {c['name'].lower() if c['kind'] == 'supporting' else c['name']}"
                                                 for c in used)
                + f". Cost {_naira(result['cost_per_kg'])} per kg, {_naira(result['batch_cost'])} for "
                  f"{result['batch_kg']:,.0f} kg.")
    if result.get("unpriced_components"):
        headline += " (Cost excludes " + " and ".join(result["unpriced_components"]) + ": no price entered.)"

    ranking = sorted((c for c in comps if c["kind"] == "protein" and c.get("cost_per_protein_point")),
                     key=lambda c: c["cost_per_protein_point"])
    rank = {c["id"]: k + 1 for k, c in enumerate(ranking)}
    prot = profile["mix"]["protein"] if profile else None
    reasons = []
    for c in protein_used:
        parts = []
        if prot and prot.get("available"):
            contrib = prot["contributions"].get(c["id"], 0)
            parts.append(f"gives {contrib:.1f} of the {prot['value']:.1f} g protein per 100 g")
        parts.append(f"costs {_naira(c['price_ngn_per_kg'])}/kg, which is {_naira(c['cost_per_protein_point'])} for each "
                     f"gram of protein per 100 g (number {rank.get(c['id'], '-')} of {len(ranking)} for value)")
        if c.get("at_max"):
            parts.append(f"it is at its maximum of {_pct(c['max_share'])} of the protein ingredients "
                         f"({c['limit_source']}), so more would be used if allowed")
        elif c.get("at_min"):
            parts.append(f"it is held at your minimum of {_pct(c['min_share'])}")
        reasons.append({"id": c["id"], "text": f"{c['name']} ({_pct(c['share_of_protein_blend'])} of the protein "
                                               "ingredients): " + "; ".join(parts) + "."})
    for c in unused:
        reasons.append({"id": c["id"], "text": f"{c['name']} is not used: other ingredients meet your requirements "
                                               "at lower cost."})
    for e in result.get("excluded_ingredients", []):
        reasons.append({"id": e["id"], "text": f"{e['name']} could not be considered: {e['reason']}"})

    constraint_notes = []
    for c in result["constraints"]:
        if c["group"] == "nutrition":
            if c["binding"] and c["cost_effect_per_kg"] > 1e-6:
                text = (f"{c['label']}: met exactly, and it sets the cost. Tightening it by 1 unit would add about "
                        f"{_naira(c['cost_effect_per_kg'])} per kg.")
            elif c["binding"]:
                text = f"{c['label']}: met exactly."
            else:
                text = f"{c['label']}: met with room to spare (mix has {c['achieved']:.2f})."
            constraint_notes.append({"binding": c["binding"], "text": text})
        elif c["binding"] and c["group"] in ("stock", "budget", "water_holding"):
            constraint_notes.append({"binding": True, "text": f"{c['label']}: this limit is reached."})

    water_oil = []
    for c in used:
        if c["kind"] != "supporting":
            continue
        reason = ""
        if c["at_max"]:
            reason = (" It is at the maximum you allowed" + (", because it lowers the cost per kg while the other "
                      "requirements are still met." if (c["price_ngn_per_kg"] or 0) < result["cost_per_kg"] else "."))
        elif c["at_min"]:
            reason = " It is at the minimum you set."
        water_oil.append(f"{c['name']}: {_pct(c['share'], 1)} of the mix ({c['kg']:,.1f} kg).{reason}")
    if result.get("water_holding_rule") and result["water_holding_rule"]["applied"]:
        water_oil.append("Added water was limited to what the protein ingredients are each reported to hold. This "
                         "assumes they do not affect each other, so treat it as a rough limit.")
    elif result.get("water_holding_rule"):
        water_oil.append("The water-holding limit was not applied: not every chosen ingredient has a value.")
    if profile:
        water_oil += _support_effects(profile)

    out = {
        "headline": headline, "ingredients": reasons, "constraints": constraint_notes, "water_oil": water_oil,
        "location": location,
    }
    if profile:
        ids = [c["id"] for c in protein_used]
        keys = sorted({"protein", "moisture"} | set(req.get("targets") or {}))
        items = evidence(dataset, ids, keys)
        out.update({
            "product": _product_notes(dataset, product, ids) if product else [],
            "functional_targets": profile["targets"],
            "availability": [a["text"] for a in profile["availability"]],
            "data": _data_notes(dataset, items), "limitations": _limitations(profile, dataset),
            "confidence": confidence_level({e["kind"] for e in items}), "evidence": items,
        })
    return out
