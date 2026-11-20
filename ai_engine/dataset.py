"""Loads and validates the shared ingredient dataset from CSV files.

Every feature reads from this one dataset, so a value shown in Ingredient
Intelligence is the same value used by Blend Simulation and Optimisation.

Files (in DATA_DIR):
    ingredients.csv             one row per ingredient
    ingredient_properties.csv   one row per reported value (long format), with
                                unit, basis, method, processing, variety,
                                Nigerian location, source and status
    ingredient_prices.csv       one row per price / supply lot
    sources.csv                 one row per reference
    formulation_experiments.csv one row per measured blend property
    product_targets.csv         sourced default targets for product categories

Data rules:
- Only records located in a Nigerian state ("Nigeria / <State> / <site>")
  are accepted; anything else is rejected and reported.
- Only verified records (status measured or sourced) are used in
  calculations. Provider estimates and assumptions are kept and shown as
  "not verified", but never used.
"""

import copy
import csv
import os
from dataclasses import dataclass, field, replace
from statistics import mean

from ai_engine import location_model
from ai_engine import properties as P
from ai_engine.locations import NIGERIA_STATES, normalise_state, parse_location

# Lower rank = weaker evidence. "predicted" values come from the prediction
# logic and are never stored in the CSV files.
STATUS_RANK = {"assumed": 0, "estimated": 1, "predicted": 2, "sourced": 3, "measured": 4}

# How the value shown for a location was obtained.
LOCATION_STATUS_LABELS = {
    "national": "all Nigerian data combined",
    "selected": "measured in the selected state",
    "location_model": "estimated for this state from other states' data (no measurements here)",
    "predicted": "estimated from this ingredient's composition (not measured)",
}

REQUIRED_FILES = (
    "ingredients.csv",
    "ingredient_properties.csv",
    "ingredient_prices.csv",
    "sources.csv",
    "formulation_experiments.csv",
)


@dataclass
class ValueRecord:
    """One reported value, converted to the property's canonical unit and basis."""
    value: float
    unit: str
    basis: str
    method: str
    processing: str
    source_id: str
    status: str
    notes: str
    converted_from_as_is: bool = False
    reported_value: float = None   # value and unit exactly as given in the source
    reported_unit: str = ""
    conversion_note: str = ""
    sample: str = ""               # sample label within a source (e.g. lot id, variety)
    location: str = ""             # "Nigeria / State / site"
    variety: str = ""
    state: str = ""
    site: str = ""


@dataclass
class AggregatedValue:
    """The value used for one ingredient property at one location."""
    key: str
    value: float
    low: float
    high: float
    n: int
    status: str  # weakest status among the records, or "predicted"
    basis: str
    records: list
    location_status: str = "national"
    prediction: dict = None  # model details when estimated
    unverified: int = 0      # unverified records (estimated / assumed) not used

    @property
    def locations(self):
        return sorted({r.state for r in self.records if r.state})

    @property
    def source_ids(self):
        return sorted({r.source_id for r in self.records if r.source_id})

    @property
    def kind(self):
        """measured / predicted, as shown to users."""
        return "predicted" if self.status == "predicted" else "measured"

    def to_dict(self, sources=None):
        d = {
            "property": self.key,
            "label": P.get(self.key).label,
            "unit": P.get(self.key).unit,
            "group": P.get(self.key).group,
            "value": round(self.value, 4),
            "low": round(self.low, 4),
            "high": round(self.high, 4),
            "n_records": self.n,
            "status": self.status,
            "kind": self.kind,
            "basis": self.basis,
            "source_ids": self.source_ids,
            "location_status": self.location_status,
            "location_note": LOCATION_STATUS_LABELS.get(self.location_status, ""),
            "states": self.locations,
            "prediction": self.prediction,
            "unverified": self.unverified,
            "records": [
                {k: getattr(r, k) for k in ("value", "unit", "basis", "method", "processing", "source_id",
                                             "status", "notes", "converted_from_as_is", "reported_value",
                                             "reported_unit", "conversion_note", "sample", "location",
                                             "variety", "state", "site")}
                for r in self.records
            ],
        }
        if sources is not None:
            d["sources"] = [sources[s].to_dict() for s in self.source_ids if s in sources]
        return d


@dataclass
class PriceInfo:
    value: float
    low: float
    high: float
    n: int
    status: str
    latest_date: str
    markets: list
    source_ids: list
    location_status: str = "national"
    states: list = field(default_factory=list)
    availability_kg: float = None   # total stock reported for the lots used
    max_share: float = None         # most restrictive max share of the protein blend (0..1)
    lots: list = field(default_factory=list)

    def to_dict(self):
        return {
            "value": round(self.value, 2), "low": round(self.low, 2), "high": round(self.high, 2),
            "unit": "NGN/kg", "n_records": self.n, "status": self.status, "latest_date": self.latest_date,
            "markets": self.markets, "source_ids": self.source_ids, "location_status": self.location_status,
            "location_note": LOCATION_STATUS_LABELS.get(self.location_status, ""), "states": self.states,
            "availability_kg": self.availability_kg, "max_share": self.max_share, "lots": self.lots,
        }


@dataclass
class Source:
    source_id: str
    citation: str
    year: str
    link: str
    notes: str

    def to_dict(self):
        return {"source_id": self.source_id, "citation": self.citation, "year": self.year,
                "link": self.link, "notes": self.notes}


@dataclass
class Ingredient:
    id: str
    name: str
    scientific_name: str
    category: str
    form: str
    description: str
    processing: str
    default_min_inclusion: float
    default_max_inclusion: float
    image_url: str
    values: dict = field(default_factory=dict)  # property key -> AggregatedValue
    price: PriceInfo = None

    def get(self, key):
        v = self.values.get(key)
        return v.value if v else None

    def summary(self):
        return {
            "id": self.id, "name": self.name, "scientific_name": self.scientific_name,
            "category": self.category, "form": self.form, "image_url": self.image_url,
            "protein": _r(self.get("protein")), "fat": _r(self.get("fat")), "fibre": _r(self.get("fibre")),
            "moisture": _r(self.get("moisture")),
            "price_ngn_per_kg": _r(self.price.value) if self.price else None,
            "data_status": self.data_status(),
        }

    def data_status(self):
        measured = sum(1 for v in self.values.values() if v.status != "predicted")
        predicted = sum(1 for v in self.values.values() if v.status == "predicted")
        return {"measured": measured, "predicted": predicted, "available": len(self.values),
                "of": len(P.PROPERTIES)}

    def to_dict(self, sources=None):
        d = self.summary()
        d.update({
            "description": self.description, "processing": self.processing,
            "default_min_inclusion": self.default_min_inclusion,
            "default_max_inclusion": self.default_max_inclusion,
            "values": {k: v.to_dict(sources) for k, v in self.values.items()},
            "price": self.price.to_dict() if self.price else None,
        })
        return d


@dataclass
class Experiment:
    """A measured property of a specific blend (training data for prediction)."""
    experiment_id: str
    source_id: str
    property: str
    value: float
    unit: str
    method: str
    shares: dict  # ingredient id -> share (0..1)
    notes: str
    location: str = ""


@dataclass
class ProductTarget:
    product: str
    property: str
    min: float
    max: float
    source_id: str
    notes: str


def _r(v, nd=2):
    return None if v is None else round(v, nd)


def _read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    # Ignore fully blank lines and comment lines starting with '#'.
    return [
        {k.strip(): (v or "").strip() for k, v in row.items() if k}
        for row in rows
        if any((v or "").strip() for v in row.values())
        and not (list(row.values())[0] or "").strip().startswith("#")
    ]


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class Dataset:
    def __init__(self, data_dir, allowed_ingredients=None):
        self.data_dir = data_dir
        self.allowed_ingredients = set(allowed_ingredients) if allowed_ingredients else None
        self.location = None      # None = all Nigerian data combined
        self.ingredients = {}
        self.sources = {}
        self.experiments = []
        self.product_targets = []
        self.records = {}          # (ingredient, property) -> [verified ValueRecord]
        self.unverified = {}       # (ingredient, property) -> [ValueRecord] estimated / assumed
        self.price_records = {}    # ingredient -> [(price, row, status, state)] verified
        self.unverified_prices = {}
        self.issues = []           # problems in the files; rows with errors are skipped
        self.excluded = []         # rows deliberately left out (scope), not errors
        self._models = {}
        self._load()

    # ------------------------------------------------------------------ load
    def _issue(self, file, row_no, message):
        self.issues.append(f"{file} row {row_no}: {message}")

    def _load(self):
        for name in REQUIRED_FILES:
            if not os.path.exists(os.path.join(self.data_dir, name)):
                self.issues.append(f"Missing data file: {name}")
        self._load_sources()
        self._load_ingredients()
        self._load_properties()
        self._load_prices()
        self._load_experiments()
        self._load_product_targets()

    def _rows(self, name):
        path = os.path.join(self.data_dir, name)
        return _read_csv(path) if os.path.exists(path) else []

    def _load_sources(self):
        for i, row in enumerate(self._rows("sources.csv"), start=2):
            sid = row.get("source_id")
            if not sid:
                self._issue("sources.csv", i, "missing source_id")
                continue
            self.sources[sid] = Source(sid, row.get("citation", ""), row.get("year", ""),
                                       row.get("doi_or_url", ""), row.get("notes", ""))

    def _load_ingredients(self):
        for i, row in enumerate(self._rows("ingredients.csv"), start=2):
            iid = row.get("id")
            if not iid or not row.get("name"):
                self._issue("ingredients.csv", i, "missing id or name")
                continue
            if self.allowed_ingredients is not None and iid not in self.allowed_ingredients:
                self.excluded.append(f"ingredients.csv row {i}: '{iid}' is outside the ingredient scope")
                continue
            lo = _float(row.get("default_min_inclusion")) or 0.0
            hi = _float(row.get("default_max_inclusion"))
            hi = 1.0 if hi is None else hi
            if not 0 <= lo <= hi <= 1:
                self._issue("ingredients.csv", i, f"inclusion limits must satisfy 0 <= min <= max <= 1 ({iid})")
                lo, hi = 0.0, 1.0
            self.ingredients[iid] = Ingredient(
                id=iid, name=row["name"], scientific_name=row.get("scientific_name", ""),
                category=row.get("category", ""), form=row.get("form", ""),
                description=row.get("description", ""), processing=row.get("processing", ""),
                default_min_inclusion=lo, default_max_inclusion=hi, image_url=row.get("image_url", ""),
            )

    def _status(self, file, i, row):
        """Validate status and source. Returns status or None."""
        status = row.get("status", "").lower()
        if status not in P.VALUE_STATUSES:
            self._issue(file, i, f"status must be one of {P.VALUE_STATUSES}, got '{status}'")
            return None
        sid = row.get("source_id", "")
        if status in P.VERIFIED_STATUSES and not sid:
            self._issue(file, i, "a measured or sourced value needs a source_id")
            return None
        if sid and sid not in self.sources:
            self._issue(file, i, f"unknown source_id '{sid}' (add it to sources.csv)")
            return None
        return status

    def _state(self, file, i, row):
        state, site = parse_location(row.get("location", ""))
        if state is None:
            self.excluded.append(f"{file} row {i}: location '{row.get('location', '')}' is not a Nigerian "
                                 "state (Nigeria only)")
        return state, site

    def _load_properties(self):
        file = "ingredient_properties.csv"
        raw = {}
        for i, row in enumerate(self._rows(file), start=2):
            iid, key = row.get("ingredient_id"), row.get("property")
            if iid not in self.ingredients:
                if self.allowed_ingredients is None or iid in self.allowed_ingredients:
                    self._issue(file, i, f"unknown ingredient_id '{iid}'")
                continue
            if key not in P.PROPERTIES:
                self._issue(file, i, f"unknown property '{key}'")
                continue
            value = _float(row.get("value"))
            if value is None:
                continue  # blank value means "not documented"
            pdef = P.get(key)
            reported_value, reported_unit = value, row.get("unit", "")
            converted = P.convert_unit(key, value, reported_unit)
            if converted is None:
                accepted = [pdef.unit] + list(P.UNIT_CONVERSIONS.get(key, {}))
                self._issue(file, i, f"unit for {key} must be one of {accepted}, got '{reported_unit}'")
                continue
            value, conversion_note = converted
            default_basis = P.DRY_BASIS if pdef.basis_convertible else (P.AS_IS_BASIS if key == "moisture" else "")
            basis = (row.get("basis") or default_basis).lower()
            if pdef.basis_convertible and basis not in (P.DRY_BASIS, P.AS_IS_BASIS):
                self._issue(file, i, f"basis must be 'dry' or 'as_is', got '{basis}'")
                continue
            status = self._status(file, i, row)
            if status is None:
                continue
            state, site = self._state(file, i, row)
            if state is None:
                continue
            raw.setdefault((iid, key), []).append(ValueRecord(
                value=value, unit=pdef.unit, basis=basis, method=row.get("method", ""),
                processing=row.get("processing", ""), source_id=row.get("source_id", ""), status=status,
                notes=row.get("notes", ""), reported_value=reported_value, reported_unit=reported_unit,
                conversion_note=conversion_note, sample=row.get("sample", ""), location=row.get("location", ""),
                variety=row.get("variety", ""), state=state, site=site,
            ))

        for (iid, key), recs in raw.items():
            pdef = P.get(key)
            if pdef.basis_convertible:
                moisture_recs = raw.get((iid, "moisture"), [])
                converted = []
                for r in recs:
                    if r.basis == P.AS_IS_BASIS:
                        # Convert with the moisture measured on the same sample in the same source.
                        same = [m.value for m in moisture_recs
                                if m.source_id == r.source_id and m.sample == r.sample]
                        if not same or same[0] >= 100:
                            self.issues.append(f"{iid}.{key}: as-is value from {r.source_id} (sample "
                                               f"'{r.sample}') skipped: no moisture for that sample to convert "
                                               "it to dry basis")
                            continue
                        moisture = same[0]
                        note = f"converted from {r.value:g} as-is using the sample's {moisture:g}% moisture"
                        r = replace(r, value=r.value * 100 / (100 - moisture), basis=P.DRY_BASIS,
                                    converted_from_as_is=True,
                                    conversion_note="; ".join(x for x in (r.conversion_note, note) if x))
                    converted.append(r)
                recs = converted
            for r in recs:
                bucket = self.records if r.status in P.VERIFIED_STATUSES else self.unverified
                bucket.setdefault((iid, key), []).append(r)

        for (iid, key), recs in self.records.items():
            self.ingredients[iid].values[key] = self._aggregate(
                key, recs, "national", len(self.unverified.get((iid, key), [])))

    @staticmethod
    def _aggregate(key, recs, location_status, unverified=0):
        vals = [r.value for r in recs]
        status = min((r.status for r in recs), key=STATUS_RANK.get)
        return AggregatedValue(key, mean(vals), min(vals), max(vals), len(vals), status, recs[0].basis, recs,
                               location_status, unverified=unverified)

    def _load_prices(self):
        file = "ingredient_prices.csv"
        for i, row in enumerate(self._rows(file), start=2):
            iid = row.get("ingredient_id")
            if iid not in self.ingredients:
                continue
            price = _float(row.get("price_ngn_per_kg"))
            if price is None or price <= 0:
                self._issue(file, i, "price_ngn_per_kg must be a positive number")
                continue
            status = self._status(file, i, row)
            if status is None:
                continue
            state, _ = self._state(file, i, row)
            if state is None:
                continue
            bucket = self.price_records if status in P.VERIFIED_STATUSES else self.unverified_prices
            bucket.setdefault(iid, []).append((price, row, status, state))
        for iid, items in self.price_records.items():
            self.ingredients[iid].price = self._price_info(items, "national")

    @staticmethod
    def _price_info(items, location_status):
        prices = [p for p, _, _, _ in items]
        avail = [a for a in (_float(r.get("availability_kg")) for _, r, _, _ in items) if a is not None]
        shares = [s for s in (_float(r.get("max_blend_share_pct")) for _, r, _, _ in items) if s is not None]
        lots = [{"price": p, "market": r.get("market", ""), "supplier": r.get("supplier", ""), "state": st,
                 "date": r.get("price_date", ""), "status": status,
                 "availability_kg": _float(r.get("availability_kg")),
                 "max_blend_share_pct": _float(r.get("max_blend_share_pct")), "lot": r.get("lot_id", "")}
                for p, r, status, st in items]
        return PriceInfo(
            value=mean(prices), low=min(prices), high=max(prices), n=len(prices),
            status=min((s for _, _, s, _ in items), key=STATUS_RANK.get),
            latest_date=max((r.get("price_date", "") for _, r, _, _ in items), default=""),
            markets=sorted({r.get("market", "") for _, r, _, _ in items if r.get("market")}),
            source_ids=sorted({r.get("source_id", "") for _, r, _, _ in items if r.get("source_id")}),
            location_status=location_status, states=sorted({st for _, _, _, st in items}),
            availability_kg=sum(avail) if avail else None,
            max_share=min(shares) / 100 if shares else None, lots=lots,
        )

    def _load_experiments(self):
        file = "formulation_experiments.csv"
        for i, row in enumerate(self._rows(file), start=2):
            key = row.get("property")
            if key not in P.PROPERTIES:
                self._issue(file, i, f"unknown property '{key}'")
                continue
            value = _float(row.get("value"))
            if value is None:
                self._issue(file, i, "value must be numeric")
                continue
            if row.get("unit") != P.get(key).unit:
                self._issue(file, i, f"unit for {key} must be '{P.get(key).unit}'")
                continue
            sid = row.get("source_id", "")
            if not sid or sid not in self.sources:
                self._issue(file, i, "experiments need a source_id listed in sources.csv")
                continue
            if row.get("location") and parse_location(row["location"])[0] is None:
                self.excluded.append(f"{file} row {i}: location is not a Nigerian state")
                continue
            shares, bad = {}, False
            for col, v in row.items():
                if col.startswith("share_") and v != "":
                    iid = col[len("share_"):]
                    s = _float(v)
                    if iid not in self.ingredients or s is None or s < 0:
                        self._issue(file, i, f"invalid share column '{col}'='{v}'")
                        bad = True
                        break
                    if s > 0:
                        shares[iid] = s
            if bad:
                continue
            total = sum(shares.values())
            if total > 1.5:
                shares = {k: s / 100 for k, s in shares.items()}
                total /= 100
            if abs(total - 1) > 0.01:
                self._issue(file, i, f"shares must sum to 1 (or 100), got {total:.3f}")
                continue
            self.experiments.append(Experiment(
                row.get("experiment_id", f"row{i}"), sid, key, value, P.get(key).unit,
                row.get("method", ""), shares, row.get("notes", ""), row.get("location", "")))

    def _load_product_targets(self):
        file = "product_targets.csv"
        for i, row in enumerate(self._rows(file), start=2):
            key = row.get("property")
            if key not in P.PROPERTIES:
                self._issue(file, i, f"unknown property '{key}'")
                continue
            sid = row.get("source_id", "")
            if not sid or sid not in self.sources:
                self._issue(file, i, "a default product target needs a source_id listed in sources.csv")
                continue
            self.product_targets.append(ProductTarget(
                row.get("product", ""), key, _float(row.get("min")), _float(row.get("max")), sid,
                row.get("notes", "")))

    # ------------------------------------------------------------- locations
    def location_model(self, iid, key):
        """Fitted location-variation model for one ingredient property (cached)."""
        if (iid, key) not in self._models:
            groups = {}
            for r in self.records.get((iid, key), []):
                groups.setdefault(r.state, []).append(r.value)
            self._models[(iid, key)] = location_model.fit(key, groups)
        return self._models[(iid, key)]

    def coverage(self):
        """Which states have verified data, per ingredient: {state: {ingredient: n_records}}."""
        out = {s: {} for s in NIGERIA_STATES}
        for (iid, _), recs in self.records.items():
            for r in recs:
                out[r.state][iid] = out[r.state].get(iid, 0) + 1
        return out

    def states_with_prices(self):
        return sorted({st for items in self.price_records.values() for *_, st in items})

    def view(self, state):
        """A copy of the dataset with values chosen for one Nigerian state.

        For each ingredient property: verified measurements from that state if
        any; otherwise the location model's estimate for a state without data,
        if the model passed validation; otherwise unavailable. Prices and stock
        come only from lots sold in that state. Nothing is scaled or invented.
        """
        if not state:
            return self
        canonical = normalise_state(state)
        if canonical is None:
            raise ValueError(f"'{state}' is not a Nigerian state or the FCT.")
        view = copy.copy(self)
        view.location = canonical
        view.ingredients = {}
        view.location_refusals = {}
        for iid, ing in self.ingredients.items():
            values = {}
            for key in P.PROPERTIES:
                recs = self.records.get((iid, key), [])
                here = [r for r in recs if r.state == canonical]
                n_unverified = len([r for r in self.unverified.get((iid, key), []) if r.state == canonical])
                if here:
                    values[key] = self._aggregate(key, here, "selected", n_unverified)
                elif recs:
                    model = self.location_model(iid, key)
                    est = model.predict_unseen()
                    if est is None:
                        view.location_refusals[(iid, key)] = model.validation.get("reason", "")
                        continue
                    values[key] = AggregatedValue(
                        key, est["value"], est["low"], est["high"], 0, "predicted", recs[0].basis, [],
                        "location_model",
                        prediction={"method": "location model (national value, checked by leaving out one "
                                              "state at a time)",
                                    "n_training": len(recs), "n_states": model.validation["n_states"],
                                    "relative_error": model.validation["relative_error"],
                                    "training_states": sorted(model.groups),
                                    "note": "No measurements from this state. Estimated from other states."})
            price = None
            lots = [it for it in self.price_records.get(iid, []) if it[3] == canonical]
            if lots:
                price = self._price_info(lots, "selected")
            view.ingredients[iid] = replace(ing, values=values, price=price)
        return view

    # ---------------------------------------------------------------- query
    def list(self, q=None, category=None, min_protein=None):
        items = list(self.ingredients.values())
        if q:
            ql = q.lower()
            items = [i for i in items if ql in i.name.lower() or ql in i.scientific_name.lower()
                     or ql in i.id.lower()]
        if category:
            items = [i for i in items if i.category.lower() == category.lower()]
        if min_protein is not None:
            items = [i for i in items if (i.get("protein") or 0) >= min_protein]
        return items

    def get(self, ingredient_id):
        return self.ingredients.get(ingredient_id)

    def categories(self):
        return sorted({i.category for i in self.ingredients.values() if i.category})

    def summary(self):
        verified = sum(len(v) for v in self.records.values())
        unverified = sum(len(v) for v in self.unverified.values())
        return {
            "ingredients": len(self.ingredients), "sources": len(self.sources),
            "experiments": len(self.experiments), "verified_records": verified,
            "unverified_records": unverified, "verified_price_lots": sum(len(v) for v in self.price_records.values()),
            "states_with_data": sorted({r.state for recs in self.records.values() for r in recs}),
            "states_with_prices": self.states_with_prices(),
            "location": self.location, "issues": self.issues, "excluded": self.excluded,
        }
