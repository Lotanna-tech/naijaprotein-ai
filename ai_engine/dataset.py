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
- Carbohydrate not reported by a source is calculated by difference for each
  sample: 100 - protein - fat - ash - fibre (dry basis), and shown as
  "calculated".
- Composition must close: the dry-matter components of a sample cannot add up
  to more than 100 g/100 g, and values labelled dry basis that only reach 100
  together with moisture are reported as probably as-is.

For a single state (`view`), a state's own verified records come first. A
state without records gets an estimate from the other states (location
model), and a state without price lots gets an estimated price (price model).
Every estimate is marked, carries a 95% interval and is graded "checked" or
"indicative".
"""

import copy
import csv
import os
from dataclasses import dataclass, field, replace
from statistics import mean

import numpy as np

from ai_engine import location_model
from ai_engine import price_model
from ai_engine import properties as P
from ai_engine.locations import NIGERIA_STATES, normalise_state, parse_location

# Lower rank = weaker evidence. "predicted" values come from the prediction
# logic and are never stored in the CSV files.
STATUS_RANK = {"assumed": 0, "estimated": 1, "predicted": 2, "sourced": 3, "measured": 4}

# How the value shown for a location was obtained (used in data exports).
LOCATION_STATUS_LABELS = {
    "national": "all Nigerian data combined",
    "selected": "measured in this state",
    "location_model": "estimated from other states",
    "price_model": "price estimated from other states",
    "predicted": "estimated from the ingredient's composition",
}

MIN_STATES_FOR_CHECK = location_model.MIN_STATES

# Carbohydrate by difference, on a dry basis, from these components of the same sample.
CARBOHYDRATE_COMPONENTS = ("protein", "fat", "ash", "fibre")
CARBOHYDRATE_METHOD = "Calculated by difference: 100 - protein - fat - ash - fibre (dry basis)"
CLOSURE_TOLERANCE = 0.5  # g/100 g

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
    derived: bool = False          # calculated from other values of the same sample (not reported)


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
        """measured / calculated / predicted, as shown to users."""
        if self.status == "predicted":
            return "predicted"
        if self.records and all(r.derived for r in self.records):
            return "calculated"
        return "measured"

    @property
    def confidence(self):
        """'measured' or 'calculated' for recorded data; for estimates 'checked' or 'indicative'."""
        if self.status != "predicted":
            return self.kind
        return (self.prediction or {}).get("confidence", location_model.CHECKED)

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
            "confidence": self.confidence,
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
                                             "variety", "state", "site", "derived")}
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
    prediction: dict = None         # price model details when the price is estimated

    @property
    def kind(self):
        return "predicted" if self.status == "predicted" else "measured"

    @property
    def confidence(self):
        return (self.prediction or {}).get("confidence", "measured") if self.status == "predicted" else "measured"

    def to_dict(self):
        return {
            "value": round(self.value, 2), "low": round(self.low, 2), "high": round(self.high, 2),
            "unit": "NGN/kg", "n_records": self.n, "status": self.status, "kind": self.kind,
            "confidence": self.confidence, "latest_date": self.latest_date,
            "markets": self.markets, "source_ids": self.source_ids, "location_status": self.location_status,
            "location_note": LOCATION_STATUS_LABELS.get(self.location_status, ""), "states": self.states,
            "availability_kg": self.availability_kg, "max_share": self.max_share, "lots": self.lots,
            "prediction": self.prediction,
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
        self._priors = None
        self._price_model = None
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
            raw[(iid, key)] = recs

        self._check_closure_and_derive_carbohydrate(raw)
        for (iid, key), recs in raw.items():
            for r in recs:
                bucket = self.records if r.status in P.VERIFIED_STATUSES else self.unverified
                bucket.setdefault((iid, key), []).append(r)

        for (iid, key), recs in self.records.items():
            self.ingredients[iid].values[key] = self._aggregate(
                key, recs, "national", len(self.unverified.get((iid, key), [])))

    def _check_closure_and_derive_carbohydrate(self, recs_by_key):
        """Per sample: check that the composition adds up, and calculate missing carbohydrate.

        recs_by_key: {(ingredient, property): [ValueRecord]} after conversion to dry basis.
        A sample is one material in one source: same source, sample label, location and processing.
        """
        keys = CARBOHYDRATE_COMPONENTS + ("carbohydrate", "moisture")
        samples = {}
        for (iid, key), recs in recs_by_key.items():
            if key not in keys:
                continue
            for r in recs:
                sk = (iid, r.source_id, r.sample, r.location, r.processing)
                samples.setdefault(sk, {}).setdefault(key, []).append(r)
        for (iid, sid, sample, _, _), by_key in samples.items():
            if any(len(v) > 1 for v in by_key.values()):
                continue  # several values for one sample label: cannot tell which belong together
            got = {k: v[0] for k, v in by_key.items()}
            label = f"{iid} sample '{sample}' from {sid}"
            parts = [k for k in CARBOHYDRATE_COMPONENTS + ("carbohydrate",) if k in got]
            dry_sum = sum(got[k].value for k in parts)
            if dry_sum > 100 + CLOSURE_TOLERANCE:
                self.issues.append(f"{label}: {', '.join(parts)} add up to {dry_sum:.2f} g/100 g dry matter, more "
                                   "than 100; check the values and their basis")
                continue
            # Labelled dry, but only reaching 100 with moisture added: the table is most likely as-is.
            if (len(parts) == 5 and "moisture" in got and not any(got[k].converted_from_as_is for k in parts)
                    and abs(dry_sum + got["moisture"].value - 100) <= CLOSURE_TOLERANCE
                    and abs(dry_sum - 100) > 2 * CLOSURE_TOLERANCE):
                self.issues.append(f"{label}: values labelled dry basis add up to 100 only together with moisture "
                                   f"({dry_sum:.2f} + {got['moisture'].value:.2f}); they look like as-is values, so "
                                   "set basis to as_is")
            if "carbohydrate" in got or not all(k in got for k in CARBOHYDRATE_COMPONENTS):
                continue
            inputs = [got[k] for k in CARBOHYDRATE_COMPONENTS]
            base = inputs[0]
            recs_by_key.setdefault((iid, "carbohydrate"), []).append(replace(
                base, value=100 - dry_sum, unit=P.get("carbohydrate").unit, basis=P.DRY_BASIS,
                method=CARBOHYDRATE_METHOD, status=min((r.status for r in inputs), key=STATUS_RANK.get),
                notes="Not reported by the source; calculated from this sample's protein, fat, ash and fibre.",
                converted_from_as_is=False, reported_value=None, reported_unit="", conversion_note="",
                derived=True))

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

    def location_prior(self, key):
        """(relative error, basis) borrowed for a state without data, for one property.

        The median leave-one-state-out relative error of this property over the ingredients that
        can be checked (2+ states). For a property never checked in any ingredient, the largest
        relative error seen for any property is used, so thin evidence gives a wide interval.
        """
        if self._priors is None:
            by_key = {}
            for (iid, k) in self.records:
                rel = self.location_model(iid, k).validation.get("relative_error")
                if rel is not None and np.isfinite(rel):
                    by_key.setdefault(k, []).append(rel)
            every = [x for xs in by_key.values() for x in xs]
            self._priors = ({k: float(np.median(xs)) for k, xs in by_key.items()},
                            max(every) if every else None)
        per_key, fallback = self._priors
        if per_key.get(key):  # a median of 0 only means the states agreed in the tests so far
            return per_key[key], "other_ingredients"
        return fallback, "largest_seen"

    def measurement_step(self, key):
        """Smallest difference between distinct verified values of a property, e.g. 2 for least gelation
        (reported in steps of 2% w/v). Results within one step can come from different samples."""
        values = sorted({round(r.value, 6) for (_, k), recs in self.records.items() if k == key for r in recs})
        steps = [b - a for a, b in zip(values, values[1:]) if b - a > 1e-9]
        return min(steps) if steps else 0.0

    def location_summary(self, iid, key):
        """Location model summary for one ingredient property, including the estimate for a new state."""
        return self.location_model(iid, key).summary(*self.location_prior(key), self.measurement_step(key) / 2)

    def price_model(self):
        """Price model fitted on every verified lot (cached)."""
        if self._price_model is None:
            lots = [(iid, st, p, row.get("price_date", ""))
                    for iid, items in self.price_records.items() for p, row, _, st in items]
            self._price_model = price_model.fit(lots)
        return self._price_model

    def national_max_share(self, iid):
        """Most restrictive supplier max share among all verified lots of an ingredient (0..1)."""
        shares = [s for s in (_float(r.get("max_blend_share_pct")) for _, r, _, _ in self.price_records.get(iid, []))
                  if s is not None]
        return min(shares) / 100 if shares else None

    def _location_estimate(self, iid, key, recs):
        """AggregatedValue for a state without records of this property, or (None, reason)."""
        model = self.location_model(iid, key)
        est = model.estimate_unseen(*self.location_prior(key), self.measurement_step(key) / 2)
        if est is None:
            return None, (f"only measured in {', '.join(sorted(model.groups))}, and the data give no measure of how "
                          "much it varies from state to state")
        checked = est["confidence"] == location_model.CHECKED
        n_states = model.validation["n_states"]
        states = f"{n_states} other state{'s' if n_states != 1 else ''}"
        note = (f"Not measured here; estimated from {states}." if checked else
                f"Not measured here; a rough estimate from {states}"
                + (", which differ a lot." if n_states >= MIN_STATES_FOR_CHECK else "."))
        prediction = {
            "method": "national value from the states with data (random-effects model)"
                      + (", checked by leaving out one state at a time" if checked else ""),
            "confidence": est["confidence"], "n_training": len(recs), "n_states": n_states,
            "relative_error": model.validation.get("relative_error"),
            "relative_sd": None if est["relative_sd"] is None else round(est["relative_sd"], 4),
            "uncertainty_from": location_model.UNCERTAINTY_LABELS[est["uncertainty_from"]],
            "training_states": sorted(model.groups), "reason": est["reason"], "note": note,
        }
        return AggregatedValue(key, est["value"], est["low"], est["high"], 0, "predicted", recs[0].basis, [],
                               "location_model", prediction=prediction), ""

    def _estimated_price(self, ing, state):
        """PriceInfo estimated by the price model, or (None, reason)."""
        model = self.price_model()
        est = model.estimate(ing.id, state)
        if est is None:
            return None, model.refusal(ing.id, state)
        name = ing.name.lower()
        join = lambda xs: xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
        if est["case"] == "missing_ingredient":
            note = (f"No market price for {name} was collected in {state}. Estimated from what other proteins cost "
                    f"in {state} and what {name} costs in {join(est['ingredient_states'])}. It may cost more if it "
                    "is not grown nearby.")
        else:
            note = (f"No market prices were collected in {state}. Estimated from prices in "
                    f"{join(list(est['weights']))}, giving nearer states more weight.")
        # Cross-check only: unverified lots never set a price, but if they fall outside the
        # estimate's range, the estimate is not called checked.
        unverified = [p for p, _, _, st in self.unverified_prices.get(ing.id, []) if st == state]
        if unverified:
            avg = mean(unverified)
            est["unverified_lots"] = {"n": len(unverified), "mean": round(avg, 2)}
            if not est["low"] <= avg <= est["high"]:
                est["confidence"] = location_model.INDICATIVE
                note += (f" Unconfirmed prices from the data provider are about \u20a6{avg:,.0f}/kg, so get a local "
                         "quote.")
        est = {**est, "note": note, "stock_note": f"Stock is not known for {state}; check with suppliers."}
        return PriceInfo(value=est["value"], low=est["low"], high=est["high"], n=0, status="predicted",
                         latest_date=est["period"][1], markets=[], source_ids=[], location_status="price_model",
                         states=est["training_states"], availability_kg=None,
                         max_share=self.national_max_share(ing.id), lots=[], prediction=est), ""

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
        any; otherwise the location model's estimate from the other states
        (graded checked or indicative, with a 95% interval). Prices and stock
        come from lots sold in that state; without a lot, the price is
        estimated by the price model and stock is unknown. Measured values are
        never replaced, and every estimate is marked as such.
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
        view.price_refusals = {}
        for iid, ing in self.ingredients.items():
            values = {}
            for key in P.PROPERTIES:
                recs = self.records.get((iid, key), [])
                here = [r for r in recs if r.state == canonical]
                n_unverified = len([r for r in self.unverified.get((iid, key), []) if r.state == canonical])
                if here:
                    values[key] = self._aggregate(key, here, "selected", n_unverified)
                elif recs:
                    value, reason = self._location_estimate(iid, key, recs)
                    if value is None:
                        view.location_refusals[(iid, key)] = reason
                    else:
                        values[key] = value
            lots = [it for it in self.price_records.get(iid, []) if it[3] == canonical]
            if lots:
                price = self._price_info(lots, "selected")
            else:
                price, reason = self._estimated_price(ing, canonical)
                if price is None:
                    view.price_refusals[iid] = reason
            view.ingredients[iid] = replace(ing, values=values, price=price)
        return view

    def without_state(self, state):
        """A copy without one state's verified records and price lots, for checking estimates (backtests)."""
        canonical = normalise_state(state)
        out = copy.copy(self)
        out.records = {k: kept for k, recs in self.records.items()
                       if (kept := [r for r in recs if r.state != canonical])}
        out.price_records = {k: kept for k, items in self.price_records.items()
                             if (kept := [it for it in items if it[3] != canonical])}
        out._models, out._priors, out._price_model = {}, None, None
        out.ingredients = {}
        for iid, ing in self.ingredients.items():
            values = {key: self._aggregate(key, recs, "national") for (i, key), recs in out.records.items()
                      if i == iid}
            price = self._price_info(out.price_records[iid], "national") if iid in out.price_records else None
            out.ingredients[iid] = replace(ing, values=values, price=price)
        return out

    def state_coverage(self, keys=("protein", "moisture")):
        """For every state: per ingredient, whether `keys` and a price are available, and how.

        {state: {ingredient: {"complete": bool, "measured": [...], "estimated": [...], "missing": [...]}}}
        """
        out = {}
        for st in NIGERIA_STATES:
            v = self.view(st)
            out[st] = {}
            for iid, ing in v.ingredients.items():
                entry = {"measured": [], "estimated": [], "missing": []}
                for k in keys:
                    val = ing.values.get(k)
                    entry["missing" if val is None else "estimated" if val.status == "predicted"
                          else "measured"].append(k)
                p = ing.price
                entry["missing" if p is None else "estimated" if p.status == "predicted" else "measured"].append(
                    "price")
                entry["complete"] = not entry["missing"]
                out[st][iid] = entry
        return out

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
