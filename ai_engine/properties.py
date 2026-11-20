"""Registry of every property the platform knows about.

Each property declares how (or whether) it can be combined in a blend:

- "linear": mass balance, sum(w_i * v_i) over all formulation components,
  after converting every value to the same basis. Valid for mass fractions
  (proximate composition, minerals, antinutrient contents) and for cost.
- "protein_weighted": amino acids expressed per 16 g N (g/100 g protein) mix in
  proportion to the protein each ingredient contributes:
  sum(w_i * P_i * v_i) / sum(w_i * P_i).
- "nonlinear": functional properties (water/oil holding and absorption,
  solubility, emulsifying, gelation, foaming, bulk density). Ingredient
  interactions mean a weighted average is not justified, so blend values come
  only from a validated model on measured blends, otherwise they are not given.

`plain` is the short, non-technical explanation shown to users.
"""

from dataclasses import dataclass

LINEAR = "linear"
PROTEIN_WEIGHTED = "protein_weighted"
NONLINEAR = "nonlinear"

# Bases a value can be reported on.
DRY_BASIS = "dry"
AS_IS_BASIS = "as_is"


@dataclass(frozen=True)
class PropertyDef:
    key: str
    label: str
    unit: str
    group: str
    blend_rule: str
    # Mass-fraction properties that can be converted between as-is and dry basis.
    basis_convertible: bool = False
    # Allowed as optimisation constraints (only mass-balance properties are).
    optimisable: bool = False
    plain: str = ""


_DEFS = [
    # Nutritional composition (g/100 g)
    PropertyDef("protein", "Protein", "g/100 g", "proximate", LINEAR, True, True,
                "Crude protein (nitrogen x 6.25 unless the source says otherwise)."),
    PropertyDef("fat", "Fat", "g/100 g", "proximate", LINEAR, True, True, "Crude fat (oil) content."),
    PropertyDef("carbohydrate", "Carbohydrate", "g/100 g", "proximate", LINEAR, True, True,
                "Carbohydrate, usually calculated by difference."),
    PropertyDef("fibre", "Fibre", "g/100 g", "proximate", LINEAR, True, True, "Crude fibre content."),
    PropertyDef("ash", "Ash (minerals)", "g/100 g", "proximate", LINEAR, True, True,
                "Total mineral content left after burning the sample."),
    PropertyDef("moisture", "Moisture", "g/100 g", "proximate", LINEAR, False, True,
                "Water content of the material as it is."),
    # Minerals (mg/100 g)
    PropertyDef("calcium", "Calcium", "mg/100 g", "mineral", LINEAR, True, True),
    PropertyDef("iron", "Iron", "mg/100 g", "mineral", LINEAR, True, True),
    PropertyDef("zinc", "Zinc", "mg/100 g", "mineral", LINEAR, True, True),
    PropertyDef("potassium", "Potassium", "mg/100 g", "mineral", LINEAR, True, True),
    PropertyDef("phosphorus", "Phosphorus", "mg/100 g", "mineral", LINEAR, True, True),
    # Essential amino acids (g/16 g N, i.e. g per 100 g protein)
    PropertyDef("lysine", "Lysine", "g/16 g N", "amino_acid", PROTEIN_WEIGHTED),
    PropertyDef("methionine", "Methionine", "g/16 g N", "amino_acid", PROTEIN_WEIGHTED),
    PropertyDef("cysteine", "Cysteine", "g/16 g N", "amino_acid", PROTEIN_WEIGHTED),
    PropertyDef("threonine", "Threonine", "g/16 g N", "amino_acid", PROTEIN_WEIGHTED),
    PropertyDef("tryptophan", "Tryptophan", "g/16 g N", "amino_acid", PROTEIN_WEIGHTED),
    PropertyDef("leucine", "Leucine", "g/16 g N", "amino_acid", PROTEIN_WEIGHTED),
    # Functional (techno-functional) properties
    PropertyDef("water_holding_capacity", "Water-holding capacity", "g/g", "functional", NONLINEAR,
                plain="How well an ingredient or mixture holds onto water, for example during cooking."),
    PropertyDef("water_absorption", "Water absorption capacity", "g/g", "functional", NONLINEAR,
                plain="How much water an ingredient takes up when mixed with water."),
    PropertyDef("oil_holding_capacity", "Oil-holding capacity", "g/g", "functional", NONLINEAR,
                plain="How well an ingredient or mixture holds onto oil."),
    PropertyDef("oil_absorption", "Oil absorption capacity", "g/g", "functional", NONLINEAR,
                plain="How much oil an ingredient takes up when mixed with oil."),
    PropertyDef("protein_solubility", "Protein solubility", "%", "functional", NONLINEAR,
                plain="How much of the protein dissolves in water. It affects binding and emulsifying."),
    PropertyDef("emulsifying_capacity", "Emulsifying capacity (ml oil/g)", "ml oil/g", "functional", NONLINEAR,
                plain="How much oil an ingredient can mix with water before the mixture separates."),
    PropertyDef("emulsifying_capacity_pct", "Emulsifying capacity (%)", "%", "functional", NONLINEAR,
                plain="How well an ingredient keeps oil and water mixed, measured as a percentage. "
                      "Not comparable with values in ml oil/g."),
    PropertyDef("emulsion_stability", "Emulsion stability", "%", "functional", NONLINEAR,
                plain="How well the oil and water mixture stays together over time or on heating."),
    PropertyDef("least_gelation", "Least gelation concentration", "% w/v", "functional", NONLINEAR,
                plain="The lowest amount needed to form a gel when heated. Lower means it gels more easily."),
    PropertyDef("foaming_capacity", "Foaming capacity", "%", "functional", NONLINEAR,
                plain="How much foam forms when whipped."),
    PropertyDef("foaming_stability", "Foaming stability", "%", "functional", NONLINEAR,
                plain="How long the foam lasts."),
    PropertyDef("bulk_density", "Bulk density", "g/mL", "functional", NONLINEAR,
                plain="How heavy the dry powder is for its volume. Affects handling and packing."),
    # Biochemical characteristics: antinutrients and bioactive compounds
    PropertyDef("trypsin_inhibitor", "Trypsin inhibitor activity", "TIU/mg", "biochemical",
                LINEAR, True, True, "Reduces protein digestion. Lowered by heat treatment."),
    PropertyDef("phytate", "Phytate", "mg/100 g", "biochemical", LINEAR, True, True,
                "Binds minerals such as iron and zinc. Lowered by soaking, germination or fermentation."),
    PropertyDef("tannin", "Tannin", "mg/100 g", "biochemical", LINEAR, True, True,
                "Can reduce protein digestibility and affect colour and taste."),
    PropertyDef("total_phenolics", "Total phenolic content", "mg GAE/100 g", "biochemical", LINEAR, True, True,
                "Plant compounds with antioxidant activity."),
    PropertyDef("protein_digestibility", "In vitro protein digestibility", "%", "biochemical", NONLINEAR,
                plain="Share of the protein broken down by digestive enzymes in a laboratory test."),
]

PROPERTIES = {p.key: p for p in _DEFS}

GROUP_LABELS = {
    "proximate": "Nutritional composition",
    "mineral": "Minerals",
    "amino_acid": "Essential amino acids",
    "functional": "Functional properties",
    "biochemical": "Biochemical characteristics",
}

GROUP_NOTES = {
    "proximate": "Dry-matter basis, except moisture. Values reported as-is are converted using the moisture of the same sample.",
    "mineral": "Dry-matter basis.",
    "amino_acid": "Grams per 100 g of protein.",
    "functional": "How the ingredient behaves in a product. Results depend on the test method and conditions, so check "
                  "the method before comparing studies. These properties do not simply average out in a blend.",
    "biochemical": "Antinutrients and bioactive compounds. These affect nutrition and health, not water or oil binding.",
}

# measured  - your own laboratory measurement
# sourced   - taken from a cited study or a verified dataset record
# estimated - an estimate made by the data provider, not verified (not used in calculations)
# assumed   - assumption or placeholder (not used in calculations)
VALUE_STATUSES = ("measured", "sourced", "estimated", "assumed")
VERIFIED_STATUSES = ("measured", "sourced")

# Units articles commonly report, and how they convert to the canonical unit.
# value_in_canonical_unit = reported_value * factor. Conversions that rely on an
# assumption carry a note, which is shown next to the value.
_ABSORPTION = {
    "g/100 g": (0.01, ""),
    "% dry sample": (0.01, "reported as g per 100 g dry sample"),
}
UNIT_CONVERSIONS = {
    "water_absorption": {**_ABSORPTION, "ml/g": (1.0, "ml water/g converted assuming water density 1.00 g/mL")},
    "water_holding_capacity": {**_ABSORPTION, "ml/g": (1.0, "ml water/g converted assuming water density 1.00 g/mL")},
    "oil_absorption": {**_ABSORPTION, "ml/g": (0.92, "ml oil/g converted assuming oil density 0.92 g/mL")},
    "oil_holding_capacity": {**_ABSORPTION, "ml/g": (0.92, "ml oil/g converted assuming oil density 0.92 g/mL")},
}


def get(key):
    return PROPERTIES[key]


def convert_unit(key, value, unit):
    """Return (value_in_canonical_unit, note) or None if the unit is not accepted."""
    if unit == get(key).unit:
        return value, ""
    conv = UNIT_CONVERSIONS.get(key, {}).get(unit)
    if conv is None:
        return None
    factor, note = conv
    return value * factor, note or f"converted from {unit}"


def by_group():
    groups = {g: [] for g in GROUP_LABELS}
    for p in _DEFS:
        groups[p.group].append(p)
    return groups
