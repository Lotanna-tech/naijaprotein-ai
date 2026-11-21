"""The four plant-based meat product categories.

For each product this module records WHICH properties matter, WHY, in plain
language, in which direction a higher value usually helps, and HOW MUCH it
matters (importance: high, medium or low). The importance levels become the
weights of the application-fit ranking (ai_engine/application.py); they are
food-science judgement for meat analogues, not measurements, so the ranking
varies them at random to check that its result does not hinge on them.

There are no numeric targets here: those must come from a cited source
(data/product_targets.csv) or from the user.
"""

from dataclasses import dataclass, field

HIGHER = "higher"
LOWER = "lower"

# Importance level -> weight points (normalised over the properties that have data).
IMPORTANCE_POINTS = {"high": 3, "medium": 2, "low": 1}


@dataclass(frozen=True)
class Relevance:
    property: str
    direction: str      # which direction usually helps this product
    why: str
    importance: str = "medium"   # high / medium / low


@dataclass(frozen=True)
class Product:
    id: str
    name: str
    examples: str
    summary: str
    needs: tuple        # plain-language characteristics
    relevant: tuple = field(default_factory=tuple)

    def relevance(self, key):
        return next((r for r in self.relevant if r.property == key), None)


PRODUCTS = {
    "beef": Product(
        "beef", "Plant-based beef", "Burger patties, minced meat",
        "Needs to bind into a firm patty or mince that stays juicy after cooking.",
        ("binding", "firmness", "moisture retention", "juiciness", "fat retention", "meat-like texture"),
        (
            Relevance("protein", HIGHER, "Protein forms the bound, meat-like structure of a patty or mince.", "high"),
            Relevance("water_holding_capacity", HIGHER, "Keeps water in the patty during cooking, for juiciness.",
                      "high"),
            Relevance("water_absorption", HIGHER, "Takes up added water, which keeps the patty juicy and lowers "
                                                  "cost.", "high"),
            Relevance("oil_holding_capacity", HIGHER, "Holds the fat added for flavour and mouthfeel.", "medium"),
            Relevance("oil_absorption", HIGHER, "Burgers are cooked: the fat added for flavour and juiciness must "
                                                "stay bound in the patty instead of running out in the pan.", "high"),
            Relevance("least_gelation", LOWER, "Forms a heat-set gel more easily, which binds the patty so it does "
                                               "not crumble.", "high"),
            Relevance("emulsifying_capacity_pct", HIGHER, "Keeps fat and water mixed in the raw patty mix.",
                      "medium"),
            Relevance("emulsifying_capacity", HIGHER, "Keeps fat and water mixed in the raw patty mix.", "medium"),
            Relevance("protein_solubility", HIGHER, "Soluble protein helps binding and emulsifying.", "medium"),
        ),
    ),
    "chicken": Product(
        "chicken", "Plant-based chicken", "Nuggets, chicken-style chunks",
        "Needs a tender, cohesive bite that holds moisture.",
        ("binding", "moisture retention", "tenderness", "cohesion", "fibrous structure"),
        (
            Relevance("protein", HIGHER, "Protein gives nuggets a cohesive, chicken-like bite.", "high"),
            Relevance("water_holding_capacity", HIGHER, "Keeps the pieces moist and tender.", "high"),
            Relevance("water_absorption", HIGHER, "Takes up added water for a moist, tender bite.", "high"),
            Relevance("least_gelation", LOWER, "Gels more easily, which holds nuggets together.", "high"),
            Relevance("protein_solubility", HIGHER, "Soluble protein helps the pieces bind and hold together.",
                      "medium"),
            Relevance("emulsifying_capacity_pct", HIGHER, "Helps hold oil in a nugget mix.", "low"),
            Relevance("emulsifying_capacity", HIGHER, "Helps hold oil in a nugget mix.", "low"),
            Relevance("oil_absorption", HIGHER, "Nuggets are usually fried: binding oil affects mouthfeel and "
                                                "frying behaviour.", "medium"),
        ),
    ),
    "chunks": Product(
        "chunks", "Plant-based meat chunks", "Textured and structured meat substitutes",
        "Needs a firm, chewy structure that rehydrates well.",
        ("structure", "hydration", "water retention", "chewiness", "texture"),
        (
            Relevance("protein", HIGHER, "Textured chunks need a high protein content to form a firm, chewy "
                                         "structure.", "high"),
            Relevance("water_absorption", HIGHER, "Rehydrates well, which affects chewiness and yield.", "high"),
            Relevance("water_holding_capacity", HIGHER, "Keeps water in the structure after cooking.", "high"),
            Relevance("least_gelation", LOWER, "Forms a gel network more easily, which supports structure.", "high"),
            Relevance("protein_solubility", HIGHER, "Soluble protein can form a stronger network on heating.",
                      "medium"),
            Relevance("bulk_density", HIGHER, "Affects how the dry mix packs and handles in processing.", "low"),
        ),
    ),
    "sausage": Product(
        "sausage", "Plant-based sausages", "Meat-free sausages",
        "Needs a stable emulsion and a gel that sets firm when cooked.",
        ("binding", "gel formation", "emulsification", "emulsion stability", "firmness", "water and oil retention"),
        (
            Relevance("emulsifying_capacity_pct", HIGHER, "Keeps fat and water mixed in the sausage batter.", "high"),
            Relevance("emulsifying_capacity", HIGHER, "Keeps fat and water mixed in the sausage batter.", "high"),
            Relevance("emulsion_stability", HIGHER, "Stops fat separating when the sausage is cooked.", "high"),
            Relevance("least_gelation", LOWER, "Sets into a firm gel more easily on cooking.", "high"),
            Relevance("protein_solubility", HIGHER, "Soluble protein forms the emulsion and the gel.", "high"),
            Relevance("protein", HIGHER, "Protein forms the gel network that holds the sausage together.", "medium"),
            Relevance("water_holding_capacity", HIGHER, "Holds water in the cooked sausage.", "medium"),
            Relevance("oil_holding_capacity", HIGHER, "Holds the fat in the sausage.", "medium"),
            Relevance("oil_absorption", HIGHER, "Sausages are cooked: binding the fat reduces fat loss.", "medium"),
        ),
    ),
}

def get(product_id):
    p = PRODUCTS.get(product_id)
    if p is None:
        raise ValueError(f"Unknown product '{product_id}'. Choose one of: {', '.join(PRODUCTS)}.")
    return p


def default_targets(dataset, product_id):
    """Sourced default targets for a product (empty if none are recorded)."""
    out = []
    for t in dataset.product_targets:
        if t.product == product_id:
            src = dataset.sources.get(t.source_id)
            out.append({"property": t.property, "min": t.min, "max": t.max, "source_id": t.source_id,
                        "source": src.citation if src else "", "notes": t.notes})
    return out


def to_dict(product):
    return {
        "id": product.id, "name": product.name, "examples": product.examples, "summary": product.summary,
        "needs": list(product.needs),
        "relevant": [{"property": r.property, "direction": r.direction, "why": r.why, "importance": r.importance}
                     for r in product.relevant],
    }
