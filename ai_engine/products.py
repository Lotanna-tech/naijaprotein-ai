"""The four plant-based meat product categories.

For each product this module records WHICH properties matter and WHY, in plain
language, and in which direction a higher value usually helps. It contains no
numeric targets: those must come from a cited source (data/product_targets.csv)
or from the user. The relevance notes are general food-science guidance for
meat analogues, not product specifications.
"""

from dataclasses import dataclass, field

HIGHER = "higher"
LOWER = "lower"


@dataclass(frozen=True)
class Relevance:
    property: str
    direction: str      # which direction usually helps this product
    why: str


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
            Relevance("water_holding_capacity", HIGHER, "Keeps water in the patty during cooking, for juiciness."),
            Relevance("water_absorption", HIGHER, "Takes up added water, which helps juiciness and lowers cost."),
            Relevance("oil_holding_capacity", HIGHER, "Holds the fat added for flavour and mouthfeel."),
            Relevance("oil_absorption", HIGHER, "Takes up added oil, reducing fat loss on cooking."),
            Relevance("least_gelation", LOWER, "Forms a heat-set gel more easily, which binds the patty."),
            Relevance("emulsifying_capacity_pct", HIGHER, "Keeps fat and water mixed in the patty mix."),
            Relevance("emulsifying_capacity", HIGHER, "Keeps fat and water mixed in the patty mix."),
            Relevance("protein_solubility", HIGHER, "Soluble protein helps binding and emulsifying."),
        ),
    ),
    "chicken": Product(
        "chicken", "Plant-based chicken", "Nuggets, chicken-style chunks",
        "Needs a tender, cohesive bite that holds moisture.",
        ("binding", "moisture retention", "tenderness", "cohesion", "fibrous structure"),
        (
            Relevance("water_holding_capacity", HIGHER, "Keeps the pieces moist and tender."),
            Relevance("water_absorption", HIGHER, "Takes up added water for a moist bite."),
            Relevance("least_gelation", LOWER, "Gels more easily, which holds nuggets together."),
            Relevance("protein_solubility", HIGHER, "Soluble protein helps the pieces bind and hold together."),
            Relevance("emulsifying_capacity_pct", HIGHER, "Helps hold oil in a nugget mix."),
            Relevance("emulsifying_capacity", HIGHER, "Helps hold oil in a nugget mix."),
            Relevance("oil_absorption", HIGHER, "Holds oil, which affects mouthfeel and frying behaviour."),
        ),
    ),
    "chunks": Product(
        "chunks", "Plant-based meat chunks", "Textured and structured meat substitutes",
        "Needs a firm, chewy structure that rehydrates well.",
        ("structure", "hydration", "water retention", "chewiness", "texture"),
        (
            Relevance("water_absorption", HIGHER, "Rehydrates well, which affects chewiness and yield."),
            Relevance("water_holding_capacity", HIGHER, "Keeps water in the structure after cooking."),
            Relevance("least_gelation", LOWER, "Forms a gel network more easily, which supports structure."),
            Relevance("protein_solubility", HIGHER, "Soluble protein can form a stronger network on heating."),
            Relevance("bulk_density", HIGHER, "Affects how the dry mix packs and handles in processing."),
        ),
    ),
    "sausage": Product(
        "sausage", "Plant-based sausages", "Meat-free sausages",
        "Needs a stable emulsion and a gel that sets firm when cooked.",
        ("binding", "gel formation", "emulsification", "emulsion stability", "firmness", "water and oil retention"),
        (
            Relevance("emulsifying_capacity_pct", HIGHER, "Keeps fat and water mixed in the sausage batter."),
            Relevance("emulsifying_capacity", HIGHER, "Keeps fat and water mixed in the sausage batter."),
            Relevance("emulsion_stability", HIGHER, "Stops fat separating when the sausage is cooked."),
            Relevance("least_gelation", LOWER, "Sets into a firm gel more easily on cooking."),
            Relevance("protein_solubility", HIGHER, "Soluble protein forms the emulsion and the gel."),
            Relevance("water_holding_capacity", HIGHER, "Holds water in the cooked sausage."),
            Relevance("oil_holding_capacity", HIGHER, "Holds the fat in the sausage."),
            Relevance("oil_absorption", HIGHER, "Holds fat, reducing fat loss on cooking."),
        ),
    ),
}

DISCLAIMER = ("These are computational estimates from ingredient data. They do not confirm the texture, taste, "
              "sensory acceptability, shelf life, microbiological safety or production performance of a real "
              "product; those need laboratory and product trials.")


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
        "relevant": [{"property": r.property, "direction": r.direction, "why": r.why} for r in product.relevant],
    }
