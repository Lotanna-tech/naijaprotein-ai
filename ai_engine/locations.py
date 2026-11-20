"""Nigerian geographic scope: the 36 states and the Federal Capital Territory.

Locations in the data files are written "Nigeria / <State> / <site>", where
the site (market, town, institute) is optional. Only Nigerian states are in
scope; records from anywhere else are rejected by the dataset loader.
"""

COUNTRY = "Nigeria"

NIGERIA_STATES = (
    "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue", "Borno",
    "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu", "Gombe", "Imo", "Jigawa",
    "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi", "Kwara", "Lagos", "Nasarawa", "Niger",
    "Ogun", "Ondo", "Osun", "Oyo", "Plateau", "Rivers", "Sokoto", "Taraba", "Yobe",
    "Zamfara", "FCT",
)

STATE_LABELS = {s: s for s in NIGERIA_STATES}
STATE_LABELS["FCT"] = "Federal Capital Territory (Abuja)"

_LOOKUP = {s.lower(): s for s in NIGERIA_STATES}
_LOOKUP.update({"abuja": "FCT", "federal capital territory": "FCT", "fct abuja": "FCT"})


def parse_location(location):
    """Return (state, site) for 'Nigeria / State / site', or (None, None) if not a Nigerian state."""
    parts = [p.strip() for p in (location or "").split("/") if p.strip()]
    if len(parts) < 2 or parts[0].lower() != COUNTRY.lower():
        return None, None
    state = _LOOKUP.get(parts[1].lower())
    if state is None:
        return None, None
    return state, " / ".join(parts[2:])


def normalise_state(value):
    """Accept a state name (any case, 'Abuja' for FCT); return the canonical name or None."""
    return _LOOKUP.get((value or "").strip().lower())
