"""Nigerian geographic scope: the 36 states and the Federal Capital Territory.

Locations in the data files are written "Nigeria / <State> / <site>", where
the site (market, town, institute) is optional. Only Nigerian states are in
scope; records from anywhere else are rejected by the dataset loader.

Each state is represented by its capital for distance calculations
(great-circle distance, used to weight nearby states more when estimating
prices for a state without a price survey).
"""

import math

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

# State capital and its approximate coordinates (decimal degrees north, east).
STATE_CAPITALS = {
    "Abia": ("Umuahia", 5.532, 7.486), "Adamawa": ("Yola", 9.204, 12.495),
    "Akwa Ibom": ("Uyo", 5.038, 7.913), "Anambra": ("Awka", 6.210, 7.074),
    "Bauchi": ("Bauchi", 10.316, 9.844), "Bayelsa": ("Yenagoa", 4.927, 6.268),
    "Benue": ("Makurdi", 7.734, 8.521), "Borno": ("Maiduguri", 11.833, 13.150),
    "Cross River": ("Calabar", 4.976, 8.342), "Delta": ("Asaba", 6.198, 6.732),
    "Ebonyi": ("Abakaliki", 6.325, 8.114), "Edo": ("Benin City", 6.335, 5.604),
    "Ekiti": ("Ado-Ekiti", 7.621, 5.221), "Enugu": ("Enugu", 6.458, 7.546),
    "Gombe": ("Gombe", 10.290, 11.167), "Imo": ("Owerri", 5.484, 7.033),
    "Jigawa": ("Dutse", 11.756, 9.339), "Kaduna": ("Kaduna", 10.522, 7.438),
    "Kano": ("Kano", 12.002, 8.592), "Katsina": ("Katsina", 12.991, 7.602),
    "Kebbi": ("Birnin Kebbi", 12.454, 4.198), "Kogi": ("Lokoja", 7.802, 6.733),
    "Kwara": ("Ilorin", 8.497, 4.543), "Lagos": ("Ikeja", 6.602, 3.352),
    "Nasarawa": ("Lafia", 8.494, 8.515), "Niger": ("Minna", 9.614, 6.557),
    "Ogun": ("Abeokuta", 7.148, 3.362), "Ondo": ("Akure", 7.257, 5.206),
    "Osun": ("Osogbo", 7.783, 4.542), "Oyo": ("Ibadan", 7.378, 3.947),
    "Plateau": ("Jos", 9.897, 8.858), "Rivers": ("Port Harcourt", 4.816, 7.050),
    "Sokoto": ("Sokoto", 13.063, 5.243), "Taraba": ("Jalingo", 8.894, 11.360),
    "Yobe": ("Damaturu", 11.747, 11.961), "Zamfara": ("Gusau", 12.163, 6.661),
    "FCT": ("Abuja", 9.077, 7.399),
}

EARTH_RADIUS_KM = 6371.0

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


def distance_km(state_a, state_b):
    """Great-circle (haversine) distance between two state capitals, in km."""
    _, lat1, lon1 = STATE_CAPITALS[state_a]
    _, lat2, lon2 = STATE_CAPITALS[state_b]
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))
