"""Bridges Flask and the calculation engine in ai_engine/.

The dataset is loaded once and reloaded automatically when any CSV file in
DATA_DIR changes, so edits from new articles show up without a restart.
"""

import os
import threading

from flask import current_app

from ai_engine.dataset import REQUIRED_FILES, Dataset
from ai_engine.predictor import Predictor

_lock = threading.Lock()
_cache = {}


def _signature(data_dir):
    sig = []
    for name in REQUIRED_FILES:
        path = os.path.join(data_dir, name)
        sig.append(os.path.getmtime(path) if os.path.exists(path) else None)
    return tuple(sig)


def get_engine():
    """Return (dataset, predictor) for the configured DATA_DIR."""
    data_dir = current_app.config["DATA_DIR"]
    sig = _signature(data_dir)
    with _lock:
        cached = _cache.get(data_dir)
        if cached is None or cached[0] != sig:
            dataset = Dataset(data_dir, current_app.config.get("INGREDIENT_SCOPE"))
            for issue in dataset.issues:
                current_app.logger.warning("Dataset: %s", issue)
            cached = (sig, dataset, Predictor(dataset), {})
            _cache[data_dir] = cached
    return cached[1], cached[2]


def get_view(location=None):
    """(view, predictor, national) for a Nigerian state, or for all Nigeria if location is empty.

    view     - values for the state: measurements from that state, else validated
               estimates from other states, else unavailable; then missing values
               estimated from composition only where the data supports it.
    national - the same for all Nigerian data combined, used to explain location effects.
    """
    from ai_engine.locations import normalise_state
    dataset, predictor = get_engine()
    views = _cache[current_app.config["DATA_DIR"]][3]
    location = (location or "").strip() or None
    if location:
        state = normalise_state(location)
        if state is None:
            raise ValueError(f"'{location}' is not a Nigerian state or the FCT.")
        location = state
    with _lock:
        for loc in {None, location}:
            if loc not in views:
                views[loc] = predictor.fill_missing(dataset.view(loc))
    return views[location], predictor, views[None]
