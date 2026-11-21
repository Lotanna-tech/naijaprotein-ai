# ai_engine

Pure-Python calculation modules used by the Flask app. They take a
`Dataset` (loaded from `data/`) and return plain dicts, so they can be used
from scripts or notebooks:

```python
from ai_engine.dataset import Dataset
from ai_engine import application

ds = Dataset("data")
kano = ds.view("Kano")
result, explanation = application.optimise(kano, "beef", list(ds.ingredients), batch_kg=100)
result["ranking"][0]["name"], result["certainty"]["label"], result["blend"]["shares"]
result, explanation = application.simulate(kano, "beef", {"soybean": 60, "cowpea": 40}, batch_kg=100)
```

`application.py` drives the Simulate and Optimise pages (ranking, robust
optimisation, certainty card, lab plan).

Which properties may be averaged is decided in `properties.py`
(`blend_rule`). Only `linear` properties can be optimisation constraints.

`ds.view("Lagos")` gives the values for one state: its own records first,
then estimates from the other states (`location_model.py`) and estimated
prices (`price_model.py`), each with a 95% range and a `confidence` of
`checked` or `indicative`.
