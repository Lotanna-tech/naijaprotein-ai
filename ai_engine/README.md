# ai_engine

Pure-Python calculation modules used by the Flask app. They take a
`Dataset` (loaded from `data/`) and return plain dicts, so they can be used
from scripts or notebooks:

```python
from ai_engine.dataset import Dataset
from ai_engine.predictor import Predictor
from ai_engine.blend import simulate
from ai_engine.optimiser import optimise
from ai_engine.explain import explain_optimisation

ds = Dataset("data")
pred = Predictor(ds)
req = dict(ingredient_ids=list(ds.ingredients), target_protein=30, batch_kg=100)
result = optimise(ds, **req)
explanation = explain_optimisation(ds, pred, req, result)
```

Which properties may be averaged is decided in `properties.py`
(`blend_rule`). Only `linear` properties can be optimisation constraints.
