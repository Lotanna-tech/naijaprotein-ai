# NaijaProtein AI

Decision support for making **plant-based meat** from five Nigerian protein
ingredients:

| Ingredient | Scientific name |
|---|---|
| Cowpea | *Vigna unguiculata* |
| Bambara groundnut | *Vigna subterranea* |
| Defatted groundnut (oil removed; not whole groundnut) | *Arachis hypogaea* |
| African yam bean | *Sphenostylis stenocarpa* |
| Soybean | *Glycine max* |

Products: **plant-based beef** (burger patties, minced meat), **plant-based
chicken** (nuggets, chunks), **plant-based meat chunks** and **plant-based
sausages**. Locations: Nigeria only, the 36 states and the Federal Capital
Territory.

It is not a chatbot. Every result comes from the verified dataset in `data/`,
mass-balance arithmetic, a statistical location model and linear programming.
Outputs are planning estimates; texture, taste, shelf life and safety need
laboratory and sensory testing.

## Features

| Feature | Where | Method |
|---|---|---|
| Ingredient information | `/ingredients/`, `/ingredients/<id>`, `/ingredients/compare` | Composition, minerals, amino acids, functional properties, prices and stock, with source, location and whether each value is measured, calculated, estimated or not available |
| Blend simulation | `/formulations/simulate` | Choose product, state, ingredients and amounts (including added water and edible oil, as % of the total mix). Mass balance on an as-is basis; amino acid score vs WHO/FAO/UNU 2007; cost and stock by state. Functional properties of a mix are never averaged |
| Optimisation | `/formulations/` | SciPy `linprog` (HiGHS): lowest cost per kg subject to nutrition targets, ingredient limits, water and oil range, stock and budget for the chosen state. Infeasible requests get a plain diagnosis |
| Prediction (internal logic, no page) | used by all features | Fills a missing value only when the data supports it (see below); never replaces a measured value |
| Explanations | shown with every result | Generated from the calculation: why each ingredient is used, which limits decide the result and what relaxing them would cost, evidence and missing data |

### Data rules

- Only `measured` and `sourced` values are used in calculations. Provider
  estimates (`estimated`) and placeholders (`assumed`) stay in the CSV files
  and are shown on profile pages as "unverified (not used)".
- Prices are never estimated. A state without verified price lots cannot be
  costed or optimised. Currently: Akwa Ibom, Benue and Kano.
- Product nutrition targets are only pre-filled from sourced rows in
  `data/product_targets.csv` (currently none). Products otherwise only say
  which properties matter and why.

### Location model

For each ingredient and property, `ai_engine/location_model.py` fits a
random-intercept model (state as a random effect; method-of-moments variance
components; empirical-Bayes shrinkage of state means). It is used for a state
without its own records only if leave-one-state-out validation passes (at
least 3 states, mean relative error ≤ 10%). The value is then marked
"estimated" with an interval. Otherwise it is "not available". Recorded state
values are never replaced. There are no environmental or variety covariates
yet because the data has none.

### Prediction of missing ingredient values

`ai_engine/predictor.py` estimates a missing property only from verified
samples of the same ingredient: at least 8 samples, leave-one-out Q² ≥ 0.5,
inputs inside the training range. Blend functional properties need a
Scheffé mixture model fitted on measured blends in
`data/formulation_experiments.csv` (currently empty, so they show "not
available"). Every refusal is shown with its reason.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
flask --app run.py db upgrade        # users and saved runs
flask --app run.py validate-data     # checks the CSV dataset
python run.py                        # http://127.0.0.1:5000
python -m pytest -q                  # tests (synthetic data in temp folders)
```

Behind the `/proxy/5000/` reverse proxy, URLs get the prefix automatically
(`URL_PREFIX`, see `app/config.py`).

## Layout

```
ai_engine/               calculation engine, no Flask dependency
  properties.py          property registry, plain-language labels, unit conversions
  locations.py           Nigerian states + FCT, location parsing
  location_model.py      per-property state model and validation
  products.py            the four products and which properties matter
  dataset.py             CSV loading, verification, basis conversion, state views
  blend.py               blend simulation (water and oil included)
  optimiser.py           linear programming and infeasibility diagnosis
  predictor.py           missing-value prediction with sufficiency checks
  explain.py             explanations from the calculations
app/
  routes/ingredients.py  ingredient pages and /api/ingredients
  routes/api.py          /api/simulate, /api/optimise, /api/formulations
  routes/formulations.py optimise, simulate and saved-runs pages
data/                    dataset (see data/README.md)
scripts/import_htf_dataset.py  imports the Hack the Fork Nigeria 2026 CSV
tests/                   pytest suite
```

## API

| Route | Body | Returns |
|---|---|---|
| `POST /api/simulate` | `{"product": "beef", "location": "Kano", "recipe": {"soybean": 30, "cowpea": 30, "water": 35, "oil": 5}, "batch_kg": 100, "targets": {"protein": {"min": 15}}, "water_price": null, "oil_price": null}` | `{result, explanation}` |
| `POST /api/optimise` | `{"product": "beef", "location": "Kano", "ingredient_ids": ["soybean", "cowpea"], "batch_kg": 100, "targets": {"protein": {"min": 15}}, "limits": {"soybean": {"min": 0, "max": 70}}, "water": {"min": 30, "max": 50}, "oil": {"min": 0, "max": 8}, "budget": null, "limit_water_to_holding": false}` | `{result, profile, explanation}` |
| `GET /api/ingredients/<id>?location=Kano` | | profile with records, sources and state provenance |

`product` is one of `beef`, `chicken`, `chunks`, `sausage`. Recipe amounts
are % of the total mix and must add up to 100. Limits are % of the protein
ingredients. Functional targets are not accepted as LP constraints; they are
checked on the result instead. Add `"save": true, "name": "..."` to save a
run. Both POST routes require login.

## Licences

Flask, NumPy, SciPy (BSD) and scikit-learn (BSD) are used as libraries; no
code was copied from them. The reuse terms of the Hack the Fork Nigeria 2026
dataset (`data/raw/`) have not been confirmed and should be checked before
publication. Published article values are cited per row in `data/sources.csv`.
