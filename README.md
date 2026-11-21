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
statistical models of how values vary between states, a Monte Carlo
multi-criteria ranking and robust linear programming. Outputs are planning
estimates to decide what to test in the laboratory; texture, taste, shelf life
and safety need laboratory and sensory testing.

Hack the Fork Nigeria 2026, Track 1, Challenge 1 (ingredient discovery and
functional optimisation): the user picks an application (beef patties,
chicken nuggets, meat chunks, sausages) and the plant proteins to consider;
the tool ranks them, recommends a blend, states its certainty and plans the
laboratory tests.

## Features

| Feature | Where | Method |
|---|---|---|
| Ingredient information | `/ingredients/`, `/ingredients/<id>`, `/ingredients/compare` | Composition, minerals, amino acids, functional properties, prices and stock, with source, location and whether each value is measured, calculated, estimated or not available |
| Simulate a blend | `/formulations/simulate` | Choose product, state and the % of each plant protein (protein blend only: no water, oil or nutrition targets). Returns the fit score for the product with its 90% range, protein, cost, how much water and oil the blend can bind, what each protein brings and a certainty card. "Optimise this blend" opens the optimiser with the same choices |
| Rank and optimise | `/formulations/` | Choose product, state and plant proteins (no limits to set). The AI ranks the proteins for the product, recommends the % and kg of each, compares with the simulated blend, gives a certainty card and a laboratory test plan. Works in all 37 states |
| Prediction (internal logic, no page) | used by all features | Fills a missing value only when the data supports it (see below); never replaces a measured value |
| Explanations | shown with every result | Plain-language reasons generated from the numbers: each protein's strong and weak points, why the blend has these shares, and a certainty card saying how sure the result is and what could change it. The method details stay in this README, not on the pages |

### What the AI does

1. **Knows what matters for the application.** `ai_engine/products.py` lists,
   per product, the properties that matter, the direction that helps and an
   importance level (high 3, medium 2, low 1 points). For a beef patty, which
   is cooked, oil absorption is high importance: the fat must stay bound in
   the patty. Weights are the points normalised over the properties with data.
   Only protein content and functional properties are scored. On both pages
   the importance of each property can be changed for a run (high, medium,
   low or off), for example to switch protein content off when every
   candidate is an isolate; the result says which levels were changed.
2. **Ranks the selected proteins under uncertainty** (`ai_engine/application.py`).
   Each value is scaled 0–1 between the lowest and highest verified value of
   the five ingredients; the fit score is 100 × Σ weight × scaled value. In
   2,000 Monte Carlo simulations every value is drawn from its lot-to-lot
   spread or estimate range and the weights from a Dirichlet distribution
   around the importance levels (about ±30%), as in SMAA (Lahdelma et al.
   1998). Output: score with a 90% range, the chance of each rank, and the
   points each property contributes.
3. **Optimises the blend robustly.** The blend's score is the share-weighted
   score of its proteins (no interaction assumed). A linear program (SciPy
   HiGHS) maximises the CVaR, the average of the blend's worst 10% of simulated
   scores (Rockafellar & Uryasev 2000), within the supplier maximum shares;
   among blends within 1 point of that, it takes the cheapest.
4. **Certainty card.** High when the top pick wins at least 80% of the
   simulations, medium from 55%, low below; lowered one level when most
   values are rough estimates. It lists the data quality, whether doubling or
   ignoring any property's importance changes the top pick, and the
   properties that matter but have no data.
5. **Narrows the work for the laboratory.** Up to 3 shortlisted proteins go
   into an augmented simplex-centroid mixture design (12 blends instead of
   1,001 at 10% steps for 5 proteins), with the properties to measure.

Why no model is "trained" for blends yet: there is no measured data on blends
(`data/formulation_experiments.csv` is empty), so a trained blend model would
be invented. The lab plan produces that data. Once at least 8 measured blends
are entered, the Scheffé mixture model in `ai_engine/predictor.py` fits them,
learns the interactions and is checked by leave-one-out cross-validation
(Q² ≥ 0.5) before it is used.

### How it was evaluated, and where not to trust it

- **Ranking backtest** (`flask --app run.py evaluate-ranking`): for each state
  with its own measurements for 3+ ingredients (Benue, Kano), its data are
  hidden, its values estimated from the other states, and the ranking
  compared with the one from its own data. Same top pick in 8 of 8 tests (4
  products × 2 states), mean Kendall τ 0.92. Only two states could be tested,
  and defatted groundnut leads in both.
- **State estimates:** leave-one-state-out errors for every property (location
  model) and about 20% for prices in a state without a survey (price model).
- **Do not trust:** absolute scores (they compare the five candidates only);
  blend properties (interactions are not modelled until lab data exist);
  rankings that rest on rough estimates; properties with no data at all
  (water- and oil-holding capacity, emulsion stability are not scored); other
  processing routes than those in the data.

### Data rules

- Only `measured` and `sourced` values are used in calculations. Provider
  estimates (`estimated`) and placeholders (`assumed`) stay in the CSV files
  and are shown on profile pages as "unverified (not used)".
- Every state gets a value for every property that was measured somewhere in
  Nigeria, and a price for every ingredient. A state's own records always come
  first; otherwise the value or price is estimated from the other states,
  marked "estimated", shown with a 95% range and graded **checked** or
  **rough estimate** (see below). Stock is never estimated.
- Carbohydrate that a source does not report is calculated per sample by
  difference: 100 − protein − fat − ash − fibre (dry basis), marked
  "calculated".
- Composition must add up: the loader reports a sample whose dry-matter
  components exceed 100 g/100 g, or whose values labelled dry basis only reach
  100 together with moisture (the signature of an as-is table).
- Products say which properties matter, why and how much
  (`ai_engine/products.py`); there are no numeric product targets.

### Location model (composition and functional properties)

For each ingredient and property, `ai_engine/location_model.py` fits a
random-intercept model (state as a random effect; method-of-moments variance
components; empirical-Bayes shrinkage of state means). A state without its own
records gets the national value μ with the predictive interval for a new
state, μ ± 1.96·√(τ² + Var μ). Each state with data is then left out in turn
and predicted from the others:

- **checked**: at least 3 states and mean relative error ≤ 10%;
- **rough estimate**: fewer states or larger differences. The standard
  deviation is the largest of the model's own, the leave-one-state-out error
  and the median error of the same property in the other ingredients
  (borrowing strength). A property never compared between states (for
  example soybean minerals, only measured in the FCT) uses the largest error
  seen for any property (currently 33%).

Recorded state values are never replaced. There are no environmental or
variety covariates yet because the data has none.

### Price model

`ai_engine/price_model.py` fits a Country-Product-Dummy model on the verified
lots, ln p = α(ingredient) + λ(state), the method the World Bank's
International Comparison Program uses to compare price levels from an
incomplete price table. A state without a price survey gets λ by
inverse-distance weighting (1/d²) of the surveyed states' levels, using the
distance between state capitals. Errors are measured by leave-one-out on the
data: currently about 20% for a state without a survey and 5% for an
ingredient missing in a surveyed state. The provider's unverified lots are never used for a price;
where they fall outside an estimate's range, the estimate is downgraded to a
rough estimate. A state's price level scales all ingredients alike, so the
optimal proportions are more reliable than the total cost.

`python scripts/export_state_values.py` writes every value and price used for
each of the 37 states to `data/derived/state_values.csv` for review (the app
does not read it). `flask --app run.py validate-data` reports the coverage.

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
  price_model.py         prices for states and ingredients without a verified lot
  products.py            the four products and which properties matter
  dataset.py             CSV loading, verification, basis conversion, state views
  application.py         ranking, blend score, robust optimisation, certainty card, lab plan
  predictor.py           missing-value prediction and the blend (mixture) model
app/
  routes/ingredients.py  ingredient pages and /api/ingredients
  routes/api.py          /api/simulate, /api/optimise, /api/formulations
  routes/formulations.py optimise, simulate and saved-runs pages
data/                    dataset (see data/README.md)
scripts/import_htf_dataset.py  imports the Hack the Fork Nigeria 2026 CSV
scripts/export_state_values.py writes data/derived/state_values.csv (every state)
tests/                   pytest suite
```

## API

| Route | Body | Returns |
|---|---|---|
| `POST /api/simulate` | `{"product": "beef", "location": "Kano", "recipe": {"soybean": 60, "cowpea": 40}, "batch_kg": 100}` | `{result, explanation}`: blend score and range, protein, cost, water/oil binding, per-protein ranking, certainty card |
| `POST /api/optimise` | `{"product": "beef", "location": "Kano", "ingredient_ids": ["soybean", "cowpea", "defatted_groundnut"], "batch_kg": 100, "budget": null, "baseline": {"soybean": 60, "cowpea": 40}}` | `{result, explanation}`: ranking, recommended blend, comparison with `baseline`, certainty card, lab plan |
| `GET /api/ingredients/<id>?location=Kano` | | profile with records, sources and state provenance |

`product` is one of `beef`, `chicken`, `chunks`, `sausage`. Recipe and
baseline amounts are % of the protein blend and must add up to 100; water and
oil are not part of the blend. `baseline` and `budget` are optional. Both
routes accept `"importance": {"protein": "off", "least_gelation": "medium"}`
(high, medium, low or off) to change how much a property counts for this run;
only protein and functional properties can be weighed. Results are
reproducible (fixed random seed). Add `"save": true, "name": "..."` to save a
run. Both POST routes require login.

## Licences

Flask, NumPy, SciPy (BSD) and scikit-learn (BSD) are used as libraries; no
code was copied from them. The reuse terms of the Hack the Fork Nigeria 2026
dataset (`data/raw/`) have not been confirmed and should be checked before
publication. Published article values are cited per row in `data/sources.csv`.
