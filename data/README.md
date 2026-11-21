# Ingredient dataset

Every feature reads these CSV files. Edit them, then run:

```bash
flask --app run.py validate-data
```

The app reloads the files automatically when they change.

## Files

| File | One row per | Key columns |
|---|---|---|
| `sources.csv` | reference (article, dataset, market survey) | `source_id`, `citation`, `year`, `doi_or_url`, `notes` |
| `ingredients.csv` | ingredient | `id`, `name`, `scientific_name`, `processing`, `default_min_inclusion`, `default_max_inclusion` (0 to 1) |
| `ingredient_properties.csv` | reported value | `ingredient_id`, `property`, `value`, `unit`, `basis`, `method`, `processing`, `source_id`, `status`, `sample`, `location`, `variety` |
| `ingredient_prices.csv` | price lot | `ingredient_id`, `price_ngn_per_kg`, `market`, `price_date`, `source_id`, `status`, `location`, `supplier`, `availability_kg`, `max_blend_share_pct`, `lot_id` |
| `formulation_experiments.csv` | measured property of a blend | `property`, `value`, `unit`, `share_<ingredient_id>`, `source_id`, `location` |
| `product_targets.csv` | sourced nutrition target for a product | `product`, `property`, `min`, `max`, `source_id`, `notes` |
| `archive/excluded_properties.csv` | removed row | as properties, plus `excluded_reason` |
| `raw/` | original external files | kept unchanged |
| `derived/state_values.csv` | value used for a state, ingredient and property (generated, not read by the app) | `state`, `ingredient_id`, `property`, `value`, `low`, `high`, `type`, `confidence`, `how_obtained`, `based_on_states`, `method` |

Ingredient ids: `cowpea`, `bambara_groundnut`, `defatted_groundnut`,
`african_yam_bean`, `soybean`. Any other id is rejected.
`defatted_groundnut` is groundnut after oil removal; never enter whole
(full-fat) groundnut values under it.

Product ids for `product_targets.csv`: `beef`, `chicken`, `chunks`,
`sausage`. The file is empty on purpose: no sourced product targets have been
found yet. Only add a row with a real `source_id`.

## Status: what is used

| `status` | Meaning | Used in calculations |
|---|---|---|
| `measured` | own laboratory result | yes |
| `sourced` | cited article or verified dataset record | yes |
| `estimated` | estimate by the data provider | no, shown as unverified |
| `assumed` | placeholder | no, shown as unverified |

Measured and sourced rows need a `source_id` that exists in `sources.csv`.
The earlier placeholder rows (`S000`) and the Sudanese samples (`S001`,
`S002`) were moved to `archive/excluded_properties.csv` with the reason.

## Location

Write `location` as `Nigeria / <State> / <site>`, e.g.
`Nigeria / Oyo / Ibadan (IITA)` or `Nigeria / FCT / Abuja main market`. The
state must be one of the 36 states or `FCT`; other countries are rejected.
For composition, use where the material was **grown**. For prices, use where
it was **bought**.

A state's own records are always used first. For a state without records,
the location model (`ai_engine/location_model.py`) estimates the value from
the other states: "checked" when its leave-one-state-out validation passes for
that ingredient and property, otherwise a "rough estimate" with a wider range.
A state without price lots gets a price from the price model
(`ai_engine/price_model.py`). Estimates are always marked and are never
written back into these files. Stock is never estimated.

To see what every state gets, run `python scripts/export_state_values.py`. It
writes `derived/state_values.csv` (one row per state, ingredient and property
or price, with value, 95% range, type, confidence and method). The file is
generated: do not edit it, and re-run the script after changing the data.

## Checks and calculated values

- **Carbohydrate by difference.** When a source does not report carbohydrate,
  it is calculated for each sample: 100 − protein − fat − ash − fibre, on a
  dry basis, and shown as "calculated". Fibre is subtracted, as in the
  sources that report it (S003).
- **Closure.** The dry-matter components of one sample may not exceed
  100 g/100 g. Values labelled `dry` that only add up to 100 together with
  moisture are reported as probably as-is. This found the S003 soybean table
  (labelled dry matter, but protein + fat + ash + fibre + carbohydrate +
  moisture = 100.0); those rows are now entered as `as_is`.

## Units and basis

- `unit` must match the property's unit (see `ai_engine/properties.py`).
  Water and oil holding values may be entered as `g/100 g`, `% dry sample`
  or `ml/g`; they are converted to g/g (oil density 0.92 g/mL) and the
  conversion is shown.
- `basis` is `dry` or `as_is` for composition. An as-is value is converted to
  dry basis only with moisture from the same source and sample; without it
  the row is skipped and reported by `validate-data`.
- A blank `value` means "not documented".

## Hack the Fork Nigeria 2026 dataset

`raw/hack-the-fork-nigeria-2026-plant-proteins.csv` is imported with
`python scripts/import_htf_dataset.py` (source `S008`; re-runnable). The
script's docstring lists how unlabelled units and basis were read
(proximate as-is, WAC/OAC g/100 g, emulsifying capacity %). Defatted soy and
defatted Bambara flour records are skipped (different ingredient forms).
Composition location is the production origin state; price lots use the
purchase state and market. For a state, the optimiser uses the lowest
`max_blend_share_pct` among its lots as the default limit and the reported
stock as an upper bound. For a state with an estimated price it uses the
lowest `max_blend_share_pct` across all verified lots and no stock limit.

## Blend experiments

Functional properties of a mix (water and oil holding, emulsion, gel) are
only estimated from `formulation_experiments.csv`: at least 8 measured
blends, 2 per model coefficient, every ingredient of the requested blend in
at least 3 of them, shares inside the measured range, leave-one-out
Q² ≥ 0.5. Example row (shares as fractions or percentages):

```csv
experiment_id,source_id,property,value,unit,method,location,share_cowpea,share_bambara_groundnut,share_defatted_groundnut,share_african_yam_bean,share_soybean,notes
E01,S001,water_absorption,1.85,g/g,"Centrifuge method",Nigeria / Oyo / Ibadan,50,0,0,0,50,
```

## Adding values from an article

1. Add the article to `sources.csv`.
2. Add one row per value to `ingredient_properties.csv` with that
   `source_id`, `status=sourced`, the Nigerian location and the sample label.
3. Run `flask --app run.py validate-data`.
