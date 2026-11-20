"""Ingredient data is not stored in the database.

The shared ingredient dataset lives in the CSV files under data/ and is loaded
by ai_engine.dataset.Dataset, so the team can update values from articles
without migrations. See data/README.md.
"""
