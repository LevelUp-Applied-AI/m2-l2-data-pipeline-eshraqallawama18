"""
Lab 2 — Learner Test File

Write your own pytest tests here. You must implement at least 3 test functions:
  - test_load_data_returns_dataframe
  - test_clean_data_no_nulls
  - test_add_features_creates_revenue

The autograder will run your tests as part of the CI check.
"""

import pandas as pd
import numpy as np
import pytest
import pandas.testing as pdt
from pipeline import load_data, clean_data, add_features

# ─── Test 1 ───────────────────────────────────────────────────────────────────
def test_load_data_returns_dataframe():
    """load_data should return a DataFrame with expected columns and rows."""
    df = load_data('data/sales_records.csv')
    
    # Assert it's a DataFrame
    assert isinstance(df, pd.DataFrame)
    
    # Assert there are rows
    assert len(df) > 0
    
    # Assert expected columns exist
    expected_cols = ['date', 'store_id', 'product_category', 'quantity', 'unit_price', 'payment_method']
    for col in expected_cols:
        assert col in df.columns

# ─── Test 2 ───────────────────────────────────────────────────────────────────
def test_clean_data_no_nulls():
    """After clean_data, quantity and unit_price should have no NaN values."""
    df = load_data('data/sales_records.csv')
    cleaned = clean_data(df)
    
    # Assert no NaNs remain
    assert cleaned['quantity'].isna().sum() == 0
    assert cleaned['unit_price'].isna().sum() == 0

# ─── Test 3 ───────────────────────────────────────────────────────────────────
def test_add_features_creates_revenue():
    """add_features should add a 'revenue' column equal to quantity * unit_price."""
    df = load_data('data/sales_records.csv')
    cleaned = clean_data(df)
    enriched = add_features(cleaned)
    
    # Assert 'revenue' column exists
    assert 'revenue' in enriched.columns
    
    # Assert revenue values match quantity * unit_price
    expected_revenue = enriched['quantity'] * enriched['unit_price']
    pdt.assert_series_equal(enriched['revenue'], expected_revenue)