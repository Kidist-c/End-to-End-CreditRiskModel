import pytest
import pandas as pd
from src.data_preprocessing import DateTimeFeature,customerAggregation

# Sample data
sample_data = pd.DataFrame({
    "CustomerId": [1, 1, 2],
    "Amount": [100, 200, 150],
    "TransactionStartTime": ["2025-12-01 10:00:00", "2025-12-02 12:00:00", "2025-12-01 15:00:00"]
})

#----------------------------------------------------------
  #Test DateTime
#-----------------------------------------------------------
def test_datetime_feature_columns():
    dt_transformer=DateTimeFeature("TransactionStartTime")
    transformed=dt_transformer.fit(sample_data)
     # Check that new columns are created
    expected_cols = ["transaction_hour", "transaction_day", "transaction_month", "transaction_year"]
    for col in expected_cols:
        assert col in transformed.columns, f"{col} missing in the DataFrame"
# -----------------------------
# Test 2: customerAggregation adds aggregated columns
# -----------------------------
def test_customer_aggregation_columns():
    transformer = customerAggregation("CustomerId", "Amount")
    transformed = transformer.fit_transform(sample_data)
    # Aggregated columns
    agg_cols = ["total_amount", "avg_amount", "transaction_count", "std_amount"]
    for col in agg_cols:
        assert col in transformed.columns, f"{col} missing in transformed DataFrame"


