import pandas as pd
from src.data_processing import build_preprocessing_pipeline

def test_pipeline_runs():
    """
    Basic smoke test to ensure pipeline runs without error.
    """
    # Minimal test data
    test_data = pd.DataFrame({
        "CustomerId": [1, 1, 2],
        "Amount": [100, -50, 200],
        "Value": [100, 50, 200],
        "TransactionStartTime": ["2024-06-01 12:00:00", "2024-06-02 14:00:00", "2024-06-03 10:00:00"],
        "ProductCategory": ["Electronics", "Electronics", "Clothing"],
        "ChannelId": ["Web", "Android", "Web"],
        "PricingStrategy": ["Standard", "Discount", "Standard"]
    })

    pipeline = build_preprocessing_pipeline()
    output = pipeline.fit_transform(test_data)

    # Assert output shape
    assert output.shape[0] == 3, "Pipeline should return same number of rows as input"
    assert output.shape[1] > 0, "Pipeline should produce features"

def test_negative_amount_flag():
    """
    Ensure negative amount flag is correctly set.
    """
    test_data = pd.DataFrame({
        "CustomerId": [1],
        "Amount": [-20],
        "Value": [20],
        "TransactionStartTime": ["2024-06-01 12:00:00"],
        "ProductCategory": ["Electronics"],
        "ChannelId": ["Web"],
        "PricingStrategy": ["Standard"]
    })

    pipeline = build_preprocessing_pipeline()
    output = pipeline.fit_transform(test_data)

    # Extract last categorical feature, which is the negative amount flag after one-hot encoding
    assert output.shape[0] == 1
