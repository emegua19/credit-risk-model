import pandas as pd
from src.rfm_clustering import RFMClustering


def test_rfm_calculation():
    """
    Test basic RFM metric calculation logic.
    """
    test_data = pd.DataFrame({
        "CustomerId": [1, 1, 2],
        "TransactionId": ["t1", "t2", "t3"],
        "TransactionStartTime": ["2024-06-01", "2024-06-10", "2024-06-05"],
        "Amount": [100, 200, 300]
    })

    rfm_model = RFMClustering(snapshot_date="2024-06-15")
    rfm_df = rfm_model.calculate_rfm(test_data)

    assert rfm_df.shape[0] == 2, "Should compute RFM for each unique CustomerId"
    assert all(col in rfm_df.columns for col in ["Recency", "Frequency", "Monetary"]), "Missing expected RFM columns"


def test_full_pipeline_assigns_labels():
    """
    Ensure the full pipeline assigns is_high_risk column.
    """
    test_data = pd.DataFrame({
        "CustomerId": [1, 1, 2, 3, 3, 3],
        "TransactionId": ["t1", "t2", "t3", "t4", "t5", "t6"],
        "TransactionStartTime": [
            "2024-06-01", "2024-06-10", "2024-06-05",
            "2024-06-02", "2024-06-08", "2024-06-09"
        ],
        "Amount": [100, 200, 300, 50, 75, 25]
    })

    rfm_model = RFMClustering(n_clusters=2, snapshot_date="2024-06-15")
    df_with_labels = rfm_model.assign_labels(test_data)

    assert "is_high_risk" in df_with_labels.columns, "Missing is_high_risk column"
    assert df_with_labels["is_high_risk"].isin([0, 1]).all(), "is_high_risk must be binary 0/1"

