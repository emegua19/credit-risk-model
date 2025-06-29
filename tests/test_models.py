import numpy as np
from src.models import CreditRiskModel

def test_evaluation_metrics():
    """
    Test that evaluation metrics return expected keys and reasonable values.
    """
    model = CreditRiskModel(data_path="dummy.csv")  # Path not used here

    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    y_proba = np.array([0.1, 0.8, 0.2, 0.4])

    metrics = model.evaluate(y_true, y_pred, y_proba)

    assert isinstance(metrics, dict), "Metrics should be returned as a dictionary."
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        assert key in metrics, f"Missing metric: {key}"
        assert 0.0 <= metrics[key] <= 1.0, f"Metric {key} out of valid range."
