import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class CreditRiskModel:
    """
    Class to handle training, evaluation, and MLflow tracking for credit risk models.
    """

    def __init__(self, data_path, experiment_name="credit-risk-model"):
        self.data_path = data_path
        self.experiment_name = experiment_name
        self.X = None
        self.y = None

    def load_data(self):
        """
        Load processed dataset, drop ID/text columns, prepare features and target.
        """
        df = pd.read_csv(self.data_path)
        if "is_high_risk" not in df.columns:
            raise ValueError("Target column 'is_high_risk' not found in dataset.")
        
        columns_to_drop = [
            "is_high_risk", "CustomerId", "TransactionId", "BatchId", "AccountId",
            "SubscriptionId", "ProductId", "CountryCode", "CurrencyCode", "ProviderId"
        ]
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]

        self.X = df.drop(columns=columns_to_drop)
        self.y = df["is_high_risk"]

        print(f"✅ Features shape: {self.X.shape}, Target shape: {self.y.shape}")

    def evaluate(self, y_true, y_pred, y_proba):
        """
        Compute evaluation metrics.
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba)
        }

    def train_model(self, model_name, model, param_grid):
        """
        Train, evaluate, and log the model to MLflow.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, stratify=self.y, test_size=0.3, random_state=42
        )

        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=model_name):
            grid = GridSearchCV(model, param_grid, cv=3, scoring="f1", n_jobs=-1)
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]

            metrics = self.evaluate(y_test, y_pred, y_proba)
            print(f"\n{model_name} Metrics:\n", metrics)

            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_model, "model")
            mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", model_name)
