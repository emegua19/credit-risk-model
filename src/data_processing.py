import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class AggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Create aggregate transaction features per customer.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Total, mean, std of transaction amount per customer
        agg_df = X.groupby("CustomerId").agg({
            "Amount": ["sum", "mean", "std", "count"]
        })
        agg_df.columns = ["TotalAmount", "MeanAmount", "StdAmount", "TransactionCount"]
        agg_df.reset_index(inplace=True)

        # Merge back to original
        X = X.merge(agg_df, on="CustomerId", how="left")
        
        return X


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """
    Extract transaction date and time features.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["TransactionStartTime"] = pd.to_datetime(X["TransactionStartTime"])

        X["TransactionHour"] = X["TransactionStartTime"].dt.hour
        X["TransactionDay"] = X["TransactionStartTime"].dt.day
        X["TransactionMonth"] = X["TransactionStartTime"].dt.month
        X["TransactionYear"] = X["TransactionStartTime"].dt.year

        X.drop(columns=["TransactionStartTime"], inplace=True)
        return X


class NegativeAmountFlag(BaseEstimator, TransformerMixin):
    """
    Binary flag for transactions with negative amounts.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["IsNegativeAmount"] = (X["Amount"] < 0).astype(int)
        return X


def build_preprocessing_pipeline():
    """
    Builds and returns the full preprocessing pipeline.
    """

    numeric_features = ["Amount", "Value", "TotalAmount", "MeanAmount", "StdAmount", "TransactionCount"]
    categorical_features = ["ProductCategory", "ChannelId", "PricingStrategy", "IsNegativeAmount"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    full_pipeline = Pipeline(steps=[
        ("aggregate", AggregateFeatures()),
        ("datetime", DateTimeFeatures()),
        ("neg_flag", NegativeAmountFlag()),
        ("preprocessor", preprocessor)
    ])

    return full_pipeline
