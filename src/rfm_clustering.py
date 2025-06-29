import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class RFMClustering:
    """
    Class to calculate RFM metrics, perform KMeans clustering, 
    and assign high-risk labels to customers.
    """

    def __init__(self, n_clusters=3, random_state=42, snapshot_date=None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.snapshot_date = snapshot_date
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.high_risk_cluster = None

    def calculate_rfm(self, df):
        df = df.copy()
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

        if self.snapshot_date is None:
            self.snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)
        else:
            self.snapshot_date = pd.to_datetime(self.snapshot_date)

        rfm = df.groupby("CustomerId").agg({
            "TransactionStartTime": lambda x: (self.snapshot_date - x.max()).days,
            "TransactionId": "count",
            "Amount": "sum"
        }).rename(columns={
            "TransactionStartTime": "Recency",
            "TransactionId": "Frequency",
            "Amount": "Monetary"
        })

        return rfm.reset_index()

    def fit(self, rfm_df):
        """
        Fit the scaler and KMeans model to the RFM data.
        """
        rfm_scaled = self.scaler.fit_transform(rfm_df[["Recency", "Frequency", "Monetary"]])
        clusters = self.kmeans.fit_predict(rfm_scaled)
        rfm_df["Cluster"] = clusters

        self.high_risk_cluster = self._identify_high_risk_cluster(rfm_df)
        rfm_df["is_high_risk"] = (rfm_df["Cluster"] == self.high_risk_cluster).astype(int)
        return rfm_df

    def _identify_high_risk_cluster(self, rfm_df):
        """
        Determine which cluster represents high-risk customers.
        """
        cluster_summary = rfm_df.groupby("Cluster").agg({
            "Recency": "mean",
            "Frequency": "mean",
            "Monetary": "mean"
        }).reset_index()

        cluster_summary["RiskScore"] = (
            cluster_summary["Recency"] - cluster_summary["Frequency"] - cluster_summary["Monetary"]
        )

        high_risk_cluster = cluster_summary.sort_values("RiskScore", ascending=False).iloc[0]["Cluster"]
        return high_risk_cluster

    def assign_labels(self, df):
        """
        Full pipeline: compute RFM, fit model, assign high-risk labels.
        Returns merged DataFrame with is_high_risk column.
        """
        rfm = self.calculate_rfm(df)
        rfm = self.fit(rfm)
        merged_df = df.merge(rfm[["CustomerId", "is_high_risk"]], on="CustomerId", how="left")
        return merged_df
