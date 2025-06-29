# src/data_explorer.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os


class DataExplorer:
    def __init__(self, filepath: str, output_dir: str = "../outputs/plots"):
        """
        Initialize the DataExplorer.
        :param filepath: Path to CSV file
        :param output_dir: Path to save visualizations
        """
        self.filepath = filepath
        self.output_dir = output_dir
        self.df = None
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        try:
            self.df = pd.read_csv(self.filepath)
            print(f"Data loaded successfully with shape: {self.df.shape}")
        except Exception as e:
            print(f"Error loading file: {e}")

    def overview(self):
        print("\n--- DATA OVERVIEW ---")
        print(f"Shape: {self.df.shape}")
        print("\nData Types:\n", self.df.dtypes)
        print("\nMissing Values:\n", self.df.isnull().sum())

    def summary_statistics(self):
        print("\n--- SUMMARY STATISTICS ---")
        print(self.df.describe())

    def missing_values(self, threshold=0):
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            "Missing Values": missing,
            "Percent (%)": missing_percent
        }).sort_values(by="Percent (%)", ascending=False)
        print("\n--- MISSING VALUES ---")
        print(missing_df[missing_df["Missing Values"] > threshold])
        return missing_df

    def plot_numerical_distributions(self, num_cols):
        for col in num_cols:
            plt.figure(figsize=(6, 4))
            sns.histplot(self.df[col].dropna(), bins=40, kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            filename = os.path.join(self.output_dir, f"distribution_{col.lower()}.png")
            plt.savefig(filename)
            print(f"Saved: {filename}")
            plt.close()

    def plot_categorical_counts(self, cat_cols, top_n=10):
        for col in cat_cols:
            plt.figure(figsize=(7, 4))
            self.df[col].value_counts().head(top_n).plot(kind='bar')
            plt.title(f"Top {top_n} Categories in {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.tight_layout()
            filename = os.path.join(self.output_dir, f"category_counts_{col.lower()}.png")
            plt.savefig(filename)
            print(f"Saved: {filename}")
            plt.close()

    def correlation_matrix(self):
        corr = self.df.corr(numeric_only=True)
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.tight_layout()
        filename = os.path.join(self.output_dir, "correlation_matrix.png")
        plt.savefig(filename)
        print(f"Saved: {filename}")
        plt.close()

    def box_plots(self, num_cols):
        for col in num_cols:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=self.df[col].dropna())
            plt.title(f"Boxplot of {col}")
            plt.tight_layout()
            filename = os.path.join(self.output_dir, f"boxplot_{col.lower()}.png")
            plt.savefig(filename)
            print(f"Saved: {filename}")
            plt.close()

    def add_time_columns(self, time_col="TransactionStartTime"):
        self.df[time_col] = pd.to_datetime(self.df[time_col], errors="coerce")
        self.df["Year"] = self.df[time_col].dt.year
        self.df["Month"] = self.df[time_col].dt.month
        self.df["Day"] = self.df[time_col].dt.day
        self.df["Hour"] = self.df[time_col].dt.hour
        print(f"Time-based columns added from {time_col}.")

    def plot_transactions_by_hour(self):
        if "Hour" not in self.df.columns:
            self.add_time_columns()
        plt.figure(figsize=(8, 4))
        self.df["Hour"].value_counts().sort_index().plot(kind="bar")
        plt.title("Transactions by Hour of Day")
        plt.xlabel("Hour")
        plt.ylabel("Count")
        plt.tight_layout()
        filename = os.path.join(self.output_dir, "transactions_by_hour.png")
        plt.savefig(filename)
        print(f"Saved: {filename}")
        plt.close()
