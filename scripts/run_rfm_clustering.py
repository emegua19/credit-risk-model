import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rfm_clustering import RFMClustering

# Define paths
RAW_DATA_PATH = "data/raw/data.csv"  # Update if filename differs
OUTPUT_PATH = "data/processed/transactions_with_labels.csv"

def main():
    print(" Starting RFM clustering and proxy label generation...")

    # Load raw data
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data file not found at {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    print(f" Loaded raw data with shape: {df.shape}")

    # RFM Clustering and high-risk label assignment
    rfm_model = RFMClustering(n_clusters=3, random_state=42)
    df_with_labels = rfm_model.assign_labels(df)

    print(f" Proxy labels assigned. Sample data:\n")
    print(df_with_labels.head(10))

    # Save result
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_with_labels.to_csv(OUTPUT_PATH, index=False)
    print(f" Labeled data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
