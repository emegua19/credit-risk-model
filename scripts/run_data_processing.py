import os
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import build_preprocessing_pipeline

# Define paths
RAW_DATA_PATH = "data/raw/data.csv"  # Update filename if different
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"

def main():
    print(" Starting data processing...")

    # Load raw data
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data file not found at {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f" Loaded raw data with shape: {df.shape}")

    # Build and apply the pipeline
    pipeline = build_preprocessing_pipeline()
    processed_array = pipeline.fit_transform(df)
    print(f" Data processed. Final shape: {processed_array.shape}")

    # Get feature names after transformation
    # Extract column names from the ColumnTransformer
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    categorical_cols = ohe.get_feature_names_out(["ProductCategory", "ChannelId", "PricingStrategy", "IsNegativeAmount"])

    final_columns = [
        "Amount", "Value", "TotalAmount", "MeanAmount", "StdAmount", "TransactionCount"
    ] + list(categorical_cols)

    # Convert to DataFrame
    processed_df = pd.DataFrame(processed_array, columns=final_columns)

    # Ensure processed folder exists
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

    # Save as CSV
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f" Processed data saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    main()
