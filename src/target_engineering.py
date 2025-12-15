# src/target_engineering.py
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from src.data_processing import load_data, save_processed_data

# -----------------------------
# Setup logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------
# Calculate RFM Metrics
# -----------------------------
def calculate_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerId').agg(
        recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
        frequency=('TransactionId', 'count'),
        monetary=('Amount', 'sum')
    ).reset_index()

    logging.info("RFM metrics calculated.")
    return rfm

# -----------------------------
# Cluster Customers
# -----------------------------
def cluster_customers(rfm_df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['recency', 'frequency', 'monetary']])
    logging.info("RFM features scaled.")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_df['cluster'] = kmeans.fit_predict(rfm_scaled)
    logging.info("K-Means clustering completed.")

    return rfm_df, kmeans

# -----------------------------
# Assign High-Risk Label
# -----------------------------
def assign_high_risk(rfm_df: pd.DataFrame) -> pd.DataFrame:
    cluster_summary = rfm_df.groupby('cluster')[['recency', 'frequency', 'monetary']].mean()
    high_risk_cluster = cluster_summary.sort_values(['frequency', 'monetary'], ascending=[True, True]).index[0]

    rfm_df['is_high_risk'] = (rfm_df['cluster'] == high_risk_cluster).astype(int)
    logging.info(f"High-risk cluster identified: {high_risk_cluster}")
    return rfm_df

# -----------------------------
# Merge Target with Processed Data
# -----------------------------
def merge_target(df: pd.DataFrame, rfm_df: pd.DataFrame) -> pd.DataFrame:
    target_df = rfm_df[['CustomerId', 'is_high_risk']]
    merged_df = df.merge(target_df, on='CustomerId', how='left')
    logging.info("Target variable merged with main dataset.")
    return merged_df

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    raw_data_path = "data/raw/transactions.csv"
    processed_data_path = "data/processed/features.npy"
    processed_with_target_path = "data/processed/features_with_target.npy"

    # Load raw data
    df = load_data(raw_data_path)

    # Calculate RFM
    rfm_df = calculate_rfm(df)

    # Cluster and assign high-risk target
    rfm_df, kmeans_model = cluster_customers(rfm_df)
    rfm_df = assign_high_risk(rfm_df)

    # Merge target into processed features
    from src.data_processing import prepare_features  # import here to avoid circular import
    X, preprocessor, feature_names = prepare_features(df)
    
    df_with_target = merge_target(df, rfm_df)

    # Save processed data
    save_processed_data(X, processed_data_path)
    df_with_target.to_csv(processed_with_target_path.replace('.npy', '.csv'), index=False)
    logging.info(f"Processed features with target saved to {processed_with_target_path}")

    print("Target variable engineering completed successfully.")
