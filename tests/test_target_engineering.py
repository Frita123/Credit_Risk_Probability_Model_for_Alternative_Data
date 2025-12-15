# tests/test_target_engineering.py
import pandas as pd
import pytest
from src.target_engineering import (
    calculate_rfm,
    cluster_customers,
    assign_high_risk,
    merge_target
)

# -----------------------------
# Sample raw transaction data
# -----------------------------
@pytest.fixture
def sample_data():
    data = {
        "TransactionId": [1, 2, 3, 4, 5],
        "CustomerId": [101, 101, 102, 103, 103],
        "Amount": [100, 200, 150, 300, 400],
        "TransactionStartTime": pd.to_datetime([
            "2025-12-01", "2025-12-02", "2025-12-01", "2025-12-03", "2025-12-04"
        ])
    }
    return pd.DataFrame(data)

# -----------------------------
# Sample RFM Data
# -----------------------------
@pytest.fixture
def sample_rfm():
    data = {
        "CustomerId": [101, 102, 103],
        "recency": [1, 4, 2],
        "frequency": [2, 1, 2],
        "monetary": [300, 150, 700]
    }
    return pd.DataFrame(data)

# -----------------------------
# Test RFM calculation
# -----------------------------
def test_calculate_rfm(sample_data):
    rfm = calculate_rfm(sample_data, snapshot_date=pd.Timestamp("2025-12-05"))
    assert "recency" in rfm.columns
    assert "frequency" in rfm.columns
    assert "monetary" in rfm.columns
    assert rfm.shape[0] == sample_data['CustomerId'].nunique()

# -----------------------------
# Test clustering
# -----------------------------
def test_cluster_customers(sample_rfm):
    clustered, kmeans_model = cluster_customers(sample_rfm, n_clusters=2, random_state=42)
    assert "cluster" in clustered.columns
    assert clustered['cluster'].nunique() <= 2

# -----------------------------
# Test high-risk assignment
# -----------------------------
def test_assign_high_risk(sample_rfm):
    clustered, _ = cluster_customers(sample_rfm, n_clusters=2, random_state=42)
    labeled = assign_high_risk(clustered)
    assert "is_high_risk" in labeled.columns
    assert labeled['is_high_risk'].isin([0, 1]).all()

# -----------------------------
# Test merging target
# -----------------------------
def test_merge_target(sample_data, sample_rfm):
    clustered, _ = cluster_customers(sample_rfm, n_clusters=2, random_state=42)
    labeled = assign_high_risk(clustered)
    merged = merge_target(sample_data, labeled)
    assert "is_high_risk" in merged.columns
    assert merged.shape[0] == sample_data.shape[0]
