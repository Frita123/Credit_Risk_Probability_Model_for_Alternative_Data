import pytest
from src.data_processing import preprocess_data
import pandas as pd

def test_preprocess_data():
    df = pd.DataFrame({"col1": [1, 2, None], "cat": ["a", "b", "c"]})
    processed = preprocess_data(df)
    assert processed.isnull().sum().sum() == 0
