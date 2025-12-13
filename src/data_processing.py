import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data(path):
    """Load CSV data from a path."""
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print(f"File not found: {path}")
        return None

def preprocess_data(df):
    """Basic preprocessing: handle missing values, encode categorical features."""
    df = df.copy()
    # Example: fill missing values
    df.fillna(0, inplace=True)
    # Example: encode categorical variables
    cat_cols = df.select_dtypes(include='object').columns
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols)
    return df
