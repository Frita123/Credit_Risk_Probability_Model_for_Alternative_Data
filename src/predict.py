def predict(model, df):
    """Make predictions using trained model."""
    df = preprocess_data(df)
    return model.predict(df)
