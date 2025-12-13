from fastapi import FastAPI
from pydantic import BaseModel
from train import train_model
import pandas as pd

app = FastAPI()

class Transaction(BaseModel):
    Amount: float
    Value: float
    # add other fields as needed

@app.post("/predict")
def predict_endpoint(transaction: Transaction):
    data = pd.DataFrame([transaction.dict()])
    # assume model is preloaded
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}
