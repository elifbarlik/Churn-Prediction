from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
import os
from typing import Dict, Any

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(model_path)

scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
scaler = joblib.load(scaler_path)

selected_path = os.path.join(os.path.dirname(__file__), "selected_features.pkl")
selected_features = joblib.load(selected_path)


app = FastAPI()

class CustomerData(BaseModel):
    data: Dict[str, Any]

    
@app.post("/predict")
def predict(data: CustomerData):
    df=pd.DataFrame([data.data])
    df_selected = df[selected_features]
    df_scaled = scaler.transform(df_selected)
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]
    return {'churn': int(prediction), 'probability':round(probability,2)}
