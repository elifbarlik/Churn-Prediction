from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler

model = joblib.load('model.pkl')
scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("selected_features.pkl")

app = FastAPI()

class CustomerData(BaseModel):
    SeniorCitizen: int
    Partner: int
    Dependents: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float

    
@app.post("/predict")
def predict(data: CustomerData):
    df=pd.DataFrame([data.dict()])
    df_selected = df[selected_features]
    df_scaled = scaler.transform(df_selected)
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]
    return {'churn': int(prediction), 'probability':round(probability,2)}
