from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from joblib import load

# Carregar modelo treinado
model = load("RF_Fraud_Model.pkl")

# Criar aplicação FastAPI
app = FastAPI(title="Detecção de Fraudes API", version="1.0")

# Definir formato de entrada
class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    NormalizedAmount: float

# Endpoint raiz
@app.get("/")
def home():
    return {"message": "API de Detecção de Fraudes funcionando!"}

# Endpoint de previsão
@app.post("/predict")
def predict(transaction: Transaction):
    data = pd.DataFrame([transaction.dict()])
    prediction = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]
    return {
        "fraude": bool(prediction),
        "probabilidade": float(prob)
    }
