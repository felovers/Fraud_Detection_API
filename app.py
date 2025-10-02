import os
from fastapi import FastAPI
from joblib import load
from huggingface_hub import hf_hub_download

import requests

app = FastAPI()

MODEL_PATH = "RF_Fraud_Model.pkl"
SCALER_PATH = "scaler.pkl"

HF_REPO = os.getenv("felovers/fraud-model")  

def ensure_model_and_scaler():
    # Se já existem, não faz nada
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("[INFO] Modelo e scaler já existem localmente.")
        return

    if HF_REPO:
        try:
            print("[INFO] Baixando modelo do Hugging Face Hub...")
            model_local = hf_hub_download(
                repo_id=HF_REPO,
                filename="RF_Fraud_Model.pkl",
                repo_type="model"
            )
            scaler_local = hf_hub_download(
                repo_id=HF_REPO,
                filename="scaler.pkl",
                repo_type="model"
            )
            # mover para nomes esperados
            os.replace(model_local, MODEL_PATH)
            os.replace(scaler_local, SCALER_PATH)
            print("[INFO] Modelo e scaler baixados do HF com sucesso.")
        except Exception as e:
            print("[WARN] Falha ao baixar do HF:", e)
            raise RuntimeError("Não foi possível baixar modelo/scaler do Hugging Face.")

try:
    ensure_model_and_scaler()
    model = load(MODEL_PATH)
    scaler = load(SCALER_PATH)
    print("[INFO] Modelo e scaler carregados com sucesso.")
except Exception as e:
    print("[ERROR] Não foi possível carregar modelo/scaler:", e)
    model = None
    scaler = None
