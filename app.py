# app.py (trecho relevante)
import os
from joblib import load
import requests
from huggingface_hub import hf_hub_download
import time

MODEL_PATH = "RF_Fraud_Model.pkl"
SCALER_PATH = "scaler.pkl"

HF_REPO = os.getenv("HF_REPO")      # ex: "felovers/fraud-model"
HF_TOKEN = os.getenv("HF_TOKEN")    # token (se repo privado)
MODEL_URL = os.getenv("MODEL_URL")  # alternativa: link direto (Google Drive/generic)

def download_file_from_url(url, dest_path):
    # download streaming
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

def ensure_model_and_scaler():
    # se já existem, não faz nada
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        return

    # 1) Tenta Hugging Face (se variável HF_REPO fornecida)
    if HF_REPO:
        try:
            print("[INFO] Baixando modelo do Hugging Face Hub...")
            # retorna caminho local do arquivo baixado
            model_local = hf_hub_download(repo_id=HF_REPO, filename="RF_Fraud_Model.pkl", repo_type="model", token=HF_TOKEN)
            scaler_local = hf_hub_download(repo_id=HF_REPO, filename="scaler.pkl", repo_type="model", token=HF_TOKEN)
            # mover/renomear para nomes esperados
            os.replace(model_local, MODEL_PATH)
            os.replace(scaler_local, SCALER_PATH)
            print("[INFO] Modelo e scaler baixados do HF com sucesso.")
            return
        except Exception as e:
            print("[WARN] Falha ao baixar do HF:", e)

    # 2) Tenta URL direta (MODEL_URL)
    if MODEL_URL:
        try:
            print("[INFO] Baixando modelo de MODEL_URL...")
            # você pode ter dois links, um para o modelo e outro para o scaler; aqui assumimos MODEL_URL e SCALER_URL
            model_url = MODEL_URL
            scaler_url = os.getenv("SCALER_URL")
            if not scaler_url:
                raise ValueError("SCALER_URL não definido")
            download_file_from_url(model_url, MODEL_PATH)
            download_file_from_url(scaler_url, SCALER_PATH)
            print("[INFO] Modelo e scaler baixados via URL com sucesso.")
            return
        except Exception as e:
            print("[WARN] Falha ao baixar via MODEL_URL:", e)

    raise RuntimeError("Modelo/scaler não encontrado localmente e não foi possível baixar automaticamente.")

# Chame ensure_model_and_scaler() antes de carregar o modelo na inicialização do app
try:
    ensure_model_and_scaler()
    model = load(MODEL_PATH)
    scaler = load(SCALER_PATH)
    print("[INFO] Modelo e scaler carregados.")
except Exception as e:
    print("[ERROR] Não foi possível carregar modelo/scaler:", e)
    model = None
    scaler = None
