import os
import io
from fastapi import FastAPI, File, UploadFile
from joblib import load
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from huggingface_hub import hf_hub_download

app = FastAPI(title="Fraud Detection API")

MODEL_PATH = "RF_Fraud_Model.pkl"
SCALER_PATH = "scaler.pkl"

HF_REPO = "felovers/fraud-model"  # Repositório público do Hugging Face

# ---------- Função para garantir modelo e scaler ----------
def ensure_model_and_scaler():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("[INFO] Modelo e scaler já existem localmente.")
        return

    try:
        print("[INFO] Baixando modelo e scaler do Hugging Face...")
        # Força download direto na raiz do projeto
        hf_hub_download(
            repo_id=HF_REPO,
            filename="RF_Fraud_Model.pkl",
            repo_type="model",
            local_dir=".",
            local_dir_use_symlinks=False
        )
        hf_hub_download(
            repo_id=HF_REPO,
            filename="scaler.pkl",
            repo_type="model",
            local_dir=".",
            local_dir_use_symlinks=False
        )
        print("[INFO] Modelo e scaler baixados com sucesso.")
    except Exception as e:
        raise RuntimeError(f"Erro ao baixar modelo/scaler: {e}")

# ---------- Inicialização ----------
try:
    ensure_model_and_scaler()
    model = load(MODEL_PATH)
    scaler = load(SCALER_PATH)
    print("[INFO] Modelo e scaler carregados com sucesso.")
except Exception as e:
    print("[ERROR] Não foi possível carregar modelo/scaler:", e)
    model = None
    scaler = None

# ---------- Endpoint raiz ----------
@app.get("/")
def root():
    return {"status": "API Online"}

# ---------- Endpoint para predição ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None or scaler is None:
        return {"error": "Modelo ou scaler não carregados."}

    # Ler CSV enviado
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        return {"error": f"Não foi possível ler o CSV: {e}"}

    if 'Amount' not in df.columns:
        return {"error": "CSV deve conter coluna 'Amount'."}

    # Normalizar Amount com scaler carregado
    df['NormalizedAmount'] = scaler.transform(df['Amount'].values.reshape(-1,1))
    df = df.drop(['Time', 'Amount'], axis=1, errors='ignore')

    # Separar X e y (se existir coluna Class)
    X = df.drop('Class', axis=1, errors='ignore')
    y_true = df['Class'] if 'Class' in df.columns else None

    # Predição
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:,1] if hasattr(model, "predict_proba") else None

    # Métricas se y_true existe
    metrics = {}
    if y_true is not None:
        metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        metrics['roc_auc_score'] = roc_auc_score(y_true, y_prob) if y_prob is not None else None
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()  # lista para JSON

    # Retornar predições e métricas
    return {
        "predictions": y_pred.tolist(),
        "probabilities": y_prob.tolist() if y_prob is not None else None,
        "metrics": metrics
    }
