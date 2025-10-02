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

CHUNK_SIZE = 5000  # número de linhas processadas por vez

# ---------- Garantir modelo e scaler ----------
def ensure_model_and_scaler():
    hf_repo = os.getenv("HF_REPO")  # pega do Render
    if hf_repo is None:
        raise RuntimeError("Variável de ambiente HF_REPO não definida.")

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("[INFO] Modelo e scaler já existem localmente.")
        return

    try:
        print("[INFO] Baixando modelo e scaler do Hugging Face...")
        hf_hub_download(
            repo_id=hf_repo,
            filename="RF_Fraud_Model.pkl",
            repo_type="model",
            local_dir=".",
            local_dir_use_symlinks=False
        )
        hf_hub_download(
            repo_id=hf_repo,
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

# ---------- Endpoint predict (com chunks) ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None or scaler is None:
        return {"error": "Modelo ou scaler não carregados."}

    # Ler CSV enviado em chunks
    try:
        contents = await file.read()
        chunks = pd.read_csv(io.BytesIO(contents), chunksize=CHUNK_SIZE)
    except Exception as e:
        return {"error": f"Não foi possível ler o CSV: {e}"}

    all_predictions = []
    all_probabilities = []
    y_true_total = []

    for chunk in chunks:
        if 'Amount' not in chunk.columns:
            return {"error": "CSV deve conter coluna 'Amount'."}

        # Normalizar Amount
        chunk['NormalizedAmount'] = scaler.transform(chunk['Amount'].values.reshape(-1,1))
        chunk = chunk.drop(['Time', 'Amount'], axis=1, errors='ignore')

        # Separar X e y
        X_chunk = chunk.drop('Class', axis=1, errors='ignore')
        y_chunk = chunk['Class'] if 'Class' in chunk.columns else None

        # Predição
        y_pred_chunk = model.predict(X_chunk)
        y_prob_chunk = model.predict_proba(X_chunk)[:,1] if hasattr(model, "predict_proba") else None

        all_predictions.extend(y_pred_chunk.tolist())
        if y_prob_chunk is not None:
            all_probabilities.extend(y_prob_chunk.tolist())
        if y_chunk is not None:
            y_true_total.extend(y_chunk.tolist())

    # Calcular métricas se y_true_total existe
    metrics = {}
    if y_true_total:
        import numpy as np
        y_pred_array = np.array(all_predictions)
        y_true_array = np.array(y_true_total)
        y_prob_array = np.array(all_probabilities) if all_probabilities else None

        metrics['classification_report'] = classification_report(y_true_array, y_pred_array, output_dict=True)
        metrics['roc_auc_score'] = roc_auc_score(y_true_array, y_prob_array) if y_prob_array is not None else None
        cm = confusion_matrix(y_true_array, y_pred_array)
        metrics['confusion_matrix'] = cm.tolist()

    return {
        "predictions": all_predictions,
        "probabilities": all_probabilities if all_probabilities else None,
        "metrics": metrics
    }
