# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io, os
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

MODEL_PATH = "RF_Fraud_Model.pkl"
SCALER_PATH = "scaler.pkl"

app = FastAPI(title="Detecção de Fraudes API")

# Permitir requisições de qualquer origem (útil para testes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carrega modelo e scaler (se existirem) na inicialização
model = None
scaler = None
if os.path.exists(MODEL_PATH):
    try:
        model = load(MODEL_PATH)
        print(f"[INFO] Modelo carregado: {MODEL_PATH}")
    except Exception as e:
        print("[WARN] Falha ao carregar modelo:", e)

if os.path.exists(SCALER_PATH):
    try:
        scaler = load(SCALER_PATH)
        print(f"[INFO] Scaler carregado: {SCALER_PATH}")
    except Exception as e:
        print("[WARN] Falha ao carregar scaler:", e)

@app.get("/")
def home():
    return {
        "message": "API de Detecção de Fraudes rodando",
        "model_loaded": bool(model),
        "scaler_loaded": bool(scaler),
    }

@app.post("/evaluate_csv")
async def evaluate_csv(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não encontrado no servidor.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Arquivo CSV inválido ou mal-formatado.")

    # Verificações mínimas
    if 'Amount' not in df.columns or 'Class' not in df.columns:
        raise HTTPException(status_code=400, detail="CSV precisa conter colunas 'Amount' e 'Class'.")

    # Normaliza 'Amount' usando o scaler salvo ou ajusta um novo (fallback)
    if scaler is not None:
        df['NormalizedAmount'] = scaler.transform(df['Amount'].values.reshape(-1, 1))
    else:
        tmp_scaler = StandardScaler()
        df['NormalizedAmount'] = tmp_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

    # Remover colunas desnecessárias (se existirem)
    drop_cols = []
    if 'Time' in df.columns:
        drop_cols.append('Time')
    drop_cols.append('Amount')
    df = df.drop(columns=drop_cols)

    X = df.drop(columns=['Class'])
    y = df['Class']

    # Previsões e métricas
    y_pred = model.predict(X)
    try:
        y_prob = model.predict_proba(X)[:, 1]
        auc = float(roc_auc_score(y, y_prob))
    except Exception:
        auc = None  # se o modelo não suportar predict_proba
    report = classification_report(y, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y, y_pred).tolist()

    return {
        "classification_report": report,
        "auc_roc": auc,
        "confusion_matrix": conf_matrix
    }
