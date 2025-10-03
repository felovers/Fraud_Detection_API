import os
import sys
import pandas as pd
import requests
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ======================
# Configurações
# ======================
API_URL = "https://fraud-detection-api-7ehe.onrender.com/predict"
CHUNK_SIZE = 10000

# Caminho correto do CSV, mesmo embutido no exe
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS  # pasta temporária criada pelo PyInstaller
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(BASE_DIR, "creditcard.csv")
print(f"[INFO] Caminho do CSV: {CSV_PATH}")

# ======================
# Carregar CSV
# ======================
print("[INFO] Carregando CSV completo...")
df = pd.read_csv(CSV_PATH)

# Separar features e target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split teste igual ao treino do modelo (80/20)
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[INFO] X_test shape: {X_test.shape}")

# ======================
# Função de predição via API
# ======================
def predict_api_chunks(X_test, chunk_size=10000):
    predictions = []
    start_time = time.time()
    
    for i in tqdm(range(0, len(X_test), chunk_size), desc="Enviando chunks"):
        chunk = X_test.iloc[i:i+chunk_size]
        csv_data = chunk.to_csv(index=False)
        
        response = requests.post(API_URL, files={"file": ("chunk.csv", csv_data)})
        
        if response.status_code == 200:
            preds_chunk = response.json().get("predictions", [])
            predictions.extend(preds_chunk)
        else:
            print(f"[ERROR] Chunk {i//chunk_size} falhou. Status code: {response.status_code}")
    
    end_time = time.time()
    print(f"[INFO] Tempo total de predição: {end_time - start_time:.2f} segundos")
    return predictions

# ======================
# Rodar predições
# ======================
print("[INFO] Solicitando predições da API...")
y_pred = predict_api_chunks(X_test, CHUNK_SIZE)

# ======================
# Avaliar métricas
# ======================
if len(y_pred) != len(y_test):
    print("[WARN] Número de predições diferente do esperado!")

print("\n=== Relatório de Classificação ===")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred)
print("=== AUC-ROC Score ===")
print(roc_auc)

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.show()

input("[INFO] Pressione Enter para encerrar o programa...")
