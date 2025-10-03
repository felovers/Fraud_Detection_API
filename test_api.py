import pandas as pd
import requests
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# URL da sua API
API_URL = "https://fraud-detection-api-7ehe.onrender.com/predict"  # ajuste se necessário

# Caminho para o CSV completo
CSV_PATH = "creditcard.csv"

# Parâmetros
CHUNK_SIZE = 10000  # enviar em blocos para evitar estouro de memória

# Carregar CSV completo
print("[INFO] Carregando CSV completo...")
df = pd.read_csv(CSV_PATH)

# Separar X e y
X = df.drop("Class", axis=1)
y = df["Class"]

# Criar split treino/teste igual ao treino original
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[INFO] X_test shape: {X_test.shape}")

# Função para enviar chunks para API e receber predições
def predict_api_chunks(X_test, chunk_size=10000):
    predictions = []
    start_time = time.time()
    for i in tqdm(range(0, len(X_test), chunk_size), desc="Enviando chunks"):
        chunk = X_test.iloc[i:i+chunk_size]
        # Converter para CSV temporário em memória
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

# Obter predições da API
print("[INFO] Solicitando predições da API...")
y_pred = predict_api_chunks(X_test, CHUNK_SIZE)

# Garantir que tamanho bate com y_test
if len(y_pred) != len(y_test):
    print("[WARN] Número de predições diferente do esperado!")

# Avaliar métricas
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
