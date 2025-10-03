import requests
import seaborn as sns
import matplotlib.pyplot as plt
import json
import time

# URL do endpoint predict
url = "https://fraud-detection-api-7ehe.onrender.com/predict"

# Caminho para seu CSV local
csv_path = "creditcard.csv"

# Medir tempo de execução
start_time = time.time()

# Enviar arquivo via POST
with open(csv_path, "rb") as f:
    files = {"file": (csv_path, f, "text/csv")}
    response = requests.post(url, files=files, timeout=300)

end_time = time.time()
elapsed_time = end_time - start_time

# Verificar resposta
if response.status_code == 200:
    data = response.json()
    metrics = data.get("metrics", {})

    print(f"\nTempo total de processamento: {elapsed_time:.2f} segundos\n")

    if not metrics:
        print("Nenhuma métrica retornada.")
    else:
        # Relatório de classificação
        print("=== Relatório de Classificação ===")
        print(json.dumps(metrics.get("classification_report", {}), indent=4))

        # AUC-ROC
        print("\n=== AUC-ROC Score ===")
        print(metrics.get("roc_auc_score"))

        # Matriz de confusão
        print("\n=== Matriz de Confusão ===")
        cm = metrics.get("confusion_matrix")
        if cm:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel("Previsto")
            plt.ylabel("Real")
            plt.title("Matriz de Confusão")
            plt.show()

else:
    print(f"Erro {response.status_code}: {response.text}")
