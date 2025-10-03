import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from joblib import dump, load
from tqdm import tqdm

# Caminhos dos arquivos
MODEL_PATH = "RF_Fraud_Model.pkl"
SCALER_PATH = "scaler.pkl"

# Se já existem, não precisa treinar de novo
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    print(f"[INFO] Modelo já existe em {MODEL_PATH}")
    print(f"[INFO] Scaler já existe em {SCALER_PATH}")
    print("[INFO] Se quiser treinar novamente, delete os arquivos e rode de novo.")
else:
    # Carregar dataset
    csv_path = "creditcard.csv"
    print(f"[INFO] Carregando dataset de {csv_path}...")
    df = pd.read_csv(csv_path)

    # Criar e aplicar scaler
    print("[INFO] Normalizando coluna 'Amount'...")
    scaler = StandardScaler()
    df['NormalizedAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df = df.drop(['Time', 'Amount'], axis=1)

    # Separar features e target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Split treino/teste
    print("[INFO] Separando dados em treino e teste (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Aplicar SMOTE apenas no treino
    print("[INFO] Aplicando SMOTE no conjunto de treino...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Treinar Random Forest
    print("[INFO] Treinando Random Forest com 500 árvores...")
    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_res, y_train_res)
    print("[INFO] Treinamento concluído.")

    # Avaliar modelo
    print("[INFO] Avaliando modelo no conjunto de teste...")
    y_pred = model.predict(X_test)
    print("\n=== Relatório de Classificação ===")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print("\n=== AUC-ROC Score ===")
    print(roc_auc)

    cm = confusion_matrix(y_test, y_pred)
    print("\n=== Matriz de Confusão ===")
    print(cm)

    # Salvar modelo e scaler
    dump(model, MODEL_PATH)
    dump(scaler, SCALER_PATH)
    print(f"[INFO] Modelo salvo em {MODEL_PATH}")
    print(f"[INFO] Scaler salvo em {SCALER_PATH}")
