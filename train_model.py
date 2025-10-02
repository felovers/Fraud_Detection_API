import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from joblib import dump, load
from tqdm import tqdm

# Caminhos
MODEL_PATH = "RF_Fraud_Model.pkl"
SCALER_PATH = "scaler.pkl"

# Se já existem, não precisa treinar de novo
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    print(f"[INFO] Modelo já existe em {MODEL_PATH}")
    print(f"[INFO] Scaler já existe em {SCALER_PATH}")
    print("[INFO] Se quiser treinar novamente, delete os arquivos e rode de novo.")
else:
    # Dataset
    csv_path = "creditcard.csv"
    df = pd.read_csv(csv_path)

    # Criar e aplicar scaler
    scaler = StandardScaler()
    df['NormalizedAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df = df.drop(['Time', 'Amount'], axis=1)

    # Separar features e target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Balanceamento com SMOTE
    print("[INFO] Aplicando SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Modelo RandomForest
    print("[INFO] Treinando modelo...")
    model = RandomForestClassifier(
        n_estimators=0,
        random_state=42,
        n_jobs=-1,
        warm_start=True
    )

    # Treino incremental com barra de progresso
    total_trees = 500
    batch_size = 50
    for i in tqdm(range(0, total_trees, batch_size), desc="Treinando árvores"):
        model.n_estimators = i + batch_size
        model.fit(X_train_res, y_train_res)

    # Avaliação
    y_pred = model.predict(X_test)
    print("\n[INFO] Relatório de Classificação:")
    print(classification_report(y_test, y_pred))
    print("AUC-ROC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

    # Salvar modelo e scaler
    dump(model, MODEL_PATH)
    dump(scaler, SCALER_PATH)
    print(f"[INFO] Modelo salvo em {MODEL_PATH}")
    print(f"[INFO] Scaler salvo em {SCALER_PATH}")
