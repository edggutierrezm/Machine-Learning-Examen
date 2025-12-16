import pandas as pd
import lightgbm as lgb # Excelente para problemas de Kaggle y datos tabulares
from joblib import dump
from sklearn.metrics import roc_auc_score

def run_training():
    # Cargar datos preprocesados
    X_train = pd.read_csv('../artifacts/X_train.csv')
    y_train = pd.read_csv('../artifacts/y_train.csv')['TARGET']

    # Definir el modelo y sus hiperparámetros (ej. para desbalanceo)
    lgb_clf = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=10000,
        learning_rate=0.01,
        # Estrategia para datos desbalanceados [cite: 33]
        class_weight='balanced', 
        n_jobs=-1,
        random_state=42
    )

    # Entrenamiento del modelo
    lgb_clf.fit(X_train, y_train)

    # 💾 Guardar el modelo campeón en artifacts [cite: 19]
    dump(lgb_clf, '../artifacts/champion_model.pkl')
    print("Modelo entrenado y guardado en artifacts/champion_model.pkl")

if __name__ == "__main__":
    run_training()