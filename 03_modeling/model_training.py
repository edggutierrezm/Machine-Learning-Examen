# Import dependencias
import pandas as pd
import lightgbm as lgb
from joblib import dump
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import os
import numpy as np
import gc


# CONFIGURACIÓN DE RUTAS
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), '../artifacts')
TRAIN_FILE = os.path.join(ARTIFACTS_PATH, 'X_final_train.csv')

def load_processed_data():
    """Carga los datos preprocesados de entrenamiento."""
    try:
        df_train = pd.read_csv(TRAIN_FILE)
        
        # Separar TARGET
        X = df_train.drop(columns=['TARGET'])
        y = df_train['TARGET'].astype(int)
        
        print(f"Datos de entrenamiento cargados. Shape: {X.shape}")
        return X, y
    except FileNotFoundError:
        print(f"ERROR: Archivo de entrenamiento no encontrado en {TRAIN_FILE}. Ejecuta la fase 02_preprocessing primero.")
        return None, None
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        return None, None


# FUNCIÓN DE LIMPIEZA DE NOMBRES DE COLUMNA (AÑADIDA)
def clean_feature_names(df):
    """Limpia los nombres de las columnas para eliminar caracteres no soportados por LightGBM."""
    new_cols = []
    for col in df.columns:
        # Reemplaza corchetes, comas, dos puntos, <, > y otros caracteres especiales por guiones bajos
        cleaned_col = col.replace('[', '_').replace(']', '_').replace('<', '_').replace(':', '_').replace(',', '_').replace('>', '_')
        # LightGBM no tolera espacios en blanco
        cleaned_col = cleaned_col.replace(' ', '_')
        new_cols.append(cleaned_col)
    df.columns = new_cols
    return df
# ==========================================================


def run_model_training():
    X, y = load_processed_data()

    if X is None:
        return

    # APLICAR LIMPIEZA: Limpiar los nombres de las columnas antes de pasarlos a LightGBM
    X = clean_feature_names(X) 

    # Parámetros del modelo LightGBM (Optimizados para AUC y desbalance)
    lgbm_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_estimators': 10000, # Número alto, se detendrá temprano con early_stopping
        'learning_rate': 0.01,
        'num_leaves': 20, 
        'max_depth': 3,
        'seed': 42,
        'n_jobs': -1, # Usa todos los núcleos disponibles
        'reg_alpha': 0.1,  # Regularización L1
        'reg_lambda': 0.1, # Regularización L2
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        # Manejo del desbalance: Usamos 'scale_pos_weight' para ponderar la clase minoritaria (TARGET=1)
        'scale_pos_weight': sum(y == 0) / sum(y == 1), 
    }

    # Inicializar el modelo
    lgb_clf = lgb.LGBMClassifier(**lgbm_params)

    # Implementación de Cross-Validation Estratificada
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(X.shape[0]) # Almacena predicciones OOF (Out-Of-Fold)
    
    print("\n--- Iniciando Entrenamiento con 5-Fold Cross-Validation ---")
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        # Entrenar el modelo con Early Stopping
        # Se utiliza para demostrar que el modelo va avanzando se utilizo en 500 o 50
        lgb_clf.fit(X_train, y_train, 
                    eval_set=[(X_valid, y_valid)],
                    callbacks=[lgb.early_stopping(stopping_rounds=500, verbose=10)], 
                    eval_metric='auc')
        
        # Realizar predicciones OOF
        oof_preds[valid_idx] = lgb_clf.predict_proba(X_valid)[:, 1]
        print(f"Fold {n_fold+1} completado.")
        
        # Limpieza de memoria
        del X_train, y_train, X_valid, y_valid; gc.collect()

    # Evaluación final del modelo OOF
    final_auc = roc_auc_score(y, oof_preds)
    print("\n" + "=" * 50)
    print(f"✅ AUC OOF FINAL (Modelo Campeón): {final_auc:.4f}")
    print("=" * 50)

    # Guardar el último modelo entrenado
    dump(lgb_clf, os.path.join(ARTIFACTS_PATH, 'champion_model.pkl'))
    print(f"\nModelo Campeón (LightGBM) guardado en artifacts/champion_model.pkl")

if __name__ == "__main__":
    run_model_training()