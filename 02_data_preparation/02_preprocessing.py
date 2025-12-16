import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
import numpy as np
import os

# Nota: La ruta de los artefactos es relativa a la carpeta 02_data_preparation
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), '../artifacts')

def run_preprocessing():
    # 1. Cargar el dataset maestro con todas las features (generado en 01_feature_engineering.py)
    df_master = pd.read_csv(os.path.join(ARTIFACTS_PATH, 'master_train_features.csv'))
    
    # Filtramos solo los datos con TARGET para entrenamiento
    df = df_master[df_master['TARGET'].notna()].copy()

    # Separar Target y Features
    X = df.drop(columns=['TARGET', 'SK_ID_CURR']) # Excluimos TARGET y la ID
    y = df['TARGET']
    
    # 2. Manejo de Nulos (Imputación con la Mediana para numéricos)
    for col in X.select_dtypes(include=np.number).columns:
        X[col] = X[col].fillna(X[col].median())
    
    # 3. Codificación de Categóricas (One-Hot Encoding)
    X = pd.get_dummies(X, dummy_na=False)

    # Nota: Si se usó One-Hot Encoding en el feature_engineering, esto puede ser redundante.
    # Asegúrate de que las columnas coincidan entre train y test si haces esto.
    
    # 4. Escalamiento de Datos (Se usa MinMaxScaler en todas las columnas numéricas)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reconvertir a DataFrame
    X_processed = pd.DataFrame(X_scaled, columns=X.columns)

    # 5. Guardar el Scaler en artifacts
    dump(scaler, os.path.join(ARTIFACTS_PATH, 'scaler.pkl'))
    print("Scaler guardado en artifacts/scaler.pkl")

    # 6. Guardar el set de entrenamiento final para la Fase 3 (Modelado)
    # Volvemos a añadir el target para el entrenamiento
    X_processed['TARGET'] = y.reset_index(drop=True)
    X_processed.to_csv(os.path.join(ARTIFACTS_PATH, 'X_final_train.csv'), index=False)
    
    print("Dataset preprocesado y listo para la fase de modelado.")

if __name__ == "__main__":
    run_preprocessing()