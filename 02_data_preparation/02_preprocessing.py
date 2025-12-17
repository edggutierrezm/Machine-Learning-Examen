# Import dependencias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
import os
import gc


# CONFIGURACIÓN DE RUTAS
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), '../artifacts')
MASTER_FILE = os.path.join(ARTIFACTS_PATH, 'master_train_features.csv')

def run_preprocessing():
    print("--- Iniciando Preprocesamiento Final ---")
    
    # 1. Cargar el dataset maestro (generado por 01_feature_engineering.py)
    try:
        df_master = pd.read_csv(MASTER_FILE)
        print(f"Dataset maestro cargado. Shape inicial: {df_master.shape}")
    except FileNotFoundError:
        print(f"ERROR: Archivo maestro no encontrado en {MASTER_FILE}. Ejecuta la fase 01_feature_engineering primero.")
        return

    # Filtramos solo los datos de entrenamiento (donde TARGET no es nulo)
    df_train = df_master[df_master['TARGET'].notna()].copy()
    
    # Separar Target, IDs y Features
    X = df_train.drop(columns=['TARGET', 'SK_ID_CURR']) 
    y = df_train['TARGET'].astype(int)
    
    # Liberar la memoria del DataFrame maestro completo
    del df_master; gc.collect()

    # 2. Manejo de Nulos (Imputación)
    print("Imputando nulos...")
    
    # Imputar variables numéricas (que son la mayoría de las 347 features) con la mediana
    for col in X.select_dtypes(include=np.number).columns:
        # Aquí también podrías usar SimpleImputer de sklearn
        X[col] = X[col].fillna(X[col].median())
        
    # Imputar variables categóricas (usando la moda si es que quedaron algunas)
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].fillna(X[col].mode()[0])
    
    # 3. Codificación de Categóricas (One-Hot Encoding para las columnas restantes)
    print("Aplicando One-Hot Encoding...")
    X = pd.get_dummies(X, dummy_na=False)

    # 4. Escalamiento de Datos (Normalización)
    print("Aplicando escalamiento de datos (MinMaxScaler)...")
    scaler = MinMaxScaler()
    
    # Ajustar y transformar los datos
    X_scaled = scaler.fit_transform(X)
    
    # Reconvertir a DataFrame para mantener nombres de columnas
    X_processed = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    #Guardar el Scaler (ESENCIAL para el Despliegue en Fase 5)
    dump(scaler, os.path.join(ARTIFACTS_PATH, 'scaler.pkl'))
    print(f"Scaler guardado en {ARTIFACTS_PATH}/scaler.pkl")

    # 5. División de Datos (Entrenamiento y Prueba para la Fase 3)
    # Usamos stratify=y para mantener la proporción del Target desbalanceado en ambos sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"\nDatos listos para modelado. Shape Final de Entrenamiento: {X_train.shape}")
    
    # 6. Guardar los sets de datos preprocesados para la Fase 3
    # NOTA: Guardamos X_train y y_train juntos para simplificar la Fase 3
    X_train['TARGET'] = y_train
    
    X_train.to_csv(os.path.join(ARTIFACTS_PATH, 'X_final_train.csv'), index=False)
    X_test.to_csv(os.path.join(ARTIFACTS_PATH, 'X_final_test.csv'), index=False)
    y_test.to_csv(os.path.join(ARTIFACTS_PATH, 'y_final_test.csv'), index=False)
    
    print("Sets de entrenamiento y prueba guardados en la carpeta /artifacts.")
    

if __name__ == "__main__":
    # Asegurar que la carpeta artifacts exista antes de guardar
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)
    run_preprocessing()