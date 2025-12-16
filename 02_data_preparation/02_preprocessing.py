import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from joblib import dump # Para guardar el scaler
import numpy as np

def run_preprocessing():
    df = pd.read_csv('processed_features.csv')
    
    # Separar Target y Features
    X = df.drop('TARGET', axis=1)
    y = df['TARGET']
    
    # Manejo de Nulos (ej. imputación simple)
    for col in X.select_dtypes(include=np.number).columns:
        X[col] = X[col].fillna(X[col].median())
    
    # Codificación de Categóricas (One-Hot Encoding, si no hay muchas)
    X = pd.get_dummies(X, dummy_na=False)
    
    # Escalamiento de Datos
    scaler = MinMaxScaler()
    # Fit solo en las columnas numéricas relevantes para evitar las dummies
    X_scaled = scaler.fit_transform(X.select_dtypes(include=np.number))
    
    # Reconvertir a DataFrame para mantener nombres de columnas
    X_processed = pd.DataFrame(X_scaled, columns=X.select_dtypes(include=np.number).columns, index=X.index)

    # 💾 Guardar el Scaler en artifacts [cite: 19]
    dump(scaler, '../artifacts/scaler.pkl')

    # División de Datos (ej. 80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Guardar los sets de datos
    X_train.to_csv('../artifacts/X_train.csv', index=False)
    # ... guardar el resto de sets

if __name__ == "__main__":
    run_preprocessing()