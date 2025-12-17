import pandas as pd
import numpy as np
from joblib import load
from flask import Flask, request, jsonify
import os

# ==========================================================
# 📌 CONFIGURACIÓN DE RUTAS Y CARGA DE ARTEFACTOS
# ==========================================================
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), '../artifacts')

MODEL_PATH = os.path.join(ARTIFACTS_PATH, 'champion_model.pkl')
SCALER_PATH = os.path.join(ARTIFACTS_PATH, 'scaler.pkl')

# ==========================================================
# 📌 FUNCIÓN DE LIMPIEZA DE NOMBRES DE COLUMNA (AÑADIDA)
# ==========================================================
# Es esencial que esta función sea EXACTAMENTE la misma que se usó en la Fase 3
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


try:
    # 1. Cargar el Modelo Campeón (LightGBM)
    MODEL = load(MODEL_PATH)
    # 2. Cargar el Escalador (MinMaxScaler)
    SCALER = load(SCALER_PATH)
    print("✅ Artefactos de despliegue cargados: Modelo y Escalador.")
except FileNotFoundError:
    print(f"ERROR: No se encontraron los artefactos en {ARTIFACTS_PATH}. Asegúrate de ejecutar Fases 2 y 3.")
    MODEL = None
    SCALER = None

# 3. Cargar la lista de características finales para el despliegue
try:
    X_train = pd.read_csv(os.path.join(ARTIFACTS_PATH, 'X_final_train.csv'))
    
    # 🚨 CLAVE: Limpiar los nombres de las columnas del DF de entrenamiento
    # Esto asegura que la lista de nombres que obtenemos sea la que LightGBM espera.
    X_train_processed = clean_feature_names(X_train.drop(columns=['TARGET'])) 
    CLEANED_FEATURE_NAMES = list(X_train_processed.columns)
    
    print(f"✅ Lista de características finales cargada ({len(CLEANED_FEATURE_NAMES)} features).")
    
    # Limpieza de memoria
    del X_train, X_train_processed; 
except FileNotFoundError:
    print("ERROR: No se encontró X_final_train.csv. No se puede obtener la lista de features.")
    CLEANED_FEATURE_NAMES = []
    
    
# ==========================================================
# 📌 API FLASK
# ==========================================================
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None or SCALER is None or not CLEANED_FEATURE_NAMES:
        return jsonify({'error': 'Servidor no inicializado. Faltan artefactos de modelado.'}), 503

    try:
        # 1. Recibir los datos del nuevo cliente (en formato JSON)
        data = request.get_json(force=True)
        
        # 2. Crear un DataFrame con los datos recibidos (simulando 1 fila)
        new_client_data = pd.DataFrame(data, index=[0])
        
        # 🚨 CLAVE 2: Limpiar los nombres de las columnas de entrada
        new_client_data = clean_feature_names(new_client_data)
        
        # 3. Asegurar que el DataFrame tiene la misma estructura que el entrenamiento
        # Creamos un DF con ceros para asegurar que todas las 469 columnas existen en el orden correcto
        X_predict = pd.DataFrame(0, index=[0], columns=CLEANED_FEATURE_NAMES)
        
        # Llenamos las columnas con los datos del cliente
        for col in new_client_data.columns:
            # Solo llenamos si la columna de entrada existe en nuestra lista esperada
            if col in CLEANED_FEATURE_NAMES:
                # Usamos .iloc[0] para asegurar que el valor se asigna correctamente
                X_predict[col] = new_client_data[col].iloc[0] 

        # 4. Aplicar el Escalador
        X_scaled = SCALER.transform(X_predict)
        
        # 5. Realizar la Predicción
        # Nota: El escalador devuelve un array numpy, así que usamos el DF X_predict
        # para la predicción, pero con los datos escalados.
        probability_default = MODEL.predict_proba(X_scaled)[:, 1][0]
        
        # 6. Definir el resultado binario (Umbral 0.5)
        default_risk = 1 if probability_default > 0.5 else 0
        
        # 7. Generar respuesta
        response = {
            'default_probability': round(probability_default, 4),
            'credit_granted': 'NO' if default_risk == 1 else 'SÍ',
            'model_decision': default_risk
        }
        
        return jsonify(response)

    except Exception as e:
        # Esto capturará errores de pandas/sklearn, incluyendo nombres de features no coincidentes
        return jsonify({'error': f'Error en el procesamiento de la solicitud: {str(e)}'}), 400

if __name__ == '__main__':
    print("\n--- INICIANDO SERVIDOR DE DESPLIEGUE ---")
    print(f"Servidor corriendo. Envía un POST a http://127.0.0.1:5000/predict")
    app.run(debug=False)