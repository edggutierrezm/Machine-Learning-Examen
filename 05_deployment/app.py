from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
import os

app = Flask(__name__)

# Rutas a los artefactos
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), '../artifacts')
MODEL_PATH = os.path.join(ARTIFACTS_PATH, 'champion_model.pkl')
SCALER_PATH = os.path.join(ARTIFACTS_PATH, 'scaler.pkl')

# Cargar el modelo y el scaler al iniciar la aplicación [cite: 22]
try:
    model = load(MODEL_PATH)
    scaler = load(SCALER_PATH)
    print("Modelo y Scaler cargados exitosamente.")
except Exception as e:
    print(f"Error al cargar artefactos: {e}")
    model = None
    scaler = None

@app.route('/evaluate_risk', methods=['POST'])
def evaluate_risk():
    """Endpoint para predecir el riesgo de incumplimiento [cite: 39]"""
    
    if not model or not scaler:
        return jsonify({"error": "Modelo o scaler no cargados"}), 500

    try:
        # Recibir los datos del solicitante en formato JSON [cite: 24]
        data = request.get_json()
        
        # Convertir a DataFrame (simulando los datos de un nuevo solicitante)
        new_applicant_df = pd.DataFrame(data, index=[0])

        # ATENCIÓN: En un escenario real, necesitarías aplicar TODAS 
        # las transformaciones de la Fase 2 aquí (imputación, one-hot encoding, 
        # y especialmente la ingeniería de features si se requiere).
        
        # Por simplicidad, escalaremos directamente los datos recibidos (asumiendo 
        # que ya tienen la misma estructura que los datos de entrenamiento).
        
        # Aplicar el scaling guardado
        applicant_scaled = scaler.transform(new_applicant_df)
        
        # Realizar la predicción (probabilidad de incumplimiento)
        probability_of_default = model.predict_proba(applicant_scaled)[:, 1][0]
        
        # Definir la decisión sugerida [cite: 39]
        if probability_of_default > 0.6: # Umbral alto
            decision = 'RECHAZAR'
        elif probability_of_default > 0.4: # Umbral medio
            decision = 'REVISIÓN MANUAL'
        else:
            decision = 'APROBAR'

        return jsonify({
            "probability_of_default": round(probability_of_default, 4),
            "decision_sugerida": decision
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Ejecutar la API
    app.run(debug=True, host='0.0.0.0', port=5000)