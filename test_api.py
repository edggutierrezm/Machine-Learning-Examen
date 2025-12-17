import requests
import json
import numpy as np

# ==========================================================
# CONFIGURACIÓN DEL CLIENTE
# ==========================================================
url = 'http://127.0.0.1:5000/predict'

# 📌 DATOS DE PRUEBA: USANDO ESPACIOS (LA ESTRUCTURA DEL MODELO ENTRENADO)
test_data = {
    # Variables Numéricas clave
    "AMT_CREDIT": [450000.0],
    "DAYS_BIRTH": [-12000],
    "EXT_SOURCE_1": [0.55],
    "EXT_SOURCE_2": [0.65],
    "EXT_SOURCE_3": [0.33],
    
    # Variables Categóricas OHE clave
    "FLAG_OWN_CAR_N": [1.0],  
    "NAME_FAMILY_STATUS_Married": [1.0],
    
    # 🚨 CORRECCIÓN: Nombres de columna con ESPACIO, tal como el modelo fue entrenado
    "FONDKAPREMONT_MODE_not specified": [0.0],
    "FONDKAPREMONT_MODE_org spec account": [0.0],
    "FONDKAPREMONT_MODE_reg oper account": [0.0],
    "FONDKAPREMONT_MODE_reg oper spec account": [0.0],
    "HOUSETYPE_MODE_block of flats": [1.0], 
}

# ==========================================================
# ENVÍO DE LA SOLICITUD
# ==========================================================
try:
    print("--- Enviando solicitud POST a la API ---")
    response = requests.post(url, json=test_data)
    
    print("-" * 50)
    print(f"Código de Estado de la Solicitud: {response.status_code}")
    print("Respuesta de la API:")
    
    if response.status_code == 200:
        print("✅ PREDICCIÓN EXITOSA. ¡PROYECTO TERMINADO!")
        print(json.dumps(response.json(), indent=4))
    else:
        print("❌ ERROR EN LA API")
        print(response.text) 
    
    print("-" * 50)

except requests.exceptions.ConnectionError:
    print("-" * 50)
    print("❌ ERROR DE CONEXIÓN: No se pudo conectar.")
    print("Asegúrate de que el servidor Flask (05_deployment/app.py) esté corriendo.")
    print("-" * 50)