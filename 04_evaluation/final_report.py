import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuración de Rutas ---
# Nota: La ruta de los artefactos es relativa a la carpeta 04_evaluation
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), '../artifacts')
MODEL_PATH = os.path.join(ARTIFACTS_PATH, 'champion_model.pkl')
DATA_PATH = os.path.join(ARTIFACTS_PATH, 'master_train_features.csv') # Usamos el set de entrenamiento maestro

def load_data_and_model():
    """Carga los datos preprocesados de prueba y el modelo campeón."""
    try:
        # Cargamos el dataset maestro completo
        df_master = pd.read_csv(DATA_PATH)
        df_test = df_master[df_master['TARGET'].notna()].sample(frac=0.2, random_state=42) # Usamos un 20% como 'test' simulado
        
        X_test = df_test.drop(columns=['TARGET', 'SK_ID_CURR']) # Asegúrate de que las columnas coincidan
        y_test = df_test['TARGET']
        
        # Cargar el modelo
        model = load(MODEL_PATH)
        return X_test, y_test, model
    except FileNotFoundError as e:
        print(f"Error: No se encontró un archivo necesario. Asegúrate de ejecutar las Fases 2 y 3. Error: {e}")
        return None, None, None
    except Exception as e:
        print(f"Error al cargar datos o modelo: {e}")
        return None, None, None

def plot_confusion_matrix(y_true, y_pred):
    """Genera y muestra la Matriz de Confusión."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['0: Pagó', '1: Incumplió'], yticklabels=['0: Pagó', '1: Incumplió'])
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Predicción del Modelo')
    plt.savefig('../reports/confusion_matrix.png')
    plt.show()

def plot_roc_curve(y_true, y_probs):
    """Genera y muestra la Curva ROC."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='orange', label=f'Curva ROC (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC')
    plt.legend()
    plt.savefig('../reports/roc_curve.png')
    plt.show()

def run_evaluation():
    X_test, y_test, model = load_data_and_model()

    if model is None:
        return

    # Predicción de Probabilidades y Clases
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Usamos un umbral para la clasificación binaria (ej. 0.5)
    y_pred = (y_probs > 0.5).astype(int) 

    # 1. Cálculo de Métricas
    auc_score = roc_auc_score(y_test, y_probs)
    
    print("=" * 50)
    print("📊 RESULTADOS DE LA EVALUACIÓN FINAL")
    print("=" * 50)
    print(f"✅ Área Bajo la Curva ROC (AUC): {auc_score:.4f}")
    print("\n--- Reporte de Clasificación (Threshold=0.5) ---")
    print(classification_report(y_test, y_pred))

    # 2. Generación de Visualizaciones
    # Asegurarse de crear la carpeta reports
    os.makedirs('../reports', exist_ok=True)
    
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_probs)
    
    print("\nVisualizaciones (Matriz de Confusión, Curva ROC) guardadas en la carpeta /reports.")

if __name__ == "__main__":
    run_evaluation()