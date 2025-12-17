import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuración de Rutas ---
# Nota: Las rutas son relativas a la carpeta 04_evaluation, por eso usamos '../artifacts'
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), '../artifacts')
MODEL_PATH = os.path.join(ARTIFACTS_PATH, 'champion_model.pkl')
X_TEST_FILE = os.path.join(ARTIFACTS_PATH, 'X_final_test.csv')
Y_TEST_FILE = os.path.join(ARTIFACTS_PATH, 'y_final_test.csv')
REPORTS_DIR = os.path.join(os.path.dirname(__file__), '../reports')

# ==========================================================
# 📌 FUNCIÓN DE LIMPIEZA DE NOMBRES DE COLUMNA (CLAVE)
# ==========================================================
# Reutilizamos la función de limpieza EXACTA que usamos en la Fase 3.
def clean_feature_names(df):
    """Limpia los nombres de las columnas para eliminar caracteres no soportados por LightGBM."""
    new_cols = []
    for col in df.columns:
        # Reemplaza corchetes, comas, dos puntos, <, > y otros caracteres especiales por guiones bajos
        cleaned_col = col.replace('[', '_').replace(']', '_').replace('<', '_').replace(':', '_').replace(',', '_').replace('>', '_')
        cleaned_col = cleaned_col.replace(' ', '_')
        new_cols.append(cleaned_col)
    df.columns = new_cols
    return df

def load_data_and_model():
    """Carga los datos preprocesados de prueba y el modelo campeón."""
    try:
        X_test = pd.read_csv(X_TEST_FILE)
        y_test = pd.read_csv(Y_TEST_FILE)
        
        # Cargar el modelo
        model = load(MODEL_PATH)
        print("Modelo y datos de prueba cargados exitosamente.")
        
        # APLICAR LIMPIEZA: Es fundamental aplicar la misma limpieza de nombres al set de prueba
        X_test = clean_feature_names(X_test)
        
        return X_test, y_test['TARGET'], model
    except FileNotFoundError as e:
        print(f"Error: No se encontró un archivo necesario. Asegúrate de ejecutar las Fases 2 y 3. Error: {e}")
        return None, None, None
    except Exception as e:
        print(f"Error al cargar datos o modelo: {e}")
        return None, None, None

def plot_confusion_matrix(y_true, y_pred):
    """Genera y guarda la Matriz de Confusión."""
    cm = confusion_matrix(y_true, y_pred)
    
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['0: Pagó', '1: Incumplió'], yticklabels=['0: Pagó', '1: Incumplió'])
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Predicción del Modelo')
    plt.savefig(os.path.join(REPORTS_DIR, 'confusion_matrix.png'))
    plt.show() 
    print("Matriz de Confusión guardada en /reports/confusion_matrix.png")

def plot_roc_curve(y_true, y_probs, auc):
    """Genera y guarda la Curva ROC."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='orange', label=f'Curva ROC (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC')
    plt.legend()
    plt.savefig(os.path.join(REPORTS_DIR, 'roc_curve.png'))
    plt.show() 
    print("Curva ROC guardada en /reports/roc_curve.png")

def run_evaluation():
    X_test, y_test, model = load_data_and_model()

    if model is None:
        return

    # Predicción de Probabilidades
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Cálculo de Métricas
    auc_score = roc_auc_score(y_test, y_probs)
    
    # Usamos un umbral para la clasificación binaria (ej. 0.5)
    y_pred = (y_probs > 0.5).astype(int) 

    print("\n" + "=" * 50)
    print("📊 RESULTADOS DE LA EVALUACIÓN FINAL")
    print("=" * 50)
    print(f"✅ Área Bajo la Curva ROC (AUC en Test): {auc_score:.4f}")
    print("\n--- Reporte de Clasificación (Threshold=0.5) ---")
    print(classification_report(y_test, y_pred))

    # 2. Generación de Visualizaciones
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_probs, auc_score)
    
if __name__ == "__main__":
    # Asegúrate de instalar seaborn y matplotlib: pip install seaborn matplotlib
    run_evaluation()