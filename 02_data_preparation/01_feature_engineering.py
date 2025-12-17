# Import dependencias
import pandas as pd
import numpy as np
import os
import gc # Herramienta esencial para liberar memoria en datasets grandes


# CONFIGURACIÓN DE RUTA Y CARGA DE DATOS
DATA_PATH = './data/'

# Funcion cargar archivo .parquet
def load_parquet_file(filename):
    """Carga un archivo .parquet de forma segura."""
    file_path = os.path.join(DATA_PATH, filename)
    try:
        df = pd.read_parquet(file_path)
        print(f"Cargado: {filename}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"ERROR: No se pudo cargar {filename}. {e}")
        return None

# UTILITY: FUNCIÓN DE AGREGACIÓN GENERAL

def aggregate_dataframe(df, group_var, prefix):
    """Calcula agregaciones básicas (mean, max, min, sum, count) para un DF por grupo."""
    
    # 1. Crear el objeto GroupBy
    # Asegúrar de que df contiene group_var, si no, fallará.
    df_groups = df.groupby(group_var) 
    
    # 2. Seleccionar columnas NUMÉRICAS para AGREGAR (excluyendo la ID de agrupación)
    agg_cols = [col for col in df.select_dtypes(include=np.number).columns if col != group_var]
    
    # 3. Calcular las agregaciones
    agg_num = df_groups[agg_cols].agg(['mean', 'max', 'min', 'sum', 'count'])
    
    # Aplanar y renombrar las columnas
    agg_num.columns = [prefix + '_' + '_'.join(col).strip().upper() 
                       for col in agg_num.columns.values]
    
    return agg_num.reset_index() # Aquí group_var se convierte en una columna nuevamente

# FUNCIÓN CLAVE: INGENIERÍA DE FEATURES DE BURÓ
def get_bureau_features(df_app):
    """
    Procesa bureau.parquet y bureau_balance.parquet.
    Esta es una agregación en dos etapas (Multi-nivel).
    """
    
    print("\n--- Iniciando Ingeniería de Bureau ---")
    bureau = load_parquet_file('bureau.parquet')
    bureau_balance = load_parquet_file('bureau_balance.parquet')
    
    # ETAPA 1: Procesar bureau_balance (Agregación por crédito SK_ID_BUREAU)
    
    bureau_balance['STATUS_C_COUNT'] = (bureau_balance['STATUS'] == 'C').astype(int)
    
    bb_agg = aggregate_dataframe(bureau_balance, 'SK_ID_BUREAU', 'BB')
    
    # Merge de las agregaciones de balance al DF de bureau
    bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')
    
    # Liberar memoria de los DFs intermedios
    del bureau_balance, bb_agg; gc.collect()
    
    # ETAPA 2: Procesar Bureau (Agregación por cliente SK_ID_CURR)
    
    # Ingeniería de Características Específicas del Buró
    # Ratio de Deuda vs. Crédito Total
    bureau['DEBT_TO_CREDIT_RATIO'] = \
        bureau['AMT_CREDIT_SUM_DEBT'] / (bureau['AMT_CREDIT_SUM'] + 0.0001)
        
    # Agregación final por cliente (SK_ID_CURR)
    bureau_agg = aggregate_dataframe(bureau, 'SK_ID_CURR', 'BUREAU')


    # ETAPA 3: Unión al DataFrame Principal
    
    df_app = df_app.merge(bureau_agg, on='SK_ID_CURR', how='left')
    
    # Liberar memoria
    del bureau, bureau_agg; gc.collect()
    
    print(f"Bureau Features añadidas. DF_App shape: {df_app.shape}")
    return df_app


# FUNCIÓN CLAVE: INGENIERÍA DE FEATURES DE SOLICITUDES PREVIAS
def get_prev_app_features(df_app):
    """Procesa previous_application.parquet y lo agrega por cliente (SK_ID_CURR)."""
    
    print("\n--- Iniciando Ingeniería de Solicitudes Previas ---")
    prev = load_parquet_file('previous_application.parquet')
    
    # Ingeniería de Features: Ratios importantes
    prev['CREDIT_TO_ANNUITY_RATIO'] = prev['AMT_APPLICATION'] / (prev['AMT_ANNUITY'] + 0.0001)
    
    # Agregación por cliente (SK_ID_CURR)
    prev_agg = aggregate_dataframe(prev, 'SK_ID_CURR', 'PREV')
    
    # Unión al DataFrame Principal
    df_app = df_app.merge(prev_agg, on='SK_ID_CURR', how='left')
    
    del prev, prev_agg; gc.collect()
    print(f"Previous App Features añadidas. DF_App shape: {df_app.shape}")
    return df_app


# FLUJO PRINCIPAL DE INGENIERÍA
def run_feature_engineering_pipeline():
    # Cargar la tabla principal
    df_app = load_parquet_file('application_.parquet')
    
    if df_app is None:
        print("No se pudo iniciar el pipeline: La tabla principal es nula.")
        return

    # 1. Integrar Features de Buró
    df_app = get_bureau_features(df_app)
    
    # 2. Integrar Features de Solicitudes Previas
    df_app = get_prev_app_features(df_app)
    
    # 3. Integrar el resto de tablas (Instalments, POS_CASH, Credit_Card...)
    # df_app = get_installments_features(df_app) 
    # df_app = get_credit_card_features(df_app) 

    print("-" * 50)
    print(f"PIPELINE COMPLETADO. Dataset Maestro Final Shape: {df_app.shape}")
    print("-" * 50)
    
    # Guardar el resultado en artifacts para la siguiente fase
    # Se recomienda guardar el dataframe maestro (con TARGET, SK_ID_CURR, y las nuevas features)
    df_app.to_csv('./artifacts/master_train_features.csv', index=False)
    print("Archivo maestro guardado en artifacts/master_train_features.csv")


if __name__ == "__main__":
    # La ruta de los datos debe estar configurada en DATA_PATH arriba
    run_feature_engineering_pipeline()