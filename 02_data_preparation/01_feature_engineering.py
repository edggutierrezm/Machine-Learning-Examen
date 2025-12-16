import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split

# --- Funciones de Utilidad ---

def aggregate_dataframe(df, group_vars, prefix):
    """Calcula agregaciones básicas (mean, max, min, sum, count) para un DF."""
    
    # 1. Agregaciones Numéricas
    num_df = df.select_dtypes(include=np.number).drop(columns=group_vars, errors='ignore')
    agg_num = num_df.groupby(group_vars[0]).agg(['mean', 'max', 'min', 'sum', 'count'])
    
    # 2. Agregaciones Categóricas (Moda y Conteo de Valores Únicos)
    cat_df = df.select_dtypes(include='object')
    agg_cat = cat_df.groupby(group_vars[0]).agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    agg_cat.rename(columns={col: col + '_MODE' for col in agg_cat.columns}, inplace=True)
    
    # 3. Combinar y Aplanar
    agg_df = pd.merge(agg_num, agg_cat, on=group_vars[0], how='left')
    
    # Aplanar y renombrar
    agg_df.columns = [prefix + '_' + '_'.join(col).strip().upper() 
                      for col in agg_num.columns.values] + list(agg_cat.columns)
    
    return agg_df.reset_index()


# --- Funciones de Ingeniería de Características Específicas ---

def get_bureau_features(df_app):
    """Procesa bureau.csv y bureau_balance.csv."""
    
    print("Iniciando Ingeniería de Bureau...")
    bureau = pd.read_csv('bureau.csv')
    bureau_balance = pd.read_csv('bureau_balance.csv')
    
    # 1. Procesar bureau_balance (datos de comportamiento)
    # Agrupar por SK_ID_BUREAU (un crédito anterior) y luego agregarlos
    bb_agg = aggregate_dataframe(bureau_balance, ['SK_ID_BUREAU'], 'BB')
    bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')
    del bb_agg, bureau_balance; gc.collect()
    
    # 2. Procesar Bureau (información del crédito)
    # Generar agregaciones por cliente (SK_ID_CURR)
    bureau_agg = aggregate_dataframe(bureau, ['SK_ID_CURR'], 'BUREAU')
    
    # 3. Ingeniería de Características personalizadas (ej. ratio)
    if 'BUREAU_AMT_CREDIT_SUM_SUM' in bureau_agg.columns and 'BUREAU_AMT_CREDIT_SUM_DEBT_SUM' in bureau_agg.columns:
        bureau_agg['BUREAU_DEBT_RATIO'] = \
            bureau_agg['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] / bureau_agg['BUREAU_AMT_CREDIT_SUM_SUM']
            
    df_app = df_app.merge(bureau_agg, on='SK_ID_CURR', how='left')
    print(f"Bureau Features listas. DF shape: {df_app.shape}")
    return df_app

def get_previous_application_features(df_app):
    """Procesa previous_application.csv."""
    
    print("Iniciando Ingeniería de Solicitudes Previas...")
    prev = pd.read_csv('previous_application.csv')

    # Ingeniería de Features: crear ratios importantes antes de agregar
    prev['APP_CREDIT_PER_ANNUITY'] = prev['AMT_APPLICATION'] / prev['AMT_ANNUITY']
    
    # Agrupar por cliente (SK_ID_CURR)
    prev_agg = aggregate_dataframe(prev, ['SK_ID_CURR'], 'PREV')
    
    df_app = df_app.merge(prev_agg, on='SK_ID_CURR', how='left')
    print(f"Previous App Features listas. DF shape: {df_app.shape}")
    return df_app

# --- Flujo Principal ---

def run_feature_engineering_pipeline():
    # 1. Cargar datos principales
    df_app_train = pd.read_csv('application_train.csv')
    df_app_test = pd.read_csv('application_test.csv') # También procesamos el set de prueba
    
    # Concatenar para asegurar consistencia en el preprocesamiento
    df_app = pd.concat([df_app_train, df_app_test], ignore_index=True, sort=False)
    
    # 2. Integrar Múltiples Fuentes de Datos
    
    # A. Integrar Buró de Crédito
    df_app = get_bureau_features(df_app)
    
    # B. Integrar Solicitudes Previas
    df_app = get_previous_application_features(df_app)

    # C. Aquí se agregarían las otras tablas (POS_CASH, INSTALLMENTS, etc.)
    # ... df_app = get_installments_features(df_app)
    # ... df_app = get_pos_cash_features(df_app)
    
    print("-" * 50)
    print(f"Dataset final con alta dimensionalidad: {df_app.shape} características.") [cite: 38]
    print("-" * 50)
    
    # Separar en Train y Test nuevamente y guardar
    df_train_final = df_app[df_app['TARGET'].notna()]
    df_test_final = df_app[df_app['TARGET'].isna()]
    
    df_train_final.to_csv('../artifacts/master_train_features.csv', index=False)
    df_test_final.to_csv('../artifacts/master_test_features.csv', index=False)
    
    print("Archivos de características maestras guardados en /artifacts.")

if __name__ == "__main__":
    run_feature_engineering_pipeline()