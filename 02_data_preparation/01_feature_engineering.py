import pandas as pd
import numpy as np
import gc # Para liberar memoria, útil en este dataset grande

def aggregate_table(df, prefix, group_by_id='SK_ID_CURR'):
    """Calcula características agregadas (mean, max, sum) y las renombra."""
    
    # Excluir IDs y TARGET
    num_aggregations = {
        'CNT_DRAWINGS_ATM_CURRENT': ['mean', 'sum'],
        'AMT_INST_MIN_REGULARITY': ['mean', 'max'],
        # ... agregar más columnas y agregaciones relevantes
    }
    
    # Crear el dataframe de agregaciones
    agg_df = df.groupby(group_by_id).agg({**num_aggregations})
    
    # Aplanar y renombrar las columnas
    agg_df.columns = [prefix + '_' + '_'.join(col).strip().upper() 
                      for col in agg_df.columns.values]
    
    return agg_df

def run_feature_engineering():
    # 1. Cargar datos
    df_app_train = pd.read_csv('application_train.csv')
    df_bureau = pd.read_csv('bureau.csv')
    df_prev = pd.read_csv('previous_application.csv')
    
    # 2. Agregar características de Bureau (agrupando por cliente SK_ID_CURR)
    bureau_agg = aggregate_table(df_bureau, 'BUREAU')
    df_app_train = df_app_train.merge(bureau_agg, on='SK_ID_CURR', how='left')
    del bureau_agg; gc.collect()

    # 3. Agregar características de Solicitudes Previas (SK_ID_PREV se convierte a SK_ID_CURR)
    prev_agg = aggregate_table(df_prev, 'PREV')
    df_app_train = df_app_train.merge(prev_agg, on='SK_ID_CURR', how='left')
    del prev_agg; gc.collect()
    
    # ... Repetir el proceso para POS_CASH_BALANCE, INSTALLMENTS_PAYMENTS, etc.
    
    print("Dataset final después de la ingeniería de características:", df_app_train.shape)
    df_app_train.to_csv('processed_features.csv', index=False)
    
if __name__ == "__main__":
    run_feature_engineering()