import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#pip install pandas pyarrow
#tener instalado el codigo de arriba

# Define la ruta a tus archivos (ajusta si es necesario)
DATA_PATH = './data/' 

def load_and_inspect_data(filename):
    """Carga un archivo parquet y muestra su información clave."""
    try:
        df = pd.read_parquet(DATA_PATH + filename)
        
        print(f"\n--- VISUALIZACIÓN INICIAL: {filename} ---")
        
        # 1. Dimensiones (Shape)
        print(f"Dimensiones del DataFrame: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        # 2. Primeras filas (Head)
        print("\nPrimeras 5 filas:")
        print(df.head())
        
        # 3. Resumen de Tipos de Datos y Nulos (Info)
        print("\nResumen de tipos de datos y valores nulos:")
        df.info()
        
        return df
        
    except Exception as e:
        print(f"ERROR: No se pudo cargar ni visualizar {filename}. Error: {e}")
        return None

# Cargar y visualizar el archivo principal
df_app = load_and_inspect_data('application_.parquet')