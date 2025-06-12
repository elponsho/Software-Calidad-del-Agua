import pandas as pd

def leer_archivo_csv(ruta):
    try:
        df = pd.read_csv(ruta)
        return df
    except Exception as e:
        print(f"Error al leer CSV: {e}")
        return None

def leer_archivo_excel(ruta):
    try:
        df = pd.read_excel(ruta)
        return df
    except Exception as e:
        print(f"Error al leer Excel: {e}")
        return None
