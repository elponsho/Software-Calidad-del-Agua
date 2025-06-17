import pandas as pd

def resumen_univariable(df):
    # ✅ Filtrar solo columnas numéricas
    df_num = df.select_dtypes(include='number')

    resumen = pd.DataFrame()
    resumen["Mínimo"] = df_num.min()
    resumen["Máximo"] = df_num.max()
    resumen["Media"] = df_num.mean()
    resumen["Mediana"] = df_num.median()
    resumen["Q1"] = df_num.quantile(0.25)
    resumen["Q3"] = df_num.quantile(0.75)

    return resumen.round(2)
