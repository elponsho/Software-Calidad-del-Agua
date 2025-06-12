import pandas as pd

def resumen_univariable(df):
    resumen = df.describe(include='all').T
    resumen["Q1"] = df.quantile(0.25)
    resumen["Q3"] = df.quantile(0.75)
    resumen = resumen[["min", "25%", "Q1", "50%", "mean", "Q3", "75%", "max"]].rename(columns={
        "25%": "Cuartil 25%",
        "50%": "Mediana",
        "75%": "Cuartil 75%",
        "mean": "Media",
        "min": "Mínimo",
        "max": "Máximo"
    })
    return resumen.round(3)
