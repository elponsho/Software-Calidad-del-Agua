import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ruta para guardar imágenes temporalmente. Se genera a partir de la
# ubicación de este archivo para evitar problemas cuando la aplicación se
# ejecuta desde otros directorios.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ruta_imagen = os.path.join(BASE_DIR, "temp_grafica.png")

def generar_boxplot(df, columna):
    plt.figure(figsize=(5, 4))
    sns.boxplot(y=df[columna])
    plt.title(f'Boxplot de {columna}')
    plt.tight_layout()
    plt.savefig(ruta_imagen)
    plt.close()

def generar_histograma(df, columna):
    plt.figure(figsize=(5, 4))
    sns.histplot(df[columna], kde=False, bins=10)
    plt.title(f'Histograma de {columna}')
    plt.tight_layout()
    plt.savefig(ruta_imagen)
    plt.close()

def generar_densidad(df, columna):
    plt.figure(figsize=(5, 4))
    sns.kdeplot(df[columna], fill=True)
    plt.title(f'Densidad de {columna}')
    plt.tight_layout()
    plt.savefig(ruta_imagen)
    plt.close()

def diagrama_dispersion(df, x, y):
    plt.figure(figsize=(5, 4))
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(f"Dispersión entre {x} y {y}")
    plt.tight_layout()
    plt.savefig(ruta_imagen)
    plt.close()

def serie_tiempo(df, col_fecha, col_valor):
    df_sorted = df.sort_values(by=col_fecha)
    plt.figure(figsize=(6, 4))
    plt.plot(df_sorted[col_fecha], df_sorted[col_valor], marker='o')
    plt.title(f"{col_valor} en el tiempo")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(ruta_imagen)
    plt.close()

def obtener_ruta_imagen():
    return os.path.abspath(ruta_imagen)
