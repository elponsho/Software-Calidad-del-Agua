import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
from scipy import stats


# CONFIGURACIÓN PARA EJECUTABLES
def get_temp_dir():
    """Obtiene directorio temporal que funciona en ejecutables"""
    if getattr(sys, 'frozen', False):
        # En ejecutable, usar directorio del ejecutable
        base_dir = os.path.dirname(sys.executable)
    else:
        # En desarrollo, usar directorio actual
        base_dir = os.getcwd()

    temp_dir = os.path.join(base_dir, "temp_graphs")

    # Crear directorio si no existe
    try:
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
    except Exception as e:
        # Si falla, usar directorio temporal del sistema
        import tempfile
        temp_dir = tempfile.gettempdir()
        print(f"⚠️ Usando directorio temporal del sistema: {temp_dir}")

    return temp_dir


# Configuración global
TEMP_DIR = get_temp_dir()
ruta_imagen = os.path.join(TEMP_DIR, "grafico_temp.png")

# Paleta de colores
COLORES = {
    'primario': '#2E8B57',
    'secundario': '#4682B4',
    'acento': '#FF6347',
    'neutro': '#708090',
    'agua': '#00CED1',
    'peligro': '#DC143C',
    'exito': '#228B22',
    'advertencia': '#FF8C00'
}


def crear_directorio_temp():
    """Crear directorio temporal si no existe"""
    try:
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)
    except Exception as e:
        print(f"⚠️ Error creando directorio temporal: {e}")


def generar_boxplot(df, columna):
    """Generar boxplot con rutas seguras para ejecutables"""
    try:
        crear_directorio_temp()

        plt.figure(figsize=(8, 6))

        data = df[columna].dropna()

        if len(data) == 0:
            plt.text(0.5, 0.5, 'No hay datos válidos', ha='center', va='center',
                     transform=plt.gca().transAxes, fontsize=14)
        else:
            # Crear boxplot
            box_plot = plt.boxplot(data,
                                   patch_artist=True,
                                   notch=True,
                                   showmeans=True,
                                   meanline=True)

            # Personalizar colores
            box_plot['boxes'][0].set_facecolor(COLORES['primario'])
            box_plot['boxes'][0].set_alpha(0.7)
            box_plot['medians'][0].set_color('white')
            box_plot['medians'][0].set_linewidth(2)

            if 'means' in box_plot:
                box_plot['means'][0].set_color(COLORES['acento'])
                box_plot['means'][0].set_linewidth(2)

            # Estadísticas
            media = data.mean()
            mediana = data.median()
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)

            stats_text = f'Media: {media:.3f}\nMediana: {mediana:.3f}\nQ1: {q1:.3f}\nQ3: {q3:.3f}'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                     verticalalignment='top', bbox=dict(boxstyle='round',
                                                        facecolor='lightblue', alpha=0.8), fontsize=10)

        plt.title(f'📦 Boxplot de {columna}', fontsize=14, fontweight='bold')
        plt.ylabel(columna, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # GUARDAR CON RUTA ABSOLUTA
        plt.savefig(ruta_imagen, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Boxplot guardado en: {ruta_imagen}")

    except Exception as e:
        print(f"❌ Error generando boxplot: {e}")
        # Crear imagen de error
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f'Error al generar boxplot:\n{str(e)}',
                 ha='center', va='center', transform=plt.gca().transAxes,
                 fontsize=12, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        plt.title(f'Error - Boxplot de {columna}')
        plt.tight_layout()
        plt.savefig(ruta_imagen, dpi=300, bbox_inches='tight')
        plt.close()


def generar_histograma(df, columna):
    """Generar histograma con rutas seguras"""
    try:
        crear_directorio_temp()

        plt.figure(figsize=(8, 6))

        data = df[columna].dropna()

        if len(data) == 0:
            plt.text(0.5, 0.5, 'No hay datos válidos', ha='center', va='center',
                     transform=plt.gca().transAxes, fontsize=14)
        else:
            # Calcular número óptimo de bins
            n_bins = max(10, min(50, int(np.sqrt(len(data)))))

            # Crear histograma
            n, bins, patches = plt.hist(data, bins=n_bins,
                                        color=COLORES['agua'],
                                        alpha=0.7,
                                        edgecolor='black',
                                        linewidth=0.8)

            # Agregar líneas de estadísticas
            media = data.mean()
            mediana = data.median()

            plt.axvline(media, color=COLORES['acento'], linestyle='--',
                        linewidth=2, label=f'Media: {media:.3f}')
            plt.axvline(mediana, color=COLORES['exito'], linestyle='--',
                        linewidth=2, label=f'Mediana: {mediana:.3f}')

            plt.legend()

        plt.title(f'📊 Histograma de {columna}', fontsize=14, fontweight='bold')
        plt.xlabel(columna, fontweight='bold')
        plt.ylabel('Frecuencia', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(ruta_imagen, dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"❌ Error generando histograma: {e}")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f'Error al generar histograma:\n{str(e)}',
                 ha='center', va='center', transform=plt.gca().transAxes,
                 fontsize=12, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        plt.title(f'Error - Histograma de {columna}')
        plt.tight_layout()
        plt.savefig(ruta_imagen, dpi=300, bbox_inches='tight')
        plt.close()


def generar_densidad(df, columna):
    """Generar gráfico de densidad con rutas seguras"""
    try:
        crear_directorio_temp()

        plt.figure(figsize=(8, 6))

        data = df[columna].dropna()

        if len(data) == 0:
            plt.text(0.5, 0.5, 'No hay datos válidos', ha='center', va='center',
                     transform=plt.gca().transAxes, fontsize=14)
        else:
            # Intentar usar KDE de scipy, si falla usar histograma normalizado
            try:
                if len(data) > 5:
                    density = stats.gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 200)
                    density_values = density(x_range)

                    plt.fill_between(x_range, density_values, alpha=0.6,
                                     color=COLORES['primario'], label='Densidad')
                    plt.plot(x_range, density_values, color=COLORES['neutro'],
                             linewidth=2)
                else:
                    raise ValueError("Pocos datos para KDE")

            except Exception:
                # Fallback: histograma normalizado
                plt.hist(data, bins=20, density=True, alpha=0.6,
                         color=COLORES['primario'], edgecolor='black',
                         label='Densidad (histograma)')

            # Agregar líneas de estadísticas
            media = data.mean()
            mediana = data.median()

            plt.axvline(media, color=COLORES['acento'], linestyle='--',
                        linewidth=2, label=f'Media: {media:.3f}')
            plt.axvline(mediana, color=COLORES['exito'], linestyle='--',
                        linewidth=2, label=f'Mediana: {mediana:.3f}')

            plt.legend()

        plt.title(f'🌊 Densidad de {columna}', fontsize=14, fontweight='bold')
        plt.xlabel(columna, fontweight='bold')
        plt.ylabel('Densidad', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(ruta_imagen, dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"❌ Error generando densidad: {e}")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f'Error al generar gráfico de densidad:\n{str(e)}',
                 ha='center', va='center', transform=plt.gca().transAxes,
                 fontsize=12, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        plt.title(f'Error - Densidad de {columna}')
        plt.tight_layout()
        plt.savefig(ruta_imagen, dpi=300, bbox_inches='tight')
        plt.close()


def diagrama_dispersion(df, x, y):
    """Generar diagrama de dispersión con rutas seguras"""
    try:
        crear_directorio_temp()

        plt.figure(figsize=(8, 6))

        # Filtrar datos válidos
        df_clean = df[[x, y]].dropna()

        if len(df_clean) == 0:
            plt.text(0.5, 0.5, 'No hay datos válidos para ambas variables',
                     ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        else:
            x_data = df_clean[x]
            y_data = df_clean[y]

            # Scatter plot
            plt.scatter(x_data, y_data, alpha=0.6, color=COLORES['agua'],
                        s=50, edgecolors='black', linewidth=0.5)

            # Línea de tendencia
            try:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                plt.plot(x_data, p(x_data), color=COLORES['acento'],
                         linestyle='--', linewidth=2, label='Tendencia')

                # Correlación
                correlation = x_data.corr(y_data)
                plt.text(0.05, 0.95, f'Correlación: {correlation:.3f}',
                         transform=plt.gca().transAxes,
                         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                         fontsize=12, fontweight='bold')

                plt.legend()

            except Exception as e:
                print(f"Error calculando tendencia: {e}")

        plt.xlabel(x, fontweight='bold')
        plt.ylabel(y, fontweight='bold')
        plt.title(f'🎯 Dispersión: {x} vs {y}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(ruta_imagen, dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"❌ Error generando dispersión: {e}")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f'Error al generar diagrama de dispersión:\n{str(e)}',
                 ha='center', va='center', transform=plt.gca().transAxes,
                 fontsize=12, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        plt.title(f'Error - Dispersión: {x} vs {y}')
        plt.tight_layout()
        plt.savefig(ruta_imagen, dpi=300, bbox_inches='tight')
        plt.close()


def serie_tiempo(df, col_fecha, col_valor):
    """Generar serie de tiempo con rutas seguras"""
    try:
        crear_directorio_temp()

        plt.figure(figsize=(10, 6))

        # Preparar datos
        df_temp = df[[col_fecha, col_valor]].copy()
        df_temp = df_temp.dropna()

        if len(df_temp) == 0:
            plt.text(0.5, 0.5, 'No hay datos válidos para la serie temporal',
                     ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        else:
            # Convertir fecha y ordenar
            try:
                df_temp[col_fecha] = pd.to_datetime(df_temp[col_fecha])
            except Exception as e:
                print(f"Error convirtiendo fechas: {e}")
                pass

            df_sorted = df_temp.sort_values(by=col_fecha)

            # Plot principal
            plt.plot(df_sorted[col_fecha], df_sorted[col_valor],
                     marker='o', linewidth=2, markersize=4,
                     color=COLORES['primario'], markerfacecolor=COLORES['agua'],
                     markeredgecolor='black', markeredgewidth=0.5,
                     label=col_valor)

            # Línea de media
            media = df_sorted[col_valor].mean()
            plt.axhline(media, color=COLORES['exito'], linestyle=':',
                        alpha=0.7, label=f'Media: {media:.3f}')

            plt.legend()

        plt.title(f'📈 {col_valor} en el tiempo', fontsize=14, fontweight='bold')
        plt.xlabel(col_fecha, fontweight='bold')
        plt.ylabel(col_valor, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(ruta_imagen, dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"❌ Error generando serie de tiempo: {e}")
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f'Error al generar serie de tiempo:\n{str(e)}',
                 ha='center', va='center', transform=plt.gca().transAxes,
                 fontsize=12, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        plt.title(f'Error - Serie de Tiempo: {col_valor}')
        plt.tight_layout()
        plt.savefig(ruta_imagen, dpi=300, bbox_inches='tight')
        plt.close()


def obtener_ruta_imagen():
    """Obtener ruta absoluta de la imagen para ejecutables"""
    return os.path.abspath(ruta_imagen)


def limpiar_imagen_temporal():
    """Limpiar imagen temporal de manera segura"""
    try:
        if os.path.exists(ruta_imagen):
            os.remove(ruta_imagen)
            print("✅ Imagen temporal limpiada")
    except Exception as e:
        print(f"⚠️ Error limpiando imagen temporal: {e}")


def limpiar_archivos_temporales():
    """Limpiar todos los archivos temporales"""
    try:
        if os.path.exists(TEMP_DIR):
            for archivo in os.listdir(TEMP_DIR):
                if archivo.endswith(('.png', '.jpg', '.jpeg')):
                    ruta_archivo = os.path.join(TEMP_DIR, archivo)
                    try:
                        os.remove(ruta_archivo)
                    except Exception as e:
                        print(f"⚠️ No se pudo eliminar {archivo}: {e}")
            print("✅ Archivos temporales limpiados")
    except Exception as e:
        print(f"⚠️ Error limpiando archivos temporales: {e}")


def debug_paths():
    """Mostrar información de rutas para debugging"""
    print("=== DEBUG DE RUTAS ===")
    print(f"Ejecutable: {getattr(sys, 'frozen', False)}")
    print(f"Directorio actual: {os.getcwd()}")
    print(f"Directorio del script: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Directorio temporal: {TEMP_DIR}")
    print(f"Ruta de imagen: {ruta_imagen}")
    if getattr(sys, 'frozen', False):
        print(f"Directorio del ejecutable: {os.path.dirname(sys.executable)}")
        if hasattr(sys, '_MEIPASS'):
            print(f"Directorio temporal PyInstaller: {sys._MEIPASS}")
    print("=" * 50)


# Para compatibilidad con el código original
def generar_grafico_temp(tipo, *args, **kwargs):
    """Función genérica para generar gráficos temporales"""
    if tipo == 'boxplot':
        return generar_boxplot(*args, **kwargs)
    elif tipo == 'histograma':
        return generar_histograma(*args, **kwargs)
    elif tipo == 'densidad':
        return generar_densidad(*args, **kwargs)
    elif tipo == 'dispersion':
        return diagrama_dispersion(*args, **kwargs)
    elif tipo == 'serie_tiempo':
        return serie_tiempo(*args, **kwargs)
    else:
        raise ValueError(f"Tipo de gráfico no reconocido: {tipo}")