import sys
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QTextEdit,
                             QLabel, QHBoxLayout, QScrollArea, QFrame,
                             QProgressBar, QApplication, QSplitter, QGroupBox,
                             QGridLayout, QTabWidget, QTableWidget, QTableWidgetItem,
                             QHeaderView, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QMutex, QMutexLocker
from PyQt5.QtGui import QFont, QTextCursor, QPixmap, QPalette, QColor
import traceback
import gc  # Para garbage collection manual

# Importaciones de matplotlib con manejo de errores
try:
    import matplotlib

    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    plt.style.use('default')
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: matplotlib not available: {e}")
    MATPLOTLIB_AVAILABLE = False

# Importaciones de sklearn con manejo de errores
try:
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, silhouette_score

    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: sklearn not available: {e}")
    SKLEARN_AVAILABLE = False
    np = None
    pd = None

import warnings

warnings.filterwarnings('ignore')


class DataCache:
    """Cache para almacenar datos y evitar regeneración innecesaria"""
    _instance = None
    _cache = {}
    _mutex = QMutex()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataCache, cls).__new__(cls)
        return cls._instance

    def get(self, key):
        with QMutexLocker(self._mutex):
            return self._cache.get(key, None)

    def set(self, key, value):
        with QMutexLocker(self._mutex):
            self._cache[key] = value

    def clear(self):
        with QMutexLocker(self._mutex):
            self._cache.clear()
            gc.collect()


def generar_datos_agua_optimizado(n_muestras=100, seed=42):
    """Función independiente para generar datos - optimizada para multiprocesamiento"""
    np.random.seed(seed)

    # Generar todos los parámetros de una vez para mejor rendimiento
    datos = {
        'pH': np.clip(np.random.normal(7.2, 0.8, n_muestras), 6.0, 8.5),
        'Oxígeno_Disuelto': np.clip(np.random.normal(8.5, 1.5, n_muestras), 4.0, 12.0),
        'Turbidez': np.clip(np.random.exponential(2.0, n_muestras), 0.1, 8.0),
        'Conductividad': np.clip(np.random.normal(500, 200, n_muestras), 100, 1200)
    }

    # Vectorizar el cálculo de calidad
    calidad_scores = np.zeros(n_muestras)

    # pH score (vectorizado)
    ph_score = np.where(
        (datos['pH'] >= 6.5) & (datos['pH'] <= 8.5),
        25,
        np.maximum(0, 25 - np.abs(datos['pH'] - 7.0) * 8)
    )

    # Oxígeno score (vectorizado)
    oxigeno_score = np.where(
        datos['Oxígeno_Disuelto'] >= 6,
        25,
        datos['Oxígeno_Disuelto'] * 4
    )

    # Turbidez score (vectorizado)
    turbidez_score = np.where(
        datos['Turbidez'] < 4,
        25,
        np.maximum(0, 25 - (datos['Turbidez'] - 4) * 5)
    )

    # Conductividad score (vectorizado)
    conductividad_score = np.where(
        (datos['Conductividad'] >= 200) & (datos['Conductividad'] <= 800),
        25,
        np.maximum(0, 25 - np.abs(datos['Conductividad'] - 500) * 0.03)
    )

    calidad_scores = ph_score + oxigeno_score + turbidez_score + conductividad_score

    # Categorías de calidad (vectorizado)
    calidades = np.select(
        [calidad_scores >= 80, calidad_scores >= 60, calidad_scores >= 40],
        ["Excelente", "Buena", "Regular"],
        default="Necesita Tratamiento"
    )

    datos['Calidad_Score'] = calidad_scores
    datos['Calidad'] = calidades

    return pd.DataFrame(datos)


def analizar_calidad_agua_proceso(n_muestras=150):
    """Función para análisis de calidad - ejecutada en proceso separado"""
    try:
        df = generar_datos_agua_optimizado(n_muestras)

        stats = {
            'total_muestras': len(df),
            'ph_promedio': float(df['pH'].mean()),
            'oxigeno_promedio': float(df['Oxígeno_Disuelto'].mean()),
            'turbidez_promedio': float(df['Turbidez'].mean()),
            'conductividad_promedio': float(df['Conductividad'].mean()),
            'calidad_promedio': float(df['Calidad_Score'].mean())
        }

        distribucion_calidad = df['Calidad'].value_counts().to_dict()

        excelente_pct = (distribucion_calidad.get('Excelente', 0) / len(df)) * 100
        necesita_tratamiento_pct = (distribucion_calidad.get('Necesita Tratamiento', 0) / len(df)) * 100

        if excelente_pct >= 70:
            estado_general = "🟢 EXCELENTE"
            mensaje = "La calidad del agua es muy buena en la mayoría de las muestras."
        elif necesita_tratamiento_pct >= 30:
            estado_general = "🔴 REQUIERE ATENCIÓN"
            mensaje = "Se detectaron problemas que necesitan tratamiento inmediato."
        else:
            estado_general = "🟡 ACEPTABLE"
            mensaje = "La calidad es aceptable pero puede mejorarse."

        return {
            'tipo': 'calidad_agua',
            'datos': df.to_dict('records'),  # Convertir a dict para serialización
            'estadisticas': stats,
            'distribucion': distribucion_calidad,
            'estado_general': estado_general,
            'mensaje': mensaje
        }
    except Exception as e:
        return {'error': str(e)}


def agrupar_muestras_proceso(n_muestras=100):
    """Función para agrupamiento - ejecutada en proceso separado"""
    try:
        df = generar_datos_agua_optimizado(n_muestras)
        X = df[['pH', 'Oxígeno_Disuelto', 'Turbidez', 'Conductividad']].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
        grupos = clustering.fit_predict(X_scaled)

        df['Grupo'] = grupos

        analisis_grupos = {}
        for grupo in range(3):
            muestras_grupo = df[df['Grupo'] == grupo]

            caracteristicas = []
            if muestras_grupo['pH'].mean() > 7.5:
                caracteristicas.append("pH elevado")
            elif muestras_grupo['pH'].mean() < 6.8:
                caracteristicas.append("pH bajo")

            if muestras_grupo['Turbidez'].mean() > 3:
                caracteristicas.append("Alta turbidez")

            if muestras_grupo['Oxígeno_Disuelto'].mean() < 6:
                caracteristicas.append("Bajo oxígeno")

            if not caracteristicas:
                caracteristicas.append("Parámetros normales")

            analisis_grupos[f"Grupo {grupo + 1}"] = {
                'cantidad': int(len(muestras_grupo)),
                'caracteristicas': caracteristicas,
                'calidad_promedio': float(muestras_grupo['Calidad_Score'].mean())
            }

        return {
            'tipo': 'agrupamiento',
            'datos': df.to_dict('records'),
            'analisis_grupos': analisis_grupos
        }
    except Exception as e:
        return {'error': str(e)}


def predecir_calidad_proceso(n_muestras=200):
    """Función para predicción - ejecutada en proceso separado"""
    try:
        df = generar_datos_agua_optimizado(n_muestras)
        X = df[['pH', 'Oxígeno_Disuelto', 'Turbidez', 'Conductividad']]
        y = df['Calidad']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Usar menos estimadores para mejor rendimiento
        modelo = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
        modelo.fit(X_train, y_train)

        precision = float(modelo.score(X_test, y_test) * 100)

        importancias = pd.DataFrame({
            'Parámetro': X.columns,
            'Importancia': modelo.feature_importances_
        }).sort_values('Importancia', ascending=False)

        ejemplos_prediccion = []
        for i in range(min(3, len(X_test))):
            muestra = X_test.iloc[i]
            prediccion = modelo.predict([muestra])[0]
            probabilidades = modelo.predict_proba([muestra])[0]
            confianza = float(max(probabilidades) * 100)

            ejemplos_prediccion.append({
                'pH': float(muestra['pH']),
                'Oxígeno': float(muestra['Oxígeno_Disuelto']),
                'Turbidez': float(muestra['Turbidez']),
                'Conductividad': float(muestra['Conductividad']),
                'Predicción': prediccion,
                'Confianza': confianza
            })

        return {
            'tipo': 'prediccion',
            'precision': precision,
            'importancias': importancias.to_dict('records'),
            'ejemplos': ejemplos_prediccion
        }
    except Exception as e:
        return {'error': str(e)}


def optimizar_sistema_proceso():
    """Función para optimización - ejecutada en proceso separado"""
    try:
        df = generar_datos_agua_optimizado(100)
        X = df[['pH', 'Oxígeno_Disuelto', 'Turbidez', 'Conductividad']]
        y = df['Calidad']

        configuraciones = [
            {'n_estimators': 30, 'max_depth': 3},
            {'n_estimators': 50, 'max_depth': 5},
            {'n_estimators': 70, 'max_depth': 7}
        ]

        resultados = []
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        for config in configuraciones:
            modelo = RandomForestClassifier(
                n_estimators=config['n_estimators'],
                max_depth=config['max_depth'],
                random_state=42,
                n_jobs=1
            )

            modelo.fit(X_train, y_train)
            precision = float(modelo.score(X_test, y_test) * 100)

            resultados.append({
                'configuracion': f"{config['n_estimators']} árboles, prof. {config['max_depth']}",
                'precision': precision
            })

        mejor = max(resultados, key=lambda x: x['precision'])

        return {
            'tipo': 'optimizacion',
            'resultados': resultados,
            'mejor_config': mejor
        }
    except Exception as e:
        return {'error': str(e)}


def comparar_metodos_proceso():
    """Función para comparación - ejecutada en proceso separado"""
    try:
        df = generar_datos_agua_optimizado(150)
        X = df[['pH', 'Oxígeno_Disuelto', 'Turbidez', 'Conductividad']]
        y = df['Calidad']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        metodos = {
            'Árbol de Decisión': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Bosque Aleatorio': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1),
            'Máquina de Vectores': SVC(random_state=42, kernel='rbf', C=1.0)
        }

        comparacion = []

        for nombre, modelo in metodos.items():
            if nombre == 'Máquina de Vectores':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                modelo.fit(X_train_scaled, y_train)
                precision = float(modelo.score(X_test_scaled, y_test) * 100)
            else:
                modelo.fit(X_train, y_train)
                precision = float(modelo.score(X_test, y_test) * 100)

            if nombre == 'Árbol de Decisión':
                ventajas = "Fácil de interpretar, rápido"
            elif nombre == 'Bosque Aleatorio':
                ventajas = "Muy preciso, robusto"
            else:
                ventajas = "Efectivo con datos complejos"

            comparacion.append({
                'metodo': nombre,
                'precision': precision,
                'ventajas': ventajas
            })

        return {
            'tipo': 'comparacion',
            'metodos': comparacion
        }
    except Exception as e:
        return {'error': str(e)}


class OptimizedMLWorker(QThread):
    """Worker thread optimizado con pool de procesos"""
    finished = pyqtSignal(dict, str)
    progress = pyqtSignal(int)
    status_update = pyqtSignal(str)

    def __init__(self, algorithm, data=None):
        super().__init__()
        self.algorithm = algorithm
        self.data = data
        self.cache = DataCache()

    def run(self):
        try:
            # Verificar cache primero
            cache_key = f"{self.algorithm}_{hash(str(self.data))}"
            cached_result = self.cache.get(cache_key)

            if cached_result:
                self.status_update.emit("📦 Usando datos en cache...")
                self.progress.emit(100)
                self.finished.emit(cached_result, self.algorithm)
                return

            self.status_update.emit("🚀 Iniciando procesamiento optimizado...")
            self.progress.emit(10)

            # Mapeo de funciones optimizadas
            process_functions = {
                "calidad_agua": analizar_calidad_agua_proceso,
                "agrupar_muestras": agrupar_muestras_proceso,
                "predecir_calidad": predecir_calidad_proceso,
                "optimizar_sistema": optimizar_sistema_proceso,
                "comparar_metodos": comparar_metodos_proceso
            }

            if self.algorithm not in process_functions:
                self.finished.emit({"error": "Análisis no reconocido"}, "error")
                return

            self.progress.emit(30)

            # Ejecutar en proceso separado con timeout
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(process_functions[self.algorithm])

                # Simular progreso mientras esperamos
                for i in range(30, 90, 10):
                    time.sleep(0.1)  # Pequeña pausa para no saturar
                    self.progress.emit(i)

                try:
                    result = future.result(timeout=30)  # Timeout de 30 segundos
                    self.progress.emit(95)

                    # Guardar en cache si es exitoso
                    if 'error' not in result:
                        self.cache.set(cache_key, result)

                    self.progress.emit(100)
                    self.finished.emit(result, self.algorithm)

                except Exception as e:
                    self.finished.emit({"error": f"Timeout o error en procesamiento: {str(e)}"}, "error")

        except Exception as e:
            self.finished.emit({"error": f"Error en worker: {str(e)}"}, "error")
        finally:
            # Limpieza de memoria
            gc.collect()


class ResultadosVisuales(QWidget):
    """Widget optimizado para mostrar resultados"""

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.current_data = None

    def setup_ui(self):
        layout = QVBoxLayout()

        self.titulo_resultado = QLabel("📊 Resultados del Análisis")
        self.titulo_resultado.setObjectName("resultTitle")
        self.titulo_resultado.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.titulo_resultado)

        self.tabs = QTabWidget()

        # Tab 1: Resumen Visual
        self.resumen_tab = QWidget()
        self.tabs.addTab(self.resumen_tab, "📋 Resumen")

        # Tab 2: Gráficos
        self.graficos_tab = QWidget()
        self.tabs.addTab(self.graficos_tab, "📊 Gráficos")

        # Tab 3: Recomendaciones
        self.recomendaciones_tab = QWidget()
        self.tabs.addTab(self.recomendaciones_tab, "💡 Recomendaciones")

        layout.addWidget(self.tabs)
        self.setLayout(layout)

        self.setup_tabs()

    def setup_tabs(self):
        """Configurar cada tab de manera optimizada"""
        # Tab resumen
        resumen_layout = QVBoxLayout()
        self.resumen_scroll = QScrollArea()
        self.resumen_content = QLabel("Ejecuta un análisis para ver los resultados")
        self.resumen_content.setAlignment(Qt.AlignTop)
        self.resumen_content.setWordWrap(True)
        self.resumen_content.setObjectName("resumenContent")
        self.resumen_scroll.setWidget(self.resumen_content)
        self.resumen_scroll.setWidgetResizable(True)
        resumen_layout.addWidget(self.resumen_scroll)
        self.resumen_tab.setLayout(resumen_layout)

        # Tab gráficos
        graficos_layout = QVBoxLayout()
        if MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(10, 6), dpi=80)  # Reducir DPI para mejor rendimiento
            self.canvas = FigureCanvas(self.figure)
            graficos_layout.addWidget(self.canvas)
        else:
            graficos_layout.addWidget(QLabel("Matplotlib no disponible"))
        self.graficos_tab.setLayout(graficos_layout)

        # Tab recomendaciones
        rec_layout = QVBoxLayout()
        self.recomendaciones_scroll = QScrollArea()
        self.recomendaciones_content = QLabel("Las recomendaciones aparecerán aquí")
        self.recomendaciones_content.setAlignment(Qt.AlignTop)
        self.recomendaciones_content.setWordWrap(True)
        self.recomendaciones_content.setObjectName("recomendacionesContent")
        self.recomendaciones_scroll.setWidget(self.recomendaciones_content)
        self.recomendaciones_scroll.setWidgetResizable(True)
        rec_layout.addWidget(self.recomendaciones_scroll)
        self.recomendaciones_tab.setLayout(rec_layout)

    def mostrar_resultados(self, resultados, tipo_analisis):
        """Mostrar resultados de manera optimizada"""
        try:
            self.current_data = resultados

            # Debug: Verificar que llegaron los resultados
            print(f"📊 Mostrando resultados para: {tipo_analisis}")
            print(f"🔍 Claves disponibles: {list(resultados.keys())}")

            if tipo_analisis == "calidad_agua":
                self.mostrar_calidad_agua(resultados)
            elif tipo_analisis == "agrupar_muestras":
                self.mostrar_agrupamiento(resultados)
            elif tipo_analisis == "predecir_calidad":
                self.mostrar_prediccion(resultados)
            elif tipo_analisis == "optimizar_sistema":
                self.mostrar_optimizacion(resultados)
            elif tipo_analisis == "comparar_metodos":
                self.mostrar_comparacion(resultados)
            else:
                self.resumen_content.setText(f"❌ Tipo de análisis no reconocido: {tipo_analisis}")

        except Exception as e:
            error_msg = f"❌ Error al mostrar resultados: {str(e)}\n\nTipo: {tipo_analisis}\nDatos: {str(resultados)[:200]}..."
            print(error_msg)
            self.resumen_content.setText(error_msg)

    def mostrar_prediccion(self, resultados):
        """Mostrar resultados de predicción optimizado"""
        try:
            precision = resultados['precision']
            importancias = resultados['importancias']
            ejemplos = resultados['ejemplos']

            # HTML corregido
            resumen_html = f"""
            <div style="font-family: Arial; font-size: 14px; padding: 20px;">
                <div style="text-align: center; background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                            color: #333; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <h2>🤖 Sistema de Predicción</h2>
                    <h3>Precisión del Sistema: {precision:.1f}%</h3>
                    <p style="font-size: 16px;">El sistema puede predecir la calidad del agua con alta confiabilidad</p>
                </div>

                <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; margin: 15px 0;">
                    <h3 style="color: #2e7d32;">📊 Importancia de Parámetros</h3>
                    <p>Estos parámetros son los más importantes para determinar la calidad:</p>
            """

            # Mostrar importancias
            if isinstance(importancias, list):
                for item in importancias:
                    porcentaje = item['Importancia'] * 100
                    resumen_html += f"""
                    <div style="margin: 8px 0;">
                        <strong>{item['Parámetro']}:</strong> {porcentaje:.1f}% de importancia
                        <div style="background: #f0f0f0; height: 8px; border-radius: 4px; margin-top: 3px;">
                            <div style="background: #4CAF50; height: 8px; width: {porcentaje}%; border-radius: 4px;"></div>
                        </div>
                    </div>
                    """

            resumen_html += """
                </div>

                <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 15px 0;">
                    <h3 style="color: #1565c0;">🔮 Ejemplos de Predicción</h3>
            """

            # Mostrar ejemplos
            for i, ejemplo in enumerate(ejemplos, 1):
                color_pred = {
                    'Excelente': '#4CAF50',
                    'Buena': '#8BC34A',
                    'Regular': '#FF9800',
                    'Necesita Tratamiento': '#F44336'
                }.get(ejemplo['Predicción'], '#666')

                resumen_html += f"""
                <div style="background: white; padding: 10px; margin: 8px 0; border-radius: 5px; 
                            border-left: 4px solid {color_pred};">
                    <h4>Muestra {i}</h4>
                    <p><strong>pH:</strong> {ejemplo['pH']:.2f} | <strong>Oxígeno:</strong> {ejemplo['Oxígeno']:.1f} mg/L | 
                       <strong>Turbidez:</strong> {ejemplo['Turbidez']:.1f} NTU | 
                       <strong>Conductividad:</strong> {ejemplo['Conductividad']:.1f}</p>
                    <p><strong>Predicción:</strong> <span style="color: {color_pred}; font-weight: bold;">
                       {ejemplo['Predicción']}</span> (Confianza: {ejemplo['Confianza']:.1f}%)</p>
                </div>
                """

            resumen_html += "</div></div>"
            self.resumen_content.setText(resumen_html)

            # Crear gráfico de importancias
            if MATPLOTLIB_AVAILABLE:
                self.crear_grafico_importancias_simple(importancias)

        except Exception as e:
            error_msg = f"❌ Error en mostrar_prediccion: {str(e)}"
            print(error_msg)
            self.mostrar_error_en_pantalla("Error en Predicción", error_msg)


    def mostrar_optimizacion(self, resultados):
        """Mostrar resultados de optimización"""
        try:
            configuraciones = resultados['resultados']
            mejor = resultados['mejor_config']

            # HTML corregido
            resumen_html = f"""
            <div style="font-family: Arial; font-size: 14px; padding: 20px;">
                <div style="text-align: center; background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                            color: #333; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <h2>⚙️ Optimización del Sistema</h2>
                    <p style="font-size: 16px;">Se probaron diferentes configuraciones para encontrar la óptima</p>
                </div>

                <div style="background: #e8f5e8; border-left: 5px solid #4CAF50; padding: 15px; 
                            margin: 15px 0; border-radius: 8px;">
                    <h3 style="color: #2e7d32;">🏆 Mejor Configuración</h3>
                    <p><strong>Configuración:</strong> {mejor['configuracion']}</p>
                    <p><strong>Precisión alcanzada:</strong> {mejor['precision']:.1f}%</p>
                    <p><strong>Recomendación:</strong> Usar esta configuración para análisis futuros</p>
                </div>

                <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 15px 0;">
                    <h3 style="color: #1565c0;">📈 Comparación de Configuraciones</h3>
            """

            for config in configuraciones:
                color = '#4CAF50' if config == mejor else '#2196F3'
                icon = '🏆' if config == mejor else '⚙️'

                resumen_html += f"""
                <div style="background: white; padding: 10px; margin: 8px 0; border-radius: 5px; 
                            border-left: 4px solid {color};">
                    <p>{icon} <strong>{config['configuracion']}</strong></p>
                    <p>Precisión: {config['precision']:.1f}%</p>
                    <div style="background: #f0f0f0; height: 8px; border-radius: 4px; margin-top: 5px;">
                        <div style="background: {color}; height: 8px; width: {config['precision']}%; border-radius: 4px;"></div>
                    </div>
                </div>
                """

            resumen_html += "</div></div>"
            self.resumen_content.setText(resumen_html)

            # Crear gráfico de comparación
            if MATPLOTLIB_AVAILABLE:
                self.crear_grafico_optimizacion_simple(configuraciones)

        except Exception as e:
            error_msg = f"❌ Error en mostrar_optimizacion: {str(e)}"
            print(error_msg)
            self.mostrar_error_en_pantalla("Error en Optimización", error_msg)

    def mostrar_agrupamiento(self, resultados):
                """Mostrar resultados de agrupamiento optimizado"""
                try:
                    analisis = resultados['analisis_grupos']

                    # HTML corregido
                    resumen_html = """
                    <div style="font-family: Arial; font-size: 14px; padding: 20px;">
                        <div style="text-align: center; background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); 
                                    color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                            <h2>🔍 Agrupamiento de Muestras</h2>
                            <p style="font-size: 16px;">Se encontraron patrones similares usando clustering</p>
                        </div>
                    """

                    colores_grupos = ['#e8f5e8', '#e3f2fd', '#fff3e0']
                    colores_bordes = ['#4CAF50', '#2196F3', '#FF9800']

                    for i, (grupo, info) in enumerate(analisis.items()):
                        color_fondo = colores_grupos[i % len(colores_grupos)]
                        color_borde = colores_bordes[i % len(colores_bordes)]

                        resumen_html += f"""
                        <div style="background: {color_fondo}; border-left: 5px solid {color_borde}; 
                                    padding: 15px; margin: 15px 0; border-radius: 8px;">
                            <h3 style="color: {color_borde}; margin-top: 0;">📊 {grupo}</h3>
                            <p><strong>Cantidad de muestras:</strong> {info['cantidad']}</p>
                            <p><strong>Calidad promedio:</strong> {info['calidad_promedio']:.1f}/100</p>
                            <p><strong>Características principales:</strong> {', '.join(info['caracteristicas'])}</p>
                        </div>
                        """

                    resumen_html += "</div>"
                    self.resumen_content.setText(resumen_html)

                    # Crear gráfico de agrupamiento
                    if MATPLOTLIB_AVAILABLE:
                        self.crear_grafico_agrupamiento_simple(resultados)

                except Exception as e:
                    error_msg = f"❌ Error en mostrar_agrupamiento: {str(e)}"
                    print(error_msg)
                    self.mostrar_error_en_pantalla("Error en Agrupamiento", error_msg)

                    def mostrar_calidad_agua(self, resultados):
                        """Mostrar resultados de análisis de calidad del agua optimizado"""
                        try:
                            stats = resultados['estadisticas']
                            distribucion = resultados['distribucion']

                            # HTML corregido sin espacios en etiquetas
                            resumen_html = f"""
                            <div style="font-family: Arial; font-size: 14px; padding: 20px;">
                                <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                            color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                                    <h2>🧪 Análisis de Calidad del Agua</h2>
                                    <h3>{resultados['estado_general']}</h3>
                                    <p style="font-size: 16px;">{resultados['mensaje']}</p>
                                </div>

                                <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; border-left: 5px solid #4CAF50;">
                                    <h3 style="color: #2e7d32; margin-top: 0;">📊 Estadísticas Generales</h3>
                                    <p><strong>Total de muestras:</strong> {stats['total_muestras']}</p>
                                    <p><strong>Calidad promedio:</strong> {stats['calidad_promedio']:.1f}/100</p>
                                    <p><strong>pH promedio:</strong> {stats['ph_promedio']:.2f}</p>
                                    <p><strong>Oxígeno:</strong> {stats['oxigeno_promedio']:.1f} mg/L</p>
                                    <p><strong>Turbidez:</strong> {stats['turbidez_promedio']:.1f} NTU</p>
                                    <p><strong>Conductividad:</strong> {stats['conductividad_promedio']:.1f} μS/cm</p>
                                </div>

                                <div style="background: #fff3e0; padding: 15px; border-radius: 8px; margin-top: 15px;">
                                    <h3 style="color: #ef6c00;">📈 Distribución de Calidad</h3>
                            """

                            # Añadir distribución de calidad
                            total_muestras = sum(distribucion.values())
                            for calidad, cantidad in distribucion.items():
                                porcentaje = (cantidad / total_muestras) * 100
                                color_barra = {
                                    'Excelente': '#4CAF50',
                                    'Buena': '#8BC34A',
                                    'Regular': '#FF9800',
                                    'Necesita Tratamiento': '#F44336'
                                }.get(calidad, '#9E9E9E')

                                resumen_html += f"""
                                    <div style="margin: 8px 0;">
                                        <strong>{calidad}:</strong> {cantidad} muestras ({porcentaje:.1f}%)
                                        <div style="background: #f0f0f0; height: 12px; border-radius: 6px; margin-top: 3px;">
                                            <div style="background: {color_barra}; height: 12px; width: {porcentaje}%; 
                                                        border-radius: 6px; transition: width 0.3s ease;"></div>
                                        </div>
                                    </div>
                                """

                            resumen_html += """
                                </div>

                                <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin-top: 15px;">
                                    <h3 style="color: #1565c0;">💡 Recomendaciones</h3>
                            """

                            # Generar recomendaciones automáticas
                            recomendaciones = self.generar_recomendaciones_calidad(stats, distribucion)
                            for rec in recomendaciones:
                                resumen_html += f"<p>• {rec}</p>"

                            resumen_html += "</div></div>"

                            self.resumen_content.setText(resumen_html)

                            # Actualizar tab de recomendaciones
                            self.actualizar_recomendaciones_calidad(stats, distribucion)

                            # Crear gráficos optimizados
                            if MATPLOTLIB_AVAILABLE:
                                self.crear_graficos_calidad_optimizado(resultados)

                        except Exception as e:
                            error_msg = f"❌ Error en mostrar_calidad_agua: {str(e)}"
                            print(error_msg)
                            self.mostrar_error_en_pantalla("Error en Calidad del Agua", error_msg)

    def mostrar_error_en_pantalla(self, titulo, mensaje):
        """Mostrar errores en la pantalla principal en lugar de solo en consola"""
        error_html = f"""
        <div style="font-family: Arial; font-size: 14px; padding: 20px;">
            <div style="text-align: center; background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); 
                        color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h2>❌ {titulo}</h2>
                <p style="font-size: 16px;">Se produjo un error durante el procesamiento</p>
            </div>

            <div style="background: #fff5f5; border-left: 5px solid #ff6b6b; padding: 15px; 
                        margin: 15px 0; border-radius: 8px;">
                <h3 style="color: #d63031; margin-top: 0;">🚨 Detalles del Error</h3>
                <p style="font-family: monospace; background: #f8f9fa; padding: 10px; 
                          border-radius: 4px; word-wrap: break-word;">{mensaje}</p>
            </div>

            <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin-top: 15px;">
                <h3 style="color: #1565c0;">💡 Posibles Soluciones</h3>
                <ul>
                    <li>Verificar que todas las dependencias estén instaladas</li>
                    <li>Reintentar el análisis</li>
                    <li>Limpiar el cache del sistema</li>
                    <li>Liberar memoria si el sistema está sobrecargado</li>
                </ul>
            </div>
        </div>
        """
        self.resumen_content.setText(error_html)

    def crear_graficos_calidad_optimizado(self, resultados):
        """Crear gráficos optimizados para calidad del agua"""
        try:
            self.figure.clear()

            # Convertir datos si es necesario
            if isinstance(resultados['datos'], list):
                datos = pd.DataFrame(resultados['datos'])
            else:
                datos = resultados['datos']

            distribucion = resultados['distribucion']

            # Crear subplots
            ax1 = self.figure.add_subplot(2, 2, 1)  # Distribución de calidad
            ax2 = self.figure.add_subplot(2, 2, 2)  # Parámetros promedio
            ax3 = self.figure.add_subplot(2, 2, 3)  # Histograma pH
            ax4 = self.figure.add_subplot(2, 2, 4)  # Scatter plot

            # Gráfico 1: Distribución de calidad (pie chart)
            colores = ['#4CAF50', '#8BC34A', '#FF9800', '#F44336']
            labels = list(distribucion.keys())
            sizes = list(distribucion.values())

            if sizes and sum(sizes) > 0:
                wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                                                   colors=colores[:len(labels)], startangle=90)
                ax1.set_title('📊 Distribución de Calidad', fontsize=10, fontweight='bold')
                # Mejorar legibilidad
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
            else:
                ax1.text(0.5, 0.5, 'Sin datos\npara mostrar', ha='center', va='center')
                ax1.set_title('📊 Distribución de Calidad', fontsize=10, fontweight='bold')

            # Gráfico 2: Parámetros promedio (bar chart)
            try:
                if hasattr(datos, 'mean'):
                    parametros = ['pH', 'Oxígeno', 'Turbidez', 'Conductividad']
                    valores = [
                        datos['pH'].mean(),
                        datos['Oxígeno_Disuelto'].mean(),
                        datos['Turbidez'].mean(),
                        datos['Conductividad'].mean() / 100  # Escalar para visualización
                    ]
                else:
                    # Procesar lista de diccionarios
                    ph_vals = [d.get('pH', 0) for d in datos]
                    ox_vals = [d.get('Oxígeno_Disuelto', 0) for d in datos]
                    turb_vals = [d.get('Turbidez', 0) for d in datos]
                    cond_vals = [d.get('Conductividad', 0) for d in datos]

                    parametros = ['pH', 'Oxígeno', 'Turbidez', 'Conductividad']
                    valores = [
                        sum(ph_vals) / len(ph_vals) if ph_vals else 0,
                        sum(ox_vals) / len(ox_vals) if ox_vals else 0,
                        sum(turb_vals) / len(turb_vals) if turb_vals else 0,
                        (sum(cond_vals) / len(cond_vals)) / 100 if cond_vals else 0
                    ]

                colores_bar = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
                barras = ax2.bar(parametros, valores, color=colores_bar, alpha=0.8, edgecolor='black',
                                 linewidth=0.5)
                ax2.set_title('📈 Parámetros Promedio', fontsize=10, fontweight='bold')
                ax2.set_ylabel('Valor')
                ax2.tick_params(axis='x', rotation=45)

                # Añadir valores en las barras
                for barra, valor in zip(barras, valores):
                    if valor > 0:
                        height = barra.get_height()
                        ax2.text(barra.get_x() + barra.get_width() / 2., height + height * 0.02,
                                 f'{valor:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

            except Exception as e:
                ax2.text(0.5, 0.5, f'Error:\n{str(e)[:30]}...', ha='center', va='center')
                ax2.set_title('📈 Parámetros Promedio', fontsize=10, fontweight='bold')

            # Gráfico 3: Histograma de pH
            try:
                if hasattr(datos, 'hist'):
                    ph_data = datos['pH'].dropna()
                else:
                    ph_data = [d.get('pH', 7) for d in datos if d.get('pH')]

                if len(ph_data) > 0:
                    ax3.hist(ph_data, bins=15, alpha=0.7, color='#3F51B5', edgecolor='black', linewidth=0.5)
                    ax3.axvline(x=7, color='red', linestyle='--', linewidth=2, label='pH Neutro')
                    ax3.axvspan(6.5, 8.5, alpha=0.2, color='green', label='Rango Aceptable')
                    ax3.set_title('📊 Distribución de pH', fontsize=10, fontweight='bold')
                    ax3.set_xlabel('pH')
                    ax3.set_ylabel('Frecuencia')
                    ax3.legend(fontsize=8)
                else:
                    ax3.text(0.5, 0.5, 'Sin datos de pH', ha='center', va='center')
                    ax3.set_title('📊 Distribución de pH', fontsize=10, fontweight='bold')
            except Exception as e:
                ax3.text(0.5, 0.5, f'Error pH:\n{str(e)[:20]}', ha='center', va='center')
                ax3.set_title('📊 Distribución de pH', fontsize=10, fontweight='bold')

            # Gráfico 4: Scatter plot Oxígeno vs Turbidez
            try:
                if hasattr(datos, 'plot'):
                    ox_data = datos['Oxígeno_Disuelto'].values
                    turb_data = datos['Turbidez'].values
                    calidad_scores = datos['Calidad_Score'].values
                else:
                    ox_data = [d.get('Oxígeno_Disuelto', 8) for d in datos]
                    turb_data = [d.get('Turbidez', 2) for d in datos]
                    calidad_scores = [d.get('Calidad_Score', 75) for d in datos]

                if len(ox_data) > 0 and len(turb_data) > 0:
                    scatter = ax4.scatter(ox_data, turb_data, c=calidad_scores,
                                          cmap='RdYlGn', alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
                    ax4.set_title('💧 Oxígeno vs Turbidez', fontsize=10, fontweight='bold')
                    ax4.set_xlabel('Oxígeno Disuelto (mg/L)')
                    ax4.set_ylabel('Turbidez (NTU)')

                    # Añadir colorbar
                    cbar = self.figure.colorbar(scatter, ax=ax4, shrink=0.8)
                    cbar.set_label('Calidad Score', rotation=270, labelpad=15, fontsize=8)
                else:
                    ax4.text(0.5, 0.5, 'Sin datos suficientes', ha='center', va='center')
                    ax4.set_title('💧 Oxígeno vs Turbidez', fontsize=10, fontweight='bold')
            except Exception as e:
                ax4.text(0.5, 0.5, f'Error scatter:\n{str(e)[:20]}', ha='center', va='center')
                ax4.set_title('💧 Oxígeno vs Turbidez', fontsize=10, fontweight='bold')

            self.figure.suptitle('🔬 Análisis Completo de Calidad del Agua', fontsize=12, fontweight='bold')
            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            print(f"❌ Error completo en gráficos: {e}")
            self.figure.clear()
            ax = self.figure.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f'❌ Error al crear gráficos:\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
            ax.set_title('Error en Visualización', fontsize=14, fontweight='bold')
            self.canvas.draw()

            # También mostrar el error en pantalla
            self.mostrar_error_en_pantalla("Error en Gráficos",
                                           f"Error al crear gráficos de calidad del agua: {str(e)}")

    def mostrar_comparacion(self, resultados):
        """Mostrar comparación de métodos - función completa"""
        try:
            metodos = resultados['metodos']
            mejor_metodo = max(metodos, key=lambda x: x['precision'])

            resumen_html = f"""
               <div style="font-family: Arial; font-size: 14px; padding: 20px;">
                   <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                               color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                       <h2>🔬 Comparación de Métodos</h2>
                       <p style="font-size: 16px;">Evaluación de diferentes enfoques de análisis</p>
                       <p><strong>Mejor método:</strong> {mejor_metodo['metodo']} ({mejor_metodo['precision']:.1f}%)</p>
                   </div>
               """

            for metodo in metodos:
                es_mejor = metodo == mejor_metodo
                color = '#4CAF50' if es_mejor else '#2196F3'
                icon = '🏆' if es_mejor else '🔬'

                resumen_html += f"""
                   <div style="background: {'#e8f5e8' if es_mejor else '#e3f2fd'}; 
                               border-left: 5px solid {color}; padding: 15px; margin: 15px 0; border-radius: 8px;">
                       <h3 style="color: {color}; margin-top: 0;">{icon} {metodo['metodo']}</h3>
                       <p><strong>Precisión:</strong> {metodo['precision']:.1f}%</p>
                       <p><strong>Ventajas:</strong> {metodo['ventajas']}</p>
                       {'<p style="color: #4CAF50; font-weight: bold;">✅ Método recomendado</p>' if es_mejor else ''}

                       <div style="background: #f0f0f0; height: 10px; border-radius: 5px; margin-top: 8px;">
                           <div style="background: {color}; height: 10px; width: {metodo['precision']}%; 
                                       border-radius: 5px; transition: width 0.3s ease;"></div>
                       </div>
                   </div>
                   """

            # Agregar análisis comparativo
            resumen_html += """
               <div style="background: #fff3e0; padding: 15px; border-radius: 8px; margin-top: 15px;">
                   <h3 style="color: #ef6c00;">📊 Análisis Comparativo</h3>
                   <p><strong>Conclusiones:</strong></p>
                   <ul>
               """

            # Generar conclusiones automáticas
            precision_max = max(m['precision'] for m in metodos)
            precision_min = min(m['precision'] for m in metodos)
            diferencia = precision_max - precision_min

            if diferencia < 5:
                resumen_html += "<li>Los métodos muestran rendimiento similar, elegir por facilidad de uso</li>"
            elif diferencia < 15:
                resumen_html += "<li>Hay diferencias moderadas en precisión, considerar el método recomendado</li>"
            else:
                resumen_html += "<li>Diferencias significativas en rendimiento, usar el método óptimo</li>"

            if precision_max > 90:
                resumen_html += "<li>Excelente precisión general del sistema de análisis</li>"
            elif precision_max > 80:
                resumen_html += "<li>Buena precisión, sistema confiable para toma de decisiones</li>"
            else:
                resumen_html += "<li>Precisión moderada, considerar mejorar el conjunto de datos</li>"

            resumen_html += """
                   </ul>
               </div>
               </div>
               """

            self.resumen_content.setText(resumen_html)

            # Crear gráfico de comparación
            if MATPLOTLIB_AVAILABLE:
                self.crear_grafico_comparacion_simple(metodos)

        except Exception as e:
            error_msg = f"❌ Error en mostrar_comparacion: {str(e)}"
            print(error_msg)
            self.mostrar_error_en_pantalla("Error en Comparación", error_msg)

    def crear_grafico_agrupamiento_simple(self, resultados):
        """Crear gráfico simple de agrupamiento"""
        try:
            self.figure.clear()
            datos = pd.DataFrame(resultados['datos']) if isinstance(resultados['datos'], list) else resultados['datos']

            ax = self.figure.add_subplot(1, 1, 1)

            # Scatter plot por grupos
            grupos_unicos = datos['Grupo'].unique()
            colores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

            for i, grupo in enumerate(grupos_unicos):
                grupo_data = datos[datos['Grupo'] == grupo]
                ax.scatter(grupo_data['pH'], grupo_data['Oxígeno_Disuelto'],
                           c=colores[i % len(colores)], label=f'Grupo {grupo + 1}',
                           alpha=0.7, s=60, edgecolors='black', linewidth=0.5)

            ax.set_xlabel('pH')
            ax.set_ylabel('Oxígeno Disuelto (mg/L)')
            ax.set_title('🔍 Agrupamiento de Muestras por pH y Oxígeno', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            print(f"Error en gráfico agrupamiento: {e}")

    def crear_grafico_importancias_simple(self, importancias):
        """Crear gráfico simple de importancias"""
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(1, 1, 1)

            if isinstance(importancias, list):
                parametros = [item['Parámetro'] for item in importancias]
                valores = [item['Importancia'] * 100 for item in importancias]

                colores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                barras = ax.barh(parametros, valores, color=colores[:len(parametros)],
                                 alpha=0.8, edgecolor='black', linewidth=0.5)

                ax.set_xlabel('Importancia (%)')
                ax.set_title('📊 Importancia de Parámetros en la Predicción', fontweight='bold')

                # Añadir valores en las barras
                for barra, valor in zip(barras, valores):
                    width = barra.get_width()
                    ax.text(width + 1, barra.get_y() + barra.get_height() / 2,
                            f'{valor:.1f}%', ha='left', va='center', fontweight='bold')

            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            print(f"Error en gráfico importancias: {e}")

    def crear_grafico_optimizacion_simple(self, configuraciones):
        """Crear gráfico simple de optimización"""
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(1, 1, 1)

            configs = [config['configuracion'] for config in configuraciones]
            precisiones = [config['precision'] for config in configuraciones]

            colores = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            barras = ax.bar(range(len(configs)), precisiones, color=colores[:len(configs)],
                            alpha=0.8, edgecolor='black', linewidth=0.5)

            ax.set_xlabel('Configuraciones')
            ax.set_ylabel('Precisión (%)')
            ax.set_title('⚙️ Comparación de Configuraciones', fontweight='bold')
            ax.set_xticks(range(len(configs)))
            ax.set_xticklabels([f'Config {i + 1}' for i in range(len(configs))], rotation=45)

            # Añadir valores en las barras
            for barra, valor in zip(barras, precisiones):
                height = barra.get_height()
                ax.text(barra.get_x() + barra.get_width() / 2., height + 0.5,
                        f'{valor:.1f}%', ha='center', va='bottom', fontweight='bold')

            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            print(f"Error en gráfico optimización: {e}")

    def crear_grafico_comparacion_simple(self, metodos):
        """Crear gráfico simple de comparación de métodos"""
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(1, 1, 1)

            nombres = [metodo['metodo'] for metodo in metodos]
            precisiones = [metodo['precision'] for metodo in metodos]

            colores = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            barras = ax.bar(nombres, precisiones, color=colores[:len(nombres)],
                            alpha=0.8, edgecolor='black', linewidth=0.5)

            ax.set_ylabel('Precisión (%)')
            ax.set_title('🔬 Comparación de Métodos de Análisis', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)

            # Añadir valores en las barras
            for barra, valor in zip(barras, precisiones):
                height = barra.get_height()
                ax.text(barra.get_x() + barra.get_width() / 2., height + 0.5,
                        f'{valor:.1f}%', ha='center', va='bottom', fontweight='bold')

            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            print(f"Error en gráfico comparación: {e}")

    def generar_recomendaciones_calidad(self, stats, distribucion):
        """Generar recomendaciones automáticas basadas en los datos"""
        recomendaciones = []

        # Análisis de pH
        ph_promedio = stats['ph_promedio']
        if ph_promedio < 6.5:
            recomendaciones.append("🔴 pH muy bajo: Considerar tratamiento de neutralización para elevar el pH")
        elif ph_promedio > 8.5:
            recomendaciones.append("🔴 pH muy alto: Implementar sistema de acidificación controlada")
        elif 6.5 <= ph_promedio <= 8.5:
            recomendaciones.append("✅ pH en rango aceptable: Mantener monitoreo regular")

        # Análisis de oxígeno
        oxigeno_promedio = stats['oxigeno_promedio']
        if oxigeno_promedio < 5.0:
            recomendaciones.append("🔴 Oxígeno bajo: Instalar sistemas de aireación o oxigenación")
        elif oxigeno_promedio < 6.0:
            recomendaciones.append("🟡 Oxígeno moderadamente bajo: Mejorar circulación del agua")
        else:
            recomendaciones.append("✅ Oxígeno adecuado: Niveles óptimos para vida acuática")

        # Análisis de turbidez
        turbidez_promedio = stats['turbidez_promedio']
        if turbidez_promedio > 4.0:
            recomendaciones.append("🔴 Turbidez alta: Implementar filtración y sedimentación")
        elif turbidez_promedio > 2.0:
            recomendaciones.append("🟡 Turbidez moderada: Monitorear fuentes de sedimentos")
        else:
            recomendaciones.append("✅ Turbidez baja: Agua clara, mantener prácticas actuales")

        # Análisis de conductividad
        conductividad_promedio = stats['conductividad_promedio']
        if conductividad_promedio > 1000:
            recomendaciones.append("🔴 Conductividad alta: Revisar fuentes de contaminación salina")
        elif conductividad_promedio < 200:
            recomendaciones.append("🟡 Conductividad baja: Agua muy pura, verificar mineralización")
        else:
            recomendaciones.append("✅ Conductividad normal: Niveles minerales apropiados")

        # Análisis de distribución de calidad
        total_muestras = sum(distribucion.values())
        pct_excelente = (distribucion.get('Excelente', 0) / total_muestras) * 100
        pct_necesita_tratamiento = (distribucion.get('Necesita Tratamiento', 0) / total_muestras) * 100

        if pct_excelente >= 80:
            recomendaciones.append("🌟 Excelente gestión: Mantener protocolos de calidad actuales")
        elif pct_necesita_tratamiento >= 25:
            recomendaciones.append("⚠️ Acción urgente: Implementar programa de mejora inmediato")
        elif pct_necesita_tratamiento >= 10:
            recomendaciones.append("🔧 Mejoras necesarias: Planificar optimización del sistema")

        # Recomendación de calidad promedio
        calidad_promedio = stats['calidad_promedio']
        if calidad_promedio >= 85:
            recomendaciones.append("🏆 Sistema excelente: Considerar como modelo de referencia")
        elif calidad_promedio >= 70:
            recomendaciones.append("👍 Sistema bueno: Pequeñas mejoras para optimización")
        elif calidad_promedio >= 50:
            recomendaciones.append("⚙️ Sistema regular: Implementar plan de mejora estructurado")
        else:
            recomendaciones.append("🚨 Sistema crítico: Requiere intervención inmediata y completa")

        return recomendaciones

    def actualizar_recomendaciones_calidad(self, stats, distribucion):
        """Actualizar el tab de recomendaciones con análisis detallado"""
        try:
            recomendaciones = self.generar_recomendaciones_calidad(stats, distribucion)

            html_recomendaciones = """
               <div style="font-family: Arial; font-size: 14px; padding: 20px;">
                   <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                               color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                       <h2>💡 Recomendaciones Inteligentes</h2>
                       <p>Análisis automático y sugerencias de mejora</p>
                   </div>

                   <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                       <h3 style="color: #2e7d32;">🎯 Acciones Recomendadas</h3>
               """

            for i, rec in enumerate(recomendaciones):
                icon_color = '#4CAF50' if '✅' in rec else '#FF9800' if '🟡' in rec else '#F44336'
                html_recomendaciones += f"""
                   <div style="background: white; padding: 12px; margin: 8px 0; border-radius: 6px; 
                               border-left: 4px solid {icon_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                       <p style="margin: 0; font-weight: 500;">{rec}</p>
                   </div>
                   """

            html_recomendaciones += """
                   </div>

                   <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                       <h3 style="color: #1565c0;">📋 Plan de Acción Sugerido</h3>
                       <ol style="padding-left: 20px;">
                           <li><strong>Inmediato (0-7 días):</strong> Abordar problemas críticos marcados en rojo</li>
                           <li><strong>Corto plazo (1-4 semanas):</strong> Implementar mejoras moderadas</li>
                           <li><strong>Mediano plazo (1-3 meses):</strong> Optimizar sistemas existentes</li>
                           <li><strong>Largo plazo (3+ meses):</strong> Establecer protocolos de mantenimiento</li>
                       </ol>
                   </div>

                   <div style="background: #fff3e0; padding: 15px; border-radius: 8px;">
                       <h3 style="color: #ef6c00;">🔄 Frecuencia de Monitoreo Recomendada</h3>
                       <p><strong>Parámetros críticos:</strong> Diario (pH, Oxígeno)</p>
                       <p><strong>Parámetros importantes:</strong> Semanal (Turbidez, Conductividad)</p>
                       <p><strong>Análisis completo:</strong> Mensual (Evaluación integral)</p>
                   </div>
               </div>
               """

            self.recomendaciones_content.setText(html_recomendaciones)

        except Exception as e:
            error_msg = f"❌ Error al generar recomendaciones: {str(e)}"
            print(error_msg)
            self.mostrar_error_en_pantalla("Error en Recomendaciones", error_msg)

    def actualizar_recomendaciones_calidad(self, stats, distribucion):
        """Actualizar el tab de recomendaciones con análisis detallado"""
        recomendaciones = self.generar_recomendaciones_calidad(stats, distribucion)

        html_recomendaciones = """
        <div style="font-family: Arial; font-size: 14px; padding: 20px;">
            <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h2>💡 Recomendaciones Inteligentes</h2>
                <p>Análisis automático y sugerencias de mejora</p>
            </div>

            <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                <h3 style="color: #2e7d32;">🎯 Acciones Recomendadas</h3>
        """

        for i, rec in enumerate(recomendaciones):
            icon_color = '#4CAF50' if '✅' in rec else '#FF9800' if '🟡' in rec else '#F44336'
            html_recomendaciones += f"""
            <div style="background: white; padding: 12px; margin: 8px 0; border-radius: 6px; 
                        border-left: 4px solid {icon_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="margin: 0; font-weight: 500;">{rec}</p>
            </div>
            """

        html_recomendaciones += """
            </div>

            <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                <h3 style="color: #1565c0;">📋 Plan de Acción Sugerido</h3>
                <ol style="padding-left: 20px;">
                    <li><strong>Inmediato (0-7 días):</strong> Abordar problemas críticos marcados en rojo</li>
                    <li><strong>Corto plazo (1-4 semanas):</strong> Implementar mejoras moderadas</li>
                    <li><strong>Mediano plazo (1-3 meses):</strong> Optimizar sistemas existentes</li>
                    <li><strong>Largo plazo (3+ meses):</strong> Establecer protocolos de mantenimiento</li>
                </ol>
            </div>

            <div style="background: #fff3e0; padding: 15px; border-radius: 8px;">
                <h3 style="color: #ef6c00;">🔄 Frecuencia de Monitoreo Recomendada</h3>
                <p><strong>Parámetros críticos:</strong> Diario (pH, Oxígeno)</p>
                <p><strong>Parámetros importantes:</strong> Semanal (Turbidez, Conductividad)</p>
                <p><strong>Análisis completo:</strong> Mensual (Evaluación integral)</p>
            </div>
        </div>
        """

        self.recomendaciones_content.setText(html_recomendaciones)

    # Función adicional para completar la función mostrar_comparacion
    def mostrar_comparacion_completa(self, resultados):
        """Mostrar comparación de métodos - función completa"""
        try:
            metodos = resultados['metodos']
            mejor_metodo = max(metodos, key=lambda x: x['precision'])

            resumen_html = f"""
            <div style="font-family: Arial; font-size: 14px; padding: 20px;">
                <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <h2>🔬 Comparación de Métodos</h2>
                    <p style="font-size: 16px;">Evaluación de diferentes enfoques de análisis</p>
                    <p><strong>Mejor método:</strong> {mejor_metodo['metodo']} ({mejor_metodo['precision']:.1f}%)</p>
                </div>
            """

            for metodo in metodos:
                es_mejor = metodo == mejor_metodo
                color = '#4CAF50' if es_mejor else '#2196F3'
                icon = '🏆' if es_mejor else '🔬'

                resumen_html += f"""
                <div style="background: {'#e8f5e8' if es_mejor else '#e3f2fd'}; 
                            border-left: 5px solid {color}; padding: 15px; margin: 15px 0; border-radius: 8px;">
                    <h3 style="color: {color}; margin-top: 0;">{icon} {metodo['metodo']}</h3>
                    <p><strong>Precisión:</strong> {metodo['precision']:.1f}%</p>
                    <p><strong>Ventajas:</strong> {metodo['ventajas']}</p>
                    {'<p style="color: #4CAF50; font-weight: bold;">✅ Método recomendado</p>' if es_mejor else ''}

                    <div style="background: #f0f0f0; height: 10px; border-radius: 5px; margin-top: 8px;">
                        <div style="background: {color}; height: 10px; width: {metodo['precision']}%; 
                                    border-radius: 5px; transition: width 0.3s ease;"></div>
                    </div>
                </div>
                """

            # Agregar análisis comparativo
            resumen_html += """
            <div style="background: #fff3e0; padding: 15px; border-radius: 8px; margin-top: 15px;">
                <h3 style="color: #ef6c00;">📊 Análisis Comparativo</h3>
                <p><strong>Conclusiones:</strong></p>
                <ul>
            """

            # Generar conclusiones automáticas
            precision_max = max(m['precision'] for m in metodos)
            precision_min = min(m['precision'] for m in metodos)
            diferencia = precision_max - precision_min

            if diferencia < 5:
                resumen_html += "<li>Los métodos muestran rendimiento similar, elegir por facilidad de uso</li>"
            elif diferencia < 15:
                resumen_html += "<li>Hay diferencias moderadas en precisión, considerar el método recomendado</li>"
            else:
                resumen_html += "<li>Diferencias significativas en rendimiento, usar el método óptimo</li>"

            if precision_max > 90:
                resumen_html += "<li>Excelente precisión general del sistema de análisis</li>"
            elif precision_max > 80:
                resumen_html += "<li>Buena precisión, sistema confiable para toma de decisiones</li>"
            else:
                resumen_html += "<li>Precisión moderada, considerar mejorar el conjunto de datos</li>"

            resumen_html += """
                </ul>
            </div>
            </div>
            """

            self.resumen_content.setText(resumen_html)

            # Crear gráfico de comparación
            if MATPLOTLIB_AVAILABLE:
                self.crear_grafico_comparacion_simple(metodos)

        except Exception as e:
            error_msg = f"❌ Error en mostrar_comparacion: {str(e)}"
            print(error_msg)
            self.resumen_content.setText(error_msg)

    # Función para añadir estilos CSS adicionales específicos para resultados
    def aplicar_estilos_resultados_adicionales(self):
        """Aplicar estilos CSS adicionales para mejorar la presentación de resultados"""
        estilos_adicionales = """
        /* Estilos específicos para resultados HTML */
        QLabel[objectName="resumenContent"] {
            background-color: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 0px;
        }

        QLabel[objectName="recomendacionesContent"] {
            background-color: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 0px;
        }

        /* Mejorar scroll areas */
        QScrollArea[objectName="resumen_scroll"] {
            background-color: #f8f9fa;
            border: 2px solid #dee2e6;
            border-radius: 8px;
        }

        QScrollArea[objectName="recomendaciones_scroll"] {
            background-color: #f8f9fa;
            border: 2px solid #dee2e6;
            border-radius: 8px;
        }

        /* Tabs mejorados */
        QTabWidget::tab-bar {
            alignment: center;
        }

        QTabBar::tab {
            min-width: 120px;
            padding: 12px 20px;
        }

        /* Canvas de matplotlib */
        QWidget[objectName="canvas"] {
            background-color: white;
            border: 1px solid #dee2e6;
            border-radius: 6px;
        }
        """

    def mostrar_calidad_agua(self, resultados):
        """Mostrar resultados de análisis de calidad del agua optimizado"""
        stats = resultados['estadisticas']
        distribucion = resultados['distribucion']

        # Resumen HTML optimizado

        def mostrar_agrupamiento(self, resultados):
            """Mostrar resultados de agrupamiento optimizado"""
            try:
                analisis = resultados['analisis_grupos']

                # HTML corregido
                resumen_html = """
                <div style="font-family: Arial; font-size: 14px; padding: 20px;">
                    <div style="text-align: center; background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); 
                                color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                        <h2>🔍 Agrupamiento de Muestras</h2>
                        <p style="font-size: 16px;">Se encontraron patrones similares usando clustering</p>
                    </div>
                """

                colores_grupos = ['#e8f5e8', '#e3f2fd', '#fff3e0']
                colores_bordes = ['#4CAF50', '#2196F3', '#FF9800']

                for i, (grupo, info) in enumerate(analisis.items()):
                    color_fondo = colores_grupos[i % len(colores_grupos)]
                    color_borde = colores_bordes[i % len(colores_bordes)]

                    resumen_html += f"""
                    <div style="background: {color_fondo}; border-left: 5px solid {color_borde}; 
                                padding: 15px; margin: 15px 0; border-radius: 8px;">
                        <h3 style="color: {color_borde}; margin-top: 0;">📊 {grupo}</h3>
                        <p><strong>Cantidad de muestras:</strong> {info['cantidad']}</p>
                        <p><strong>Calidad promedio:</strong> {info['calidad_promedio']:.1f}/100</p>
                        <p><strong>Características principales:</strong> {', '.join(info['caracteristicas'])}</p>
                    </div>
                    """

                resumen_html += "</div>"
                self.resumen_content.setText(resumen_html)

                # Crear gráfico de agrupamiento
                if MATPLOTLIB_AVAILABLE:
                    self.crear_grafico_agrupamiento_simple(resultados)

            except Exception as e:
                error_msg = f"❌ Error en mostrar_agrupamiento: {str(e)}"
                print(error_msg)
                self.resumen_content.setText(error_msg)

    def mostrar_calidad_agua(self, resultados):
        """Mostrar resultados de análisis de calidad del agua optimizado"""
        try:
            stats = resultados['estadisticas']
            distribucion = resultados['distribucion']

            # HTML corregido sin espacios en etiquetas
            resumen_html = f"""
            <div style="font-family: Arial; font-size: 14px; padding: 20px;">
                <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <h2>🧪 Análisis de Calidad del Agua</h2>
                    <h3>{resultados['estado_general']}</h3>
                    <p style="font-size: 16px;">{resultados['mensaje']}</p>
                </div>

                <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; border-left: 5px solid #4CAF50;">
                    <h3 style="color: #2e7d32; margin-top: 0;">📊 Estadísticas Generales</h3>
                    <p><strong>Total de muestras:</strong> {stats['total_muestras']}</p>
                    <p><strong>Calidad promedio:</strong> {stats['calidad_promedio']:.1f}/100</p>
                    <p><strong>pH promedio:</strong> {stats['ph_promedio']:.2f}</p>
                    <p><strong>Oxígeno:</strong> {stats['oxigeno_promedio']:.1f} mg/L</p>
                    <p><strong>Turbidez:</strong> {stats['turbidez_promedio']:.1f} NTU</p>
                    <p><strong>Conductividad:</strong> {stats['conductividad_promedio']:.1f} μS/cm</p>
                </div>

                <div style="background: #fff3e0; padding: 15px; border-radius: 8px; margin-top: 15px;">
                    <h3 style="color: #ef6c00;">📈 Distribución de Calidad</h3>
            """

            # Añadir distribución de calidad
            total_muestras = sum(distribucion.values())
            for calidad, cantidad in distribucion.items():
                porcentaje = (cantidad / total_muestras) * 100
                color_barra = {
                    'Excelente': '#4CAF50',
                    'Buena': '#8BC34A',
                    'Regular': '#FF9800',
                    'Necesita Tratamiento': '#F44336'
                }.get(calidad, '#9E9E9E')

                resumen_html += f"""
                    <div style="margin: 8px 0;">
                        <strong>{calidad}:</strong> {cantidad} muestras ({porcentaje:.1f}%)
                        <div style="background: #f0f0f0; height: 12px; border-radius: 6px; margin-top: 3px;">
                            <div style="background: {color_barra}; height: 12px; width: {porcentaje}%; 
                                        border-radius: 6px; transition: width 0.3s ease;"></div>
                        </div>
                    </div>
                """

            resumen_html += """
                </div>

                <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin-top: 15px;">
                    <h3 style="color: #1565c0;">💡 Recomendaciones</h3>
            """

            # Generar recomendaciones automáticas
            recomendaciones = self.generar_recomendaciones_calidad(stats, distribucion)
            for rec in recomendaciones:
                resumen_html += f"<p>• {rec}</p>"

            resumen_html += "</div></div>"

            self.resumen_content.setText(resumen_html)

            # Actualizar tab de recomendaciones
            self.actualizar_recomendaciones_calidad(stats, distribucion)

            # Crear gráficos optimizados
            if MATPLOTLIB_AVAILABLE:
                self.crear_graficos_calidad_optimizado(resultados)

        except Exception as e:
            error_msg = f"❌ Error en mostrar_calidad_agua: {str(e)}"
            print(error_msg)
            self.resumen_content.setText(error_msg)


class SegmentacionML(QWidget):
    """Versión optimizada de la aplicación principal"""

    def __init__(self):
        super().__init__()
        self.worker = None
        self.cache = DataCache()
        self.setup_ui()
        self.apply_styles()
        self.setup_connections()

        # Timer para limpieza de memoria
        self.cleanup_timer = QTimer()
        self.cleanup_timer.timeout.connect(self.cleanup_memory)
        self.cleanup_timer.start(60000)  # Cada minuto

    def cleanup_memory(self):
        """Limpieza periódica de memoria"""
        gc.collect()

    def setup_ui(self):
        self.setWindowTitle("💧 Análisis Inteligente de Calidad del Agua - Optimizado")
        self.setMinimumSize(1200, 800)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Título
        title = QLabel("💧 Sistema Optimizado de Análisis de Calidad del Agua")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        subtitle = QLabel("Procesamiento paralelo e inteligencia artificial para análisis eficiente")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle)

        # Splitter horizontal
        splitter = QSplitter(Qt.Horizontal)

        # Panel izquierdo - Controles
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)

        # Panel derecho - Resultados visuales
        self.resultados_widget = ResultadosVisuales()
        splitter.addWidget(self.resultados_widget)

        splitter.setSizes([400, 800])
        main_layout.addWidget(splitter)

        # Barra de estado
        status_layout = QHBoxLayout()

        self.status_label = QLabel("✅ Sistema optimizado listo")
        self.status_label.setObjectName("statusLabel")
        status_layout.addWidget(self.status_label)

        # Información del sistema
        cpu_count = multiprocessing.cpu_count()
        system_info = QLabel(f"🖥️ CPUs disponibles: {cpu_count} | Modo: Multiproceso")
        system_info.setObjectName("systemInfo")
        status_layout.addWidget(system_info)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("progressBar")
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(300)
        status_layout.addWidget(self.progress_bar)

        main_layout.addLayout(status_layout)
        self.setLayout(main_layout)

    def create_control_panel(self):
        """Crear panel de controles optimizado"""
        group = QGroupBox("🎛️ Análisis Disponibles")
        layout = QVBoxLayout()
        layout.setSpacing(15)

        # Información del sistema optimizado
        info_frame = QFrame()
        info_frame.setObjectName("infoFrame")
        info_layout = QVBoxLayout()

        info_title = QLabel("⚡ Sistema Optimizado")
        info_title.setObjectName("infoTitle")
        info_layout.addWidget(info_title)

        info_text = QLabel("""
        Sistema optimizado con procesamiento paralelo y cache inteligente:

        ✅ Multiprocesamiento para análisis pesados
        ✅ Cache automático de resultados
        ✅ Gestión optimizada de memoria
        ✅ Timeouts para prevenir bloqueos
        """)
        info_text.setObjectName("infoText")
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)

        info_frame.setLayout(info_layout)
        layout.addWidget(info_frame)

        # Análisis optimizados
        analisis = [
            ("🔬 Análisis Rápido de Calidad", "calidad_agua",
             "Evaluación optimizada de parámetros del agua"),
            ("🔍 Clustering Inteligente", "agrupar_muestras",
             "Agrupamiento acelerado con algoritmos optimizados"),
            ("🤖 Predicción ML Eficiente", "predecir_calidad",
             "Modelo predictivo con procesamiento paralelo"),
            ("⚙️ Optimización Automática", "optimizar_sistema",
             "Búsqueda optimizada de mejores parámetros"),
            ("📊 Comparación Rápida", "comparar_metodos",
             "Evaluación eficiente de múltiples algoritmos")
        ]

        self.analysis_buttons = {}

        # Grid layout optimizado
        buttons_widget = QWidget()
        buttons_layout = QGridLayout(buttons_widget)
        buttons_layout.setSpacing(10)

        for i, (name, key, description) in enumerate(analisis):
            btn_frame = QFrame()
            btn_frame.setObjectName("buttonFrame")
            btn_frame.setMinimumHeight(100)
            btn_frame.setMaximumHeight(120)

            btn_layout = QVBoxLayout(btn_frame)
            btn_layout.setSpacing(6)
            btn_layout.setContentsMargins(12, 12, 12, 12)

            # Botón optimizado
            btn = QPushButton(name)
            btn.setObjectName("analysisBtn")
            btn.setMinimumHeight(40)
            btn.setMaximumHeight(45)
            btn.clicked.connect(lambda checked, k=key: self.run_analysis_optimized(k))

            # Descripción compacta
            desc_label = QLabel(description)
            desc_label.setObjectName("descLabel")
            desc_label.setWordWrap(True)
            desc_label.setAlignment(Qt.AlignTop)

            # Indicador de estado
            status_indicator = QLabel("⏳")
            status_indicator.setObjectName("statusIndicator")
            status_indicator.setVisible(False)
            status_indicator.setAlignment(Qt.AlignCenter)
            status_indicator.setMaximumHeight(20)

            btn_layout.addWidget(btn, 0)
            btn_layout.addWidget(desc_label, 1)
            btn_layout.addWidget(status_indicator, 0)

            row = i // 2
            col = i % 2
            buttons_layout.addWidget(btn_frame, row, col)

            self.analysis_buttons[key] = btn
            btn.status_indicator = status_indicator

        layout.addWidget(buttons_widget)
        layout.addStretch()

        # Controles de utilidad
        utility_frame = QFrame()
        utility_frame.setObjectName("utilityFrame")
        utility_layout = QHBoxLayout(utility_frame)
        utility_layout.setSpacing(10)

        clear_cache_btn = QPushButton("🗑️ Limpiar Cache")
        clear_cache_btn.setObjectName("clearBtn")
        clear_cache_btn.clicked.connect(self.clear_cache)

        clear_results_btn = QPushButton("📄 Limpiar Resultados")
        clear_results_btn.setObjectName("clearBtn")
        clear_results_btn.clicked.connect(self.clear_results)

        memory_btn = QPushButton("🧹 Liberar Memoria")
        memory_btn.setObjectName("memoryBtn")
        memory_btn.clicked.connect(self.force_cleanup)

        utility_layout.addWidget(clear_cache_btn)
        utility_layout.addWidget(clear_results_btn)
        utility_layout.addWidget(memory_btn)
        utility_layout.addStretch()

        layout.addWidget(utility_frame)

        # Separador
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # Botón regresar
        self.btn_regresar = QPushButton("← Regresar al Menú Principal")
        self.btn_regresar.setObjectName("backBtn")
        layout.addWidget(self.btn_regresar)

        group.setLayout(layout)
        return group

    def setup_connections(self):
        """Configurar conexiones optimizadas"""
        pass

    def run_analysis_optimized(self, analysis_type):
        """Ejecutar análisis optimizado"""
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Procesamiento en Curso",
                                    "Hay un análisis ejecutándose. El sistema está optimizado "
                                    "para procesar de manera eficiente.")
            return

        # Verificar disponibilidad del sistema
        if not SKLEARN_AVAILABLE:
            QMessageBox.warning(self, "Sistema No Disponible",
                                "Las librerías de machine learning no están disponibles.\n"
                                "Instala: pip install scikit-learn pandas numpy")
            return

        # Obtener botón activo
        sender_btn = self.analysis_buttons[analysis_type]

        # Mostrar indicador de procesamiento
        if hasattr(sender_btn, 'status_indicator'):
            sender_btn.status_indicator.setText("⚡")
            sender_btn.status_indicator.setVisible(True)

        # Deshabilitar botones temporalmente
        for btn in self.analysis_buttons.values():
            btn.setEnabled(False)

        # Mostrar progreso
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Actualizar estado
        analysis_names = {
            "calidad_agua": "🔬 Procesando análisis de calidad...",
            "agrupar_muestras": "🔍 Ejecutando clustering...",
            "predecir_calidad": "🤖 Entrenando modelo ML...",
            "optimizar_sistema": "⚙️ Optimizando parámetros...",
            "comparar_metodos": "📊 Comparando algoritmos..."
        }

        self.status_label.setText(analysis_names.get(analysis_type, "⏳ Procesando..."))

        # Crear worker optimizado
        self.worker = OptimizedMLWorker(analysis_type)
        self.worker.finished.connect(
            lambda results, atype: self.on_analysis_finished_optimized(results, atype, sender_btn))
        self.worker.progress.connect(self.update_progress)
        self.worker.status_update.connect(self.status_label.setText)
        self.worker.start()

    def update_progress(self, value):
        """Actualizar progreso de manera optimizada"""
        self.progress_bar.setValue(value)

        # Actualizar indicador visual del botón activo
        for btn in self.analysis_buttons.values():
            if hasattr(btn, 'status_indicator') and btn.status_indicator.isVisible():
                if value < 30:
                    btn.status_indicator.setText("⏳")
                elif value < 70:
                    btn.status_indicator.setText("⚡")
                else:
                    btn.status_indicator.setText("✅")
                break

    def on_analysis_finished_optimized(self, results, analysis_type, sender_btn):
        """Manejar finalización optimizada"""
        try:
            # Restaurar estado de botones
            for btn in self.analysis_buttons.values():
                btn.setEnabled(True)
                if hasattr(btn, 'status_indicator'):
                    btn.status_indicator.setVisible(False)

            # Ocultar progreso
            self.progress_bar.setVisible(False)

            print(
                f"📋 Resultados recibidos para {analysis_type}: {list(results.keys()) if isinstance(results, dict) else 'No dict'}")

            if "error" in results:
                self.status_label.setText(f"❌ Error: {results['error']}")
                QMessageBox.critical(self, "Error de Procesamiento",
                                     f"Error durante el análisis:\n{results['error']}\n\n"
                                     f"El sistema está optimizado para manejar errores de manera eficiente.")
                return

            # Mostrar resultados
            self.resultados_widget.mostrar_resultados(results, analysis_type)
            self.status_label.setText("✅ Análisis completado con éxito")

            # Notificación de éxito optimizada
            analysis_names = {
                "calidad_agua": "Análisis de Calidad",
                "agrupar_muestras": "Clustering de Datos",
                "predecir_calidad": "Predicción ML",
                "optimizar_sistema": "Optimización",
                "comparar_metodos": "Comparación de Métodos"
            }

            QMessageBox.information(self, "Procesamiento Completado",
                                    f"✅ {analysis_names.get(analysis_type, 'Análisis')} completado exitosamente.\n\n"
                                    f"🚀 Procesamiento optimizado ejecutado\n"
                                    f"📊 Resultados disponibles en las pestañas")

        except Exception as e:
            error_msg = f"❌ Error en finalización: {str(e)}"
            print(error_msg)
            self.status_label.setText(error_msg)

            # Restaurar botones incluso si hay error
            for btn in self.analysis_buttons.values():
                btn.setEnabled(True)
                if hasattr(btn, 'status_indicator'):
                    btn.status_indicator.setVisible(False)
            self.progress_bar.setVisible(False)

            def crear_graficos_calidad_optimizado(self, resultados):
                """Crear gráficos optimizados para calidad del agua"""
                try:
                    self.figure.clear()

                    # Convertir datos si es necesario
                    if isinstance(resultados['datos'], list):
                        datos = pd.DataFrame(resultados['datos'])
                    else:
                        datos = resultados['datos']

                    distribucion = resultados['distribucion']

                    # Crear subplots
                    ax1 = self.figure.add_subplot(2, 2, 1)  # Distribución de calidad
                    ax2 = self.figure.add_subplot(2, 2, 2)  # Parámetros promedio
                    ax3 = self.figure.add_subplot(2, 2, 3)  # Histograma pH
                    ax4 = self.figure.add_subplot(2, 2, 4)  # Scatter plot

                    # Gráfico 1: Distribución de calidad (pie chart)
                    colores = ['#4CAF50', '#8BC34A', '#FF9800', '#F44336']
                    labels = list(distribucion.keys())
                    sizes = list(distribucion.values())

                    if sizes and sum(sizes) > 0:
                        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                                                           colors=colores[:len(labels)], startangle=90)
                        ax1.set_title('📊 Distribución de Calidad', fontsize=10, fontweight='bold')
                        # Mejorar legibilidad
                        for autotext in autotexts:
                            autotext.set_color('white')
                            autotext.set_fontweight('bold')
                    else:
                        ax1.text(0.5, 0.5, 'Sin datos\npara mostrar', ha='center', va='center')
                        ax1.set_title('📊 Distribución de Calidad', fontsize=10, fontweight='bold')

                    # Gráfico 2: Parámetros promedio (bar chart)
                    try:
                        if hasattr(datos, 'mean'):
                            parametros = ['pH', 'Oxígeno', 'Turbidez', 'Conductividad']
                            valores = [
                                datos['pH'].mean(),
                                datos['Oxígeno_Disuelto'].mean(),
                                datos['Turbidez'].mean(),
                                datos['Conductividad'].mean() / 100  # Escalar para visualización
                            ]
                        else:
                            # Procesar lista de diccionarios
                            ph_vals = [d.get('pH', 0) for d in datos]
                            ox_vals = [d.get('Oxígeno_Disuelto', 0) for d in datos]
                            turb_vals = [d.get('Turbidez', 0) for d in datos]
                            cond_vals = [d.get('Conductividad', 0) for d in datos]

                            parametros = ['pH', 'Oxígeno', 'Turbidez', 'Conductividad']
                            valores = [
                                sum(ph_vals) / len(ph_vals) if ph_vals else 0,
                                sum(ox_vals) / len(ox_vals) if ox_vals else 0,
                                sum(turb_vals) / len(turb_vals) if turb_vals else 0,
                                (sum(cond_vals) / len(cond_vals)) / 100 if cond_vals else 0
                            ]

                        colores_bar = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
                        barras = ax2.bar(parametros, valores, color=colores_bar, alpha=0.8, edgecolor='black',
                                         linewidth=0.5)
                        ax2.set_title('📈 Parámetros Promedio', fontsize=10, fontweight='bold')
                        ax2.set_ylabel('Valor')
                        ax2.tick_params(axis='x', rotation=45)

                        # Añadir valores en las barras
                        for barra, valor in zip(barras, valores):
                            if valor > 0:
                                height = barra.get_height()
                                ax2.text(barra.get_x() + barra.get_width() / 2., height + height * 0.02,
                                         f'{valor:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

                    except Exception as e:
                        ax2.text(0.5, 0.5, f'Error:\n{str(e)[:30]}...', ha='center', va='center')
                        ax2.set_title('📈 Parámetros Promedio', fontsize=10, fontweight='bold')

                    # Gráfico 3: Histograma de pH
                    try:
                        if hasattr(datos, 'hist'):
                            ph_data = datos['pH'].dropna()
                        else:
                            ph_data = [d.get('pH', 7) for d in datos if d.get('pH')]

                        if len(ph_data) > 0:
                            ax3.hist(ph_data, bins=15, alpha=0.7, color='#3F51B5', edgecolor='black', linewidth=0.5)
                            ax3.axvline(x=7, color='red', linestyle='--', linewidth=2, label='pH Neutro')
                            ax3.axvspan(6.5, 8.5, alpha=0.2, color='green', label='Rango Aceptable')
                            ax3.set_title('📊 Distribución de pH', fontsize=10, fontweight='bold')
                            ax3.set_xlabel('pH')
                            ax3.set_ylabel('Frecuencia')
                            ax3.legend(fontsize=8)
                        else:
                            ax3.text(0.5, 0.5, 'Sin datos de pH', ha='center', va='center')
                            ax3.set_title('📊 Distribución de pH', fontsize=10, fontweight='bold')
                    except Exception as e:
                        ax3.text(0.5, 0.5, f'Error pH:\n{str(e)[:20]}', ha='center', va='center')
                        ax3.set_title('📊 Distribución de pH', fontsize=10, fontweight='bold')

                    # Gráfico 4: Scatter plot Oxígeno vs Turbidez
                    try:
                        if hasattr(datos, 'plot'):
                            ox_data = datos['Oxígeno_Disuelto'].values
                            turb_data = datos['Turbidez'].values
                            calidad_scores = datos['Calidad_Score'].values
                        else:
                            ox_data = [d.get('Oxígeno_Disuelto', 8) for d in datos]
                            turb_data = [d.get('Turbidez', 2) for d in datos]
                            calidad_scores = [d.get('Calidad_Score', 75) for d in datos]

                        if len(ox_data) > 0 and len(turb_data) > 0:
                            scatter = ax4.scatter(ox_data, turb_data, c=calidad_scores,
                                                  cmap='RdYlGn', alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
                            ax4.set_title('💧 Oxígeno vs Turbidez', fontsize=10, fontweight='bold')
                            ax4.set_xlabel('Oxígeno Disuelto (mg/L)')
                            ax4.set_ylabel('Turbidez (NTU)')

                            # Añadir colorbar
                            cbar = self.figure.colorbar(scatter, ax=ax4, shrink=0.8)
                            cbar.set_label('Calidad Score', rotation=270, labelpad=15, fontsize=8)
                        else:
                            ax4.text(0.5, 0.5, 'Sin datos suficientes', ha='center', va='center')
                            ax4.set_title('💧 Oxígeno vs Turbidez', fontsize=10, fontweight='bold')
                    except Exception as e:
                        ax4.text(0.5, 0.5, f'Error scatter:\n{str(e)[:20]}', ha='center', va='center')
                        ax4.set_title('💧 Oxígeno vs Turbidez', fontsize=10, fontweight='bold')

                    self.figure.suptitle('🔬 Análisis Completo de Calidad del Agua', fontsize=12, fontweight='bold')
                    self.figure.tight_layout()
                    self.canvas.draw()

                except Exception as e:
                    print(f"❌ Error completo en gráficos: {e}")
                    self.figure.clear()
                    ax = self.figure.add_subplot(1, 1, 1)
                    ax.text(0.5, 0.5, f'❌ Error al crear gráficos:\n{str(e)}',
                            ha='center', va='center', transform=ax.transAxes, fontsize=12,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
                    ax.set_title('Error en Visualización', fontsize=14, fontweight='bold')
                    self.canvas.draw()

    def clear_cache(self):
        """Limpiar cache del sistema"""
        reply = QMessageBox.question(self, "Limpiar Cache",
                                     "¿Limpiar el cache del sistema?\n\n"
                                     "Esto mejorará el uso de memoria pero los próximos "
                                     "análisis tomarán más tiempo.",
                                     QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.cache.clear()
            self.status_label.setText("🗑️ Cache limpiado - Memoria optimizada")

    def clear_results(self):
        """Limpiar resultados visuales"""
        self.resultados_widget.resumen_content.setText("Ejecuta un análisis para ver los resultados")
        self.resultados_widget.recomendaciones_content.setText("Las recomendaciones aparecerán aquí")

        if MATPLOTLIB_AVAILABLE:
            self.resultados_widget.figure.clear()
            self.resultados_widget.canvas.draw()

        self.status_label.setText("📄 Resultados limpiados")

    def force_cleanup(self):
        """Forzar limpieza de memoria"""
        self.cache.clear()
        gc.collect()
        self.status_label.setText("🧹 Limpieza de memoria completada")

        QMessageBox.information(self, "Limpieza Completada",
                                "🧹 Limpieza de memoria ejecutada:\n\n"
                                "✅ Cache limpiado\n"
                                "✅ Garbage collection ejecutado\n"
                                "✅ Memoria optimizada")

    def apply_styles(self):
        """Aplicar estilos optimizados"""
        styles = """
        QWidget {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', 'Arial', sans-serif;
        }

        #title {
            font-size: 24px;
            font-weight: bold;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #4a90e2, stop:0.5 #357abd, stop:1 #1e5f99);
            color: white;
            padding: 18px;
            border-radius: 10px;
            margin-bottom: 8px;
        }

        #subtitle {
            font-size: 13px;
            color: #7f8c8d;
            font-style: italic;
            margin-bottom: 15px;
            padding: 5px;
        }

        #systemInfo {
            font-size: 11px;
            color: #27ae60;
            font-weight: bold;
            background-color: #e8f5e8;
            padding: 8px 12px;
            border-radius: 6px;
            border-left: 3px solid #27ae60;
        }

        QGroupBox {
            font-weight: bold;
            border: 2px solid #ddd;
            border-radius: 8px;
            margin-top: 15px;
            padding-top: 20px;
            background-color: white;
            font-size: 13px;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 10px 0 10px;
            color: #2c3e50;
            font-size: 14px;
            font-weight: bold;
        }

        #infoFrame {
            background-color: #e8f4fd;
            border: 2px solid #b3d9ff;
            border-radius: 6px;
            padding: 12px;
            margin: 5px;
        }

        #infoTitle {
            font-weight: bold;
            color: #2980b9;
            font-size: 13px;
            margin-bottom: 6px;
        }

        #infoText {
            color: #34495e;
            font-size: 11px;
            line-height: 1.4;
        }

        #buttonFrame {
            background-color: #ffffff;
            border: 2px solid #e1e8ed;
            border-radius: 8px;
            padding: 10px;
            margin: 3px;
        }

        #buttonFrame:hover {
            border-color: #4a90e2;
            background-color: #f8fcff;
            box-shadow: 0 2px 8px rgba(74, 144, 226, 0.1);
        }

        #analysisBtn {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #4a90e2, stop:1 #357abd);
            color: white;
            border: none;
            padding: 10px 12px;
            font-size: 12px;
            font-weight: bold;
            border-radius: 6px;
        }

        #analysisBtn:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #5ba0f2, stop:1 #4a90e2);
        }

        #analysisBtn:disabled {
            background-color: #bdc3c7;
            color: #7f8c8d;
        }

        #descLabel {
            color: #5a6c7d;
            font-size: 10px;
            font-weight: normal;
            line-height: 1.3;
        }

        #statusIndicator {
            color: #27ae60;
            font-size: 16px;
            font-weight: bold;
        }

        #utilityFrame {
            background-color: #f8f9fa;
            border-radius: 6px;
            padding: 8px;
            margin: 5px 0;
        }

        #clearBtn, #memoryBtn {
            background-color: #6c757d;
            color: white;
            border: none;
            padding: 6px 12px;
            font-weight: bold;
            border-radius: 4px;
            font-size: 11px;
            min-width: 100px;
        }

        #clearBtn:hover, #memoryBtn:hover {
            background-color: #5a6268;
        }

        #memoryBtn {
            background-color: #28a745;
        }

        #memoryBtn:hover {
            background-color: #218838;
        }

        #backBtn {
            background-color: #6f7d8c;
            color: white;
            border: none;
            padding: 10px 16px;
            font-weight: bold;
            border-radius: 6px;
            font-size: 12px;
        }

        #backBtn:hover {
            background-color: #5a6c7d;
        }

        #statusLabel {
            background-color: #ffffff;
            color: #2c3e50;
            padding: 10px 16px;
            border-radius: 6px;
            border-left: 4px solid #4a90e2;
            font-weight: bold;
            font-size: 12px;
        }

        #progressBar {
            border: 2px solid #ddd;
            border-radius: 6px;
            text-align: center;
            font-weight: bold;
            font-size: 11px;
            height: 22px;
        }

        #progressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #27ae60, stop:1 #2ecc71);
            border-radius: 4px;
        }

        #resultTitle {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            padding: 12px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #f8f9fa, stop:1 #e9ecef);
            border-radius: 6px;
            margin-bottom: 8px;
        }

        QTabWidget::pane {
            border: 2px solid #ddd;
            background-color: white;
            border-radius: 6px;
            padding: 5px;
        }

        QTabBar::tab {
            background-color: #ecf0f1;
            color: #2c3e50;
            padding: 10px 20px;
            margin-right: 2px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            font-weight: bold;
            font-size: 12px;
        }

        QTabBar::tab:selected {
            background-color: #4a90e2;
            color: white;
        }

        QTabBar::tab:hover:!selected {
            background-color: #d5dbdb;
        }

        #resumenContent, #recomendacionesContent {
            background-color: white;
            padding: 12px;
            border-radius: 6px;
            font-size: 12px;
            line-height: 1.4;
        }

        QScrollArea {
            border: 1px solid #ddd;
            border-radius: 6px;
            background-color: white;
        }

        QScrollBar:vertical {
            background-color: #f1f1f1;
            width: 10px;
            border-radius: 5px;
        }

        QScrollBar::handle:vertical {
            background-color: #c1c1c1;
            border-radius: 5px;
            min-height: 20px;
        }

        QScrollBar::handle:vertical:hover {
            background-color: #a8a8a8;
        }
        """

        self.setStyleSheet(styles)

    def closeEvent(self, event):
        """Manejar cierre de aplicación de manera optimizada"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(self, "Cerrar Aplicación",
                                         "Hay un proceso ejecutándose. ¿Deseas cerrarlo?",
                                         QMessageBox.Yes | QMessageBox.No)

            if reply == QMessageBox.Yes:
                self.worker.terminate()
                self.worker.wait(3000)  # Esperar máximo 3 segundos
            else:
                event.ignore()
                return

        # Limpieza final
        self.cache.clear()
        self.cleanup_timer.stop()
        gc.collect()
        event.accept()


# Ejemplo de uso optimizado
if __name__ == "__main__":
    # Configurar multiprocesamiento para Windows
    multiprocessing.set_start_method('spawn', force=True)

    app = QApplication(sys.argv)

    # Verificar dependencias
    missing_deps = []
    if not SKLEARN_AVAILABLE:
        missing_deps.append("scikit-learn, pandas, numpy")
    if not MATPLOTLIB_AVAILABLE:
        missing_deps.append("matplotlib")

    if missing_deps:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Dependencias Faltantes")
        msg.setText("Faltan las siguientes dependencias:")
        msg.setInformativeText("\n".join(missing_deps))
        msg.setDetailedText("Instala con:\npip install scikit-learn pandas numpy matplotlib")
        msg.exec_()

    window = SegmentacionML()
    window.show()

    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        window.close()
