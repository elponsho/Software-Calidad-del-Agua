import sys
import os
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                             QHBoxLayout, QScrollArea, QFrame, QProgressBar,
                             QSplitter, QGroupBox, QGridLayout, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont

# Agregar el directorio actual al path para importaciones
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Importar sistema de temas
try:
    from darkmode.theme_manager import ThemedWidget, ThemeManager
except ImportError:
    try:
        from ..darkmode.theme_manager import ThemedWidget, ThemeManager
    except ImportError:
        # Fallback si no existe darkmode
        class ThemedWidget:
            def __init__(self):
                pass
        class ThemeManager:
            @staticmethod
            def toggle_theme():
                pass
            @staticmethod
            def is_dark_theme():
                return False

# Importar funciones ML con múltiples intentos de ruta
ML_FUNCTIONS_AVAILABLE = False
ML_FUNCTIONS = None

# Intento 1: Importación directa (si está en el mismo directorio)
try:
    from ml_functions_supervisado import (
        regresion_multiple_proceso, svm_proceso,
        random_forest_proceso, regresion_lineal_proceso
    )
    ML_FUNCTIONS_AVAILABLE = True
    print("✅ Funciones ML cargadas desde importación directa")
except ImportError as e:
    print(f"⚠️ Intento 1 falló: {e}")

# Intento 2: Importación relativa (si está en la misma carpeta)
if not ML_FUNCTIONS_AVAILABLE:
    try:
        from .ml_functions_supervisado import (
            regresion_multiple_proceso, svm_proceso,
            random_forest_proceso, regresion_lineal_proceso
        )
        ML_FUNCTIONS_AVAILABLE = True
        print("✅ Funciones ML cargadas desde importación relativa")
    except ImportError as e:
        print(f"⚠️ Intento 2 falló: {e}")

# Intento 3: Buscar el archivo en el directorio actual
if not ML_FUNCTIONS_AVAILABLE:
    try:
        ml_file_path = os.path.join(current_dir, 'ml_functions_supervisado.py')
        if os.path.exists(ml_file_path):
            # Importar usando importlib
            import importlib.util
            spec = importlib.util.spec_from_file_location("ml_functions_supervisado", ml_file_path)
            ml_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ml_module)

            # Extraer las funciones
            regresion_multiple_proceso = ml_module.regresion_multiple_proceso
            svm_proceso = ml_module.svm_proceso
            random_forest_proceso = ml_module.random_forest_proceso
            regresion_lineal_proceso = ml_module.regresion_lineal_proceso

            ML_FUNCTIONS_AVAILABLE = True
            print("✅ Funciones ML cargadas usando importlib")
        else:
            print(f"❌ Archivo no encontrado en: {ml_file_path}")
    except Exception as e:
        print(f"⚠️ Intento 3 falló: {e}")

# Intento 4: Buscar en carpeta padre
if not ML_FUNCTIONS_AVAILABLE:
    try:
        parent_dir = os.path.dirname(current_dir)
        ml_file_path = os.path.join(parent_dir, 'ml_functions_supervisado.py')
        if os.path.exists(ml_file_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("ml_functions_supervisado", ml_file_path)
            ml_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ml_module)

            regresion_multiple_proceso = ml_module.regresion_multiple_proceso
            svm_proceso = ml_module.svm_proceso
            random_forest_proceso = ml_module.random_forest_proceso
            regresion_lineal_proceso = ml_module.regresion_lineal_proceso

            ML_FUNCTIONS_AVAILABLE = True
            print("✅ Funciones ML cargadas desde carpeta padre")
        else:
            print(f"❌ Archivo no encontrado en carpeta padre: {ml_file_path}")
    except Exception as e:
        print(f"⚠️ Intento 4 falló: {e}")

# Verificar dependencias ML
try:
    import numpy as np
    import pandas as pd
    import sklearn
    ML_DEPS_AVAILABLE = True
    print("✅ Dependencias ML disponibles")
except ImportError as e:
    print(f"❌ Dependencias ML faltantes: {e}")
    ML_DEPS_AVAILABLE = False

# Mostrar estado final
if ML_FUNCTIONS_AVAILABLE and ML_DEPS_AVAILABLE:
    print("🎉 Sistema ML completamente disponible")
elif ML_FUNCTIONS_AVAILABLE:
    print("⚠️ Funciones disponibles pero faltan dependencias")
elif ML_DEPS_AVAILABLE:
    print("⚠️ Dependencias OK pero faltan funciones ML")
else:
    print("❌ Sistema ML no disponible")

# Importar otros módulos
try:
    from resultados_visuales import ResultadosVisuales
    print("✅ ResultadosVisuales importado")
except ImportError:
    try:
        from .resultados_visuales import ResultadosVisuales
        print("✅ ResultadosVisuales importado (relativo)")
    except ImportError:
        print("⚠️ Usando ResultadosVisuales simplificado")
        # Crear una versión simple si no existe
        class ResultadosVisuales(QWidget):
            def __init__(self):
                super().__init__()
                self.setup_ui()

            def setup_ui(self):
                layout = QVBoxLayout()
                self.label = QLabel("📊 Panel de Resultados\n\nLos resultados aparecerán aquí cuando ejecutes un análisis.")
                self.label.setAlignment(Qt.AlignCenter)
                self.label.setStyleSheet("font-size: 14px; color: #666; padding: 20px;")
                layout.addWidget(self.label)
                self.setLayout(layout)

            def mostrar_resultados(self, results, algorithm):
                text = f"✅ Análisis completado: {algorithm}\n\n"
                if 'error' in results:
                    text += f"❌ Error: {results['error']}"
                else:
                    text += "📈 Resultados generados correctamente\n"
                    if 'r2_score' in results:
                        text += f"📊 R² Score: {results['r2_score']:.3f}\n"
                    if 'accuracy' in results:
                        text += f"🎯 Precisión: {results['accuracy']:.1f}%\n"
                    text += f"🔍 Análisis completado exitosamente"
                self.label.setText(text)

            def limpiar_resultados(self):
                self.label.setText("📊 Panel de Resultados\n\nLos resultados aparecerán aquí cuando ejecutes un análisis.")

try:
    from data_cache import DataCache
    print("✅ DataCache importado")
except ImportError:
    try:
        from .data_cache import DataCache
        print("✅ DataCache importado (relativo)")
    except ImportError:
        print("⚠️ Usando DataCache simplificado")
        # Crear una versión simple si no existe
        class DataCache:
            def __init__(self):
                self.cache = {}

            def get(self, key):
                return self.cache.get(key)

            def set(self, key, value):
                self.cache[key] = value

            def clear(self):
                self.cache.clear()

import gc


class SupervisadoWorker(QThread):
    """Worker thread para análisis supervisado"""
    finished = pyqtSignal(dict, str)
    progress = pyqtSignal(int)
    status_update = pyqtSignal(str)

    def __init__(self, algorithm):
        super().__init__()
        self.algorithm = algorithm
        self.cache = DataCache()

    def run(self):
        try:
            # Verificar que las funciones ML estén disponibles
            if not ML_FUNCTIONS_AVAILABLE:
                self.finished.emit({
                    "error": "Funciones ML no disponibles. Verifica que ml_functions_supervisado.py esté en el directorio correcto."
                }, "error")
                return

            if not ML_DEPS_AVAILABLE:
                self.finished.emit({
                    "error": "Dependencias ML faltantes. Instala: pip install scikit-learn pandas numpy"
                }, "error")
                return

            # Verificar cache
            cache_key = f"supervisado_{self.algorithm}"
            cached_result = self.cache.get(cache_key)

            if cached_result:
                self.status_update.emit("📦 Usando datos en cache...")
                self.progress.emit(100)
                self.finished.emit(cached_result, self.algorithm)
                return

            self.status_update.emit("🚀 Iniciando análisis supervisado...")
            self.progress.emit(10)

            # Mapeo de funciones reales
            process_functions = {
                "regresion_multiple": regresion_multiple_proceso,
                "svm": svm_proceso,
                "random_forest": random_forest_proceso,
                "regresion_lineal": regresion_lineal_proceso
            }

            if self.algorithm not in process_functions:
                self.finished.emit({"error": "Algoritmo no reconocido"}, "error")
                return

            self.progress.emit(30)
            self.status_update.emit(f"🔬 Ejecutando {self.algorithm}...")

            # Ejecutar función directamente
            try:
                print(f"🔄 Ejecutando función: {self.algorithm}")
                result = process_functions[self.algorithm]()
                self.progress.emit(95)

                if 'error' not in result:
                    self.cache.set(cache_key, result)
                    self.status_update.emit("✅ Análisis completado exitosamente")
                    print(f"✅ {self.algorithm} completado exitosamente")
                else:
                    self.status_update.emit(f"❌ Error en análisis: {result['error']}")
                    print(f"❌ Error en {self.algorithm}: {result['error']}")

                self.progress.emit(100)
                self.finished.emit(result, self.algorithm)

            except Exception as e:
                error_msg = f"Error ejecutando {self.algorithm}: {str(e)}"
                print(f"❌ {error_msg}")
                self.finished.emit({"error": error_msg}, "error")

        except Exception as e:
            self.finished.emit({"error": f"Error en worker: {str(e)}"}, "error")
        finally:
            gc.collect()


class SupervisadoWindow(QWidget, ThemedWidget):
    """Ventana de Aprendizaje Supervisado"""

    def __init__(self):
        QWidget.__init__(self)
        ThemedWidget.__init__(self)

        self.worker = None
        self.cache = DataCache()
        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):
        """Configurar interfaz de usuario"""
        self.setWindowTitle("🎯 Aprendizaje Supervisado - ML Calidad del Agua")
        self.setMinimumSize(1200, 800)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header_layout = self.create_header()
        main_layout.addLayout(header_layout)

        # Splitter principal
        splitter = QSplitter(Qt.Horizontal)

        # Panel izquierdo - Controles
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)

        # Panel derecho - Resultados
        self.resultados_widget = ResultadosVisuales()
        splitter.addWidget(self.resultados_widget)

        splitter.setSizes([400, 800])
        main_layout.addWidget(splitter)

        # Barra de estado
        status_layout = self.create_status_bar()
        main_layout.addLayout(status_layout)

        self.setLayout(main_layout)

    def create_header(self):
        """Crear header de la ventana"""
        header_layout = QHBoxLayout()

        # Título
        title = QLabel("🎯 Aprendizaje Supervisado")
        title.setObjectName("windowTitle")
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # Descripción
        desc = QLabel("Predicción y clasificación con algoritmos supervisados")
        desc.setObjectName("windowDesc")

        # Layout vertical para título y descripción
        title_layout = QVBoxLayout()
        title_layout.setSpacing(5)
        title_layout.addWidget(title)
        title_layout.addWidget(desc)

        header_layout.addLayout(title_layout)
        header_layout.addStretch()

        # Botón de cerrar
        close_button = QPushButton("✕")
        close_button.setObjectName("closeButton")
        close_button.setFixedSize(30, 30)
        close_button.clicked.connect(self.close)

        header_layout.addWidget(close_button)

        return header_layout

    def create_control_panel(self):
        """Crear panel de controles"""
        group = QGroupBox("🎯 Algoritmos de Aprendizaje Supervisado")
        group.setObjectName("controlGroup")

        layout = QVBoxLayout()
        layout.setSpacing(20)

        # Información introductoria
        intro_frame = self.create_intro_section()
        layout.addWidget(intro_frame)

        # Grid de algoritmos
        algorithms_grid = self.create_algorithms_grid()
        layout.addWidget(algorithms_grid)

        layout.addStretch()

        # Controles de utilidad
        utility_controls = self.create_utility_controls()
        layout.addWidget(utility_controls)

        group.setLayout(layout)
        return group

    def create_intro_section(self):
        """Crear sección introductoria"""
        intro_frame = QFrame()
        intro_frame.setObjectName("introFrame")

        layout = QVBoxLayout(intro_frame)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        title = QLabel("📚 Acerca del Aprendizaje Supervisado")
        title.setObjectName("introTitle")

        # Estado del sistema detallado
        status_parts = []

        if ML_FUNCTIONS_AVAILABLE:
            status_parts.append("✅ Funciones ML")
        else:
            status_parts.append("❌ Funciones ML")

        if ML_DEPS_AVAILABLE:
            status_parts.append("✅ Dependencias")
        else:
            status_parts.append("❌ Dependencias")

        status_text = " | ".join(status_parts)

        if ML_FUNCTIONS_AVAILABLE and ML_DEPS_AVAILABLE:
            status_color = "#28a745"
            status_icon = "🎉"
        elif ML_FUNCTIONS_AVAILABLE or ML_DEPS_AVAILABLE:
            status_color = "#ffc107"
            status_icon = "⚠️"
        else:
            status_color = "#dc3545"
            status_icon = "❌"

        status_label = QLabel(f"{status_icon} {status_text}")
        status_label.setStyleSheet(f"color: {status_color}; font-weight: bold; font-size: 12px;")

        # Información de depuración
        debug_info = QLabel(f"📂 Directorio actual: {os.path.basename(current_dir)}")
        debug_info.setStyleSheet("color: #6c757d; font-size: 10px;")

        description = QLabel(
            "Los algoritmos supervisados aprenden de datos etiquetados para hacer "
            "predicciones o clasificaciones. En este módulo encontrarás:\n\n"
            "• Regresión para predecir valores numéricos\n"
            "• Clasificación para categorizar muestras\n"
            "• Evaluación de precisión y métricas de rendimiento"
        )
        description.setObjectName("introText")
        description.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(status_label)
        layout.addWidget(debug_info)
        layout.addWidget(description)

        return intro_frame

    def create_algorithms_grid(self):
        """Crear grid de algoritmos"""
        algorithms_frame = QFrame()
        algorithms_frame.setObjectName("algorithmsFrame")

        grid_layout = QGridLayout(algorithms_frame)
        grid_layout.setSpacing(15)
        grid_layout.setContentsMargins(15, 15, 15, 15)

        # Definir algoritmos
        algorithms = [
            {
                "name": "📈 Regresión Múltiple",
                "key": "regresion_multiple",
                "description": "Predicción de scores usando múltiples variables",
                "details": "• R², MSE, MAE\n• Análisis de coeficientes\n• Gráficos de residuos",
                "color": "#4CAF50"
            },
            {
                "name": "🎯 SVM (Support Vector Machine)",
                "key": "svm",
                "description": "Clasificación con máquinas de vectores",
                "details": "• Kernel RBF\n• Matriz de confusión\n• Métricas por clase",
                "color": "#2196F3"
            },
            {
                "name": "🌳 Random Forest",
                "key": "random_forest",
                "description": "Ensemble de árboles de decisión",
                "details": "• Clasificación + Regresión\n• Importancia de variables\n• 100 árboles",
                "color": "#FF9800"
            },
            {
                "name": "📏 Regresión Lineal Completa",
                "key": "regresion_lineal",
                "description": "Simple, múltiple y regularizada",
                "details": "• Ridge y Lasso\n• Comparación R²\n• Análisis coeficientes",
                "color": "#9C27B0"
            }
        ]

        self.algorithm_buttons = {}

        for i, algo in enumerate(algorithms):
            button_frame = self.create_algorithm_button(algo)
            row = i // 2
            col = i % 2
            grid_layout.addWidget(button_frame, row, col)

        return algorithms_frame

    def create_algorithm_button(self, algo_config):
        """Crear botón de algoritmo personalizado"""
        button_frame = QFrame()
        button_frame.setObjectName("algorithmButton")
        button_frame.setMinimumHeight(160)
        button_frame.setMaximumHeight(180)
        button_frame.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(button_frame)
        layout.setSpacing(8)
        layout.setContentsMargins(15, 15, 15, 15)

        # Header del botón
        header_layout = QHBoxLayout()

        # Nombre del algoritmo
        name_label = QLabel(algo_config["name"])
        name_label.setObjectName("algorithmName")
        name_label.setWordWrap(True)

        # Indicador de estado
        status_indicator = QLabel("⚪")
        status_indicator.setObjectName("statusIndicator")
        status_indicator.setFixedSize(20, 20)
        status_indicator.setAlignment(Qt.AlignCenter)

        header_layout.addWidget(name_label)
        header_layout.addStretch()
        header_layout.addWidget(status_indicator)

        # Descripción
        desc_label = QLabel(algo_config["description"])
        desc_label.setObjectName("algorithmDesc")
        desc_label.setWordWrap(True)

        # Detalles técnicos
        details_label = QLabel(algo_config["details"])
        details_label.setObjectName("algorithmDetails")
        details_label.setWordWrap(True)

        # Botón de acción
        action_button = QPushButton("🚀 Ejecutar Análisis")
        action_button.setObjectName("executeButton")
        action_button.setMinimumHeight(35)

        # Determinar disponibilidad
        is_available = ML_FUNCTIONS_AVAILABLE and ML_DEPS_AVAILABLE
        action_button.setEnabled(is_available)

        if not is_available:
            if not ML_FUNCTIONS_AVAILABLE and not ML_DEPS_AVAILABLE:
                action_button.setText("❌ Funciones y Deps faltantes")
                action_button.setToolTip("Faltan tanto las funciones ML como las dependencias")
            elif not ML_FUNCTIONS_AVAILABLE:
                action_button.setText("❌ Funciones faltantes")
                action_button.setToolTip("ml_functions_supervisado.py no encontrado")
            else:
                action_button.setText("❌ Dependencias faltantes")
                action_button.setToolTip("pip install scikit-learn pandas numpy")

        action_button.clicked.connect(
            lambda: self.run_algorithm(algo_config["key"])
        )

        layout.addLayout(header_layout)
        layout.addWidget(desc_label)
        layout.addWidget(details_label)
        layout.addStretch()
        layout.addWidget(action_button)

        # Guardar referencias
        self.algorithm_buttons[algo_config["key"]] = {
            'frame': button_frame,
            'button': action_button,
            'status': status_indicator
        }

        # Efecto click en el frame solo si está habilitado
        if is_available:
            button_frame.mousePressEvent = lambda event: self.run_algorithm(algo_config["key"])

        return button_frame

    def create_utility_controls(self):
        """Crear controles de utilidad"""
        utility_frame = QFrame()
        utility_frame.setObjectName("utilityFrame")

        layout = QHBoxLayout(utility_frame)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Botón limpiar cache
        clear_cache_btn = QPushButton("🗑️ Limpiar Cache")
        clear_cache_btn.setObjectName("utilityButton")
        clear_cache_btn.clicked.connect(self.clear_cache)

        # Botón limpiar resultados
        clear_results_btn = QPushButton("📄 Limpiar Resultados")
        clear_results_btn.setObjectName("utilityButton")
        clear_results_btn.clicked.connect(self.clear_results)

        # Botón verificar dependencias
        check_deps_btn = QPushButton("🔍 Verificar Sistema")
        check_deps_btn.setObjectName("utilityButton")
        check_deps_btn.clicked.connect(self.check_system)

        # Botón buscar archivo ML
        find_ml_btn = QPushButton("🔎 Buscar ML")
        find_ml_btn.setObjectName("utilityButton")
        find_ml_btn.clicked.connect(self.find_ml_file)

        layout.addWidget(clear_cache_btn)
        layout.addWidget(clear_results_btn)
        layout.addWidget(check_deps_btn)
        layout.addWidget(find_ml_btn)
        layout.addStretch()

        return utility_frame

    def create_status_bar(self):
        """Crear barra de estado"""
        status_layout = QHBoxLayout()

        # Etiqueta de estado
        if ML_FUNCTIONS_AVAILABLE and ML_DEPS_AVAILABLE:
            status_text = "✅ Sistema supervisado completamente funcional"
        elif ML_FUNCTIONS_AVAILABLE:
            status_text = "⚠️ Funciones OK - Dependencias faltantes"
        elif ML_DEPS_AVAILABLE:
            status_text = "⚠️ Dependencias OK - Funciones faltantes"
        else:
            status_text = "❌ Sistema no disponible"

        self.status_label = QLabel(status_text)
        self.status_label.setObjectName("statusLabel")

        # Información del sistema
        cpu_count = multiprocessing.cpu_count()
        mode = "Completo" if (ML_FUNCTIONS_AVAILABLE and ML_DEPS_AVAILABLE) else "Limitado"
        system_info = QLabel(f"🖥️ CPUs: {cpu_count} | 🎯 Modo: {mode}")
        system_info.setObjectName("systemInfo")

        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("progressBar")
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)

        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(system_info)
        status_layout.addWidget(self.progress_bar)

        return status_layout

    def run_algorithm(self, algorithm_key):
        """Ejecutar algoritmo seleccionado"""
        # Verificar que el sistema esté disponible
        if not (ML_FUNCTIONS_AVAILABLE and ML_DEPS_AVAILABLE):
            error_details = []
            if not ML_FUNCTIONS_AVAILABLE:
                error_details.append("❌ Funciones ML no encontradas")
            if not ML_DEPS_AVAILABLE:
                error_details.append("❌ Dependencias faltantes")

            QMessageBox.critical(
                self,
                "Sistema No Disponible",
                f"El sistema de Machine Learning no está disponible.\n\n"
                f"Estado actual:\n"
                f"{'✅' if ML_FUNCTIONS_AVAILABLE else '❌'} Funciones ML\n"
                f"{'✅' if ML_DEPS_AVAILABLE else '❌'} Dependencias\n\n"
                f"Para solucionar:\n"
                f"1. Verifica que ml_functions_supervisado.py esté en el directorio\n"
                f"2. Instala dependencias: pip install scikit-learn pandas numpy\n"
                f"3. Usa el botón 'Buscar ML' para localizar el archivo"
            )
            return

        if self.worker and self.worker.isRunning():
            QMessageBox.information(
                self,
                "Procesamiento en Curso",
                "Ya hay un análisis ejecutándose. Por favor espera."
            )
            return

        # Actualizar UI
        self.update_ui_for_processing(algorithm_key, True)

        # Crear y ejecutar worker
        self.worker = SupervisadoWorker(algorithm_key)
        self.worker.finished.connect(self.on_algorithm_finished)
        self.worker.progress.connect(self.update_progress)
        self.worker.status_update.connect(self.update_status)
        self.worker.start()

    def update_ui_for_processing(self, algorithm_key, is_processing):
        """Actualizar UI durante procesamiento"""
        # Deshabilitar todos los botones
        for key, components in self.algorithm_buttons.items():
            if ML_FUNCTIONS_AVAILABLE and ML_DEPS_AVAILABLE:
                components['button'].setEnabled(not is_processing)

            if key == algorithm_key and is_processing:
                components['status'].setText("⚡")
                components['status'].setStyleSheet("color: #ffc107; font-weight: bold;")
            elif not is_processing:
                components['status'].setText("⚪")
                components['status'].setStyleSheet("")

        # Mostrar/ocultar barra de progreso
        self.progress_bar.setVisible(is_processing)
        if is_processing:
            self.progress_bar.setValue(0)

    def update_progress(self, value):
        """Actualizar progreso"""
        self.progress_bar.setValue(value)

    def update_status(self, message):
        """Actualizar mensaje de estado"""
        self.status_label.setText(message)

    def on_algorithm_finished(self, results, algorithm_key):
        """Manejar finalización del algoritmo"""
        # Restaurar UI
        self.update_ui_for_processing(algorithm_key, False)

        if "error" in results:
            self.status_label.setText(f"❌ Error: {results['error']}")
            self.algorithm_buttons[algorithm_key]['status'].setText("❌")
            self.algorithm_buttons[algorithm_key]['status'].setStyleSheet("color: #dc3545;")

            QMessageBox.critical(
                self,
                "Error en Análisis",
                f"Error durante la ejecución:\n{results['error']}"
            )
            return

        # Marcar como completado
        self.algorithm_buttons[algorithm_key]['status'].setText("✅")
        self.algorithm_buttons[algorithm_key]['status'].setStyleSheet("color: #28a745; font-weight: bold;")

        # Mostrar resultados
        self.resultados_widget.mostrar_resultados(results, algorithm_key)

        # Mensaje de éxito
        success_messages = {
            "regresion_multiple": "✅ Regresión múltiple completada - Métricas R² calculadas",
            "svm": "✅ SVM completado - Clasificación con kernel RBF",
            "random_forest": "✅ Random Forest completado - Análisis dual",
            "regresion_lineal": "✅ Regresión lineal completada - Comparación de métodos"
        }

        self.status_label.setText(success_messages.get(algorithm_key, "✅ Análisis completado"))

    def clear_cache(self):
        """Limpiar cache del sistema"""
        self.cache.clear()
        self.status_label.setText("🗑️ Cache limpiado")

        # Resetear indicadores de estado
        for components in self.algorithm_buttons.values():
            components['status'].setText("⚪")
            components['status'].setStyleSheet("")

    def clear_results(self):
        """Limpiar resultados visuales"""
        self.resultados_widget.limpiar_resultados()
        self.status_label.setText("📄 Resultados limpiados")

    def find_ml_file(self):
        """Buscar archivo ml_functions_supervisado.py"""
        search_paths = [
            current_dir,
            os.path.dirname(current_dir),
            os.path.join(current_dir, '..'),
            os.path.join(current_dir, '..', 'machine_learning'),
            os.path.join(os.path.dirname(current_dir), 'machine_learning'),
        ]

        found_paths = []
        for search_path in search_paths:
            try:
                full_path = os.path.abspath(search_path)
                ml_file = os.path.join(full_path, 'ml_functions_supervisado.py')
                if os.path.exists(ml_file):
                    found_paths.append(ml_file)
            except:
                continue

        if found_paths:
            paths_text = "\n".join([f"• {path}" for path in found_paths])
            QMessageBox.information(
                self,
                "🔎 Archivos ML Encontrados",
                f"Se encontraron los siguientes archivos:\n\n{paths_text}\n\n"
                f"Directorio actual: {current_dir}\n\n"
                f"Para que funcione automáticamente, el archivo debe estar en:\n"
                f"• {current_dir}\n"
                f"• O ser importable desde Python"
            )
        else:
            QMessageBox.warning(
                self,
                "🔎 Archivo ML No Encontrado",
                f"No se encontró ml_functions_supervisado.py en:\n\n"
                f"• {current_dir}\n"
                f"• Carpetas padre\n"
                f"• Carpetas relacionadas\n\n"
                f"Asegúrate de que el archivo esté en el lugar correcto."
            )

    def check_system(self):
        """Verificar estado del sistema"""
        # Reimportar para verificar estado actual
        global ML_FUNCTIONS_AVAILABLE, ML_DEPS_AVAILABLE

        # Verificar dependencias
        deps_details = []
        try:
            import numpy as np
            deps_details.append(f"✅ NumPy {np.__version__}")
        except ImportError:
            deps_details.append("❌ NumPy faltante")

        try:
            import pandas as pd
            deps_details.append(f"✅ Pandas {pd.__version__}")
        except ImportError:
            deps_details.append("❌ Pandas faltante")

        try:
            import sklearn
            deps_details.append(f"✅ Scikit-learn {sklearn.__version__}")
            ML_DEPS_AVAILABLE = True
        except ImportError:
            deps_details.append("❌ Scikit-learn faltante")
            ML_DEPS_AVAILABLE = False

        # Verificar funciones
        funcs_details = []
        if ML_FUNCTIONS_AVAILABLE:
            try:
                # Intentar llamar a una función de prueba
                test_result = regresion_multiple_proceso(50)
                if 'error' not in test_result:
                    funcs_details.append("✅ Funciones funcionando correctamente")
                else:
                    funcs_details.append(f"⚠️ Funciones con errores: {test_result['error']}")
            except Exception as e:
                funcs_details.append(f"❌ Error al probar funciones: {str(e)}")
        else:
            funcs_details.append("❌ Funciones ML no importadas")

        # Información de archivos
        file_info = []
        ml_file_path = os.path.join(current_dir, 'ml_functions_supervisado.py')
        if os.path.exists(ml_file_path):
            file_info.append(f"✅ Archivo encontrado: {ml_file_path}")
        else:
            file_info.append(f"❌ Archivo no encontrado en: {current_dir}")

        QMessageBox.information(
            self,
            "🔍 Estado del Sistema",
            f"<h3>Estado del Sistema ML</h3>"
            f"<p><b>📦 Dependencias:</b></p>"
            f"<p>{'<br>'.join(deps_details)}</p>"
            f"<p><b>🔧 Funciones ML:</b></p>"
            f"<p>{'<br>'.join(funcs_details)}</p>"
            f"<p><b>📁 Archivos:</b></p>"
            f"<p>{'<br>'.join(file_info)}</p>"
            f"<p><b>🖥️ Sistema:</b></p>"
            f"<p>CPUs: {multiprocessing.cpu_count()}<br>"
            f"Python: {sys.version.split()[0]}<br>"
            f"Directorio: {os.path.basename(current_dir)}</p>"
            f"<hr>"
            f"<p><b>Para instalar dependencias:</b><br>"
            f"<code>pip install scikit-learn pandas numpy matplotlib</code></p>"
        )

        # Actualizar UI después de verificar
        self.setup_ui()

    def apply_styles(self):
        """Aplicar estilos CSS"""
        styles = """
        QWidget {
            font-family: 'Segoe UI', Arial, sans-serif;
        }

        #windowTitle {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }

        #windowDesc {
            font-size: 14px;
            color: #6c757d;
        }

        #closeButton {
            background: #dc3545;
            color: white;
            border: none;
            border-radius: 15px;
            font-weight: bold;
        }

        #closeButton:hover {
            background: #c82333;
        }

        #controlGroup {
            font-size: 16px;
            font-weight: bold;
            color: #495057;
            border: 2px solid #dee2e6;
            border-radius: 10px;
            margin: 5px;
        }

        #introFrame {
            background: linear-gradient(135deg, #e3f2fd 0%, #f8f9fa 100%);
            border: 1px solid #90caf9;
            border-radius: 8px;
        }

        #introTitle {
            font-size: 16px;
            font-weight: bold;
            color: #1565c0;
        }

        #introText {
            font-size: 13px;
            color: #37474f;
            line-height: 1.4;
        }

        #algorithmsFrame {
            background: #ffffff;
            border: 1px solid #e9ecef;
            border-radius: 8px;
        }

        #algorithmButton {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border: 2px solid #e9ecef;
            border-radius: 10px;
            margin: 2px;
        }

        #algorithmButton:hover {
            border-color: #007bff;
            background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
        }

        #algorithmName {
            font-size: 14px;
            font-weight: bold;
            color: #2c3e50;
        }

        #algorithmDesc {
            font-size: 12px;
            color: #6c757d;
            font-weight: 500;
        }

        #algorithmDetails {
            font-size: 11px;
            color: #868e96;
            line-height: 1.3;
        }

        #executeButton {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            border: none;
            border-radius: 6px;
            font-weight: bold;
            font-size: 12px;
        }

        #executeButton:hover {
            background: linear-gradient(135deg, #218838 0%, #1e7e34 100%);
        }

        #executeButton:disabled {
            background: #6c757d;
        }

        #utilityFrame {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
        }

        #utilityButton {
            background: #6c757d;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 12px;
            font-size: 11px;
            font-weight: bold;
        }

        #utilityButton:hover {
            background: #5a6268;
        }

        #statusLabel {
            font-size: 13px;
            color: #28a745;
            font-weight: 500;
        }

        #systemInfo {
            font-size: 11px;
            color: #6c757d;
        }

        #progressBar {
            border: 1px solid #dee2e6;
            border-radius: 4px;
            text-align: center;
        }

        #progressBar::chunk {
            background: linear-gradient(90deg, #007bff 0%, #28a745 100%);
            border-radius: 3px;
        }
        """

        self.setStyleSheet(styles)

    def closeEvent(self, event):
        """Manejar cierre de ventana"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Cerrar Ventana",
                "Hay un análisis ejecutándose. ¿Deseas interrumpirlo?",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.worker.terminate()
                self.worker.wait(2000)
            else:
                event.ignore()
                return

        event.accept()


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = SupervisadoWindow()
    window.show()
    sys.exit(app.exec_())