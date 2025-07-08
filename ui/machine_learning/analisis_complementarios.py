"""
analisis_complementarios.py - Ventana de Análisis Complementarios
Contiene: Análisis de Calidad, Optimización, Comparación de Métodos
"""

import sys
import multiprocessing
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                             QHBoxLayout, QFrame, QProgressBar, QSplitter,
                             QGroupBox, QGridLayout, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

# Importar sistema de temas
from darkmode import ThemedWidget, ThemeManager
from resultados_visuales import ResultadosVisuales
from data_cache import DataCache


# Funciones simuladas para análisis complementarios
def analizar_calidad_agua_proceso():
    """Análisis básico de calidad del agua"""
    import time
    time.sleep(2)  # Simular procesamiento

    return {
        'tipo': 'calidad_agua',
        'estadisticas': {
            'total_muestras': 150,
            'ph_promedio': 7.2,
            'oxigeno_promedio': 8.5,
            'turbidez_promedio': 2.1,
            'conductividad_promedio': 520,
            'calidad_promedio': 75.5
        },
        'distribucion': {
            'Excelente': 45,
            'Buena': 60,
            'Regular': 30,
            'Necesita Tratamiento': 15
        },
        'estado_general': '🟢 BUENA',
        'mensaje': 'La calidad general del agua es buena'
    }


def optimizar_sistema_proceso():
    """Optimización de hiperparámetros"""
    import time
    time.sleep(3)  # Simular procesamiento

    return {
        'tipo': 'optimizacion',
        'resultados': [
            {'configuracion': '50 árboles, prof. 5', 'precision': 87.5},
            {'configuracion': '100 árboles, prof. 7', 'precision': 92.1},
            {'configuracion': '150 árboles, prof. 10', 'precision': 89.8}
        ],
        'mejor_config': {'configuracion': '100 árboles, prof. 7', 'precision': 92.1}
    }


def comparar_metodos_proceso():
    """Comparación de diferentes algoritmos ML"""
    import time
    time.sleep(2.5)  # Simular procesamiento

    return {
        'tipo': 'comparacion',
        'metodos': [
            {'metodo': 'Random Forest', 'precision': 92.1, 'ventajas': 'Robusto y preciso'},
            {'metodo': 'SVM', 'precision': 89.3, 'ventajas': 'Efectivo con datos complejos'},
            {'metodo': 'Árbol de Decisión', 'precision': 84.7, 'ventajas': 'Fácil interpretación'}
        ]
    }


class ComplementariosWorker(QThread):
    """Worker thread para análisis complementarios"""
    finished = pyqtSignal(dict, str)
    progress = pyqtSignal(int)
    status_update = pyqtSignal(str)

    def __init__(self, algorithm):
        super().__init__()
        self.algorithm = algorithm
        self.cache = DataCache()

    def run(self):
        try:
            # Verificar cache
            cache_key = f"complementario_{self.algorithm}"
            cached_result = self.cache.get(cache_key)

            if cached_result:
                self.status_update.emit("📦 Usando datos en cache...")
                self.progress.emit(100)
                self.finished.emit(cached_result, self.algorithm)
                return

            self.status_update.emit("📊 Iniciando análisis complementario...")
            self.progress.emit(10)

            # Mapeo de funciones
            process_functions = {
                "calidad_agua": analizar_calidad_agua_proceso,
                "optimizar_sistema": optimizar_sistema_proceso,
                "comparar_metodos": comparar_metodos_proceso
            }

            if self.algorithm not in process_functions:
                self.finished.emit({"error": "Análisis no reconocido"}, "error")
                return

            self.progress.emit(30)

            # Ejecutar función
            result = process_functions[self.algorithm]()
            self.progress.emit(95)

            if 'error' not in result:
                self.cache.set(cache_key, result)

            self.progress.emit(100)
            self.finished.emit(result, self.algorithm)

        except Exception as e:
            self.finished.emit({"error": f"Error en worker: {str(e)}"}, "error")


class AnalisisComplementarios(QWidget, ThemedWidget):
    """Ventana de Análisis Complementarios"""

    def __init__(self):
        QWidget.__init__(self)
        ThemedWidget.__init__(self)

        self.worker = None
        self.cache = DataCache()
        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):
        """Configurar interfaz de usuario"""
        self.setWindowTitle("📊 Análisis Complementarios - ML Calidad del Agua")
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

        # Título y descripción
        title_layout = QVBoxLayout()
        title_layout.setSpacing(5)

        title = QLabel("📊 Análisis Complementarios")
        title.setObjectName("windowTitle")

        desc = QLabel("Evaluación, optimización y comparación de sistemas ML")
        desc.setObjectName("windowDesc")

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
        group = QGroupBox("📊 Análisis Complementarios del Sistema")
        group.setObjectName("controlGroup")

        layout = QVBoxLayout()
        layout.setSpacing(25)

        # Información introductoria
        intro_frame = self.create_intro_section()
        layout.addWidget(intro_frame)

        # Análisis disponibles
        analysis_section = self.create_analysis_section()
        layout.addWidget(analysis_section)

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
        layout.setSpacing(12)
        layout.setContentsMargins(20, 15, 20, 15)

        title = QLabel("🔧 Herramientas de Evaluación")
        title.setObjectName("introTitle")

        description = QLabel(
            "Conjunto de herramientas para evaluar, optimizar y comparar "
            "el rendimiento de los sistemas de machine learning:\n\n"
            "🧪 <b>Análisis de Calidad:</b> Evaluación completa de parámetros del agua\n"
            "⚙️ <b>Optimización:</b> Búsqueda automática de mejores configuraciones\n"
            "📊 <b>Comparación:</b> Evaluación comparativa entre algoritmos"
        )
        description.setObjectName("introText")
        description.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(description)

        return intro_frame

    def create_analysis_section(self):
        """Crear sección de análisis"""
        analysis_frame = QFrame()
        analysis_frame.setObjectName("analysisFrame")

        layout = QVBoxLayout(analysis_frame)
        layout.setSpacing(20)
        layout.setContentsMargins(15, 20, 15, 20)

        # Definir análisis
        analyses = [
            {
                "name": "🧪 Análisis de Calidad del Agua",
                "key": "calidad_agua",
                "subtitle": "Evaluación Completa de Parámetros",
                "description": "Análisis exhaustivo de pH, oxígeno disuelto, turbidez y conductividad",
                "features": [
                    "• Cálculo automático de scores de calidad",
                    "• Clasificación en categorías (Excelente, Buena, etc.)",
                    "• Estadísticas descriptivas completas",
                    "• Distribución visual de resultados",
                    "• Recomendaciones de mejora automáticas"
                ],
                "time": "8-12 segundos",
                "color": "#4CAF50",
                "icon": "🧪"
            },
            {
                "name": "⚙️ Optimización de Hiperparámetros",
                "key": "optimizar_sistema",
                "subtitle": "Búsqueda Automática de Configuraciones",
                "description": "Encuentra automáticamente los mejores parámetros para algoritmos ML",
                "features": [
                    "• Grid search de hiperparámetros",
                    "• Evaluación de múltiples configuraciones",
                    "• Análisis de trade-offs precisión/complejidad",
                    "• Recomendación de configuración óptima",
                    "• Visualización comparativa de resultados"
                ],
                "time": "15-20 segundos",
                "color": "#FF9800",
                "icon": "⚙️"
            },
            {
                "name": "📊 Comparación de Algoritmos",
                "key": "comparar_metodos",
                "subtitle": "Evaluación Comparativa de ML",
                "description": "Compara el rendimiento de diferentes algoritmos de machine learning",
                "features": [
                    "• Evaluación simultánea de múltiples algoritmos",
                    "• Métricas de precisión y rendimiento",
                    "• Análisis de ventajas y desventajas",
                    "• Recomendación del mejor algoritmo",
                    "• Gráficos comparativos detallados"
                ],
                "time": "12-18 segundos",
                "color": "#2196F3",
                "icon": "📊"
            }
        ]

        self.analysis_buttons = {}

        for analysis in analyses:
            analysis_widget = self.create_analysis_widget(analysis)
            layout.addWidget(analysis_widget)

        return analysis_frame

    def create_analysis_widget(self, analysis_config):
        """Crear widget de análisis completo"""
        analysis_frame = QFrame()
        analysis_frame.setObjectName("analysisWidget")
        analysis_frame.setMinimumHeight(180)
        analysis_frame.setMaximumHeight(200)

        main_layout = QHBoxLayout(analysis_frame)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 15, 20, 15)

        # Panel izquierdo - Información
        info_layout = QVBoxLayout()
        info_layout.setSpacing(8)

        # Header con icono y nombre
        header_layout = QHBoxLayout()

        icon_label = QLabel(analysis_config["icon"])
        icon_label.setObjectName("analysisIcon")

        name_label = QLabel(analysis_config["name"])
        name_label.setObjectName("analysisName")

        header_layout.addWidget(icon_label)
        header_layout.addWidget(name_label)
        header_layout.addStretch()

        # Subtítulo
        subtitle_label = QLabel(analysis_config["subtitle"])
        subtitle_label.setObjectName("analysisSubtitle")

        # Descripción
        desc_label = QLabel(analysis_config["description"])
        desc_label.setObjectName("analysisDesc")
        desc_label.setWordWrap(True)

        # Características (primeras 3)
        features_text = "\n".join(analysis_config["features"][:3])
        features_label = QLabel(features_text)
        features_label.setObjectName("analysisFeatures")
        features_label.setWordWrap(True)

        info_layout.addLayout(header_layout)
        info_layout.addWidget(subtitle_label)
        info_layout.addWidget(desc_label)
        info_layout.addWidget(features_label)
        info_layout.addStretch()

        # Panel derecho - Controles
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(10)
        controls_layout.setAlignment(Qt.AlignCenter)

        # Indicador de estado
        status_indicator = QLabel("⚪")
        status_indicator.setObjectName("statusIndicator")
        status_indicator.setFixedSize(35, 35)
        status_indicator.setAlignment(Qt.AlignCenter)

        # Tiempo estimado
        time_label = QLabel(f"⏱️ {analysis_config['time']}")
        time_label.setObjectName("timeLabel")
        time_label.setAlignment(Qt.AlignCenter)

        # Botón de ejecución
        execute_button = QPushButton("🚀 Ejecutar")
        execute_button.setObjectName("executeButton")
        execute_button.setMinimumHeight(35)
        execute_button.setMinimumWidth(120)
        execute_button.clicked.connect(
            lambda: self.run_analysis(analysis_config["key"])
        )

        # Botón de información
        info_button = QPushButton("ℹ️ Info")
        info_button.setObjectName("infoButton")
        info_button.setMaximumWidth(80)
        info_button.clicked.connect(
            lambda: self.show_analysis_info(analysis_config)
        )

        controls_layout.addWidget(status_indicator)
        controls_layout.addWidget(time_label)
        controls_layout.addWidget(execute_button)
        controls_layout.addWidget(info_button)

        # Agregar a layout principal
        main_layout.addLayout(info_layout, 3)  # 75% del espacio
        main_layout.addLayout(controls_layout, 1)  # 25% del espacio

        # Guardar referencias
        self.analysis_buttons[analysis_config["key"]] = {
            'frame': analysis_frame,
            'button': execute_button,
            'status': status_indicator,
            'info_button': info_button
        }

        return analysis_frame

    def create_utility_controls(self):
        """Crear controles de utilidad"""
        utility_frame = QFrame()
        utility_frame.setObjectName("utilityFrame")

        layout = QHBoxLayout(utility_frame)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 10, 15, 10)

        # Botones de utilidad
        clear_cache_btn = QPushButton("🗑️ Limpiar Cache")
        clear_cache_btn.setObjectName("utilityButton")
        clear_cache_btn.clicked.connect(self.clear_cache)

        clear_results_btn = QPushButton("📄 Limpiar Resultados")
        clear_results_btn.setObjectName("utilityButton")
        clear_results_btn.clicked.connect(self.clear_results)

        help_button = QPushButton("❓ Ayuda")
        help_button.setObjectName("utilityButton")
        help_button.clicked.connect(self.show_help)

        layout.addWidget(clear_cache_btn)
        layout.addWidget(clear_results_btn)
        layout.addWidget(help_button)
        layout.addStretch()

        return utility_frame

    def create_status_bar(self):
        """Crear barra de estado"""
        status_layout = QHBoxLayout()

        # Etiqueta de estado
        self.status_label = QLabel("✅ Sistema complementario listo")
        self.status_label.setObjectName("statusLabel")

        # Información del sistema
        cpu_count = multiprocessing.cpu_count()
        system_info = QLabel(f"🖥️ CPUs: {cpu_count} | 📊 Análisis Complementarios")
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
            background: linear-gradient(135deg, #fff3e0 0%, #f8f9fa 100%);
            border: 1px solid #ffb74d;
            border-radius: 8px;
        }

        #introTitle {
            font-size: 16px;
            font-weight: bold;
            color: #ef6c00;
        }

        #introText {
            font-size: 13px;
            color: #37474f;
            line-height: 1.5;
        }

        #analysisFrame {
            background: #ffffff;
            border: 1px solid #e9ecef;
            border-radius: 8px;
        }

        #analysisWidget {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border: 2px solid #e9ecef;
            border-radius: 12px;
            margin: 5px 0;
        }

        #analysisWidget:hover {
            border-color: #007bff;
            background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
        }

        #analysisIcon {
            font-size: 20px;
            margin-right: 8px;
        }

        #analysisName {
            font-size: 15px;
            font-weight: bold;
            color: #2c3e50;
        }

        #analysisSubtitle {
            font-size: 12px;
            color: #007bff;
            font-weight: 600;
            font-style: italic;
        }

        #analysisDesc {
            font-size: 12px;
            color: #6c757d;
            margin: 5px 0;
        }

        #analysisFeatures {
            font-size: 10px;
            color: #868e96;
            line-height: 1.4;
        }

        #statusIndicator {
            font-size: 20px;
            border: 2px solid #dee2e6;
            border-radius: 17px;
            background: #f8f9fa;
        }

        #timeLabel {
            font-size: 10px;
            color: #6c757d;
            font-weight: 500;
        }

        #executeButton {
            background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
            color: white;
            border: none;
            border-radius: 6px;
            font-weight: bold;
            font-size: 12px;
        }

        #executeButton:hover {
            background: linear-gradient(135deg, #0056b3 0%, #004085 100%);
        }

        #executeButton:disabled {
            background: #6c757d;
        }

        #infoButton {
            background: #28a745;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 10px;
            font-weight: bold;
            padding: 5px 8px;
        }

        #infoButton:hover {
            background: #218838;
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
            color: #007bff;
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

    def run_analysis(self, analysis_key):
        """Ejecutar análisis seleccionado"""
        if self.worker and self.worker.isRunning():
            QMessageBox.information(
                self,
                "Procesamiento en Curso",
                "Ya hay un análisis ejecutándose. Por favor espera."
            )
            return

        # Actualizar UI
        self.update_ui_for_processing(analysis_key, True)

        # Crear y ejecutar worker
        self.worker = ComplementariosWorker(analysis_key)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.progress.connect(self.update_progress)
        self.worker.status_update.connect(self.update_status)
        self.worker.start()

    def update_ui_for_processing(self, analysis_key, is_processing):
        """Actualizar UI durante procesamiento"""
        # Deshabilitar todos los botones
        for key, components in self.analysis_buttons.items():
            components['button'].setEnabled(not is_processing)
            components['info_button'].setEnabled(not is_processing)

            if key == analysis_key and is_processing:
                components['status'].setText("⚡")
                components['status'].setStyleSheet("color: #ffc107; font-weight: bold; font-size: 24px;")
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

    def on_analysis_finished(self, results, analysis_key):
        """Manejar finalización del análisis"""
        # Restaurar UI
        self.update_ui_for_processing(analysis_key, False)

        if "error" in results:
            self.status_label.setText(f"❌ Error: {results['error']}")
            self.analysis_buttons[analysis_key]['status'].setText("❌")
            self.analysis_buttons[analysis_key]['status'].setStyleSheet("color: #dc3545; font-size: 24px;")

            QMessageBox.critical(
                self,
                "Error en Análisis",
                f"Error durante la ejecución:\n{results['error']}"
            )
            return

        # Marcar como completado
        self.analysis_buttons[analysis_key]['status'].setText("✅")
        self.analysis_buttons[analysis_key]['status'].setStyleSheet(
            "color: #28a745; font-weight: bold; font-size: 24px;")

        # Mostrar resultados
        self.resultados_widget.mostrar_resultados(results, analysis_key)

        # Mensaje de éxito
        success_messages = {
            "calidad_agua": "✅ Análisis de calidad completado - Parámetros evaluados",
            "optimizar_sistema": "✅ Optimización completada - Mejores parámetros encontrados",
            "comparar_metodos": "✅ Comparación completada - Algoritmos evaluados"
        }

        self.status_label.setText(success_messages.get(analysis_key, "✅ Análisis completado"))

    def show_analysis_info(self, analysis_config):
        """Mostrar información detallada del análisis"""
        info_texts = {
            "calidad_agua": (
                "<b>🧪 Análisis de Calidad del Agua</b><br><br>"
                "<b>Objetivo:</b> Evaluar la calidad del agua mediante análisis integral de parámetros.<br><br>"
                "<b>Parámetros evaluados:</b><br>"
                "• <b>pH:</b> Acidez/alcalinidad (rango óptimo: 6.5-8.5)<br>"
                "• <b>Oxígeno Disuelto:</b> Disponibilidad para vida acuática (>6 mg/L)<br>"
                "• <b>Turbidez:</b> Claridad del agua (<4 NTU)<br>"
                "• <b>Conductividad:</b> Contenido de sales (200-800 μS/cm)<br><br>"
                "<b>Resultados:</b><br>"
                "• Score de calidad (0-100)<br>"
                "• Clasificación automática<br>"
                "• Estadísticas descriptivas<br>"
                "• Recomendaciones de mejora"
            ),
            "optimizar_sistema": (
                "<b>⚙️ Optimización de Hiperparámetros</b><br><br>"
                "<b>Objetivo:</b> Encontrar la configuración óptima para algoritmos ML.<br><br>"
                "<b>Proceso:</b><br>"
                "• <b>Grid Search:</b> Búsqueda sistemática de parámetros<br>"
                "• <b>Validación Cruzada:</b> Evaluación robusta<br>"
                "• <b>Métricas múltiples:</b> Precisión, tiempo, complejidad<br><br>"
                "<b>Parámetros optimizados:</b><br>"
                "• Número de estimadores (árboles)<br>"
                "• Profundidad máxima<br>"
                "• Criterios de división<br><br>"
                "<b>Resultado:</b> Configuración recomendada con mejor balance precisión/eficiencia"
            ),
            "comparar_metodos": (
                "<b>📊 Comparación de Algoritmos ML</b><br><br>"
                "<b>Objetivo:</b> Evaluar y comparar diferentes algoritmos de machine learning.<br><br>"
                "<b>Algoritmos comparados:</b><br>"
                "• <b>Random Forest:</b> Ensemble de árboles de decisión<br>"
                "• <b>SVM:</b> Máquinas de vectores de soporte<br>"
                "• <b>Árbol de Decisión:</b> Modelo interpretable<br><br>"
                "<b>Métricas evaluadas:</b><br>"
                "• Precisión de clasificación<br>"
                "• Tiempo de entrenamiento<br>"
                "• Interpretabilidad<br>"
                "• Robustez<br><br>"
                "<b>Resultado:</b> Recomendación del algoritmo más adecuado para el caso de uso"
            )
        }

        info_text = info_texts.get(analysis_config["key"], "Información no disponible")

        QMessageBox.information(
            self,
            f"Información - {analysis_config['name']}",
            info_text
        )

    def show_help(self):
        """Mostrar ayuda general"""
        help_text = (
            "<b>📊 Análisis Complementarios - Guía de Uso</b><br><br>"
            "<b>Propósito:</b><br>"
            "Herramientas adicionales para evaluar, optimizar y comparar "
            "el rendimiento de sistemas de machine learning.<br><br>"
            "<b>Flujo recomendado:</b><br>"
            "1. <b>Análisis de Calidad:</b> Evalúa tus datos iniciales<br>"
            "2. <b>Comparación de Métodos:</b> Identifica el mejor algoritmo<br>"
            "3. <b>Optimización:</b> Afina los parámetros del mejor método<br><br>"
            "<b>Interpretación:</b><br>"
            "• <b>Scores >80:</b> Excelente calidad<br>"
            "• <b>Precisión >90%:</b> Modelo muy confiable<br>"
            "• <b>Optimización:</b> Mejora típica de 3-5% en precisión<br><br>"
            "<b>Consejos:</b><br>"
            "• Ejecuta análisis en orden sugerido<br>"
            "• Compara resultados entre métodos<br>"
            "• Usa optimización solo después de seleccionar algoritmo"
        )

        QMessageBox.information(self, "❓ Ayuda - Análisis Complementarios", help_text)

    def clear_cache(self):
        """Limpiar cache del sistema"""
        self.cache.clear()
        self.status_label.setText("🗑️ Cache limpiado")

        # Resetear indicadores de estado
        for components in self.analysis_buttons.values():
            components['status'].setText("⚪")
            components['status'].setStyleSheet("")

    def clear_results(self):
        """Limpiar resultados visuales"""
        self.resultados_widget.limpiar_resultados()
        self.status_label.setText("📄 Resultados limpiados")

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
    window = AnalisisComplementarios()
    window.show()
    sys.exit(app.exec_())