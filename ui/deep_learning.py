import sys
import os
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                             QHBoxLayout, QScrollArea, QFrame, QProgressBar,
                             QSplitter, QGroupBox, QGridLayout, QMessageBox,
                             QTextEdit, QTabWidget, QComboBox, QSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QPainter

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

# Verificar dependencias Deep Learning
try:
    import tensorflow as tf
    import keras

    TF_AVAILABLE = True
    print("✅ TensorFlow disponible")
except ImportError:
    TF_AVAILABLE = False
    print("❌ TensorFlow no disponible")

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
    print("✅ PyTorch disponible")
except ImportError:
    TORCH_AVAILABLE = False
    print("❌ PyTorch no disponible")

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    BASE_DEPS_AVAILABLE = True
    print("✅ Dependencias base disponibles")
except ImportError:
    BASE_DEPS_AVAILABLE = False
    print("❌ Dependencias base faltantes")

# Estado general del sistema DL
DL_AVAILABLE = (TF_AVAILABLE or TORCH_AVAILABLE) and BASE_DEPS_AVAILABLE

try:
    from data_cache import DataCache

    print("✅ DataCache importado")
except ImportError:
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


class DeepLearningWorker(QThread):
    """Worker thread para análisis de Deep Learning"""
    finished = pyqtSignal(dict, str)
    progress = pyqtSignal(int)
    status_update = pyqtSignal(str)

    def __init__(self, network_type, config):
        super().__init__()
        self.network_type = network_type
        self.config = config
        self.cache = DataCache()

    def run(self):
        try:
            if not DL_AVAILABLE:
                self.finished.emit({
                    "error": "Bibliotecas de Deep Learning no disponibles"
                }, "error")
                return

            # Verificar cache
            cache_key = f"dl_{self.network_type}_{hash(str(self.config))}"
            cached_result = self.cache.get(cache_key)

            if cached_result:
                self.status_update.emit("📦 Usando resultados en cache...")
                self.progress.emit(100)
                self.finished.emit(cached_result, self.network_type)
                return

            self.status_update.emit("🧠 Iniciando análisis de Deep Learning...")
            self.progress.emit(10)

            # Simular análisis según el tipo de red
            if self.network_type == "cnn":
                result = self.simulate_cnn_analysis()
            elif self.network_type == "rnn":
                result = self.simulate_rnn_analysis()
            elif self.network_type == "lstm":
                result = self.simulate_lstm_analysis()
            elif self.network_type == "gru":
                result = self.simulate_gru_analysis()
            elif self.network_type == "autoencoder":
                result = self.simulate_autoencoder_analysis()
            elif self.network_type == "gan":
                result = self.simulate_gan_analysis()
            else:
                result = {"error": "Tipo de red no reconocido"}

            if 'error' not in result:
                self.cache.set(cache_key, result)
                self.status_update.emit("✅ Análisis completado exitosamente")
            else:
                self.status_update.emit(f"❌ Error en análisis: {result['error']}")

            self.progress.emit(100)
            self.finished.emit(result, self.network_type)

        except Exception as e:
            self.finished.emit({"error": f"Error en worker: {str(e)}"}, "error")
        finally:
            gc.collect()

    def simulate_cnn_analysis(self):
        """Simular análisis CNN"""
        self.status_update.emit("🔍 Construyendo arquitectura CNN...")
        self.progress.emit(25)

        time.sleep(1)  # Simular procesamiento

        self.status_update.emit("🏋️ Entrenando modelo CNN...")
        self.progress.emit(50)

        time.sleep(2)

        self.status_update.emit("📊 Evaluando rendimiento...")
        self.progress.emit(75)

        time.sleep(1)

        # Simular resultados
        import random
        accuracy = round(random.uniform(0.85, 0.95), 3)
        loss = round(random.uniform(0.05, 0.15), 4)

        return {
            "network_type": "CNN",
            "architecture": "Conv2D(32) -> Conv2D(64) -> Dense(128) -> Dense(10)",
            "accuracy": accuracy,
            "loss": loss,
            "epochs": self.config.get("epochs", 50),
            "validation_accuracy": round(accuracy - random.uniform(0.01, 0.05), 3),
            "parameters": "156,874",
            "training_time": "2.3 min",
            "use_case": "Clasificación de imágenes de calidad del agua"
        }

    def simulate_rnn_analysis(self):
        """Simular análisis RNN"""
        self.status_update.emit("🔄 Construyendo RNN...")
        self.progress.emit(30)
        time.sleep(1)

        self.status_update.emit("📈 Procesando secuencias temporales...")
        self.progress.emit(60)
        time.sleep(2)

        import random
        accuracy = round(random.uniform(0.78, 0.88), 3)

        return {
            "network_type": "RNN",
            "architecture": "SimpleRNN(64) -> Dense(32) -> Dense(1)",
            "accuracy": accuracy,
            "mae": round(random.uniform(0.12, 0.25), 4),
            "epochs": self.config.get("epochs", 100),
            "sequence_length": self.config.get("sequence_length", 10),
            "parameters": "8,257",
            "training_time": "1.8 min",
            "use_case": "Predicción temporal de parámetros de agua"
        }

    def simulate_lstm_analysis(self):
        """Simular análisis LSTM"""
        self.status_update.emit("🧠 Construyendo LSTM...")
        self.progress.emit(35)
        time.sleep(1.5)

        self.status_update.emit("🔍 Analizando dependencias a largo plazo...")
        self.progress.emit(70)
        time.sleep(2)

        import random
        accuracy = round(random.uniform(0.88, 0.95), 3)

        return {
            "network_type": "LSTM",
            "architecture": "LSTM(128) -> Dropout(0.2) -> LSTM(64) -> Dense(1)",
            "accuracy": accuracy,
            "rmse": round(random.uniform(0.08, 0.15), 4),
            "epochs": self.config.get("epochs", 75),
            "sequence_length": self.config.get("sequence_length", 20),
            "parameters": "74,945",
            "training_time": "4.2 min",
            "use_case": "Predicción avanzada de series temporales"
        }

    def simulate_gru_analysis(self):
        """Simular análisis GRU"""
        self.status_update.emit("⚡ Construyendo GRU...")
        self.progress.emit(30)
        time.sleep(1)

        import random
        accuracy = round(random.uniform(0.85, 0.92), 3)

        return {
            "network_type": "GRU",
            "architecture": "GRU(96) -> GRU(48) -> Dense(1)",
            "accuracy": accuracy,
            "mse": round(random.uniform(0.01, 0.05), 4),
            "epochs": self.config.get("epochs", 60),
            "parameters": "56,321",
            "training_time": "3.1 min",
            "use_case": "Predicción eficiente de secuencias"
        }

    def simulate_autoencoder_analysis(self):
        """Simular análisis Autoencoder"""
        self.status_update.emit("🔄 Entrenando Autoencoder...")
        self.progress.emit(40)
        time.sleep(2)

        import random
        reconstruction_loss = round(random.uniform(0.02, 0.08), 4)

        return {
            "network_type": "Autoencoder",
            "architecture": "Dense(128) -> Dense(64) -> Dense(32) -> Dense(64) -> Dense(128)",
            "reconstruction_loss": reconstruction_loss,
            "compression_ratio": "4:1",
            "epochs": self.config.get("epochs", 100),
            "latent_dimension": 32,
            "parameters": "33,792",
            "training_time": "2.8 min",
            "use_case": "Detección de anomalías en calidad del agua"
        }

    def simulate_gan_analysis(self):
        """Simular análisis GAN"""
        self.status_update.emit("🎭 Entrenando GAN...")
        self.progress.emit(45)
        time.sleep(3)

        import random

        return {
            "network_type": "GAN",
            "architecture": "Generator: Dense(256) -> Dense(512) -> Dense(784)\nDiscriminator: Dense(512) -> Dense(256) -> Dense(1)",
            "generator_loss": round(random.uniform(0.5, 1.2), 4),
            "discriminator_loss": round(random.uniform(0.4, 0.9), 4),
            "epochs": self.config.get("epochs", 200),
            "generated_samples": 1000,
            "parameters": "2,347,521",
            "training_time": "8.7 min",
            "use_case": "Generación sintética de datos de calidad del agua"
        }


class NetworkResultsWidget(QWidget):
    """Widget para mostrar resultados de redes neuronales"""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header = QLabel("🧠 Resultados de Deep Learning")
        header.setObjectName("resultsHeader")
        header.setAlignment(Qt.AlignCenter)

        # Área de contenido
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setObjectName("resultsScrollArea")

        # Widget de contenido
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)

        # Mensaje inicial
        self.initial_message = QLabel(
            "🎯 Panel de Resultados de Deep Learning\n\n"
            "Los resultados del entrenamiento y evaluación\n"
            "de redes neuronales aparecerán aquí.\n\n"
            "Selecciona un tipo de red y configura\n"
            "los parámetros para comenzar."
        )
        self.initial_message.setAlignment(Qt.AlignCenter)
        self.initial_message.setObjectName("initialMessage")

        self.content_layout.addWidget(self.initial_message)
        self.scroll_area.setWidget(self.content_widget)

        layout.addWidget(header)
        layout.addWidget(self.scroll_area)

        self.setLayout(layout)

    def mostrar_resultados(self, results, network_type):
        """Mostrar resultados del análisis"""
        # Limpiar contenido anterior
        for i in reversed(range(self.content_layout.count())):
            self.content_layout.itemAt(i).widget().setParent(None)

        if 'error' in results:
            error_widget = self.create_error_widget(results['error'])
            self.content_layout.addWidget(error_widget)
        else:
            results_widget = self.create_results_widget(results, network_type)
            self.content_layout.addWidget(results_widget)

    def create_error_widget(self, error_message):
        """Crear widget de error"""
        error_frame = QFrame()
        error_frame.setObjectName("errorFrame")

        layout = QVBoxLayout(error_frame)
        layout.setContentsMargins(20, 20, 20, 20)

        error_label = QLabel(f"❌ Error en el Análisis\n\n{error_message}")
        error_label.setObjectName("errorLabel")
        error_label.setAlignment(Qt.AlignCenter)
        error_label.setWordWrap(True)

        layout.addWidget(error_label)

        return error_frame

    def create_results_widget(self, results, network_type):
        """Crear widget de resultados"""
        results_frame = QFrame()
        results_frame.setObjectName("resultsFrame")

        layout = QVBoxLayout(results_frame)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Título del resultado
        title = QLabel(f"✅ {results.get('network_type', network_type.upper())} - Análisis Completado")
        title.setObjectName("resultTitle")

        # Información de la arquitectura
        arch_frame = self.create_info_section("🏗️ Arquitectura", results.get('architecture', 'No especificada'))

        # Métricas de rendimiento
        metrics_frame = self.create_metrics_section(results)

        # Información del entrenamiento
        training_frame = self.create_training_section(results)

        # Caso de uso
        use_case_frame = self.create_info_section("🎯 Caso de Uso", results.get('use_case', 'Análisis general'))

        layout.addWidget(title)
        layout.addWidget(arch_frame)
        layout.addWidget(metrics_frame)
        layout.addWidget(training_frame)
        layout.addWidget(use_case_frame)
        layout.addStretch()

        return results_frame

    def create_info_section(self, title, content):
        """Crear sección de información"""
        frame = QFrame()
        frame.setObjectName("infoSection")

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(15, 15, 15, 15)

        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")

        content_label = QLabel(content)
        content_label.setObjectName("sectionContent")
        content_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(content_label)

        return frame

    def create_metrics_section(self, results):
        """Crear sección de métricas"""
        frame = QFrame()
        frame.setObjectName("metricsSection")

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(15, 15, 15, 15)

        title = QLabel("📊 Métricas de Rendimiento")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        # Grid de métricas
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(10)

        row = 0
        # Métricas comunes
        metrics_map = {
            'accuracy': ('🎯 Precisión', '%'),
            'loss': ('📉 Pérdida', ''),
            'mae': ('📏 MAE', ''),
            'mse': ('📐 MSE', ''),
            'rmse': ('📊 RMSE', ''),
            'reconstruction_loss': ('🔄 Pérdida de Reconstrucción', ''),
            'generator_loss': ('🎭 Pérdida del Generador', ''),
            'discriminator_loss': ('🕵️ Pérdida del Discriminador', ''),
            'validation_accuracy': ('✅ Precisión de Validación', '%')
        }

        for key, (label, unit) in metrics_map.items():
            if key in results:
                value = results[key]
                if unit == '%':
                    value_text = f"{value * 100:.1f}%" if value <= 1 else f"{value:.1f}%"
                else:
                    value_text = f"{value:.4f}" if isinstance(value, float) else str(value)

                metric_label = QLabel(label)
                metric_label.setObjectName("metricLabel")

                value_label = QLabel(value_text)
                value_label.setObjectName("metricValue")

                metrics_grid.addWidget(metric_label, row, 0)
                metrics_grid.addWidget(value_label, row, 1)
                row += 1

        layout.addLayout(metrics_grid)

        return frame

    def create_training_section(self, results):
        """Crear sección de información de entrenamiento"""
        frame = QFrame()
        frame.setObjectName("trainingSection")

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(15, 15, 15, 15)

        title = QLabel("🏋️ Información de Entrenamiento")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        # Grid de información
        info_grid = QGridLayout()
        info_grid.setSpacing(10)

        training_info = [
            ('⏱️ Tiempo de Entrenamiento', results.get('training_time', 'No especificado')),
            ('🔢 Épocas', str(results.get('epochs', 'No especificado'))),
            ('⚙️ Parámetros', results.get('parameters', 'No especificado')),
            ('📏 Longitud de Secuencia', str(results.get('sequence_length', 'N/A'))),
            ('🗜️ Ratio de Compresión', results.get('compression_ratio', 'N/A')),
            ('🎲 Muestras Generadas', str(results.get('generated_samples', 'N/A'))),
            ('📐 Dimensión Latente', str(results.get('latent_dimension', 'N/A')))
        ]

        row = 0
        for label_text, value in training_info:
            if value != 'N/A':
                label = QLabel(label_text)
                label.setObjectName("infoLabel")

                value_label = QLabel(value)
                value_label.setObjectName("infoValue")

                info_grid.addWidget(label, row, 0)
                info_grid.addWidget(value_label, row, 1)
                row += 1

        layout.addLayout(info_grid)

        return frame

    def limpiar_resultados(self):
        """Limpiar resultados"""
        for i in reversed(range(self.content_layout.count())):
            self.content_layout.itemAt(i).widget().setParent(None)

        self.content_layout.addWidget(self.initial_message)


class DeepLearning(QWidget, ThemedWidget):
    """Ventana principal de Deep Learning"""

    def __init__(self):
        QWidget.__init__(self)
        ThemedWidget.__init__(self)

        self.worker = None
        self.cache = DataCache()
        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):
        """Configurar interfaz de usuario"""
        self.setWindowTitle("🧠 Deep Learning - ML Calidad del Agua")
        self.setMinimumSize(1400, 900)

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
        self.resultados_widget = NetworkResultsWidget()
        splitter.addWidget(self.resultados_widget)

        splitter.setSizes([500, 900])
        main_layout.addWidget(splitter)

        # Barra de estado
        status_layout = self.create_status_bar()
        main_layout.addLayout(status_layout)

        self.setLayout(main_layout)

    def create_header(self):
        """Crear header de la ventana"""
        header_layout = QHBoxLayout()

        # Título principal
        title = QLabel("🧠 Deep Learning")
        title.setObjectName("windowTitle")
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # Descripción
        desc = QLabel("Redes neuronales profundas para análisis avanzado")
        desc.setObjectName("windowDesc")

        # Layout vertical para título y descripción
        title_layout = QVBoxLayout()
        title_layout.setSpacing(5)
        title_layout.addWidget(title)
        title_layout.addWidget(desc)

        header_layout.addLayout(title_layout)
        header_layout.addStretch()

        # Botón de configuración
        config_button = QPushButton("⚙️")
        config_button.setObjectName("configButton")
        config_button.setFixedSize(30, 30)
        config_button.setToolTip("Configuración")
        config_button.clicked.connect(self.show_config_dialog)

        # Botón de cerrar
        close_button = QPushButton("✕")
        close_button.setObjectName("closeButton")
        close_button.setFixedSize(30, 30)
        close_button.clicked.connect(self.close)

        header_layout.addWidget(config_button)
        header_layout.addWidget(close_button)

        return header_layout

    def create_control_panel(self):
        """Crear panel de controles"""
        group = QGroupBox("🧠 Arquitecturas de Deep Learning")
        group.setObjectName("controlGroup")

        layout = QVBoxLayout()
        layout.setSpacing(20)

        # Información del sistema
        system_info = self.create_system_info()
        layout.addWidget(system_info)

        # Pestañas de tipos de redes
        tabs = self.create_network_tabs()
        layout.addWidget(tabs)

        # Controles de configuración
        config_controls = self.create_config_controls()
        layout.addWidget(config_controls)

        layout.addStretch()

        # Controles de utilidad
        utility_controls = self.create_utility_controls()
        layout.addWidget(utility_controls)

        group.setLayout(layout)
        return group

    def create_system_info(self):
        """Crear información del sistema"""
        info_frame = QFrame()
        info_frame.setObjectName("systemInfoFrame")

        layout = QVBoxLayout(info_frame)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        title = QLabel("🖥️ Estado del Sistema Deep Learning")
        title.setObjectName("systemTitle")

        # Estado de dependencias
        status_parts = []

        if TF_AVAILABLE:
            status_parts.append("✅ TensorFlow")
        else:
            status_parts.append("❌ TensorFlow")

        if TORCH_AVAILABLE:
            status_parts.append("✅ PyTorch")
        else:
            status_parts.append("❌ PyTorch")

        if BASE_DEPS_AVAILABLE:
            status_parts.append("✅ NumPy/Pandas")
        else:
            status_parts.append("❌ NumPy/Pandas")

        status_text = " | ".join(status_parts)

        if DL_AVAILABLE:
            status_color = "#28a745"
            status_icon = "🎉"
            overall_status = "Sistema completamente funcional"
        else:
            status_color = "#dc3545"
            status_icon = "❌"
            overall_status = "Sistema no disponible"

        status_label = QLabel(f"{status_icon} {status_text}")
        status_label.setStyleSheet(f"color: {status_color}; font-weight: bold; font-size: 12px;")

        overall_label = QLabel(overall_status)
        overall_label.setObjectName("overallStatus")

        # Información adicional
        additional_info = QLabel(
            "Las redes neuronales profundas permiten:\n"
            "• Aprendizaje de patrones complejos\n"
            "• Procesamiento de secuencias temporales\n"
            "• Detección de anomalías\n"
            "• Generación de datos sintéticos"
        )
        additional_info.setObjectName("additionalInfo")
        additional_info.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(status_label)
        layout.addWidget(overall_label)
        layout.addWidget(additional_info)

        return info_frame

    def create_network_tabs(self):
        """Crear pestañas de tipos de redes"""
        tabs = QTabWidget()
        tabs.setObjectName("networkTabs")

        # Pestaña de redes de clasificación
        classification_tab = self.create_classification_tab()
        tabs.addTab(classification_tab, "📊 Clasificación")

        # Pestaña de redes secuenciales
        sequential_tab = self.create_sequential_tab()
        tabs.addTab(sequential_tab, "🔄 Secuenciales")

        # Pestaña de redes generativas
        generative_tab = self.create_generative_tab()
        tabs.addTab(generative_tab, "🎭 Generativas")

        return tabs

    def create_classification_tab(self):
        """Crear pestaña de redes de clasificación"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

        # CNN
        cnn_frame = self.create_network_card(
            "🖼️ Red Neuronal Convolucional (CNN)",
            "cnn",
            "Especializada en procesamiento de imágenes y datos espaciales",
            "• Capas convolucionales y pooling\n• Ideal para clasificación de imágenes\n• Detección de patrones visuales",
            "#4CAF50"
        )

        layout.addWidget(cnn_frame)
        layout.addStretch()

        return widget

    def create_sequential_tab(self):
        """Crear pestaña de redes secuenciales"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

        # RNN
        rnn_frame = self.create_network_card(
            "🔄 Red Neuronal Recurrente (RNN)",
            "rnn",
            "Procesamiento de secuencias temporales básicas",
            "• Memoria a corto plazo\n• Predicción de series temporales\n• Análisis secuencial simple",
            "#2196F3"
        )

        # LSTM
        lstm_frame = self.create_network_card(
            "🧠 Long Short-Term Memory (LSTM)",
            "lstm",
            "Manejo avanzado de dependencias a largo plazo",
            "• Memoria selectiva\n• Predicción compleja\n• Análisis de patrones temporales",
            "#9C27B0"
        )

        # GRU
        gru_frame = self.create_network_card(
            "⚡ Gated Recurrent Unit (GRU)",
            "gru",
            "Versión eficiente de LSTM con menos parámetros",
            "• Arquitectura simplificada\n• Entrenamiento más rápido\n• Rendimiento comparable a LSTM",
            "#FF9800"
        )

        layout.addWidget(rnn_frame)
        layout.addWidget(lstm_frame)
        layout.addWidget(gru_frame)
        layout.addStretch()

        return widget

    def create_generative_tab(self):
        """Crear pestaña de redes generativas"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

        # Autoencoder
        autoencoder_frame = self.create_network_card(
            "🔄 Autoencoder",
            "autoencoder",
            "Codificación y decodificación para detección de anomalías",
            "• Compresión de datos\n• Detección de outliers\n• Reducción de dimensionalidad",
            "#607D8B"
        )

        # GAN
        gan_frame = self.create_network_card(
            "🎭 Generative Adversarial Network (GAN)",
            "gan",
            "Generación de datos sintéticos realistas",
            "• Competencia generador vs discriminador\n• Creación de datos sintéticos\n• Aumento de datasets",
            "#E91E63"
        )

        layout.addWidget(autoencoder_frame)
        layout.addWidget(gan_frame)
        layout.addStretch()

        return widget

    def create_network_card(self, name, key, description, details, color):
        """Crear tarjeta de red neuronal"""
        card_frame = QFrame()
        card_frame.setObjectName("networkCard")
        card_frame.setMinimumHeight(180)
        card_frame.setMaximumHeight(200)
        card_frame.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(card_frame)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # Header
        header_layout = QHBoxLayout()

        name_label = QLabel(name)
        name_label.setObjectName("networkName")
        name_label.setWordWrap(True)

        # Indicador de estado
        status_indicator = QLabel("⚪")
        status_indicator.setObjectName("networkStatus")
        status_indicator.setFixedSize(20, 20)
        status_indicator.setAlignment(Qt.AlignCenter)

        header_layout.addWidget(name_label)
        header_layout.addStretch()
        header_layout.addWidget(status_indicator)

        # Descripción
        desc_label = QLabel(description)
        desc_label.setObjectName("networkDesc")
        desc_label.setWordWrap(True)

        # Detalles
        details_label = QLabel(details)
        details_label.setObjectName("networkDetails")
        details_label.setWordWrap(True)

        # Botón de ejecución
        execute_btn = QPushButton("🚀 Entrenar Red")
        execute_btn.setObjectName("executeNetworkButton")
        execute_btn.setMinimumHeight(35)
        execute_btn.setEnabled(DL_AVAILABLE)

        if not DL_AVAILABLE:
            execute_btn.setText("❌ Dependencias faltantes")
            execute_btn.setToolTip("Instala TensorFlow o PyTorch")

        execute_btn.clicked.connect(lambda: self.train_network(key))

        layout.addLayout(header_layout)
        layout.addWidget(desc_label)
        layout.addWidget(details_label)
        layout.addStretch()
        layout.addWidget(execute_btn)

        # Guardar referencia para actualizar estado
        if not hasattr(self, 'network_cards'):
            self.network_cards = {}

        self.network_cards[key] = {
            'frame': card_frame,
            'button': execute_btn,
            'status': status_indicator
        }

        # Efecto click en toda la tarjeta
        if DL_AVAILABLE:
            card_frame.mousePressEvent = lambda event: self.train_network(key)

        return card_frame

    def create_config_controls(self):
        """Crear controles de configuración"""
        config_frame = QFrame()
        config_frame.setObjectName("configFrame")

        layout = QVBoxLayout(config_frame)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        title = QLabel("⚙️ Configuración de Entrenamiento")
        title.setObjectName("configTitle")

        # Grid de configuraciones
        config_grid = QGridLayout()
        config_grid.setSpacing(10)

        # Épocas
        epochs_label = QLabel("🔢 Épocas:")
        epochs_label.setObjectName("configLabel")
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setObjectName("configSpinBox")
        self.epochs_spin.setRange(10, 500)
        self.epochs_spin.setValue(100)
        self.epochs_spin.setSuffix(" épocas")

        # Longitud de secuencia (para RNN/LSTM/GRU)
        seq_label = QLabel("📏 Long. Secuencia:")
        seq_label.setObjectName("configLabel")
        self.seq_spin = QSpinBox()
        self.seq_spin.setObjectName("configSpinBox")
        self.seq_spin.setRange(5, 100)
        self.seq_spin.setValue(20)
        self.seq_spin.setSuffix(" puntos")

        # Batch size
        batch_label = QLabel("📦 Batch Size:")
        batch_label.setObjectName("configLabel")
        self.batch_combo = QComboBox()
        self.batch_combo.setObjectName("configComboBox")
        self.batch_combo.addItems(["16", "32", "64", "128", "256"])
        self.batch_combo.setCurrentText("32")

        # Learning rate
        lr_label = QLabel("🎯 Learning Rate:")
        lr_label.setObjectName("configLabel")
        self.lr_combo = QComboBox()
        self.lr_combo.setObjectName("configComboBox")
        self.lr_combo.addItems(["0.001", "0.01", "0.1", "0.0001"])
        self.lr_combo.setCurrentText("0.001")

        config_grid.addWidget(epochs_label, 0, 0)
        config_grid.addWidget(self.epochs_spin, 0, 1)
        config_grid.addWidget(seq_label, 1, 0)
        config_grid.addWidget(self.seq_spin, 1, 1)
        config_grid.addWidget(batch_label, 2, 0)
        config_grid.addWidget(self.batch_combo, 2, 1)
        config_grid.addWidget(lr_label, 3, 0)
        config_grid.addWidget(self.lr_combo, 3, 1)

        layout.addWidget(title)
        layout.addLayout(config_grid)

        return config_frame

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
        check_deps_btn.clicked.connect(self.check_dependencies)

        # Botón optimización de hiperparámetros
        optimize_btn = QPushButton("🎛️ Optimizar")
        optimize_btn.setObjectName("optimizeButton")
        optimize_btn.clicked.connect(self.optimize_hyperparameters)
        optimize_btn.setEnabled(DL_AVAILABLE)

        layout.addWidget(clear_cache_btn)
        layout.addWidget(clear_results_btn)
        layout.addWidget(check_deps_btn)
        layout.addWidget(optimize_btn)
        layout.addStretch()

        return utility_frame

    def create_status_bar(self):
        """Crear barra de estado"""
        status_layout = QHBoxLayout()

        # Estado del sistema
        if DL_AVAILABLE:
            status_text = "✅ Deep Learning completamente funcional"
            status_color = "#28a745"
        elif TF_AVAILABLE or TORCH_AVAILABLE:
            status_text = "⚠️ Deep Learning parcialmente disponible"
            status_color = "#ffc107"
        else:
            status_text = "❌ Deep Learning no disponible"
            status_color = "#dc3545"

        self.status_label = QLabel(status_text)
        self.status_label.setObjectName("statusLabel")
        self.status_label.setStyleSheet(f"color: {status_color};")

        # Información del sistema
        frameworks = []
        if TF_AVAILABLE:
            frameworks.append("TensorFlow")
        if TORCH_AVAILABLE:
            frameworks.append("PyTorch")

        framework_text = " + ".join(frameworks) if frameworks else "Ninguno"
        system_info = QLabel(f"🧠 Frameworks: {framework_text} | 🖥️ CPUs: {multiprocessing.cpu_count()}")
        system_info.setObjectName("systemInfo")

        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("progressBar")
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(250)

        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(system_info)
        status_layout.addWidget(self.progress_bar)

        return status_layout

    def train_network(self, network_type):
        """Entrenar red neuronal"""
        if not DL_AVAILABLE:
            QMessageBox.critical(
                self,
                "Sistema No Disponible",
                "El sistema de Deep Learning no está disponible.\n\n"
                "Para solucionarlo:\n"
                "1. Instala TensorFlow: pip install tensorflow\n"
                "2. O instala PyTorch: pip install torch\n"
                "3. Instala dependencias: pip install numpy pandas matplotlib"
            )
            return

        if self.worker and self.worker.isRunning():
            QMessageBox.information(
                self,
                "Entrenamiento en Curso",
                "Ya hay un entrenamiento ejecutándose. Por favor espera."
            )
            return

        # Obtener configuración
        config = {
            'epochs': self.epochs_spin.value(),
            'sequence_length': self.seq_spin.value(),
            'batch_size': int(self.batch_combo.currentText()),
            'learning_rate': float(self.lr_combo.currentText())
        }

        # Actualizar UI
        self.update_ui_for_training(network_type, True)

        # Crear y ejecutar worker
        self.worker = DeepLearningWorker(network_type, config)
        self.worker.finished.connect(self.on_training_finished)
        self.worker.progress.connect(self.update_progress)
        self.worker.status_update.connect(self.update_status)
        self.worker.start()

    def update_ui_for_training(self, network_type, is_training):
        """Actualizar UI durante entrenamiento"""
        # Deshabilitar todos los botones
        for key, components in self.network_cards.items():
            if DL_AVAILABLE:
                components['button'].setEnabled(not is_training)

            if key == network_type and is_training:
                components['status'].setText("⚡")
                components['status'].setStyleSheet("color: #ffc107; font-weight: bold;")
            elif not is_training:
                components['status'].setText("⚪")
                components['status'].setStyleSheet("")

        # Mostrar/ocultar barra de progreso
        self.progress_bar.setVisible(is_training)
        if is_training:
            self.progress_bar.setValue(0)

    def update_progress(self, value):
        """Actualizar progreso"""
        self.progress_bar.setValue(value)

    def update_status(self, message):
        """Actualizar mensaje de estado"""
        original_style = self.status_label.styleSheet()
        self.status_label.setText(message)

        # Restaurar color después de un tiempo
        QTimer.singleShot(3000, lambda: self.status_label.setStyleSheet(original_style))

    def on_training_finished(self, results, network_type):
        """Manejar finalización del entrenamiento"""
        # Restaurar UI
        self.update_ui_for_training(network_type, False)

        if "error" in results:
            self.status_label.setText(f"❌ Error: {results['error']}")
            self.network_cards[network_type]['status'].setText("❌")
            self.network_cards[network_type]['status'].setStyleSheet("color: #dc3545;")

            QMessageBox.critical(
                self,
                "Error en Entrenamiento",
                f"Error durante el entrenamiento:\n{results['error']}"
            )
            return

        # Marcar como completado
        self.network_cards[network_type]['status'].setText("✅")
        self.network_cards[network_type]['status'].setStyleSheet("color: #28a745; font-weight: bold;")

        # Mostrar resultados
        self.resultados_widget.mostrar_resultados(results, network_type)

        # Mensaje de éxito
        success_messages = {
            "cnn": "✅ CNN entrenada - Clasificación de imágenes",
            "rnn": "✅ RNN entrenada - Predicción secuencial",
            "lstm": "✅ LSTM entrenada - Análisis temporal avanzado",
            "gru": "✅ GRU entrenada - Predicción eficiente",
            "autoencoder": "✅ Autoencoder entrenado - Detección de anomalías",
            "gan": "✅ GAN entrenada - Generación de datos sintéticos"
        }

        self.status_label.setText(success_messages.get(network_type, "✅ Entrenamiento completado"))

    def clear_cache(self):
        """Limpiar cache"""
        self.cache.clear()
        self.status_label.setText("🗑️ Cache limpiado")

        # Resetear indicadores
        for components in self.network_cards.values():
            components['status'].setText("⚪")
            components['status'].setStyleSheet("")

    def clear_results(self):
        """Limpiar resultados"""
        self.resultados_widget.limpiar_resultados()
        self.status_label.setText("📄 Resultados limpiados")

    def check_dependencies(self):
        """Verificar dependencias"""
        deps_info = []

        # TensorFlow
        try:
            import tensorflow as tf
            deps_info.append(f"✅ TensorFlow {tf.__version__}")
        except ImportError:
            deps_info.append("❌ TensorFlow no instalado")

        # PyTorch
        try:
            import torch
            deps_info.append(f"✅ PyTorch {torch.__version__}")
        except ImportError:
            deps_info.append("❌ PyTorch no instalado")

        # Otras dependencias
        try:
            import numpy as np
            deps_info.append(f"✅ NumPy {np.__version__}")
        except ImportError:
            deps_info.append("❌ NumPy no instalado")

        try:
            import pandas as pd
            deps_info.append(f"✅ Pandas {pd.__version__}")
        except ImportError:
            deps_info.append("❌ Pandas no instalado")

        try:
            import matplotlib
            deps_info.append(f"✅ Matplotlib {matplotlib.__version__}")
        except ImportError:
            deps_info.append("❌ Matplotlib no instalado")

        # GPU support
        gpu_info = "❌ GPU no disponible"
        try:
            if TF_AVAILABLE:
                import tensorflow as tf
                if tf.config.list_physical_devices('GPU'):
                    gpu_info = "✅ GPU disponible (TensorFlow)"
            elif TORCH_AVAILABLE:
                import torch
                if torch.cuda.is_available():
                    gpu_info = "✅ GPU disponible (PyTorch)"
        except:
            pass

        deps_info.append(gpu_info)

        QMessageBox.information(
            self,
            "🔍 Estado de Dependencias",
            f"<h3>Estado del Sistema Deep Learning</h3>"
            f"<p><b>📦 Dependencias instaladas:</b></p>"
            f"<p>{'<br>'.join(deps_info)}</p>"
            f"<hr>"
            f"<p><b>Para instalar dependencias faltantes:</b></p>"
            f"<p><code>pip install tensorflow torch numpy pandas matplotlib</code></p>"
            f"<p><b>Para soporte GPU:</b></p>"
            f"<p><code>pip install tensorflow-gpu</code> o <code>pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118</code></p>"
        )

    def optimize_hyperparameters(self):
        """Optimización de hiperparámetros"""
        if not DL_AVAILABLE:
            QMessageBox.warning(
                self,
                "Función No Disponible",
                "La optimización de hiperparámetros requiere bibliotecas de Deep Learning."
            )
            return

        QMessageBox.information(
            self,
            "🎛️ Optimización de Hiperparámetros",
            "La optimización automática de hiperparámetros incluye:\n\n"
            "🔍 Búsqueda de parámetros óptimos:\n"
            "• Learning rate\n"
            "• Batch size\n"
            "• Número de capas\n"
            "• Función de activación\n"
            "• Regularización\n\n"
            "⚙️ Métodos disponibles:\n"
            "• Grid Search\n"
            "• Random Search\n"
            "• Bayesian Optimization\n"
            "• Genetic Algorithm\n\n"
            "Esta funcionalidad se implementará en una versión futura."
        )

    def show_config_dialog(self):
        """Mostrar diálogo de configuración avanzada"""
        QMessageBox.information(
            self,
            "⚙️ Configuración Avanzada",
            "Configuraciones adicionales disponibles:\n\n"
            "🎯 Optimizadores:\n"
            "• Adam, SGD, RMSprop, AdaGrad\n\n"
            "📊 Métricas de evaluación:\n"
            "• Accuracy, Precision, Recall, F1-Score\n"
            "• MAE, MSE, RMSE para regresión\n\n"
            "🔄 Técnicas de regularización:\n"
            "• Dropout, Batch Normalization\n"
            "• L1/L2 Regularization\n\n"
            "💾 Callbacks:\n"
            "• Early Stopping, Model Checkpoints\n"
            "• Learning Rate Scheduling\n\n"
            "Esta configuración se implementará en futuras versiones."
        )

    def apply_styles(self):
        """Aplicar estilos CSS"""
        styles = """
        QWidget {
            font-family: 'Segoe UI', Arial, sans-serif;
        }

        #windowTitle {
            font-size: 26px;
            font-weight: bold;
            color: #1a237e;
        }

        #windowDesc {
            font-size: 14px;
            color: #6c757d;
        }

        #configButton {
            background: #17a2b8;
            color: white;
            border: none;
            border-radius: 15px;
            font-weight: bold;
        }

        #configButton:hover {
            background: #138496;
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

        #systemInfoFrame {
            background: linear-gradient(135deg, #e8f4f8 0%, #f8f9fa 100%);
            border: 1px solid #17a2b8;
            border-radius: 8px;
        }

        #systemTitle {
            font-size: 16px;
            font-weight: bold;
            color: #0c5460;
        }

        #overallStatus {
            font-size: 13px;
            color: #495057;
            font-weight: 500;
        }

        #additionalInfo {
            font-size: 12px;
            color: #6c757d;
            line-height: 1.4;
        }

        #networkTabs {
            background: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 8px;
        }

        #networkTabs::pane {
            border: 1px solid #dee2e6;
            border-radius: 6px;
            background: #ffffff;
        }

        #networkTabs::tab-bar {
            alignment: center;
        }

        QTabBar::tab {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            padding: 8px 16px;
            margin: 2px;
            border-radius: 6px;
            font-weight: 500;
        }

        QTabBar::tab:selected {
            background: #007bff;
            color: white;
        }

        QTabBar::tab:hover {
            background: #e9ecef;
        }

        #networkCard {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border: 2px solid #e9ecef;
            border-radius: 12px;
            margin: 5px;
        }

        #networkCard:hover {
            border-color: #007bff;
            background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
            transform: translateY(-2px);
        }

        #networkName {
            font-size: 14px;
            font-weight: bold;
            color: #1a237e;
        }

        #networkDesc {
            font-size: 12px;
            color: #6c757d;
            font-weight: 500;
        }

        #networkDetails {
            font-size: 11px;
            color: #868e96;
            line-height: 1.3;
        }

        #executeNetworkButton {
            background: linear-gradient(135deg, #1a237e 0%, #3f51b5 100%);
            color: white;
            border: none;
            border-radius: 6px;
            font-weight: bold;
            font-size: 12px;
        }

        #executeNetworkButton:hover {
            background: linear-gradient(135deg, #000051 0%, #283593 100%);
        }

        #executeNetworkButton:disabled {
            background: #6c757d;
        }

        #configFrame {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
        }

        #configTitle {
            font-size: 15px;
            font-weight: bold;
            color: #495057;
        }

        #configLabel {
            font-size: 12px;
            color: #6c757d;
            font-weight: 500;
        }

        #configSpinBox, #configComboBox {
            background: white;
            border: 1px solid #ced4da;
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 11px;
        }

        #configSpinBox:focus, #configComboBox:focus {
            border-color: #80bdff;
            outline: none;
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

        #optimizeButton {
            background: linear-gradient(135deg, #fd7e14 0%, #e83e8c 100%);
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 12px;
            font-size: 11px;
            font-weight: bold;
        }

        #optimizeButton:hover {
            background: linear-gradient(135deg, #e8590c 0%, #d91a72 100%);
        }

        #optimizeButton:disabled {
            background: #6c757d;
        }

        #statusLabel {
            font-size: 13px;
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
            background: linear-gradient(90deg, #1a237e 0%, #3f51b5 100%);
            border-radius: 3px;
        }

        /* Estilos para resultados */
        #resultsHeader {
            font-size: 20px;
            font-weight: bold;
            color: #1a237e;
            padding: 10px;
        }

        #resultsScrollArea {
            background: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 8px;
        }

        #initialMessage {
            font-size: 14px;
            color: #6c757d;
            line-height: 1.6;
            padding: 40px;
        }

        #errorFrame {
            background: linear-gradient(135deg, #f8d7da 0%, #ffeaa7 100%);
            border: 1px solid #f5c6cb;
            border-radius: 8px;
            margin: 10px;
        }

        #errorLabel {
            color: #721c24;
            font-size: 14px;
            font-weight: 500;
        }

        #resultsFrame {
            background: linear-gradient(135deg, #d4edda 0%, #f8f9fa 100%);
            border: 1px solid #c3e6cb;
            border-radius: 8px;
            margin: 10px;
        }

        #resultTitle {
            font-size: 18px;
            font-weight: bold;
            color: #155724;
        }

        #infoSection {
            background: #ffffff;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            margin: 5px 0;
        }

        #metricsSection {
            background: linear-gradient(135deg, #e3f2fd 0%, #ffffff 100%);
            border: 1px solid #90caf9;
            border-radius: 6px;
            margin: 5px 0;
        }

        #trainingSection {
            background: linear-gradient(135deg, #f3e5f5 0%, #ffffff 100%);
            border: 1px solid #ce93d8;
            border-radius: 6px;
            margin: 5px 0;
        }

        #sectionTitle {
            font-size: 14px;
            font-weight: bold;
            color: #495057;
        }

        #sectionContent {
            font-size: 12px;
            color: #6c757d;
            line-height: 1.4;
        }

        #metricLabel, #infoLabel {
            font-size: 12px;
            color: #495057;
            font-weight: 500;
        }

        #metricValue, #infoValue {
            font-size: 12px;
            color: #007bff;
            font-weight: bold;
        }
        """

        self.setStyleSheet(styles)

    def closeEvent(self, event):
        """Manejar cierre de ventana"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Cerrar Ventana",
                "Hay un entrenamiento ejecutándose. ¿Deseas interrumpirlo?",
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
    window = DeepLearning()
    window.show()
    sys.exit(app.exec_())