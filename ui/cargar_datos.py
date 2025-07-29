import sys
import pandas as pd
import json
import requests
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QHBoxLayout, QFrame,
                             QLabel, QPushButton, QFileDialog, QMessageBox,
                             QTabWidget, QWidget, QTableWidget, QTableWidgetItem,
                             QTextEdit, QProgressBar, QGroupBox, QGridLayout,
                             QCheckBox, QSplitter, QDialog, QLineEdit, QComboBox,
                             QDialogButtonBox, QFormLayout, QSpacerItem, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QAbstractTableModel, QVariant
from PyQt5.QtGui import QFont
import numpy as np
from datetime import datetime
import os
from ui.machine_learning.data_manager import get_data_manager, has_shared_data
from PyQt5.QtWidgets import QPushButton, QSizePolicy



# Importar sistema de temas
try:
    from darkmode.ui_theme_manager import ThemedWidget, ThemeManager
except ImportError:
    try:
        # from darkmode import ThemedWidget, ThemeManager  # COMENTADA
        raise ImportError("Comentado temporalmente")
    except ImportError:
        class ThemedWidget:
            def __init__(self):
                pass

            def apply_theme(self):
                pass


        class ThemeManager:
            def __init__(self):
                pass
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


class DataLoadingThread(QThread):
    """Hilo para carga de datos en background"""

    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    data_loaded = pyqtSignal(object)  # DataFrame
    error_occurred = pyqtSignal(str)

    def __init__(self, file_path, file_type):
        super().__init__()
        self.file_path = file_path
        self.file_type = file_type

    def run(self):
        """Ejecutar carga de datos"""
        try:
            self.status_update.emit("🔄 Iniciando carga de datos...")
            self.progress_update.emit(10)

            if self.file_type == 'csv':
                df = self.load_csv_optimized()
            elif self.file_type == 'excel':
                df = self.load_excel_optimized()
            else:
                raise ValueError(f"Tipo de archivo no soportado: {self.file_type}")

            self.progress_update.emit(90)
            self.status_update.emit("✅ Procesando datos...")

            # Validaciones básicas
            if df.empty:
                raise ValueError("El archivo está vacío")

            self.progress_update.emit(100)
            self.status_update.emit("✅ Datos cargados exitosamente")
            self.data_loaded.emit(df)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def load_csv_optimized(self):
        """Carga optimizada de CSV"""
        self.status_update.emit("📄 Analizando archivo CSV...")
        self.progress_update.emit(20)

        # Detectar encoding automáticamente
        encoding = 'utf-8'
        try:
            import chardet
            with open(self.file_path, 'rb') as f:
                sample = f.read(10000)
                encoding_result = chardet.detect(sample)
                encoding = encoding_result.get('encoding', 'utf-8')
            self.status_update.emit(f"🔍 Encoding detectado: {encoding}")
        except ImportError:
            self.status_update.emit("🔍 Usando encoding UTF-8")
        except Exception:
            self.status_update.emit("🔍 Usando encoding UTF-8 por defecto")

        self.progress_update.emit(30)

        try:
            # Detectar separador automáticamente
            with open(self.file_path, 'r', encoding=encoding) as f:
                first_line = f.readline()
                if '\t' in first_line and first_line.count('\t') > first_line.count(','):
                    separator = '\t'
                    self.status_update.emit("🔍 Separador detectado: Tabulador")
                elif ';' in first_line and first_line.count(';') > first_line.count(','):
                    separator = ';'
                    self.status_update.emit("🔍 Separador detectado: Punto y coma")
                else:
                    separator = ','
                    self.status_update.emit("🔍 Separador detectado: Coma")

            self.progress_update.emit(50)
            df = pd.read_csv(self.file_path, encoding=encoding, sep=separator, low_memory=False)

        except UnicodeDecodeError:
            # Fallback con diferentes encodings
            encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            separators_to_try = ['\t', ',', ';']
            df = None

            for enc in encodings_to_try:
                for sep in separators_to_try:
                    try:
                        self.status_update.emit(f"🔄 Probando: {enc} con separador '{sep if sep != '\t' else 'TAB'}'")
                        df = pd.read_csv(self.file_path, encoding=enc, sep=sep, low_memory=False)
                        if len(df.columns) > 1:
                            break
                    except:
                        continue
                if df is not None and len(df.columns) > 1:
                    break

            if df is None or len(df.columns) <= 1:
                try:
                    df = pd.read_csv(self.file_path, encoding='utf-8', sep=None, engine='python', low_memory=False)
                except:
                    df = pd.read_csv(self.file_path, encoding='utf-8', errors='ignore', sep=None, engine='python',
                                     low_memory=False)

        self.progress_update.emit(80)
        return df

    def load_excel_optimized(self):
        """Carga optimizada de Excel"""
        self.status_update.emit("📊 Cargando archivo Excel...")
        self.progress_update.emit(20)

        engine = 'openpyxl' if self.file_path.endswith('.xlsx') else 'xlrd'
        self.progress_update.emit(40)
        df = pd.read_excel(self.file_path, engine=engine)
        self.progress_update.emit(80)
        return df


class APIConnectionDialog(QDialog):
    """Diálogo para configurar conexión API"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🌐 Configurar Conexión API")
        self.setModal(True)
        self.setFixedSize(500, 400)
        self.setup_ui()

    def setup_ui(self):
        """Configurar interfaz del diálogo"""
        layout = QVBoxLayout()

        # Título
        title = QLabel("🌐 Conexión a API de Calidad del Agua")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50; margin: 10px;")
        layout.addWidget(title)

        # Formulario
        form_layout = QFormLayout()

        # Tipo de API
        self.api_type = QComboBox()
        self.api_type.addItems([
            "IDEAM (Colombia)",
            "EPA (Estados Unidos)",
            "CONAGUA (México)",
            "ANA (Brasil)",
            "API Personalizada"
        ])
        self.api_type.currentTextChanged.connect(self.on_api_type_changed)
        form_layout.addRow("📡 Tipo de API:", self.api_type)

        # URL Base
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://api.ejemplo.com/agua")
        form_layout.addRow("🔗 URL Base:", self.url_input)

        # Token/API Key
        self.token_input = QLineEdit()
        self.token_input.setPlaceholderText("Token de autenticación (opcional)")
        self.token_input.setEchoMode(QLineEdit.Password)
        form_layout.addRow("🔑 API Key:", self.token_input)

        # Endpoint específico
        self.endpoint_input = QLineEdit()
        self.endpoint_input.setPlaceholderText("/datos/calidad")
        form_layout.addRow("📍 Endpoint:", self.endpoint_input)

        # Parámetros adicionales
        self.params_input = QTextEdit()
        self.params_input.setMaximumHeight(80)
        self.params_input.setPlaceholderText('{"region": "Bogotá", "limit": 1000}')
        form_layout.addRow("⚙️ Parámetros JSON:", self.params_input)

        layout.addLayout(form_layout)

        # Información de ayuda
        help_text = QLabel(
            "💡 Consejos:\n"
            "• Selecciona el tipo de API para configuración automática\n"
            "• Los parámetros JSON son opcionales\n"
            "• Se realizará una prueba de conexión antes de cargar datos"
        )
        help_text.setStyleSheet(
            "color: #6c757d; font-size: 11px; margin: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px;")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        # Spacer
        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Botones
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        test_button = QPushButton("🧪 Probar Conexión")
        test_button.clicked.connect(self.test_connection)
        button_box.addButton(test_button, QDialogButtonBox.ActionRole)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

        # Configurar API por defecto
        self.on_api_type_changed("IDEAM (Colombia)")

    def on_api_type_changed(self, api_type):
        """Configurar automáticamente según el tipo de API"""
        configs = {
            "IDEAM (Colombia)": {
                "url": "http://dhime.ideam.gov.co/api",
                "endpoint": "/datos/calidad-agua",
                "params": '{"departamento": "cundinamarca", "limite": 1000}'
            },
            "EPA (Estados Unidos)": {
                "url": "https://www.waterqualitydata.us/data",
                "endpoint": "/Result/search",
                "params": '{"statecode": "US:11", "sampleMedia": "Water"}'
            },
            "CONAGUA (México)": {
                "url": "https://snia.conagua.gob.mx/api",
                "endpoint": "/calidad-agua",
                "params": '{"estado": "mexico", "limite": 1000}'
            },
            "ANA (Brasil)": {
                "url": "https://dadosabertos.ana.gov.br/api",
                "endpoint": "/qualidade-agua",
                "params": '{"estado": "SP", "limite": 1000}'
            },
            "API Personalizada": {
                "url": "",
                "endpoint": "",
                "params": "{}"
            }
        }

        if api_type in configs:
            config = configs[api_type]
            self.url_input.setText(config["url"])
            self.endpoint_input.setText(config["endpoint"])
            self.params_input.setPlainText(config["params"])

    def test_connection(self):
        """Probar conexión a la API"""
        try:
            url = self.url_input.text().strip()
            endpoint = self.endpoint_input.text().strip()

            if not url:
                QMessageBox.warning(self, "Error", "Por favor ingresa una URL válida")
                return

            full_url = url + endpoint

            # Preparar headers
            headers = {"Content-Type": "application/json"}
            token = self.token_input.text().strip()
            if token:
                headers["Authorization"] = f"Bearer {token}"

            # Preparar parámetros
            params = {}
            params_text = self.params_input.toPlainText().strip()
            if params_text:
                try:
                    params = json.loads(params_text)
                except json.JSONDecodeError:
                    QMessageBox.warning(self, "Error", "Los parámetros JSON no son válidos")
                    return

            # Realizar petición de prueba con timeout
            response = requests.get(full_url, headers=headers, params=params, timeout=10)

            if response.status_code == 200:
                QMessageBox.information(
                    self,
                    "✅ Conexión Exitosa",
                    f"Conexión establecida correctamente\n"
                    f"Estado: {response.status_code}\n"
                    f"Tamaño respuesta: {len(response.content)} bytes"
                )
            else:
                QMessageBox.warning(
                    self,
                    "⚠️ Conexión con Advertencias",
                    f"La API respondió con código: {response.status_code}\n"
                    f"Mensaje: {response.text[:200]}..."
                )

        except requests.exceptions.Timeout:
            QMessageBox.critical(self, "❌ Error", "Timeout: La API no respondió en 10 segundos")
        except requests.exceptions.ConnectionError:
            QMessageBox.critical(self, "❌ Error", "Error de conexión: Verifica la URL y tu conexión a internet")
        except Exception as e:
            QMessageBox.critical(self, "❌ Error", f"Error inesperado: {str(e)}")

    def get_config(self):
        """Obtener configuración de la API"""
        params = {}
        params_text = self.params_input.toPlainText().strip()
        if params_text:
            try:
                params = json.loads(params_text)
            except:
                params = {}

        return {
            "api_type": self.api_type.currentText(),
            "url": self.url_input.text().strip(),
            "endpoint": self.endpoint_input.text().strip(),
            "token": self.token_input.text().strip(),
            "params": params
        }


class APILoadingThread(QThread):
    """Hilo para carga de datos desde API"""

    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    data_loaded = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        """Ejecutar carga desde API"""
        try:
            self.status_update.emit("🌐 Conectando a la API...")
            self.progress_update.emit(10)

            # Construir URL completa
            full_url = self.config["url"] + self.config["endpoint"]

            # Preparar headers
            headers = {"Content-Type": "application/json"}
            if self.config["token"]:
                headers["Authorization"] = f"Bearer {self.config['token']}"

            self.progress_update.emit(20)
            self.status_update.emit("📡 Enviando petición...")

            # Realizar petición
            response = requests.get(
                full_url,
                headers=headers,
                params=self.config["params"],
                timeout=30
            )

            self.progress_update.emit(50)

            if response.status_code != 200:
                raise Exception(f"API respondió con código {response.status_code}: {response.text}")

            self.status_update.emit("📊 Procesando datos de la API...")
            self.progress_update.emit(70)

            # Procesar respuesta JSON
            data = response.json()

            # Convertir a DataFrame según la estructura
            df = self.process_api_data(data)

            self.progress_update.emit(90)
            self.status_update.emit("✅ Datos cargados desde API")

            self.progress_update.emit(100)
            self.data_loaded.emit(df)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def process_api_data(self, data):
        """Procesar datos de la API y convertir a DataFrame"""
        self.status_update.emit("🔄 Convirtiendo datos a formato tabular...")

        # Intentar diferentes estructuras de respuesta comunes
        if isinstance(data, dict):
            # Buscar datos en claves comunes
            for key in ['data', 'results', 'records', 'items', 'datos']:
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break

            # Si aún es un dict, intentar convertir directamente
            if isinstance(data, dict):
                # Si es un dict con listas, convertir
                if any(isinstance(v, list) for v in data.values()):
                    df = pd.DataFrame(data)
                else:
                    # Si es un solo registro, convertir a lista
                    df = pd.DataFrame([data])
            else:
                df = pd.DataFrame(data)
        elif isinstance(data, list):
            if len(data) > 0:
                df = pd.DataFrame(data)
            else:
                raise Exception("La API devolvió una lista vacía")
        else:
            raise Exception("Formato de datos no reconocido en la respuesta de la API")

        # Validar que tenemos un DataFrame válido
        if df.empty:
            raise Exception("No se pudieron extraer datos válidos de la respuesta de la API")

        # Limpiar nombres de columnas
        df.columns = df.columns.astype(str)
        df.columns = [col.replace(' ', '_').replace('-', '_') for col in df.columns]

        return df


class StatusCard(QFrame):
    """Tarjeta de estado simplificada"""

    def __init__(self, title, icon, value="0", unit=""):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel)
        self.setFixedHeight(80)
        self.unit = unit

        layout = QHBoxLayout()
        layout.setContentsMargins(10, 5, 10, 5)  # Añadir márgenes consistentes

        # Icono
        icon_label = QLabel(icon)
        icon_label.setFont(QFont("Arial", 24))
        icon_label.setFixedWidth(50)
        icon_label.setAlignment(Qt.AlignCenter)

        # Contenido
        content_layout = QVBoxLayout()
        content_layout.setSpacing(2)  # Reducir espaciado

        self.title_label = QLabel(title)
        self.title_label.setFont(QFont("Arial", 9))
        self.title_label.setStyleSheet("color: #666;")

        self.value_label = QLabel(value)
        self.value_label.setFont(QFont("Arial", 16, QFont.Bold))

        content_layout.addWidget(self.title_label)
        content_layout.addWidget(self.value_label)

        layout.addWidget(icon_label)
        layout.addLayout(content_layout)
        layout.addStretch()

        self.setLayout(layout)

    def update_value(self, new_value):
        self.value_label.setText(str(new_value))


class CargaDatos(QWidget, ThemedWidget):
    """Pantalla de carga de datos simplificada"""

    # Señales para cuando los datos están listos
    datos_cargados = pyqtSignal(object)  # Emite el DataFrame cuando está listo
    data_loaded_signal = pyqtSignal(object)  # Alias para compatibilidad

    def __init__(self):
        QWidget.__init__(self)
        ThemedWidget.__init__(self)
        self.df = None
        self.loading_thread = None
        self.setup_ui()
        self.apply_styles()

        # Conectar las señales
        self.datos_cargados.connect(self.data_loaded_signal.emit)

    def setup_ui(self):
        """Configurar la interfaz de usuario simplificada"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Header simplificado
        self.create_header(main_layout)

        # Cards de estado
        self.create_status_cards(main_layout)

        # Splitter para área principal
        splitter = QSplitter(Qt.Horizontal)

        # Panel de control (izquierda) - simplificado
        self.create_control_panel(splitter)

        # Panel de vista previa (derecha)
        self.create_preview_panel(splitter)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 900])

        main_layout.addWidget(splitter)

        # Footer con botones principales - CORREGIDO
        self.create_footer(main_layout)

        self.setLayout(main_layout)

    def create_header(self, parent_layout):
        """Crear header simplificado"""
        header_frame = QFrame()
        header_frame.setObjectName("headerFrame")
        header_frame.setFixedHeight(80)  # Aumentado de 60 a 80

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(20, 15, 20, 15)  # Márgenes más generosos

        # Título - CORREGIDO
        title_layout = QVBoxLayout()
        title_layout.setSpacing(5)

        title_label = QLabel("💧 Control de Calidad del Agua")
        title_label.setObjectName("mainTitle")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))  # Aumentado font-size
        title_label.setStyleSheet("color: #2c3e50; margin: 0px; padding: 0px;")  # Estilo directo

        subtitle_label = QLabel("Carga de Datos")
        subtitle_label.setFont(QFont("Arial", 12))  # Aumentado font-size
        subtitle_label.setStyleSheet("color: #666; margin: 0px; padding: 0px;")

        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)

        # Botón tema - CORREGIDO
        self.dark_mode_button = QPushButton("🌙")
        self.dark_mode_button.setObjectName("darkModeButton")
        self.dark_mode_button.setFixedSize(50, 50)  # Aumentado de 45 a 50
        self.dark_mode_button.setStyleSheet("""
            QPushButton {
                background-color: #495057;
                color: white;
                border: none;
                border-radius: 25px;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #343a40;
            }
        """)
        self.dark_mode_button.clicked.connect(self.toggle_theme)

        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        header_layout.addWidget(self.dark_mode_button)

        header_frame.setLayout(header_layout)
        parent_layout.addWidget(header_frame)

    def create_status_cards(self, parent_layout):
        """Crear tarjetas de estado"""
        status_frame = QFrame()
        status_frame.setObjectName("statusFrame")

        status_layout = QHBoxLayout()
        status_layout.setSpacing(15)
        status_layout.setContentsMargins(10, 10, 10, 10)  # Añadir márgenes

        self.file_card = StatusCard("Archivo", "📁", "Ninguno")
        self.rows_card = StatusCard("Registros", "📊", "0")
        self.cols_card = StatusCard("Variables", "📋", "0")
        self.size_card = StatusCard("Memoria", "💾", "0 MB")

        status_layout.addWidget(self.file_card)
        status_layout.addWidget(self.rows_card)
        status_layout.addWidget(self.cols_card)
        status_layout.addWidget(self.size_card)

        status_frame.setLayout(status_layout)
        parent_layout.addWidget(status_frame)

    def create_control_panel(self, parent_widget):
        """Crear panel de control simplificado"""
        control_frame = QFrame()
        control_frame.setObjectName("controlFrame")
        control_frame.setFixedWidth(300)

        control_layout = QVBoxLayout()
        control_layout.setSpacing(20)
        control_layout.setContentsMargins(15, 15, 15, 15)  # Añadir márgenes

        # Título del panel
        panel_title = QLabel("📂 Fuentes de Datos")
        panel_title.setObjectName("panelTitle")
        panel_title.setFont(QFont("Arial", 12, QFont.Bold))
        control_layout.addWidget(panel_title)

        # Información del archivo
        info_group = QGroupBox("ℹ️ Información del Archivo")
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(10, 15, 10, 10)

        self.file_info_label = QLabel("No hay archivo seleccionado")
        self.file_info_label.setObjectName("fileInfoLabel")
        self.file_info_label.setWordWrap(True)
        self.file_info_label.setMinimumHeight(60)
        info_layout.addWidget(self.file_info_label)

        info_group.setLayout(info_layout)
        control_layout.addWidget(info_group)

        # Botones de carga principales - CORREGIDOS
        load_group = QGroupBox("📥 Cargar Datos")
        load_layout = QGridLayout()
        load_layout.setSpacing(10)
        load_layout.setContentsMargins(10, 15, 10, 10)

        # Definir altura mínima para todos los botones
        button_height = 55  # Aumentado de 50 a 55

        self.csv_button = QPushButton("📄 Archivo CSV")
        self.csv_button.setMinimumHeight(button_height)
        self.csv_button.setMinimumWidth(150)  # Aumentado de 120 a 150
        self.csv_button.setObjectName("loadButton")
        self.csv_button.clicked.connect(self.load_csv)

        self.excel_button = QPushButton("📊 Archivo Excel")
        self.excel_button.setMinimumHeight(button_height)
        self.excel_button.setMinimumWidth(150)  # Aumentado de 120 a 150
        self.excel_button.setObjectName("loadButton")
        self.excel_button.clicked.connect(self.load_excel)

        self.sample_button = QPushButton("🎯 Datos de Ejemplo")  # Texto completo
        self.sample_button.setMinimumHeight(button_height)
        self.sample_button.setMinimumWidth(130)  # Aumentado de 100 a 130
        self.sample_button.setObjectName("sampleButton")
        self.sample_button.clicked.connect(self.load_sample_data)

        self.api_button = QPushButton("🌐 Conectar API")  # Texto completo
        self.api_button.setMinimumHeight(button_height)
        self.api_button.setMinimumWidth(130)  # Aumentado de 100 a 130
        self.api_button.setObjectName("apiButton")
        self.api_button.clicked.connect(self.load_from_api)

        load_layout.addWidget(self.csv_button, 0, 0, 1, 2)
        load_layout.addWidget(self.excel_button, 1, 0, 1, 2)
        load_layout.addWidget(self.sample_button, 2, 0)
        load_layout.addWidget(self.api_button, 2, 1)

        # Configurar expansión de columnas
        load_layout.setColumnStretch(0, 1)
        load_layout.setColumnStretch(1, 1)

        load_group.setLayout(load_layout)
        control_layout.addWidget(load_group)

        # Estado del sistema
        system_group = QGroupBox("⚙️ Estado")
        system_layout = QVBoxLayout()
        system_layout.setContentsMargins(10, 15, 10, 10)

        self.status_label = QLabel("✅ Sistema listo para cargar datos")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(40)  # Altura mínima

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("progressBar")
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(25)  # Altura mínima

        system_layout.addWidget(self.status_label)
        system_layout.addWidget(self.progress_bar)

        system_group.setLayout(system_layout)
        control_layout.addWidget(system_group)

        control_layout.addStretch()
        control_frame.setLayout(control_layout)
        parent_widget.addWidget(control_frame)

    def create_preview_panel(self, parent_widget):
        """Crear panel de vista previa"""
        preview_frame = QFrame()
        preview_frame.setObjectName("previewFrame")

        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(15, 15, 15, 15)

        # Header del panel
        header_layout = QHBoxLayout()
        panel_title = QLabel("👁️ Vista Previa de Datos")
        panel_title.setObjectName("panelTitle")
        panel_title.setFont(QFont("Arial", 12, QFont.Bold))

        self.data_info_label = QLabel("Sin datos")
        self.data_info_label.setAlignment(Qt.AlignRight)
        self.data_info_label.setStyleSheet("color: #666; font-size: 10px;")

        header_layout.addWidget(panel_title)
        header_layout.addStretch()
        header_layout.addWidget(self.data_info_label)

        preview_layout.addLayout(header_layout)

        # Tabs simplificados
        self.tab_widget = QTabWidget()

        # Tab 1: Vista de tabla
        self.create_table_tab()

        # Tab 2: Estadísticas básicas
        self.create_stats_tab()

        preview_layout.addWidget(self.tab_widget)
        preview_frame.setLayout(preview_layout)
        parent_widget.addWidget(preview_frame)

    def create_table_tab(self):
        """Crear tab de tabla"""
        self.table_tab = QWidget()
        table_layout = QVBoxLayout()
        table_layout.setContentsMargins(10, 10, 10, 10)

        # Información de la tabla
        self.table_info_label = QLabel("📊 Datos no disponibles")
        self.table_info_label.setStyleSheet("font-size: 10px; color: #666;")
        table_layout.addWidget(self.table_info_label)

        # Tabla optimizada para mostrar solo primeras 100 filas
        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setSortingEnabled(True)
        self.data_table.setSelectionBehavior(QTableWidget.SelectRows)

        table_layout.addWidget(self.data_table)

        self.table_tab.setLayout(table_layout)
        self.tab_widget.addTab(self.table_tab, "📋 Tabla")

    def create_stats_tab(self):
        """Crear tab de estadísticas básicas"""
        self.stats_tab = QWidget()
        stats_layout = QVBoxLayout()
        stats_layout.setContentsMargins(10, 10, 10, 10)

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setPlainText("Carga un archivo para ver las estadísticas...")
        self.stats_text.setFont(QFont("Consolas", 9))
        stats_layout.addWidget(self.stats_text)

        self.stats_tab.setLayout(stats_layout)
        self.tab_widget.addTab(self.stats_tab, "📊 Estadísticas")

    def create_footer(self, parent_layout):
        """Crear footer con botones principales - CORREGIDO"""
        footer_frame = QFrame()
        footer_frame.setObjectName("footerFrame")
        footer_frame.setFixedHeight(80)  # Aumentar altura de 70 a 80

        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(25, 20, 25, 20)  # Márgenes más generosos
        footer_layout.setSpacing(20)  # Más espaciado entre elementos

        # Info
        self.footer_info_label = QLabel("💡 Selecciona un archivo para comenzar")
        self.footer_info_label.setStyleSheet("font-size: 12px; color: #666; font-weight: normal;")
        self.footer_info_label.setWordWrap(True)

        # Botones principales - SIGNIFICATIVAMENTE AUMENTADOS
        button_height = 55  # Aumentado de 50 a 55

        self.export_button = QPushButton("📤 Exportar")
        self.export_button.setObjectName("secondaryButton")
        self.export_button.setMinimumHeight(button_height)
        self.export_button.setMinimumWidth(160)  # Aumentado de 140 a 160
        self.export_button.setMaximumHeight(button_height)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.exportar_datos)

        self.clear_button = QPushButton("🗑️ Limpiar")
        self.clear_button.setObjectName("dangerButton")
        self.clear_button.setMinimumHeight(button_height)
        self.clear_button.setMinimumWidth(150)  # Aumentado de 130 a 150
        self.clear_button.setMaximumHeight(button_height)
        self.clear_button.setEnabled(False)
        self.clear_button.clicked.connect(self.clear_data)

        self.btn_cargar = QPushButton("✅ Usar Estos Datos")
        self.btn_cargar.setObjectName("primaryButton")
        self.btn_cargar.setMinimumHeight(button_height)
        self.btn_cargar.setMinimumWidth(1240)  # Aumentado de 200 a 240
        self.btn_cargar.setMaximumHeight(button_height)
        self.btn_cargar.setEnabled(False)
        self.btn_cargar.clicked.connect(self.usar_datos)

        footer_layout.addWidget(self.footer_info_label)
        footer_layout.addStretch()
        footer_layout.addWidget(self.export_button)
        footer_layout.addWidget(self.clear_button)
        footer_layout.addWidget(self.btn_cargar)

        footer_frame.setLayout(footer_layout)
        parent_layout.addWidget(footer_frame)

    def toggle_theme(self):
        """Alternar tema"""
        try:
            theme_manager = ThemeManager()
            theme_manager.toggle_theme()
            if theme_manager.is_dark_theme():
                self.dark_mode_button.setText("☀️")
            else:
                self.dark_mode_button.setText("🌙")
        except Exception:
            if self.dark_mode_button.text() == "🌙":
                self.dark_mode_button.setText("☀️")
            else:
                self.dark_mode_button.setText("🌙")

    def load_csv(self):
        """Cargar archivo CSV"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo CSV", "", "Archivos CSV (*.csv);;Todos los archivos (*.*)"
        )
        if file_path:
            self.load_file_threaded(file_path, 'csv')

    def load_excel(self):
        """Cargar archivo Excel"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo Excel", "", "Archivos Excel (*.xlsx *.xls);;Todos los archivos (*.*)"
        )
        if file_path:
            self.load_file_threaded(file_path, 'excel')

    def load_file_threaded(self, file_path, file_type):
        """Cargar archivo usando thread"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.set_loading_state(True)

        self.loading_thread = DataLoadingThread(file_path, file_type)
        self.loading_thread.progress_update.connect(self.progress_bar.setValue)
        self.loading_thread.status_update.connect(self.status_label.setText)
        self.loading_thread.data_loaded.connect(self.on_data_loaded)
        self.loading_thread.error_occurred.connect(self.on_loading_error)
        self.loading_thread.finished.connect(self.on_loading_finished)
        self.loading_thread.start()

    def set_loading_state(self, loading):
        """Configurar estado de carga"""
        self.csv_button.setEnabled(not loading)
        self.excel_button.setEnabled(not loading)
        self.sample_button.setEnabled(not loading)
        self.api_button.setEnabled(not loading)

    def on_data_loaded(self, df):
        """Callback cuando los datos se cargan exitosamente"""
        self.df = df
        self.update_display()
        self.update_file_info_from_df()

    def on_loading_error(self, error_msg):
        """Callback cuando ocurre un error"""
        QMessageBox.critical(self, "Error", f"Error al cargar archivo:\n{error_msg}")
        self.status_label.setText("❌ Error al cargar archivo")

    def on_loading_finished(self):
        """Callback cuando termina la carga"""
        self.progress_bar.setVisible(False)
        self.set_loading_state(False)
        self.loading_thread = None

    def load_from_api(self):
        """Cargar datos desde API"""
        dialog = APIConnectionDialog(self)

        if dialog.exec_() == QDialog.Accepted:
            config = dialog.get_config()

            # Validar configuración
            if not config["url"] or not config["endpoint"]:
                QMessageBox.warning(
                    self,
                    "Configuración Incompleta",
                    "Por favor completa al menos la URL y el endpoint"
                )
                return

            # Confirmar carga
            reply = QMessageBox.question(
                self,
                "🌐 Confirmar Carga desde API",
                f"¿Deseas cargar datos desde:\n\n"
                f"🔗 API: {config['api_type']}\n"
                f"📍 URL: {config['url']}{config['endpoint']}\n"
                f"⚙️ Parámetros: {len(config['params'])} configurados\n\n"
                f"Esto puede tomar varios segundos dependiendo de la cantidad de datos.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Yes:
                self.load_api_threaded(config)

    def load_api_threaded(self, config):
        """Cargar datos desde API usando thread"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.set_loading_state(True)

        self.loading_thread = APILoadingThread(config)
        self.loading_thread.progress_update.connect(self.progress_bar.setValue)
        self.loading_thread.status_update.connect(self.status_label.setText)
        self.loading_thread.data_loaded.connect(self.on_api_data_loaded)
        self.loading_thread.error_occurred.connect(self.on_api_loading_error)
        self.loading_thread.finished.connect(self.on_loading_finished)
        self.loading_thread.start()

    def on_api_data_loaded(self, df):
        """Callback cuando los datos de API se cargan exitosamente"""
        self.df = df
        self.update_display()

        # Actualizar información específica para API
        api_info = f"🌐 Datos cargados desde API\n"
        api_info += f"📊 {len(df):,} registros obtenidos\n"
        api_info += f"📋 {len(df.columns)} variables disponibles\n"
        api_info += f"⏰ Cargado: {datetime.now().strftime('%H:%M:%S')}"

        self.file_info_label.setText(api_info)
        self.file_card.update_value("API_data.json")

    def on_api_loading_error(self, error_msg):
        """Callback cuando ocurre un error en la carga de API"""
        QMessageBox.critical(
            self,
            "❌ Error de API",
            f"Error al cargar datos desde la API:\n\n{error_msg}\n\n"
            f"Posibles soluciones:\n"
            f"• Verificar la conexión a internet\n"
            f"• Revisar la URL y endpoint\n"
            f"• Comprobar el token de autenticación\n"
            f"• Contactar al administrador de la API"
        )
        self.status_label.setText("❌ Error al cargar desde API")

    def load_sample_data(self):
        """Cargar datos de ejemplo"""
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(10)
            self.status_label.setText("🎯 Generando datos de ejemplo...")

            np.random.seed(42)
            n_samples = 1000

            # Generar datos de ejemplo realistas para calidad del agua
            data = {
                'Estacion': [f"EST_{str(i + 1).zfill(3)}" for i in np.random.choice(50, n_samples)],
                'Fecha': pd.date_range('2020-01-01', '2024-12-31', periods=n_samples),
                'pH': np.round(np.random.normal(7.2, 0.8, n_samples), 2),
                'Oxigeno_Disuelto': np.round(np.random.normal(8.5, 1.5, n_samples), 2),
                'Turbidez': np.round(np.random.exponential(2.0, n_samples), 2),
                'Conductividad': np.round(np.random.normal(250, 80, n_samples), 1),
                'DBO5': np.round(np.random.exponential(3, n_samples), 2),
                'Temperatura': np.round(np.random.normal(22, 5, n_samples), 1),
                'Coliformes_Totales': np.random.poisson(50, n_samples),
                'Nitratos': np.round(np.random.exponential(2, n_samples), 2),
                'Fosforo_Total': np.round(np.random.exponential(0.1, n_samples), 3)
            }

            self.progress_bar.setValue(50)

            # Calcular índice de calidad simple
            ph_score = 100 * np.exp(-0.5 * ((data['pH'] - 7.0) / 1.5) ** 2)
            do_score = np.minimum(100, (data['Oxigeno_Disuelto'] / 8.0) * 100)
            data['Indice_Calidad'] = np.round((ph_score + do_score) / 2, 1)

            # Clasificación
            data['Clasificacion'] = np.where(data['Indice_Calidad'] >= 80, 'Excelente',
                                             np.where(data['Indice_Calidad'] >= 60, 'Buena',
                                                      np.where(data['Indice_Calidad'] >= 40, 'Regular', 'Mala')))

            self.df = pd.DataFrame(data)
            self.progress_bar.setValue(90)

            self.update_display()
            self.file_info_label.setText(
                f"🎯 Dataset de ejemplo\n"
                f"📊 {n_samples:,} muestras × {len(data)} variables\n"
                f"🧪 Parámetros fisicoquímicos y bacteriológicos"
            )
            self.file_card.update_value("datos_ejemplo.csv")

            self.progress_bar.setValue(100)
            self.progress_bar.setVisible(False)
            self.status_label.setText("✅ Datos de ejemplo cargados")

        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Error al generar datos de ejemplo: {str(e)}")

    def update_display(self):
        """Actualizar visualización de datos"""
        if self.df is not None:
            self.update_status_cards()
            self.update_table()
            self.update_statistics()

            # Habilitar botones
            self.btn_cargar.setEnabled(True)
            self.export_button.setEnabled(True)
            self.clear_button.setEnabled(True)

            # Actualizar info
            self.data_info_label.setText(f"{len(self.df):,} filas × {len(self.df.columns)} columnas")
            self.footer_info_label.setText(f"✅ Datos listos • {len(self.df):,} registros")

    def update_status_cards(self):
        """Actualizar tarjetas de estado"""
        if self.df is not None:
            self.rows_card.update_value(f"{len(self.df):,}")
            self.cols_card.update_value(len(self.df.columns))

            size_mb = self.df.memory_usage(deep=True).sum() / (1024 * 1024)
            self.size_card.update_value(f"{size_mb:.1f} MB")

    def update_table(self):
        """Actualizar tabla (solo primeras 100 filas)"""
        if self.df is not None:
            # Mostrar solo las primeras 100 filas para mejor rendimiento
            display_rows = min(100, len(self.df))
            df_display = self.df.head(display_rows)

            self.table_info_label.setText(
                f"📊 Mostrando {display_rows:,} de {len(self.df):,} filas"
            )

            self.data_table.setRowCount(display_rows)
            self.data_table.setColumnCount(len(df_display.columns))
            self.data_table.setHorizontalHeaderLabels(df_display.columns.tolist())

            self.data_table.setSortingEnabled(False)

            for i in range(display_rows):
                for j, col in enumerate(df_display.columns):
                    value = df_display.iloc[i, j]

                    if pd.isna(value):
                        display_value = "N/A"
                    elif isinstance(value, float):
                        display_value = f"{value:.3f}"
                    elif isinstance(value, pd.Timestamp):
                        display_value = value.strftime("%Y-%m-%d")
                    else:
                        display_value = str(value)

                    item = QTableWidgetItem(display_value)
                    self.data_table.setItem(i, j, item)

            self.data_table.setSortingEnabled(True)
            self.data_table.resizeColumnsToContents()

    def update_statistics(self):
        """Actualizar estadísticas básicas"""
        if self.df is not None:
            stats_text = "=== 📊 RESUMEN DE DATOS ===\n\n"

            # Información general
            stats_text += f"📋 INFORMACIÓN GENERAL\n"
            stats_text += f"{'=' * 40}\n"
            stats_text += f"Registros: {len(self.df):,}\n"
            stats_text += f"Variables: {len(self.df.columns)}\n"
            stats_text += f"Memoria: {self.df.memory_usage(deep=True).sum() / (1024 * 1024):.1f} MB\n\n"

            # Tipos de datos
            stats_text += f"📝 TIPOS DE VARIABLES\n"
            stats_text += f"{'=' * 40}\n"
            for col, dtype in self.df.dtypes.items():
                stats_text += f"  • {col}: {dtype}\n"

            # Valores faltantes
            stats_text += f"\n⚠️ VALORES FALTANTES\n"
            stats_text += f"{'=' * 40}\n"
            null_counts = self.df.isnull().sum()
            total_nulls = null_counts.sum()

            if total_nulls == 0:
                stats_text += "✅ No hay valores faltantes\n"
            else:
                stats_text += f"Total: {total_nulls:,} valores faltantes\n"
                missing_data = null_counts[null_counts > 0]
                for col, count in missing_data.items():
                    percentage = (count / len(self.df)) * 100
                    stats_text += f"  • {col}: {count:,} ({percentage:.1f}%)\n"

            # Estadísticas para columnas numéricas
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                stats_text += f"\n📈 ESTADÍSTICAS NUMÉRICAS\n"
                stats_text += f"{'=' * 40}\n"
                desc_stats = self.df[numeric_cols].describe()
                stats_text += desc_stats.round(3).to_string()

            self.stats_text.setPlainText(stats_text)

    def update_file_info_from_df(self):
        """Actualizar información del archivo desde DataFrame"""
        if self.df is not None:
            size_mb = self.df.memory_usage(deep=True).sum() / (1024 * 1024)
            info_text = f"📊 Archivo cargado exitosamente\n"
            info_text += f"💾 {size_mb:.1f} MB en memoria\n"
            info_text += f"📋 {len(self.df):,} × {len(self.df.columns)} (filas × columnas)"
            self.file_info_label.setText(info_text)

    def usar_datos(self):
        """Emitir señal con los datos cargados y notificar al DataManager."""
        if self.df is not None:
            # Registrar los datos globalmente
            dm = get_data_manager()  # Obtener la instancia del DataManager
            dm.set_data(self.df, source="carga_manual")

            # ¡CRUCIAL! Notificar al DataManager Singleton que los datos han cambiado
            dm._notify_observers('data_changed')  # Esto hará que DataManager llame a update() en SupervisadoWindow

            # Emitir señal con los datos (esto sigue siendo útil si otros widgets escuchan esta señal específica)
            self.datos_cargados.emit(self.df)

            # Mostrar mensaje informativo
            size_mb = self.df.memory_usage(deep=True).sum() / (1024 * 1024)
            QMessageBox.information(
                self,
                "✅ Datos Preparados",
                f"Los datos han sido preparados para análisis:\n\n"
                f"📊 Registros: {len(self.df):,}\n"
                f"📋 Variables: {len(self.df.columns)}\n"
                f"💾 Memoria: {size_mb:.1f} MB\n\n"
                f"Los datos están listos para el análisis."
            )

    def exportar_datos(self):
        """Exportar datos"""
        if self.df is not None:
            try:
                file_path, selected_filter = QFileDialog.getSaveFileName(
                    self,
                    "Exportar datos",
                    f"datos_calidad_agua_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "CSV (*.csv);;Excel (*.xlsx)"
                )

                if file_path:
                    if selected_filter.startswith("Excel"):
                        if not file_path.endswith('.xlsx'):
                            file_path += '.xlsx'
                        self.df.to_excel(file_path, index=False)
                    else:
                        if not file_path.endswith('.csv'):
                            file_path += '.csv'
                        self.df.to_csv(file_path, index=False, encoding='utf-8')

                    QMessageBox.information(
                        self,
                        "✅ Exportación Exitosa",
                        f"Datos exportados:\n\n"
                        f"📁 Archivo: {os.path.basename(file_path)}\n"
                        f"📊 Registros: {len(self.df):,}\n"
                        f"📋 Variables: {len(self.df.columns)}"
                    )
                    print("¿Datos cargados?", has_shared_data())  # Debería imprimir True


            except Exception as e:
                QMessageBox.critical(self, "❌ Error", f"Error al exportar: {str(e)}")

    def clear_data(self):
        """Limpiar todos los datos"""
        if self.df is not None:
            reply = QMessageBox.question(
                self,
                "🗑️ Confirmar Limpieza",
                f"¿Seguro que quieres limpiar los datos?\n\n"
                f"Se perderán {len(self.df):,} registros.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.df = None

                # Resetear cards
                self.file_card.update_value("Ninguno")
                self.rows_card.update_value("0")
                self.cols_card.update_value("0")
                self.size_card.update_value("0 MB")

                # Limpiar tabla
                self.data_table.clear()
                self.data_table.setRowCount(0)
                self.data_table.setColumnCount(0)

                # Limpiar texto
                self.stats_text.setPlainText("Carga un archivo para ver las estadísticas...")

                # Resetear labels
                self.file_info_label.setText("No hay archivo seleccionado")
                self.status_label.setText("✅ Sistema listo para cargar datos")
                self.data_info_label.setText("Sin datos")
                self.table_info_label.setText("📊 Datos no disponibles")
                self.footer_info_label.setText("💡 Selecciona un archivo para comenzar")

                # Deshabilitar botones
                self.btn_cargar.setEnabled(False)
                self.export_button.setEnabled(False)
                self.clear_button.setEnabled(False)

    def get_dataframe(self):
        """Obtener el DataFrame actual"""
        return self.df

    def apply_styles(self):
        """Aplicar estilos CSS - CORREGIDOS"""
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
                background-color: #f8f9fa;
            }

            /* HEADER Y TÍTULOS - CORREGIDOS */
            #headerFrame {
                background-color: #ffffff;
                border-bottom: 2px solid #e9ecef;
                border-radius: 8px;
                min-height: 80px;
            }

            #mainTitle {
                color: #2c3e50;
                font-size: 20px;  /* Aumentado de 18px a 20px */
                font-weight: bold;
                margin: 0px;
                padding: 0px;
                line-height: 1.2;
            }

            #panelTitle {
                color: #495057;
                font-size: 14px;  /* Aumentado de 12px a 14px */
                font-weight: bold;
                padding: 5px;
            }

            #statusFrame {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 10px;
                padding: 10px;
            }

            #controlFrame {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 10px;
                padding: 15px;
            }

            #previewFrame {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 10px;
                padding: 15px;
            }

            #footerFrame {
                background-color: #ffffff;
                border-top: 2px solid #e9ecef;
                border-radius: 8px;
                padding: 10px;
                min-height: 80px;  /* Aumentado de 70px a 80px */
            }

            /* BOTONES DE CARGA - CORREGIDOS CON TAMAÑOS ADECUADOS */
            #loadButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 14px 20px;  /* Aumentado padding */
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;  /* Aumentado font-size */
                text-align: center;
                min-height: 55px;  /* Aumentado de 50 a 55 */
                max-height: 55px;
                min-width: 150px;  /* Aumentado de 120 a 150 */
            }

            #loadButton:hover {
                background-color: #0056b3;
            }

            #loadButton:disabled {
                background-color: #6c757d;
                color: #adb5bd;
            }

            #sampleButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 14px 18px;  /* Aumentado padding */
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;  /* Aumentado font-size */
                text-align: center;
                min-height: 55px;  /* Aumentado de 50 a 55 */
                max-height: 55px;
                min-width: 130px;  /* Aumentado de 100 a 130 */
            }

            #sampleButton:hover {
                background-color: #218838;
            }

            #sampleButton:disabled {
                background-color: #6c757d;
                color: #adb5bd;
            }

            #apiButton {
                background-color: #17a2b8;
                color: white;
                border: none;
                padding: 14px 18px;  /* Aumentado padding */
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;  /* Aumentado font-size */
                text-align: center;
                min-height: 55px;  /* Aumentado de 50 a 55 */
                max-height: 55px;
                min-width: 130px;  /* Aumentado de 100 a 130 */
            }

            #apiButton:hover {
                background-color: #138496;
            }

            #apiButton:disabled {
                background-color: #6c757d;
                color: #adb5bd;
            }

            /* BOTONES DEL FOOTER - SIGNIFICATIVAMENTE AUMENTADOS */
            #primaryButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 16px 28px;  /* Aumentado significativamente */
                border-radius: 6px;
                font-weight: bold;
                font-size: 15px;  /* Aumentado de 14px a 15px */
                text-align: center;
                min-height: 55px;  /* Aumentado de 50px a 55px */
                max-height: 55px;
                min-width: 240px;  /* Aumentado de 200px a 240px */
            }

            #primaryButton:hover {
                background-color: #218838;
            }

            #primaryButton:disabled {
                background-color: #6c757d;
                color: #adb5bd;
            }

            #secondaryButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 16px 24px;  /* Aumentado significativamente */
                border-radius: 6px;
                font-weight: bold;
                font-size: 15px;  /* Aumentado de 14px a 15px */
                text-align: center;
                min-height: 55px;  /* Aumentado de 50px a 55px */
                max-height: 55px;
                min-width: 160px;  /* Aumentado de 140px a 160px */
            }

            #secondaryButton:hover {
                background-color: #5a6268;
            }

            #secondaryButton:disabled {
                background-color: #adb5bd;
                color: #6c757d;
            }

            #dangerButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 16px 24px;  /* Aumentado significativamente */
                border-radius: 6px;
                font-weight: bold;
                font-size: 15px;  /* Aumentado de 14px a 15px */
                text-align: center;
                min-height: 55px;  /* Aumentado de 50px a 55px */
                max-height: 55px;
                min-width: 150px;  /* Aumentado de 130px a 150px */
            }

            #dangerButton:hover {
                background-color: #c82333;
            }

            #dangerButton:disabled {
                background-color: #adb5bd;
                color: #6c757d;
            }

            /* BOTÓN DE TEMA - CORREGIDO */
            #darkModeButton {
                background-color: #495057;
                color: white;
                border: none;
                border-radius: 22px;
                font-size: 18px;
                font-weight: bold;
                min-width: 45px;
                max-width: 45px;
                min-height: 45px;
                max-height: 45px;
                text-align: center;
            }

            #darkModeButton:hover {
                background-color: #343a40;
            }

            /* CARDS DE ESTADO - CORREGIDAS */
            StatusCard {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                margin: 5px;
                padding: 5px;
            }

            /* BARRA DE PROGRESO - CORREGIDA */
            QProgressBar {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background-color: #f8f9fa;
                text-align: center;
                font-weight: bold;
                min-height: 25px;
                max-height: 25px;
            }

            QProgressBar::chunk {
                background-color: #007bff;
                border-radius: 3px;
            }

            /* TABLA - CORREGIDA */
            QTableWidget {
                gridline-color: #dee2e6;
                selection-background-color: #007bff;
                alternate-background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                font-size: 12px;
            }

            QTableWidget::item {
                padding: 6px;
                border: none;
            }

            QTableWidget::item:selected {
                background-color: #007bff;
                color: white;
            }

            QTableWidget QHeaderView::section {
                background-color: #e9ecef;
                padding: 8px;
                border: 1px solid #dee2e6;
                font-weight: bold;
                font-size: 12px;
            }

            /* TEXTO Y AREAS DE TEXTO - CORREGIDAS */
            QTextEdit {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background-color: #ffffff;
                padding: 10px;
                font-size: 12px;
            }

            /* GRUPOS - CORREGIDOS */
            QGroupBox {
                font-weight: bold;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: #f8f9fa;
                font-size: 12px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                color: #495057;
                background-color: #f8f9fa;
                font-size: 12px;
                font-weight: bold;
            }

            /* TABS - CORREGIDOS */
            QTabWidget::pane {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background-color: #ffffff;
            }

            QTabBar::tab {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                padding: 10px 15px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-size: 12px;
                font-weight: bold;
                min-width: 80px;
            }

            QTabBar::tab:selected {
                background-color: #ffffff;
                border-bottom: 1px solid #ffffff;
                color: #007bff;
            }

            QTabBar::tab:hover {
                background-color: #e9ecef;
            }

            /* LABELS DE INFORMACIÓN - CORREGIDOS */
            #fileInfoLabel {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 10px;
                color: #495057;
                font-size: 12px;
                min-height: 60px;
            }

            #statusLabel {
                color: #28a745;
                font-weight: bold;
                padding: 5px;
                font-size: 12px;
                min-height: 40px;
            }

            /* INPUTS Y FORMULARIOS - CORREGIDOS */
            QLineEdit {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
                font-size: 12px;
                background-color: #ffffff;
            }

            QLineEdit:focus {
                border: 2px solid #007bff;
            }

            QComboBox {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
                font-size: 12px;
                background-color: #ffffff;
                min-width: 150px;
            }

            QComboBox:focus {
                border: 2px solid #007bff;
            }

            QComboBox::drop-down {
                border: none;
                width: 20px;
            }

            QComboBox::down-arrow {
                image: none;
                border: none;
                width: 0;
                height: 0;
            }

            /* MENSAJES Y DIÁLOGOS - CORREGIDOS */
            QMessageBox {
                background-color: #ffffff;
                font-size: 12px;
            }

            QMessageBox QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                min-width: 80px;
                min-height: 30px;
            }

            QMessageBox QPushButton:hover {
                background-color: #0056b3;
            }

            /* SCROLLBARS - CORREGIDOS */
            QScrollBar:vertical {
                background-color: #f8f9fa;
                width: 12px;
                border-radius: 6px;
            }

            QScrollBar::handle:vertical {
                background-color: #6c757d;
                border-radius: 6px;
                min-height: 20px;
            }

            QScrollBar::handle:vertical:hover {
                background-color: #495057;
            }

            QScrollBar:horizontal {
                background-color: #f8f9fa;
                height: 12px;
                border-radius: 6px;
            }

            QScrollBar::handle:horizontal {
                background-color: #6c757d;
                border-radius: 6px;
                min-width: 20px;
            }

            QScrollBar::handle:horizontal:hover {
                background-color: #495057;
            }

            /* SPLITTER - CORREGIDO */
            QSplitter::handle {
                background-color: #dee2e6;
                width: 2px;
                height: 2px;
            }

            QSplitter::handle:hover {
                background-color: #007bff;
            }

            /* TOOLTIPS - CORREGIDOS */
            QToolTip {
                background-color: #343a40;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 4px;
                font-size: 11px;
            }
        """)

    def closeEvent(self, event):
        """Manejar evento de cierre"""
        if self.loading_thread and self.loading_thread.isRunning():
            self.loading_thread.terminate()
            self.loading_thread.wait()
        event.accept()


# Para ejecutar standalone
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CargaDatos()
    window.setWindowTitle("💧 Control de Calidad del Agua - Carga de Datos")
    window.resize(1200, 700)
    window.show()


    def on_data_loaded(df):
        print(f"Datos cargados: {df.shape}")
        print(f"Columnas: {list(df.columns)}")


    window.datos_cargados.connect(on_data_loaded)
    sys.exit(app.exec_())