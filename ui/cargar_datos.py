# cargar_datos.py
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QTextEdit, QFileDialog, QLabel, QTableWidget,
                             QTableWidgetItem, QTabWidget, QGroupBox, QSplitter,
                             QMessageBox, QProgressBar, QFrame, QScrollArea,
                             QHeaderView, QApplication, QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor
import sys
import traceback


class DataLoader(QThread):
    """Worker thread para cargar datos de manera asíncrona"""
    data_loaded = pyqtSignal(object)  # DataFrame
    error_occurred = pyqtSignal(str)  # Error message
    progress_updated = pyqtSignal(int)  # Progress percentage

    def __init__(self, file_path, file_type):
        super().__init__()
        self.file_path = file_path
        self.file_type = file_type

    def run(self):
        try:
            self.progress_updated.emit(20)

            if self.file_type == 'csv':
                df = self.leer_archivo_csv(self.file_path)
            elif self.file_type == 'excel':
                df = self.leer_archivo_excel(self.file_path)
            else:
                raise ValueError(f"Tipo de archivo no soportado: {self.file_type}")

            self.progress_updated.emit(80)

            if df is not None:
                self.data_loaded.emit(df)
            else:
                self.error_occurred.emit("No se pudo cargar el archivo")

            self.progress_updated.emit(100)

        except Exception as e:
            self.error_occurred.emit(f"Error al cargar archivo: {str(e)}")

    def leer_archivo_csv(self, ruta):
        """Leer archivo CSV con múltiples opciones de encoding"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        separators = [',', ';', '\t', '|']

        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(ruta, encoding=encoding, sep=sep)
                    if not df.empty and len(df.columns) > 1:
                        return df
                except:
                    continue

        # Último intento con pandas auto-detection
        try:
            df = pd.read_csv(ruta)
            return df
        except Exception as e:
            print(f"Error al leer CSV: {e}")
            return None

    def leer_archivo_excel(self, ruta):
        """Leer archivo Excel con manejo de múltiples hojas"""
        try:
            # Intentar leer todas las hojas
            excel_file = pd.ExcelFile(ruta)

            if len(excel_file.sheet_names) == 1:
                # Solo una hoja
                df = pd.read_excel(ruta)
            else:
                # Múltiples hojas - tomar la primera que tenga datos
                for sheet in excel_file.sheet_names:
                    try:
                        df = pd.read_excel(ruta, sheet_name=sheet)
                        if not df.empty:
                            break
                    except:
                        continue
                else:
                    df = pd.read_excel(ruta, sheet_name=0)  # Fallback a primera hoja

            return df

        except Exception as e:
            print(f"Error al leer Excel: {e}")
            return None


class DataPreviewWidget(QWidget):
    """Widget optimizado para mostrar vista previa de datos"""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Tab widget para diferentes vistas
        self.tabs = QTabWidget()

        # Tab 1: Vista de tabla
        self.table_tab = QWidget()
        table_layout = QVBoxLayout()

        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.data_table.setSortingEnabled(True)
        table_layout.addWidget(self.data_table)

        self.table_tab.setLayout(table_layout)
        self.tabs.addTab(self.table_tab, "📊 Vista de Tabla")

        # Tab 2: Información estadística
        self.stats_tab = QWidget()
        stats_layout = QVBoxLayout()

        self.stats_scroll = QScrollArea()
        self.stats_content = QLabel()
        self.stats_content.setAlignment(Qt.AlignTop)
        self.stats_content.setWordWrap(True)
        self.stats_content.setStyleSheet("""
            QLabel {
                background-color: white;
                padding: 15px;
                border-radius: 8px;
                font-family: 'Courier New', monospace;
            }
        """)
        self.stats_scroll.setWidget(self.stats_content)
        self.stats_scroll.setWidgetResizable(True)

        stats_layout.addWidget(self.stats_scroll)
        self.stats_tab.setLayout(stats_layout)
        self.tabs.addTab(self.stats_tab, "📈 Estadísticas")

        # Tab 3: Vista raw (texto)
        self.raw_tab = QWidget()
        raw_layout = QVBoxLayout()

        self.raw_text = QTextEdit()
        self.raw_text.setReadOnly(True)
        self.raw_text.setFont(QFont("Courier New", 9))
        raw_layout.addWidget(self.raw_text)

        self.raw_tab.setLayout(raw_layout)
        self.tabs.addTab(self.raw_tab, "📄 Vista Raw")

        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def mostrar_datos(self, df):
        """Mostrar datos en todas las vistas"""
        try:
            # Vista de tabla
            self.llenar_tabla(df)

            # Vista estadística
            self.mostrar_estadisticas(df)

            # Vista raw
            self.mostrar_raw(df)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al mostrar datos: {str(e)}")

    def llenar_tabla(self, df):
        """Llenar la tabla con los datos del DataFrame"""
        try:
            # Limitar a las primeras 1000 filas para rendimiento
            df_display = df.head(1000) if len(df) > 1000 else df

            self.data_table.setRowCount(len(df_display))
            self.data_table.setColumnCount(len(df_display.columns))

            # Headers
            self.data_table.setHorizontalHeaderLabels([str(col) for col in df_display.columns])

            # Datos
            for i, row in enumerate(df_display.itertuples(index=False)):
                for j, value in enumerate(row):
                    # Convertir a string y manejar valores NaN
                    if pd.isna(value):
                        display_value = "NaN"
                    else:
                        display_value = str(value)

                    item = QTableWidgetItem(display_value)
                    self.data_table.setItem(i, j, item)

            # Ajustar columnas
            self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
            self.data_table.resizeColumnsToContents()

            # Limitar ancho máximo de columnas
            for i in range(self.data_table.columnCount()):
                if self.data_table.columnWidth(i) > 200:
                    self.data_table.setColumnWidth(i, 200)

        except Exception as e:
            print(f"Error al llenar tabla: {e}")

    def mostrar_estadisticas(self, df):
        """Mostrar estadísticas descriptivas del DataFrame"""
        try:
            stats_html = f"""
            <div style="font-family: Arial; font-size: 12px; line-height: 1.4;">
                <h2 style="color: #2c3e50; text-align: center;">📊 Análisis de Datos</h2>

                <div style="background: #e8f4fd; padding: 15px; border-radius: 8px; margin: 10px 0;">
                    <h3 style="color: #2980b9;">🔍 Información General</h3>
                    <p><strong>Filas:</strong> {len(df):,}</p>
                    <p><strong>Columnas:</strong> {len(df.columns)}</p>
                    <p><strong>Tamaño en memoria:</strong> {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB</p>
                    <p><strong>Valores faltantes:</strong> {df.isnull().sum().sum():,}</p>
                </div>

                <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; margin: 10px 0;">
                    <h3 style="color: #27ae60;">📋 Tipos de Datos</h3>
            """

            # Información de tipos de datos
            for dtype, count in df.dtypes.value_counts().items():
                stats_html += f"<p><strong>{dtype}:</strong> {count} columna(s)</p>"

            stats_html += "</div>"

            # Estadísticas numéricas
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                stats_html += """
                <div style="background: #fff3e0; padding: 15px; border-radius: 8px; margin: 10px 0;">
                    <h3 style="color: #ef6c00;">📈 Estadísticas Numéricas</h3>
                """

                describe_stats = numeric_df.describe()
                for col in describe_stats.columns[:5]:  # Limitar a 5 columnas
                    stats_html += f"""
                    <h4 style="color: #d84315;">{col}</h4>
                    <p>Media: {describe_stats[col]['mean']:.3f} | 
                       Mediana: {numeric_df[col].median():.3f} | 
                       Std: {describe_stats[col]['std']:.3f}</p>
                    """

                stats_html += "</div>"

            # Información de valores faltantes por columna
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                stats_html += """
                <div style="background: #ffebee; padding: 15px; border-radius: 8px; margin: 10px 0;">
                    <h3 style="color: #c62828;">⚠️ Valores Faltantes</h3>
                """

                for col in missing_data[missing_data > 0].head(10).index:
                    missing_count = missing_data[col]
                    missing_pct = (missing_count / len(df)) * 100
                    stats_html += f"<p><strong>{col}:</strong> {missing_count:,} ({missing_pct:.1f}%)</p>"

                stats_html += "</div>"

            # Muestra de datos únicos para columnas categóricas
            categorical_df = df.select_dtypes(include=['object'])
            if not categorical_df.empty:
                stats_html += """
                <div style="background: #f3e5f5; padding: 15px; border-radius: 8px; margin: 10px 0;">
                    <h3 style="color: #7b1fa2;">🏷️ Datos Categóricos</h3>
                """

                for col in categorical_df.columns[:5]:  # Limitar a 5 columnas
                    unique_count = df[col].nunique()
                    top_values = df[col].value_counts().head(3)

                    stats_html += f"""
                    <h4 style="color: #4a148c;">{col}</h4>
                    <p>Valores únicos: {unique_count}</p>
                    <p>Más frecuentes: {', '.join([f"{val} ({count})" for val, count in top_values.items()])}</p>
                    """

                stats_html += "</div>"

            stats_html += "</div>"

            self.stats_content.setText(stats_html)

        except Exception as e:
            self.stats_content.setText(f"Error al generar estadísticas: {str(e)}")

    def mostrar_raw(self, df):
        """Mostrar vista raw de los datos"""
        try:
            # Mostrar información básica y muestra de datos
            raw_text = f"Dataset Information:\n"
            raw_text += f"Shape: {df.shape}\n"
            raw_text += f"Columns: {list(df.columns)}\n\n"

            # Tipos de datos
            raw_text += "Data Types:\n"
            raw_text += str(df.dtypes) + "\n\n"

            # Primeras filas
            raw_text += "First 10 rows:\n"
            raw_text += str(df.head(10)) + "\n\n"

            # Últimas filas
            raw_text += "Last 5 rows:\n"
            raw_text += str(df.tail(5)) + "\n\n"

            # Información adicional
            raw_text += "Missing values per column:\n"
            raw_text += str(df.isnull().sum()) + "\n"

            self.raw_text.setText(raw_text)

        except Exception as e:
            self.raw_text.setText(f"Error al mostrar vista raw: {str(e)}")


class CargaDatos(QWidget):
    """Widget principal para carga de datos con diseño mejorado"""

    data_loaded_signal = pyqtSignal(object)  # Señal para cuando se cargan datos exitosamente

    def __init__(self):
        super().__init__()
        self.setWindowTitle("💾 Sistema de Carga de Datos")
        self.setMinimumSize(1000, 700)
        self.df = None
        self.loader_thread = None
        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Título principal
        title = QLabel("💾 Sistema de Carga de Datos")
        title.setObjectName("mainTitle")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        subtitle = QLabel("Carga y visualiza archivos CSV, Excel y más")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle)

        # Panel de controles
        control_group = QGroupBox("🎛️ Opciones de Carga")
        control_layout = QVBoxLayout()

        # Información del sistema
        info_frame = QFrame()
        info_frame.setObjectName("infoFrame")
        info_layout = QHBoxLayout()

        self.file_info_label = QLabel("📁 No hay archivo seleccionado")
        self.file_info_label.setObjectName("fileInfoLabel")
        info_layout.addWidget(self.file_info_label)

        info_layout.addStretch()
        info_frame.setLayout(info_layout)
        control_layout.addWidget(info_frame)

        # Botones de carga
        buttons_frame = QFrame()
        buttons_frame.setObjectName("buttonsFrame")
        buttons_layout = QGridLayout()

        self.btn_csv = QPushButton("📄 Archivo CSV")
        self.btn_csv.setObjectName("loadButton")
        self.btn_csv.setMinimumHeight(60)
        self.btn_csv.clicked.connect(self.cargar_csv)

        self.btn_excel = QPushButton("📊 Archivo Excel")
        self.btn_excel.setObjectName("loadButton")
        self.btn_excel.setMinimumHeight(60)
        self.btn_excel.clicked.connect(self.cargar_excel)

        self.btn_api = QPushButton("🌐 Desde API")
        self.btn_api.setObjectName("apiButton")
        self.btn_api.setMinimumHeight(60)
        self.btn_api.setEnabled(False)  # Por implementar

        self.btn_sample = QPushButton("🔬 Datos de Ejemplo")
        self.btn_sample.setObjectName("sampleButton")
        self.btn_sample.setMinimumHeight(60)
        self.btn_sample.clicked.connect(self.cargar_datos_ejemplo)

        buttons_layout.addWidget(self.btn_csv, 0, 0)
        buttons_layout.addWidget(self.btn_excel, 0, 1)
        buttons_layout.addWidget(self.btn_api, 1, 0)
        buttons_layout.addWidget(self.btn_sample, 1, 1)

        buttons_frame.setLayout(buttons_layout)
        control_layout.addWidget(buttons_frame)

        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("progressBar")
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)

        # Información de estado
        self.status_label = QLabel("✅ Sistema listo para cargar datos")
        self.status_label.setObjectName("statusLabel")
        control_layout.addWidget(self.status_label)

        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        # Widget de vista previa
        preview_group = QGroupBox("👁️ Vista Previa de Datos")
        preview_layout = QVBoxLayout()

        self.data_preview = DataPreviewWidget()
        preview_layout.addWidget(self.data_preview)

        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group)

        # Botones de acción
        action_layout = QHBoxLayout()

        self.btn_cargar = QPushButton("✅ Usar Estos Datos")
        self.btn_cargar.setObjectName("useDataButton")
        self.btn_cargar.setMinimumHeight(45)
        self.btn_cargar.setEnabled(False)
        self.btn_cargar.clicked.connect(self.confirmar_carga)

        self.btn_limpiar = QPushButton("🗑️ Limpiar")
        self.btn_limpiar.setObjectName("clearButton")
        self.btn_limpiar.setMinimumHeight(45)
        self.btn_limpiar.clicked.connect(self.limpiar_datos)

        self.btn_exportar = QPushButton("💾 Exportar Muestra")
        self.btn_exportar.setObjectName("exportButton")
        self.btn_exportar.setMinimumHeight(45)
        self.btn_exportar.setEnabled(False)
        self.btn_exportar.clicked.connect(self.exportar_muestra)

        action_layout.addWidget(self.btn_cargar)
        action_layout.addWidget(self.btn_limpiar)
        action_layout.addWidget(self.btn_exportar)
        action_layout.addStretch()

        main_layout.addLayout(action_layout)
        self.setLayout(main_layout)

    def apply_styles(self):
        """Aplicar estilos CSS modernos"""
        self.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }

            #mainTitle {
                font-size: 28px;
                font-weight: bold;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:0.5 #764ba2, stop:1 #f093fb);
                color: white;
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 10px;
            }

            #subtitle {
                font-size: 14px;
                color: #6c757d;
                font-style: italic;
                margin-bottom: 20px;
                padding: 8px;
            }

            QGroupBox {
                font-weight: bold;
                border: 2px solid #dee2e6;
                border-radius: 10px;
                margin-top: 20px;
                padding-top: 15px;
                background-color: white;
                font-size: 14px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 20px;
                padding: 0 15px 0 15px;
                color: #495057;
                font-size: 16px;
                font-weight: bold;
            }

            #infoFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e3f2fd, stop:1 #bbdefb);
                border: 2px solid #2196f3;
                border-radius: 8px;
                padding: 12px;
                margin: 8px;
            }

            #fileInfoLabel {
                font-weight: bold;
                color: #1976d2;
                font-size: 13px;
            }

            #buttonsFrame {
                background-color: #ffffff;
                border-radius: 10px;
                padding: 15px;
                margin: 8px;
            }

            #loadButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4CAF50, stop:1 #388E3C);
                color: white;
                border: none;
                padding: 15px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
                margin: 5px;
            }

            #loadButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #66BB6A, stop:1 #4CAF50);
            }

            #loadButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #388E3C, stop:1 #2E7D32);
            }

            #apiButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FF9800, stop:1 #F57C00);
                color: white;
                border: none;
                padding: 15px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
                margin: 5px;
            }

            #apiButton:disabled {
                background-color: #e0e0e0;
                color: #9e9e9e;
            }

            #sampleButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #9C27B0, stop:1 #7B1FA2);
                color: white;
                border: none;
                padding: 15px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
                margin: 5px;
            }

            #sampleButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #BA68C8, stop:1 #9C27B0);
            }

            #progressBar {
                border: 2px solid #dee2e6;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                font-size: 12px;
                height: 25px;
                margin: 10px 5px;
            }

            #progressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4CAF50, stop:1 #8BC34A);
                border-radius: 6px;
            }

            #statusLabel {
                background-color: #e8f5e8;
                color: #2e7d32;
                padding: 12px 16px;
                border-radius: 8px;
                border-left: 5px solid #4CAF50;
                font-weight: bold;
                font-size: 13px;
                margin: 5px;
            }

            #useDataButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2196F3, stop:1 #1976D2);
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
                margin: 5px;
            }

            #useDataButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #42A5F5, stop:1 #2196F3);
            }

            #useDataButton:disabled {
                background-color: #e0e0e0;
                color: #9e9e9e;
            }

            #clearButton, #exportButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #757575, stop:1 #616161);
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
                margin: 5px;
            }

            #clearButton:hover, #exportButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #9E9E9E, stop:1 #757575);
            }

            #exportButton:disabled {
                background-color: #e0e0e0;
                color: #9e9e9e;
            }

            QTabWidget::pane {
                border: 2px solid #dee2e6;
                background-color: white;
                border-radius: 8px;
            }

            QTabBar::tab {
                background-color: #f8f9fa;
                color: #495057;
                padding: 12px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: bold;
                font-size: 12px;
            }

            QTabBar::tab:selected {
                background-color: #2196F3;
                color: white;
            }

            QTabBar::tab:hover:!selected {
                background-color: #e9ecef;
            }

            QTableWidget {
                gridline-color: #dee2e6;
                background-color: white;
                alternate-background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 6px;
            }

            QHeaderView::section {
                background-color: #495057;
                color: white;
                padding: 8px;
                border: none;
                font-weight: bold;
            }

            QTextEdit, QScrollArea {
                border: 2px solid #dee2e6;
                border-radius: 8px;
                background-color: white;
                padding: 8px;
            }
        """)

    def cargar_csv(self):
        """Cargar archivo CSV"""
        ruta, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar archivo CSV",
            "",
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)"
        )
        if ruta:
            self.iniciar_carga(ruta, 'csv')

    def cargar_excel(self):
        """Cargar archivo Excel"""
        ruta, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar archivo Excel",
            "",
            "Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        if ruta:
            self.iniciar_carga(ruta, 'excel')

    def cargar_datos_ejemplo(self):
        """Cargar datos de ejemplo para demostración"""
        try:
            # Generar datos de ejemplo
            np.random.seed(42)
            n_samples = 200

            data = {
                'ID': range(1, n_samples + 1),
                'Nombre': [f'Usuario_{i}' for i in range(1, n_samples + 1)],
                'Edad': np.random.randint(18, 65, n_samples),
                'Salario': np.random.normal(50000, 15000, n_samples).round(2),
                'Departamento': np.random.choice(['IT', 'Marketing', 'Ventas', 'RRHH', 'Finanzas'], n_samples),
                'Experiencia': np.random.randint(0, 20, n_samples),
                'Puntuacion': np.random.uniform(1, 10, n_samples).round(1),
                'Activo': np.random.choice([True, False], n_samples),
                'Fecha_Ingreso': pd.date_range(start='2020-01-01', end='2024-01-01', periods=n_samples)
            }

            # Introducir algunos valores faltantes para realismo
            df_ejemplo = pd.DataFrame(data)
            df_ejemplo.loc[np.random.choice(df_ejemplo.index, 20, replace=False), 'Salario'] = np.nan
            df_ejemplo.loc[np.random.choice(df_ejemplo.index, 15, replace=False), 'Experiencia'] = np.nan

            self.df = df_ejemplo
            self.file_info_label.setText("📊 Datos de ejemplo cargados (200 registros)")
            self.status_label.setText("✅ Datos de ejemplo listos para análisis")

            # Mostrar en vista previa
            self.data_preview.mostrar_datos(self.df)

            # Habilitar botones
            self.btn_cargar.setEnabled(True)
            self.btn_exportar.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al generar datos de ejemplo: {str(e)}")

    def iniciar_carga(self, ruta_archivo, tipo_archivo):
        """Iniciar carga de archivo en thread separado"""
        try:
            # Mostrar información del archivo
            import os
            filename = os.path.basename(ruta_archivo)
            filesize = os.path.getsize(ruta_archivo) / (1024 * 1024)  # MB

            self.file_info_label.setText(f"📁 {filename} ({filesize:.2f} MB)")
            self.status_label.setText("⏳ Cargando archivo...")

            # Deshabilitar botones
            self.btn_csv.setEnabled(False)
            self.btn_excel.setEnabled(False)
            self.btn_sample.setEnabled(False)

            # Mostrar barra de progreso
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # Crear y ejecutar thread de carga
            self.loader_thread = DataLoader(ruta_archivo, tipo_archivo)
            self.loader_thread.data_loaded.connect(self.on_data_loaded)
            self.loader_thread.error_occurred.connect(self.on_error_occurred)
            self.loader_thread.progress_updated.connect(self.progress_bar.setValue)
            self.loader_thread.finished.connect(self.on_loading_finished)
            self.loader_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al iniciar carga: {str(e)}")
            self.on_loading_finished()

    def on_data_loaded(self, df):
        """Callback cuando los datos se cargan exitosamente"""
        try:
            self.df = df

            # Actualizar información
            rows, cols = df.shape
            self.status_label.setText(f"✅ Datos cargados: {rows:,} filas, {cols} columnas")

            # Mostrar vista previa
            self.data_preview.mostrar_datos(self.df)

            # Habilitar botones
            self.btn_cargar.setEnabled(True)
            self.btn_exportar.setEnabled(True)

            QMessageBox.information(
                self,
                "Carga Exitosa",
                f"✅ Archivo cargado correctamente!\n\n"
                f"📊 {rows:,} filas y {cols} columnas\n"
                f"💾 {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB en memoria"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al procesar datos: {str(e)}")

    def on_error_occurred(self, error_message):
        """Callback cuando ocurre un error durante la carga"""
        QMessageBox.critical(self, "Error de Carga", f"❌ {error_message}")
        self.status_label.setText(f"❌ Error: {error_message}")

    def on_loading_finished(self):
        """Callback cuando termina el proceso de carga"""
        # Restaurar botones
        self.btn_csv.setEnabled(True)
        self.btn_excel.setEnabled(True)
        self.btn_sample.setEnabled(True)

        # Ocultar barra de progreso
        self.progress_bar.setVisible(False)

    def confirmar_carga(self):
        """Confirmar el uso de los datos cargados"""
        if self.df is not None:
            reply = QMessageBox.question(
                self,
                "Confirmar Carga",
                f"¿Usar estos datos para el análisis?\n\n"
                f"📊 {len(self.df):,} filas\n"
                f"📋 {len(self.df.columns)} columnas\n"
                f"💾 {self.df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                # Emitir señal con los datos
                self.data_loaded_signal.emit(self.df)

                QMessageBox.information(
                    self,
                    "Datos Confirmados",
                    "✅ Los datos han sido cargados exitosamente al sistema.\n\n"
                    "Ahora puedes proceder con el análisis."
                )

    def limpiar_datos(self):
        """Limpiar todos los datos y resetear la interfaz"""
        reply = QMessageBox.question(
            self,
            "Limpiar Datos",
            "¿Estás seguro de que quieres limpiar todos los datos?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.df = None
            self.file_info_label.setText("📁 No hay archivo seleccionado")
            self.status_label.setText("✅ Sistema listo para cargar datos")

            # Deshabilitar botones
            self.btn_cargar.setEnabled(False)
            self.btn_exportar.setEnabled(False)

            # Limpiar vista previa
            self.data_preview.data_table.clear()
            self.data_preview.data_table.setRowCount(0)
            self.data_preview.data_table.setColumnCount(0)
            self.data_preview.stats_content.setText("")
            self.data_preview.raw_text.setText("")

    def exportar_muestra(self):
        """Exportar una muestra de los datos"""
        if self.df is not None:
            try:
                # Preguntar cuántas filas exportar
                from PyQt5.QtWidgets import QInputDialog

                filas, ok = QInputDialog.getInt(
                    self,
                    "Exportar Muestra",
                    "¿Cuántas filas exportar?",
                    min(100, len(self.df)),  # Default
                    1,  # Minimum
                    len(self.df)  # Maximum
                )

                if ok:
                    # Seleccionar archivo de destino
                    ruta, _ = QFileDialog.getSaveFileName(
                        self,
                        "Guardar muestra",
                        f"muestra_{filas}_filas.csv",
                        "CSV Files (*.csv);;Excel Files (*.xlsx)"
                    )

                    if ruta:
                        df_muestra = self.df.head(filas)

                        if ruta.endswith('.xlsx'):
                            df_muestra.to_excel(ruta, index=False)
                        else:
                            df_muestra.to_csv(ruta, index=False)

                        QMessageBox.information(
                            self,
                            "Exportación Exitosa",
                            f"✅ Muestra exportada correctamente!\n\n"
                            f"📄 {filas} filas guardadas en:\n{ruta}"
                        )

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al exportar: {str(e)}")

    def get_loaded_data(self):
        """Obtener los datos cargados"""
        return self.df


# Funciones auxiliares para compatibilidad
def leer_archivo_csv(ruta):
    """Función auxiliar para mantener compatibilidad"""
    loader = DataLoader(ruta, 'csv')
    # Esta función ahora es más simple ya que la lógica está en DataLoader
    try:
        return pd.read_csv(ruta)
    except Exception as e:
        print(f"Error al leer CSV: {e}")
        return None


def leer_archivo_excel(ruta):
    """Función auxiliar para mantener compatibilidad"""
    try:
        return pd.read_excel(ruta)
    except Exception as e:
        print(f"Error al leer Excel: {e}")
        return None


# Ejemplo de uso y prueba del sistema
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Crear ventana principal
    window = CargaDatos()


    # Conectar señal para cuando se cargan datos
    def on_data_loaded(df):
        print(f"¡Datos cargados! Shape: {df.shape}")
        print(f"Columnas: {list(df.columns)}")


    window.data_loaded_signal.connect(on_data_loaded)

    window.show()

    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        window.close()