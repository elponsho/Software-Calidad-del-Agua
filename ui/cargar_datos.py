import sys
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QHBoxLayout, QFrame,
                             QLabel, QPushButton, QFileDialog, QMessageBox,
                             QTabWidget, QWidget, QTableWidget, QTableWidgetItem,
                             QTextEdit, QProgressBar, QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon

# Importar sistema de temas
from darkmode import ThemedWidget, ThemeManager


class StatusCard(QFrame):
    """Tarjeta de estado personalizada"""

    def __init__(self, title, icon, value="0"):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel)
        self.setFixedHeight(80)

        layout = QHBoxLayout()

        # Icono (emoji como placeholder)
        icon_label = QLabel(icon)
        icon_label.setFont(QFont("Arial", 24))
        icon_label.setFixedWidth(50)
        icon_label.setAlignment(Qt.AlignCenter)

        # Contenido
        content_layout = QVBoxLayout()

        self.title_label = QLabel(title)
        self.title_label.setFont(QFont("Arial", 10))

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
    """Pantalla de carga de datos que hereda de ThemedWidget"""

    def __init__(self):
        QWidget.__init__(self)
        ThemedWidget.__init__(self)
        self.df = None
        self.setup_ui()

    def setup_ui(self):
        """Configurar la interfaz de usuario"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Header con título y botón de tema
        self.create_header(main_layout)

        # Cards de estado
        self.create_status_cards(main_layout)

        # Área principal con controles y vista previa
        self.create_main_area(main_layout)

        # Footer con botones de acción
        self.create_footer(main_layout)

        self.setLayout(main_layout)

    def create_header(self, parent_layout):
        """Crear el header con título y botón de tema"""
        header_frame = QFrame()
        header_frame.setObjectName("headerFrame")
        header_frame.setFixedHeight(80)

        header_layout = QHBoxLayout()

        # Título
        title_label = QLabel("Carga de Datos")
        title_label.setObjectName("mainTitle")

        # Botón para cambiar tema
        self.dark_mode_button = QPushButton("🌙")
        self.dark_mode_button.setObjectName("darkModeButton")
        self.dark_mode_button.setFixedSize(50, 50)
        self.dark_mode_button.clicked.connect(self.toggle_theme)

        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.dark_mode_button)

        header_frame.setLayout(header_layout)
        parent_layout.addWidget(header_frame)

    def create_status_cards(self, parent_layout):
        """Crear las tarjetas de estado"""
        status_frame = QFrame()
        status_frame.setObjectName("statusFrame")

        status_layout = QHBoxLayout()
        status_layout.setSpacing(15)

        # Cards de estado
        self.file_card = StatusCard("Archivo", "📁", "Ninguno seleccionado")
        self.rows_card = StatusCard("Filas", "📊", "0")
        self.cols_card = StatusCard("Columnas", "📋", "0")
        self.size_card = StatusCard("Tamaño", "💾", "0 MB")

        status_layout.addWidget(self.file_card)
        status_layout.addWidget(self.rows_card)
        status_layout.addWidget(self.cols_card)
        status_layout.addWidget(self.size_card)

        status_frame.setLayout(status_layout)
        parent_layout.addWidget(status_frame)

    def create_main_area(self, parent_layout):
        """Crear el área principal con controles y vista previa"""
        main_frame = QFrame()
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)

        # Panel de control (izquierda)
        self.create_control_panel(main_layout)

        # Panel de vista previa (derecha)
        self.create_preview_panel(main_layout)

        main_frame.setLayout(main_layout)
        parent_layout.addWidget(main_frame)

    def create_control_panel(self, parent_layout):
        """Crear el panel de control"""
        control_frame = QFrame()
        control_frame.setObjectName("controlFrame")
        control_frame.setFixedWidth(320)

        control_layout = QVBoxLayout()

        # Título del panel
        panel_title = QLabel("Cargar Datos")
        panel_title.setObjectName("panelTitle")
        control_layout.addWidget(panel_title)

        # Información del archivo
        self.file_info_label = QLabel("No hay archivo seleccionado")
        self.file_info_label.setObjectName("fileInfoLabel")
        self.file_info_label.setWordWrap(True)
        control_layout.addWidget(self.file_info_label)

        # Grupo de botones de carga
        load_group = QGroupBox("Fuentes de Datos")
        load_layout = QGridLayout()

        # Botón CSV
        self.csv_button = QPushButton("📄 Archivo CSV")
        self.csv_button.setObjectName("csvButton")
        self.csv_button.clicked.connect(self.load_csv)

        # Botón Excel
        self.excel_button = QPushButton("📊 Archivo Excel")
        self.excel_button.setObjectName("excelButton")
        self.excel_button.clicked.connect(self.load_excel)

        # Botón datos de ejemplo
        self.sample_button = QPushButton("🎯 Datos de Ejemplo")
        self.sample_button.setObjectName("sampleButton")
        self.sample_button.clicked.connect(self.load_sample_data)

        # Botón API (deshabilitado)
        self.api_button = QPushButton("🌐 Desde API")
        self.api_button.setObjectName("apiButton")
        self.api_button.setEnabled(False)

        load_layout.addWidget(self.csv_button, 0, 0)
        load_layout.addWidget(self.excel_button, 0, 1)
        load_layout.addWidget(self.sample_button, 1, 0)
        load_layout.addWidget(self.api_button, 1, 1)

        load_group.setLayout(load_layout)
        control_layout.addWidget(load_group)

        # Estado del sistema
        system_group = QGroupBox("Estado del Sistema")
        system_layout = QVBoxLayout()

        self.status_label = QLabel("✅ Sistema listo para cargar datos")
        self.status_label.setObjectName("statusLabel")

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("progressBar")
        self.progress_bar.setVisible(False)

        system_layout.addWidget(self.status_label)
        system_layout.addWidget(self.progress_bar)

        system_group.setLayout(system_layout)
        control_layout.addWidget(system_group)

        control_layout.addStretch()
        control_frame.setLayout(control_layout)
        parent_layout.addWidget(control_frame)

    def create_preview_panel(self, parent_layout):
        """Crear el panel de vista previa"""
        preview_frame = QFrame()
        preview_frame.setObjectName("previewFrame")

        preview_layout = QVBoxLayout()

        # Título del panel
        panel_title = QLabel("👁️ Vista Previa de Datos")
        panel_title.setObjectName("panelTitle")
        preview_layout.addWidget(panel_title)

        # Tabs para diferentes vistas
        self.tab_widget = QTabWidget()

        # Tab 1: Vista de tabla
        self.table_tab = QWidget()
        table_layout = QVBoxLayout()

        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        table_layout.addWidget(self.data_table)

        self.table_tab.setLayout(table_layout)
        self.tab_widget.addTab(self.table_tab, "📋 Vista de Tabla")

        # Tab 2: Estadísticas
        self.stats_tab = QWidget()
        stats_layout = QVBoxLayout()

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setPlainText("Carga un archivo para ver las estadísticas...")
        stats_layout.addWidget(self.stats_text)

        self.stats_tab.setLayout(stats_layout)
        self.tab_widget.addTab(self.stats_tab, "📊 Estadísticas")

        # Tab 3: Vista raw
        self.raw_tab = QWidget()
        raw_layout = QVBoxLayout()

        self.raw_text = QTextEdit()
        self.raw_text.setReadOnly(True)
        self.raw_text.setPlainText("Carga un archivo para ver los datos en crudo...")
        raw_layout.addWidget(self.raw_text)

        self.raw_tab.setLayout(raw_layout)
        self.tab_widget.addTab(self.raw_tab, "🔍 Vista Raw")

        preview_layout.addWidget(self.tab_widget)
        preview_frame.setLayout(preview_layout)
        parent_layout.addWidget(preview_frame)

    def create_footer(self, parent_layout):
        """Crear el footer con botones de acción"""
        footer_frame = QFrame()
        footer_frame.setObjectName("footerFrame")
        footer_frame.setFixedHeight(80)

        footer_layout = QHBoxLayout()

        # Botón usar estos datos
        self.btn_cargar = QPushButton("✅ Usar Estos Datos")
        self.btn_cargar.setObjectName("primaryButton")
        self.btn_cargar.setEnabled(False)

        # Botón exportar muestra
        self.export_button = QPushButton("📤 Exportar Muestra")
        self.export_button.setObjectName("secondaryButton")
        self.export_button.setEnabled(False)

        # Botón limpiar
        self.clear_button = QPushButton("🗑️ Limpiar")
        self.clear_button.setObjectName("dangerButton")
        self.clear_button.clicked.connect(self.clear_data)
        self.clear_button.setEnabled(False)

        footer_layout.addStretch()
        footer_layout.addWidget(self.export_button)
        footer_layout.addWidget(self.clear_button)
        footer_layout.addWidget(self.btn_cargar)

        footer_frame.setLayout(footer_layout)
        parent_layout.addWidget(footer_frame)

    def toggle_theme(self):
        """Alternar entre tema claro y oscuro"""
        theme_manager = ThemeManager()
        theme_manager.toggle_theme()

        # Actualizar el icono del botón
        if theme_manager.is_dark_theme():
            self.dark_mode_button.setText("☀️")
        else:
            self.dark_mode_button.setText("🌙")

    def load_csv(self):
        """Cargar archivo CSV"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo CSV", "", "Archivos CSV (*.csv)"
        )
        if file_path:
            try:
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(50)

                self.df = pd.read_csv(file_path)
                self.update_display()
                self.update_file_info(file_path)

                self.progress_bar.setValue(100)
                self.progress_bar.setVisible(False)
                self.status_label.setText("✅ Archivo CSV cargado exitosamente")

            except Exception as e:
                self.progress_bar.setVisible(False)
                QMessageBox.critical(self, "Error", f"Error al cargar CSV: {str(e)}")
                self.status_label.setText("❌ Error al cargar archivo")

    def load_excel(self):
        """Cargar archivo Excel"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo Excel", "", "Archivos Excel (*.xlsx *.xls)"
        )
        if file_path:
            try:
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(50)

                self.df = pd.read_excel(file_path)
                self.update_display()
                self.update_file_info(file_path)

                self.progress_bar.setValue(100)
                self.progress_bar.setVisible(False)
                self.status_label.setText("✅ Archivo Excel cargado exitosamente")

            except Exception as e:
                self.progress_bar.setVisible(False)
                QMessageBox.critical(self, "Error", f"Error al cargar Excel: {str(e)}")
                self.status_label.setText("❌ Error al cargar archivo")

    def load_sample_data(self):
        """Cargar datos de ejemplo"""
        try:
            # Crear datos de ejemplo para calidad del agua
            import numpy as np
            np.random.seed(42)

            n_samples = 1000
            data = {
                'pH': np.random.normal(7.2, 0.8, n_samples),
                'Turbidez': np.random.exponential(2, n_samples),
                'Oxigeno_Disuelto': np.random.normal(8.5, 1.2, n_samples),
                'Conductividad': np.random.normal(500, 100, n_samples),
                'Temperatura': np.random.normal(20, 5, n_samples),
                'Coliformes': np.random.poisson(10, n_samples),
                'Calidad': np.random.choice(['Excelente', 'Buena', 'Regular', 'Mala'], n_samples,
                                            p=[0.3, 0.4, 0.2, 0.1])
            }

            self.df = pd.DataFrame(data)
            self.update_display()
            self.file_info_label.setText("📊 Datos de ejemplo de calidad del agua\n1000 muestras con 7 variables")
            self.status_label.setText("✅ Datos de ejemplo cargados exitosamente")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al generar datos de ejemplo: {str(e)}")
            self.status_label.setText("❌ Error al cargar datos de ejemplo")

    def update_display(self):
        """Actualizar la visualización de datos"""
        if self.df is not None:
            # Actualizar cards de estado
            self.rows_card.update_value(f"{len(self.df):,}")
            self.cols_card.update_value(len(self.df.columns))

            # Calcular tamaño aproximado
            size_mb = self.df.memory_usage(deep=True).sum() / (1024 * 1024)
            self.size_card.update_value(f"{size_mb:.1f} MB")

            # Actualizar tabla
            self.update_table()

            # Actualizar estadísticas
            self.update_statistics()

            # Actualizar vista raw
            self.update_raw_view()

            # Habilitar botones
            self.btn_cargar.setEnabled(True)
            self.export_button.setEnabled(True)
            self.clear_button.setEnabled(True)

    def update_table(self):
        """Actualizar la tabla de vista previa"""
        if self.df is not None:
            # Mostrar solo las primeras 100 filas
            df_preview = self.df.head(100)

            self.data_table.setRowCount(len(df_preview))
            self.data_table.setColumnCount(len(df_preview.columns))
            self.data_table.setHorizontalHeaderLabels(df_preview.columns.tolist())

            for i, row in df_preview.iterrows():
                for j, value in enumerate(row):
                    item = QTableWidgetItem(str(value))
                    self.data_table.setItem(i, j, item)

            self.data_table.resizeColumnsToContents()

    def update_statistics(self):
        """Actualizar las estadísticas"""
        if self.df is not None:
            stats_text = "=== ESTADÍSTICAS DESCRIPTIVAS ===\n\n"

            # Información general
            stats_text += f"Número de filas: {len(self.df):,}\n"
            stats_text += f"Número de columnas: {len(self.df.columns)}\n"
            stats_text += f"Memoria utilizada: {self.df.memory_usage(deep=True).sum() / (1024 * 1024):.1f} MB\n\n"

            # Tipos de datos
            stats_text += "=== TIPOS DE DATOS ===\n"
            for col, dtype in self.df.dtypes.items():
                stats_text += f"{col}: {dtype}\n"

            stats_text += "\n=== VALORES NULOS ===\n"
            null_counts = self.df.isnull().sum()
            for col, null_count in null_counts.items():
                percentage = (null_count / len(self.df)) * 100
                stats_text += f"{col}: {null_count} ({percentage:.1f}%)\n"

            # Estadísticas descriptivas para columnas numéricas
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                stats_text += "\n=== ESTADÍSTICAS NUMÉRICAS ===\n"
                stats_text += self.df[numeric_cols].describe().to_string()

            self.stats_text.setPlainText(stats_text)

    def update_raw_view(self):
        """Actualizar la vista raw"""
        if self.df is not None:
            # Mostrar las primeras 20 filas en formato crudo
            raw_text = "=== VISTA CRUDA DE LOS DATOS ===\n\n"
            raw_text += "Primeras 20 filas:\n\n"
            raw_text += self.df.head(20).to_string()

            self.raw_text.setPlainText(raw_text)

    def update_file_info(self, file_path):
        """Actualizar información del archivo"""
        import os
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB

        info_text = f"📁 {file_name}\n"
        info_text += f"💾 {file_size:.1f} MB\n"
        info_text += f"📊 {len(self.df)} filas, {len(self.df.columns)} columnas"

        self.file_info_label.setText(info_text)
        self.file_card.update_value(file_name)

    def clear_data(self):
        """Limpiar todos los datos"""
        reply = QMessageBox.question(
            self, "Confirmar", "¿Estás seguro de que quieres limpiar todos los datos?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.df = None

            # Resetear cards
            self.file_card.update_value("Ninguno seleccionado")
            self.rows_card.update_value("0")
            self.cols_card.update_value("0")
            self.size_card.update_value("0 MB")

            # Limpiar tabla
            self.data_table.clear()
            self.data_table.setRowCount(0)
            self.data_table.setColumnCount(0)

            # Limpiar textos
            self.stats_text.setPlainText("Carga un archivo para ver las estadísticas...")
            self.raw_text.setPlainText("Carga un archivo para ver los datos en crudo...")

            # Resetear labels
            self.file_info_label.setText("No hay archivo seleccionado")
            self.status_label.setText("✅ Sistema listo para cargar datos")

            # Deshabilitar botones
            self.btn_cargar.setEnabled(False)
            self.export_button.setEnabled(False)
            self.clear_button.setEnabled(False)


# Para ejecutar standalone
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CargaDatos()
    window.show()
    sys.exit(app.exec_())