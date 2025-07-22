"""
wqi_window.py - Ventana visual para cálculo del Índice de Calidad del Agua
Interfaz simplificada y estética consistente con el resto de la aplicación
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox,
    QTextEdit, QSplitter, QMessageBox, QFileDialog, QCheckBox, QFrame,
    QTabWidget, QProgressBar, QGridLayout, QApplication
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont

# Importar el motor de cálculo WQI
from ui.machine_learning.wqi_calculator import WQICalculationEngine, TemporalAnalysisEngine, WQICalculationWorker

# Importar sistema de temas
try:
    from darkmode import ThemedWidget
except ImportError:
    class ThemedWidget:
        def __init__(self):
            pass

# Importar data manager
try:
    from ui.machine_learning.data_manager import get_data_manager
except ImportError:
    def get_data_manager():
        return None


class StatusCard(QFrame):
    """Tarjeta de estado para WQI"""
    def __init__(self, title, icon, value="--", color="#2196f3"):
        super().__init__()
        self.setObjectName("wqiStatusCard")
        self.setFixedHeight(100)
        self.color = color

        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(5)

        # Icono y título
        header_layout = QHBoxLayout()
        icon_label = QLabel(icon)
        icon_label.setObjectName("cardIcon")
        icon_label.setFont(QFont("Arial", 28))

        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        title_label.setFont(QFont("Arial", 11))

        header_layout.addWidget(icon_label)
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        # Valor
        self.value_label = QLabel(str(value))
        self.value_label.setObjectName("cardValue")
        self.value_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.value_label.setAlignment(Qt.AlignCenter)

        layout.addLayout(header_layout)
        layout.addWidget(self.value_label)

        self.setLayout(layout)
        self.update_style()

    def update_value(self, value, color=None):
        """Actualizar valor de la tarjeta"""
        self.value_label.setText(str(value))
        if color:
            self.color = color
            self.update_style()

    def update_style(self):
        """Actualizar estilo según color"""
        self.setStyleSheet(f"""
            #wqiStatusCard {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {self.color}20, stop:1 {self.color}10);
                border: 2px solid {self.color};
                border-radius: 12px;
            }}
            #cardValue {{
                color: {self.color};
            }}
        """)


class WQIWindow(QWidget, ThemedWidget):
    """Ventana principal para cálculo de WQI"""

    # Señal para regresar al menú
    regresar_menu = pyqtSignal()

    def __init__(self):
        QWidget.__init__(self)
        ThemedWidget.__init__(self)

        self.data = None
        self.current_results = None
        self.calculation_worker = None

        self.setWindowTitle("💧 Índice de Calidad del Agua (WQI)")
        self.setMinimumSize(1200, 800)

        self.setup_ui()
        self.apply_styles()
        self.check_for_data()

    def setup_ui(self):
        """Configurar interfaz principal"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Header
        self.create_header(main_layout)

        # Tarjetas de estado
        self.create_status_cards(main_layout)

        # Área principal con tabs
        self.create_main_area(main_layout)

        # Footer con controles
        self.create_footer(main_layout)

        self.setLayout(main_layout)

    def create_header(self, parent_layout):
        """Crear header consistente con otras ventanas"""
        header_frame = QFrame()
        header_frame.setObjectName("headerFrame")
        header_frame.setFixedHeight(80)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(20, 15, 20, 15)

        # Título principal
        title_layout = QVBoxLayout()
        title_layout.setSpacing(5)

        title_label = QLabel("💧 Índice de Calidad del Agua (WQI)")
        title_label.setObjectName("mainTitle")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))

        subtitle_label = QLabel("Cálculo mediante ponderación de parámetros fisicoquímicos")
        subtitle_label.setObjectName("subtitle")
        subtitle_label.setFont(QFont("Arial", 12))

        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)

        # Información del dataset
        self.dataset_info_label = QLabel("📊 Sin datos cargados")
        self.dataset_info_label.setObjectName("datasetInfo")
        self.dataset_info_label.setAlignment(Qt.AlignRight)

        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        header_layout.addWidget(self.dataset_info_label)

        header_frame.setLayout(header_layout)
        parent_layout.addWidget(header_frame)

    def create_status_cards(self, parent_layout):
        """Crear tarjetas de estado WQI"""
        cards_frame = QFrame()
        cards_frame.setObjectName("cardsFrame")

        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(15)

        # Crear tarjetas
        self.wqi_card = StatusCard("WQI Actual", "💧", "--", "#2196f3")
        self.quality_card = StatusCard("Calidad", "🏆", "--", "#4caf50")
        self.samples_card = StatusCard("Muestras", "📊", "0", "#ff9800")
        self.trend_card = StatusCard("Tendencia", "📈", "--", "#9c27b0")

        cards_layout.addWidget(self.wqi_card)
        cards_layout.addWidget(self.quality_card)
        cards_layout.addWidget(self.samples_card)
        cards_layout.addWidget(self.trend_card)

        cards_frame.setLayout(cards_layout)
        parent_layout.addWidget(cards_frame)

    def create_main_area(self, parent_layout):
        """Crear área principal con tabs"""
        self.main_tabs = QTabWidget()
        self.main_tabs.setObjectName("mainTabs")

        # Tab 1: Configuración
        self.create_config_tab()

        # Tab 2: Resultados
        self.create_results_tab()

        # Tab 3: Análisis
        self.create_analysis_tab()

        parent_layout.addWidget(self.main_tabs)

    def create_config_tab(self):
        """Tab de configuración"""
        config_widget = QWidget()
        config_layout = QHBoxLayout()
        config_layout.setSpacing(15)

        # Panel izquierdo - Parámetros
        left_panel = self.create_parameters_panel()
        config_layout.addWidget(left_panel, 2)

        # Panel derecho - Método y controles
        right_panel = self.create_method_panel()
        config_layout.addWidget(right_panel, 1)

        config_widget.setLayout(config_layout)
        self.main_tabs.addTab(config_widget, "⚙️ Configuración")

    def create_parameters_panel(self):
        """Panel de parámetros"""
        params_group = QGroupBox("📊 Parámetros del Agua")
        params_layout = QVBoxLayout()

        # Info
        info_label = QLabel("Selecciona los parámetros a incluir en el cálculo:")
        info_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        params_layout.addWidget(info_label)

        # Tabla de parámetros
        self.params_table = QTableWidget()
        self.params_table.setColumnCount(4)
        self.params_table.setHorizontalHeaderLabels([
            "Incluir", "Parámetro", "Peso (%)", "Rango Óptimo"
        ])

        # Llenar tabla con parámetros estándar
        self.setup_parameters_table()

        params_layout.addWidget(self.params_table)

        # Botones de control
        buttons_layout = QHBoxLayout()

        normalize_btn = QPushButton("⚖️ Normalizar Pesos")
        normalize_btn.setObjectName("secondaryButton")
        normalize_btn.clicked.connect(self.normalize_weights)

        reset_btn = QPushButton("🔄 Restaurar")
        reset_btn.setObjectName("secondaryButton")
        reset_btn.clicked.connect(self.reset_parameters)

        buttons_layout.addWidget(normalize_btn)
        buttons_layout.addWidget(reset_btn)
        buttons_layout.addStretch()

        params_layout.addLayout(buttons_layout)

        params_group.setLayout(params_layout)
        return params_group

    def create_method_panel(self):
        """Panel de método y cálculo"""
        method_group = QGroupBox("🧮 Método de Cálculo")
        method_layout = QVBoxLayout()

        # Selector de método
        method_label = QLabel("Método:")
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "NSF WQI - National Sanitation Foundation",
            "CCME WQI - Canadian Council",
            "Aritmético Ponderado"
        ])
        self.method_combo.currentTextChanged.connect(self.on_method_changed)

        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_combo)

        # Descripción del método
        self.method_description = QTextEdit()
        self.method_description.setReadOnly(True)
        self.method_description.setMaximumHeight(150)
        self.method_description.setObjectName("methodDescription")

        method_layout.addWidget(self.method_description)

        # Fórmula
        formula_label = QLabel("📐 Fórmula:")
        formula_label.setFont(QFont("Arial", 11, QFont.Bold))

        self.formula_label = QLabel()
        self.formula_label.setObjectName("formulaLabel")
        self.formula_label.setAlignment(Qt.AlignCenter)
        self.formula_label.setMinimumHeight(50)

        method_layout.addWidget(formula_label)
        method_layout.addWidget(self.formula_label)

        # Estado y progreso
        self.status_label = QLabel("✅ Listo para calcular")
        self.status_label.setObjectName("statusLabel")

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        method_layout.addWidget(self.status_label)
        method_layout.addWidget(self.progress_bar)

        method_layout.addStretch()

        # Botón calcular
        self.calculate_btn = QPushButton("🧮 Calcular WQI")
        self.calculate_btn.setObjectName("primaryButton")
        self.calculate_btn.setMinimumHeight(50)
        self.calculate_btn.clicked.connect(self.calculate_wqi)

        method_layout.addWidget(self.calculate_btn)

        method_group.setLayout(method_layout)

        # Actualizar descripción inicial
        self.on_method_changed(self.method_combo.currentText())

        return method_group

    def create_results_tab(self):
        """Tab de resultados"""
        results_widget = QWidget()
        results_layout = QVBoxLayout()

        # Resumen de resultados
        summary_group = QGroupBox("📊 Resumen de Resultados")
        summary_layout = QHBoxLayout()

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)

        summary_layout.addWidget(self.results_text)
        summary_group.setLayout(summary_layout)

        results_layout.addWidget(summary_group)

        # Tabla de resultados detallados
        details_group = QGroupBox("📋 Resultados Detallados")
        details_layout = QVBoxLayout()

        self.results_table = QTableWidget()
        self.results_table.setAlternatingRowColors(True)

        details_layout.addWidget(self.results_table)
        details_group.setLayout(details_layout)

        results_layout.addWidget(details_group)

        results_widget.setLayout(results_layout)
        self.main_tabs.addTab(results_widget, "📊 Resultados")

    def create_analysis_tab(self):
        """Tab de análisis"""
        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout()

        # Controles de análisis
        controls_layout = QHBoxLayout()

        trend_btn = QPushButton("📈 Análisis Temporal")
        trend_btn.setObjectName("analysisButton")
        trend_btn.clicked.connect(self.analyze_trends)

        compare_btn = QPushButton("⚖️ Comparar Métodos")
        compare_btn.setObjectName("analysisButton")
        compare_btn.clicked.connect(self.compare_methods)

        export_btn = QPushButton("📤 Exportar Informe")
        export_btn.setObjectName("analysisButton")
        export_btn.clicked.connect(self.export_report)

        controls_layout.addWidget(trend_btn)
        controls_layout.addWidget(compare_btn)
        controls_layout.addWidget(export_btn)
        controls_layout.addStretch()

        analysis_layout.addLayout(controls_layout)

        # Área de análisis
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setPlaceholderText(
            "Los análisis aparecerán aquí...\n\n"
            "• Análisis temporal: Tendencias del WQI en el tiempo\n"
            "• Comparación: Diferencias entre métodos de cálculo\n"
            "• Exportar: Generar informe completo en PDF/Excel"
        )

        analysis_layout.addWidget(self.analysis_text)

        analysis_widget.setLayout(analysis_layout)
        self.main_tabs.addTab(analysis_widget, "📈 Análisis")

    def create_footer(self, parent_layout):
        """Crear footer con controles principales"""
        footer_frame = QFrame()
        footer_frame.setObjectName("footerFrame")

        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(20, 15, 20, 15)

        # Información
        self.info_label = QLabel("💡 Configura los parámetros y calcula el WQI")
        self.info_label.setObjectName("infoLabel")

        # Botones
        load_data_btn = QPushButton("📂 Cargar Datos")
        load_data_btn.setObjectName("secondaryButton")
        load_data_btn.clicked.connect(self.load_data)

        help_btn = QPushButton("❓ Ayuda")
        help_btn.setObjectName("secondaryButton")
        help_btn.clicked.connect(self.show_help)

        self.back_btn = QPushButton("◀ Regresar")
        self.back_btn.setObjectName("backButton")
        self.back_btn.clicked.connect(self.go_back)

        footer_layout.addWidget(self.info_label)
        footer_layout.addStretch()
        footer_layout.addWidget(load_data_btn)
        footer_layout.addWidget(help_btn)
        footer_layout.addWidget(self.back_btn)

        footer_frame.setLayout(footer_layout)
        parent_layout.addWidget(footer_frame)

    def setup_parameters_table(self):
        """Configurar tabla de parámetros"""
        parameters = WQICalculationEngine.NSF_PARAMETERS

        self.params_table.setRowCount(len(parameters))

        for i, (param, config) in enumerate(parameters.items()):
            # Checkbox
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            self.params_table.setCellWidget(i, 0, checkbox)

            # Nombre
            self.params_table.setItem(i, 1, QTableWidgetItem(param.replace('_', ' ')))

            # Peso
            weight_spin = QSpinBox()
            weight_spin.setRange(0, 100)
            weight_spin.setValue(int(config['weight'] * 100))
            weight_spin.setSuffix("%")
            self.params_table.setCellWidget(i, 2, weight_spin)

            # Rango óptimo
            optimal_text = f"{config['optimal'][0]} - {config['optimal'][1]}"
            self.params_table.setItem(i, 3, QTableWidgetItem(optimal_text))

        self.params_table.resizeColumnsToContents()

    def check_for_data(self):
        """Verificar si hay datos disponibles"""
        try:
            dm = get_data_manager()
            if dm and dm.has_data():
                self.data = dm.get_data()
                self.update_dataset_info()
                self.check_parameter_mapping()
                return True
        except:
            pass

        self.info_label.setText("⚠️ No hay datos cargados, usa 'Cargar Datos'")
        return False

    def check_parameter_mapping(self):
        """Verificar y mapear parámetros disponibles"""
        if self.data is None:
            return

        # Importar el mapper
        try:
            from ui.machine_learning.wqi_parameter_mapper import WQIParameterMapper
        except ImportError:
            # Si no existe el mapper, usar mapeo manual básico
            self.parameter_mapping = self.get_manual_parameter_mapping()
            return

        # Mapear parámetros automáticamente
        self.parameter_mapping = WQIParameterMapper.map_parameters(self.data.columns.tolist())
        available_params = list(self.parameter_mapping.keys())

        if available_params:
            self.info_label.setText(f"✅ {len(available_params)} parámetros WQI detectados")
            self.update_parameters_table_with_mapping(available_params)
        else:
            self.info_label.setText("⚠️ No se encontraron parámetros WQI en los datos")

    def get_manual_parameter_mapping(self):
        """Mapeo manual básico para compatibilidad"""
        mapping = {}
        columns = self.data.columns.tolist()

        # Mapeo manual de los nombres más comunes
        mappings = {
            'pH': ['pH', 'ph', 'PH'],
            'Oxigeno_Disuelto': ['DO', 'Oxigeno_Disuelto', 'DissolvedOxygen'],
            'DBO5': ['BOD5', 'DBO5', 'BOD'],
            'Coliformes_Fecales': ['FC', 'Coliformes_Fecales', 'TC'],
            'Temperatura': ['WT', 'Temperatura', 'Temperature'],
            'Fosforo_Total': ['TP', 'Fosforo_Total', 'TotalPhosphorus'],
            'Nitrato': ['NO3', 'Nitrato', 'Nitrate'],
            'Turbiedad': ['TBD', 'Turbiedad', 'Turbidity'],
            'Solidos_Totales': ['TSS', 'Solidos_Totales', 'TS']
        }

        for standard, alternatives in mappings.items():
            for col in columns:
                if col in alternatives:
                    mapping[standard] = col
                    break

        return mapping

    def update_parameters_table_with_mapping(self, available_params):
        """Actualizar tabla mostrando solo parámetros disponibles"""
        # Marcar como no disponibles los parámetros que no están en los datos
        for i in range(self.params_table.rowCount()):
            param_name = self.params_table.item(i, 1).text().replace(' ', '_')
            checkbox = self.params_table.cellWidget(i, 0)

            if param_name in available_params:
                checkbox.setEnabled(True)
                checkbox.setChecked(True)
                # Mostrar nombre real de la columna
                if hasattr(self, 'parameter_mapping') and param_name in self.parameter_mapping:
                    real_name = self.parameter_mapping[param_name]
                    self.params_table.item(i, 1).setToolTip(f"Columna en datos: {real_name}")
            else:
                checkbox.setEnabled(False)
                checkbox.setChecked(False)
                self.params_table.item(i, 1).setToolTip("No disponible en los datos")

    def update_dataset_info(self):
        """Actualizar información del dataset"""
        if self.data is not None:
            info_text = f"📊 {len(self.data):,} muestras × {len(self.data.columns)} variables"
            self.dataset_info_label.setText(info_text)
            self.samples_card.update_value(f"{len(self.data):,}")
        else:
            self.dataset_info_label.setText("📊 Sin datos cargados")
            self.samples_card.update_value("0")

    def on_method_changed(self, method_text):
        """Actualizar descripción del método"""
        descriptions = {
            "NSF WQI - National Sanitation Foundation": {
                "desc": "Método más utilizado internacionalmente. Combina 9 parámetros "
                       "mediante un producto ponderado. Ideal para comparaciones globales.",
                "formula": "WQI = Π(Qi^Wi)"
            },
            "CCME WQI - Canadian Council": {
                "desc": "Método canadiense que evalúa el cumplimiento de objetivos de calidad. "
                       "Considera frecuencia, alcance y amplitud de las excedencias.",
                "formula": "WQI = 100 - √(F1² + F2² + F3²) / 1.732"
            },
            "Aritmético Ponderado": {
                "desc": "Método simple que calcula el promedio ponderado de los índices. "
                       "Fácil de interpretar y comunicar a no especialistas.",
                "formula": "WQI = Σ(Wi × Qi)"
            }
        }

        method_key = method_text
        if method_key in descriptions:
            self.method_description.setPlainText(descriptions[method_key]["desc"])
            self.formula_label.setText(descriptions[method_key]["formula"])

    def normalize_weights(self):
        """Normalizar pesos a 100%"""
        total_weight = 0
        active_params = []

        for i in range(self.params_table.rowCount()):
            checkbox = self.params_table.cellWidget(i, 0)
            if checkbox.isChecked():
                weight_spin = self.params_table.cellWidget(i, 2)
                total_weight += weight_spin.value()
                active_params.append((i, weight_spin))

        if total_weight > 0 and len(active_params) > 0:
            for i, weight_spin in active_params:
                normalized = int((weight_spin.value() / total_weight) * 100)
                weight_spin.setValue(normalized)

            self.status_label.setText("✅ Pesos normalizados a 100%")
        else:
            QMessageBox.warning(self, "Advertencia", "Selecciona al menos un parámetro")

    def reset_parameters(self):
        """Restaurar parámetros por defecto"""
        self.setup_parameters_table()
        self.status_label.setText("✅ Parámetros restaurados")

    def load_data(self):
        """Cargar datos desde archivo"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Cargar datos", "", "CSV (*.csv);;Excel (*.xlsx)"
        )

        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.data = pd.read_csv(file_path)
                else:
                    self.data = pd.read_excel(file_path)

                self.update_dataset_info()
                self.check_parameter_mapping()

                # Guardar en data manager si existe
                try:
                    dm = get_data_manager()
                    if dm:
                        dm.set_data(self.data, source="wqi_window")
                except:
                    pass

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al cargar datos: {str(e)}")

    def calculate_wqi(self):
        """Calcular WQI"""
        if self.data is None:
            if not self.check_for_data():
                QMessageBox.warning(self, "Sin datos", "Por favor carga datos primero")
                return

        # Obtener método
        method_map = {
            "NSF WQI - National Sanitation Foundation": "NSF",
            "CCME WQI - Canadian Council": "CCME",
            "Aritmético Ponderado": "Weighted_Arithmetic"
        }
        method = method_map.get(self.method_combo.currentText(), "NSF")

        # Obtener parámetros activos y mapearlos
        parameters = {}
        weights = {}

        for i in range(self.params_table.rowCount()):
            checkbox = self.params_table.cellWidget(i, 0)
            if checkbox.isChecked() and checkbox.isEnabled():
                param_name = self.params_table.item(i, 1).text().replace(' ', '_')
                weight = self.params_table.cellWidget(i, 2).value() / 100.0

                if param_name in WQICalculationEngine.NSF_PARAMETERS:
                    parameters[param_name] = WQICalculationEngine.NSF_PARAMETERS[param_name]
                    weights[param_name] = weight

        if not parameters:
            QMessageBox.warning(self, "Sin parámetros", "No hay parámetros disponibles en los datos")
            return

        # Usar el mapeo de parámetros si existe
        if hasattr(self, 'parameter_mapping'):
            # Verificar que los parámetros mapeados existen en los datos
            missing = []
            for param in parameters.keys():
                if param not in self.parameter_mapping:
                    missing.append(param)
                elif self.parameter_mapping[param] not in self.data.columns:
                    missing.append(f"{param} ({self.parameter_mapping[param]})")

            if missing:
                QMessageBox.warning(
                    self, "Parámetros faltantes",
                    f"Los siguientes parámetros no están mapeados correctamente:\n{', '.join(missing)}"
                )
                return
        else:
            # Sin mapeo, verificar directamente
            missing = [p for p in parameters.keys() if p not in self.data.columns]
            if missing:
                QMessageBox.warning(
                    self, "Parámetros faltantes",
                    f"Los siguientes parámetros no están en los datos:\n{', '.join(missing)}"
                )
                return

        # Iniciar cálculo con mapeo
        self.start_calculation_with_mapping(method, parameters, weights)

    def start_calculation_with_mapping(self, method, parameters, weights):
        """Iniciar cálculo con mapeo de parámetros"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.calculate_btn.setEnabled(False)

        # Crear un DataFrame temporal con columnas mapeadas
        mapped_data = self.data.copy()

        if hasattr(self, 'parameter_mapping'):
            # Renombrar columnas según el mapeo
            rename_dict = {}
            for standard_name, data_name in self.parameter_mapping.items():
                if data_name in mapped_data.columns and standard_name != data_name:
                    rename_dict[data_name] = standard_name

            if rename_dict:
                mapped_data = mapped_data.rename(columns=rename_dict)

        # Usar el worker con los datos mapeados
        self.calculation_worker = WQICalculationWorker(mapped_data, method, parameters, weights)
        self.calculation_worker.progress_updated.connect(self.progress_bar.setValue)
        self.calculation_worker.status_updated.connect(self.status_label.setText)
        self.calculation_worker.calculation_finished.connect(self.on_calculation_finished)
        self.calculation_worker.error_occurred.connect(self.on_calculation_error)

        self.calculation_worker.start()

    def on_calculation_finished(self, results):
        """Manejar finalización del cálculo"""
        self.current_results = results
        self.progress_bar.setVisible(False)
        self.calculate_btn.setEnabled(True)

        # Actualizar tarjetas
        stats = results['statistics']
        if stats:
            wqi_mean = stats['mean']
            self.wqi_card.update_value(f"{wqi_mean:.1f}")

            # Clasificar calidad promedio
            classification = WQICalculationEngine.classify_water_quality(wqi_mean)
            self.quality_card.update_value(classification['label'], classification['color'])

        # Actualizar resultados
        self.update_results_display()

        # Cambiar a tab de resultados
        self.main_tabs.setCurrentIndex(1)

        self.status_label.setText("✅ Cálculo completado")
        self.info_label.setText(f"✅ WQI calculado para {results['total_samples']} muestras")

    def on_calculation_error(self, error_message):
        """Manejar error en cálculo"""
        self.progress_bar.setVisible(False)
        self.calculate_btn.setEnabled(True)
        self.status_label.setText("❌ Error en cálculo")
        QMessageBox.critical(self, "Error", error_message)

    def update_results_display(self):
        """Actualizar display de resultados"""
        if not self.current_results:
            return

        results = self.current_results
        stats = results['statistics']

        # Resumen textual
        summary_text = f"=== 📊 RESULTADOS WQI ===\n\n"
        summary_text += f"Método: {results['method']}\n"
        summary_text += f"Muestras analizadas: {results['total_samples']}\n\n"

        if stats:
            summary_text += f"📈 ESTADÍSTICAS:\n"
            summary_text += f"  • WQI Promedio: {stats['mean']:.2f}\n"
            summary_text += f"  • Desv. Estándar: {stats['std']:.2f}\n"
            summary_text += f"  • Mínimo: {stats['min']:.2f}\n"
            summary_text += f"  • Máximo: {stats['max']:.2f}\n\n"

        if 'quality_distribution' in results:
            summary_text += f"🏆 DISTRIBUCIÓN DE CALIDAD:\n"
            for quality, count in results['quality_distribution'].items():
                percentage = (count / results['total_samples']) * 100
                summary_text += f"  • {quality}: {count} ({percentage:.1f}%)\n"

        self.results_text.setPlainText(summary_text)

        # Tabla de resultados
        self.results_table.setRowCount(min(100, len(results['results'])))
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Muestra", "WQI", "Clasificación"])

        for i, result in enumerate(results['results'][:100]):
            self.results_table.setItem(i, 0, QTableWidgetItem(str(result.get('index', i))))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{result['wqi']:.2f}"))

            classification = result.get('classification', {})
            class_item = QTableWidgetItem(classification.get('label', 'N/A'))
            self.results_table.setItem(i, 2, class_item)

        self.results_table.resizeColumnsToContents()

    def analyze_trends(self):
        """Analizar tendencias temporales"""
        if self.current_results is None:
            QMessageBox.warning(self, "Sin resultados", "Calcula WQI primero")
            return

        try:
            # Crear DataFrame temporal con resultados
            wqi_values = [r['wqi'] for r in self.current_results['results']]
            temp_df = self.data.copy()
            temp_df['WQI'] = wqi_values

            # Buscar columna de fecha
            date_column = None
            for col in temp_df.columns:
                if 'fecha' in col.lower() or 'date' in col.lower():
                    date_column = col
                    break

            if date_column:
                analysis = TemporalAnalysisEngine.analyze_temporal_trends(temp_df, date_column, 'WQI')

                analysis_text = "=== 📈 ANÁLISIS TEMPORAL ===\n\n"

                if 'tendencia_general' in analysis:
                    trend = analysis['tendencia_general']
                    analysis_text += f"📊 TENDENCIA GENERAL:\n"
                    analysis_text += f"  • Dirección: {trend['direccion']}\n"
                    analysis_text += f"  • Correlación: {trend['correlacion']:.3f}\n"
                    analysis_text += f"  • Significativa: {'Sí' if trend['significativa'] else 'No'}\n\n"

                    # Actualizar tarjeta de tendencia
                    icon = "📈" if trend['direccion'] == 'Mejorando' else "📉" if trend['direccion'] == 'Empeorando' else "➡️"
                    color = "#4caf50" if trend['direccion'] == 'Mejorando' else "#f44336" if trend['direccion'] == 'Empeorando' else "#ff9800"
                    self.trend_card.update_value(trend['direccion'], color)

                if 'variacion_estacional' in analysis:
                    seasonal = analysis['variacion_estacional']
                    analysis_text += f"📅 VARIACIÓN ESTACIONAL:\n"
                    analysis_text += f"  • Mejor mes: {seasonal['mejor_mes']}\n"
                    analysis_text += f"  • Peor mes: {seasonal['peor_mes']}\n\n"

                self.analysis_text.setPlainText(analysis_text)
            else:
                self.analysis_text.setPlainText("⚠️ No se encontró columna de fecha para análisis temporal")

        except Exception as e:
            self.analysis_text.setPlainText(f"❌ Error en análisis temporal: {str(e)}")

    def compare_methods(self):
        """Comparar diferentes métodos de cálculo"""
        if self.data is None:
            QMessageBox.warning(self, "Sin datos", "Carga datos primero")
            return

        try:
            # Obtener parámetros activos
            parameters = {}
            weights = {}

            for i in range(self.params_table.rowCount()):
                checkbox = self.params_table.cellWidget(i, 0)
                if checkbox.isChecked():
                    param_name = self.params_table.item(i, 1).text().replace(' ', '_')
                    weight = self.params_table.cellWidget(i, 2).value() / 100.0

                    if param_name in WQICalculationEngine.NSF_PARAMETERS:
                        parameters[param_name] = WQICalculationEngine.NSF_PARAMETERS[param_name]
                        weights[param_name] = weight

            comparison_text = "=== ⚖️ COMPARACIÓN DE MÉTODOS ===\n\n"

            methods = ['NSF', 'CCME', 'Weighted_Arithmetic']
            method_results = {}

            for method in methods:
                method_wqi = []
                for _, row in self.data.iterrows():
                    if method == 'NSF':
                        result = WQICalculationEngine.calculate_nsf_wqi(row, parameters, weights)
                    elif method == 'CCME':
                        result = WQICalculationEngine.calculate_ccme_wqi(row, parameters)
                    else:
                        result = WQICalculationEngine.calculate_weighted_arithmetic_wqi(row, parameters, weights)

                    method_wqi.append(result.get('wqi', 0))

                method_results[method] = method_wqi

                comparison_text += f"📊 {method}:\n"
                comparison_text += f"  • Media: {np.mean(method_wqi):.2f}\n"
                comparison_text += f"  • Desv. Est.: {np.std(method_wqi):.2f}\n"
                comparison_text += f"  • Rango: {np.min(method_wqi):.2f} - {np.max(method_wqi):.2f}\n\n"

            # Correlaciones
            comparison_text += "🔗 CORRELACIONES:\n"
            for i, method1 in enumerate(methods):
                for method2 in methods[i+1:]:
                    corr = np.corrcoef(method_results[method1], method_results[method2])[0, 1]
                    comparison_text += f"  • {method1} vs {method2}: {corr:.3f}\n"

            self.analysis_text.setPlainText(comparison_text)

        except Exception as e:
            self.analysis_text.setPlainText(f"❌ Error en comparación: {str(e)}")

    def export_report(self):
        """Exportar informe completo"""
        if self.current_results is None:
            QMessageBox.warning(self, "Sin resultados", "Calcula WQI primero")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Exportar Informe WQI",
            f"informe_wqi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            "Excel (*.xlsx);;CSV (*.csv)"
        )

        if file_path:
            try:
                # Crear DataFrame con resultados
                results_data = []
                for result in self.current_results['results']:
                    results_data.append({
                        'Índice': result.get('index', ''),
                        'WQI': result.get('wqi', 0),
                        'Clasificación': result.get('classification', {}).get('label', ''),
                        'Color': result.get('classification', {}).get('color', '')
                    })

                results_df = pd.DataFrame(results_data)

                if file_path.endswith('.xlsx'):
                    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                        # Hoja de resultados
                        results_df.to_excel(writer, sheet_name='Resultados', index=False)

                        # Hoja de estadísticas
                        stats_df = pd.DataFrame([self.current_results['statistics']])
                        stats_df.to_excel(writer, sheet_name='Estadísticas', index=False)

                        # Hoja de distribución
                        dist_df = pd.DataFrame(
                            list(self.current_results['quality_distribution'].items()),
                            columns=['Calidad', 'Cantidad']
                        )
                        dist_df.to_excel(writer, sheet_name='Distribución', index=False)
                else:
                    results_df.to_csv(file_path, index=False)

                QMessageBox.information(self, "✅ Éxito", "Informe exportado correctamente")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al exportar: {str(e)}")

    def show_help(self):
        """Mostrar ayuda"""
        help_text = """
💧 ÍNDICE DE CALIDAD DEL AGUA (WQI)

El WQI combina múltiples parámetros fisicoquímicos en un índice único 
entre 0 y 100 que indica la calidad general del agua.

📊 MÉTODOS DISPONIBLES:

• NSF WQI: Método más utilizado, usa producto ponderado
• CCME WQI: Evalúa cumplimiento de objetivos
• Aritmético: Promedio ponderado simple

🎯 INTERPRETACIÓN:
• 90-100: Excelente
• 70-89: Buena
• 50-69: Regular
• 25-49: Deficiente
• 0-24: Muy Deficiente

💡 PASOS:
1. Cargar datos con parámetros del agua
2. Seleccionar parámetros a incluir
3. Ajustar pesos (deben sumar 100%)
4. Elegir método de cálculo
5. Calcular WQI
6. Analizar resultados y tendencias
"""

        QMessageBox.information(self, "❓ Ayuda - WQI", help_text)

    def go_back(self):
        """Regresar al menú principal"""
        self.regresar_menu.emit()
        self.close()

    def apply_styles(self):
        """Aplicar estilos consistentes con el resto de la aplicación"""
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
                background-color: #f8f9fa;
            }
            
            /* Header */
            #headerFrame {
                background-color: #ffffff;
                border-bottom: 2px solid #e9ecef;
                border-radius: 8px;
            }
            
            #mainTitle {
                color: #2c3e50;
                font-size: 20px;
                font-weight: bold;
            }
            
            #subtitle {
                color: #666;
                font-size: 12px;
            }
            
            #datasetInfo {
                color: #666;
                font-size: 11px;
                font-style: italic;
            }
            
            /* Tarjetas de estado */
            #cardsFrame {
                background-color: transparent;
                padding: 10px 0;
            }
            
            #cardIcon {
                color: inherit;
            }
            
            #cardTitle {
                color: #666;
            }
            
            #cardValue {
                color: inherit;
            }
            
            /* Tabs */
            #mainTabs::pane {
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
                font-weight: bold;
                min-width: 100px;
            }
            
            QTabBar::tab:selected {
                background-color: #ffffff;
                border-bottom: 1px solid #ffffff;
                color: #007bff;
            }
            
            QTabBar::tab:hover {
                background-color: #e9ecef;
            }
            
            /* Grupos */
            QGroupBox {
                font-weight: bold;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: #ffffff;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                color: #495057;
                background-color: #ffffff;
            }
            
            /* Tabla */
            QTableWidget {
                gridline-color: #dee2e6;
                selection-background-color: #007bff;
                alternate-background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
            
            QTableWidget::item {
                padding: 6px;
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
            }
            
            /* Botones */
            #primaryButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
                min-height: 40px;
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
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                min-height: 35px;
            }
            
            #secondaryButton:hover {
                background-color: #5a6268;
            }
            
            #analysisButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                min-height: 35px;
            }
            
            #analysisButton:hover {
                background-color: #0056b3;
            }
            
            #backButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                min-height: 35px;
            }
            
            #backButton:hover {
                background-color: #c82333;
            }
            
            /* Otros elementos */
            QTextEdit {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background-color: #ffffff;
                padding: 10px;
            }
            
            #methodDescription {
                background-color: #f8f9fa;
                font-size: 11px;
                color: #666;
            }
            
            #formulaLabel {
                background-color: #e3f2fd;
                border: 2px solid #2196f3;
                border-radius: 6px;
                padding: 10px;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                font-weight: bold;
                color: #1565c0;
            }
            
            #statusLabel {
                color: #28a745;
                font-weight: bold;
                padding: 5px;
            }
            
            #infoLabel {
                color: #666;
                font-size: 12px;
            }
            
            #footerFrame {
                background-color: #ffffff;
                border-top: 2px solid #e9ecef;
                border-radius: 8px;
                padding: 10px;
            }
            
            QProgressBar {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background-color: #f8f9fa;
                text-align: center;
                font-weight: bold;
            }
            
            QProgressBar::chunk {
                background-color: #007bff;
                border-radius: 3px;
            }
            
            QSpinBox, QDoubleSpinBox {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 5px;
                background-color: #ffffff;
            }
            
            QComboBox {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 6px;
                background-color: #ffffff;
                min-width: 200px;
            }
            
            QCheckBox {
                spacing: 5px;
            }
            
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)


# Para pruebas independientes
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WQIWindow()
    window.show()
    sys.exit(app.exec_())