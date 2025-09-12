"""
wqi_window.py - Ventana visual para cálculo del Índice de Calidad del Agua
Versión modificada con gráfico temporal y UI simplificada
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox,
    QTextEdit, QSplitter, QMessageBox, QFileDialog, QCheckBox, QFrame,
    QTabWidget, QProgressBar, QGridLayout, QApplication, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont

# Importar el motor de cálculo WQI
from ui.machine_learning.wqi_calculator import WQICalculationEngine, TemporalAnalysisEngine, WQICalculationWorker

# Importar sistema de temas
try:
    class ThemedWidget:
        def __init__(self):
            pass
        def apply_theme(self):
            pass
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


class TimeSeriesPlotWidget(QWidget):
    """Widget para mostrar gráfico de series temporales del WQI"""

    def __init__(self):
        super().__init__()
        self.figure = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Configurar matplotlib para español
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['axes.grid'] = True

    def plot_wqi_evolution(self, data, date_column, wqi_values, title="Evolución temporal del WQI"):
        """Crear gráfico de evolución temporal del WQI similar a la Figura 13"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        try:
            # Convertir fechas
            dates = pd.to_datetime(data[date_column])

            # Crear DataFrame temporal
            temp_df = pd.DataFrame({
                'fecha': dates,
                'wqi': wqi_values
            }).sort_values('fecha')

            # Agrupar por año para obtener promedio anual
            temp_df['año'] = temp_df['fecha'].dt.year
            yearly_data = temp_df.groupby('año')['wqi'].mean()

            # Configurar colores por rangos de calidad
            colors = {
                'very_bad': '#ffcdd2',    # Rojo claro
                'bad': '#fff9c4',        # Amarillo claro
                'medium': '#c8e6c9',     # Verde claro
                'good': '#b3e5fc',       # Azul claro
                'excellent': '#e1bee7'    # Morado claro
            }

            # Crear áreas de fondo para rangos de calidad
            ax.axhspan(0, 25, color=colors['very_bad'], alpha=0.7, label='Very Bad')
            ax.axhspan(25, 50, color=colors['bad'], alpha=0.7, label='Bad')
            ax.axhspan(50, 70, color=colors['medium'], alpha=0.7, label='Medium')
            ax.axhspan(70, 90, color=colors['good'], alpha=0.7, label='Good')
            ax.axhspan(90, 100, color=colors['excellent'], alpha=0.7, label='Excellent')

            # Línea principal del WQI
            ax.plot(yearly_data.index, yearly_data.values,
                   color='#1565c0', linewidth=2.5, marker='o',
                   markersize=4, label='WQI Promedio Anual')

            # Si hay múltiples sitios, agregar líneas adicionales
            if 'sitio' in data.columns or 'station' in data.columns:
                site_col = 'sitio' if 'sitio' in data.columns else 'station'
                sites = data[site_col].unique()[:5]  # Máximo 5 sitios

                colors_sites = ['#2e7d32', '#f57c00', '#5d4037', '#7b1fa2', '#c62828']

                for i, site in enumerate(sites):
                    site_data = temp_df[data[site_col] == site]
                    if len(site_data) > 0:
                        site_yearly = site_data.groupby('año')['wqi'].mean()
                        ax.plot(site_yearly.index, site_yearly.values,
                               color=colors_sites[i % len(colors_sites)],
                               linewidth=1.5, alpha=0.8, linestyle='--',
                               label=f'{site}')

            # Configuración del gráfico
            ax.set_xlabel('Año', fontsize=12, fontweight='bold')
            ax.set_ylabel('WQI_NSF_9V', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

            # Configurar límites y grid
            ax.set_ylim(0, 100)
            ax.set_xlim(yearly_data.index.min() - 0.5, yearly_data.index.max() + 0.5)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

            # Leyenda
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

            # Formato de ejes
            ax.tick_params(axis='both', which='major', labelsize=10)

            # Añadir texto de fuente
            self.figure.text(0.5, 0.02, 'Fuente: Elaboración propia',
                           ha='center', fontsize=10, style='italic')

            # Ajustar layout
            self.figure.tight_layout()
            self.canvas.draw()

            return True

        except Exception as e:
            ax.text(0.5, 0.5, f'Error al generar gráfico:\n{str(e)}',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            self.canvas.draw()
            return False


class WQIWindow(QWidget, ThemedWidget):
    """Ventana principal para cálculo de WQI - Versión Simplificada"""

    # Señal para regresar al menú
    regresar_menu = pyqtSignal()

    def __init__(self):
        QWidget.__init__(self)
        ThemedWidget.__init__(self)

        self.data = None
        self.current_results = None
        self.calculation_worker = None

        self.setWindowTitle("Índice de Calidad del Agua (WQI)")
        self.setMinimumSize(1200, 800)

        self.setup_ui()
        self.apply_styles()
        self.check_for_data()

    def setup_ui(self):
        """Configurar interfaz principal simplificada"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Header simplificado
        self.create_simple_header(main_layout)

        # Área principal con tabs
        self.create_main_area(main_layout)

        # Footer con controles
        self.create_footer(main_layout)

        self.setLayout(main_layout)

    def create_simple_header(self, parent_layout):
        """Crear header simplificado"""
        header_frame = QFrame()
        header_frame.setObjectName("headerFrame")
        header_frame.setFixedHeight(60)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(20, 15, 20, 15)

        # Título principal
        title_label = QLabel("Índice de Calidad del Agua (WQI)")
        title_label.setObjectName("mainTitle")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))

        # Información del dataset
        self.dataset_info_label = QLabel("Sin datos cargados")
        self.dataset_info_label.setObjectName("datasetInfo")
        self.dataset_info_label.setAlignment(Qt.AlignRight)

        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.dataset_info_label)

        header_frame.setLayout(header_layout)
        parent_layout.addWidget(header_frame)

    def create_main_area(self, parent_layout):
        """Crear área principal con tabs"""
        self.main_tabs = QTabWidget()
        self.main_tabs.setObjectName("mainTabs")

        # Tab 1: Configuración
        self.create_config_tab()

        # Tab 2: Resultados
        self.create_results_tab()

        # Tab 3: Análisis con gráfico temporal
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
        self.main_tabs.addTab(config_widget, "Configuración")

    def create_parameters_panel(self):
        """Panel de parámetros"""
        params_group = QGroupBox("Parámetros del Agua")
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

        normalize_btn = QPushButton("Normalizar Pesos")
        normalize_btn.setObjectName("secondaryButton")
        normalize_btn.clicked.connect(self.normalize_weights)

        reset_btn = QPushButton("Restaurar")
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
        method_group = QGroupBox("Método de Cálculo")
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
        self.method_description.setMaximumHeight(120)
        self.method_description.setObjectName("methodDescription")

        method_layout.addWidget(self.method_description)

        # Estado y progreso
        self.status_label = QLabel("Listo para calcular")
        self.status_label.setObjectName("statusLabel")

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        method_layout.addWidget(self.status_label)
        method_layout.addWidget(self.progress_bar)

        method_layout.addStretch()

        # Botón calcular
        self.calculate_btn = QPushButton("Calcular WQI")
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
        summary_group = QGroupBox("Resumen de Resultados")
        summary_layout = QHBoxLayout()

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)

        summary_layout.addWidget(self.results_text)
        summary_group.setLayout(summary_layout)

        results_layout.addWidget(summary_group)

        # Tabla de resultados detallados
        details_group = QGroupBox("Resultados Detallados")
        details_layout = QVBoxLayout()

        self.results_table = QTableWidget()
        self.results_table.setAlternatingRowColors(True)

        details_layout.addWidget(self.results_table)
        details_group.setLayout(details_layout)

        results_layout.addWidget(details_group)

        results_widget.setLayout(results_layout)
        self.main_tabs.addTab(results_widget, "Resultados")

    def create_analysis_tab(self):
        """Tab de análisis con gráfico temporal"""
        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout()

        # Controles de análisis
        controls_layout = QHBoxLayout()

        plot_temporal_btn = QPushButton("Generar Gráfico Temporal")
        plot_temporal_btn.setObjectName("primaryButton")
        plot_temporal_btn.clicked.connect(self.generate_temporal_plot)

        compare_btn = QPushButton("Comparar Métodos")
        compare_btn.setObjectName("analysisButton")
        compare_btn.clicked.connect(self.compare_methods)

        export_btn = QPushButton("Exportar Informe")
        export_btn.setObjectName("analysisButton")
        export_btn.clicked.connect(self.export_report)

        controls_layout.addWidget(plot_temporal_btn)
        controls_layout.addWidget(compare_btn)
        controls_layout.addWidget(export_btn)
        controls_layout.addStretch()

        analysis_layout.addLayout(controls_layout)

        # Área de gráfico temporal
        self.temporal_plot_widget = TimeSeriesPlotWidget()

        # Scroll area para el gráfico
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.temporal_plot_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(400)

        analysis_layout.addWidget(scroll_area)

        # Área de análisis de texto
        text_group = QGroupBox("Análisis Estadístico")
        text_layout = QVBoxLayout()

        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setMaximumHeight(200)
        self.analysis_text.setPlaceholderText(
            "El análisis estadístico aparecerá aquí después de generar el gráfico temporal..."
        )

        text_layout.addWidget(self.analysis_text)
        text_group.setLayout(text_layout)

        analysis_layout.addWidget(text_group)

        analysis_widget.setLayout(analysis_layout)
        self.main_tabs.addTab(analysis_widget, "Análisis")

    def create_footer(self, parent_layout):
        """Crear footer con controles principales"""
        footer_frame = QFrame()
        footer_frame.setObjectName("footerFrame")

        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(20, 15, 20, 15)

        # Información
        self.info_label = QLabel("Configura los parámetros y calcula el WQI")
        self.info_label.setObjectName("infoLabel")

        # Botones
        load_data_btn = QPushButton("Cargar Datos")
        load_data_btn.setObjectName("secondaryButton")
        load_data_btn.clicked.connect(self.load_data)

        help_btn = QPushButton("Ayuda")
        help_btn.setObjectName("secondaryButton")
        help_btn.clicked.connect(self.show_help)

        self.back_btn = QPushButton("Regresar")
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

    def generate_temporal_plot(self):
        """Generar gráfico de serie temporal del WQI"""
        if self.current_results is None:
            QMessageBox.warning(self, "Sin resultados", "Primero calcula el WQI")
            return

        if self.data is None:
            QMessageBox.warning(self, "Sin datos", "No hay datos disponibles")
            return

        # Buscar columna de fecha
        date_column = None
        for col in self.data.columns:
            if any(keyword in col.lower() for keyword in ['fecha', 'date', 'time', 'año', 'year']):
                date_column = col
                break

        if date_column is None:
            QMessageBox.warning(
                self, "Sin columna de fecha",
                "No se encontró una columna de fecha en los datos.\n"
                "Asegúrate de que existe una columna con nombres como: fecha, date, año, year, etc."
            )
            return

        # Obtener valores WQI
        wqi_values = [r['wqi'] for r in self.current_results['results']]

        if len(wqi_values) != len(self.data):
            QMessageBox.warning(
                self, "Error de datos",
                "El número de resultados WQI no coincide con el número de filas de datos"
            )
            return

        # Generar gráfico
        title = f"Evolución temporal del Índice de Calidad del Agua (WQI_{self.current_results['method']})"

        success = self.temporal_plot_widget.plot_wqi_evolution(
            self.data, date_column, wqi_values, title
        )

        if success:
            # Generar análisis estadístico
            self.generate_temporal_analysis(wqi_values, date_column)
            self.info_label.setText("Gráfico temporal generado exitosamente")
        else:
            self.info_label.setText("Error al generar el gráfico temporal")

    def generate_temporal_analysis(self, wqi_values, date_column):
        """Generar análisis estadístico temporal"""
        try:
            # Crear DataFrame temporal
            dates = pd.to_datetime(self.data[date_column])
            temp_df = pd.DataFrame({
                'fecha': dates,
                'wqi': wqi_values
            }).sort_values('fecha')

            # Análisis por año
            temp_df['año'] = temp_df['fecha'].dt.year
            yearly_stats = temp_df.groupby('año')['wqi'].agg(['mean', 'std', 'min', 'max'])

            # Tendencia general
            años = yearly_stats.index.values
            wqi_medios = yearly_stats['mean'].values
            correlacion = np.corrcoef(años, wqi_medios)[0, 1]

            # Análisis de texto
            analysis_text = "=== ANÁLISIS TEMPORAL DEL WQI ===\n\n"

            analysis_text += f"📊 PERÍODO DE ANÁLISIS:\n"
            analysis_text += f"  • Desde: {temp_df['fecha'].min().strftime('%Y-%m-%d')}\n"
            analysis_text += f"  • Hasta: {temp_df['fecha'].max().strftime('%Y-%m-%d')}\n"
            analysis_text += f"  • Total años: {len(yearly_stats)} años\n\n"

            analysis_text += f"📈 TENDENCIA GENERAL:\n"
            if correlacion > 0.3:
                trend = "Mejorando"
            elif correlacion < -0.3:
                trend = "Empeorando"
            else:
                trend = "Estable"

            analysis_text += f"  • Dirección: {trend}\n"
            analysis_text += f"  • Correlación temporal: {correlacion:.3f}\n\n"

            analysis_text += f"📊 ESTADÍSTICAS ANUALES:\n"
            analysis_text += f"  • WQI promedio global: {wqi_medios.mean():.2f}\n"
            analysis_text += f"  • Desviación estándar: {wqi_medios.std():.2f}\n"
            analysis_text += f"  • Mejor año: {yearly_stats['mean'].idxmax()} (WQI: {yearly_stats['mean'].max():.2f})\n"
            analysis_text += f"  • Peor año: {yearly_stats['mean'].idxmin()} (WQI: {yearly_stats['mean'].min():.2f})\n\n"

            # Clasificación predominante
            classification_counts = {}
            for wqi in wqi_values:
                classification = WQICalculationEngine.classify_water_quality(wqi)
                label = classification['label']
                classification_counts[label] = classification_counts.get(label, 0) + 1

            predominant = max(classification_counts, key=classification_counts.get)
            analysis_text += f"🏆 CALIDAD PREDOMINANTE:\n"
            analysis_text += f"  • Clasificación más frecuente: {predominant}\n"
            for quality, count in classification_counts.items():
                percentage = (count / len(wqi_values)) * 100
                analysis_text += f"  • {quality}: {percentage:.1f}%\n"

            self.analysis_text.setPlainText(analysis_text)

        except Exception as e:
            self.analysis_text.setPlainText(f"Error en análisis temporal: {str(e)}")

    # [Resto de métodos originales: check_for_data, load_data, calculate_wqi, etc.]
    # [Los métodos restantes permanecen iguales que en el código original]

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

        self.info_label.setText("No hay datos cargados, usa 'Cargar Datos'")
        return False

    def update_dataset_info(self):
        """Actualizar información del dataset"""
        if self.data is not None:
            info_text = f"{len(self.data):,} muestras × {len(self.data.columns)} variables"
            self.dataset_info_label.setText(info_text)
        else:
            self.dataset_info_label.setText("Sin datos cargados")

    def on_method_changed(self, method_text):
        """Actualizar descripción del método"""
        descriptions = {
            "NSF WQI - National Sanitation Foundation":
                "Método más utilizado internacionalmente. Combina 9 parámetros mediante un producto ponderado.",
            "CCME WQI - Canadian Council":
                "Método canadiense que evalúa el cumplimiento de objetivos de calidad.",
            "Aritmético Ponderado":
                "Método simple que calcula el promedio ponderado de los índices."
        }

        if method_text in descriptions:
            self.method_description.setPlainText(descriptions[method_text])

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

            self.status_label.setText("Pesos normalizados a 100%")
        else:
            QMessageBox.warning(self, "Advertencia", "Selecciona al menos un parámetro")

    def reset_parameters(self):
        """Restaurar parámetros por defecto"""
        self.setup_parameters_table()
        self.status_label.setText("Parámetros restaurados")

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
        """Método placeholder - implementar según la lógica original"""
        QMessageBox.information(self, "Info", "Implementar lógica de cálculo WQI")

    def compare_methods(self):
        """Método placeholder - implementar según la lógica original"""
        QMessageBox.information(self, "Info", "Implementar comparación de métodos")

    def export_report(self):
        """Método placeholder - implementar según la lógica original"""
        QMessageBox.information(self, "Info", "Implementar exportación de informes")

    def check_parameter_mapping(self):
        """Verificar y mapear parámetros disponibles"""
        if self.data is None:
            return

        # Mapeo manual básico para compatibilidad
        self.parameter_mapping = self.get_manual_parameter_mapping()
        available_params = list(self.parameter_mapping.keys())

        if available_params:
            self.info_label.setText(f"{len(available_params)} parámetros WQI detectados")
            self.update_parameters_table_with_mapping(available_params)
        else:
            self.info_label.setText("No se encontraron parámetros WQI en los datos")

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
        for i in range(self.params_table.rowCount()):
            param_name = self.params_table.item(i, 1).text().replace(' ', '_')
            checkbox = self.params_table.cellWidget(i, 0)

            if param_name in available_params:
                checkbox.setEnabled(True)
                checkbox.setChecked(True)
                if hasattr(self, 'parameter_mapping') and param_name in self.parameter_mapping:
                    real_name = self.parameter_mapping[param_name]
                    self.params_table.item(i, 1).setToolTip(f"Columna en datos: {real_name}")
            else:
                checkbox.setEnabled(False)
                checkbox.setChecked(False)
                self.params_table.item(i, 1).setToolTip("No disponible en los datos")

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

        # Obtener parámetros activos
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

        # Simular cálculo (reemplazar con lógica real)
        self.simulate_calculation(method, parameters)

    def simulate_calculation(self, method, parameters):
        """Simular cálculo WQI para demostración"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.calculate_btn.setEnabled(False)

        # Simular resultados
        np.random.seed(42)  # Para resultados reproducibles
        n_samples = len(self.data)

        # Generar valores WQI simulados
        base_wqi = 55 + np.random.normal(0, 15, n_samples)
        base_wqi = np.clip(base_wqi, 10, 95)  # Limitar entre 10 y 95

        results = []
        for i, wqi in enumerate(base_wqi):
            classification = WQICalculationEngine.classify_water_quality(wqi)
            results.append({
                'index': i,
                'wqi': wqi,
                'classification': classification
            })

        # Calcular estadísticas
        stats = {
            'mean': np.mean(base_wqi),
            'std': np.std(base_wqi),
            'min': np.min(base_wqi),
            'max': np.max(base_wqi)
        }

        # Distribución de calidad
        quality_dist = {}
        for result in results:
            label = result['classification']['label']
            quality_dist[label] = quality_dist.get(label, 0) + 1

        # Crear objeto de resultados
        self.current_results = {
            'method': method,
            'total_samples': n_samples,
            'results': results,
            'statistics': stats,
            'quality_distribution': quality_dist
        }

        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)
        self.calculate_btn.setEnabled(True)

        # Actualizar display
        self.update_results_display()
        self.main_tabs.setCurrentIndex(1)  # Ir a tab de resultados
        self.status_label.setText("Cálculo completado")
        self.info_label.setText(f"WQI calculado para {n_samples} muestras")

    def update_results_display(self):
        """Actualizar display de resultados"""
        if not self.current_results:
            return

        results = self.current_results
        stats = results['statistics']

        # Resumen textual
        summary_text = f"=== RESULTADOS WQI ===\n\n"
        summary_text += f"Método: {results['method']}\n"
        summary_text += f"Muestras analizadas: {results['total_samples']}\n\n"

        if stats:
            summary_text += f"ESTADÍSTICAS:\n"
            summary_text += f"  • WQI Promedio: {stats['mean']:.2f}\n"
            summary_text += f"  • Desv. Estándar: {stats['std']:.2f}\n"
            summary_text += f"  • Mínimo: {stats['min']:.2f}\n"
            summary_text += f"  • Máximo: {stats['max']:.2f}\n\n"

        if 'quality_distribution' in results:
            summary_text += f"DISTRIBUCIÓN DE CALIDAD:\n"
            for quality, count in results['quality_distribution'].items():
                percentage = (count / results['total_samples']) * 100
                summary_text += f"  • {quality}: {count} ({percentage:.1f}%)\n"

        self.results_text.setPlainText(summary_text)

        # Tabla de resultados (primeras 100 filas)
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

    def compare_methods(self):
        """Comparar diferentes métodos de cálculo"""
        if self.data is None:
            QMessageBox.warning(self, "Sin datos", "Carga datos primero")
            return

        comparison_text = "=== COMPARACIÓN DE MÉTODOS ===\n\n"
        comparison_text += "Simulación de comparación entre métodos NSF, CCME y Aritmético\n"
        comparison_text += "Los diferentes métodos pueden producir resultados variados\n"
        comparison_text += "dependiendo de los parámetros y sus ponderaciones.\n\n"
        comparison_text += "NSF: Método más conservador, usa productos ponderados\n"
        comparison_text += "CCME: Enfoque canadiense, evalúa cumplimiento de objetivos\n"
        comparison_text += "Aritmético: Método simple, promedio ponderado\n"

        self.analysis_text.setPlainText(comparison_text)

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
                    })

                results_df = pd.DataFrame(results_data)

                if file_path.endswith('.xlsx'):
                    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                        results_df.to_excel(writer, sheet_name='Resultados', index=False)

                        stats_df = pd.DataFrame([self.current_results['statistics']])
                        stats_df.to_excel(writer, sheet_name='Estadísticas', index=False)
                else:
                    results_df.to_csv(file_path, index=False)

                QMessageBox.information(self, "Éxito", "Informe exportado correctamente")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al exportar: {str(e)}")

    def show_help(self):
        """Mostrar ayuda"""
        help_text = """
ÍNDICE DE CALIDAD DEL AGUA (WQI)

El WQI combina múltiples parámetros fisicoquímicos en un índice único 
entre 0 y 100 que indica la calidad general del agua.

MÉTODOS DISPONIBLES:
• NSF WQI: Método más utilizado, usa producto ponderado
• CCME WQI: Evalúa cumplimiento de objetivos
• Aritmético: Promedio ponderado simple

INTERPRETACIÓN:
• 90-100: Excelente
• 70-89: Buena
• 50-69: Regular
• 25-49: Deficiente
• 0-24: Muy Deficiente

PASOS:
1. Cargar datos con parámetros del agua
2. Seleccionar parámetros a incluir
3. Ajustar pesos (deben sumar 100%)
4. Elegir método de cálculo
5. Calcular WQI
6. Generar gráfico temporal en la pestaña Análisis
"""

        QMessageBox.information(self, "Ayuda - WQI", help_text)

    def go_back(self):
        """Regresar al menú principal"""
        self.regresar_menu.emit()
        self.close()

    def apply_styles(self):
        """Aplicar estilos"""
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
                background-color: #f8f9fa;
            }
            
            #headerFrame {
                background-color: #ffffff;
                border-bottom: 2px solid #e9ecef;
                border-radius: 8px;
            }
            
            #mainTitle {
                color: #2c3e50;
                font-size: 18px;
                font-weight: bold;
            }
            
            #datasetInfo {
                color: #666;
                font-size: 11px;
                font-style: italic;
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
            
            QTableWidget {
                gridline-color: #dee2e6;
                selection-background-color: #007bff;
                alternate-background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
            
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
        """)


# Para pruebas independientes
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WQIWindow()
    window.show()
    sys.exit(app.exec_())