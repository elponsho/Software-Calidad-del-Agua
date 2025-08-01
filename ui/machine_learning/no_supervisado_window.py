import sys
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os

# Importaciones de PyQt5
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QApplication, QMessageBox, QTabWidget, QGridLayout, QGroupBox,
    QProgressBar, QTextEdit, QSplitter, QComboBox, QSpinBox,
    QCheckBox, QTableWidget, QTableWidgetItem, QDialog,
    QDialogButtonBox, QFormLayout, QFileDialog, QListWidget,
    QListWidgetItem, QScrollArea, QSlider, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, pyqtSlot
from PyQt5.QtGui import QFont, QColor

# Importar gestor de datos compartido
try:
    from .data_manager import get_data_manager, has_shared_data, get_shared_data
    DATA_MANAGER_AVAILABLE = True
    print("✅ DataManager importado correctamente")
except ImportError as e:
    DATA_MANAGER_AVAILABLE = False
    print(f"⚠️ DataManager no disponible: {e}")

    # Fallback para testing
    def get_data_manager():
        return None

    def has_shared_data():
        return False

    def get_shared_data():
        return None

# Importar funciones ML No Supervisado SIN SCIPY
ML_AVAILABLE = False
try:
    from .ml_functions_no_supervisado import (
        clustering_jerarquico_completo,
        kmeans_optimizado_completo,
        dbscan_optimizado,
        pca_completo_avanzado,
        analisis_exploratorio_completo,
        generar_datos_agua_realistas
    )

    # Importaciones de matplotlib
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    ML_AVAILABLE = True
    print("✅ Librerías ML No Supervisado cargadas correctamente")

except ImportError as e:
    ML_AVAILABLE = False
    print(f"❌ Error cargando ML No Supervisado: {e}")

# Importar sistema de temas
try:
    # from darkmode.ui_theme_manager import ThemedWidget, ThemeManager  # COMENTADA
    class ThemedWidget:
        def __init__(self):
            pass

        def apply_theme(self):
            pass

    class ThemeManager:
        def __init__(self):
            pass
except ImportError:
    class ThemedWidget(QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class ThemeManager:
        @staticmethod
        def toggle_theme():
            pass

        @staticmethod
        def is_dark_theme():
            return False


# ==================== WORKER THREAD PARA ML NO SUPERVISADO ====================

class MLNoSupervisadoWorker(QThread):
    """Worker thread para análisis ML No Supervisado"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, analysis_type: str, data: pd.DataFrame, **kwargs):
        super().__init__()
        self.analysis_type = analysis_type
        self.data = data.copy()
        self.kwargs = kwargs
        self._is_cancelled = False

    def run(self):
        """Ejecutar análisis"""
        try:
            if not ML_AVAILABLE:
                raise ImportError("Librerías de ML No Supervisado no disponibles")

            self.log.emit(f"🚀 Iniciando {self.analysis_type}")
            self.progress.emit(10)

            result = None

            # Mapear tipos de análisis a funciones
            if self.analysis_type == 'clustering_jerarquico':
                result = self._run_clustering_jerarquico()
            elif self.analysis_type == 'kmeans_optimizado':
                result = self._run_kmeans_optimizado()
            elif self.analysis_type == 'dbscan':
                result = self._run_dbscan()
            elif self.analysis_type == 'pca_avanzado':
                result = self._run_pca_avanzado()
            elif self.analysis_type == 'analisis_exploratorio':
                result = self._run_analisis_exploratorio()
            else:
                raise ValueError(f"Tipo de análisis desconocido: {self.analysis_type}")

            if self._is_cancelled:
                self.log.emit("❌ Análisis cancelado")
                return

            self.progress.emit(100)
            self.status.emit("✅ Análisis completado")
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))
            self.log.emit(f"❌ Error: {str(e)}")
            print(traceback.format_exc())

    def cancel(self):
        """Cancelar análisis"""
        self._is_cancelled = True

    def _run_clustering_jerarquico(self):
        """Ejecutar clustering jerárquico"""
        self.status.emit("Ejecutando clustering jerárquico...")
        self.progress.emit(30)

        # Parámetros corregidos para clustering jerárquico
        valid_kwargs = {
            'variables': self.kwargs.get('variables', []),
            'metodos': self.kwargs.get('metodos', ['ward']),
            'metricas': self.kwargs.get('metricas', ['euclidean']),
            'max_clusters': self.kwargs.get('max_clusters', 10),
            'escalado': self.kwargs.get('escalado', 'standard'),
            'verbose': self.kwargs.get('verbose', True)
        }

        return clustering_jerarquico_completo(self.data, **valid_kwargs)

    def _run_kmeans_optimizado(self):
        self.status.emit("Ejecutando K-Means optimizado...")
        self.progress.emit(30)

        # Parámetros corregidos para K-Means
        valid_kwargs = {
            'variables': self.kwargs.get('variables', []),
            'k_range': self.kwargs.get('k_range', range(2, 9)),
            'criterios_optimo': self.kwargs.get('criterios_optimo', ['silhouette']),
            'escalado': self.kwargs.get('escalado', 'standard'),
            'random_state': self.kwargs.get('random_state', 42),
            'verbose': self.kwargs.get('verbose', True)
        }

        return kmeans_optimizado_completo(self.data, **valid_kwargs)

    def _run_dbscan(self):
        self.status.emit("Ejecutando DBSCAN...")
        self.progress.emit(30)

        # Parámetros corregidos para DBSCAN
        valid_kwargs = {
            'variables': self.kwargs.get('variables', []),
            'optimizar_parametros': self.kwargs.get('optimizar_parametros', True),
            'escalado': self.kwargs.get('escalado', 'standard'),
            'verbose': self.kwargs.get('verbose', True)
        }

        # Añadir parámetros opcionales si están disponibles
        if 'contamination' in self.kwargs:
            valid_kwargs['contamination'] = self.kwargs['contamination']

        return dbscan_optimizado(self.data, **valid_kwargs)

    def _run_pca_avanzado(self):
        self.status.emit("Ejecutando PCA avanzado...")
        self.progress.emit(30)

        # Parámetros corregidos para PCA
        valid_kwargs = {
            'variables': self.kwargs.get('variables', []),
            'metodos': self.kwargs.get('metodos', ['linear']),
            'explicar_varianza_objetivo': self.kwargs.get('explicar_varianza_objetivo', 0.95),
            'escalado': self.kwargs.get('escalado', 'standard'),
            'random_state': self.kwargs.get('random_state', 42),
            'verbose': self.kwargs.get('verbose', True)
        }

        # Añadir parámetros específicos de kernel si están disponibles
        if 'max_components' in self.kwargs:
            valid_kwargs['max_components'] = self.kwargs['max_components']
        if 'kernel_type' in self.kwargs:
            valid_kwargs['kernel_type'] = self.kwargs['kernel_type']
        if 'gamma' in self.kwargs:
            valid_kwargs['gamma'] = self.kwargs['gamma']

        return pca_completo_avanzado(self.data, **valid_kwargs)

    def _run_analisis_exploratorio(self):
        self.status.emit("Ejecutando análisis exploratorio...")
        self.progress.emit(30)

        # Parámetros corregidos para análisis exploratorio
        valid_kwargs = {
            'variables': self.kwargs.get('variables', []),
            'escalado': self.kwargs.get('escalado', 'standard'),
            'handle_outliers': self.kwargs.get('handle_outliers', True),
            'verbose': self.kwargs.get('verbose', True)
        }

        # Añadir parámetros opcionales si están disponibles
        if 'outlier_method' in self.kwargs:
            valid_kwargs['outlier_method'] = self.kwargs['outlier_method']
        if 'random_state' in self.kwargs:
            valid_kwargs['random_state'] = self.kwargs['random_state']

        return analisis_exploratorio_completo(self.data, **valid_kwargs)


# ==================== WIDGET DE SELECCIÓN DE VARIABLES ====================

class VariableSelectionWidget(QWidget):
    """Widget para selección de variables para análisis no supervisado"""
    variables_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.data = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Título
        title = QLabel("📊 Selección de Variables")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title)

        # Controles de selección
        controls_layout = QHBoxLayout()

        self.select_all_btn = QPushButton("Todas")
        self.select_all_btn.setMinimumHeight(35)
        self.select_all_btn.clicked.connect(self._select_all_variables)
        controls_layout.addWidget(self.select_all_btn)

        self.select_none_btn = QPushButton("Ninguna")
        self.select_none_btn.setMinimumHeight(35)
        self.select_none_btn.clicked.connect(self._select_none_variables)
        controls_layout.addWidget(self.select_none_btn)

        self.auto_select_btn = QPushButton("🤖 Auto")
        self.auto_select_btn.setMinimumHeight(35)
        self.auto_select_btn.clicked.connect(self._auto_select_variables)
        controls_layout.addWidget(self.auto_select_btn)

        layout.addLayout(controls_layout)

        # Lista de variables con scroll
        self.variables_list = QListWidget()
        self.variables_list.setSelectionMode(QListWidget.MultiSelection)
        self.variables_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.variables_list.setMinimumHeight(150)
        layout.addWidget(self.variables_list)

        # Info de selección
        self.selection_info_label = QLabel("0 variables seleccionadas")
        self.selection_info_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.selection_info_label)

        self.setLayout(layout)

    def set_data(self, data: pd.DataFrame):
        """Establecer datos"""
        self.data = data
        self._update_variables_list()

    def clear_data(self):
        """Limpiar datos"""
        self.data = None
        self.variables_list.clear()
        self.selection_info_label.setText("0 variables disponibles")

    def _update_variables_list(self):
        """Actualizar lista de variables"""
        if self.data is None:
            return

        # Limpiar
        self.variables_list.clear()

        # Obtener columnas numéricas, excluyendo las no relevantes para análisis
        exclude_cols = ['Points', 'Sampling_date', 'Classification_6V', 'Classification_7V', 'Classification_9V']
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        for col in numeric_cols:
            col_data = self.data[col]
            missing_pct = (col_data.isnull().sum() / len(col_data)) * 100

            # Crear item con información
            item_text = f"{col} (missing: {missing_pct:.1f}%)"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, col)

            # Colorear según calidad
            if missing_pct > 50:
                item.setForeground(QColor('red'))
            elif missing_pct > 20:
                item.setForeground(QColor('orange'))

            self.variables_list.addItem(item)

            # Seleccionar automáticamente si tiene pocos missing
            if missing_pct < 30:
                item.setSelected(True)

        self._update_selection_info()

    def _select_all_variables(self):
        """Seleccionar todas las variables"""
        for i in range(self.variables_list.count()):
            self.variables_list.item(i).setSelected(True)

    def _select_none_variables(self):
        """Deseleccionar todas las variables"""
        for i in range(self.variables_list.count()):
            self.variables_list.item(i).setSelected(False)

    def _auto_select_variables(self):
        """Selección automática basada en calidad de datos"""
        if self.data is None:
            return

        # Deseleccionar primero
        self._select_none_variables()

        # Seleccionar variables con:
        # 1. Menos del 20% de valores faltantes
        # 2. Varianza suficiente
        selected_count = 0

        for i in range(self.variables_list.count()):
            item = self.variables_list.item(i)
            col_name = item.data(Qt.UserRole)

            col_data = self.data[col_name]
            missing_pct = (col_data.isnull().sum() / len(col_data)) * 100

            # Verificar varianza (evitar variables constantes)
            variance = col_data.var()

            if missing_pct < 20 and not np.isnan(variance) and variance > 0:
                item.setSelected(True)
                selected_count += 1

        if selected_count > 0:
            QMessageBox.information(
                self, "Selección Automática",
                f"Se seleccionaron {selected_count} variables con buena calidad de datos"
            )
        else:
            QMessageBox.warning(
                self, "Selección Automática",
                "No se encontraron variables que cumplan los criterios de calidad"
            )

    def _on_selection_changed(self):
        """Cuando cambia la selección"""
        self._update_selection_info()
        self.variables_changed.emit()

    def _update_selection_info(self):
        """Actualizar información de selección"""
        n_selected = len(self.get_selected_variables())
        n_total = self.variables_list.count()
        self.selection_info_label.setText(f"{n_selected} de {n_total} variables seleccionadas")

    def get_selected_variables(self) -> list:
        """Obtener variables seleccionadas"""
        variables = []
        for item in self.variables_list.selectedItems():
            variables.append(item.data(Qt.UserRole))
        return variables

    def is_valid_selection(self) -> bool:
        """Verificar si la selección es válida"""
        return len(self.get_selected_variables()) >= 2


# ==================== WIDGET DE CONFIGURACIÓN ====================

class ConfigurationWidget(QWidget):
    """Widget para configuración de análisis con scroll"""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        # Layout principal
        main_layout = QVBoxLayout()
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Crear scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setMinimumHeight(300)

        # Widget de contenido
        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setSpacing(15)

        # ===== CONFIGURACIÓN DE CLUSTERING =====
        clustering_group = QGroupBox("🎯 Configuración de Clustering")
        clustering_layout = QFormLayout()
        clustering_layout.setSpacing(10)

        # K-Means
        kmeans_label = QLabel("K-Means Optimizado:")
        kmeans_label.setStyleSheet("font-weight: bold; color: #34495e;")
        clustering_layout.addRow(kmeans_label)

        self.kmeans_k_min = QSpinBox()
        self.kmeans_k_min.setRange(2, 10)
        self.kmeans_k_min.setValue(2)
        self.kmeans_k_min.setMinimumHeight(30)
        clustering_layout.addRow("K mínimo:", self.kmeans_k_min)

        self.kmeans_k_max = QSpinBox()
        self.kmeans_k_max.setRange(3, 15)
        self.kmeans_k_max.setValue(8)
        self.kmeans_k_max.setMinimumHeight(30)
        clustering_layout.addRow("K máximo:", self.kmeans_k_max)

        # Separador
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFrameShadow(QFrame.Sunken)
        clustering_layout.addRow(separator1)

        # DBSCAN
        dbscan_label = QLabel("DBSCAN:")
        dbscan_label.setStyleSheet("font-weight: bold; color: #34495e;")
        clustering_layout.addRow(dbscan_label)

        self.dbscan_optimize = QCheckBox("Optimizar parámetros automáticamente")
        self.dbscan_optimize.setChecked(True)
        self.dbscan_optimize.setMinimumHeight(30)
        clustering_layout.addRow("", self.dbscan_optimize)

        # Separador
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        clustering_layout.addRow(separator2)

        # CLUSTERING JERÁRQUICO
        hierarchical_label = QLabel("Clustering Jerárquico:")
        hierarchical_label.setStyleSheet("font-weight: bold; color: #34495e;")
        clustering_layout.addRow(hierarchical_label)

        self.hierarchical_method = QComboBox()
        self.hierarchical_method.addItems(['ward', 'complete', 'average', 'single'])
        self.hierarchical_method.setCurrentText('ward')
        self.hierarchical_method.setMinimumHeight(30)
        clustering_layout.addRow("Método de enlace:", self.hierarchical_method)

        self.hierarchical_metric = QComboBox()
        self.hierarchical_metric.addItems(['euclidean', 'manhattan', 'cosine'])
        self.hierarchical_metric.setCurrentText('euclidean')
        self.hierarchical_metric.setMinimumHeight(30)
        clustering_layout.addRow("Métrica de distancia:", self.hierarchical_metric)

        self.hierarchical_max_clusters = QSpinBox()
        self.hierarchical_max_clusters.setRange(2, 20)
        self.hierarchical_max_clusters.setValue(10)
        self.hierarchical_max_clusters.setMinimumHeight(30)
        clustering_layout.addRow("Clusters máximos:", self.hierarchical_max_clusters)

        clustering_group.setLayout(clustering_layout)
        content_layout.addWidget(clustering_group)

        # ===== CONFIGURACIÓN DE PCA =====
        pca_group = QGroupBox("📊 Configuración de PCA")
        pca_layout = QFormLayout()
        pca_layout.setSpacing(10)

        self.pca_variance_threshold = QDoubleSpinBox()
        self.pca_variance_threshold.setRange(0.8, 0.99)
        self.pca_variance_threshold.setValue(0.95)
        self.pca_variance_threshold.setSingleStep(0.05)
        self.pca_variance_threshold.setDecimals(2)
        self.pca_variance_threshold.setMinimumHeight(30)
        pca_layout.addRow("Varianza objetivo:", self.pca_variance_threshold)

        self.pca_kernel_methods = QCheckBox("Incluir Kernel PCA (no lineal)")
        self.pca_kernel_methods.setChecked(False)
        self.pca_kernel_methods.setMinimumHeight(30)
        pca_layout.addRow("", self.pca_kernel_methods)

        self.pca_max_components = QSpinBox()
        self.pca_max_components.setRange(2, 50)
        self.pca_max_components.setValue(10)
        self.pca_max_components.setMinimumHeight(30)
        pca_layout.addRow("Componentes máximos:", self.pca_max_components)

        pca_group.setLayout(pca_layout)
        content_layout.addWidget(pca_group)

        # ===== PREPROCESAMIENTO =====
        preprocessing_group = QGroupBox("⚙️ Preprocesamiento")
        preprocessing_layout = QFormLayout()
        preprocessing_layout.setSpacing(10)

        self.scaling_method = QComboBox()
        self.scaling_method.addItems(['standard', 'robust', 'minmax', 'quantile', 'none'])
        self.scaling_method.setCurrentText('standard')
        self.scaling_method.setMinimumHeight(30)
        preprocessing_layout.addRow("Método de escalado:", self.scaling_method)

        self.handle_outliers = QCheckBox("Detectar y manejar outliers")
        self.handle_outliers.setChecked(True)
        self.handle_outliers.setMinimumHeight(30)
        preprocessing_layout.addRow("", self.handle_outliers)

        self.outlier_method = QComboBox()
        self.outlier_method.addItems(['isolation_forest', 'zscore', 'iqr'])
        self.outlier_method.setCurrentText('isolation_forest')
        self.outlier_method.setMinimumHeight(30)
        preprocessing_layout.addRow("Método detección:", self.outlier_method)

        preprocessing_group.setLayout(preprocessing_layout)
        content_layout.addWidget(preprocessing_group)

        # ===== CONFIGURACIÓN GENERAL =====
        general_group = QGroupBox("🔧 Configuración General")
        general_layout = QFormLayout()
        general_layout.setSpacing(10)

        self.random_state = QSpinBox()
        self.random_state.setRange(0, 9999)
        self.random_state.setValue(42)
        self.random_state.setMinimumHeight(30)
        general_layout.addRow("Semilla aleatoria:", self.random_state)

        self.verbose_output = QCheckBox("Salida detallada")
        self.verbose_output.setChecked(True)
        self.verbose_output.setMinimumHeight(30)
        general_layout.addRow("", self.verbose_output)

        general_group.setLayout(general_layout)
        content_layout.addWidget(general_group)

        # Espacio al final
        content_layout.addStretch()

        content_widget.setLayout(content_layout)
        scroll_area.setWidget(content_widget)

        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)

    def get_config(self) -> dict:
        """Obtener configuración actual"""
        return {
            # K-Means
            'kmeans_k_range': range(self.kmeans_k_min.value(), self.kmeans_k_max.value() + 1),

            # DBSCAN
            'dbscan_optimize': self.dbscan_optimize.isChecked(),

            # Clustering Jerárquico
            'hierarchical_method': self.hierarchical_method.currentText(),
            'hierarchical_metric': self.hierarchical_metric.currentText(),
            'hierarchical_max_clusters': self.hierarchical_max_clusters.value(),

            # PCA
            'pca_variance_threshold': self.pca_variance_threshold.value(),
            'pca_include_kernel': self.pca_kernel_methods.isChecked(),
            'pca_max_components': self.pca_max_components.value(),

            # Preprocesamiento
            'scaling_method': self.scaling_method.currentText(),
            'handle_outliers': self.handle_outliers.isChecked(),
            'outlier_method': self.outlier_method.currentText(),

            # General
            'random_state': self.random_state.value(),
            'verbose': self.verbose_output.isChecked()
        }


# ==================== WIDGET DE RESULTADOS CON VISUALIZACIONES ====================

class ResultsVisualizationWidget(QWidget):
    """Widget para visualización de resultados No Supervisado con gráficos completos"""

    def __init__(self):
        super().__init__()
        self.current_results = None
        self.current_data = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Tabs para diferentes vistas
        self.tabs = QTabWidget()

        # Tab de resumen
        self.summary_widget = self._create_summary_tab()
        self.tabs.addTab(self.summary_widget, "📋 Resumen")

        # Tab de métricas
        self.metrics_widget = self._create_metrics_tab()
        self.tabs.addTab(self.metrics_widget, "📊 Métricas")

        # Tab de visualizaciones
        if ML_AVAILABLE:
            self.viz_widget = self._create_viz_tab()
            self.tabs.addTab(self.viz_widget, "📈 Visualizaciones")

        # Tab de detalles
        self.details_widget = self._create_details_tab()
        self.tabs.addTab(self.details_widget, "🔍 Detalles")

        layout.addWidget(self.tabs)

        # Toolbar
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)

        self.setLayout(layout)

    def _create_summary_tab(self) -> QWidget:
        """Crear tab de resumen"""
        widget = QWidget()
        layout = QVBoxLayout()

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setText("No hay resultados para mostrar.\n\nSelecciona variables y ejecuta un análisis.")
        layout.addWidget(self.summary_text)

        widget.setLayout(layout)
        return widget

    def _create_metrics_tab(self) -> QWidget:
        """Crear tab de métricas"""
        widget = QWidget()
        layout = QVBoxLayout()

        self.metrics_table = QTableWidget()
        self.metrics_table.setAlternatingRowColors(True)
        layout.addWidget(self.metrics_table)

        widget.setLayout(layout)
        return widget

    def _create_viz_tab(self) -> QWidget:
        """Crear tab de visualizaciones"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Canvas para matplotlib
        self.figure = Figure(figsize=(12, 10))
        self.canvas = FigureCanvas(self.figure)

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        # Controles
        controls_layout = QHBoxLayout()

        self.viz_type_combo = QComboBox()
        self.viz_type_combo.addItems([
            "Vista General", "Clusters 2D", "PCA Biplot", "Correlaciones",
            "Distribuciones", "Outliers", "Dendrograma"
        ])
        self.viz_type_combo.currentTextChanged.connect(self._update_visualization)
        controls_layout.addWidget(QLabel("Visualización:"))
        controls_layout.addWidget(self.viz_type_combo)

        self.save_fig_btn = QPushButton("💾 Guardar")
        self.save_fig_btn.clicked.connect(self._save_figure)
        controls_layout.addWidget(self.save_fig_btn)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        widget.setLayout(layout)
        return widget

    def _create_details_tab(self) -> QWidget:
        """Crear tab de detalles"""
        widget = QWidget()
        layout = QVBoxLayout()

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setFont(QFont("Consolas", 9))
        self.details_text.setText("No hay detalles técnicos para mostrar.")
        layout.addWidget(self.details_text)

        widget.setLayout(layout)
        return widget

    def _create_toolbar(self) -> QWidget:
        """Crear toolbar"""
        toolbar = QWidget()
        layout = QHBoxLayout()

        self.export_results_btn = QPushButton("📄 Exportar Resultados")
        self.export_results_btn.setMinimumHeight(35)
        self.export_results_btn.clicked.connect(self._export_results)
        self.export_results_btn.setEnabled(False)
        layout.addWidget(self.export_results_btn)

        self.generate_report_btn = QPushButton("📊 Generar Reporte")
        self.generate_report_btn.setMinimumHeight(35)
        self.generate_report_btn.clicked.connect(self._generate_report)
        self.generate_report_btn.setEnabled(False)
        layout.addWidget(self.generate_report_btn)

        layout.addStretch()

        self.status_label = QLabel("Sin resultados")
        self.status_label.setStyleSheet("color: #666;")
        layout.addWidget(self.status_label)

        toolbar.setLayout(layout)
        return toolbar

    def update_results(self, results: dict, analysis_type: str):
        """Actualizar con nuevos resultados"""
        self.current_results = results

        if 'error' in results:
            self._show_error(results['error'])
            return

        # Actualizar cada componente
        self._update_summary(results, analysis_type)
        self._update_metrics(results)
        self._update_details(results)

        if ML_AVAILABLE:
            self._update_visualization()

        self.status_label.setText(f"✅ {analysis_type} completado")
        self.status_label.setStyleSheet("color: green;")

        # Habilitar botones
        self.export_results_btn.setEnabled(True)
        self.generate_report_btn.setEnabled(True)

    def set_current_data(self, data):
        """Establecer datos actuales para visualizaciones"""
        self.current_data = data

    def _update_summary(self, results: dict, analysis_type: str):
        """Actualizar resumen"""
        summary = f"📊 Resumen - {analysis_type.replace('_', ' ').title()}\n"
        summary += "=" * 50 + "\n\n"

        # Información general
        summary += f"🔍 Tipo de análisis: {results.get('tipo', 'N/A')}\n"
        summary += f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        if 'variables_utilizadas' in results:
            summary += f"📈 Variables analizadas: {len(results['variables_utilizadas'])}\n"
            summary += f"📝 Variables: {', '.join(results['variables_utilizadas'][:5])}"
            if len(results['variables_utilizadas']) > 5:
                summary += f" (y {len(results['variables_utilizadas']) - 5} más)"
            summary += "\n\n"

        # Resultados específicos por tipo
        if results.get('tipo') == 'kmeans_optimizado':
            k_optimo = results.get('recomendacion_k', 'N/A')
            summary += f"🎯 K óptimo recomendado: {k_optimo}\n"

            if k_optimo != 'N/A' and 'resultados_por_k' in results:
                if k_optimo in results['resultados_por_k']:
                    best_result = results['resultados_por_k'][k_optimo]
                    summary += f"📊 Silhouette Score: {best_result.get('silhouette_score', 'N/A'):.3f}\n"
                    summary += f"📊 Davies-Bouldin Score: {best_result.get('davies_bouldin_score', 'N/A'):.3f}\n"

        elif results.get('tipo') == 'clustering_jerarquico_completo':
            if 'mejor_configuracion' in results:
                mejor_config = results['mejor_configuracion']
                summary += f"🎯 Mejor configuración: {mejor_config}\n"

        elif results.get('tipo') == 'pca_completo_avanzado':
            if 'linear' in results.get('resultados_por_metodo', {}):
                linear_result = results['resultados_por_metodo']['linear']
                n_comp = linear_result.get('componentes_recomendados', 'N/A')
                summary += f"📊 Componentes recomendados: {n_comp}\n"

        elif results.get('tipo') == 'dbscan_optimizado':
            if 'mejor_configuracion' in results:
                config = results['mejor_configuracion']
                summary += f"🎯 Clusters encontrados: {config.get('n_clusters', 'N/A')}\n"
                summary += f"🔍 Outliers detectados: {config.get('n_noise', 'N/A')}\n"

        # Recomendaciones
        if 'recomendaciones' in results:
            summary += "\n💡 Recomendaciones:\n"
            for i, rec in enumerate(results['recomendaciones'][:3], 1):
                summary += f"{i}. {rec}\n"

        self.summary_text.setText(summary)

    def _update_metrics(self, results: dict):
        """Actualizar métricas"""
        metrics_data = []

        # Métricas generales
        if 'variables_utilizadas' in results:
            metrics_data.append(("Variables utilizadas", len(results['variables_utilizadas'])))

        # Métricas específicas por tipo
        tipo = results.get('tipo', '')

        if tipo == 'kmeans_optimizado':
            k_optimo = results.get('recomendacion_k')
            if k_optimo and 'resultados_por_k' in results:
                if k_optimo in results['resultados_por_k']:
                    best_result = results['resultados_por_k'][k_optimo]
                    metrics_data.extend([
                        ("K óptimo", k_optimo),
                        ("Silhouette Score", f"{best_result.get('silhouette_score', 0):.3f}"),
                        ("Davies-Bouldin Score", f"{best_result.get('davies_bouldin_score', 0):.3f}"),
                        ("Inercia", f"{best_result.get('inercia', 0):.2f}")
                    ])

        elif tipo == 'pca_completo_avanzado':
            if 'linear' in results.get('resultados_por_metodo', {}):
                linear_result = results['resultados_por_metodo']['linear']
                metrics_data.extend([
                    ("Componentes recomendados", linear_result.get('componentes_recomendados', 'N/A')),
                    ("Varianza PC1",
                     f"{linear_result['analisis']['varianza_explicada'][0] * 100:.1f}%" if linear_result.get(
                         'analisis') else 'N/A'),
                    ("Varianza PC2", f"{linear_result['analisis']['varianza_explicada'][1] * 100:.1f}%" if len(
                        linear_result.get('analisis', {}).get('varianza_explicada', [])) > 1 else 'N/A')
                ])

        elif tipo == 'dbscan_optimizado':
            if 'mejor_configuracion' in results:
                config = results['mejor_configuracion']
                metrics_data.extend([
                    ("Clusters", config.get('n_clusters', 'N/A')),
                    ("Outliers", config.get('n_noise', 'N/A')),
                    ("% Outliers", f"{config.get('noise_ratio', 0) * 100:.1f}%"),
                    ("Silhouette Score", f"{config.get('silhouette_score', 0):.3f}"),
                    ("Eps óptimo", f"{config.get('eps', 0):.3f}"),
                    ("Min samples", config.get('min_samples', 'N/A'))
                ])

        # Llenar tabla
        self.metrics_table.setRowCount(len(metrics_data))
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Métrica", "Valor"])

        for i, (metric, value) in enumerate(metrics_data):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(str(metric)))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(str(value)))

        self.metrics_table.resizeColumnsToContents()

    def _update_details(self, results: dict):
        """Actualizar detalles técnicos"""
        details = json.dumps(results, indent=2, default=str, ensure_ascii=False)
        # Limitar longitud para evitar sobrecarga
        if len(details) > 50000:
            details = details[:50000] + "\n\n... (Resultado truncado por longitud)"

        self.details_text.setText(details)

    def _update_visualization(self):
        """Actualizar visualización"""
        if not self.current_results or not ML_AVAILABLE:
            return

        viz_type = self.viz_type_combo.currentText()

        try:
            self.figure.clear()

            if viz_type == "Vista General":
                self._create_overview_plot()
            elif viz_type == "Clusters 2D":
                self._create_clusters_2d_plot()
            elif viz_type == "PCA Biplot":
                self._create_pca_biplot()
            elif viz_type == "Correlaciones":
                self._create_correlation_plot()
            elif viz_type == "Distribuciones":
                self._create_distributions_plot()
            elif viz_type == "Outliers":
                self._create_outliers_plot()
            elif viz_type == "Dendrograma":
                self._create_dendrogram_plot()

            self.canvas.draw()

        except Exception as e:
            print(f"Error en visualización: {e}")
            self._show_viz_error(str(e))

    def _create_overview_plot(self):
        """Crear gráfico de vista general"""
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Vista general en desarrollo', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title('Vista General')
        ax.axis('off')

    def _create_clusters_2d_plot(self):
        """Crear visualización 2D de clusters"""
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Visualización 2D de clusters en desarrollo', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title('Clusters 2D')
        ax.axis('off')

    def _create_pca_biplot(self):
        """Crear biplot de PCA"""
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, 'PCA Biplot en desarrollo', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title('PCA Biplot')
        ax.axis('off')

    def _create_correlation_plot(self):
        """Crear gráfico de correlaciones"""
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Gráfico de correlaciones en desarrollo', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title('Correlaciones')
        ax.axis('off')

    def _create_distributions_plot(self):
        """Crear gráfico de distribuciones"""
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Gráfico de distribuciones en desarrollo', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title('Distribuciones')
        ax.axis('off')

    def _create_outliers_plot(self):
        """Crear gráfico de outliers"""
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Gráfico de outliers en desarrollo', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title('Outliers')
        ax.axis('off')

    def _create_dendrogram_plot(self):
        """Crear dendrograma"""
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Dendrograma en desarrollo', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title('Dendrograma')
        ax.axis('off')

    def _show_viz_error(self, error_msg):
        """Mostrar error en visualización"""
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, f'Error en visualización:\n{error_msg}',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, bbox=dict(boxstyle='round', facecolor='mistyrose'))
        ax.set_title('Error en Visualización')
        ax.axis('off')

    def _save_figure(self):
        """Guardar figura"""
        if not hasattr(self, 'figure'):
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Guardar Gráfico",
            f"grafico_no_supervisado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "PNG (*.png);;PDF (*.pdf)"
        )

        if filepath:
            self.figure.savefig(filepath, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Éxito", "Gráfico guardado correctamente")

    def _export_results(self):
        """Exportar resultados"""
        if not self.current_results:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Exportar Resultados",
            f"resultados_no_supervisado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON (*.json)"
        )

        if filepath:
            try:
                # Preparar datos para exportación
                export_data = {}
                for key, value in self.current_results.items():
                    if isinstance(value, np.ndarray):
                        export_data[key] = value.tolist()
                    elif isinstance(value, pd.DataFrame):
                        export_data[key] = value.to_dict()
                    else:
                        export_data[key] = value

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

                QMessageBox.information(self, "Éxito", "Resultados exportados correctamente")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al exportar: {str(e)}")

    def _generate_report(self):
        """Generar reporte completo"""
        QMessageBox.information(
            self, "Generar Reporte",
            "La funcionalidad de generación de reportes automáticos estará disponible próximamente.\n\n"
            "Por ahora, puedes:\n"
            "- Exportar los resultados en JSON\n"
            "- Guardar los gráficos como imágenes\n"
            "- Copiar el contenido del resumen"
        )

    def _show_error(self, error_msg: str):
        """Mostrar error"""
        self.summary_text.setText(f"❌ Error: {error_msg}")
        self.metrics_table.setRowCount(0)
        self.details_text.setText(f"Error: {error_msg}")
        self.status_label.setText("❌ Error en análisis")
        self.status_label.setStyleSheet("color: red;")


# ==================== VENTANA PRINCIPAL ====================

class NoSupervisadoWindow(QWidget, ThemedWidget):
    """Ventana principal para ML No Supervisado"""

    def __init__(self):
        QWidget.__init__(self)
        ThemedWidget.__init__(self)

        print("🚀 NoSupervisadoWindow: Inicializando...")

        self.current_data = None
        self.current_worker = None
        self.analysis_history = []

        # Registrar como observador del DataManager
        if DATA_MANAGER_AVAILABLE:
            dm = get_data_manager()
            if dm is not None:
                dm.add_observer(self)
                print("✅ NoSupervisadoWindow: Registrada como observador del DataManager")
            else:
                print("⚠️ NoSupervisadoWindow: DataManager no disponible")
        else:
            print("⚠️ NoSupervisadoWindow: DataManager no importado")

        self.setup_ui()

        # Verificar datos al inicio
        self.check_data_availability()
        print("✅ NoSupervisadoWindow: Inicialización completada")

    def setup_ui(self):
        """Configurar interfaz de usuario"""
        self.setWindowTitle("🔍 Machine Learning No Supervisado")
        self.setMinimumSize(1400, 900)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Header
        header = self.create_header()
        main_layout.addWidget(header)

        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Contenido principal
        content_splitter = QSplitter(Qt.Horizontal)

        # Panel izquierdo con scroll
        left_panel = self.create_left_panel()
        content_splitter.addWidget(left_panel)

        # Panel derecho
        self.results_widget = ResultsVisualizationWidget()
        content_splitter.addWidget(self.results_widget)

        content_splitter.setSizes([450, 950])
        main_layout.addWidget(content_splitter)

        # Log
        log_widget = self.create_log_widget()
        main_layout.addWidget(log_widget)

        self.setLayout(main_layout)
        self.apply_styles()

    # ==================== PATRÓN OBSERVER ====================

    def update(self, event_type: str = ""):
        """Método llamado por el DataManager cuando los datos cambian"""
        print(f"🔔 NoSupervisadoWindow: Recibida notificación '{event_type}'")

        if event_type in ['data_changed', 'session_imported']:
            self.check_data_availability()
        elif event_type == 'data_cleared':
            self.current_data = None
            self.update_data_info()
            self.enable_analysis_buttons(False)
            self.log("🗑️ Datos limpiados del sistema")

    # ==================== GESTIÓN DE DATOS ====================

    def check_data_availability(self):
        """Verificar disponibilidad de datos"""
        if DATA_MANAGER_AVAILABLE:
            if has_shared_data():
                self.current_data = get_shared_data()
                print(f"✅ Datos cargados: {self.current_data.shape if self.current_data is not None else 'None'}")
                self.update_data_info()
                self.enable_analysis_buttons(True)
                self.log("✅ Datos cargados desde el sistema")

                # Pasar datos a widget de resultados para visualizaciones
                self.results_widget.set_current_data(self.current_data)
            else:
                print("⚠️ No hay datos disponibles en el DataManager")
                self.current_data = None
                self.update_data_info()
                self.enable_analysis_buttons(False)
                self.log("⚠️ No hay datos disponibles. Carga datos desde el módulo de Cargar Datos")
        else:
            print("❌ DataManager no disponible")
            self.current_data = None
            self.update_data_info()
            self.enable_analysis_buttons(False)
            self.log("❌ Sistema de datos no disponible")

    def update_data_info(self):
        """Actualizar información de datos"""
        if self.current_data is not None:
            n_rows, n_cols = self.current_data.shape
            numeric_cols = len(self.current_data.select_dtypes(include=[np.number]).columns)

            info = f"📊 Dataset: {n_rows:,} filas × {n_cols} columnas ({numeric_cols} numéricas)"
            self.data_info_label.setText(info)

            # Actualizar widget de selección de variables
            self.variable_selection.set_data(self.current_data)
        else:
            self.data_info_label.setText("❌ No hay datos cargados")
            self.variable_selection.clear_data()

    def enable_analysis_buttons(self, enabled: bool):
        """Habilitar/deshabilitar botones de análisis"""
        buttons = [
            self.kmeans_btn, self.hierarchical_btn, self.dbscan_btn,
            self.pca_btn, self.exploratory_btn
        ]
        for btn in buttons:
            btn.setEnabled(enabled)

    # ==================== CONFIGURACIÓN DE UI ====================

    def create_header(self) -> QWidget:
        """Crear header de la ventana"""
        header = QFrame()
        header.setFrameStyle(QFrame.Box)
        header.setMaximumHeight(80)

        layout = QHBoxLayout()

        # Información del título
        title_layout = QVBoxLayout()

        title = QLabel("🔍 Machine Learning No Supervisado")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #2c3e50;")
        title_layout.addWidget(title)

        self.data_info_label = QLabel("Verificando datos...")
        self.data_info_label.setStyleSheet("color: #666; font-size: 12px;")
        title_layout.addWidget(self.data_info_label)

        layout.addLayout(title_layout)
        layout.addStretch()

        # Botones de acción
        self.refresh_btn = QPushButton("🔄 Actualizar")
        self.refresh_btn.setMinimumHeight(35)
        self.refresh_btn.clicked.connect(self.check_data_availability)
        layout.addWidget(self.refresh_btn)

        self.demo_btn = QPushButton("🎲 Demo")
        self.demo_btn.setMinimumHeight(35)
        self.demo_btn.clicked.connect(self.load_demo_data)
        layout.addWidget(self.demo_btn)

        self.help_btn = QPushButton("❓ Ayuda")
        self.help_btn.setMinimumHeight(35)
        self.help_btn.clicked.connect(self.show_help)
        layout.addWidget(self.help_btn)

        header.setLayout(layout)
        return header

    def create_left_panel(self) -> QWidget:
        """Crear panel izquierdo con scroll mejorado"""
        # Widget principal
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Crear scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setMinimumWidth(400)

        # Widget de contenido del scroll
        content_widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 10)

        # Widget de selección de variables
        self.variable_selection = VariableSelectionWidget()
        self.variable_selection.variables_changed.connect(self.on_variables_changed)
        layout.addWidget(self.variable_selection)

        # Widget de configuración
        self.configuration = ConfigurationWidget()
        layout.addWidget(self.configuration)

        # Botones de análisis
        analysis_group = QGroupBox("🚀 Análisis Disponibles")
        analysis_layout = QVBoxLayout()
        analysis_layout.setSpacing(12)

        # Clustering
        clustering_label = QLabel("🎯 Algoritmos de Clustering")
        clustering_label.setStyleSheet("""
            font-weight: bold; 
            color: #2c3e50; 
            font-size: 13px;
            background-color: #ecf0f1;
            padding: 8px;
            border-radius: 4px;
        """)
        analysis_layout.addWidget(clustering_label)

        self.kmeans_btn = QPushButton("🔹 K-Means Optimizado")
        self.kmeans_btn.setMinimumHeight(40)
        self.kmeans_btn.setToolTip("Clustering con optimización automática del número de clusters")
        self.kmeans_btn.clicked.connect(lambda: self.run_analysis('kmeans_optimizado'))
        analysis_layout.addWidget(self.kmeans_btn)

        self.hierarchical_btn = QPushButton("🔸 Clustering Jerárquico")
        self.hierarchical_btn.setMinimumHeight(40)
        self.hierarchical_btn.setToolTip("Clustering basado en dendrogramas con múltiples métodos de enlace")
        self.hierarchical_btn.clicked.connect(lambda: self.run_analysis('clustering_jerarquico'))
        analysis_layout.addWidget(self.hierarchical_btn)

        self.dbscan_btn = QPushButton("🔺 DBSCAN")
        self.dbscan_btn.setMinimumHeight(40)
        self.dbscan_btn.setToolTip("Clustering basado en densidad con detección automática de outliers")
        self.dbscan_btn.clicked.connect(lambda: self.run_analysis('dbscan'))
        analysis_layout.addWidget(self.dbscan_btn)

        # Separador visual
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFrameShadow(QFrame.Sunken)
        separator1.setStyleSheet("color: #bdc3c7;")
        analysis_layout.addWidget(separator1)

        # Reducción dimensional
        pca_label = QLabel("📊 Reducción Dimensional")
        pca_label.setStyleSheet("""
            font-weight: bold; 
            color: #2c3e50; 
            font-size: 13px;
            background-color: #e8f5e8;
            padding: 8px;
            border-radius: 4px;
        """)
        analysis_layout.addWidget(pca_label)

        self.pca_btn = QPushButton("🔹 PCA Avanzado")
        self.pca_btn.setMinimumHeight(40)
        self.pca_btn.setToolTip("Análisis de Componentes Principales lineal y no lineal (Kernel PCA)")
        self.pca_btn.clicked.connect(lambda: self.run_analysis('pca_avanzado'))
        analysis_layout.addWidget(self.pca_btn)

        # Separador visual
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        separator2.setStyleSheet("color: #bdc3c7;")
        analysis_layout.addWidget(separator2)

        # Análisis exploratorio
        exp_label = QLabel("🔍 Análisis Exploratorio")
        exp_label.setStyleSheet("""
            font-weight: bold; 
            color: #2c3e50; 
            font-size: 13px;
            background-color: #fff2e8;
            padding: 8px;
            border-radius: 4px;
        """)
        analysis_layout.addWidget(exp_label)

        self.exploratory_btn = QPushButton("🔹 Análisis Completo")
        self.exploratory_btn.setMinimumHeight(40)
        self.exploratory_btn.setToolTip("Análisis exploratorio completo: correlaciones, distribuciones y outliers")
        self.exploratory_btn.clicked.connect(lambda: self.run_analysis('analisis_exploratorio'))
        analysis_layout.addWidget(self.exploratory_btn)

        # Espacio flexible
        analysis_layout.addStretch()

        # Separador visual
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.HLine)
        separator3.setFrameShadow(QFrame.Sunken)
        separator3.setStyleSheet("color: #bdc3c7;")
        analysis_layout.addWidget(separator3)

        # Botones de control
        control_label = QLabel("⚙️ Control de Análisis")
        control_label.setStyleSheet("""
            font-weight: bold; 
            color: #2c3e50; 
            font-size: 13px;
            background-color: #ffeaea;
            padding: 8px;
            border-radius: 4px;
        """)
        analysis_layout.addWidget(control_label)

        self.cancel_btn = QPushButton("❌ Cancelar Análisis")
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setVisible(False)
        self.cancel_btn.setToolTip("Cancelar el análisis actual")
        self.cancel_btn.clicked.connect(self.cancel_analysis)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        analysis_layout.addWidget(self.cancel_btn)

        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

        # Espacio al final
        layout.addStretch()

        content_widget.setLayout(layout)
        scroll_area.setWidget(content_widget)

        main_layout.addWidget(scroll_area)
        main_widget.setLayout(main_layout)

        return main_widget

    def create_log_widget(self) -> QWidget:
        """Crear widget de log"""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Box)
        widget.setMaximumHeight(120)

        layout = QVBoxLayout()

        # Header
        header_layout = QHBoxLayout()
        log_title = QLabel("📝 Log de Actividad")
        log_title.setStyleSheet("font-weight: bold; color: #2c3e50;")
        header_layout.addWidget(log_title)

        clear_btn = QPushButton("🗑️ Limpiar")
        clear_btn.setMinimumHeight(30)
        clear_btn.clicked.connect(self.clear_log)
        header_layout.addWidget(clear_btn)

        layout.addLayout(header_layout)

        # Área de texto
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setMaximumHeight(80)
        layout.addWidget(self.log_text)

        widget.setLayout(layout)
        return widget

    # ==================== CARGA DE DATOS DEMO ====================

    def load_demo_data(self):
        """Cargar datos de demostración"""
        if not ML_AVAILABLE:
            QMessageBox.warning(
                self, "Error",
                "Las librerías ML no están disponibles para generar datos demo"
            )
            return

        try:
            self.log("🎲 Generando datos de demostración...")

            # Generar datos usando la función del módulo ML
            demo_data = generar_datos_agua_realistas(n_muestras=200, incluir_outliers=True)

            self.current_data = demo_data
            self.update_data_info()
            self.enable_analysis_buttons(True)

            # Pasar datos a widget de resultados
            self.results_widget.set_current_data(self.current_data)

            self.log("✅ Datos de demostración generados exitosamente")
            QMessageBox.information(
                self, "Datos Demo",
                f"Se generaron {len(demo_data)} muestras con {demo_data.shape[1]} variables\n\n"
                "Los datos incluyen parámetros de calidad del agua realistas con:\n"
                "• Múltiples estaciones de monitoreo\n"
                "• Correlaciones naturales entre variables\n"
                "• Outliers para pruebas de robustez"
            )

        except Exception as e:
            self.log(f"❌ Error generando datos demo: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error generando datos demo:\n{str(e)}")

    # ==================== EJECUCIÓN DE ANÁLISIS ====================

    def run_analysis(self, analysis_type: str):
        """Ejecutar análisis específico"""
        if not self.validate_selection():
            return

        if not ML_AVAILABLE:
            QMessageBox.critical(
                self, "Error",
                "Las librerías de Machine Learning no están disponibles.\n"
                "Verifica que scikit-learn, matplotlib estén instalados."
            )
            return

        # Obtener configuración
        variables = self.variable_selection.get_selected_variables()
        config = self.configuration.get_config()

        # Configurar kwargs base comunes
        base_kwargs = {
            'variables': variables,
            'escalado': config['scaling_method'],
            'verbose': config['verbose']
        }

        # Configurar kwargs específicos según el tipo de análisis
        if analysis_type == 'kmeans_optimizado':
            kwargs = {
                **base_kwargs,
                'k_range': config['kmeans_k_range'],
                'criterios_optimo': ['silhouette', 'elbow', 'gap'],
                'random_state': config['random_state']
            }
        elif analysis_type == 'clustering_jerarquico':
            kwargs = {
                **base_kwargs,
                'metodos': [config['hierarchical_method']],
                'metricas': [config['hierarchical_metric']],
                'max_clusters': config['hierarchical_max_clusters']
            }
        elif analysis_type == 'dbscan':
            kwargs = {
                **base_kwargs,
                'optimizar_parametros': config['dbscan_optimize']
            }

        elif analysis_type == 'pca_avanzado':
            metodos = ['linear']
            if config['pca_include_kernel']:
                metodos.append('kernel')
            kwargs = {
                **base_kwargs,
                'metodos': metodos,
                'explicar_varianza_objetivo': config['pca_variance_threshold'],
                'random_state': config['random_state']
            }
            # Añadir parámetros de kernel PCA si está habilitado
            if config['pca_include_kernel']:
                kwargs.update({
                    'max_components': config['pca_max_components']
                })

        elif analysis_type == 'analisis_exploratorio':
            kwargs = {
                **base_kwargs,
                'handle_outliers': config['handle_outliers']
            }
            # Añadir método de outliers si está habilitado
            if config['handle_outliers']:
                kwargs['outlier_method'] = config['outlier_method']
                kwargs['random_state'] = config['random_state']

        # Mostrar progreso
        self.show_progress(True)
        self.log(f"🚀 Iniciando análisis: {analysis_type}")
        self.log(f"📊 Variables seleccionadas: {len(variables)}")
        self.log(f"⚙️ Configuración: {config['scaling_method']} scaling")

        # Crear worker
        self.current_worker = MLNoSupervisadoWorker(analysis_type, self.current_data, **kwargs)

        # Conectar señales
        self.current_worker.progress.connect(self.progress_bar.setValue)
        self.current_worker.status.connect(self.log)
        self.current_worker.finished.connect(self.on_analysis_finished)
        self.current_worker.error.connect(self.on_analysis_error)
        self.current_worker.log.connect(self.log)

        # Iniciar
        self.current_worker.start()

    def cancel_analysis(self):
        """Cancelar análisis en curso"""
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.cancel()
            self.current_worker.terminate()
            self.log("❌ Análisis cancelado por el usuario")
            self.show_progress(False)

    def validate_selection(self) -> bool:
        """Validar selección de variables"""
        if self.current_data is None:
            QMessageBox.warning(self, "Sin Datos", "No hay datos cargados")
            return False

        if not self.variable_selection.is_valid_selection():
            QMessageBox.warning(
                self, "Selección Inválida",
                "Selecciona al menos 2 variables para el análisis"
            )
            return False

        return True

    # ==================== CALLBACKS ====================

    @pyqtSlot(dict)
    def on_analysis_finished(self, results: dict):
        """Cuando termina el análisis"""
        self.show_progress(False)

        # Guardar en historial
        analysis_entry = {
            'timestamp': datetime.now(),
            'type': self.current_worker.analysis_type if self.current_worker else 'unknown',
            'results': results,
            'variables': self.variable_selection.get_selected_variables(),
            'config': self.configuration.get_config()
        }
        self.analysis_history.append(analysis_entry)

        # Actualizar resultados
        self.results_widget.update_results(
            results,
            self.current_worker.analysis_type if self.current_worker else 'unknown'
        )

        self.log("✅ Análisis completado exitosamente")

        # Mostrar resumen rápido en el log
        if 'tipo' in results:
            if results['tipo'] == 'kmeans_optimizado' and 'recomendacion_k' in results:
                self.log(f"🎯 K óptimo recomendado: {results['recomendacion_k']}")
            elif results['tipo'] == 'dbscan_optimizado' and 'mejor_configuracion' in results:
                config = results['mejor_configuracion']
                self.log(f"🎯 DBSCAN: {config.get('n_clusters', 0)} clusters, {config.get('n_noise', 0)} outliers")
            elif results['tipo'] == 'pca_completo_avanzado':
                if 'linear' in results.get('resultados_por_metodo', {}):
                    linear = results['resultados_por_metodo']['linear']
                    self.log(f"📊 PCA: {linear.get('componentes_recomendados', 'N/A')} componentes recomendados")

    @pyqtSlot(str)
    def on_analysis_error(self, error_msg: str):
        """Cuando ocurre un error"""
        self.show_progress(False)
        self.log(f"❌ Error: {error_msg}")
        QMessageBox.critical(self, "Error en Análisis",
                            f"Error durante el análisis:\n\n{error_msg}\n\n"
                            "Revisa los datos y la configuración.")

    def on_variables_changed(self):
        """Cuando cambian las variables seleccionadas"""
        n_selected = len(self.variable_selection.get_selected_variables())
        if n_selected > 0:
            self.log(f"📊 {n_selected} variables seleccionadas")

    # ==================== UTILIDADES ====================

    def show_progress(self, show: bool):
        """Mostrar/ocultar progreso"""
        self.progress_bar.setVisible(show)
        self.cancel_btn.setVisible(show)
        if show:
            self.progress_bar.setValue(0)

        # Deshabilitar botones durante análisis
        self.enable_analysis_buttons(not show)

    def log(self, message: str):
        """Añadir mensaje al log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

        # Auto-scroll
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_log(self):
        """Limpiar log"""
        self.log_text.clear()
        self.log("📝 Log limpiado")

    def show_help(self):
        """Mostrar ayuda"""
        help_dialog = QDialog(self)
        help_dialog.setWindowTitle("Ayuda - Machine Learning No Supervisado")
        help_dialog.setModal(True)
        help_dialog.resize(800, 700)

        layout = QVBoxLayout()

        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
        <h2>🔍 Machine Learning No Supervisado</h2>

        <h3>📌 ¿Qué es el ML No Supervisado?</h3>
        <p>El aprendizaje no supervisado busca patrones ocultos en datos sin etiquetas predefinidas. 
        Es ideal para explorar datos y descubrir estructuras subyacentes.</p>

        <h3>🎯 Técnicas de Clustering:</h3>
        <ul>
        <li><b>K-Means Optimizado:</b> Agrupa datos en K clusters esféricos con selección automática del K óptimo</li>
        <li><b>Clustering Jerárquico:</b> Crea dendrogramas con jerarquías de grupos, ideal para estructuras anidadas</li>
        <li><b>DBSCAN:</b> Detecta clusters de densidad variable y outliers automáticamente</li>
        </ul>

        <h3>📊 Reducción Dimensional:</h3>
        <ul>
        <li><b>PCA Linear:</b> Proyecta datos en componentes principales lineales</li>
        <li><b>Kernel PCA:</b> PCA no lineal para patrones complejos con kernels RBF, polinomial, etc.</li>
        </ul>

        <h3>🔍 Análisis Exploratorio:</h3>
        <ul>
        <li><b>Correlaciones:</b> Detecta relaciones lineales y no lineales entre variables</li>
        <li><b>Distribuciones:</b> Analiza patrones de distribución y normalidad</li>
        <li><b>Outliers:</b> Identifica valores atípicos con múltiples métodos</li>
        </ul>

        <h3>🚀 Cómo usar (Flujo recomendado):</h3>
        <ol>
        <li><b>Cargar datos:</b> Desde el módulo "Cargar Datos" o usa el botón Demo</li>
        <li><b>Seleccionar variables:</b> Usa el botón "Auto" para selección inteligente</li>
        <li><b>Configurar análisis:</b> Ajusta parámetros según tus necesidades</li>
        <li><b>Ejecutar análisis:</b> Comienza con "Análisis Exploratorio" para entender los datos</li>
        <li><b>Revisar resultados:</b> Usa las pestañas de visualización y métricas</li>
        <li><b>Refinar análisis:</b> Ajusta parámetros y repite según resultados</li>
        </ol>

        <h3>💡 Consejos y Mejores Prácticas:</h3>
        <ul>
        <li><b>Orden recomendado:</b> Análisis Exploratorio → PCA → K-Means → DBSCAN → Jerárquico</li>
        <li><b>Selección de variables:</b> Evita variables con >50% de valores faltantes</li>
        <li><b>Escalado:</b> Standard es recomendado para la mayoría de casos</li>
        <li><b>K-Means:</b> Prueba rangos de 2-8 clusters inicialmente</li>
        <li><b>PCA:</b> 95% de varianza explicada es un buen punto de partida</li>
        <li><b>DBSCAN:</b> Ideal cuando no conoces el número de clusters</li>
        <li><b>Outliers:</b> Isolation Forest es robusto para datos multidimensionales</li>
        </ul>
        """)
        layout.addWidget(help_text)

        # Botones
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(help_dialog.accept)
        layout.addWidget(buttons)

        help_dialog.setLayout(layout)
        help_dialog.exec_()

    def apply_styles(self):
        """Aplicar estilos personalizados"""
        style = """
        /* Estilos generales */
        QWidget {
            font-family: 'Segoe UI', Arial, sans-serif;
        }

        /* GroupBox styling */
        QGroupBox {
            font-weight: bold;
            border: 2px solid #bdc3c7;
            border-radius: 8px;
            margin-top: 1ex;
            padding-top: 10px;
            background-color: #fafafa;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 8px 0 8px;
            background-color: white;
            border-radius: 4px;
        }

        /* Botones principales */
        QPushButton {
            background-color: #3498db;
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 12px;
        }

        QPushButton:hover {
            background-color: #2980b9;
        }

        QPushButton:pressed {
            background-color: #21618c;
        }

        QPushButton:disabled {
            background-color: #bdc3c7;
            color: #7f8c8d;
        }

        /* Botones específicos de análisis */
        QPushButton[text*="K-Means"] {
            background-color: #e74c3c;
        }
        QPushButton[text*="K-Means"]:hover {
            background-color: #c0392b;
        }

        QPushButton[text*="Jerárquico"] {
            background-color: #f39c12;
        }
        QPushButton[text*="Jerárquico"]:hover {
            background-color: #e67e22;
        }

        QPushButton[text*="DBSCAN"] {
            background-color: #9b59b6;
        }
        QPushButton[text*="DBSCAN"]:hover {
            background-color: #8e44ad;
        }

        QPushButton[text*="PCA"] {
            background-color: #27ae60;
        }
        QPushButton[text*="PCA"]:hover {
            background-color: #229954;
        }

        QPushButton[text*="Exploratorio"] {
            background-color: #17a2b8;
        }
        QPushButton[text*="Exploratorio"]:hover {
            background-color: #138496;
        }

        /* Progress bar */
        QProgressBar {
            border: 2px solid #bdc3c7;
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
            background-color: white;
        }

        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #3498db, stop:1 #2980b9);
            border-radius: 6px;
        }

        /* Tabs */
        QTabWidget::pane {
            border: 2px solid #bdc3c7;
            border-radius: 8px;
            background-color: white;
        }

        QTabBar::tab {
            background: #ecf0f1;
            border: 1px solid #bdc3c7;
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
        }

        QTabBar::tab:selected {
            background: #3498db;
            color: white;
            font-weight: bold;
        }

        QTabBar::tab:hover {
            background: #d5dbdb;
        }
        """

        self.setStyleSheet(style)

    def closeEvent(self, event):
        """Manejar el cierre de la ventana"""
        # Cancelar análisis en curso si existe
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.cancel()
            self.current_worker.terminate()
            self.current_worker.wait(3000)  # Esperar máximo 3 segundos

        # Desconectar del DataManager si está disponible
        if DATA_MANAGER_AVAILABLE:
            dm = get_data_manager()
            if dm is not None:
                try:
                    dm.remove_observer(self)
                    print("✅ NoSupervisadoWindow: Desconectada del DataManager")
                except:
                    pass

        event.accept()


# ==================== FUNCIÓN PRINCIPAL PARA TESTING ====================

def main():
    """Función principal para testing independiente"""
    app = QApplication(sys.argv)

    # Configurar estilo de la aplicación
    app.setStyle('Fusion')

    # Crear ventana principal
    window = NoSupervisadoWindow()
    window.show()

    # Ejecutar aplicación
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()