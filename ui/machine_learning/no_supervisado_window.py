
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

# Importar funciones ML No Supervisado
ML_AVAILABLE = False
try:
    from .ml_functions_no_supervisado import (
        clustering_jerarquico_completo,
        kmeans_optimizado_completo,
        dbscan_optimizado,
        pca_completo_avanzado,
        analisis_exploratorio_completo,
        generar_datos_agua_realistas, _crear_visualizacion_pca_puntos_muestreo,
        _crear_visualizacion_exploratorio_puntos_muestreo
    )

    # Importaciones de matplotlib
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import seaborn as sns

    # Configurar estilo de seaborn
    sns.set_style("whitegrid")

    # Importar scipy para dendrogramas
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy import stats

    ML_AVAILABLE = True
    print("✅ Librerías ML No Supervisado cargadas correctamente")

except ImportError as e:
    ML_AVAILABLE = False
    print(f"❌ Error cargando ML No Supervisado: {e}")

# Importar sistema de temas
try:
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

        valid_kwargs = {
            'variables': self.kwargs.get('variables', []),
            'metodos': self.kwargs.get('metodos', ['ward']),
            'metricas': self.kwargs.get('metricas', ['euclidean']),
            'max_clusters': self.kwargs.get('max_clusters', 10),
            'escalado': self.kwargs.get('escalado', 'standard'),
            'verbose': self.kwargs.get('verbose', True)
        }

        resultado = clustering_jerarquico_completo(self.data, **valid_kwargs)

        # Guardar datos originales para visualización
        resultado['datos_originales'] = self.data[valid_kwargs['variables']].copy()

        return resultado

    def _run_kmeans_optimizado(self):
        self.status.emit("Ejecutando K-Means optimizado...")
        self.progress.emit(30)

        valid_kwargs = {
            'variables': self.kwargs.get('variables', []),
            'k_range': self.kwargs.get('k_range', range(2, 9)),
            'criterios_optimo': self.kwargs.get('criterios_optimo', ['silhouette']),
            'escalado': self.kwargs.get('escalado', 'standard'),
            'random_state': self.kwargs.get('random_state', 42),
            'verbose': self.kwargs.get('verbose', True)
        }

        resultado = kmeans_optimizado_completo(self.data, **valid_kwargs)

        # Guardar datos originales para visualización
        resultado['datos_originales'] = self.data[valid_kwargs['variables']].copy()

        return resultado

    def _run_dbscan(self):
        self.status.emit("Ejecutando DBSCAN...")
        self.progress.emit(30)

        valid_kwargs = {
            'variables': self.kwargs.get('variables', []),
            'optimizar_parametros': self.kwargs.get('optimizar_parametros', True),
            'escalado': self.kwargs.get('escalado', 'standard'),
            'verbose': self.kwargs.get('verbose', True)
        }

        if 'contamination' in self.kwargs:
            valid_kwargs['contamination'] = self.kwargs['contamination']

        resultado = dbscan_optimizado(self.data, **valid_kwargs)

        # Guardar datos originales para visualización
        resultado['datos_originales'] = self.data[valid_kwargs['variables']].copy()

        return resultado

    def _run_pca_avanzado(self):
        self.status.emit("Ejecutando PCA avanzado...")
        self.progress.emit(30)

        valid_kwargs = {
            'variables': self.kwargs.get('variables', []),
            'metodos': self.kwargs.get('metodos', ['linear']),
            'explicar_varianza_objetivo': self.kwargs.get('explicar_varianza_objetivo', 0.95),
            'escalado': self.kwargs.get('escalado', 'standard'),
            'random_state': self.kwargs.get('random_state', 42),
            'verbose': self.kwargs.get('verbose', True)
        }

        if 'max_components' in self.kwargs:
            valid_kwargs['max_components'] = self.kwargs['max_components']
        if 'kernel_type' in self.kwargs:
            valid_kwargs['kernel_type'] = self.kwargs['kernel_type']
        if 'gamma' in self.kwargs:
            valid_kwargs['gamma'] = self.kwargs['gamma']

        resultado = pca_completo_avanzado(self.data, **valid_kwargs)

        # Guardar datos originales para visualización
        resultado['datos_originales'] = self.data[valid_kwargs['variables']].copy()

        return resultado

    def _run_analisis_exploratorio(self):
        self.status.emit("Ejecutando análisis exploratorio...")
        self.progress.emit(30)

        valid_kwargs = {
            'variables': self.kwargs.get('variables', []),
            'escalado': self.kwargs.get('escalado', 'standard'),
            'handle_outliers': self.kwargs.get('handle_outliers', True),
            'verbose': self.kwargs.get('verbose', True)
        }

        if 'outlier_method' in self.kwargs:
            valid_kwargs['outlier_method'] = self.kwargs['outlier_method']
        if 'random_state' in self.kwargs:
            valid_kwargs['random_state'] = self.kwargs['random_state']

        resultado = analisis_exploratorio_completo(self.data, **valid_kwargs)

        # Guardar datos originales para visualización
        resultado['datos_originales'] = self.data[valid_kwargs['variables']].copy()

        return resultado


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


# ==================== WIDGET DE CONFIGURACIÓN MEJORADO ====================

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
        self.hierarchical_metric.addItems(['euclidean', 'manhattan', 'cosine', 'chebyshev'])
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

        # ===== PREPROCESAMIENTO MEJORADO =====
        preprocessing_group = QGroupBox("⚙️ Preprocesamiento Avanzado")
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
        self.outlier_method.addItems(['isolation_forest', 'zscore', 'iqr', 'local_outlier'])
        self.outlier_method.setCurrentText('isolation_forest')
        self.outlier_method.setMinimumHeight(30)
        preprocessing_layout.addRow("Método detección:", self.outlier_method)

        self.outlier_contamination = QDoubleSpinBox()
        self.outlier_contamination.setRange(0.01, 0.5)
        self.outlier_contamination.setValue(0.1)
        self.outlier_contamination.setSingleStep(0.01)
        self.outlier_contamination.setDecimals(2)
        self.outlier_contamination.setMinimumHeight(30)
        preprocessing_layout.addRow("Contaminación:", self.outlier_contamination)

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
            'outlier_contamination': self.outlier_contamination.value(),

            # General
            'random_state': self.random_state.value(),
            'verbose': self.verbose_output.isChecked()
        }


# ==================== WIDGET DE RESULTADOS CON VISUALIZACIONES IMPLEMENTADAS ====================

class ResultsVisualizationWidget(QWidget):
    """Widget para visualización de resultados No Supervisado"""

    def __init__(self):
        super().__init__()
        self.current_results = None
        self.current_analysis_type = None
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

    def _save_figure(self):
        """Guardar figura actual en PNG."""
        if not hasattr(self, 'figure'):
            QMessageBox.warning(self, "Sin figura", "No hay figura para guardar.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar Figura",
            "",
            "Imagen PNG (*.png);;Imagen JPEG (*.jpg)"
        )
        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Éxito", f"Figura guardada en:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo guardar la figura:\n{e}")

    def _generate_report(self):
        """Generar reporte de los resultados actuales."""
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        import pandas as pd

        if not hasattr(self, 'current_results') or not self.current_results:
            QMessageBox.warning(self, "Sin datos", "No hay resultados para generar el reporte.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar Reporte",
            "",
            "Archivo de texto (*.txt);;Archivo CSV (*.csv)"
        )
        if not file_path:
            return

        try:
            mejor_config = self.current_results.get('mejor_configuracion', {})
            lines = []
            lines.append("=== Reporte de Resultados No Supervisado ===\n")
            if mejor_config:
                lines.append("Parámetros óptimos:\n")
                for key, val in mejor_config.items():
                    lines.append(f"- {key}: {val}\n")
            else:
                lines.append("No se encontraron parámetros óptimos.\n")

            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

            QMessageBox.information(self, "Éxito", f"Reporte guardado en:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo generar el reporte:\n{e}")

<<<<<<< HEAD
    def _export_results(self):
        """Exportar resultados actuales a CSV."""
        if not hasattr(self, 'current_results') or 'datos_originales' not in self.current_results:
            QMessageBox.warning(self, "Sin datos", "No hay resultados para exportar.")
            return

=======
    def _plot_dbscan_clusters_fixed(self, ax, datos, labels):
        """Graficar clusters DBSCAN con PCA - VERSIÓN CORREGIDA"""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            # Seleccionar columnas numéricas válidas
            numeric_cols = datos.select_dtypes(include=[np.number]).columns
            exclude_cols = ['Points', 'WQI_IDEAM_6V', 'WQI_IDEAM_7V', 'WQI_NSF_9V']
            valid_cols = [col for col in numeric_cols if col not in exclude_cols]

            if len(valid_cols) < 2:
                ax.text(0.5, 0.5, 'Insuficientes variables numéricas',
                        ha='center', va='center', transform=ax.transAxes)
                return

            # Preparar datos para PCA
            datos_numeric = datos[valid_cols].dropna()

            # Ajustar labels al tamaño de datos limpios
            if len(datos_numeric) != len(labels):
                valid_indices = datos_numeric.index
                original_indices = datos.index
                mask = np.isin(original_indices, valid_indices)
                labels_clean = labels[mask]
            else:
                labels_clean = labels

            if len(datos_numeric) != len(labels_clean):
                ax.text(0.5, 0.5, 'Incompatibilidad en dimensiones de datos',
                        ha='center', va='center', transform=ax.transAxes)
                return

            # Aplicar PCA para reducir a 2D
            scaler = StandardScaler()
            datos_scaled = scaler.fit_transform(datos_numeric)

            pca = PCA(n_components=2)
            datos_2d = pca.fit_transform(datos_scaled)

            # Colores por cluster
            unique_labels = set(labels_clean)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Outliers en negro
                    col = 'black'
                    marker = 'x'
                    label = 'Outliers'
                    s = 100
                    alpha = 0.8
                else:
                    marker = 'o'
                    label = f'Cluster {k}'
                    s = 60
                    alpha = 0.7

                class_member_mask = (labels_clean == k)
                xy = datos_2d[class_member_mask]

                if len(xy) > 0:  # Solo graficar si hay puntos
                    ax.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker,
                               s=s, label=label, alpha=alpha, edgecolors='black', linewidth=0.5)

            var_exp = pca.explained_variance_ratio_
            ax.set_xlabel(f'PC1 ({var_exp[0] * 100:.1f}%)')
            ax.set_ylabel(f'PC2 ({var_exp[1] * 100:.1f}%)')
            ax.set_title('Clusters DBSCAN (PCA)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}',
                    ha='center', va='center', transform=ax.transAxes)

    def _plot_dbscan_info_fixed(self, ax, mejor_config):
        """Mostrar información de parámetros DBSCAN"""
        try:
            info_text = f"Parámetros DBSCAN:\n\n"
            info_text += f"Eps: {mejor_config.get('eps', 0):.3f}\n"
            info_text += f"Min Samples: {mejor_config.get('min_samples', 0)}\n\n"
            info_text += f"Resultados:\n"
            info_text += f"Clusters: {mejor_config.get('n_clusters', 0)}\n"
            info_text += f"Outliers: {mejor_config.get('n_noise', 0)}\n"
            info_text += f"Silhouette: {mejor_config.get('silhouette_score', 0):.3f}\n"

            ax.text(0.05, 0.95, info_text, fontsize=11, va='top', ha='left',
                    family='monospace', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            ax.set_title('Configuración DBSCAN')
            ax.axis('off')

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}',
                    ha='center', va='center', transform=ax.transAxes)

    def _plot_dbscan_distribution_fixed(self, ax, labels):
        """Graficar distribución de puntos por cluster"""
        try:
            unique_labels = sorted([l for l in set(labels) if l != -1])
            outliers_count = list(labels).count(-1)

            # Contar puntos por cluster
            cluster_counts = [list(labels).count(label) for label in unique_labels]
            cluster_names = [f'Cluster {label}' for label in unique_labels]

            if outliers_count > 0:
                cluster_counts.append(outliers_count)
                cluster_names.append('Outliers')

            if cluster_counts:
                colors = ['red' if name == 'Outliers' else 'skyblue' for name in cluster_names]
                bars = ax.bar(cluster_names, cluster_counts, color=colors, alpha=0.7, edgecolor='black')

                # Añadir valores en barras
                for bar, count in zip(bars, cluster_counts):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                            f'{count}', ha='center', va='bottom', fontweight='bold')

                ax.set_ylabel('Número de Puntos')
                ax.set_title('Distribución por Cluster')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3, axis='y')
            else:
                ax.text(0.5, 0.5, 'Sin datos de distribución',
                        ha='center', va='center', transform=ax.transAxes)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}',
                    ha='center', va='center', transform=ax.transAxes)

    def _plot_dbscan_stats_fixed(self, ax, mejor_config):
        """Mostrar estadísticas adicionales"""
        try:
            n_clusters = mejor_config.get('n_clusters', 0)
            n_outliers = mejor_config.get('n_noise', 0)
            total_points = mejor_config.get('total_points', n_clusters * 10 + n_outliers)

            # Crear gráfico de texto informativo
            info_text = f"Estadísticas DBSCAN\n"
            info_text += "=" * 20 + "\n\n"
            info_text += f"Total de puntos: {total_points}\n"
            info_text += f"Clusters válidos: {n_clusters}\n"
            info_text += f"Puntos outliers: {n_outliers}\n\n"

            if total_points > 0:
                cluster_ratio = (total_points - n_outliers) / total_points * 100
                outlier_ratio = n_outliers / total_points * 100
                info_text += f"% en clusters: {cluster_ratio:.1f}%\n"
                info_text += f"% outliers: {outlier_ratio:.1f}%\n\n"

            # Evaluación de calidad
            silhouette = mejor_config.get('silhouette_score', 0)
            if silhouette > 0.7:
                calidad = "Excelente"
            elif silhouette > 0.5:
                calidad = "Buena"
            else:
                calidad = "Regular"

            info_text += f"Calidad clustering: {calidad}\n"
            info_text += f"Silhouette Score: {silhouette:.3f}"

            ax.text(0.05, 0.95, info_text, fontsize=10, va='top', ha='left',
                    family='monospace', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            ax.set_title('Estadísticas Detalladas')
            ax.axis('off')

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}',
                    ha='center', va='center', transform=ax.transAxes)

    def _create_pca_plots_internal(self):
        """Crear plots PCA directamente en self.figure - VERSIÓN CORREGIDA"""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            # Obtener datos originales
            if 'datos_originales' not in self.current_results:
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, 'No hay datos originales para PCA',
                        ha='center', va='center', transform=ax.transAxes)
                return

            datos = self.current_results['datos_originales']
            resultados = self.current_results.get('resultados_por_metodo', {})

            if 'linear' not in resultados:
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, 'No hay resultados de PCA lineal',
                        ha='center', va='center', transform=ax.transAxes)
                return

            # Obtener variables numéricas válidas
            numeric_cols = datos.select_dtypes(include=[np.number]).columns
            exclude_cols = ['Points', 'WQI_IDEAM_6V', 'WQI_IDEAM_7V', 'WQI_NSF_9V']
            variables_pca = [col for col in numeric_cols if col not in exclude_cols]

            if len(variables_pca) < 2:
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, 'Se necesitan al menos 2 variables numéricas para PCA',
                        ha='center', va='center', transform=ax.transAxes)
                return

            # Preparar datos - CORREGIDO: manejo seguro de datos
            datos_pca = datos[variables_pca].dropna()

            if len(datos_pca) < 3:
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, 'Datos insuficientes después de limpiar valores faltantes',
                        ha='center', va='center', transform=ax.transAxes)
                return

            # Aplicar PCA
            scaler = StandardScaler()
            datos_scaled = scaler.fit_transform(datos_pca)

            pca = PCA(n_components=min(5, len(variables_pca), len(datos_pca) - 1))
            datos_pca_transformed = pca.fit_transform(datos_scaled)

            # Layout 2x2
            gs = self.figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

            # 1. Biplot PC1 vs PC2
            ax1 = self.figure.add_subplot(gs[0, 0])
            if datos_pca_transformed.shape[1] >= 2:
                scatter = ax1.scatter(datos_pca_transformed[:, 0], datos_pca_transformed[:, 1],
                                      alpha=0.6, s=50, c='blue', edgecolors='black', linewidth=0.5)

                # Añadir flechas de variables (loadings)
                loadings = pca.components_[:2, :].T * np.sqrt(pca.explained_variance_[:2])

                for i, (var, loading) in enumerate(zip(variables_pca, loadings)):
                    if i < 8:  # Solo mostrar las primeras 8 variables para claridad
                        ax1.arrow(0, 0, loading[0] * 2, loading[1] * 2,
                                  head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
                        ax1.text(loading[0] * 2.2, loading[1] * 2.2, var[:6],
                                 fontsize=8, ha='center', va='center')

                ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
                ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
                ax1.set_title('Biplot PCA')
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'Solo 1 componente disponible',
                         ha='center', va='center', transform=ax1.transAxes)

            # 2. Varianza explicada
            ax2 = self.figure.add_subplot(gs[0, 1])
            var_exp = pca.explained_variance_ratio_
            var_cum = np.cumsum(var_exp)

            x = range(1, len(var_exp) + 1)
            bars = ax2.bar(x, var_exp * 100, alpha=0.6, color='skyblue', label='Individual')

            ax2_twin = ax2.twinx()
            ax2_twin.plot(x, var_cum * 100, 'ro-', linewidth=2, markersize=6, label='Acumulada')
            ax2_twin.axhline(y=95, color='green', linestyle='--', alpha=0.7, label='95%')

            ax2.set_xlabel('Componente Principal')
            ax2.set_ylabel('Varianza Explicada (%)', color='blue')
            ax2_twin.set_ylabel('Varianza Acumulada (%)', color='red')
            ax2.set_title('Varianza Explicada')
            ax2.legend(loc='upper left')
            ax2_twin.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)

            # 3. Contribuciones de variables al PC1
            ax3 = self.figure.add_subplot(gs[1, 0])
            if len(pca.components_) > 0:
                loadings_pc1 = pca.components_[0]
                # Tomar las variables más importantes
                abs_loadings = np.abs(loadings_pc1)
                top_indices = np.argsort(abs_loadings)[-8:][::-1]  # Top 8 variables

                top_vars = [variables_pca[i] for i in top_indices]
                top_loadings = [loadings_pc1[i] for i in top_indices]

                colors = ['red' if x < 0 else 'blue' for x in top_loadings]
                bars = ax3.barh(range(len(top_vars)), top_loadings, color=colors, alpha=0.7)

                ax3.set_yticks(range(len(top_vars)))
                ax3.set_yticklabels([var[:10] for var in top_vars])  # Truncar nombres largos
                ax3.set_xlabel('Loading')
                ax3.set_title('Contribuciones Variables PC1')
                ax3.grid(True, alpha=0.3)

                # Añadir valores
                for i, (bar, val) in enumerate(zip(bars, top_loadings)):
                    ax3.text(val + 0.01 if val >= 0 else val - 0.01, i,
                             f'{val:.2f}', va='center',
                             ha='left' if val >= 0 else 'right', fontsize=8)

            # 4. Información general
            ax4 = self.figure.add_subplot(gs[1, 1])
            n_comp_95 = np.argmax(var_cum >= 0.95) + 1

            info_text = f"Resumen PCA:\n\n"
            info_text += f"Variables analizadas: {len(variables_pca)}\n"
            info_text += f"Muestras: {len(datos_pca)}\n"
            info_text += f"Componentes calculados: {len(var_exp)}\n"
            info_text += f"Componentes para 95%: {n_comp_95}\n\n"
            info_text += f"PC1: {var_exp[0] * 100:.1f}% varianza\n"
            if len(var_exp) > 1:
                info_text += f"PC2: {var_exp[1] * 100:.1f}% varianza\n"
            info_text += f"Total PC1+PC2: {var_cum[1] * 100:.1f}%" if len(
                var_cum) > 1 else f"Solo PC1: {var_cum[0] * 100:.1f}%"

            ax4.text(0.05, 0.95, info_text, fontsize=10, va='top', ha='left',
                     family='monospace', transform=ax4.transAxes,
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            ax4.set_title('Información PCA')
            ax4.axis('off')

            self.figure.suptitle('Análisis de Componentes Principales (PCA)',
                                 fontsize=16, fontweight='bold')

        except Exception as e:
            print(f"Error en PCA plots: {e}")
            import traceback
            traceback.print_exc()

            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Error en PCA: {str(e)[:100]}',
                    ha='center', va='center', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='mistyrose'))
            ax.set_title('Error en PCA')
            ax.axis('off')

    def _export_results(self):
        """Exportar resultados actuales a CSV."""
        if not hasattr(self, 'current_results') or 'datos_originales' not in self.current_results:
            QMessageBox.warning(self, "Sin datos", "No hay resultados para exportar.")
            return

>>>>>>> 73b8f33 (Vivan los papus)
        df = self.current_results['datos_originales'].copy()
        mejor_config = self.current_results.get('mejor_configuracion', {})
        if 'cluster_labels' in mejor_config:
            df['Cluster'] = mejor_config['cluster_labels'][:len(df)]

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar Resultados",
            "",
            "Archivo CSV (*.csv)"
        )
        if file_path:
            try:
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                QMessageBox.information(self, "Éxito", f"Resultados exportados a:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo exportar:\n{e}")

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
        self.current_analysis_type = analysis_type

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
                summary += f"🎯 Mejor configuración:\n"
                summary += f"  - Método: {mejor_config.get('metodo', 'N/A')}\n"
                summary += f"  - Métrica: {mejor_config.get('metrica', 'N/A')}\n"
                summary += f"  - Clusters sugeridos: {mejor_config.get('n_clusters_sugeridos', 'N/A')}\n"

        elif results.get('tipo') == 'pca_completo_avanzado':
            if 'linear' in results.get('resultados_por_metodo', {}):
                linear_result = results['resultados_por_metodo']['linear']
                n_comp = linear_result.get('componentes_recomendados', 'N/A')
                summary += f"📊 Componentes recomendados: {n_comp}\n"
                if 'analisis' in linear_result:
                    var_exp = linear_result['analisis'].get('varianza_acumulada', [])
                    if var_exp and n_comp != 'N/A' and n_comp <= len(var_exp):
                        summary += f"📊 Varianza explicada: {var_exp[n_comp -1 ] *100:.1f}%\n"

        elif results.get('tipo') == 'dbscan_optimizado':
            if 'mejor_configuracion' in results:
                config = results['mejor_configuracion']
                summary += f"🎯 Clusters encontrados: {config.get('n_clusters', 'N/A')}\n"
                summary += f"🔍 Outliers detectados: {config.get('n_noise', 'N/A')}\n"
                summary += f"📊 Eps óptimo: {config.get('eps', 'N/A'):.3f}\n"

        elif results.get('tipo') == 'analisis_exploratorio_completo':
            if 'estadisticas_basicas' in results:
                summary += f"📊 Variables analizadas: {len(results['estadisticas_basicas'])}\n"
            if 'outliers' in results:
                outliers_info = results['outliers']
                summary += f"🔍 Outliers detectados: {outliers_info.get('total_outliers', 'N/A')}\n"

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
                if 'analisis' in linear_result:
                    metrics_data.extend([
                        ("Componentes recomendados", linear_result.get('componentes_recomendados', 'N/A')),
                        ("Varianza PC1", f"{linear_result['analisis']['varianza_explicada'][0 ] *100:.1f}%"),
                        ("Varianza PC2", f"{linear_result['analisis']['varianza_explicada'][1 ] *100:.1f}%"
                        if len(linear_result['analisis']['varianza_explicada']) > 1 else 'N/A'),
                        ("Varianza Acumulada (2 PCs)",
                         f"{linear_result['analisis']['varianza_acumulada'][1 ] *100:.1f}%"
                         if len(linear_result['analisis']['varianza_acumulada']) > 1 else 'N/A')
                    ])

        elif tipo == 'clustering_jerarquico_completo':
            if 'mejor_configuracion' in results:
                config = results['mejor_configuracion']
                metrics_data.extend([
                    ("Método", config.get('metodo', 'N/A')),
                    ("Métrica", config.get('metrica', 'N/A')),
                    ("Clusters sugeridos", config.get('n_clusters_sugeridos', 'N/A')),
                    ("Silhouette Score", f"{config.get('silhouette_score', 0):.3f}"),
                    ("Calinski-Harabasz", f"{config.get('calinski_harabasz_score', 0):.1f}"),
                    ("Davies-Bouldin", f"{config.get('davies_bouldin_score', 0):.3f}")
                ])

        elif tipo == 'dbscan_optimizado':
            if 'mejor_configuracion' in results:
                config = results['mejor_configuracion']
                metrics_data.extend([
                    ("Clusters", config.get('n_clusters', 'N/A')),
                    ("Outliers", config.get('n_noise', 'N/A')),
                    ("% Outliers", f"{config.get('noise_ratio', 0 ) *100:.1f}%"),
                    ("Silhouette Score", f"{config.get('silhouette_score', 0):.3f}"),
                    ("Eps óptimo", f"{config.get('eps', 0):.3f}"),
                    ("Min samples", config.get('min_samples', 'N/A'))
                ])

        elif tipo == 'analisis_exploratorio_completo':
            if 'correlaciones' in results:
                corr_data = results['correlaciones']
                metrics_data.append(("Correlaciones fuertes",
                                     len(corr_data.get('correlaciones_fuertes', []))))
            if 'outliers' in results:
                outlier_data = results['outliers']
                metrics_data.extend([
                    ("Total outliers", outlier_data.get('total_outliers', 'N/A')),
                    ("% Outliers", f"{outlier_data.get('porcentaje_outliers', 0):.1f}%")
                ])

        # Llenar tabla
        self.metrics_table.setRowCount(len(metrics_data))
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Métrica", "Valor"])

        for i, (metric, value) in enumerate(metrics_data):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(str(metric)))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(str(value)))

        self.metrics_table.resizeColumnsToContents()

    def _create_pca_visualization(self):
        """Crear visualización para PCA con enfoque en puntos de muestreo"""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            resultados = self.current_results.get('resultados_por_metodo', {})

            if 'linear' not in resultados:
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, 'No hay resultados de PCA lineal para visualizar',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('PCA - Sin Resultados')
                return

            # NO usar figura externa - trabajar directamente con self.figure
            # self.figure ya está limpia desde _update_visualization

            # Usar la función de visualización especializada
            if 'datos_originales_escalados' in self.current_results:
                # Crear la visualización directamente en self.figure
                self._create_pca_plots_internal()
            else:
                # Visualización PCA tradicional como fallback
                self._create_traditional_pca_visualization()

        except Exception as e:
            print(f"Error en visualización PCA: {e}")
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Error en PCA: {str(e)[:100]}',
                    ha='center', va='center', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='mistyrose'))
            ax.set_title('Error en PCA')
            ax.axis('off')

    def _create_traditional_pca_visualization(self):
        """Visualización PCA tradicional como fallback"""
        try:
            resultados = self.current_results.get('resultados_por_metodo', {})

            if 'linear' not in resultados:
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, 'No hay resultados de PCA para visualizar',
                        ha='center', va='center', transform=ax.transAxes)
                return

            linear_result = resultados['linear']
            analisis = linear_result.get('analisis', {})

            # Crear subplot 2x2
            fig = self.figure
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

            # 1. Varianza explicada
            ax1 = fig.add_subplot(gs[0, 0])
            var_explicada = analisis.get('varianza_explicada', [])
            if var_explicada:
                x = range(1, len(var_explicada) + 1)
                ax1.bar(x, [v * 100 for v in var_explicada], alpha=0.7)
                ax1.set_xlabel('Componente Principal')
                ax1.set_ylabel('Varianza Explicada (%)')
                ax1.set_title('Varianza por Componente')
                ax1.grid(True, alpha=0.3)

            # 2. Varianza acumulada
            ax2 = fig.add_subplot(gs[0, 1])
            var_acumulada = analisis.get('varianza_acumulada', [])
            if var_acumulada:
                x = range(1, len(var_acumulada) + 1)
                ax2.plot(x, [v * 100 for v in var_acumulada], 'o-', linewidth=2)
                ax2.axhline(y=95, color='red', linestyle='--', label='95%')
                ax2.set_xlabel('Componente Principal')
                ax2.set_ylabel('Varianza Acumulada (%)')
                ax2.set_title('Varianza Acumulada')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            # 3. Información de componentes
            ax3 = fig.add_subplot(gs[1, :])
            componentes_info = analisis.get('componentes_info', [])
            if componentes_info and len(componentes_info) > 0:
                # Mostrar contribuciones del primer componente
                pc1_info = componentes_info[0]
                top_vars = pc1_info.get('top_variables', [])[:5]

                if top_vars:
                    variables = [var['variable'] for var in top_vars]
                    loadings = [var['loading'] for var in top_vars]

                    bars = ax3.barh(range(len(variables)), loadings)
                    ax3.set_yticks(range(len(variables)))
                    ax3.set_yticklabels(variables)
                    ax3.set_xlabel('Loading')
                    ax3.set_title('Variables más importantes en PC1')
                    ax3.grid(True, alpha=0.3)

            plt.suptitle('Análisis de Componentes Principales', fontsize=14, fontweight='bold')

        except Exception as e:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Error en PCA tradicional: {str(e)[:100]}',
                    ha='center', va='center', transform=ax.transAxes)

    def _show_error_visualization(self):
        """Mostrar visualización de error"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Error generando visualización',
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='mistyrose'))
        ax.set_title('Error en Visualización')
        ax.axis('off')

    # 5. Fix missing functions in ResultsVisualizationWidget class
    # Add these methods to the ResultsVisualizationWidget class:

    def _show_error(self, error_msg):
        """Mostrar error en lugar de resultados"""
        self.summary_text.setText(f"❌ Error en el análisis:\n\n{error_msg}")
        self.status_label.setText("❌ Error en análisis")
        self.status_label.setStyleSheet("color: red;")

        # Limpiar otros tabs
        self.metrics_table.setRowCount(0)
        self.details_text.setText(f"Error: {error_msg}")

        # Deshabilitar botones
        self.export_results_btn.setEnabled(False)
        self.generate_report_btn.setEnabled(False)

    # 6. Add the missing _create_exploratory_visualization method
    def _create_exploratory_visualization(self):
        """Crear visualización para análisis exploratorio"""
        try:
            self.figure.clear()

            # Usar la función de visualización exploratoria
            if 'datos_originales' in self.current_results:
                self.figure = _crear_visualizacion_exploratorio_puntos_muestreo(
                    self.current_results, figsize=(16, 12)
                )
            else:
                # Fallback básico
                ax = self.figure.add_subplot(111)

                # Mostrar información básica del análisis exploratorio
                calidad = self.current_results.get('calidad_datos', {})
                outliers = self.current_results.get('outliers', {})
                correlaciones = self.current_results.get('correlaciones', {})

                info_text = "Resumen Análisis Exploratorio:\n\n"

                if calidad:
                    score = calidad.get('quality_score', 0)
                    info_text += f"Calidad de datos: {score:.1f}/100\n"
                    info_text += f"Calificación: {calidad.get('calificacion', 'N/A')}\n\n"

                if outliers:
                    consenso = outliers.get('consenso', {})
                    if consenso:
                        info_text += f"Outliers detectados: {consenso.get('total_unico', 0)}\n"
                        info_text += f"Porcentaje outliers: {consenso.get('porcentaje', 0):.1f}%\n\n"

                if correlaciones:
                    corr_fuertes = correlaciones.get('correlaciones_fuertes', [])
                    info_text += f"Correlaciones fuertes: {len(corr_fuertes)}\n"
                    multicolineal = correlaciones.get('multicolinealidad', 'N/A')
                    info_text += f"Multicolinealidad: {multicolineal}\n"

                ax.text(0.1, 0.9, info_text, transform=ax.transAxes,
                        fontsize=12, va='top', ha='left',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
                ax.set_title('Análisis Exploratorio - Resumen')
                ax.axis('off')

            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            print(f"Error en visualización exploratoria: {e}")
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Error en visualización exploratoria:\n{str(e)[:100]}',
                    ha='center', va='center', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='mistyrose'))
            ax.set_title('Error en Visualización')
            ax.axis('off')
            self.canvas.draw()

    def _update_details(self, results: dict):
        """Actualizar detalles técnicos"""
        # Limitar el contenido para evitar problemas de memoria
        results_copy = results.copy()

        # Eliminar datos grandes para el JSON
        if 'datos_originales' in results_copy:
            del results_copy['datos_originales']
        if 'datos_escalados' in results_copy:
            del results_copy['datos_escalados']
        if 'linkage_matrix' in results_copy:
            results_copy['linkage_matrix'] = "... (matriz muy grande, omitida)"

        details = json.dumps(results_copy, indent=2, default=str, ensure_ascii=False)

        # Limitar longitud para evitar sobrecarga
        if len(details) > 50000:
            details = details[:50000] + "\n\n... (Resultado truncado por longitud)"

        self.details_text.setText(details)

    def _update_visualization(self):
        """Actualizar visualización según el tipo de análisis"""
        if not self.current_results or not ML_AVAILABLE:
            return

        try:
            # IMPORTANTE: Limpiar completamente la figura antes de crear nueva
            self.figure.clear()
            plt.close('all')  # Cerrar todas las figuras previas

            tipo = self.current_results.get('tipo', '')

            if tipo == 'kmeans_optimizado':
                self._create_kmeans_visualization()
            elif tipo == 'clustering_jerarquico_completo':
                self._create_hierarchical_visualization()
            elif tipo == 'dbscan_optimizado':
                self._create_dbscan_visualization()
            elif tipo == 'pca_completo_avanzado':
                self._create_pca_visualization()
            elif tipo == 'analisis_exploratorio_completo':
                self._create_exploratory_visualization()
            else:
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, f'Visualización no disponible para: {tipo}',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Visualización No Disponible')

<<<<<<< HEAD
            self.figure.tight_layout()
            self.canvas.draw()
=======
            # IMPORTANTE: Aplicar layout y dibujar
            self.figure.tight_layout()
            self.canvas.draw_idle()  # Usar draw_idle() en lugar de draw()
>>>>>>> 73b8f33 (Vivan los papus)

        except Exception as e:
            print(f"Error en visualización: {e}")
            import traceback
            traceback.print_exc()

<<<<<<< HEAD
            # Mostrar error en el canvas
=======
            # IMPORTANTE: Limpiar y mostrar error
>>>>>>> 73b8f33 (Vivan los papus)
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Error generando visualización:\n{str(e)[:100]}',
                    ha='center', va='center', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='mistyrose'))
            ax.set_title('Error en Visualización')
            ax.axis('off')
<<<<<<< HEAD
            self.canvas.draw()
=======
            self.canvas.draw_idle()
>>>>>>> 73b8f33 (Vivan los papus)

    def _create_kmeans_visualization(self):
        """Crear visualización para K-Means"""
        resultados = self.current_results.get('resultados_por_k', {})

        if not resultados:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No hay resultados de K-Means para visualizar',
                    ha='center', va='center', transform=ax.transAxes)
            return

        # Crear subplots
        fig = self.figure

        # Si tenemos los datos y clusters, hacer visualización 2D
        k_optimo = self.current_results.get('recomendacion_k')
        if k_optimo and k_optimo in resultados and 'datos_originales' in self.current_results:
            # 2 subplots: métricas y clusters
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            # Plot 1: Silhouette score vs K
            k_vals = sorted(list(resultados.keys()))
            silhouette_vals = [resultados[k]['silhouette_score'] for k in k_vals]

            ax1.plot(k_vals, silhouette_vals, 'bo-', linewidth=2, markersize=8)
            ax1.set_xlabel('Número de Clusters (K)', fontsize=11)
            ax1.set_ylabel('Silhouette Score', fontsize=11)
            ax1.set_title('Evaluación de K óptimo', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)

            # Marcar el K óptimo
            if k_optimo in resultados:
                best_score = resultados[k_optimo]['silhouette_score']
                ax1.plot(k_optimo, best_score, 'ro', markersize=12, label=f'K óptimo = {k_optimo}')
                ax1.legend()

            # Plot 2: Visualización de clusters (PCA 2D si hay muchas dimensiones)
            try:
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler

                datos = self.current_results['datos_originales'].dropna()

                # Si hay más de 2 dimensiones, usar PCA
                if datos.shape[1] > 2:
                    scaler = StandardScaler()
                    datos_scaled = scaler.fit_transform(datos)
                    pca = PCA(n_components=2)
                    datos_2d = pca.fit_transform(datos_scaled)
                    xlabel = f'PC1 ({pca.explained_variance_ratio_[0 ] *100:.1f}%)'
                    ylabel = f'PC2 ({pca.explained_variance_ratio_[1 ] *100:.1f}%)'
                else:
                    datos_2d = datos.values
                    xlabel = datos.columns[0] if datos.shape[1] > 0 else 'X'
                    ylabel = datos.columns[1] if datos.shape[1] > 1 else 'Y'

                # Obtener clusters del k óptimo
                if 'cluster_labels' in resultados[k_optimo]:
                    labels = resultados[k_optimo]['cluster_labels'][:len(datos_2d)]

                    # Graficar puntos coloreados por cluster
                    scatter = ax2.scatter(datos_2d[:, 0], datos_2d[:, 1],
                                          c=labels, cmap='viridis',
                                          alpha=0.6, edgecolors='black', linewidth=0.5)
                    ax2.set_xlabel(xlabel, fontsize=11)
                    ax2.set_ylabel(ylabel, fontsize=11)
                    ax2.set_title(f'Clusters K-Means (K={k_optimo})', fontsize=12, fontweight='bold')

                    # Añadir colorbar
                    cbar = plt.colorbar(scatter, ax=ax2)
                    cbar.set_label('Cluster', fontsize=10)

                    # Marcar centroides si están disponibles
                    if 'centroides' in resultados[k_optimo]:
                        centroides = resultados[k_optimo]['centroides']
                        if centroides.shape[1] > 2:
                            # Transformar centroides con el mismo PCA
                            centroides_scaled = scaler.transform(centroides)
                            centroides_2d = pca.transform(centroides_scaled)
                        else:
                            centroides_2d = centroides

                        ax2.scatter(centroides_2d[:, 0], centroides_2d[:, 1],
                                    c='red', marker='*', s=300, edgecolors='black',
                                    linewidth=2, label='Centroides')
                        ax2.legend()
                else:
                    ax2.scatter(datos_2d[:, 0], datos_2d[:, 1], alpha=0.6)
                    ax2.set_xlabel(xlabel, fontsize=11)
                    ax2.set_ylabel(ylabel, fontsize=11)
                    ax2.set_title('Datos sin clusters', fontsize=12)

            except Exception as e:
                print(f"Error en visualización de clusters: {e}")
                ax2.text(0.5, 0.5, 'Error visualizando clusters',
                         ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Error')
        else:
            # Solo mostrar gráfico de métricas
            ax = fig.add_subplot(111)

            k_vals = sorted(list(resultados.keys()))
            silhouette_vals = [resultados[k]['silhouette_score'] for k in k_vals]
            davies_vals = [resultados[k]['davies_bouldin_score'] for k in k_vals]

            ax2 = ax.twinx()

            line1 = ax.plot(k_vals, silhouette_vals, 'bo-', linewidth=2,
                            markersize=8, label='Silhouette (↑ mejor)')
            line2 = ax2.plot(k_vals, davies_vals, 'rs-', linewidth=2,
                             markersize=8, label='Davies-Bouldin (↓ mejor)')

            ax.set_xlabel('Número de Clusters (K)', fontsize=12)
            ax.set_ylabel('Silhouette Score', fontsize=12, color='b')
            ax2.set_ylabel('Davies-Bouldin Score', fontsize=12, color='r')
            ax.set_title('Métricas de Evaluación K-Means', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Combinar leyendas
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='best')

            # Marcar K óptimo
            k_optimo = self.current_results.get('recomendacion_k')
            if k_optimo and k_optimo in resultados:
                ax.axvline(k_optimo, color='green', linestyle='--', alpha=0.7,
                           label=f'K óptimo = {k_optimo}')

    def _create_hierarchical_visualization(self):
        """Crear dendrograma para clustering jerárquico"""
        try:
            from scipy.cluster.hierarchy import dendrogram, linkage
            from sklearn.preprocessing import StandardScaler

            # Obtener datos
            if 'datos_originales' not in self.current_results:
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, 'No hay datos para crear dendrograma',
                        ha='center', va='center', transform=ax.transAxes)
                return

            datos = self.current_results['datos_originales'].dropna()

            # Limitar número de muestras para visualización clara
            max_samples = 100
            if len(datos) > max_samples:
                datos = datos.sample(n=max_samples, random_state=42)
                print(f"Datos limitados a {max_samples} muestras para dendrograma")

            # Escalar datos
            scaler = StandardScaler()
            datos_scaled = scaler.fit_transform(datos)

            # Obtener configuración
            mejor_config = self.current_results.get('mejor_configuracion', {})
            metodo = mejor_config.get('metodo', 'ward')
            metrica = mejor_config.get('metrica', 'euclidean')

            # Ward solo funciona con euclidean
            if metodo == 'ward':
                metrica = 'euclidean'

            # Crear linkage matrix
            Z = linkage(datos_scaled, method=metodo, metric=metrica)

            # Crear dendrograma
            ax = self.figure.add_subplot(111)

            dendro = dendrogram(
                Z,
                ax=ax,
                orientation='top',
                distance_sort='descending',
                show_leaf_counts=True,
                leaf_font_size=10,
                color_threshold=0.7 * np.max(Z[:, 2]),
                above_threshold_color='gray'
            )

            ax.set_title(f'Dendrograma - Clustering Jerárquico ({metodo} + {metrica})',
                         fontsize=14, fontweight='bold', pad=20)
            ax.set_ylabel('Distancia', fontsize=12)
            ax.set_xlabel('Índice de Muestra', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')

            # Añadir línea de corte sugerida
            n_clusters = mejor_config.get('n_clusters_sugeridos', 3)
            if n_clusters > 1 and len(Z) >= n_clusters - 1:
                altura_corte = Z[-(n_clusters -1), 2]
                ax.axhline(y=altura_corte, color='red', linestyle='--',
                           linewidth=2, alpha=0.8)
                ax.text(ax.get_xlim()[1] * 0.7, altura_corte * 1.05,
                        f'{n_clusters} clusters sugeridos',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        fontsize=10, fontweight='bold')

        except Exception as e:
            print(f"Error creando dendrograma: {e}")
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Error creando dendrograma:\n{str(e)[:100]}',
                    ha='center', va='center', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='mistyrose'))
            ax.set_title('Error en Dendrograma')
            ax.axis('off')

    def _create_dbscan_visualization(self):
        """Crear visualización para DBSCAN"""
        try:
<<<<<<< HEAD
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

=======
>>>>>>> 73b8f33 (Vivan los papus)
            mejor_config = self.current_results.get('mejor_configuracion', {})

            if not mejor_config or 'datos_originales' not in self.current_results:
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, 'No hay resultados de DBSCAN para visualizar',
                        ha='center', va='center', transform=ax.transAxes)
<<<<<<< HEAD
                return

            # Obtener datos
            datos = self.current_results['datos_originales'].dropna()

            # Crear figura
            fig = self.figure

            # Si tenemos labels, crear visualización
            if 'cluster_labels' in mejor_config:
                labels = mejor_config['cluster_labels'][:len(datos)]

                # Reducir dimensionalidad si es necesario
                if datos.shape[1] > 2:
                    scaler = StandardScaler()
                    datos_scaled = scaler.fit_transform(datos)
                    pca = PCA(n_components=2)
                    datos_2d = pca.fit_transform(datos_scaled)
                    var_exp = pca.explained_variance_ratio_

                    # 2 subplots
                    ax1 = fig.add_subplot(121)
                    ax2 = fig.add_subplot(122)

                    # Plot 1: Clusters
                    unique_labels = set(labels)
                    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

                    for k, col in zip(unique_labels, colors):
                        if k == -1:
                            # Ruido/outliers en negro
                            col = 'black'
                            marker = 'x'
                            label = 'Outliers'
                        else:
                            marker = 'o'
                            label = f'Cluster {k}'

                        class_member_mask = (labels == k)
                        xy = datos_2d[class_member_mask]
                        ax1.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker,
                                    s=50, label=label, alpha=0.7, edgecolors='black', linewidth=0.5)

                    ax1.set_xlabel(f'PC1 ({var_exp[0] * 100:.1f}%)', fontsize=11)
                    ax1.set_ylabel(f'PC2 ({var_exp[1] * 100:.1f}%)', fontsize=11)
                    ax1.set_title(f'DBSCAN - {mejor_config.get("n_clusters", 0)} Clusters',
                                  fontsize=12, fontweight='bold')
                    ax1.legend(loc='best', fontsize=9)
                    ax1.grid(True, alpha=0.3)

                    # Plot 2: Información de parámetros
                    info_text = f"Parámetros Óptimos:\n\n"
                    info_text += f"Eps: {mejor_config.get('eps', 0):.3f}\n"
                    info_text += f"Min Samples: {mejor_config.get('min_samples', 0)}\n\n"

                    info_text += f"Resultados:\n"
                    info_text += f"Clusters: {mejor_config.get('n_clusters', 0)}\n"
                    info_text += f"Outliers: {mejor_config.get('n_outliers', 0)}\n"

                    silhouette = mejor_config.get('silhouette', None)
                    if silhouette is not None:
                        info_text += f"Silhouette Score: {silhouette:.3f}\n"

                    davies_bouldin = mejor_config.get('davies_bouldin', None)
                    if davies_bouldin is not None:
                        info_text += f"Davies-Bouldin Index: {davies_bouldin:.3f}\n"

                    calinski_harabasz = mejor_config.get('calinski_harabasz', None)
                    if calinski_harabasz is not None:
                        info_text += f"Calinski-Harabasz Index: {calinski_harabasz:.3f}\n"

                    # Mostrar texto
                    ax2.axis('off')
                    ax2.text(0.05, 0.95, info_text, fontsize=10, va='top', ha='left',
                             family='monospace', transform=ax2.transAxes)

        except Exception as e:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Error en visualización DBSCAN:\n{str(e)}",
                    ha='center', va='center', transform=ax.transAxes)
=======
                ax.set_title('DBSCAN - Sin Resultados')
                return

            # Verificar si hay error en los resultados
            if 'error' in self.current_results:
                ax = self.figure.add_subplot(111)
                error_msg = self.current_results.get('mensaje_error', self.current_results['error'])
                ax.text(0.5, 0.5, f'Error en DBSCAN:\n\n{error_msg}',
                        ha='center', va='center', transform=ax.transAxes,
                        bbox=dict(boxstyle='round', facecolor='mistyrose'))
                ax.set_title('Error en DBSCAN')
                ax.axis('off')
                return

            # Crear la visualización
            self._create_dbscan_plots_internal()

        except Exception as e:
            print(f"Error en visualización DBSCAN: {e}")
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Error en DBSCAN:\n{str(e)[:50]}",
                    ha='center', va='center', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='mistyrose'))
            ax.set_title('Error en DBSCAN')
            ax.axis('off')

    def _create_dbscan_plots_internal(self):
        """Crear plots DBSCAN directamente en self.figure - VERSIÓN CORREGIDA"""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            datos = self.current_results['datos_originales']
            mejor_config = self.current_results['mejor_configuracion']

            # CORREGIDO: Verificar que labels existe y es válido
            if 'labels' not in mejor_config and 'cluster_labels' not in mejor_config:
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, 'No hay etiquetas de clusters disponibles',
                        ha='center', va='center', transform=ax.transAxes)
                return

            # Obtener labels con nombres alternativos
            labels = mejor_config.get('labels', mejor_config.get('cluster_labels', []))

            if not labels or len(labels) == 0:
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, 'Etiquetas de clusters vacías',
                        ha='center', va='center', transform=ax.transAxes)
                return

            labels = np.array(labels)

            # Verificar compatibilidad de dimensiones
            if len(labels) != len(datos):
                # Ajustar si es necesario
                min_len = min(len(labels), len(datos))
                labels = labels[:min_len]
                datos = datos.iloc[:min_len]

            # Layout 2x2
            gs = self.figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

            # 1. Clusters con PCA - CORREGIDO
            ax1 = self.figure.add_subplot(gs[0, 0])
            self._plot_dbscan_clusters_fixed(ax1, datos, labels)

            # 2. Información de parámetros
            ax2 = self.figure.add_subplot(gs[0, 1])
            self._plot_dbscan_info_fixed(ax2, mejor_config)

            # 3. Distribución de clusters
            ax3 = self.figure.add_subplot(gs[1, 0])
            self._plot_dbscan_distribution_fixed(ax3, labels)

            # 4. Estadísticas
            ax4 = self.figure.add_subplot(gs[1, 1])
            self._plot_dbscan_stats_fixed(ax4, mejor_config)

            self.figure.suptitle('DBSCAN - Análisis de Clusters y Outliers',
                                 fontsize=16, fontweight='bold')

        except Exception as e:
            print(f"Error en DBSCAN plots: {e}")
            import traceback
            traceback.print_exc()

            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Error en DBSCAN: {str(e)[:100]}',
                    ha='center', va='center', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='mistyrose'))
            ax.set_title('Error en DBSCAN')
            ax.axis('off')

    # 7. Métodos auxiliares para DBSCAN
    def _plot_dbscan_clusters(self, ax, datos, labels):
        """Graficar clusters DBSCAN con PCA - VERSIÓN CORREGIDA"""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            # Seleccionar columnas numéricas válidas
            numeric_cols = datos.select_dtypes(include=[np.number]).columns
            exclude_cols = ['Points', 'WQI_IDEAM_6V', 'WQI_IDEAM_7V', 'WQI_NSF_9V']
            valid_cols = [col for col in numeric_cols if col not in exclude_cols]

            if len(valid_cols) < 2:
                ax.text(0.5, 0.5, 'Insuficientes variables numéricas',
                        ha='center', va='center', transform=ax.transAxes)
                return

            # Preparar datos para PCA
            datos_numeric = datos[valid_cols].dropna()

            # Ajustar labels al tamaño de datos limpios
            if len(datos_numeric) != len(labels):
                # Mantener solo los índices válidos
                valid_indices = datos_numeric.index
                original_indices = datos.index
                mask = np.isin(original_indices, valid_indices)
                labels_clean = labels[mask]
            else:
                labels_clean = labels

            if len(datos_numeric) != len(labels_clean):
                ax.text(0.5, 0.5, 'Incompatibilidad en dimensiones de datos',
                        ha='center', va='center', transform=ax.transAxes)
                return

            # Aplicar PCA para reducir a 2D
            scaler = StandardScaler()
            datos_scaled = scaler.fit_transform(datos_numeric)

            pca = PCA(n_components=2)
            datos_2d = pca.fit_transform(datos_scaled)

            # Colores por cluster
            unique_labels = set(labels_clean)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Outliers en negro
                    col = 'black'
                    marker = 'x'
                    label = 'Outliers'
                    s = 100
                    alpha = 0.8
                else:
                    marker = 'o'
                    label = f'Cluster {k}'
                    s = 60
                    alpha = 0.7

                class_member_mask = (labels_clean == k)
                xy = datos_2d[class_member_mask]

                if len(xy) > 0:  # Solo graficar si hay puntos
                    ax.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker,
                               s=s, label=label, alpha=alpha, edgecolors='black', linewidth=0.5)

            var_exp = pca.explained_variance_ratio_
            ax.set_xlabel(f'PC1 ({var_exp[0] * 100:.1f}%)')
            ax.set_ylabel(f'PC2 ({var_exp[1] * 100:.1f}%)')
            ax.set_title('Clusters DBSCAN (PCA)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}',
                    ha='center', va='center', transform=ax.transAxes)

    def _plot_dbscan_info(self, ax, mejor_config):
        """Mostrar información de parámetros DBSCAN"""
        try:
            info_text = f"Parámetros DBSCAN:\n\n"
            info_text += f"Eps: {mejor_config.get('eps', 0):.3f}\n"
            info_text += f"Min Samples: {mejor_config.get('min_samples', 0)}\n\n"
            info_text += f"Resultados:\n"
            info_text += f"Clusters: {mejor_config.get('n_clusters', 0)}\n"
            info_text += f"Outliers: {mejor_config.get('n_noise', 0)}\n"
            info_text += f"Silhouette: {mejor_config.get('silhouette_score', 0):.3f}\n"

            ax.text(0.05, 0.95, info_text, fontsize=11, va='top', ha='left',
                    family='monospace', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            ax.set_title('Configuración DBSCAN')
            ax.axis('off')

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}',
                    ha='center', va='center', transform=ax.transAxes)

    def _plot_dbscan_distribution(self, ax, labels):
        """Graficar distribución de puntos por cluster"""
        try:
            unique_labels = sorted([l for l in set(labels) if l != -1])
            outliers_count = list(labels).count(-1)

            # Contar puntos por cluster
            cluster_counts = [list(labels).count(label) for label in unique_labels]
            cluster_names = [f'Cluster {label}' for label in unique_labels]

            if outliers_count > 0:
                cluster_counts.append(outliers_count)
                cluster_names.append('Outliers')

            if cluster_counts:
                colors = ['red' if name == 'Outliers' else 'skyblue' for name in cluster_names]
                bars = ax.bar(cluster_names, cluster_counts, color=colors, alpha=0.7, edgecolor='black')

                # Añadir valores en barras
                for bar, count in zip(bars, cluster_counts):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                            f'{count}', ha='center', va='bottom', fontweight='bold')

                ax.set_ylabel('Número de Puntos')
                ax.set_title('Distribución por Cluster')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3, axis='y')
            else:
                ax.text(0.5, 0.5, 'Sin datos de distribución',
                        ha='center', va='center', transform=ax.transAxes)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}',
                    ha='center', va='center', transform=ax.transAxes)

    def _plot_dbscan_stats(self, ax, mejor_config):
        """Mostrar estadísticas adicionales"""
        try:
            n_clusters = mejor_config.get('n_clusters', 0)
            n_outliers = mejor_config.get('n_noise', 0)
            total_points = n_clusters * 10 + n_outliers  # Estimación

            # Crear gráfico de pie
            if n_clusters > 0 or n_outliers > 0:
                labels_pie = []
                sizes = []
                colors = []

                if n_clusters > 0:
                    labels_pie.append(f'Clusters\n({n_clusters})')
                    sizes.append(70)  # Porcentaje aproximado
                    colors.append('lightblue')

                if n_outliers > 0:
                    labels_pie.append(f'Outliers\n({n_outliers})')
                    sizes.append(30)
                    colors.append('red')

                ax.pie(sizes, labels=labels_pie, colors=colors, autopct='%1.1f%%',
                       startangle=90, alpha=0.7)
                ax.set_title('Proporción Clusters vs Outliers')
            else:
                ax.text(0.5, 0.5, 'Sin estadísticas disponibles',
                        ha='center', va='center', transform=ax.transAxes)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}',
                    ha='center', va='center', transform=ax.transAxes)

    # 8. Agregar botón de limpiar gráfico en _create_viz_tab
    def _create_viz_tab(self) -> QWidget:
        """Crear tab de visualizaciones con botón de limpiar"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Canvas para matplotlib
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        # Controles mejorados
        controls_layout = QHBoxLayout()

        self.save_fig_btn = QPushButton("💾 Guardar Gráfico")
        self.save_fig_btn.clicked.connect(self._save_figure)
        controls_layout.addWidget(self.save_fig_btn)

        # NUEVO: Botón de limpiar gráfico
        self.clear_fig_btn = QPushButton("🗑️ Limpiar Gráfico")
        self.clear_fig_btn.clicked.connect(self._clear_figure)
        self.clear_fig_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        controls_layout.addWidget(self.clear_fig_btn)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        widget.setLayout(layout)
        return widget

    # 9. Nuevo método para limpiar figura
    def _clear_figure(self):
        """Limpiar completamente la figura"""
        try:
            self.figure.clear()
            plt.close('all')

            # Mostrar mensaje de figura limpia
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'Gráfico limpiado\n\nEjecuta un nuevo análisis para ver visualizaciones',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            ax.set_title('Figura Limpia')
            ax.axis('off')

            self.canvas.draw_idle()

            # Log de la acción
            if hasattr(self.parent(), 'log'):
                self.parent().log("🗑️ Gráfico limpiado manualmente")

        except Exception as e:
            print(f"Error limpiando figura: {e}")
            QMessageBox.warning(self, "Error", f"No se pudo limpiar el gráfico:\n{e}")
>>>>>>> 73b8f33 (Vivan los papus)


    def _export_results(self):
        """Exportar resultados actuales a un archivo CSV"""
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        import pandas as pd

        try:
            if not hasattr(self, 'current_results') or 'datos_originales' not in self.current_results:
                QMessageBox.warning(self, "Sin datos", "No hay resultados para exportar.")
                return

            # Obtener DataFrame original y etiquetas de cluster
            df = self.current_results['datos_originales'].copy()
            mejor_config = self.current_results.get('mejor_configuracion', {})
            if 'cluster_labels' in mejor_config:
                df['Cluster'] = mejor_config['cluster_labels'][:len(df)]

            # Diálogo para guardar archivo
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Exportar Resultados",
                "",
                "Archivo CSV (*.csv);;Todos los archivos (*.*)"
            )

            if file_path:
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                QMessageBox.information(self, "Exportación completada",
                                        f"Resultados exportados a:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudieron exportar los resultados:\n{str(e)}")

    def _create_metrics_tab(self) -> QWidget:
        """Crear tab de métricas"""
        widget = QWidget()
        layout = QVBoxLayout()

        self.metrics_table = QTableWidget()
        self.metrics_table.setAlternatingRowColors(True)
        layout.addWidget(self.metrics_table)

        widget.setLayout(layout)
        return widget

<<<<<<< HEAD
    def _create_viz_tab(self) -> QWidget:
        """Crear tab de visualizaciones"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Canvas para matplotlib
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        # Controles
        controls_layout = QHBoxLayout()

        self.save_fig_btn = QPushButton("💾 Guardar Gráfico")
        self.save_fig_btn.clicked.connect(self._save_figure)
        controls_layout.addWidget(self.save_fig_btn)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        widget.setLayout(layout)
        return widget


=======
>>>>>>> 73b8f33 (Vivan los papus)
# ==================== VENTANA PRINCIPAL MEJORADA ====================

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

    # ==================== CONFIGURACIÓN DE UI MEJORADA ====================

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

        # Botones de análisis mejorados
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
                "Verifica que scikit-learn, matplotlib y seaborn estén instalados."
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
            # Añadir contamination solo si se maneja outliers
            if config['handle_outliers']:
                kwargs['contamination'] = config['outlier_contamination']

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

        <h3>📈 Interpretación de Resultados:</h3>
        <ul>
        <li><b>Silhouette Score:</b> >0.7 excelente, 0.5-0.7 bueno, <0.5 débil</li>
        <li><b>Davies-Bouldin:</b> Menor es mejor (clusters más separados)</li>
        <li><b>Varianza PCA:</b> Primer componente debería explicar >30% idealmente</li>
        <li><b>Outliers:</b> 5-10% es normal, >20% puede indicar problemas en datos</li>
        </ul>

        <h3>🚨 Solución de Problemas:</h3>
        <ul>
        <li><b>Error "No clusters":</b> Verifica escalado y selección de variables</li>
        <li><b>Resultados inconsistentes:</b> Fija la semilla aleatoria</li>
        <li><b>Análisis lento:</b> Reduce variables o usa muestreo</li>
        <li><b>PCA sin sentido:</b> Verifica correlaciones entre variables</li>
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
        """Aplicar estilos personalizados mejorados"""
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

        /* TabWidget */
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

        /* ListWidget */
        QListWidget {
            border: 2px solid #bdc3c7;
            border-radius: 6px;
            selection-background-color: #3498db;
            background-color: white;
            alternate-background-color: #f8f9fa;
        }

        QListWidget::item {
            padding: 8px;
            border-bottom: 1px solid #ecf0f1;
        }

        QListWidget::item:hover {
            background-color: #e8f4fd;
        }

        QListWidget::item:selected {
            background-color: #3498db;
            color: white;
        }

        /* TextEdit */
        QTextEdit {
            border: 2px solid #bdc3c7;
            border-radius: 6px;
            font-family: 'Segoe UI', Arial, sans-serif;
            background-color: white;
            selection-background-color: #3498db;
        }

        /* TableWidget */
        QTableWidget {
            border: 2px solid #bdc3c7;
            gridline-color: #ecf0f1;
            selection-background-color: #3498db;
            alternate-background-color: #f8f9fa;
            background-color: white;
        }

        QTableWidget::item {
            padding: 8px;
            border-bottom: 1px solid #ecf0f1;
        }

        QTableWidget::item:selected {
            background-color: #3498db;
            color: white;
        }

        QHeaderView::section {
            background-color: #34495e;
            color: white;
            padding: 8px;
            border: none;
            font-weight: bold;
        }

        /* Frame */
        QFrame {
            border: 1px solid #bdc3c7;
            border-radius: 6px;
            background-color: white;
        }

        /* ScrollArea */
        QScrollArea {
            border: 1px solid #bdc3c7;
            border-radius: 6px;
            background-color: white;
        }

        /* ComboBox */
        QComboBox {
            border: 2px solid #bdc3c7;
            border-radius: 4px;
            padding: 4px 8px;
            background-color: white;
        }

        QComboBox:hover {
            border-color: #3498db;
        }

        /* SpinBox */
        QSpinBox, QDoubleSpinBox {
            border: 2px solid #bdc3c7;
            border-radius: 4px;
            padding: 4px 8px;
            background-color: white;
        }

        QSpinBox:hover, QDoubleSpinBox:hover {
            border-color: #3498db;
        }

        /* CheckBox */
        QCheckBox {
            spacing: 8px;
        }

        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border: 2px solid #bdc3c7;
            border-radius: 3px;
            background-color: white;
        }

        QCheckBox::indicator:checked {
            background-color: #3498db;
            border-color: #3498db;
        }
        """

        self.setStyleSheet(style)


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