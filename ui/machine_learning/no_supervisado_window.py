"""
no_supervisado_window.py - Sistema ML No Supervisado CORREGIDO
Sistema de Machine Learning No Supervisado para análisis de calidad del agua
MEJORADO: Integración completa con DataManager y funciones ML avanzadas
"""

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
        generar_datos_agua_realistas
    )

    # Importaciones de matplotlib
    import matplotlib

    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import seaborn as sns

    ML_AVAILABLE = True
    print("✅ Librerías ML No Supervisado cargadas correctamente")

except ImportError as e:
    ML_AVAILABLE = False
    print(f"❌ Error cargando ML No Supervisado: {e}")

# Importar sistema de temas
try:
    from darkmode.theme_manager import ThemedWidget, ThemeManager
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
        self.status.emit("Ejecutando clustering jerárquico...")
        self.progress.emit(30)
        return clustering_jerarquico_completo(self.data, **self.kwargs)

    def _run_kmeans_optimizado(self):
        self.status.emit("Ejecutando K-Means optimizado...")
        self.progress.emit(30)
        return kmeans_optimizado_completo(self.data, **self.kwargs)

    def _run_dbscan(self):
        self.status.emit("Ejecutando DBSCAN...")
        self.progress.emit(30)
        return dbscan_optimizado(self.data, **self.kwargs)

    def _run_pca_avanzado(self):
        self.status.emit("Ejecutando PCA avanzado...")
        self.progress.emit(30)
        return pca_completo_avanzado(self.data, **self.kwargs)

    def _run_analisis_exploratorio(self):
        self.status.emit("Ejecutando análisis exploratorio...")
        self.progress.emit(30)
        return analisis_exploratorio_completo(self.data, **self.kwargs)


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
        self.select_all_btn.clicked.connect(self._select_all_variables)
        controls_layout.addWidget(self.select_all_btn)

        self.select_none_btn = QPushButton("Ninguna")
        self.select_none_btn.clicked.connect(self._select_none_variables)
        controls_layout.addWidget(self.select_none_btn)

        self.auto_select_btn = QPushButton("🤖 Auto")
        self.auto_select_btn.clicked.connect(self._auto_select_variables)
        controls_layout.addWidget(self.auto_select_btn)

        layout.addLayout(controls_layout)

        # Lista de variables
        self.variables_list = QListWidget()
        self.variables_list.setSelectionMode(QListWidget.MultiSelection)
        self.variables_list.itemSelectionChanged.connect(self._on_selection_changed)
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

        # Obtener columnas numéricas
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()

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
    """Widget para configuración de análisis"""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Clustering Configuration
        clustering_group = QGroupBox("🎯 Configuración de Clustering")
        clustering_layout = QFormLayout()

        # K-Means
        self.kmeans_k_min = QSpinBox()
        self.kmeans_k_min.setRange(2, 10)
        self.kmeans_k_min.setValue(2)
        clustering_layout.addRow("K-Means K mínimo:", self.kmeans_k_min)

        self.kmeans_k_max = QSpinBox()
        self.kmeans_k_max.setRange(3, 15)
        self.kmeans_k_max.setValue(8)
        clustering_layout.addRow("K-Means K máximo:", self.kmeans_k_max)

        # DBSCAN
        self.dbscan_optimize = QCheckBox("Optimizar parámetros DBSCAN automáticamente")
        self.dbscan_optimize.setChecked(True)
        clustering_layout.addRow("", self.dbscan_optimize)

        clustering_group.setLayout(clustering_layout)
        layout.addWidget(clustering_group)

        # PCA Configuration
        pca_group = QGroupBox("📊 Configuración de PCA")
        pca_layout = QFormLayout()

        self.pca_variance_threshold = QDoubleSpinBox()
        self.pca_variance_threshold.setRange(0.8, 0.99)
        self.pca_variance_threshold.setValue(0.95)
        self.pca_variance_threshold.setSingleStep(0.05)
        self.pca_variance_threshold.setSuffix("%")
        self.pca_variance_threshold.setDecimals(0)
        self.pca_variance_threshold.setValue(95)
        pca_layout.addRow("Varianza objetivo:", self.pca_variance_threshold)

        self.pca_kernel_methods = QCheckBox("Incluir Kernel PCA")
        self.pca_kernel_methods.setChecked(False)
        pca_layout.addRow("", self.pca_kernel_methods)

        pca_group.setLayout(pca_layout)
        layout.addWidget(pca_group)

        # Preprocessing
        preprocessing_group = QGroupBox("⚙️ Preprocesamiento")
        preprocessing_layout = QFormLayout()

        self.scaling_method = QComboBox()
        self.scaling_method.addItems(['standard', 'robust', 'minmax', 'none'])
        preprocessing_layout.addRow("Método de escalado:", self.scaling_method)

        self.handle_outliers = QCheckBox("Incluir manejo de outliers")
        self.handle_outliers.setChecked(True)
        preprocessing_layout.addRow("", self.handle_outliers)

        preprocessing_group.setLayout(preprocessing_layout)
        layout.addWidget(preprocessing_group)

        layout.addStretch()
        self.setLayout(layout)

    def get_config(self) -> dict:
        """Obtener configuración actual"""
        return {
            'kmeans_k_range': range(self.kmeans_k_min.value(), self.kmeans_k_max.value() + 1),
            'dbscan_optimize': self.dbscan_optimize.isChecked(),
            'pca_variance_threshold': self.pca_variance_threshold.value() / 100.0,
            'pca_include_kernel': self.pca_kernel_methods.isChecked(),
            'scaling_method': self.scaling_method.currentText(),
            'handle_outliers': self.handle_outliers.isChecked()
        }


# ==================== WIDGET DE RESULTADOS ====================

class ResultsVisualizationWidget(QWidget):
    """Widget para visualización de resultados No Supervisado"""

    def __init__(self):
        super().__init__()
        self.current_results = None
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
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        # Controles
        controls_layout = QHBoxLayout()

        self.viz_type_combo = QComboBox()
        self.viz_type_combo.addItems([
            "Vista General", "Clusters", "PCA", "Correlaciones", "Dendrograma"
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
        layout.addWidget(self.details_text)

        widget.setLayout(layout)
        return widget

    def _create_toolbar(self) -> QWidget:
        """Crear toolbar"""
        toolbar = QWidget()
        layout = QHBoxLayout()

        self.export_results_btn = QPushButton("📄 Exportar Resultados")
        self.export_results_btn.clicked.connect(self._export_results)
        layout.addWidget(self.export_results_btn)

        self.generate_report_btn = QPushButton("📊 Generar Reporte")
        self.generate_report_btn.clicked.connect(self._generate_report)
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

    def _update_summary(self, results: dict, analysis_type: str):
        """Actualizar resumen"""
        summary = f"📊 Resumen - {analysis_type.title()}\n"
        summary += "=" * 50 + "\n\n"

        # Información general
        summary += f"Tipo de análisis: {results.get('tipo', 'N/A')}\n"

        if 'variables_utilizadas' in results:
            summary += f"Variables analizadas: {len(results['variables_utilizadas'])}\n"
            summary += f"Variables: {', '.join(results['variables_utilizadas'][:5])}"
            if len(results['variables_utilizadas']) > 5:
                summary += f" (y {len(results['variables_utilizadas']) - 5} más)"
            summary += "\n"

        # Resultados específicos por tipo
        if results.get('tipo') == 'kmeans_optimizado':
            k_optimo = results.get('recomendacion_k', 'N/A')
            summary += f"\n🎯 K óptimo recomendado: {k_optimo}\n"

            if k_optimo != 'N/A' and 'resultados_por_k' in results:
                if k_optimo in results['resultados_por_k']:
                    best_result = results['resultados_por_k'][k_optimo]
                    summary += f"Silhouette Score: {best_result.get('silhouette_score', 'N/A'):.3f}\n"

        elif results.get('tipo') == 'clustering_jerarquico_completo':
            if results.get('mejor_configuracion'):
                mejor_config = results['mejor_configuracion']
                summary += f"\n🎯 Mejor configuración: {mejor_config}\n"

        elif results.get('tipo') == 'pca_completo_avanzado':
            if 'linear' in results.get('resultados_por_metodo', {}):
                linear_result = results['resultados_por_metodo']['linear']
                n_comp = linear_result.get('componentes_recomendados', 'N/A')
                summary += f"\n📊 Componentes recomendados: {n_comp}\n"

        elif results.get('tipo') == 'dbscan_optimizado':
            if 'mejor_configuracion' in results:
                config = results['mejor_configuracion']
                summary += f"\n🎯 Clusters encontrados: {config.get('n_clusters', 'N/A')}\n"
                summary += f"Outliers detectados: {config.get('n_noise', 'N/A')}\n"

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
            elif viz_type == "Clusters":
                self._create_clusters_plot()
            elif viz_type == "PCA":
                self._create_pca_plot()
            elif viz_type == "Correlaciones":
                self._create_correlation_plot()
            elif viz_type == "Dendrograma":
                self._create_dendrogram_plot()

            self.canvas.draw()

        except Exception as e:
            print(f"Error en visualización: {e}")

    def _create_overview_plot(self):
        """Crear gráfico de vista general"""
        tipo = self.current_results.get('tipo', '')

        if tipo == 'kmeans_optimizado':
            # Gráfico de Silhouette vs K
            resultados = self.current_results.get('resultados_por_k', {})
            if resultados:
                ax = self.figure.add_subplot(111)
                k_vals = list(resultados.keys())
                silhouette_vals = [resultados[k]['silhouette_score'] for k in k_vals]

                ax.plot(k_vals, silhouette_vals, 'bo-', linewidth=2, markersize=8)
                ax.set_xlabel('Número de Clusters (K)')
                ax.set_ylabel('Silhouette Score')
                ax.set_title('Evaluación de K óptimo - K-Means')
                ax.grid(True, alpha=0.3)

                # Marcar el mejor K
                k_optimo = self.current_results.get('recomendacion_k')
                if k_optimo in resultados:
                    best_score = resultados[k_optimo]['silhouette_score']
                    ax.plot(k_optimo, best_score, 'ro', markersize=12,
                            label=f'K óptimo = {k_optimo}')
                    ax.legend()

        elif tipo == 'pca_completo_avanzado':
            # Gráfico de varianza explicada
            if 'linear' in self.current_results.get('resultados_por_metodo', {}):
                linear_result = self.current_results['resultados_por_metodo']['linear']
                if 'analisis' in linear_result:
                    ax = self.figure.add_subplot(111)
                    varianza = linear_result['analisis']['varianza_explicada']
                    varianza_acum = linear_result['analisis']['varianza_acumulada']

                    componentes = range(1, min(11, len(varianza) + 1))

                    ax.bar(componentes, [v * 100 for v in varianza[:10]],
                           alpha=0.7, color='steelblue', label='Individual')
                    ax.plot(componentes, [v * 100 for v in varianza_acum[:10]],
                            'ro-', linewidth=2, label='Acumulada')

                    ax.set_xlabel('Componente Principal')
                    ax.set_ylabel('Varianza Explicada (%)')
                    ax.set_title('Análisis de Componentes Principales')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

        elif tipo == 'dbscan_optimizado':
            # Gráfico de configuraciones probadas
            if 'todas_configuraciones' in self.current_results:
                configs = self.current_results['todas_configuraciones'][:10]  # Top 10
                ax = self.figure.add_subplot(111)

                scores = [c['score_compuesto'] for c in configs]
                labels = [f"eps={c['eps']:.2f}\nmin_s={c['min_samples']}" for c in configs]

                bars = ax.bar(range(len(scores)), scores, color='darkgreen', alpha=0.7)
                ax.set_xlabel('Configuración')
                ax.set_ylabel('Score Compuesto')
                ax.set_title('Top 10 Configuraciones DBSCAN')
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')

                # Destacar la mejor
                if scores:
                    bars[0].set_color('gold')

        self.figure.tight_layout()

    def _create_clusters_plot(self):
        """Crear gráfico de clusters"""
        tipo = self.current_results.get('tipo', '')

        # Este es un placeholder - en una implementación real necesitarías
        # los datos transformados y las etiquetas de clusters
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Visualización de Clusters\n(Requiere datos transformados)',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, bbox=dict(boxstyle='round', facecolor='wheat'))
        ax.set_title('Visualización de Clusters')

    def _create_pca_plot(self):
        """Crear gráfico de PCA"""
        if self.current_results.get('tipo') == 'pca_completo_avanzado':
            if 'linear' in self.current_results.get('resultados_por_metodo', {}):
                linear_result = self.current_results['resultados_por_metodo']['linear']

                if 'contribuciones' in linear_result:
                    # Gráfico de contribuciones de variables a PC1 y PC2
                    ax = self.figure.add_subplot(111)
                    contribuciones = linear_result['contribuciones']

                    variables = list(contribuciones.keys())[:10]  # Top 10
                    pc1_contrib = [contribuciones[var]['PC1'] for var in variables]
                    pc2_contrib = [contribuciones[var]['PC2'] for var in variables]

                    x = np.arange(len(variables))
                    width = 0.35

                    ax.bar(x - width / 2, pc1_contrib, width, label='PC1', alpha=0.8)
                    ax.bar(x + width / 2, pc2_contrib, width, label='PC2', alpha=0.8)

                    ax.set_xlabel('Variables')
                    ax.set_ylabel('Contribución')
                    ax.set_title('Contribución de Variables a PC1 y PC2')
                    ax.set_xticks(x)
                    ax.set_xticklabels(variables, rotation=45, ha='right')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

        self.figure.tight_layout()

    def _create_correlation_plot(self):
        """Crear gráfico de correlaciones"""
        if self.current_results.get('tipo') == 'analisis_exploratorio_completo':
            if 'correlaciones' in self.current_results:
                corr_data = self.current_results['correlaciones']

                if 'correlaciones_fuertes' in corr_data:
                    # Gráfico de correlaciones fuertes
                    ax = self.figure.add_subplot(111)
                    correlaciones_fuertes = corr_data['correlaciones_fuertes'][:10]  # Top 10

                    if correlaciones_fuertes:
                        variables = [f"{c['variable_1']} vs\n{c['variable_2']}"
                                     for c in correlaciones_fuertes]
                        correlaciones = [c['pearson'] for c in correlaciones_fuertes]

                        colors = ['red' if c < 0 else 'blue' for c in correlaciones]

                        bars = ax.barh(variables, correlaciones, color=colors, alpha=0.7)
                        ax.set_xlabel('Correlación de Pearson')
                        ax.set_title('Correlaciones Fuertes Detectadas')
                        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                        ax.grid(True, alpha=0.3)
                    else:
                        ax.text(0.5, 0.5, 'No se encontraron\ncorrelaciones fuertes',
                                ha='center', va='center', transform=ax.transAxes,
                                fontsize=14)

        self.figure.tight_layout()

    def _create_dendrogram_plot(self):
        """Crear dendrograma"""
        if self.current_results.get('tipo') == 'clustering_jerarquico_completo':
            # Placeholder para dendrograma
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'Dendrograma\n(Visualización simplificada)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=14, bbox=dict(boxstyle='round', facecolor='lightblue'))
            ax.set_title('Clustering Jerárquico - Dendrograma')

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

        # Panel izquierdo
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
        self.refresh_btn.clicked.connect(self.check_data_availability)
        layout.addWidget(self.refresh_btn)

        self.demo_btn = QPushButton("🎲 Demo")
        self.demo_btn.clicked.connect(self.load_demo_data)
        layout.addWidget(self.demo_btn)

        self.help_btn = QPushButton("❓ Ayuda")
        self.help_btn.clicked.connect(self.show_help)
        layout.addWidget(self.help_btn)

        header.setLayout(layout)
        return header

    def create_left_panel(self) -> QWidget:
        """Crear panel izquierdo de configuración"""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)

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

        # Clustering
        clustering_label = QLabel("🎯 Clustering")
        clustering_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        analysis_layout.addWidget(clustering_label)

        self.kmeans_btn = QPushButton("K-Means Optimizado")
        self.kmeans_btn.clicked.connect(lambda: self.run_analysis('kmeans_optimizado'))
        analysis_layout.addWidget(self.kmeans_btn)

        self.hierarchical_btn = QPushButton("Clustering Jerárquico")
        self.hierarchical_btn.clicked.connect(lambda: self.run_analysis('clustering_jerarquico'))
        analysis_layout.addWidget(self.hierarchical_btn)

        self.dbscan_btn = QPushButton("DBSCAN")
        self.dbscan_btn.clicked.connect(lambda: self.run_analysis('dbscan'))
        analysis_layout.addWidget(self.dbscan_btn)

        # Separador
        analysis_layout.addWidget(QLabel(""))

        # Reducción dimensional
        pca_label = QLabel("📊 Reducción Dimensional")
        pca_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        analysis_layout.addWidget(pca_label)

        self.pca_btn = QPushButton("PCA Avanzado")
        self.pca_btn.clicked.connect(lambda: self.run_analysis('pca_avanzado'))
        analysis_layout.addWidget(self.pca_btn)

        # Separador
        analysis_layout.addWidget(QLabel(""))

        # Análisis exploratorio
        exp_label = QLabel("🔍 Análisis Exploratorio")
        exp_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        analysis_layout.addWidget(exp_label)

        self.exploratory_btn = QPushButton("Análisis Completo")
        self.exploratory_btn.clicked.connect(lambda: self.run_analysis('analisis_exploratorio'))
        analysis_layout.addWidget(self.exploratory_btn)

        # Botón cancelar
        analysis_layout.addWidget(QLabel(""))
        self.cancel_btn = QPushButton("❌ Cancelar Análisis")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_analysis)
        analysis_layout.addWidget(self.cancel_btn)

        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

        layout.addStretch()
        panel.setLayout(layout)
        return panel

    def create_log_widget(self) -> QWidget:
        """Crear widget de log"""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Box)
        widget.setMaximumHeight(120)

        layout = QVBoxLayout()

        # Header
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("📝 Log de Actividad"))

        clear_btn = QPushButton("🗑️ Limpiar")
        clear_btn.clicked.connect(self.clear_log)
        header_layout.addWidget(clear_btn)

        layout.addLayout(header_layout)

        # Área de texto
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
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

        # Configurar kwargs según el tipo de análisis
        kwargs = {
            'variables': variables,
            'escalado': config['scaling_method']
        }

        if analysis_type == 'kmeans_optimizado':
            kwargs.update({
                'k_range': config['kmeans_k_range'],
                'criterios_optimo': ['silhouette', 'elbow', 'gap']
            })
        elif analysis_type == 'clustering_jerarquico':
            kwargs.update({
                'metodos': ['ward', 'complete', 'average'],
                'metricas': ['euclidean', 'manhattan'],
                'max_clusters': max(config['kmeans_k_range'])
            })
        elif analysis_type == 'dbscan':
            kwargs.update({
                'optimizar_parametros': config['dbscan_optimize']
            })
        elif analysis_type == 'pca_avanzado':
            metodos = ['linear']
            if config['pca_include_kernel']:
                metodos.append('kernel')
            kwargs.update({
                'metodos': metodos,
                'explicar_varianza_objetivo': config['pca_variance_threshold']
            })

        # Mostrar progreso
        self.show_progress(True)
        self.log(f"🚀 Iniciando análisis: {analysis_type}")

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
            self.log("❌ Análisis cancelado")
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
            'variables': self.variable_selection.get_selected_variables()
        }
        self.analysis_history.append(analysis_entry)

        # Actualizar resultados
        self.results_widget.update_results(
            results,
            self.current_worker.analysis_type if self.current_worker else 'unknown'
        )

        self.log("✅ Análisis completado exitosamente")

    @pyqtSlot(str)
    def on_analysis_error(self, error_msg: str):
        """Cuando ocurre un error"""
        self.show_progress(False)
        self.log(f"❌ Error: {error_msg}")
        QMessageBox.critical(self, "Error en Análisis", error_msg)

    def on_variables_changed(self):
        """Cuando cambian las variables seleccionadas"""
        # Aquí se puede añadir lógica adicional si es necesario
        pass

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
        help_dialog.resize(700, 600)

        layout = QVBoxLayout()

        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
        <h2>🔍 Machine Learning No Supervisado</h2>

        <h3>📌 ¿Qué es el ML No Supervisado?</h3>
        <p>El aprendizaje no supervisado busca patrones ocultos en datos sin etiquetas predefinidas.</p>

        <h3>🎯 Técnicas de Clustering:</h3>
        <ul>
        <li><b>K-Means:</b> Agrupa datos en K clusters esféricos</li>
        <li><b>Clustering Jerárquico:</b> Crea dendrogramas con jerarquías de grupos</li>
        <li><b>DBSCAN:</b> Detecta clusters de densidad variable y outliers</li>
        </ul>

        <h3>📊 Reducción Dimensional:</h3>
        <ul>
        <li><b>PCA Linear:</b> Proyecta datos en componentes principales</li>
        <li><b>Kernel PCA:</b> PCA no lineal para patrones complejos</li>
        </ul>

        <h3>🔍 Análisis Exploratorio:</h3>
        <ul>
        <li><b>Correlaciones:</b> Detecta relaciones entre variables</li>
        <li><b>Distribuciones:</b> Analiza patrones en los datos</li>
        <li><b>Outliers:</b> Identifica valores atípicos</li>
        </ul>

        <h3>🚀 Cómo usar:</h3>
        <ol>
        <li>Carga datos desde el módulo "Cargar Datos" o usa el botón Demo</li>
        <li>Selecciona las variables numéricas a analizar</li>
        <li>Configura los parámetros según tu análisis</li>
        <li>Ejecuta el algoritmo deseado</li>
        <li>Revisa los resultados en las pestañas de visualización</li>
        </ol>

        <h3>💡 Consejos:</h3>
        <ul>
        <li>Para clustering, comienza con K-Means para obtener una idea general</li>
        <li>DBSCAN es excelente para detectar outliers</li>
        <li>PCA ayuda a visualizar datos de alta dimensionalidad</li>
        <li>El análisis exploratorio es ideal para comenzar cualquier proyecto</li>
        </ul>

        <h3>⚙️ Configuración:</h3>
        <ul>
        <li><b>Escalado:</b> Standard es recomendado para la mayoría de casos</li>
        <li><b>K-Means:</b> Prueba rangos de 2-8 clusters inicialmente</li>
        <li><b>PCA:</b> 95% de varianza explicada es un buen punto de partida</li>
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
        QGroupBox {
            font-weight: bold;
            border: 2px solid #cccccc;
            border-radius: 5px;
            margin-top: 1ex;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }

        QPushButton {
            background-color: #3498db;
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
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

        QProgressBar {
            border: 2px solid #bdc3c7;
            border-radius: 5px;
            text-align: center;
        }

        QProgressBar::chunk {
            background-color: #27ae60;
            border-radius: 3px;
        }

        QTabWidget::pane {
            border: 1px solid #bdc3c7;
        }

        QTabBar::tab {
            background: #ecf0f1;
            border: 1px solid #bdc3c7;
            padding: 8px 12px;
            margin-right: 2px;
        }

        QTabBar::tab:selected {
            background: #3498db;
            color: white;
        }

        QListWidget {
            border: 1px solid #bdc3c7;
            border-radius: 4px;
            selection-background-color: #3498db;
        }

        QTextEdit {
            border: 1px solid #bdc3c7;
            border-radius: 4px;
            font-family: 'Segoe UI', Arial, sans-serif;
        }

        QTableWidget {
            border: 1px solid #bdc3c7;
            gridline-color: #ecf0f1;
            selection-background-color: #3498db;
        }

        QTableWidget::item {
            padding: 4px;
        }

        QFrame {
            border: 1px solid #bdc3c7;
            border-radius: 4px;
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