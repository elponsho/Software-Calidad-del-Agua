"""
no_supervisado_window.py - Compatible con PyInstaller SIN THREADING
Ventana principal para Machine Learning No Supervisado
Versión completamente secuencial
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
    QListWidgetItem, QScrollArea, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor

# Importar gestor de datos
try:
    from .data_manager import get_data_manager, has_shared_data, get_shared_data
    DATA_MANAGER_AVAILABLE = True
    print("✅ DataManager importado correctamente")
except ImportError as e:
    DATA_MANAGER_AVAILABLE = False
    print(f"⚠️ DataManager no disponible: {e}")

    def get_data_manager():
        return None

    def has_shared_data():
        return False

    def get_shared_data():
        return None

# Importar funciones ML No Supervisado
ML_AVAILABLE = True
try:
    from .ml_functions_no_supervisado import (
        generar_datos_agua_realistas,
        kmeans_optimizado_completo,
        dbscan_optimizado,
        analisis_exploratorio_completo,
        pca_completo_avanzado,
        clustering_jerarquico_completo,
        generar_visualizaciones_ml_no_supervisado
    )

    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    ML_AVAILABLE = True
    print("✅ Librerías ML No Supervisado cargadas correctamente")

except ImportError as e:
    ML_AVAILABLE = False
    print(f"⚠️ Librerías ML No Supervisado no disponibles: {e}")

    try:
        import matplotlib
        matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
    except ImportError:
        class Figure:
            def __init__(self, *args, **kwargs):
                pass
            def clear(self):
                pass
            def add_subplot(self, *args, **kwargs):
                return MockAxes()
            def tight_layout(self):
                pass
            def savefig(self, *args, **kwargs):
                pass

        class FigureCanvas:
            def __init__(self, figure):
                self.figure = figure
            def draw(self):
                pass
            def draw_idle(self):
                pass

        class MockAxes:
            def text(self, *args, **kwargs):
                pass
            def set_title(self, *args, **kwargs):
                pass
            def axis(self, *args, **kwargs):
                pass

    def kmeans_optimizado_completo(*args, **kwargs):
        return {'error': 'ML no disponible', 'tipo': 'kmeans_optimizado'}

    def dbscan_optimizado(*args, **kwargs):
        return {'error': 'ML no disponible', 'tipo': 'dbscan_optimizado'}

    def pca_completo_avanzado(*args, **kwargs):
        return {'error': 'ML no disponible', 'tipo': 'pca_completo_avanzado'}

    def analisis_exploratorio_completo(*args, **kwargs):
        return {'error': 'ML no disponible', 'tipo': 'analisis_exploratorio_completo'}

    def clustering_jerarquico_completo(*args, **kwargs):
        return {'error': 'ML no disponible', 'tipo': 'clustering_jerarquico_completo'}

    def generar_datos_agua_realistas(*args, **kwargs):
        return pd.DataFrame(np.random.randn(10, 5), columns=['A', 'B', 'C', 'D', 'E'])

    def generar_visualizaciones_ml_no_supervisado(*args, **kwargs):
        return Figure()

# Sistema de temas simplificado
try:
    class ThemedWidget:
        def __init__(self):
            pass
        def apply_theme(self):
            pass
except ImportError:
    class ThemedWidget(QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)


# ==================== WIDGET DE SELECCIÓN DE VARIABLES ====================

class VariableSelectionWidget(QWidget):
    """Widget para selección de variables"""

    def __init__(self):
        super().__init__()
        self.data = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        title = QLabel("📊 Selección de Variables")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title)

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

        self.variables_list = QListWidget()
        self.variables_list.setSelectionMode(QListWidget.MultiSelection)
        self.variables_list.setMinimumHeight(150)
        layout.addWidget(self.variables_list)

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

        self.variables_list.clear()

        exclude_cols = ['Points', 'Sampling_date', 'Classification_6V', 'Classification_7V', 'Classification_9V']
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        for col in numeric_cols:
            col_data = self.data[col]
            missing_pct = (col_data.isnull().sum() / len(col_data)) * 100

            item_text = f"{col} (missing: {missing_pct:.1f}%)"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, col)

            if missing_pct > 50:
                item.setForeground(QColor('red'))
            elif missing_pct > 20:
                item.setForeground(QColor('orange'))

            self.variables_list.addItem(item)

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

        self._select_none_variables()
        selected_count = 0

        for i in range(self.variables_list.count()):
            item = self.variables_list.item(i)
            col_name = item.data(Qt.UserRole)

            col_data = self.data[col_name]
            missing_pct = (col_data.isnull().sum() / len(col_data)) * 100
            variance = col_data.var()

            if missing_pct < 20 and not np.isnan(variance) and variance > 0:
                item.setSelected(True)
                selected_count += 1

        if selected_count > 0:
            QMessageBox.information(
                self, "Selección Automática",
                f"Se seleccionaron {selected_count} variables con buena calidad"
            )
        else:
            QMessageBox.warning(
                self, "Selección Automática",
                "No se encontraron variables que cumplan los criterios"
            )

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
        main_layout = QVBoxLayout()
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setMinimumHeight(300)

        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setSpacing(15)

        # ===== CLUSTERING =====
        clustering_group = QGroupBox("🎯 Configuración de Clustering")
        clustering_layout = QFormLayout()
        clustering_layout.setSpacing(10)

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

        self.dbscan_optimize = QCheckBox("Optimizar parámetros DBSCAN automáticamente")
        self.dbscan_optimize.setChecked(True)
        self.dbscan_optimize.setMinimumHeight(30)
        clustering_layout.addRow("", self.dbscan_optimize)

        self.hierarchical_method = QComboBox()
        self.hierarchical_method.addItems(['ward', 'complete', 'average', 'single'])
        self.hierarchical_method.setCurrentText('ward')
        self.hierarchical_method.setMinimumHeight(30)
        clustering_layout.addRow("Método jerárquico:", self.hierarchical_method)

        self.hierarchical_metric = QComboBox()
        self.hierarchical_metric.addItems(['euclidean', 'manhattan', 'cosine'])
        self.hierarchical_metric.setCurrentText('euclidean')
        self.hierarchical_metric.setMinimumHeight(30)
        clustering_layout.addRow("Métrica distancia:", self.hierarchical_metric)

        self.hierarchical_max_clusters = QSpinBox()
        self.hierarchical_max_clusters.setRange(2, 20)
        self.hierarchical_max_clusters.setValue(10)
        self.hierarchical_max_clusters.setMinimumHeight(30)
        clustering_layout.addRow("Clusters máximos:", self.hierarchical_max_clusters)

        clustering_group.setLayout(clustering_layout)
        content_layout.addWidget(clustering_group)

        # ===== PCA =====
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

        pca_group.setLayout(pca_layout)
        content_layout.addWidget(pca_group)

        # ===== PREPROCESAMIENTO =====
        preprocessing_group = QGroupBox("⚙️ Preprocesamiento")
        preprocessing_layout = QFormLayout()
        preprocessing_layout.setSpacing(10)

        self.scaling_method = QComboBox()
        self.scaling_method.addItems(['standard', 'robust', 'minmax', 'none'])
        self.scaling_method.setCurrentText('standard')
        self.scaling_method.setMinimumHeight(30)
        preprocessing_layout.addRow("Método de escalado:", self.scaling_method)

        self.handle_outliers = QCheckBox("Detectar y manejar outliers")
        self.handle_outliers.setChecked(True)
        self.handle_outliers.setMinimumHeight(30)
        preprocessing_layout.addRow("", self.handle_outliers)

        preprocessing_group.setLayout(preprocessing_layout)
        content_layout.addWidget(preprocessing_group)

        # ===== GENERAL =====
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

        content_layout.addStretch()

        content_widget.setLayout(content_layout)
        scroll_area.setWidget(content_widget)

        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)

    def get_config(self) -> dict:
        """Obtener configuración actual"""
        return {
            'kmeans_k_range': range(self.kmeans_k_min.value(), self.kmeans_k_max.value() + 1),
            'dbscan_optimize': self.dbscan_optimize.isChecked(),
            'hierarchical_method': self.hierarchical_method.currentText(),
            'hierarchical_metric': self.hierarchical_metric.currentText(),
            'hierarchical_max_clusters': self.hierarchical_max_clusters.value(),
            'pca_variance_threshold': self.pca_variance_threshold.value(),
            'scaling_method': self.scaling_method.currentText(),
            'handle_outliers': self.handle_outliers.isChecked(),
            'random_state': self.random_state.value(),
            'verbose': self.verbose_output.isChecked()
        }


# ==================== WIDGET DE RESULTADOS ====================

class ResultsVisualizationWidget(QWidget):
    """Widget para visualización de resultados"""

    def __init__(self):
        super().__init__()
        self.current_results = None
        self.current_analysis_type = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        self.tabs = QTabWidget()

        self.summary_widget = self._create_summary_tab()
        self.tabs.addTab(self.summary_widget, "📋 Resumen")

        self.metrics_widget = self._create_metrics_tab()
        self.tabs.addTab(self.metrics_widget, "📊 Métricas")

        if ML_AVAILABLE:
            self.viz_widget = self._create_viz_tab()
            self.tabs.addTab(self.viz_widget, "📈 Visualizaciones")

        self.details_widget = self._create_details_tab()
        self.tabs.addTab(self.details_widget, "🔍 Detalles")

        layout.addWidget(self.tabs)

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

        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        controls_layout = QHBoxLayout()

        self.save_fig_btn = QPushButton("💾 Guardar Gráfico")
        self.save_fig_btn.clicked.connect(self._save_figure)
        controls_layout.addWidget(self.save_fig_btn)

        self.clear_fig_btn = QPushButton("🗑️ Limpiar")
        self.clear_fig_btn.clicked.connect(self._clear_figure)
        controls_layout.addWidget(self.clear_fig_btn)

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

        # NUEVO: Botón para dendrograma completo
        self.dendrograma_btn = QPushButton("🌳 Dendrograma Ampliado")
        self.dendrograma_btn.setMinimumHeight(35)
        self.dendrograma_btn.clicked.connect(self._mostrar_dendrograma_ampliado)
        self.dendrograma_btn.setEnabled(False)
        layout.addWidget(self.dendrograma_btn)

        layout.addStretch()

        self.status_label = QLabel("Sin resultados")
        self.status_label.setStyleSheet("color: #666;")
        layout.addWidget(self.status_label)

        toolbar.setLayout(layout)
        return toolbar

    def _mostrar_dendrograma_ampliado(self):
        """Mostrar dendrograma en ventana ampliada"""
        if not self.current_results:
            QMessageBox.warning(self, "Sin datos", "No hay resultados para mostrar.")
            return

        tipo = self.current_results.get('tipo', '')

        if tipo != 'clustering_jerarquico_completo':
            QMessageBox.information(
                self, "Tipo incorrecto",
                "El dendrograma ampliado solo está disponible para Clustering Jerárquico."
            )
            return

        try:
            from .ml_functions_no_supervisado import generar_dendrograma_completo

            linkage_matrix = np.array(self.current_results.get('linkage_matrix', []))
            n_samples = len(self.current_results.get('datos_originales', []))

            if len(linkage_matrix) == 0:
                QMessageBox.warning(self, "Sin datos", "No hay matriz de linkage disponible.")
                return

            # Generar dendrograma completo
            fig = generar_dendrograma_completo(
                linkage_matrix,
                n_samples,
                truncate_mode='level',
                p=12,
                mostrar_linea_corte=True
            )

            # Crear ventana de diálogo
            dialog = QDialog(self)
            dialog.setWindowTitle("🌳 Dendrograma Ampliado - Clustering Jerárquico")
            dialog.setModal(False)
            dialog.resize(1400, 900)

            layout = QVBoxLayout()

            # Canvas con el dendrograma
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)

            # Botones
            btn_layout = QHBoxLayout()

            save_btn = QPushButton("💾 Guardar Imagen")
            save_btn.clicked.connect(lambda: self._save_dendrograma(fig))
            btn_layout.addWidget(save_btn)

            close_btn = QPushButton("❌ Cerrar")
            close_btn.clicked.connect(dialog.close)
            btn_layout.addWidget(close_btn)

            btn_layout.addStretch()
            layout.addLayout(btn_layout)

            dialog.setLayout(layout)
            dialog.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error mostrando dendrograma:\n{e}")

    def _save_dendrograma(self, fig):
        """Guardar dendrograma"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Dendrograma", "", "Imagen PNG (*.png);;Imagen JPEG (*.jpg)"
        )
        if file_path:
            try:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Éxito", f"Dendrograma guardado en:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo guardar:\n{e}")

    def update_results(self, results: dict, analysis_type: str):
        """Actualizar con nuevos resultados"""
        self.current_results = results
        self.current_analysis_type = analysis_type

        if 'error' in results:
            self._show_error(results['error'])
            return

        self._update_summary(results, analysis_type)
        self._update_metrics(results)
        self._update_details(results)

        if ML_AVAILABLE:
            self._update_visualization()

        self.status_label.setText(f"✅ {analysis_type} completado")
        self.status_label.setStyleSheet("color: green;")

        self.export_results_btn.setEnabled(True)
        self.generate_report_btn.setEnabled(True)

        # Habilitar botón de dendrograma solo para clustering jerárquico
        if results.get('tipo') == 'clustering_jerarquico_completo':
            self.dendrograma_btn.setEnabled(True)
        else:
            self.dendrograma_btn.setEnabled(False)

    def _update_summary(self, results: dict, analysis_type: str):
        """Actualizar resumen"""
        summary = f"📊 Resumen - {analysis_type.replace('_', ' ').title()}\n"
        summary += "=" * 50 + "\n\n"

        summary += f"🔍 Tipo: {results.get('tipo', 'N/A')}\n"
        summary += f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        if 'variables_utilizadas' in results:
            summary += f"📈 Variables: {len(results['variables_utilizadas'])}\n"
            summary += f"📝 Variables: {', '.join(results['variables_utilizadas'][:5])}"
            if len(results['variables_utilizadas']) > 5:
                summary += f" (y {len(results['variables_utilizadas']) - 5} más)"
            summary += "\n\n"

        if results.get('tipo') == 'kmeans_optimizado':
            k_optimo = results.get('recomendacion_k', 'N/A')
            summary += f"🎯 K óptimo: {k_optimo}\n"

        elif results.get('tipo') == 'dbscan_optimizado':
            config = results.get('mejor_configuracion', {})
            summary += f"🎯 Clusters: {config.get('n_clusters', 'N/A')}\n"
            summary += f"🔍 Outliers: {config.get('n_noise', 'N/A')}\n"

        elif results.get('tipo') == 'pca_completo_avanzado':
            if 'linear' in results.get('resultados_por_metodo', {}):
                linear_result = results['resultados_por_metodo']['linear']
                n_comp = linear_result.get('componentes_recomendados', 'N/A')
                summary += f"📊 Componentes: {n_comp}\n"

        elif results.get('tipo') == 'clustering_jerarquico_completo':
            mejor_config = results.get('mejor_configuracion', {})
            summary += f"🎯 K óptimo: {mejor_config.get('n_clusters_sugeridos', 'N/A')}\n"

        if 'recomendaciones' in results:
            summary += "\n💡 Recomendaciones:\n"
            for i, rec in enumerate(results['recomendaciones'][:3], 1):
                summary += f"{i}. {rec}\n"

        self.summary_text.setText(summary)

    def _update_metrics(self, results: dict):
        """Actualizar métricas"""
        metrics_data = []

        if 'variables_utilizadas' in results:
            metrics_data.append(("Variables utilizadas", len(results['variables_utilizadas'])))

        tipo = results.get('tipo', '')

        if tipo == 'kmeans_optimizado':
            k_optimo = results.get('recomendacion_k')
            if k_optimo and 'resultados_por_k' in results:
                if k_optimo in results['resultados_por_k']:
                    best_result = results['resultados_por_k'][k_optimo]
                    metrics_data.extend([
                        ("K óptimo", k_optimo),
                        ("Silhouette Score", f"{best_result.get('silhouette_score', 0):.3f}"),
                        ("Inercia", f"{best_result.get('inertia', 0):.2f}")
                    ])

        elif tipo == 'dbscan_optimizado':
            config = results.get('mejor_configuracion', {})
            metrics_data.extend([
                ("Clusters", config.get('n_clusters', 'N/A')),
                ("Outliers", config.get('n_noise', 'N/A')),
                ("% Outliers", f"{config.get('noise_ratio', 0) * 100:.1f}%"),
                ("Silhouette Score", f"{config.get('silhouette_score', 0):.3f}")
            ])

        elif tipo == 'pca_completo_avanzado':
            if 'linear' in results.get('resultados_por_metodo', {}):
                linear_result = results['resultados_por_metodo']['linear']
                if 'analisis' in linear_result:
                    analisis = linear_result['analisis']
                    metrics_data.extend([
                        ("Componentes recomendados", linear_result.get('componentes_recomendados', 'N/A')),
                        ("Varianza PC1", f"{analisis['varianza_explicada'][0] * 100:.1f}%"),
                        ("Varianza PC2", f"{analisis['varianza_explicada'][1] * 100:.1f}%"
                        if len(analisis['varianza_explicada']) > 1 else 'N/A')
                    ])

        elif tipo == 'clustering_jerarquico_completo':
            mejor_config = results.get('mejor_configuracion', {})
            metrics_data.extend([
                ("K sugerido", mejor_config.get('n_clusters_sugeridos', 'N/A')),
                ("Método", mejor_config.get('metodo', 'N/A')),
                ("Silhouette", f"{mejor_config.get('silhouette_score', 0):.3f}")
            ])

        self.metrics_table.setRowCount(len(metrics_data))
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Métrica", "Valor"])

        for i, (metric, value) in enumerate(metrics_data):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(str(metric)))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(str(value)))

        self.metrics_table.resizeColumnsToContents()

    def _update_details(self, results: dict):
        """Actualizar detalles técnicos"""
        results_copy = results.copy()

        if 'datos_originales' in results_copy:
            del results_copy['datos_originales']

        try:
            details = json.dumps(results_copy, indent=2, default=str, ensure_ascii=False)

            if len(details) > 50000:
                details = details[:50000] + "\n\n... (Resultado truncado)"

            self.details_text.setText(details)
        except Exception as e:
            self.details_text.setText(f"Error mostrando detalles: {e}")

    def _update_visualization(self):
        """Actualizar visualización"""
        if not self.current_results or not ML_AVAILABLE:
            return

        try:
            self.figure.clear()
            plt.close('all')

            if 'error' not in self.current_results:
                nueva_figura = generar_visualizaciones_ml_no_supervisado(
                    self.current_results, figsize=(12, 8)
                )

                self.figure = nueva_figura
                self.canvas.figure = self.figure

            self.canvas.draw_idle()

        except Exception as e:
            print(f"Error en visualización: {e}")
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Error generando visualización:\n{str(e)[:100]}',
                    ha='center', va='center', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='mistyrose'))
            ax.set_title('Error en Visualización')
            ax.axis('off')
            self.canvas.draw_idle()

    def _show_error(self, error_msg):
        """Mostrar error"""
        self.summary_text.setText(f"❌ Error en el análisis:\n\n{error_msg}")
        self.status_label.setText("❌ Error en análisis")
        self.status_label.setStyleSheet("color: red;")

        self.metrics_table.setRowCount(0)
        self.details_text.setText(f"Error: {error_msg}")

        self.export_results_btn.setEnabled(False)
        self.generate_report_btn.setEnabled(False)

    def _save_figure(self):
        """Guardar figura"""
        if not hasattr(self, 'figure'):
            QMessageBox.warning(self, "Sin figura", "No hay figura para guardar.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Figura", "", "Imagen PNG (*.png);;Imagen JPEG (*.jpg)"
        )
        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Éxito", f"Figura guardada en:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo guardar:\n{e}")

    def _clear_figure(self):
        """Limpiar figura"""
        try:
            self.figure.clear()
            plt.close('all')

            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'Gráfico limpiado\n\nEjecuta un nuevo análisis',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgreen'))
            ax.set_title('Figura Limpia')
            ax.axis('off')

            self.canvas.draw_idle()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"No se pudo limpiar:\n{e}")

    def _export_results(self):
        """Exportar resultados"""
        if not self.current_results or 'datos_originales' not in self.current_results:
            QMessageBox.warning(self, "Sin datos", "No hay resultados para exportar.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Exportar Resultados", "", "Archivo CSV (*.csv)"
        )
        if file_path:
            try:
                df = self.current_results['datos_originales'].copy()

                if self.current_results.get('tipo') == 'kmeans_optimizado':
                    k_opt = self.current_results.get('recomendacion_k')
                    if k_opt and 'resultados_por_k' in self.current_results:
                        if k_opt in self.current_results['resultados_por_k']:
                            labels = self.current_results['resultados_por_k'][k_opt]['labels']
                            df['Cluster'] = labels[:len(df)]

                elif self.current_results.get('tipo') == 'dbscan_optimizado':
                    config = self.current_results.get('mejor_configuracion', {})
                    if 'labels' in config:
                        labels = config['labels']
                        df['Cluster'] = labels[:len(df)]

                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                QMessageBox.information(self, "Éxito", f"Resultados exportados a:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo exportar:\n{e}")

    def _generate_report(self):
        """Generar reporte"""
        if not self.current_results:
            QMessageBox.warning(self, "Sin datos", "No hay resultados para el reporte.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Reporte", "", "Archivo de texto (*.txt)"
        )
        if not file_path:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("=== Reporte de ML No Supervisado ===\n\n")
                f.write(f"Tipo: {self.current_results.get('tipo', 'N/A')}\n")
                f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                if 'recomendaciones' in self.current_results:
                    f.write("Recomendaciones:\n")
                    for i, rec in enumerate(self.current_results['recomendaciones'], 1):
                        f.write(f"{i}. {rec}\n")

            QMessageBox.information(self, "Éxito", f"Reporte guardado en:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo generar el reporte:\n{e}")


# ==================== VENTANA PRINCIPAL ====================

class NoSupervisadoWindow(QWidget, ThemedWidget):
    """Ventana principal para ML No Supervisado (SIN THREADING)"""

    def __init__(self):
        QWidget.__init__(self)
        ThemedWidget.__init__(self)

        print("🚀 NoSupervisadoWindow: Inicializando (SIN THREADING)...")

        self.current_data = None
        self.analysis_history = []
        self.is_processing = False

        if DATA_MANAGER_AVAILABLE:
            dm = get_data_manager()
            if dm is not None:
                dm.add_observer(self)
                print("✅ Registrada como observador del DataManager")
            else:
                print("⚠️ DataManager no disponible")
        else:
            print("⚠️ DataManager no importado")

        self.setup_ui()
        self.check_data_availability()
        print("✅ NoSupervisadoWindow: Inicialización completada")

    def setup_ui(self):
        """Configurar interfaz de usuario"""
        self.setWindowTitle("🔍 Machine Learning No Supervisado (Secuencial)")
        self.setMinimumSize(1400, 900)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        header = self.create_header()
        main_layout.addWidget(header)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        content_splitter = QSplitter(Qt.Horizontal)

        left_panel = self.create_left_panel()
        content_splitter.addWidget(left_panel)

        self.results_widget = ResultsVisualizationWidget()
        content_splitter.addWidget(self.results_widget)

        content_splitter.setSizes([450, 950])
        main_layout.addWidget(content_splitter)

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
                print("⚠️ No hay datos disponibles")
                self.current_data = None
                self.update_data_info()
                self.enable_analysis_buttons(False)
                self.log("⚠️ No hay datos disponibles")
        else:
            print("❌ DataManager no disponible")
            self.current_data = None
            self.update_data_info()
            self.enable_analysis_buttons(False)

    def update_data_info(self):
        """Actualizar información de datos"""
        if self.current_data is not None:
            n_rows, n_cols = self.current_data.shape
            numeric_cols = len(self.current_data.select_dtypes(include=[np.number]).columns)

            info = f"📊 Dataset: {n_rows:,} filas × {n_cols} columnas ({numeric_cols} numéricas)"
            self.data_info_label.setText(info)

            self.variable_selection.set_data(self.current_data)
        else:
            self.data_info_label.setText("❌ No hay datos cargados")
            self.variable_selection.clear_data()

    def enable_analysis_buttons(self, enabled: bool):
        """Habilitar/deshabilitar botones de análisis"""
        if self.is_processing:
            enabled = False

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

        title_layout = QVBoxLayout()

        title = QLabel("🔍 Machine Learning No Supervisado (Secuencial)")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        title_layout.addWidget(title)

        self.data_info_label = QLabel("Verificando datos...")
        self.data_info_label.setStyleSheet("color: #666; font-size: 11px;")
        title_layout.addWidget(self.data_info_label)

        layout.addLayout(title_layout)
        layout.addStretch()

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
        """Crear panel izquierdo"""
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(400)

        content_widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 10)

        self.variable_selection = VariableSelectionWidget()
        layout.addWidget(self.variable_selection)

        self.configuration = ConfigurationWidget()
        layout.addWidget(self.configuration)

        analysis_group = QGroupBox("🚀 Análisis Disponibles")
        analysis_layout = QVBoxLayout()
        analysis_layout.setSpacing(12)

        clustering_label = QLabel("🎯 Algoritmos de Clustering")
        clustering_label.setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 13px;")
        analysis_layout.addWidget(clustering_label)

        self.kmeans_btn = QPushButton("🔹 K-Means Optimizado")
        self.kmeans_btn.setMinimumHeight(40)
        self.kmeans_btn.setToolTip("Clustering con optimización automática")
        self.kmeans_btn.clicked.connect(lambda: self.run_analysis('kmeans_optimizado'))
        analysis_layout.addWidget(self.kmeans_btn)

        self.hierarchical_btn = QPushButton("🔸 Clustering Jerárquico")
        self.hierarchical_btn.setMinimumHeight(40)
        self.hierarchical_btn.setToolTip("Clustering basado en jerarquías")
        self.hierarchical_btn.clicked.connect(lambda: self.run_analysis('clustering_jerarquico'))
        analysis_layout.addWidget(self.hierarchical_btn)

        self.dbscan_btn = QPushButton("🔺 DBSCAN")
        self.dbscan_btn.setMinimumHeight(40)
        self.dbscan_btn.setToolTip("Clustering basado en densidad")
        self.dbscan_btn.clicked.connect(lambda: self.run_analysis('dbscan'))
        analysis_layout.addWidget(self.dbscan_btn)

        pca_label = QLabel("📊 Reducción Dimensional")
        pca_label.setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 13px;")
        analysis_layout.addWidget(pca_label)

        self.pca_btn = QPushButton("🔹 PCA Avanzado")
        self.pca_btn.setMinimumHeight(40)
        self.pca_btn.setToolTip("Análisis de Componentes Principales")
        self.pca_btn.clicked.connect(lambda: self.run_analysis('pca_avanzado'))
        analysis_layout.addWidget(self.pca_btn)

        exp_label = QLabel("🔍 Análisis Exploratorio")
        exp_label.setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 13px;")
        analysis_layout.addWidget(exp_label)

        self.exploratory_btn = QPushButton("🔹 Análisis Completo")
        self.exploratory_btn.setMinimumHeight(40)
        self.exploratory_btn.setToolTip("Análisis exploratorio completo")
        self.exploratory_btn.clicked.connect(lambda: self.run_analysis('analisis_exploratorio'))
        analysis_layout.addWidget(self.exploratory_btn)

        analysis_layout.addStretch()
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

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

        header_layout = QHBoxLayout()
        log_title = QLabel("📝 Log de Actividad")
        log_title.setStyleSheet("font-weight: bold; color: #2c3e50;")
        header_layout.addWidget(log_title)

        clear_btn = QPushButton("🗑️ Limpiar")
        clear_btn.setMinimumHeight(30)
        clear_btn.clicked.connect(self.clear_log)
        header_layout.addWidget(clear_btn)

        layout.addLayout(header_layout)

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
                "Las librerías ML no están disponibles"
            )
            return

        try:
            self.log("🎲 Generando datos de demostración...")

            demo_data = generar_datos_agua_realistas(n_muestras=200, incluir_outliers=True)

            self.current_data = demo_data
            self.update_data_info()
            self.enable_analysis_buttons(True)

            self.log("✅ Datos de demostración generados")
            QMessageBox.information(
                self, "Datos Demo",
                f"Se generaron {len(demo_data)} muestras con {demo_data.shape[1]} variables"
            )

        except Exception as e:
            self.log(f"❌ Error generando datos demo: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error:\n{str(e)}")

    # ==================== EJECUCIÓN DE ANÁLISIS (SECUENCIAL) ====================

    def run_analysis(self, analysis_type: str):
        """Ejecutar análisis específico (SECUENCIAL, SIN THREADING)"""
        if not self.validate_selection():
            return

        if not ML_AVAILABLE:
            QMessageBox.critical(
                self, "Error",
                "Las librerías ML no están disponibles."
            )
            return

        if self.is_processing:
            QMessageBox.warning(self, "Procesando", "Ya hay un análisis en curso")
            return

        # Marcar como procesando
        self.is_processing = True
        self.enable_analysis_buttons(False)
        self.show_progress(True)

        # Obtener configuración
        variables = self.variable_selection.get_selected_variables()
        config = self.configuration.get_config()

        # Preparar kwargs BASE (comunes a todos)
        kwargs_base = {
            'variables': variables,
            'escalado': config['scaling_method'],
            'verbose': config['verbose'],
            'progress_callback': self.update_progress,
            'status_callback': self.update_status
        }

        # Añadir parámetros específicos según el tipo de análisis
        kwargs = kwargs_base.copy()

        if analysis_type == 'kmeans_optimizado':
            kwargs['k_range'] = config['kmeans_k_range']
            kwargs['random_state'] = config['random_state']

        elif analysis_type == 'clustering_jerarquico':
            # Clustering jerárquico NO usa random_state
            kwargs['metodos'] = [config['hierarchical_method']]
            kwargs['metricas'] = [config['hierarchical_metric']]
            kwargs['max_clusters'] = config['hierarchical_max_clusters']

        elif analysis_type == 'dbscan':
            # DBSCAN NO usa random_state
            kwargs['optimizar_parametros'] = config['dbscan_optimize']

        elif analysis_type == 'pca_avanzado':
            # PCA NO usa random_state
            kwargs['explicar_varianza_objetivo'] = config['pca_variance_threshold']

        elif analysis_type == 'analisis_exploratorio':
            # Análisis exploratorio NO usa random_state
            kwargs['handle_outliers'] = config['handle_outliers']

        self.log(f"Iniciando análisis: {analysis_type}")
        self.log(f"Variables: {len(variables)}")

        # Ejecutar análisis de forma SECUENCIAL usando QTimer
        # para no bloquear la UI completamente
        QTimer.singleShot(100, lambda: self._execute_analysis(analysis_type, kwargs))

    def _execute_analysis(self, analysis_type: str, kwargs: dict):
        """Ejecutar el análisis real"""
        try:
            self.log("Ejecutando análisis...")
            QApplication.processEvents()  # Permitir actualización de UI

            # Llamar a la función correspondiente (SECUENCIAL)
            if analysis_type == 'kmeans_optimizado':
                resultado = kmeans_optimizado_completo(self.current_data, **kwargs)
            elif analysis_type == 'clustering_jerarquico':
                resultado = clustering_jerarquico_completo(self.current_data, **kwargs)
            elif analysis_type == 'dbscan':
                resultado = dbscan_optimizado(self.current_data, **kwargs)
            elif analysis_type == 'pca_avanzado':
                resultado = pca_completo_avanzado(self.current_data, **kwargs)
            elif analysis_type == 'analisis_exploratorio':
                resultado = analisis_exploratorio_completo(self.current_data, **kwargs)
            else:
                raise ValueError(f"Tipo desconocido: {analysis_type}")

            # Procesar resultado
            self.on_analysis_finished(resultado, analysis_type)

        except Exception as e:
            self.on_analysis_error(str(e))
            print(f"Error en análisis: {e}")
            print(traceback.format_exc())

    def validate_selection(self) -> bool:
        """Validar selección de variables"""
        if self.current_data is None:
            QMessageBox.warning(self, "Sin Datos", "No hay datos cargados")
            return False

        if not self.variable_selection.is_valid_selection():
            QMessageBox.warning(
                self, "Selección Inválida",
                "Selecciona al menos 2 variables"
            )
            return False

        return True

    # ==================== CALLBACKS ====================

    def on_analysis_finished(self, results: dict, analysis_type: str):
        """Cuando termina el análisis"""
        self.show_progress(False)
        self.is_processing = False
        self.enable_analysis_buttons(True)

        # Guardar en historial
        analysis_entry = {
            'timestamp': datetime.now(),
            'type': analysis_type,
            'results': results,
            'variables': self.variable_selection.get_selected_variables(),
            'config': self.configuration.get_config()
        }
        self.analysis_history.append(analysis_entry)

        # Actualizar resultados
        self.results_widget.update_results(results, analysis_type)

        self.log("✅ Análisis completado")

    def on_analysis_error(self, error_msg: str):
        """Cuando ocurre un error"""
        self.show_progress(False)
        self.is_processing = False
        self.enable_analysis_buttons(True)

        self.log(f"❌ Error: {error_msg}")
        QMessageBox.critical(self, "Error en Análisis",
                             f"Error durante el análisis:\n\n{error_msg}")

    def update_progress(self, value: int):
        """Actualizar barra de progreso"""
        try:
            self.progress_bar.setValue(value)
            QApplication.processEvents()  # Permitir actualización de UI
        except:
            pass

    def update_status(self, message: str):
        """Actualizar estado"""
        try:
            self.log(message)
            QApplication.processEvents()  # Permitir actualización de UI
        except:
            pass

    # ==================== UTILIDADES ====================

    def show_progress(self, show: bool):
        """Mostrar/ocultar progreso"""
        self.progress_bar.setVisible(show)

        if show:
            self.progress_bar.setValue(0)

        # Deshabilitar botones durante análisis
        if not show:
            self.enable_analysis_buttons(True)

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
        help_dialog.resize(800, 600)

        layout = QVBoxLayout()

        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
            <h2>Machine Learning No Supervisado (Secuencial)</h2>

            <h3>⚡ Versión SIN Threading</h3>
            <p>Esta versión ejecuta todos los análisis de forma <b>secuencial</b>, 
            sin usar hilos paralelos. Esto garantiza máxima compatibilidad con 
            PyInstaller y evita problemas de concurrencia.</p>

            <h3>Técnicas disponibles:</h3>
            <ul>
            <li><b>K-Means:</b> Agrupa datos en K clusters esféricos</li>
            <li><b>DBSCAN:</b> Detecta clusters de densidad variable y outliers</li>
            <li><b>Clustering Jerárquico:</b> Crea jerarquías de grupos (implementación manual)</li>
            <li><b>PCA:</b> Reduce dimensionalidad manteniendo varianza</li>
            <li><b>Análisis Exploratorio:</b> Examina correlaciones y distribuciones</li>
            </ul>

            <h3>Flujo recomendado:</h3>
            <ol>
            <li>Cargar datos desde "Cargar Datos" o usar Demo</li>
            <li>Seleccionar variables (botón "Auto" para selección inteligente)</li>
            <li>Configurar parámetros según necesidades</li>
            <li>Ejecutar "Análisis Exploratorio" primero</li>
            <li>Aplicar técnicas de clustering o PCA</li>
            <li>Revisar resultados en las pestañas</li>
            </ol>

            <h3>Consejos:</h3>
            <ul>
            <li>Standard scaling recomendado para mayoría de casos</li>
            <li>Evita variables con >50% valores faltantes</li>
            <li>K-Means funciona bien con clusters esféricos</li>
            <li>DBSCAN ideal cuando no conoces el número de clusters</li>
            <li>PCA útil con muchas variables correlacionadas</li>
            <li><b>Los análisis pueden tardar más que la versión con threads</b></li>
            </ul>

            <h3>⚠️ Importante:</h3>
            <p>Durante el análisis la interfaz puede parecer bloqueada brevemente. 
            Esto es normal en la versión secuencial. El progreso se actualiza periódicamente.</p>
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
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
            }

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

            QListWidget {
                border: 2px solid #bdc3c7;
                border-radius: 6px;
                selection-background-color: #3498db;
                background-color: white;
            }

            QTextEdit {
                border: 2px solid #bdc3c7;
                border-radius: 6px;
                background-color: white;
            }
            """

        self.setStyleSheet(style)


# ==================== FUNCIÓN PRINCIPAL ====================

def main():
    """Función principal para testing independiente"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = NoSupervisadoWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()