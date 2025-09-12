"""
supervisado_window.py - Interfaz GUI Mejorada para Machine Learning Supervisado
Versión con tamaño fijo, botón de limpiar gráficas y visualizaciones completas
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
    QListWidgetItem, QSlider, QDoubleSpinBox, QScrollArea,
    QButtonGroup, QRadioButton, QDesktopWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, pyqtSlot
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor

# Importar gestor de datos compartido
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

# Importar funciones ML optimizadas
ML_AVAILABLE = False
try:
    from .ml_functions_supervisado import (
        verificar_datos,
        preparar_datos_supervisado_optimizado,
        regresion_lineal_simple,
        regresion_lineal_multiple,
        arbol_decision,
        random_forest,
        svm_modelo,
        comparar_modelos_supervisado,
        generar_visualizaciones_ml,
        visualizar_arbol_decision,
        visualizar_importancia_features,
        crear_dashboard_arbol,
        exportar_modelo,
        cargar_modelo,
        limpiar_memoria
    )

    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    ML_AVAILABLE = True
    print("✅ Librerías ML Supervisado cargadas correctamente")

except ImportError as e:
    ML_AVAILABLE = False
    print(f"❌ Error cargando ML: {e}")

# ==================== WORKER THREAD OPTIMIZADO ====================

class MLAnalysisWorker(QThread):
    """Worker optimizado para análisis ML"""
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
                raise ImportError("Librerías de ML no disponibles")

            self.log.emit(f"🚀 Iniciando {self.analysis_type}")
            self.progress.emit(10)

            result = None

            if self.analysis_type == 'regresion_simple':
                result = self._run_regresion_simple()
            elif self.analysis_type == 'regresion_multiple':
                result = self._run_regresion_multiple()
            elif self.analysis_type == 'arbol_decision':
                result = self._run_arbol_decision()
            elif self.analysis_type == 'random_forest':
                result = self._run_random_forest()
            elif self.analysis_type == 'svm':
                result = self._run_svm()
            elif self.analysis_type == 'comparar_modelos':
                result = self._run_comparar_modelos()
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

    def _run_regresion_simple(self):
        self.status.emit("Ejecutando regresión lineal simple...")
        self.progress.emit(30)
        filtered_kwargs = {k: v for k, v in self.kwargs.items()
                          if k in ['x_column', 'y_column', 'optimize_params']}
        return regresion_lineal_simple(self.data, **filtered_kwargs)

    def _run_regresion_multiple(self):
        self.status.emit("Ejecutando regresión lineal múltiple...")
        self.progress.emit(30)
        filtered_kwargs = {k: v for k, v in self.kwargs.items()
                          if k in ['target_column', 'feature_columns', 'regularization',
                                   'alpha', 'optimize_params']}
        return regresion_lineal_multiple(self.data, **filtered_kwargs)

    def _run_arbol_decision(self):
        self.status.emit("Entrenando árbol de decisión...")
        self.progress.emit(30)
        filtered_kwargs = {k: v for k, v in self.kwargs.items()
                          if k in ['target_column', 'feature_columns', 'max_depth',
                                   'min_samples_split', 'min_samples_leaf', 'optimize_params']}
        return arbol_decision(self.data, **filtered_kwargs)

    def _run_random_forest(self):
        self.status.emit("Entrenando Random Forest...")
        self.progress.emit(30)
        filtered_kwargs = {k: v for k, v in self.kwargs.items()
                          if k in ['target_column', 'feature_columns', 'n_estimators',
                                   'max_depth', 'max_features', 'optimize_params', 'n_jobs']}
        return random_forest(self.data, **filtered_kwargs)

    def _run_svm(self):
        self.status.emit("Entrenando SVM...")
        self.progress.emit(30)
        filtered_kwargs = {k: v for k, v in self.kwargs.items()
                          if k in ['target_column', 'feature_columns', 'kernel',
                                   'C', 'gamma', 'optimize_params']}
        return svm_modelo(self.data, **filtered_kwargs)

    def _run_comparar_modelos(self):
        self.status.emit("Comparando modelos...")
        self.progress.emit(20)
        filtered_kwargs = {k: v for k, v in self.kwargs.items()
                          if k in ['target_column', 'feature_columns', 'modelos',
                                   'optimize_all', 'cv_folds', 'n_jobs']}
        return comparar_modelos_supervisado(self.data, **filtered_kwargs)

# ==================== WIDGET DE SELECCIÓN DE VARIABLES MEJORADO ====================

class VariableSelectionWidget(QWidget):
    """Widget mejorado para selección de variables con etiquetas Y/X"""
    variables_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.data = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(5, 5, 5, 5)

        # Variable dependiente (Y) - Variable Objetivo
        target_group = QGroupBox("🎯 Variable Dependiente (Y)")
        target_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #2c3e50;
                font-size: 12px;
                padding-top: 15px;
            }
            QGroupBox::title {
                padding: 0 8px;
            }
        """)
        target_layout = QVBoxLayout()
        target_layout.setSpacing(5)

        self.target_combo = QComboBox()
        self.target_combo.setMinimumHeight(30)  # Altura mínima para el combo
        self.target_combo.currentTextChanged.connect(self._on_target_changed)
        target_layout.addWidget(self.target_combo)

        self.target_info_label = QLabel("Variable a predecir (Y)")
        self.target_info_label.setStyleSheet("color: #666; font-size: 10px;")
        self.target_info_label.setWordWrap(True)
        target_layout.addWidget(self.target_info_label)

        target_group.setLayout(target_layout)
        layout.addWidget(target_group)

        # Variables independientes (X) - Variables Predictoras
        features_group = QGroupBox("📊 Variables Independientes (X)")
        features_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #2c3e50;
                font-size: 12px;
                padding-top: 15px;
            }
            QGroupBox::title {
                padding: 0 8px;
            }
        """)
        features_layout = QVBoxLayout()
        features_layout.setSpacing(5)

        # Controles de selección con botones más pequeños
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(5)

        self.select_all_btn = QPushButton("Todas")
        self.select_all_btn.setMaximumHeight(25)
        self.select_all_btn.clicked.connect(self._select_all_features)
        controls_layout.addWidget(self.select_all_btn)

        self.select_none_btn = QPushButton("Ninguna")
        self.select_none_btn.setMaximumHeight(25)
        self.select_none_btn.clicked.connect(self._select_none_features)
        controls_layout.addWidget(self.select_none_btn)

        self.auto_select_btn = QPushButton("🤖 Auto")
        self.auto_select_btn.setMaximumHeight(25)
        self.auto_select_btn.clicked.connect(self._auto_select_features)
        controls_layout.addWidget(self.auto_select_btn)

        features_layout.addLayout(controls_layout)

        # Lista de características con altura mínima
        self.features_list = QListWidget()
        self.features_list.setSelectionMode(QListWidget.MultiSelection)
        self.features_list.setMinimumHeight(150)  # Altura mínima para la lista
        self.features_list.itemSelectionChanged.connect(self._on_features_changed)
        features_layout.addWidget(self.features_list)

        self.features_info_label = QLabel("Variables explicativas (X)")
        self.features_info_label.setStyleSheet("color: #666; font-size: 10px;")
        self.features_info_label.setWordWrap(True)
        features_layout.addWidget(self.features_info_label)

        features_group.setLayout(features_layout)
        layout.addWidget(features_group)

        self.setLayout(layout)

    def set_data(self, data: pd.DataFrame):
        """Establecer datos"""
        self.data = data
        self._update_variables()

    def clear_data(self):
        """Limpiar datos"""
        self.data = None
        self.target_combo.clear()
        self.features_list.clear()
        self.target_info_label.setText("Sin datos disponibles")
        self.features_info_label.setText("0 variables disponibles")

    def _update_variables(self):
        """Actualizar lista de variables"""
        if self.data is None:
            return

        self.target_combo.clear()
        self.features_list.clear()

        # Obtener columnas numéricas
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()

        # Obtener columnas categóricas con pocas categorías
        categorical_cols = []
        for col in self.data.select_dtypes(include=['object', 'category']).columns:
            if self.data[col].nunique() <= 10:
                categorical_cols.append(col)

        all_cols = numeric_cols + categorical_cols

        # Actualizar target combo
        self.target_combo.addItems(all_cols)

        # Seleccionar target por defecto
        default_targets = ['WQI', 'Calidad', 'Quality', 'Target', 'y', 'Y', 'Indice_Calidad']
        for target in default_targets:
            if target in all_cols:
                self.target_combo.setCurrentText(target)
                break

        self._update_features_list()

    def _update_features_list(self):
        """Actualizar lista de características"""
        self.features_list.clear()

        if self.data is None:
            return

        target = self.target_combo.currentText()
        if not target:
            return

        # Columnas numéricas (excluyendo target)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != target]

        for col in numeric_cols:
            col_data = self.data[col]
            missing_pct = (col_data.isnull().sum() / len(col_data)) * 100

            item_text = f"{col} (missing: {missing_pct:.1f}%)"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, col)

            # Colorear según calidad
            if missing_pct > 50:
                item.setForeground(QColor('red'))
            elif missing_pct > 20:
                item.setForeground(QColor('orange'))

            self.features_list.addItem(item)

            # Seleccionar automáticamente si tiene pocos missing
            if missing_pct < 20:
                item.setSelected(True)

    def _on_target_changed(self):
        """Cuando cambia la variable objetivo"""
        self._update_features_list()
        self._update_target_info()
        self.variables_changed.emit()

    def _on_features_changed(self):
        """Cuando cambian las características"""
        self._update_features_info()
        self.variables_changed.emit()

    def _update_target_info(self):
        """Actualizar información de target"""
        target = self.target_combo.currentText()
        if not target or self.data is None:
            self.target_info_label.setText("Selecciona una variable objetivo (Y)")
            return

        target_data = self.data[target]
        n_unique = target_data.nunique()

        if target_data.dtype in [np.float64, np.float32, np.int64, np.int32]:
            info = f"Variable Y: Numérica con {n_unique} valores únicos"
            if n_unique <= 10:
                info += " → Clasificación sugerida"
            else:
                info += " → Regresión sugerida"
        else:
            info = f"Variable Y: Categórica con {n_unique} categorías → Clasificación"

        self.target_info_label.setText(info)

    def _update_features_info(self):
        """Actualizar información de características"""
        n_selected = len(self.get_selected_features())
        n_total = self.features_list.count()
        self.features_info_label.setText(f"{n_selected} de {n_total} variables X seleccionadas")

    def _select_all_features(self):
        """Seleccionar todas"""
        for i in range(self.features_list.count()):
            self.features_list.item(i).setSelected(True)

    def _select_none_features(self):
        """Deseleccionar todas"""
        for i in range(self.features_list.count()):
            self.features_list.item(i).setSelected(False)

    def _auto_select_features(self):
        """Selección automática basada en correlación"""
        if not ML_AVAILABLE or self.data is None:
            QMessageBox.warning(self, "Error", "Funcionalidad no disponible")
            return

        target = self.target_combo.currentText()
        if not target:
            return

        try:
            # Calcular correlaciones
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            correlations = {}

            for col in numeric_cols:
                if col != target:
                    mask = ~(self.data[col].isnull() | self.data[target].isnull())
                    if mask.sum() > 10:
                        corr = abs(self.data.loc[mask, col].corr(self.data.loc[mask, target]))
                        correlations[col] = corr

            # Seleccionar las mejores
            sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            n_to_select = min(10, len(sorted_features))

            self._select_none_features()

            selected_count = 0
            for col, corr in sorted_features:
                if selected_count >= n_to_select:
                    break

                for i in range(self.features_list.count()):
                    item = self.features_list.item(i)
                    if item.data(Qt.UserRole) == col:
                        item.setSelected(True)
                        selected_count += 1
                        break

            QMessageBox.information(
                self, "Selección Automática",
                f"Se seleccionaron las {selected_count} variables X con mayor correlación con Y"
            )

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error en selección automática: {str(e)}")

    def get_target_variable(self) -> str:
        """Obtener variable objetivo"""
        return self.target_combo.currentText()

    def get_selected_features(self) -> list:
        """Obtener características seleccionadas"""
        features = []
        for item in self.features_list.selectedItems():
            features.append(item.data(Qt.UserRole))
        return features

    def is_valid_selection(self) -> bool:
        """Verificar si la selección es válida"""
        return bool(self.get_target_variable() and self.get_selected_features())

# ==================== WIDGET DE RESULTADOS MEJORADO ====================

class ResultsVisualizationWidget(QWidget):
    """Widget mejorado para visualización de resultados con botón de limpiar"""

    def __init__(self):
        super().__init__()
        self.current_results = None
        self.current_model = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Tabs para diferentes vistas
        self.tabs = QTabWidget()

        # Tab de métricas
        self.metrics_widget = self._create_metrics_tab()
        self.tabs.addTab(self.metrics_widget, "📊 Métricas")

        # Tab de visualizaciones
        if ML_AVAILABLE:
            self.viz_widget = self._create_viz_tab()
            self.tabs.addTab(self.viz_widget, "📈 Visualizaciones")

        # Tab de modelo
        self.model_widget = self._create_model_tab()
        self.tabs.addTab(self.model_widget, "🔧 Modelo")

        layout.addWidget(self.tabs)

        # Toolbar mejorada
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)

        self.setLayout(layout)

    def _create_metrics_tab(self) -> QWidget:
        """Crear tab de métricas"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Tabla de métricas
        self.metrics_table = QTableWidget()
        self.metrics_table.setAlternatingRowColors(True)
        layout.addWidget(QLabel("📊 Métricas de Rendimiento"))
        layout.addWidget(self.metrics_table)

        widget.setLayout(layout)
        return widget

    def _create_viz_tab(self) -> QWidget:
        """Crear tab de visualizaciones con botón de limpiar"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Controles superiores
        controls_layout = QHBoxLayout()

        self.clear_viz_btn = QPushButton("🗑️ Limpiar Gráficos")
        self.clear_viz_btn.clicked.connect(self._clear_visualizations)
        self.clear_viz_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        controls_layout.addWidget(self.clear_viz_btn)

        self.save_fig_btn = QPushButton("💾 Guardar Gráfico")
        self.save_fig_btn.clicked.connect(self._save_figure)
        controls_layout.addWidget(self.save_fig_btn)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Canvas para matplotlib
        self.figure = Figure(figsize=(16, 12))
        self.canvas = FigureCanvas(self.figure)

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        widget.setLayout(layout)
        return widget

    def _create_model_tab(self) -> QWidget:
        """Crear tab de modelo"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Información del modelo
        self.model_info_text = QTextEdit()
        self.model_info_text.setReadOnly(True)
        self.model_info_text.setMaximumHeight(200)
        layout.addWidget(QLabel("📋 Información del Modelo"))
        layout.addWidget(self.model_info_text)

        # Parámetros
        self.params_table = QTableWidget()
        self.params_table.setColumnCount(2)
        self.params_table.setHorizontalHeaderLabels(["Parámetro", "Valor"])
        layout.addWidget(QLabel("⚙️ Parámetros"))
        layout.addWidget(self.params_table)

        widget.setLayout(layout)
        return widget

    def _create_toolbar(self) -> QWidget:
        """Crear toolbar mejorada"""
        toolbar = QWidget()
        layout = QHBoxLayout()

        self.export_model_btn = QPushButton("💾 Exportar Modelo")
        self.export_model_btn.clicked.connect(self._export_model)
        layout.addWidget(self.export_model_btn)

        self.export_results_btn = QPushButton("📄 Exportar Resultados")
        self.export_results_btn.clicked.connect(self._export_results)
        layout.addWidget(self.export_results_btn)

        layout.addStretch()

        self.status_label = QLabel("Sin resultados")
        self.status_label.setStyleSheet("color: #666;")
        layout.addWidget(self.status_label)

        toolbar.setLayout(layout)
        return toolbar

    def _clear_visualizations(self):
        """Limpiar todas las visualizaciones"""
        self.figure.clear()
        self.canvas.draw()
        self.status_label.setText("📊 Gráficos limpiados")
        self.status_label.setStyleSheet("color: #3498db;")
        QMessageBox.information(self, "Gráficos Limpiados",
                               "Se han limpiado todas las visualizaciones")

    def update_results(self, results: dict, analysis_type: str):
        """Actualizar con nuevos resultados"""
        self.current_results = results
        self.current_model = results.get('modelo')

        if 'error' in results:
            self._show_error(results['error'])
            return

        self._update_metrics(results)
        self._update_model_info(results)

        if ML_AVAILABLE:
            self._update_visualization()

        self.status_label.setText(f"✅ {analysis_type} completado")
        self.status_label.setStyleSheet("color: green;")

        # Habilitar botones
        self.export_model_btn.setEnabled(self.current_model is not None)
        self.export_results_btn.setEnabled(True)

    def _update_metrics(self, results: dict):
        """Actualizar métricas"""
        metrics = results.get('metricas', {})

        if results.get('tipo') == 'comparar_modelos':
            self._show_model_comparison(results)
        else:
            self._show_individual_metrics(metrics)

    def _show_individual_metrics(self, metrics: dict):
        """Mostrar métricas individuales"""
        all_metrics = set()
        for split_metrics in metrics.values():
            if isinstance(split_metrics, dict):
                all_metrics.update(split_metrics.keys())

        self.metrics_table.setRowCount(len(all_metrics))
        self.metrics_table.setColumnCount(3)
        self.metrics_table.setHorizontalHeaderLabels(['Métrica', 'Train', 'Test'])

        for i, metric_name in enumerate(sorted(all_metrics)):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(metric_name.upper()))

            train_val = metrics.get('train', {}).get(metric_name, 'N/A')
            if isinstance(train_val, (int, float)):
                train_item = QTableWidgetItem(f"{train_val:.4f}")
            else:
                train_item = QTableWidgetItem(str(train_val))
            self.metrics_table.setItem(i, 1, train_item)

            test_val = metrics.get('test', {}).get(metric_name, 'N/A')
            if isinstance(test_val, (int, float)):
                test_item = QTableWidgetItem(f"{test_val:.4f}")
            else:
                test_item = QTableWidgetItem(str(test_val))
            self.metrics_table.setItem(i, 2, test_item)

        self.metrics_table.resizeColumnsToContents()

    def _show_model_comparison(self, results: dict):
        """Mostrar comparación de modelos"""
        ranking = results.get('ranking', [])

        self.metrics_table.setRowCount(len(ranking))
        self.metrics_table.setColumnCount(4)
        self.metrics_table.setHorizontalHeaderLabels([
            'Ranking', 'Modelo', 'Métrica', 'Score'
        ])

        for i, item in enumerate(ranking):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))

            model_item = QTableWidgetItem(item['modelo'])
            if i == 0:
                model_item.setBackground(QColor(255, 215, 0))
            self.metrics_table.setItem(i, 1, model_item)

            self.metrics_table.setItem(i, 2, QTableWidgetItem(item['metrica'].upper()))
            self.metrics_table.setItem(i, 3, QTableWidgetItem(f"{item['score']:.4f}"))

        self.metrics_table.resizeColumnsToContents()

    def _update_model_info(self, results: dict):
        """Actualizar información del modelo"""
        info_text = f"Tipo: {results.get('tipo', 'N/A')}\n"

        if results.get('es_clasificacion') is not None:
            info_text += f"Problema: {'Clasificación' if results['es_clasificacion'] else 'Regresión'}\n"

        if 'target_column' in results:
            info_text += f"Variable Y (objetivo): {results['target_column']}\n"

        if 'feature_columns' in results:
            info_text += f"Variables X (predictoras): {len(results['feature_columns'])}\n"

        self.model_info_text.setText(info_text)

        # Parámetros
        params = results.get('parametros', {})
        self.params_table.setRowCount(len(params))

        for i, (param, value) in enumerate(params.items()):
            self.params_table.setItem(i, 0, QTableWidgetItem(param))
            self.params_table.setItem(i, 1, QTableWidgetItem(str(value)))

        self.params_table.resizeColumnsToContents()

    def _update_visualization(self):
        """Actualizar visualización con gráficos mejorados"""
        if not self.current_results:
            return

        try:
            self.figure.clear()

            # Para árboles de decisión, usar las funciones especializadas
            if self.current_results.get('tipo') == 'arbol_decision':
                self.figure = crear_dashboard_arbol(
                    self.current_results,
                    max_depth_visual=3,
                    figsize=(16, 12)
                )
            else:
                # Para otros modelos, usar la función general
                self.figure = generar_visualizaciones_ml(
                    self.current_results,
                    figsize=(16, 12)
                )

            self.canvas.figure = self.figure
            self.canvas.draw()
        except Exception as e:
            print(f"Error en visualización: {e}")
            # Fallback a visualización básica
            self._show_basic_visualization()

    def _show_basic_visualization(self):
        """Visualización básica de respaldo"""
        self.figure.clear()
        ax = self.figure.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, 'Error en visualización\nVer consola para detalles',
                ha='center', va='center', transform=ax.transAxes)
        self.canvas.draw()

    def _save_figure(self):
        """Guardar figura"""
        if not hasattr(self, 'figure'):
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Guardar Gráfico",
            f"grafico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "PNG (*.png);;PDF (*.pdf)"
        )

        if filepath:
            self.figure.savefig(filepath, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Éxito", "Gráfico guardado correctamente")

    def _export_model(self):
        """Exportar modelo"""
        if not self.current_model or not ML_AVAILABLE:
            QMessageBox.warning(self, "Error", "No hay modelo para exportar")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Exportar Modelo",
            f"modelo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
            "Pickle (*.pkl)"
        )

        if filepath:
            try:
                if exportar_modelo(self.current_model, filepath):
                    QMessageBox.information(self, "Éxito", "Modelo exportado correctamente")
                else:
                    QMessageBox.warning(self, "Error", "No se pudo exportar el modelo")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al exportar: {str(e)}")

    def _export_results(self):
        """Exportar resultados"""
        if not self.current_results:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Exportar Resultados",
            f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON (*.json)"
        )

        if filepath:
            try:
                export_data = {}
                for key, value in self.current_results.items():
                    if key != 'modelo':
                        if isinstance(value, np.ndarray):
                            export_data[key] = value.tolist()
                        elif isinstance(value, pd.DataFrame):
                            export_data[key] = value.to_dict()
                        else:
                            export_data[key] = value

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)

                QMessageBox.information(self, "Éxito", "Resultados exportados correctamente")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al exportar: {str(e)}")

    def _show_error(self, error_msg: str):
        """Mostrar error"""
        self.model_info_text.setText(f"❌ Error: {error_msg}")
        self.metrics_table.setRowCount(0)
        self.params_table.setRowCount(0)
        self.status_label.setText("❌ Error en análisis")
        self.status_label.setStyleSheet("color: red;")

# ==================== VENTANA PRINCIPAL MEJORADA ====================

class SupervisadoWindow(QWidget):
    """Ventana principal mejorada con tamaño fijo"""

    # Tamaño fijo para todas las pantallas
    WINDOW_WIDTH = 1600
    WINDOW_HEIGHT = 900

    def __init__(self):
        QWidget.__init__(self)
        print("🚀 SupervisadoWindow: Inicializando...")

        self.current_data = None
        self.current_worker = None
        self.analysis_history = []

        # Registrar como observador del DataManager
        if DATA_MANAGER_AVAILABLE:
            dm = get_data_manager()
            if dm is not None:
                dm.add_observer(self)
                print("✅ SupervisadoWindow: Registrada como observador del DataManager")

        self.setup_ui()
        self.center_window()
        self.check_data_availability()
        print("✅ SupervisadoWindow: Inicialización completada")

    def center_window(self):
        """Centrar ventana en la pantalla con tamaño mínimo y redimensionable"""
        # Cambiar de setFixedSize a setMinimumSize
        self.setMinimumSize(1200, 700)  # Tamaño mínimo
        self.resize(1600, 900)  # Tamaño inicial

        # Obtener el centro de la pantalla
        screen = QDesktopWidget().screenGeometry()
        x = (screen.width() - 1600) // 2
        y = (screen.height() - 900) // 2

        self.move(x, y)

    def setup_ui(self):
        """Configurar interfaz de usuario redimensionable"""
        self.setWindowTitle("🤖 Machine Learning Supervisado")

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

        # Contenido principal con splitter
        content_splitter = QSplitter(Qt.Horizontal)

        # Panel izquierdo
        left_panel = self.create_left_panel()
        content_splitter.addWidget(left_panel)

        # Panel derecho
        self.results_widget = ResultsVisualizationWidget()
        content_splitter.addWidget(self.results_widget)

        # Proporciones responsivas: 30% izquierda, 70% derecha
        content_splitter.setSizes([400, 1000])
        content_splitter.setStretchFactor(0, 0)  # Panel izquierdo no se estira mucho
        content_splitter.setStretchFactor(1, 1)  # Panel derecho se estira más

        # Establecer tamaños mínimos para evitar que se colapsen
        content_splitter.setChildrenCollapsible(False)

        main_layout.addWidget(content_splitter)

        # Log con altura fija pero redimensionable
        log_widget = self.create_log_widget()
        main_layout.addWidget(log_widget)

        # Establecer proporciones del layout principal
        main_layout.setStretchFactor(header, 0)  # Header fijo
        main_layout.setStretchFactor(self.progress_bar, 0)  # Progress bar fijo
        main_layout.setStretchFactor(content_splitter, 1)  # Contenido principal se expande
        main_layout.setStretchFactor(log_widget, 0)  # Log con altura fija

        self.setLayout(main_layout)
        self.apply_styles()


    def update(self, event_type: str = ""):
        """Método llamado por el DataManager cuando los datos cambian"""
        print(f"🔔 SupervisadoWindow: Recibida notificación '{event_type}'")

        if event_type in ['data_changed', 'session_imported']:
            self.check_data_availability()
        elif event_type == 'data_cleared':
            self.current_data = None
            self.update_data_info()
            self.enable_analysis_buttons(False)
            self.log("🗑️ Datos limpiados del sistema")

    def check_data_availability(self):
        """Verificar disponibilidad de datos"""
        if DATA_MANAGER_AVAILABLE:
            if has_shared_data():
                self.current_data = get_shared_data()
                self.update_data_info()
                self.enable_analysis_buttons(True)
                self.log("✅ Datos cargados desde el sistema")
            else:
                self.current_data = None
                self.update_data_info()
                self.enable_analysis_buttons(False)
                self.log("⚠️ No hay datos disponibles")

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
            self.reg_simple_btn, self.reg_multiple_btn, self.tree_btn,
            self.forest_btn, self.svm_btn, self.compare_btn
        ]
        for btn in buttons:
            btn.setEnabled(enabled)

    def create_header(self) -> QWidget:
        """Crear header de la ventana con altura dinámica"""
        header = QFrame()
        header.setFrameStyle(QFrame.Box)
        # Cambiar de setMaximumHeight a setMinimumHeight
        header.setMinimumHeight(100)  # Altura mínima más generosa
        header.setMaximumHeight(120)  # Permitir un poco más de altura

        layout = QHBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)  # Añadir márgenes

        # Información del título
        title_layout = QVBoxLayout()

        title = QLabel("🤖 Machine Learning Supervisado")
        # Ajustar el estilo del título
        title.setStyleSheet("""
            font-size: 18px; 
            font-weight: bold; 
            color: #2c3e50;
            margin: 5px 0px;
        """)
        title.setWordWrap(True)  # Permitir que se ajuste el texto
        title_layout.addWidget(title)

        self.data_info_label = QLabel("Verificando datos...")
        self.data_info_label.setStyleSheet("""
            color: #666; 
            font-size: 11px;
            margin: 2px 0px;
        """)
        self.data_info_label.setWordWrap(True)  # Permitir ajuste de línea
        title_layout.addWidget(self.data_info_label)

        layout.addLayout(title_layout)
        layout.addStretch()

        # Botones de acción en layout vertical para ahorrar espacio horizontal
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(5)

        buttons_row1 = QHBoxLayout()
        self.refresh_btn = QPushButton("🔄 Actualizar")
        self.refresh_btn.clicked.connect(self.check_data_availability)
        buttons_row1.addWidget(self.refresh_btn)

        self.history_btn = QPushButton("📜 Historial")
        self.history_btn.clicked.connect(self.show_history)
        buttons_row1.addWidget(self.history_btn)

        buttons_row2 = QHBoxLayout()
        self.help_btn = QPushButton("❓ Ayuda")
        self.help_btn.clicked.connect(self.show_help)
        buttons_row2.addWidget(self.help_btn)

        buttons_layout.addLayout(buttons_row1)
        buttons_layout.addLayout(buttons_row2)

        layout.addLayout(buttons_layout)

        header.setLayout(layout)
        return header

    def create_left_panel(self) -> QWidget:
        """Crear panel izquierdo de configuración con proporciones mejoradas"""
        panel = QWidget()
        panel.setMinimumWidth(350)  # Ancho mínimo
        panel.setMaximumWidth(450)  # Ancho máximo

        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(5, 5, 5, 5)

        # Widget de selección de variables (debe ocupar más espacio)
        self.variable_selection = VariableSelectionWidget()
        self.variable_selection.variables_changed.connect(self.on_variables_changed)
        layout.addWidget(self.variable_selection, 3)  # Factor de stretch 3

        # Opciones de análisis (compacto)
        options_group = QGroupBox("⚙️ Opciones de Análisis")
        options_group.setMaximumHeight(120)
        options_layout = QVBoxLayout()
        options_layout.setSpacing(5)

        self.optimize_checkbox = QCheckBox("Optimizar hiperparámetros automáticamente")
        self.optimize_checkbox.setChecked(True)
        options_layout.addWidget(self.optimize_checkbox)

        # Layout horizontal para los spinboxes
        params_layout = QHBoxLayout()

        self.cv_folds_spin = QSpinBox()
        self.cv_folds_spin.setRange(3, 10)
        self.cv_folds_spin.setValue(5)
        self.cv_folds_spin.setPrefix("CV: ")
        self.cv_folds_spin.setMaximumWidth(80)
        params_layout.addWidget(QLabel("Folds:"))
        params_layout.addWidget(self.cv_folds_spin)

        self.test_size_spin = QDoubleSpinBox()
        self.test_size_spin.setRange(10, 50)
        self.test_size_spin.setValue(20)
        self.test_size_spin.setSingleStep(5)
        self.test_size_spin.setSuffix("%")
        self.test_size_spin.setDecimals(0)
        self.test_size_spin.setMaximumWidth(80)
        params_layout.addWidget(QLabel("Test:"))
        params_layout.addWidget(self.test_size_spin)

        params_layout.addStretch()
        options_layout.addLayout(params_layout)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group, 0)  # Sin stretch

        # Botones de análisis
        analysis_group = QGroupBox("🚀 Ejecutar Análisis")
        analysis_layout = QVBoxLayout()
        analysis_layout.setSpacing(8)

        # Grid de modelos con botones más compactos
        models_layout = QGridLayout()
        models_layout.setSpacing(5)

        button_style = """
            QPushButton {
                font-size: 11px;
                padding: 6px 8px;
                min-height: 30px;
            }
        """

        self.reg_simple_btn = QPushButton("📊 Regresión Simple")
        self.reg_simple_btn.setStyleSheet(button_style)
        self.reg_simple_btn.clicked.connect(lambda: self.run_analysis('regresion_simple'))
        models_layout.addWidget(self.reg_simple_btn, 0, 0)

        self.reg_multiple_btn = QPushButton("📈 Regresión Múltiple")
        self.reg_multiple_btn.setStyleSheet(button_style)
        self.reg_multiple_btn.clicked.connect(lambda: self.run_analysis('regresion_multiple'))
        models_layout.addWidget(self.reg_multiple_btn, 0, 1)

        self.tree_btn = QPushButton("🌳 Árbol Decisión")
        self.tree_btn.setStyleSheet(button_style)
        self.tree_btn.clicked.connect(lambda: self.run_analysis('arbol_decision'))
        models_layout.addWidget(self.tree_btn, 1, 0)

        self.forest_btn = QPushButton("🌲 Random Forest")
        self.forest_btn.setStyleSheet(button_style)
        self.forest_btn.clicked.connect(lambda: self.run_analysis('random_forest'))
        models_layout.addWidget(self.forest_btn, 1, 1)

        self.svm_btn = QPushButton("📷 SVM")
        self.svm_btn.setStyleSheet(button_style)
        self.svm_btn.clicked.connect(lambda: self.run_analysis('svm'))
        models_layout.addWidget(self.svm_btn, 2, 0)

        analysis_layout.addLayout(models_layout)

        # Botón de comparación
        self.compare_btn = QPushButton("⚖️ Comparar Todos los Modelos")
        self.compare_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-weight: bold;
                padding: 8px;
                font-size: 12px;
                min-height: 35px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        self.compare_btn.clicked.connect(self.compare_models)
        analysis_layout.addWidget(self.compare_btn)

        # Botón cancelar
        self.cancel_btn = QPushButton("❌ Cancelar Análisis")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_analysis)
        analysis_layout.addWidget(self.cancel_btn)

        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group, 1)  # Factor de stretch 1

        # Eliminar addStretch() para mejor control del espacio
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

    def run_analysis(self, analysis_type: str):
        """Ejecutar análisis específico"""
        if not self.validate_selection():
            return

        if not ML_AVAILABLE:
            QMessageBox.critical(
                self, "Error",
                "Las librerías de Machine Learning no están disponibles"
            )
            return

        target = self.variable_selection.get_target_variable()
        features = self.variable_selection.get_selected_features()

        base_kwargs = {
            'target_column': target,
            'feature_columns': features,
            'optimize_params': self.optimize_checkbox.isChecked()
        }

        if analysis_type == 'regresion_simple':
            if len(features) != 1:
                QMessageBox.warning(
                    self, "Error",
                    "La regresión simple requiere exactamente una variable X"
                )
                return
            kwargs = {
                'x_column': features[0],
                'y_column': target,
                'optimize_params': self.optimize_checkbox.isChecked()
            }
        else:
            kwargs = base_kwargs

        self.show_progress(True)
        self.log(f"🚀 Iniciando análisis: {analysis_type}")

        self.current_worker = MLAnalysisWorker(analysis_type, self.current_data, **kwargs)

        self.current_worker.progress.connect(self.progress_bar.setValue)
        self.current_worker.status.connect(self.log)
        self.current_worker.finished.connect(self.on_analysis_finished)
        self.current_worker.error.connect(self.on_analysis_error)
        self.current_worker.log.connect(self.log)

        self.current_worker.start()

    def compare_models(self):
        """Comparar múltiples modelos"""
        if not self.validate_selection():
            return

        if not ML_AVAILABLE:
            QMessageBox.critical(self, "Error", "Funcionalidad ML no disponible")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Seleccionar Modelos")
        dialog.setModal(True)
        dialog.resize(400, 300)

        layout = QVBoxLayout()

        layout.addWidget(QLabel("Selecciona los modelos a comparar:"))

        model_checks = {}
        models_info = [
            ('linear', 'Regresión Lineal / Logística'),
            ('tree', 'Árbol de Decisión'),
            ('forest', 'Random Forest'),
            ('svm', 'Support Vector Machine'),
        ]

        for model_id, model_name in models_info:
            check = QCheckBox(model_name)
            check.setChecked(True)
            model_checks[model_id] = check
            layout.addWidget(check)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        dialog.setLayout(layout)

        if dialog.exec_() == QDialog.Accepted:
            selected_models = [model for model, check in model_checks.items() if check.isChecked()]

            if not selected_models:
                QMessageBox.warning(self, "Error", "Selecciona al menos un modelo")
                return

            target = self.variable_selection.get_target_variable()
            features = self.variable_selection.get_selected_features()

            kwargs = {
                'target_column': target,
                'feature_columns': features,
                'modelos': selected_models,
                'optimize_all': self.optimize_checkbox.isChecked(),
                'cv_folds': self.cv_folds_spin.value(),
                'n_jobs': -1
            }

            self.show_progress(True)
            self.log(f"🚀 Comparando {len(selected_models)} modelos...")

            self.current_worker = MLAnalysisWorker('comparar_modelos', self.current_data, **kwargs)
            self.current_worker.progress.connect(self.progress_bar.setValue)
            self.current_worker.status.connect(self.log)
            self.current_worker.finished.connect(self.on_analysis_finished)
            self.current_worker.error.connect(self.on_analysis_error)
            self.current_worker.log.connect(self.log)
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
                "Selecciona una variable Y (objetivo) y al menos una variable X (predictora)"
            )
            return False

        if ML_AVAILABLE:
            target = self.variable_selection.get_target_variable()
            features = self.variable_selection.get_selected_features()

            try:
                verification = verificar_datos(self.current_data, target, features)

                if not verification['valid']:
                    QMessageBox.critical(
                        self, "Datos Inválidos",
                        "Problemas con los datos:\n" + "\n".join(verification['issues'])
                    )
                    return False

                if verification['warnings']:
                    reply = QMessageBox.warning(
                        self, "Advertencias",
                        "Se encontraron advertencias en los datos:\n" +
                        "\n".join(verification['warnings'][:3]) +
                        "\n\n¿Continuar?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No
                    )
                    if reply == QMessageBox.No:
                        return False

            except Exception as e:
                self.log(f"⚠️ Error en verificación: {str(e)}")

        return True

    @pyqtSlot(dict)
    def on_analysis_finished(self, results: dict):
        """Cuando termina el análisis"""
        self.show_progress(False)

        # Guardar en historial
        analysis_entry = {
            'timestamp': datetime.now(),
            'type': self.current_worker.analysis_type if self.current_worker else 'unknown',
            'results': results,
            'target': self.variable_selection.get_target_variable(),
            'features': self.variable_selection.get_selected_features()
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
        pass

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

    def show_history(self):
        """Mostrar historial de análisis"""
        if not self.analysis_history:
            QMessageBox.information(self, "Historial", "No hay análisis previos")
            return

        history_text = "📜 Historial de Análisis:\n\n"
        for i, entry in enumerate(reversed(self.analysis_history[-10:])):
            history_text += f"{i+1}. {entry['timestamp'].strftime('%H:%M:%S')} - "
            history_text += f"{entry['type']} (Variable Y: {entry['target']})\n"

        QMessageBox.information(self, "Historial", history_text)

    def show_help(self):
        """Mostrar ayuda"""
        help_dialog = QDialog(self)
        help_dialog.setWindowTitle("Ayuda - Machine Learning Supervisado")
        help_dialog.setModal(True)
        help_dialog.resize(700, 600)

        layout = QVBoxLayout()

        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
        <h2>🤖 Machine Learning Supervisado</h2>
        
        <h3>📌 Conceptos Básicos:</h3>
        <ul>
        <li><b>Variable Y (Dependiente/Objetivo):</b> Lo que queremos predecir</li>
        <li><b>Variables X (Independientes/Predictoras):</b> Las variables que usamos para hacer la predicción</li>
        </ul>
        
        <h3>🔄 Pasos para usar:</h3>
        <ol>
        <li>Asegúrate de que los datos estén cargados desde el módulo "Cargar Datos"</li>
        <li>Selecciona la variable Y (lo que quieres predecir)</li>
        <li>Selecciona las variables X (predictoras)</li>
        <li>Configura las opciones de análisis si lo deseas</li>
        <li>Ejecuta el modelo deseado</li>
        </ol>
        
        <h3>📊 Tipos de Problemas:</h3>
        <ul>
        <li><b>Regresión:</b> Predecir valores numéricos continuos (ej: precio, temperatura)</li>
        <li><b>Clasificación:</b> Predecir categorías o clases (ej: si/no, alto/medio/bajo)</li>
        </ul>
        
        <h3>🔧 Modelos Disponibles:</h3>
        <ul>
        <li><b>Regresión Simple:</b> Relación lineal entre una X y una Y</li>
        <li><b>Regresión Múltiple:</b> Múltiples X predicen una Y</li>
        <li><b>Árbol de Decisión:</b> Crea reglas de decisión en forma de árbol</li>
        <li><b>Random Forest:</b> Conjunto de árboles de decisión</li>
        <li><b>SVM:</b> Support Vector Machine para clasificación/regresión</li>
        </ul>
        
        <h3>📈 Visualizaciones:</h3>
        <ul>
        <li><b>Árbol de Decisión:</b> Muestra la estructura del árbol con nodos</li>
        <li><b>Random Forest:</b> Muestra matriz de confusión, métricas y comparaciones</li>
        <li><b>SVM:</b> Muestra importancia aproximada de características</li>
        <li>Usa el botón "🗑️ Limpiar Gráficos" para limpiar las visualizaciones</li>
        </ul>
        
        <h3>💡 Consejos:</h3>
        <ul>
        <li>Usa "Selección Automática" para elegir las mejores variables X</li>
        <li>Activa "Optimizar hiperparámetros" para mejores resultados</li>
        <li>Compara múltiples modelos para encontrar el mejor</li>
        <li>Revisa las métricas para evaluar el rendimiento</li>
        <li>Para clasificación: Accuracy, Precision, Recall, F1-Score</li>
        <li>Para regresión: R², RMSE, MAE</li>
        </ul>
        
        <h3>⚠️ Nota sobre el tamaño de ventana:</h3>
        <p>La ventana tiene un tamaño fijo de 1600x900 píxeles para garantizar
        una visualización consistente en todas las pantallas.</p>
        """)

        layout.addWidget(help_text)

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
        
        /* Lists and Tables */
        QListWidget, QTableWidget {
            border: 2px solid #bdc3c7;
            border-radius: 6px;
            background-color: white;
            alternate-background-color: #f8f9fa;
        }
        
        QListWidget::item:selected {
            background-color: #3498db;
            color: white;
        }
        
        /* Text areas */
        QTextEdit {
            border: 2px solid #bdc3c7;
            border-radius: 6px;
            background-color: white;
        }
        
        /* Combo boxes and spin boxes */
        QComboBox, QSpinBox, QDoubleSpinBox {
            border: 2px solid #bdc3c7;
            border-radius: 4px;
            padding: 4px 8px;
            background-color: white;
        }
        
        QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
            border-color: #3498db;
        }
        """

        self.setStyleSheet(style)

    def closeEvent(self, event):
        """Manejar cierre de la ventana"""
        # Cancelar worker si está ejecutándose
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.cancel()
            self.current_worker.terminate()
            self.current_worker.wait()

        # Remover como observador
        if DATA_MANAGER_AVAILABLE:
            dm = get_data_manager()
            if dm is not None:
                dm.remove_observer(self)

        # Limpiar memoria ML
        if ML_AVAILABLE:
            try:
                limpiar_memoria()
            except:
                pass

        event.accept()

# ==================== FUNCIÓN PRINCIPAL PARA TESTING ====================

def main():
    """Función principal para testing"""
    app = QApplication(sys.argv)

    # Crear ventana con tamaño fijo
    window = SupervisadoWindow()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()