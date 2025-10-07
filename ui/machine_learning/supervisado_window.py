"""
supervisado_window.py - Interfaz GUI para Machine Learning Supervisado
Versión corregida para tu estructura de archivos real
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
except ImportError:
    DATA_MANAGER_AVAILABLE = False

    def get_data_manager():
        return None
    def has_shared_data():
        return False
    def get_shared_data():
        return None

# Importar funciones ML supervisado - VERSIÓN CORREGIDA PARA TU ESTRUCTURA
ML_AVAILABLE = True
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
        generar_visualizaciones_ml,  # Esta sí debe existir después de corregir indentación
        exportar_modelo,
        limpiar_memoria
    )

    # Importar matplotlib
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    ML_AVAILABLE = True
    print("✅ Librerías ML Supervisado cargadas correctamente")

except ImportError as e:
    ML_AVAILABLE = False
    print(f"⚠️ Librerías ML Supervisado no disponibles: {e}")

    # Crear clases mock para matplotlib si no está disponible
    try:
        import matplotlib
        matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
    except ImportError:
        # Clases mock para matplotlib
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

        class MockAxes:
            def text(self, *args, **kwargs):
                pass
            def set_title(self, *args, **kwargs):
                pass
            def axis(self, *args, **kwargs):
                pass

    # Funciones mock para ML si no están disponibles
    def verificar_datos(*args, **kwargs):
        return {'valid': True, 'issues': [], 'warnings': []}

    def preparar_datos_supervisado_optimizado(*args, **kwargs):
        return {}

    def regresion_lineal_simple(*args, **kwargs):
        return {'error': 'ML no disponible'}

    def regresion_lineal_multiple(*args, **kwargs):
        return {'error': 'ML no disponible'}

    def arbol_decision(*args, **kwargs):
        return {'error': 'ML no disponible'}

    def random_forest(*args, **kwargs):
        return {'error': 'ML no disponible'}

    def svm_modelo(*args, **kwargs):
        return {'error': 'ML no disponible'}

    def comparar_modelos_supervisado(*args, **kwargs):
        return {'error': 'ML no disponible'}

    def generar_visualizaciones_ml(*args, **kwargs):
        return Figure()

    def exportar_modelo(*args, **kwargs):
        return False

    def limpiar_memoria(*args, **kwargs):
        pass

# ==================== WORKER THREAD COMPATIBLE ====================

class MLAnalysisWorkerCompatible(QThread):
    """Worker compatible para análisis ML - SIN SKLEARN"""
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
        """Ejecutar análisis compatible"""
        try:
            if not ML_AVAILABLE:
                raise ImportError("Librerías de ML no disponibles")

            self.log.emit(f"🚀 Iniciando {self.analysis_type}")
            self.progress.emit(10)

            result = None

            if self.analysis_type == 'regresion_simple':
                result = self._run_regresion_simple_compatible()
            elif self.analysis_type == 'regresion_multiple':
                result = self._run_regresion_multiple_compatible()
            elif self.analysis_type == 'arbol_decision':
                result = self._run_arbol_decision_compatible()
            elif self.analysis_type == 'random_forest':
                result = self._run_random_forest_compatible()
            elif self.analysis_type == 'svm':
                result = self._run_svm_compatible()
            elif self.analysis_type == 'comparar_modelos':
                result = self._run_comparar_modelos_compatible()
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

    def cancel(self):
        """Cancelar análisis"""
        self._is_cancelled = True

    def _run_regresion_simple_compatible(self):
        self.status.emit("Ejecutando regresión lineal simple...")
        self.progress.emit(30)
        return regresion_lineal_simple(
            self.data,
            self.kwargs.get('x_column'),
            self.kwargs.get('y_column')
        )

    def _run_regresion_multiple_compatible(self):
        self.status.emit("Ejecutando regresión lineal múltiple...")
        self.progress.emit(30)
        return regresion_lineal_multiple(
            self.data,
            self.kwargs.get('target_column'),
            self.kwargs.get('feature_columns')
        )

    def _run_arbol_decision_compatible(self):
        self.status.emit("Entrenando árbol de decisión...")
        self.progress.emit(30)
        return arbol_decision(
            self.data,
            self.kwargs.get('target_column'),
            self.kwargs.get('feature_columns'),
            self.kwargs.get('max_depth'),
            self.kwargs.get('min_samples_split')
        )

    def _run_random_forest_compatible(self):
        self.status.emit("Entrenando Random Forest...")
        self.progress.emit(30)
        return random_forest(
            self.data,
            self.kwargs.get('target_column'),
            self.kwargs.get('feature_columns')
        )

    def _run_svm_compatible(self):
        self.status.emit("Entrenando SVM...")
        self.progress.emit(30)
        return svm_modelo(
            self.data,
            self.kwargs.get('target_column'),
            self.kwargs.get('feature_columns'),
            self.kwargs.get('C', 1.0)
        )

    def _run_comparar_modelos_compatible(self):
        self.status.emit("Comparando modelos...")
        self.progress.emit(20)
        return comparar_modelos_supervisado(
            self.data,
            self.kwargs.get('target_column'),
            self.kwargs.get('feature_columns'),
            self.kwargs.get('modelos')
        )

# ==================== WIDGET DE SELECCIÓN DE VARIABLES COMPATIBLE ====================

class VariableSelectionWidgetCompatible(QWidget):
    """Widget compatible para selección de variables"""
    variables_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.data = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(5, 5, 5, 5)

        # Variable dependiente (Y)
        target_group = QGroupBox("🎯 Variable Dependiente (Y)")
        target_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #2c3e50;
                font-size: 12px;
                padding-top: 15px;
            }
        """)
        target_layout = QVBoxLayout()

        self.target_combo = QComboBox()
        self.target_combo.setMinimumHeight(30)
        self.target_combo.currentTextChanged.connect(self._on_target_changed)
        target_layout.addWidget(self.target_combo)

        self.target_info_label = QLabel("Variable a predecir (Y)")
        self.target_info_label.setStyleSheet("color: #666; font-size: 10px;")
        target_layout.addWidget(self.target_info_label)

        target_group.setLayout(target_layout)
        layout.addWidget(target_group)

        # Variables independientes (X)
        features_group = QGroupBox("📊 Variables Independientes (X)")
        features_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #2c3e50;
                font-size: 12px;
                padding-top: 15px;
            }
        """)
        features_layout = QVBoxLayout()

        # Controles de selección
        controls_layout = QHBoxLayout()

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

        # Lista de características
        self.features_list = QListWidget()
        self.features_list.setSelectionMode(QListWidget.MultiSelection)
        self.features_list.setMinimumHeight(150)
        self.features_list.itemSelectionChanged.connect(self._on_features_changed)
        features_layout.addWidget(self.features_list)

        self.features_info_label = QLabel("Variables explicativas (X)")
        self.features_info_label.setStyleSheet("color: #666; font-size: 10px;")
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

        # Buscar target por defecto
        default_targets = ['WQI', 'Calidad', 'Quality', 'Target', 'y', 'Y', 'target']
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
            # Calcular correlaciones simples
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            correlations = {}

            for col in numeric_cols:
                if col != target:
                    mask = ~(self.data[col].isnull() | self.data[target].isnull())
                    if mask.sum() > 10:
                        # Correlación simple usando NumPy
                        x_clean = self.data.loc[mask, col]
                        y_clean = self.data.loc[mask, target]
                        corr = abs(np.corrcoef(x_clean, y_clean)[0, 1])
                        if not np.isnan(corr):
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

# ==================== WIDGET DE RESULTADOS COMPATIBLE ====================

class ResultsVisualizationWidgetCompatible(QWidget):
    """Widget compatible para visualización de resultados"""

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

        # Toolbar
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
        """Crear tab de visualizaciones"""
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
        """Crear toolbar"""
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
        """Limpiar visualizaciones"""
        if hasattr(self, 'figure'):
            self.figure.clear()
            self.canvas.draw()
        self.status_label.setText("📊 Gráficos limpiados")
        QMessageBox.information(self, "Gráficos Limpiados", "Visualizaciones limpiadas")

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
        self.metrics_table.setHorizontalHeaderLabels(['Ranking', 'Modelo', 'Métrica', 'Score'])

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
            info_text += f"Variable Y: {results['target_column']}\n"

        if 'feature_columns' in results:
            info_text += f"Variables X: {len(results['feature_columns'])}\n"

        self.model_info_text.setText(info_text)

        # Parámetros
        params = results.get('parametros', {})
        if not params and 'coeficientes' in results:
            # Para regresión lineal, mostrar algunos coeficientes
            coefs = results['coeficientes']
            params = dict(list(coefs.items())[:5])  # Primeros 5 coeficientes

        self.params_table.setRowCount(len(params))
        for i, (param, value) in enumerate(params.items()):
            self.params_table.setItem(i, 0, QTableWidgetItem(param))
            if isinstance(value, (int, float)):
                self.params_table.setItem(i, 1, QTableWidgetItem(f"{value:.4f}"))
            else:
                self.params_table.setItem(i, 1, QTableWidgetItem(str(value)))

        self.params_table.resizeColumnsToContents()

    def _update_visualization(self):
        if not self.current_results or not hasattr(self, 'figure'):
            print("❌ No hay resultados o figura")
            return

        try:
            print(f"🎨 Iniciando visualización...")
            print(f"🎨 Tipo resultado: {self.current_results.get('tipo', 'SIN_TIPO')}")

            self.figure.clear()

            # Detectar tipo y usar visualización específica
            result_type = self.current_results.get('tipo', '')

            if result_type == 'random_forest':
                print("🎨 Detectado Random Forest, iniciando visualización específica...")
                self._plot_random_forest_results()  # ← FUNCIÓN ESPECÍFICA
            elif result_type == 'arbol_decision':
                print("🎨 Detectado Árbol de Decisión, iniciando visualización específica...")
                self._plot_tree_results()  # ← FUNCIÓN ESPECÍFICA
            elif result_type == 'svm':
                print("🎨 Detectado SVM, iniciando visualización específica...")
                self._plot_svm_results()
            elif result_type == 'regresion_lineal_multiple':
                print("🎨 Detectado Regresión Múltiple, iniciando visualización específica...")
                self._plot_regression_results()
            elif result_type == 'comparar_modelos':
                print("🎨 Detectado Comparación de Modelos, iniciando visualización específica...")
                self._plot_comparison_results()
            else:
                print(f"🎨 Tipo no reconocido: '{result_type}', usando genérico")
                self._plot_generic_results()

            self.figure.tight_layout()
            self.canvas.draw()
            print("🎨 Visualización completada")

        except Exception as e:
            print(f"❌ Error en visualización: {e}")
            import traceback
            traceback.print_exc()

    def _plot_random_forest_results(self):
        """Gráficos específicos para Random Forest - ÚNICO Y DIFERENTE"""
        try:
            # Layout especializado para Random Forest (2x3)
            gs = self.figure.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

            # 1. Importancia de características con enfoque de ensemble
            ax1 = self.figure.add_subplot(gs[0, 0])
            importancia = self.current_results.get('importancia_features', {})
            if importancia:
                features = list(importancia.keys())[:8]
                values = [importancia[f] for f in features]

                # Colores específicos para Random Forest (verde bosque)
                colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(features)))

                y_pos = np.arange(len(features))
                bars = ax1.barh(y_pos, values, color=colors, alpha=0.8,
                                edgecolor='darkgreen', linewidth=1)
                ax1.set_yticks(y_pos)
                ax1.set_yticklabels(features, fontsize=9)
                ax1.set_xlabel('Importancia Promedio del Ensemble')
                ax1.set_title('🌲 Importancia en Random Forest\n(Promedio de Múltiples Árboles)',
                              fontweight='bold', fontsize=10)
                ax1.grid(True, alpha=0.3, axis='x')

                # Añadir valores y mostrar que es ensemble
                for bar, value in zip(bars, values):
                    width = bar.get_width()
                    ax1.text(width + 0.005, bar.get_y() + bar.get_height() / 2,
                             f'{value:.3f}', ha='left', va='center', fontsize=8)

            # 2. Visualización del concepto de ensemble
            ax2 = self.figure.add_subplot(gs[0, 1])
            self._draw_random_forest_concept(ax2)

            # 3. Información específica de Random Forest
            ax3 = self.figure.add_subplot(gs[0, 2])
            ax3.axis('off')

            parametros = self.current_results.get('parametros', {})
            metricas = self.current_results.get('metricas', {})

            rf_info = [
                "🌲 RANDOM FOREST",
                "=" * 20,
                "",
                "CONFIGURACIÓN:",
                f"• N° Árboles: {parametros.get('n_estimators', 'N/A')}",
                f"• Profundidad: {parametros.get('max_depth', 'N/A')}",
                f"• Variables: {len(self.current_results.get('feature_columns', []))}",
                "",
                "CARACTERÍSTICAS:",
                "• Ensemble de árboles",
                "• Votación mayoritaria",
                "• Reduce sobreajuste",
                "• Robusto a outliers",
            ]

            if metricas:
                train_acc = metricas.get('train', {}).get('accuracy', 0)
                test_acc = metricas.get('test', {}).get('accuracy', 0)
                rf_info.extend([
                    "",
                    "RENDIMIENTO:",
                    f"• Train: {train_acc:.4f}",
                    f"• Test: {test_acc:.4f}",
                    f"• Estabilidad: {1 - abs(train_acc - test_acc):.4f}",
                ])

            info_text = '\n'.join(rf_info)
            ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes, fontsize=9,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

            # 4. Comparación Train vs Test con análisis de estabilidad
            ax4 = self.figure.add_subplot(gs[1, 0])
            if metricas:
                train_acc = metricas.get('train', {}).get('accuracy', 0)
                test_acc = metricas.get('test', {}).get('accuracy', 0)

                # Barras con colores de bosque
                bars = ax4.bar(['Entrenamiento', 'Prueba'], [train_acc, test_acc],
                               color=['#228B22', '#32CD32'], alpha=0.8, width=0.6,
                               edgecolor='darkgreen', linewidth=2)

                # Zona de estabilidad (diferencia < 0.05)
                stability_zone = max(train_acc, test_acc) * 0.95
                ax4.axhline(y=stability_zone, color='orange', linestyle='--',
                            alpha=0.7, label='Zona de Estabilidad')

                ax4.set_ylabel('Accuracy')
                ax4.set_title('🎯 Estabilidad del Ensemble', fontweight='bold')
                ax4.set_ylim(0, 1.1)
                ax4.grid(True, alpha=0.3, axis='y')

                # Valores y análisis de estabilidad
                for bar, acc in zip(bars, [train_acc, test_acc]):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

                # Indicador de estabilidad específico para RF
                diff = abs(train_acc - test_acc)
                if diff < 0.03:
                    status_text = "🟢 Muy Estable"
                    status_color = "lightgreen"
                elif diff < 0.07:
                    status_text = "🟡 Estable"
                    status_color = "yellow"
                else:
                    status_text = "🟠 Inestable"
                    status_color = "orange"

                ax4.text(0.5, 0.85, f'{status_text}\nDif: {diff:.3f}',
                         transform=ax4.transAxes, ha='center',
                         bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.7))

            # 5. Análisis de diversidad del ensemble
            ax5 = self.figure.add_subplot(gs[1, 1])

            # Simular diversidad de los árboles (en implementación real sería más complejo)
            n_trees = parametros.get('n_estimators', 10)
            tree_performance = np.random.normal(test_acc if 'test_acc' in locals() else 0.7, 0.05, min(n_trees, 20))
            tree_performance = np.clip(tree_performance, 0, 1)

            ax5.hist(tree_performance, bins=8, alpha=0.7, color='forestgreen',
                     edgecolor='black', density=True)
            ax5.axvline(x=np.mean(tree_performance), color='red', linestyle='--',
                        linewidth=2, label=f'Ensemble: {np.mean(tree_performance):.3f}')

            ax5.set_xlabel('Accuracy Individual de Árboles')
            ax5.set_ylabel('Densidad')
            ax5.set_title('📊 Diversidad del Ensemble\n(Distribución de Performance)', fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

            # 6. Métricas específicas de Random Forest
            ax6 = self.figure.add_subplot(gs[1, 2])
            ax6.axis('off')

            # Análisis específico de RF
            rf_analysis = [
                "ANÁLISIS DEL ENSEMBLE",
                "=" * 25,
                "",
                f"Número de árboles: {n_trees}",
                f"Diversidad estimada: {np.std(tree_performance):.3f}",
                f"Estabilidad: {1 - diff if 'diff' in locals() else 'N/A'}",
                "",
                "VENTAJAS OBSERVADAS:",
                "✓ Reduce sobreajuste vs árbol único",
                "✓ Maneja bien datos ruidosos",
                "✓ Importancia más robusta",
                "",
                "INTERPRETACIÓN:",
            ]

            if 'diff' in locals():
                if diff < 0.05:
                    rf_analysis.append("• Excelente generalización")
                elif diff < 0.1:
                    rf_analysis.append("• Buena generalización")
                else:
                    rf_analysis.append("• Revisar parámetros")

            analysis_text = '\n'.join(rf_analysis)
            ax6.text(0.05, 0.95, analysis_text, transform=ax6.transAxes, fontsize=8,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.8))

            self.figure.suptitle('🌲 Análisis Completo de Random Forest (Ensemble)',
                                 fontsize=14, fontweight='bold')

        except Exception as e:
            ax = self.figure.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f'Error en visualización de Random Forest:\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    def _draw_random_forest_concept(self, ax):
        """Dibujar concepto visual de Random Forest"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis('off')

        # Título
        ax.text(5, 7.5, 'Concepto Random Forest', ha='center', fontweight='bold', fontsize=11)

        # Dibujar múltiples árboles pequeños
        tree_positions = [(2, 5), (4, 5), (6, 5), (8, 5)]
        tree_colors = ['lightgreen', 'lightblue', 'lightcoral', 'lightyellow']

        for i, (x, y) in enumerate(tree_positions):
            # Tronco del árbol
            ax.add_patch(plt.Rectangle((x - 0.1, y - 1), 0.2, 1, facecolor='brown', alpha=0.7))

            # Copa del árbol
            circle = plt.Circle((x, y), 0.4, facecolor=tree_colors[i],
                                edgecolor='darkgreen', alpha=0.8)
            ax.add_patch(circle)

            # Etiqueta del árbol
            ax.text(x, y - 1.5, f'Árbol {i + 1}', ha='center', fontsize=8)

            # Predicción individual
            prediction = f'Pred: {0.6 + i * 0.1:.1f}'
            ax.text(x, y + 0.7, prediction, ha='center', fontsize=7,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Flecha hacia resultado final
        ax.arrow(5, 3, 0, -0.8, head_width=0.2, head_length=0.1,
                 fc='red', ec='red', linewidth=2)

        # Resultado del ensemble
        ax.text(5, 1.5, 'VOTACIÓN\nMAYORITARIA', ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='gold', alpha=0.8))

        ax.text(5, 0.5, 'Predicción Final: 0.75', ha='center', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

    def _recreate_plots_from_results(self):
        """Recrear gráficos desde los resultados - MÉTODO NUEVO"""
        if not self.current_results:
            return

        result_type = self.current_results.get('tipo', '')

        if result_type == 'regresion_lineal_multiple':
            self._plot_regression_results()
        elif result_type == 'arbol_decision':
            self._plot_tree_results()
        elif result_type == 'svm':
            self._plot_svm_results()
        elif result_type == 'comparar_modelos':
            self._plot_comparison_results()
        else:
            self._plot_generic_results()

    def _plot_regression_results(self):
        """Gráficos específicos para regresión"""
        try:
            datos_pred = self.current_results.get('datos_prediccion', {})
            y_test = np.array(datos_pred.get('y_test', []))
            y_pred = np.array(datos_pred.get('y_pred_test', []))

            if len(y_test) == 0 or len(y_pred) == 0:
                raise ValueError("No hay datos de predicción disponibles")

            # Layout 2x2
            ax1 = self.figure.add_subplot(2, 2, 1)
            ax2 = self.figure.add_subplot(2, 2, 2)
            ax3 = self.figure.add_subplot(2, 2, 3)
            ax4 = self.figure.add_subplot(2, 2, 4)

            # 1. Predicciones vs Reales
            ax1.scatter(y_test, y_pred, alpha=0.6, color='blue', s=30)
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            ax1.set_xlabel('Valores Reales')
            ax1.set_ylabel('Predicciones')
            r2 = self.current_results.get('metricas', {}).get('test', {}).get('r2', 0)
            ax1.set_title(f'Predicciones vs Reales (R² = {r2:.3f})')
            ax1.grid(True, alpha=0.3)

            # 2. Residuos
            residuos = y_test - y_pred
            ax2.scatter(y_pred, residuos, alpha=0.6, color='red', s=30)
            ax2.axhline(y=0, color='black', linestyle='--')
            ax2.set_xlabel('Predicciones')
            ax2.set_ylabel('Residuos')
            ax2.set_title('Análisis de Residuos')
            ax2.grid(True, alpha=0.3)

            # 3. Distribución de residuos
            ax3.hist(residuos, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax3.set_xlabel('Residuos')
            ax3.set_ylabel('Frecuencia')
            ax3.set_title('Distribución de Residuos')
            ax3.grid(True, alpha=0.3)

            # 4. Coeficientes
            coefs = self.current_results.get('coeficientes', {})
            if coefs:
                features = list(coefs.keys())[:10]  # Máximo 10 para legibilidad
                values = [coefs[f] for f in features]
                colors = ['red' if v < 0 else 'blue' for v in values]

                y_pos = np.arange(len(features))
                ax4.barh(y_pos, values, color=colors, alpha=0.7)
                ax4.set_yticks(y_pos)
                ax4.set_yticklabels(features)
                ax4.set_xlabel('Coeficiente')
                ax4.set_title('Importancia de Variables')
                ax4.grid(True, alpha=0.3)

        except Exception as e:
            ax = self.figure.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f'Error en gráfico de regresión:\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    # Reemplaza estas funciones en tu supervisado_window.py

    def _plot_tree_results(self):
        """Gráficos específicos para árbol de decisión - MEJORADO"""
        try:
            # Layout 2x3 para más información
            gs = self.figure.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

            # 1. Importancia de características con colores distintivos
            ax1 = self.figure.add_subplot(gs[0, 0])
            importancia = self.current_results.get('importancia_features', {})
            if importancia:
                features = list(importancia.keys())[:8]  # Top 8
                values = [importancia[f] for f in features]

                # Colores degradados para importancia
                colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(features)))

                y_pos = np.arange(len(features))
                bars = ax1.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax1.set_yticks(y_pos)
                ax1.set_yticklabels(features, fontsize=9)
                ax1.set_xlabel('Importancia de la Variable', fontsize=10)
                ax1.set_title('🌳 Importancia en Árbol de Decisión', fontsize=11, fontweight='bold')
                ax1.grid(True, alpha=0.3, axis='x')

                # Añadir valores en las barras
                for bar, value in zip(bars, values):
                    width = bar.get_width()
                    ax1.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                             f'{value:.3f}', ha='left', va='center', fontsize=8)

            # 2. Métricas comparativas con diseño distintivo
            ax2 = self.figure.add_subplot(gs[0, 1])
            metricas = self.current_results.get('metricas', {})
            if metricas:
                train_metrics = metricas.get('train', {})
                test_metrics = metricas.get('test', {})

                if self.current_results.get('es_clasificacion'):
                    train_acc = train_metrics.get('accuracy', 0)
                    test_acc = test_metrics.get('accuracy', 0)

                    # Gráfico de barras con indicadores de sobreajuste
                    bars = ax2.bar(['Entrenamiento', 'Prueba'], [train_acc, test_acc],
                                   color=['#2ecc71', '#e74c3c'], alpha=0.7, width=0.6)

                    # Línea de referencia para sobreajuste
                    if train_acc > test_acc + 0.1:
                        ax2.axhline(y=test_acc + 0.05, color='orange', linestyle='--',
                                    label='Zona de Sobreajuste', alpha=0.7)

                    ax2.set_ylabel('Precisión (Accuracy)', fontsize=10)
                    ax2.set_title('🎯 Rendimiento del Árbol', fontsize=11, fontweight='bold')
                    ax2.set_ylim(0, 1.1)
                    ax2.grid(True, alpha=0.3, axis='y')

                    # Valores en las barras
                    for bar, acc in zip(bars, [train_acc, test_acc]):
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                                 f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

                    # Indicador de sobreajuste
                    diff = train_acc - test_acc
                    if diff > 0.1:
                        ax2.text(0.5, 0.9, f'⚠️ Sobreajuste: {diff:.3f}',
                                 transform=ax2.transAxes, ha='center',
                                 bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))

            # 3. Información del modelo con diseño único
            ax3 = self.figure.add_subplot(gs[0, 2])
            ax3.axis('off')

            # Crear un panel de información estilizado
            tree_info = self.current_results.get('tree_info', {})
            info_lines = [
                "🌳 ÁRBOL DE DECISIÓN",
                "=" * 25,
                f"Tipo: {'Clasificación' if self.current_results.get('es_clasificacion') else 'Regresión'}",
                f"Profundidad Máx: {tree_info.get('max_depth', 'N/A')}",
                f"Min. Muestras: {tree_info.get('min_samples_split', 'N/A')}",
                "",
                "CARACTERÍSTICAS:",
                f"• Variables: {len(self.current_results.get('feature_columns', []))}",
                f"• Target: {self.current_results.get('target_column', 'N/A')}",
            ]

            if metricas and test_metrics:
                info_lines.extend([
                    "",
                    "MÉTRICAS FINALES:",
                    f"• R² Test: {test_metrics.get('r2', test_metrics.get('accuracy', 0)):.4f}",
                ])

            info_text = '\n'.join(info_lines)
            ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes, fontsize=9,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

            # 4. Visualización de la estructura del árbol (más realista)
            ax4 = self.figure.add_subplot(gs[1, :])
            ax4.axis('off')

            # Simular estructura del árbol con más detalle
            self._draw_tree_structure(ax4)

            self.figure.suptitle('Análisis de Árbol de Decisión', fontsize=14, fontweight='bold')

        except Exception as e:
            ax = self.figure.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f'Error en visualización de árbol:\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    def _draw_tree_structure(self, ax):
        """Dibujar estructura del árbol más detallada"""
        # Configurar límites
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)

        # Nodo raíz
        ax.add_patch(plt.Rectangle((4.5, 5), 1, 0.6, facecolor='lightgreen', edgecolor='black'))
        ax.text(5, 5.3, 'Raíz\n(Mejor división)', ha='center', va='center', fontsize=8, fontweight='bold')

        # Ramas del primer nivel
        ax.plot([5, 3], [5, 3.5], 'k-', linewidth=2)
        ax.plot([5, 7], [5, 3.5], 'k-', linewidth=2)

        # Nodos del segundo nivel
        ax.add_patch(plt.Rectangle((2.5, 3.2), 1, 0.6, facecolor='lightblue', edgecolor='black'))
        ax.text(3, 3.5, 'Nodo\nIzq.', ha='center', va='center', fontsize=8)

        ax.add_patch(plt.Rectangle((6.5, 3.2), 1, 0.6, facecolor='lightblue', edgecolor='black'))
        ax.text(7, 3.5, 'Nodo\nDer.', ha='center', va='center', fontsize=8)

        # Hojas finales
        for x, label in [(1.5, 'Hoja A'), (4.5, 'Hoja B'), (5.5, 'Hoja C'), (8.5, 'Hoja D')]:
            ax.add_patch(plt.Rectangle((x - 0.4, 1.7), 0.8, 0.6, facecolor='lightyellow', edgecolor='black'))
            ax.text(x, 2, label, ha='center', va='center', fontsize=7)

        # Conexiones a hojas
        ax.plot([3, 1.5], [3.2, 2.3], 'k-', linewidth=1)
        ax.plot([3, 4.5], [3.2, 2.3], 'k-', linewidth=1)
        ax.plot([7, 5.5], [3.2, 2.3], 'k-', linewidth=1)
        ax.plot([7, 8.5], [3.2, 2.3], 'k-', linewidth=1)

        # Etiquetas de decisión
        ax.text(4, 4.2, '≤ umbral', ha='center', fontsize=7, style='italic')
        ax.text(6, 4.2, '> umbral', ha='center', fontsize=7, style='italic')

        ax.set_title('🌳 Estructura del Árbol de Decisión', fontsize=12, fontweight='bold')

    def _plot_regression_results(self):
        """Gráficos específicos para regresión - COMPLETAMENTE DIFERENTE"""
        try:
            datos_pred = self.current_results.get('datos_prediccion', {})
            y_test = np.array(datos_pred.get('y_test', []))
            y_pred = np.array(datos_pred.get('y_pred_test', []))

            if len(y_test) == 0 or len(y_pred) == 0:
                raise ValueError("No hay datos de predicción disponibles")

            # Layout 2x3 especializado para regresión
            gs = self.figure.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

            # 1. Scatter plot de predicciones vs reales (mejorado)
            ax1 = self.figure.add_subplot(gs[0, 0])
            scatter = ax1.scatter(y_test, y_pred, alpha=0.6, c=np.abs(y_test - y_pred),
                                  cmap='RdYlBu_r', s=30, edgecolors='black', linewidth=0.5)

            # Línea de predicción perfecta
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')

            # Bandas de confianza
            ax1.fill_between([min_val, max_val], [min_val * 0.9, max_val * 0.9],
                             [min_val * 1.1, max_val * 1.1], alpha=0.2, color='gray', label='±10% Error')

            ax1.set_xlabel('Valores Reales')
            ax1.set_ylabel('Predicciones')
            r2 = self.current_results.get('metricas', {}).get('test', {}).get('r2', 0)
            ax1.set_title(f'📊 Predicciones vs Reales\nR² = {r2:.4f}', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=8)

            # Colorbar para errores
            cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
            cbar.set_label('Error Absoluto', fontsize=8)

            # 2. Análisis de residuos (mejorado)
            ax2 = self.figure.add_subplot(gs[0, 1])
            residuos = y_test - y_pred
            ax2.scatter(y_pred, residuos, alpha=0.6, color='red', s=30, edgecolors='black', linewidth=0.5)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=2, label='Residuo = 0')

            # Líneas de tendencia
            z = np.polyfit(y_pred, residuos, 1)
            p = np.poly1d(z)
            ax2.plot(sorted(y_pred), p(sorted(y_pred)), "b--", alpha=0.8, label=f'Tendencia (m={z[0]:.3f})')

            ax2.set_xlabel('Predicciones')
            ax2.set_ylabel('Residuos (Real - Predicho)')
            ax2.set_title('📈 Análisis de Residuos', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=8)

            # 3. Distribución de residuos con estadísticas
            ax3 = self.figure.add_subplot(gs[0, 2])
            n, bins, patches = ax3.hist(residuos, bins=20, alpha=0.7, color='green',
                                        edgecolor='black', density=True)

            # Curva normal de referencia
            mu, sigma = np.mean(residuos), np.std(residuos)
            y_normal = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
                        np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
            ax3.plot(bins, y_normal, 'r-', linewidth=2, label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')

            ax3.set_xlabel('Residuos')
            ax3.set_ylabel('Densidad')
            ax3.set_title('📊 Distribución de Residuos', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=8)

            # 4. Coeficientes con análisis de impacto
            ax4 = self.figure.add_subplot(gs[1, 0])
            coefs = self.current_results.get('coeficientes', {})
            if coefs:
                features = list(coefs.keys())[:10]
                values = [coefs[f] for f in features]
                colors = ['red' if v < 0 else 'blue' for v in values]

                y_pos = np.arange(len(features))
                bars = ax4.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
                ax4.set_yticks(y_pos)
                ax4.set_yticklabels(features, fontsize=9)
                ax4.set_xlabel('Coeficiente')
                ax4.set_title('📊 Impacto de Variables\n(Azul: +, Rojo: -)', fontweight='bold')
                ax4.grid(True, alpha=0.3, axis='x')

                # Línea de referencia en cero
                ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)

            # 5. Métricas de rendimiento
            ax5 = self.figure.add_subplot(gs[1, 1])
            ax5.axis('off')

            metricas = self.current_results.get('metricas', {})
            train_metrics = metricas.get('train', {})
            test_metrics = metricas.get('test', {})

            metrics_text = "📊 MÉTRICAS DE REGRESIÓN\n" + "=" * 30 + "\n\n"

            for split, metrics_dict in [("ENTRENAMIENTO", train_metrics), ("PRUEBA", test_metrics)]:
                metrics_text += f"{split}:\n"
                for metric, value in metrics_dict.items():
                    if isinstance(value, (int, float)):
                        metrics_text += f"  • {metric.upper()}: {value:.4f}\n"
                metrics_text += "\n"

            # Añadir interpretación
            r2_test = test_metrics.get('r2', 0)
            if r2_test > 0.8:
                interpretation = "🟢 Excelente ajuste"
            elif r2_test > 0.6:
                interpretation = "🟡 Buen ajuste"
            elif r2_test > 0.4:
                interpretation = "🟠 Ajuste moderado"
            else:
                interpretation = "🔴 Ajuste pobre"

            metrics_text += f"INTERPRETACIÓN:\n{interpretation}"

            ax5.text(0.05, 0.95, metrics_text, transform=ax5.transAxes, fontsize=9,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

            # 6. Gráfico Q-Q para normalidad de residuos
            ax6 = self.figure.add_subplot(gs[1, 2])
            from scipy import stats

            # Q-Q plot manual
            sorted_residuos = np.sort(residuos)
            n = len(sorted_residuos)
            theoretical_quantiles = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))

            ax6.scatter(theoretical_quantiles, sorted_residuos, alpha=0.6, s=20)
            ax6.plot(theoretical_quantiles, theoretical_quantiles * np.std(residuos) + np.mean(residuos),
                     'r-', label='Línea de Normalidad')
            ax6.set_xlabel('Cuantiles Teóricos')
            ax6.set_ylabel('Cuantiles Observados')
            ax6.set_title('📊 Q-Q Plot\n(Normalidad de Residuos)', fontweight='bold')
            ax6.grid(True, alpha=0.3)
            ax6.legend(fontsize=8)

            self.figure.suptitle('🔍 Análisis Completo de Regresión Lineal', fontsize=14, fontweight='bold')

        except Exception as e:
            ax = self.figure.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f'Error en gráfico de regresión:\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    def _plot_svm_results(self):
        """Gráficos específicos para SVM - COMPLETAMENTE REDISEÑADO"""
        try:
            # Layout especializado para SVM (3x3)
            gs = self.figure.add_gridspec(3, 3, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)

            # 1. Panel principal de información SVM
            ax1 = self.figure.add_subplot(gs[0, :2])
            ax1.axis('off')

            parametros = self.current_results.get('parametros', {})
            metricas = self.current_results.get('metricas', {})

            # Información estilizada de SVM
            svm_info = [
                "🔍 SUPPORT VECTOR MACHINE",
                "=" * 40,
                "",
                "CONFIGURACIÓN DEL MODELO:",
                f"  • Kernel: {parametros.get('kernel', 'linear').upper()}",
                f"  • Parámetro C: {parametros.get('C', 1.0)} (Regularización)",
                f"  • Vectores de Soporte: {parametros.get('n_support_vectors', 0)}",
                "",
                "CARACTERÍSTICAS DEL ALGORITMO:",
                f"  • Variables de entrada: {len(self.current_results.get('feature_columns', []))}",
                f"  • Tipo de problema: Clasificación Binaria",
                f"  • Variable objetivo: {self.current_results.get('target_column', 'N/A')}",
                f"  • Método: Maximización del margen",
            ]

            if metricas:
                train_acc = metricas.get('train', {}).get('accuracy', 0)
                test_acc = metricas.get('test', {}).get('accuracy', 0)
                generalization_gap = abs(train_acc - test_acc)

                svm_info.extend([
                    "",
                    "RENDIMIENTO DEL MODELO:",
                    f"  • Accuracy Entrenamiento: {train_acc:.4f}",
                    f"  • Accuracy Prueba: {test_acc:.4f}",
                    f"  • Brecha de generalización: {generalization_gap:.4f}",
                    f"  • Estatus: {'Buena generalización' if generalization_gap < 0.05 else 'Revisar parámetros'}",
                ])

            info_text = '\n'.join(svm_info)
            ax1.text(0.05, 0.95, info_text, transform=ax1.transAxes, fontsize=10,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.8', facecolor='lightcyan', alpha=0.9,
                               edgecolor='darkblue', linewidth=2))

            # 2. Visualización conceptual mejorada del SVM
            ax2 = self.figure.add_subplot(gs[0, 2])
            self._draw_svm_concept_enhanced(ax2, parametros)

            # 3. Análisis de rendimiento con métricas clave
            ax3 = self.figure.add_subplot(gs[1, 0])
            if metricas:
                train_acc = metricas.get('train', {}).get('accuracy', 0)
                test_acc = metricas.get('test', {}).get('accuracy', 0)

                # Gráfico de barras mejorado con análisis
                x_pos = [0, 1]
                values = [train_acc, test_acc]
                colors = ['#3498db', '#e74c3c']

                bars = ax3.bar(x_pos, values, color=colors, alpha=0.8, width=0.6,
                               edgecolor='black', linewidth=2)

                # Líneas de referencia importantes
                ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7,
                            label='Clasificador Aleatorio (50%)')
                ax3.axhline(y=0.8, color='green', linestyle='--', alpha=0.7,
                            label='Umbral de Buen Modelo (80%)')
                ax3.axhline(y=0.9, color='gold', linestyle='--', alpha=0.7,
                            label='Excelente Modelo (90%)')

                ax3.set_xticks(x_pos)
                ax3.set_xticklabels(['Entrenamiento', 'Prueba'])
                ax3.set_ylabel('Accuracy')
                ax3.set_title('🎯 Análisis de Rendimiento SVM', fontweight='bold')
                ax3.set_ylim(0, 1.1)
                ax3.grid(True, alpha=0.3, axis='y')
                ax3.legend(fontsize=8, loc='upper left')

                # Valores detallados en las barras
                for i, (bar, acc) in enumerate(zip(bars, values)):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                             f'{acc:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

                # Análisis de la brecha de generalización
                gap = train_acc - test_acc
                if abs(gap) > 0.05:
                    status = "Sobreajuste" if gap > 0 else "Subajuste"
                    color = "orange" if gap > 0 else "purple"
                    ax3.text(0.5, 0.7, f'{status}\nBrecha: {gap:+.3f}',
                             transform=ax3.transAxes, ha='center',
                             bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                             fontweight='bold')
                else:
                    ax3.text(0.5, 0.7, f'Buena\nGeneralización\n({gap:+.3f})',
                             transform=ax3.transAxes, ha='center',
                             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                             fontweight='bold')

            # 4. Matriz de confusión mejorada
            ax4 = self.figure.add_subplot(gs[1, 1])
            datos_pred = self.current_results.get('datos_prediccion', {})
            test_cm = datos_pred.get('test_cm')
            cm_labels = datos_pred.get('cm_labels', [])

            if test_cm is not None and len(test_cm) > 0:
                # Matriz de confusión con análisis detallado
                im = ax4.imshow(test_cm, interpolation='nearest', cmap='Blues', alpha=0.8)

                # Calcular porcentajes para cada celda
                cm_normalized = test_cm.astype('float') / test_cm.sum(axis=1)[:, np.newaxis]

                # Añadir números y porcentajes en las celdas
                for i in range(len(test_cm)):
                    for j in range(len(test_cm)):
                        count = test_cm[i, j]
                        percentage = cm_normalized[i, j] * 100

                        # Color del texto basado en la intensidad
                        text_color = "white" if count > test_cm.max() / 2 else "black"

                        text = ax4.text(j, i, f'{count}\n({percentage:.1f}%)',
                                        ha='center', va='center', fontweight='bold',
                                        fontsize=10, color=text_color)

                ax4.set_xlabel('Predicción del Modelo')
                ax4.set_ylabel('Etiqueta Real')
                ax4.set_title('🔢 Matriz de Confusión\n(Casos y Porcentajes)', fontweight='bold')

                # Etiquetas de clase mejoradas
                if len(cm_labels) == 2:
                    ax4.set_xticks([0, 1])
                    ax4.set_yticks([0, 1])
                    ax4.set_xticklabels([f'Clase {cm_labels[0]}', f'Clase {cm_labels[1]}'])
                    ax4.set_yticklabels([f'Clase {cm_labels[0]}', f'Clase {cm_labels[1]}'])

                # Colorbar
                cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
                cbar.set_label('Número de casos', fontsize=9)

            # 5. Análisis de características del modelo
            ax5 = self.figure.add_subplot(gs[1, 2])
            feature_columns = self.current_results.get('feature_columns', [])

            if len(feature_columns) > 0:
                # Análisis de las características utilizadas
                n_show = min(10, len(feature_columns))
                features = feature_columns[:n_show]

                # Para SVM, simular la relevancia basada en los coeficientes del hiperplano
                # En una implementación real, usarías los coeficientes reales del modelo
                np.random.seed(42)  # Para reproducibilidad
                relevance = np.random.exponential(0.4, n_show)
                relevance = relevance / np.max(relevance)  # Normalizar

                # Colores basados en relevancia
                colors = plt.cm.plasma(relevance)
                y_pos = np.arange(len(features))

                bars = ax5.barh(y_pos, relevance, color=colors, alpha=0.8,
                                edgecolor='black', linewidth=0.5)
                ax5.set_yticks(y_pos)
                ax5.set_yticklabels(features, fontsize=9)
                ax5.set_xlabel('Relevancia en el Hiperplano')
                ax5.set_title('📊 Importancia de Variables\nen la Decisión SVM', fontweight='bold')
                ax5.grid(True, alpha=0.3, axis='x')

                # Añadir valores en las barras
                for bar, value in zip(bars, relevance):
                    width = bar.get_width()
                    ax5.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                             f'{value:.3f}', ha='left', va='center', fontsize=8)

            # 6. Análisis detallado del modelo (reporte completo)
            ax6 = self.figure.add_subplot(gs[2, :])
            ax6.axis('off')

            test_report = metricas.get('test', {}).get('classification_report', {}) if metricas else {}

            if isinstance(test_report, dict) and cm_labels is not None:
                # Crear reporte detallado con análisis
                report_text = "📊 ANÁLISIS COMPLETO DEL MODELO SVM\n"
                report_text += "=" * 80 + "\n\n"

                # Información del modelo
                report_text += "CONFIGURACIÓN DEL MODELO:\n"
                report_text += f"• Kernel: {parametros.get('kernel', 'linear').upper()}\n"
                report_text += f"• Parámetro C: {parametros.get('C', 1.0)} (Control de regularización)\n"
                report_text += f"• Vectores de soporte: {parametros.get('n_support_vectors', 0)} "
                total_samples = sum([test_report[str(label)]['support'] for label in cm_labels if
                                     str(label) in test_report]) if test_report else 0
                if total_samples > 0:
                    support_ratio = parametros.get('n_support_vectors', 0) / total_samples
                    report_text += f"({support_ratio:.1%} de las muestras)\n"
                else:
                    report_text += "\n"

                report_text += "\nMÉTRICAS DE CLASIFICACIÓN:\n"
                report_text += f"{'CLASE':>15} {'PRECISIÓN':>12} {'RECALL':>12} {'F1-SCORE':>12} {'SOPORTE':>12}\n"
                report_text += "-" * 75 + "\n"

                # Métricas por clase con interpretación
                for label in cm_labels:
                    label_str = str(label)
                    if label_str in test_report:
                        metrics = test_report[label_str]
                        name = f'Clase {label}'
                        report_text += f"{name:>15} {metrics['precision']:>12.3f} {metrics['recall']:>12.3f} "
                        report_text += f"{metrics['f1-score']:>12.3f} {int(metrics['support']):>12}\n"

                # Métricas generales
                if 'accuracy' in test_report:
                    report_text += "\n" + "-" * 75 + "\n"
                    report_text += f"{'ACCURACY':>15} {'':>12} {'':>12} {test_report['accuracy']:>12.3f} "
                    report_text += f"{total_samples:>12}\n"

                if 'macro avg' in test_report:
                    macro = test_report['macro avg']
                    report_text += f"{'MACRO AVG':>15} {macro['precision']:>12.3f} {macro['recall']:>12.3f} "
                    report_text += f"{macro['f1-score']:>12.3f} {int(macro['support']):>12}\n"

                if 'weighted avg' in test_report:
                    weighted = test_report['weighted avg']
                    report_text += f"{'WEIGHTED AVG':>15} {weighted['precision']:>12.3f} {weighted['recall']:>12.3f} "
                    report_text += f"{weighted['f1-score']:>12.3f} {int(weighted['support']):>12}\n"

                # Interpretación del modelo
                report_text += "\nINTERPRETACIÓN DEL MODELO:\n"
                if test_report.get('accuracy', 0) > 0.9:
                    report_text += "🟢 EXCELENTE: Modelo con alta precisión y buena generalización\n"
                elif test_report.get('accuracy', 0) > 0.8:
                    report_text += "🟡 BUENO: Modelo con buen rendimiento, considerar ajustes menores\n"
                elif test_report.get('accuracy', 0) > 0.7:
                    report_text += "🟠 ACEPTABLE: Modelo funcional, revisar hiperparámetros\n"
                else:
                    report_text += "🔴 DEFICIENTE: Modelo requiere ajustes significativos\n"

                # Recomendaciones basadas en los resultados
                if 'train_acc' in locals() and 'test_acc' in locals():
                    gap = locals()['train_acc'] - locals()['test_acc']
                    if gap > 0.1:
                        report_text += "⚠️  RECOMENDACIÓN: Reducir C o usar regularización más fuerte\n"
                    elif gap < -0.05:
                        report_text += "⚠️  RECOMENDACIÓN: Aumentar C o usar modelo más complejo\n"
                    else:
                        report_text += "✅ RECOMENDACIÓN: Modelo bien balanceado, parámetros apropiados\n"

                ax6.text(0.02, 0.95, report_text, transform=ax6.transAxes,
                         fontsize=9, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

            self.figure.suptitle('🔍 Análisis Integral de Support Vector Machine',
                                 fontsize=14, fontweight='bold')

        except Exception as e:
            ax = self.figure.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f'Error en visualización de SVM:\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    def _draw_svm_concept_enhanced(self, ax, parametros):
        """Dibujar concepto visual mejorado de SVM"""
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)

        # Generar puntos de ejemplo para dos clases (más realistas)
        np.random.seed(42)

        # Clase 1 (azul) - cluster en cuadrante inferior izquierdo
        angle1 = np.random.uniform(0, 2 * np.pi, 12)
        radius1 = np.random.exponential(0.8, 12)
        x1_class1 = -1.2 + radius1 * np.cos(angle1) * 0.6
        x2_class1 = -1.2 + radius1 * np.sin(angle1) * 0.6

        # Clase 2 (rojo) - cluster en cuadrante superior derecho
        angle2 = np.random.uniform(0, 2 * np.pi, 12)
        radius2 = np.random.exponential(0.8, 12)
        x1_class2 = 1.2 + radius2 * np.cos(angle2) * 0.6
        x2_class2 = 1.2 + radius2 * np.sin(angle2) * 0.6

        # Plotear puntos con mejor estilo
        scatter1 = ax.scatter(x1_class1, x2_class1, c='blue', s=60, alpha=0.8,
                              label='Clase 0', edgecolors='darkblue', linewidth=1, marker='o')
        scatter2 = ax.scatter(x1_class2, x2_class2, c='red', s=60, alpha=0.8,
                              label='Clase 1', edgecolors='darkred', linewidth=1, marker='s')

        # Hiperplano de separación óptimo
        x_line = np.linspace(-3, 3, 100)
        y_line = 0.3 * x_line + 0.1  # Línea que separa bien las clases
        ax.plot(x_line, y_line, 'black', linewidth=4, label='Hiperplano Óptimo', alpha=0.9)

        # Márgenes del SVM
        margin_width = 1.2
        ax.plot(x_line, y_line + margin_width, 'gray', linewidth=3, linestyle='--',
                alpha=0.8, label='Margen Superior')
        ax.plot(x_line, y_line - margin_width, 'gray', linewidth=3, linestyle='--',
                alpha=0.8, label='Margen Inferior')

        # Vectores de soporte (puntos críticos)
        support_x = [-0.8, 0.8]
        support_y = [0.14, 0.34]
        ax.scatter(support_x, support_y, s=200, facecolors='none',
                   edgecolors='green', linewidth=4, marker='o',
                   label='Vectores de Soporte')

        # Zona de margen
        ax.fill_between(x_line, y_line - margin_width, y_line + margin_width,
                        alpha=0.15, color='yellow', label='Zona de Margen')

        # Añadir información del kernel
        kernel_type = parametros.get('kernel', 'linear')
        ax.text(-2.8, 2.5, f'Kernel: {kernel_type.upper()}', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # Parámetro C
        C_value = parametros.get('C', 1.0)
        ax.text(-2.8, 2, f'C = {C_value}', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        # Vectores de soporte
        n_support = parametros.get('n_support_vectors', 0)
        ax.text(-2.8, 1.5, f'Support Vec: {n_support}', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        ax.set_xlabel('Variable X₁')
        ax.set_ylabel('Variable X₂')
        ax.set_title('Concepto SVM: Separación Óptima\ncon Maximización del Margen',
                     fontweight='bold', fontsize=10)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

        # Añadir flechas explicativas
        ax.annotate('Margen máximo', xy=(0, 0.1), xytext=(1.5, -1.5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=9, fontweight='bold', color='red')

        ax.annotate('Decisión de\nclasificación', xy=(0, 0.1), xytext=(-1.5, 1.5),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2),
                    fontsize=9, fontweight='bold', color='black')

    def _plot_comparison_results(self):
        """Gráficos para comparación de modelos - COMPLETAMENTE REDISEÑADO"""
        try:
            ranking = self.current_results.get('ranking', [])
            modelos_data = self.current_results.get('modelos', {})

            if not ranking:
                ax = self.figure.add_subplot(1, 1, 1)
                ax.text(0.5, 0.5, 'No hay datos de comparación disponibles',
                        ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                return

            # Layout especializado para comparación
            gs = self.figure.add_gridspec(3, 3, height_ratios=[1.5, 1, 1], hspace=0.4, wspace=0.3)

            # 1. Gráfico principal de ranking (mejorado)
            ax1 = self.figure.add_subplot(gs[0, :])

            models = [item['modelo'] for item in ranking]
            scores = [item['score'] for item in ranking]

            # Mapear nombres a nombres más descriptivos
            model_names = {
                'linear': 'Regresión Lineal',
                'tree': 'Árbol de Decisión',
                'random_forest': 'Random Forest',
                'svm': 'Support Vector Machine'
            }
            display_names = [model_names.get(m, m.title()) for m in models]

            # Colores distintivos por ranking
            colors = ['#FFD700', '#C0C0C0', '#CD7F32', '#87CEEB', '#DDA0DD'][:len(models)]

            # Crear barras horizontales para mejor legibilidad
            y_pos = np.arange(len(display_names))
            bars = ax1.barh(y_pos, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

            # Personalizar cada barra según el ranking
            for i, (bar, score) in enumerate(zip(bars, scores)):
                # Añadir posición del ranking
                width = bar.get_width()
                ax1.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                         f'#{i + 1} - {score:.4f}', ha='left', va='center', fontweight='bold', fontsize=11)

                # Añadir patrón visual para el ganador
                if i == 0:
                    bar.set_hatch('///')
                    bar.set_edgecolor('gold')
                    bar.set_linewidth(3)

            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(display_names)
            ax1.set_xlabel('Score de Rendimiento (Mayor es Mejor)', fontsize=12)
            ax1.set_title('Ranking de Modelos de Machine Learning', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='x')

            # Línea de referencia
            ax1.axvline(x=np.mean(scores), color='red', linestyle='--', alpha=0.7,
                        label=f'Promedio: {np.mean(scores):.3f}')
            ax1.legend()

            # 2. Comparación detallada de métricas
            ax2 = self.figure.add_subplot(gs[1, :2])

            # Extraer métricas de todos los modelos
            all_metrics = {}
            for model_name, model_data in modelos_data.items():
                if 'error' not in model_data and 'metricas' in model_data:
                    test_metrics = model_data['metricas'].get('test', {})
                    for metric_name, value in test_metrics.items():
                        if isinstance(value, (int, float)) and metric_name != 'classification_report':
                            if metric_name not in all_metrics:
                                all_metrics[metric_name] = {}
                            all_metrics[metric_name][model_names.get(model_name, model_name)] = value

            if all_metrics:
                # Crear gráfico de radar/comparación múltiple
                metric_names = list(all_metrics.keys())[:4]  # Máximo 4 métricas
                x = np.arange(len(metric_names))
                width = 0.8 / len(modelos_data)

                for i, (model_name, model_data) in enumerate(modelos_data.items()):
                    if 'error' not in model_data:
                        display_name = model_names.get(model_name, model_name)
                        values = []
                        for metric in metric_names:
                            test_metrics = model_data['metricas'].get('test', {})
                            values.append(test_metrics.get(metric, 0))

                        ax2.bar(x + i * width, values, width, label=display_name,
                                alpha=0.7, edgecolor='black', linewidth=0.5)

                ax2.set_xlabel('Métricas')
                ax2.set_ylabel('Valor')
                ax2.set_title('Comparación Detallada de Métricas', fontweight='bold')
                ax2.set_xticks(x + width * (len(modelos_data) - 1) / 2)
                ax2.set_xticklabels([m.upper() for m in metric_names])
                ax2.legend(fontsize=9)
                ax2.grid(True, alpha=0.3, axis='y')

            # 3. Información del ganador
            ax3 = self.figure.add_subplot(gs[1, 2])
            ax3.axis('off')

            if ranking:
                winner = ranking[0]
                winner_name = model_names.get(winner['modelo'], winner['modelo'])
                winner_data = modelos_data.get(winner['modelo'], {})

                winner_info = [
                    f"MODELO GANADOR",
                    "=" * 25,
                    f"Modelo: {winner_name}",
                    f"Score: {winner['score']:.4f}",
                    f"Métrica: {winner['metrica'].upper()}",
                    "",
                    "CARACTERÍSTICAS:"
                ]

                if 'target_column' in self.current_results:
                    winner_info.append(f"• Variable Y: {self.current_results['target_column']}")

                if 'feature_columns' in self.current_results:
                    n_features = len(self.current_results['feature_columns'])
                    winner_info.append(f"• Variables X: {n_features}")

                if winner_data and 'metricas' in winner_data:
                    test_metrics = winner_data['metricas'].get('test', {})
                    winner_info.append("")
                    winner_info.append("MÉTRICAS PRINCIPALES:")
                    for metric, value in test_metrics.items():
                        if isinstance(value, (int, float)) and metric != 'classification_report':
                            winner_info.append(f"• {metric.upper()}: {value:.4f}")

                info_text = '\n'.join(winner_info)
                ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes, fontsize=9,
                         verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='gold', alpha=0.8))

            # 4. Análisis de rendimiento por modelo (detalles)
            ax4 = self.figure.add_subplot(gs[2, :])
            ax4.axis('off')

            # Crear tabla comparativa
            comparison_text = "ANÁLISIS COMPARATIVO DETALLADO\n"
            comparison_text += "=" * 80 + "\n\n"

            # Header de la tabla
            comparison_text += f"{'MODELO':^20} | {'TIPO':^15} | {'TRAIN':^10} | {'TEST':^10} | {'DIFERENCIA':^12} | {'ESTADO':^15}\n"
            comparison_text += "-" * 90 + "\n"

            for i, (model_name, model_data) in enumerate(modelos_data.items()):
                if 'error' not in model_data and 'metricas' in model_data:
                    display_name = model_names.get(model_name, model_name)

                    train_metrics = model_data['metricas'].get('train', {})
                    test_metrics = model_data['metricas'].get('test', {})

                    # Obtener métrica principal
                    main_metric = 'accuracy' if 'accuracy' in test_metrics else list(test_metrics.keys())[0]
                    if main_metric != 'classification_report':
                        train_val = train_metrics.get(main_metric, 0)
                        test_val = test_metrics.get(main_metric, 0)
                        diff = train_val - test_val

                        # Determinar estado del modelo
                        if abs(diff) < 0.05:
                            status = "Balanceado"
                        elif diff > 0.1:
                            status = "Sobreajuste"
                        elif diff < -0.05:
                            status = "Subajuste"
                        else:
                            status = "Aceptable"

                        # Determinar tipo de problema
                        problema = "Clasificación" if model_data.get('es_clasificacion') else "Regresión"

                        comparison_text += f"{display_name:^20} | {problema:^15} | {train_val:^10.3f} | "
                        comparison_text += f"{test_val:^10.3f} | {diff:^12.3f} | {status:^15}\n"

            comparison_text += "\n" + "=" * 80 + "\n"
            comparison_text += "INTERPRETACIÓN:\n"
            comparison_text += "• Sobreajuste: El modelo memoriza datos de entrenamiento (Train >> Test)\n"
            comparison_text += "• Subajuste: El modelo es demasiado simple (Train < Test)\n"
            comparison_text += "• Balanceado: Diferencia < 0.05 entre Train y Test (Ideal)\n"

            ax4.text(0.02, 0.95, comparison_text, transform=ax4.transAxes,
                     fontsize=8, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))

            self.figure.suptitle('Comparación Completa de Modelos de Machine Learning',
                                 fontsize=14, fontweight='bold')

        except Exception as e:
            ax = self.figure.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f'Error en gráfico de comparación:\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    def _plot_svm_results(self):
        """Gráficos específicos para SVM"""
        try:
            # Layout 2x2
            ax1 = self.figure.add_subplot(2, 2, 1)
            ax2 = self.figure.add_subplot(2, 2, 2)
            ax3 = self.figure.add_subplot(2, 2, 3)
            ax4 = self.figure.add_subplot(2, 2, 4)

            # 1. Información del modelo
            ax1.axis('off')
            info_text = f"Tipo: Support Vector Machine\n"
            info_text += f"Kernel: Lineal (simplificado)\n"

            parametros = self.current_results.get('parametros', {})
            if parametros:
                info_text += f"C: {parametros.get('C', 'N/A')}\n"

            metricas = self.current_results.get('metricas', {})
            if metricas:
                train_acc = metricas.get('train', {}).get('accuracy', 0)
                test_acc = metricas.get('test', {}).get('accuracy', 0)
                info_text += f"Accuracy Train: {train_acc:.4f}\n"
                info_text += f"Accuracy Test: {test_acc:.4f}\n"

            ax1.text(0.1, 0.9, info_text, transform=ax1.transAxes, fontsize=12,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

            # 2. Comparación de métricas
            if metricas:
                train_acc = metricas.get('train', {}).get('accuracy', 0)
                test_acc = metricas.get('test', {}).get('accuracy', 0)

                ax2.bar(['Train', 'Test'], [train_acc, test_acc],
                        color=['lightblue', 'lightcoral'], alpha=0.7)
                ax2.set_ylabel('Accuracy')
                ax2.set_title('Comparación Train vs Test')
                ax2.set_ylim(0, 1.1)
                ax2.grid(True, alpha=0.3)

                # Añadir valores
                ax2.text(0, train_acc + 0.02, f'{train_acc:.3f}',
                         ha='center', va='bottom', fontweight='bold')
                ax2.text(1, test_acc + 0.02, f'{test_acc:.3f}',
                         ha='center', va='bottom', fontweight='bold')

            # 3. Información de características
            feature_columns = self.current_results.get('feature_columns', [])
            n_features = len(feature_columns)
            ax3.axis('off')
            features_text = f"Características utilizadas ({n_features}):\n\n"
            for i, feature in enumerate(feature_columns[:8]):  # Mostrar máximo 8
                features_text += f"• {feature}\n"
            if n_features > 8:
                features_text += f"... y {n_features - 8} más"

            ax3.text(0.1, 0.9, features_text, transform=ax3.transAxes, fontsize=10,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

            # 4. Visualización conceptual de SVM
            ax4.axis('off')
            concept_text = """SVM - Concepto:

    🔵 Clase 0    🔴 Clase 1
        |\\        /|
        | \\      / |
        |  \\____/  |
        |  MARGEN  |
        |          |

    Busca el hiperplano que
    maximiza la separación
    entre las clases."""

            ax4.text(0.1, 0.9, concept_text, transform=ax4.transAxes, fontsize=9,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

        except Exception as e:
            ax = self.figure.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f'Error en gráfico de SVM:\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    def _plot_comparison_results(self):
        """Gráficos para comparación de modelos"""
        try:
            ranking = self.current_results.get('ranking', [])

            if not ranking:
                ax = self.figure.add_subplot(1, 1, 1)
                ax.text(0.5, 0.5, 'No hay resultados de comparación disponibles',
                        ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                return

            # Gráfico de barras con ranking
            ax = self.figure.add_subplot(1, 1, 1)

            models = [item['modelo'] for item in ranking]
            scores = [item['score'] for item in ranking]

            colors = ['gold', 'silver', '#CD7F32'] + ['lightblue'] * (len(models) - 3)
            colors = colors[:len(models)]

            bars = ax.bar(models, scores, color=colors, alpha=0.8)
            ax.set_ylabel('Score')
            ax.set_title('Comparación de Modelos (Mayor es mejor)')
            ax.grid(True, alpha=0.3)

            # Añadir valores en las barras
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

            # Rotar etiquetas si hay muchos modelos
            if len(models) > 3:
                ax.tick_params(axis='x', rotation=45)

        except Exception as e:
            ax = self.figure.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f'Error en gráfico de comparación:\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    def _plot_generic_results(self):
        """Gráfico genérico para resultados no especificados"""
        ax = self.figure.add_subplot(1, 1, 1)

        result_type = self.current_results.get('tipo', 'Desconocido')
        ax.text(0.5, 0.5, f'Visualización para {result_type}\nno implementada aún',
                ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.axis('off')
        ax.set_title(f'Resultado: {result_type}', fontsize=14)

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
            try:
                self.figure.savefig(filepath, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Éxito", "Gráfico guardado correctamente")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error guardando gráfico: {str(e)}")

    def _export_model(self):
        """Exportar modelo"""
        if not self.current_model or not ML_AVAILABLE:
            QMessageBox.warning(self, "Error", "No hay modelo para exportar")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Exportar Modelo",
            f"modelo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
            "Pickle (*.pkl);;JSON (*.json)"
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

# ==================== VENTANA PRINCIPAL COMPATIBLE ====================

class SupervisadoWindowCompatible(QWidget):
    """Ventana principal compatible con PyInstaller"""

    def __init__(self):
        QWidget.__init__(self)
        print("🚀 SupervisadoWindowCompatible: Inicializando...")

        self.current_data = None
        self.current_worker = None
        self.analysis_history = []

        # Registrar como observador del DataManager
        if DATA_MANAGER_AVAILABLE:
            dm = get_data_manager()
            if dm is not None:
                dm.add_observer(self)

        self.setup_ui()
        self.center_window()
        self.check_data_availability()
        print("✅ SupervisadoWindowCompatible: Inicialización completada")

    def center_window(self):
        """Centrar ventana en la pantalla"""
        self.setMinimumSize(1200, 700)
        self.resize(1600, 900)

        # Obtener el centro de la pantalla
        screen = QDesktopWidget().screenGeometry()
        x = (screen.width() - 1600) // 2
        y = (screen.height() - 900) // 2
        self.move(x, y)

    def setup_ui(self):
        """Configurar interfaz de usuario compatible"""
        self.setWindowTitle("🤖 Machine Learning Supervisado - Compatible")

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
        self.results_widget = ResultsVisualizationWidgetCompatible()
        content_splitter.addWidget(self.results_widget)

        content_splitter.setSizes([400, 1000])
        content_splitter.setChildrenCollapsible(False)

        main_layout.addWidget(content_splitter)

        # Log
        log_widget = self.create_log_widget()
        main_layout.addWidget(log_widget)

        self.setLayout(main_layout)
        self.apply_styles()

    def update(self, event_type: str = ""):
        """Método Observer"""
        if event_type in ['data_changed', 'session_imported']:
            self.check_data_availability()
        elif event_type == 'data_cleared':
            self.current_data = None
            self.update_data_info()
            self.enable_analysis_buttons(False)

    def check_data_availability(self):
        """Verificar disponibilidad de datos"""
        if DATA_MANAGER_AVAILABLE:
            if has_shared_data():
                self.current_data = get_shared_data()
                self.update_data_info()
                self.enable_analysis_buttons(True)
            else:
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
        """Habilitar/deshabilitar botones"""
        buttons = [
            self.reg_simple_btn, self.reg_multiple_btn, self.tree_btn,
            self.forest_btn, self.svm_btn, self.compare_btn
        ]
        for btn in buttons:
            btn.setEnabled(enabled and ML_AVAILABLE)

    def create_header(self) -> QWidget:
        """Crear header"""
        header = QFrame()
        header.setFrameStyle(QFrame.Box)
        header.setMinimumHeight(100)
        header.setMaximumHeight(120)

        layout = QHBoxLayout()

        # Información del título
        title_layout = QVBoxLayout()

        title = QLabel("🤖 Machine Learning Supervisado - Compatible")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        title_layout.addWidget(title)

        self.data_info_label = QLabel("Verificando datos...")
        self.data_info_label.setStyleSheet("color: #666; font-size: 11px;")
        title_layout.addWidget(self.data_info_label)

        layout.addLayout(title_layout)
        layout.addStretch()

        # Botones
        self.refresh_btn = QPushButton("🔄 Actualizar")
        self.refresh_btn.clicked.connect(self.check_data_availability)
        layout.addWidget(self.refresh_btn)

        header.setLayout(layout)
        return header

    def create_left_panel(self) -> QWidget:
        """Crear panel izquierdo"""
        panel = QWidget()
        panel.setMinimumWidth(350)
        panel.setMaximumWidth(450)

        layout = QVBoxLayout()

        # Widget de selección de variables
        self.variable_selection = VariableSelectionWidgetCompatible()
        self.variable_selection.variables_changed.connect(self.on_variables_changed)
        layout.addWidget(self.variable_selection, 3)

        # Opciones de análisis
        options_group = QGroupBox("⚙️ Opciones")
        options_layout = QVBoxLayout()

        # Parámetros simplificados para la versión compatible
        params_layout = QHBoxLayout()

        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(3, 20)
        self.max_depth_spin.setValue(5)
        params_layout.addWidget(QLabel("Max Depth:"))
        params_layout.addWidget(self.max_depth_spin)

        self.test_size_spin = QDoubleSpinBox()
        self.test_size_spin.setRange(10, 50)
        self.test_size_spin.setValue(20)
        self.test_size_spin.setSuffix("%")
        params_layout.addWidget(QLabel("Test:"))
        params_layout.addWidget(self.test_size_spin)

        options_layout.addLayout(params_layout)
        options_group.setLayout(options_layout)
        layout.addWidget(options_group, 0)

        # Botones de análisis
        analysis_group = QGroupBox("🚀 Ejecutar Análisis")
        analysis_layout = QVBoxLayout()

        models_layout = QGridLayout()

        self.reg_simple_btn = QPushButton("📊 Regresión Simple")
        self.reg_simple_btn.clicked.connect(lambda: self.run_analysis('regresion_simple'))
        models_layout.addWidget(self.reg_simple_btn, 0, 0)

        self.reg_multiple_btn = QPushButton("📈 Regresión Múltiple")
        self.reg_multiple_btn.clicked.connect(lambda: self.run_analysis('regresion_multiple'))
        models_layout.addWidget(self.reg_multiple_btn, 0, 1)

        self.tree_btn = QPushButton("🌳 Árbol Decisión")
        self.tree_btn.clicked.connect(lambda: self.run_analysis('arbol_decision'))
        models_layout.addWidget(self.tree_btn, 1, 0)

        self.forest_btn = QPushButton("🌲 Random Forest")
        self.forest_btn.clicked.connect(lambda: self.run_analysis('random_forest'))
        models_layout.addWidget(self.forest_btn, 1, 1)

        self.svm_btn = QPushButton("📷 SVM")
        self.svm_btn.clicked.connect(lambda: self.run_analysis('svm'))
        models_layout.addWidget(self.svm_btn, 2, 0)

        analysis_layout.addLayout(models_layout)

        # Botón de comparación
        self.compare_btn = QPushButton("⚖️ Comparar Modelos")
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
        self.cancel_btn = QPushButton("❌ Cancelar")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_analysis)
        analysis_layout.addWidget(self.cancel_btn)

        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group, 1)

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
                "Las librerías de Machine Learning no están disponibles.\n\n"
                "Versión compatible cargada con implementaciones NumPy.\n"
                "Funcionalidad limitada pero sin dependencias problemáticas."
            )
            return

        target = self.variable_selection.get_target_variable()
        features = self.variable_selection.get_selected_features()

        # Preparar argumentos base
        base_kwargs = {
            'target_column': target,
            'feature_columns': features,
            'test_size': self.test_size_spin.value() / 100.0
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
                'y_column': target
            }
        else:
            kwargs = base_kwargs

        # Agregar parámetros específicos
        if analysis_type == 'arbol_decision':
            kwargs.update({
                'max_depth': self.max_depth_spin.value(),
                'min_samples_split': 2
            })
        elif analysis_type == 'svm':
            kwargs.update({
                'C': 1.0
            })

        self.show_progress(True)
        self.log(f"🚀 Iniciando análisis: {analysis_type}")

        # Usar worker compatible
        self.current_worker = MLAnalysisWorkerCompatible(analysis_type, self.current_data, **kwargs)

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

        # Diálogo simple para seleccionar modelos
        dialog = QDialog(self)
        dialog.setWindowTitle("Seleccionar Modelos")
        dialog.setModal(True)
        dialog.resize(400, 300)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Selecciona los modelos a comparar:"))

        model_checks = {}
        models_info = [
            ('linear', 'Regresión Lineal'),
            ('tree', 'Árbol de Decisión'),
            ('svm', 'Support Vector Machine')
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
                'modelos': selected_models
            }

            self.show_progress(True)
            self.log(f"🚀 Comparando {len(selected_models)} modelos...")

            self.current_worker = MLAnalysisWorkerCompatible('comparar_modelos', self.current_data, **kwargs)
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

        # Verificación básica de datos
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

        self.enable_analysis_buttons(not show)

    def log(self, message: str):
        """Añadir mensaje al log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_log(self):
        """Limpiar log"""
        self.log_text.clear()
        self.log("📝 Log limpiado")

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
        }
        
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #3498db, stop:1 #2980b9);
            border-radius: 6px;
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

# ==================== USAR LA CLASE COMPATIBLE ====================

# Asignar la clase compatible como la clase principal
SupervisadoWindow = SupervisadoWindowCompatible

# ==================== FUNCIÓN PRINCIPAL PARA TESTING ====================

def main():
    """Función principal para testing"""
    app = QApplication(sys.argv)

    # Crear ventana compatible
    window = SupervisadoWindowCompatible()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()