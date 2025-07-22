"""
supervisado_window.py - Interfaz GUI Optimizada para Machine Learning Supervisado
Sistema mejorado con mejor gestión de datos y conexión con el sistema principal
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
    QButtonGroup, QRadioButton
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

    # Fallback para testing
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
        exportar_modelo,
        cargar_modelo,
        limpiar_memoria
    )

    # Importaciones de matplotlib
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import seaborn as sns

    ML_AVAILABLE = True
    print("✅ Librerías ML Supervisado cargadas correctamente")

except ImportError as e:
    ML_AVAILABLE = False
    print(f"❌ Error cargando ML: {e}")

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

            # Mapear tipos de análisis a funciones
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
        # Filtrar parámetros para regresión simple
        filtered_kwargs = {k: v for k, v in self.kwargs.items()
                          if k in ['x_column', 'y_column', 'optimize_params']}
        return regresion_lineal_simple(self.data, **filtered_kwargs)

    def _run_regresion_multiple(self):
        self.status.emit("Ejecutando regresión lineal múltiple...")
        self.progress.emit(30)
        # Filtrar parámetros para regresión múltiple
        filtered_kwargs = {k: v for k, v in self.kwargs.items()
                          if k in ['target_column', 'feature_columns', 'regularization',
                                   'alpha', 'optimize_params']}
        return regresion_lineal_multiple(self.data, **filtered_kwargs)

    def _run_arbol_decision(self):
        self.status.emit("Entrenando árbol de decisión...")
        self.progress.emit(30)
        # Filtrar parámetros para árbol de decisión
        filtered_kwargs = {k: v for k, v in self.kwargs.items()
                          if k in ['target_column', 'feature_columns', 'max_depth',
                                   'min_samples_split', 'min_samples_leaf', 'optimize_params']}
        return arbol_decision(self.data, **filtered_kwargs)

    def _run_random_forest(self):
        self.status.emit("Entrenando Random Forest...")
        self.progress.emit(30)
        # Filtrar parámetros para random forest
        filtered_kwargs = {k: v for k, v in self.kwargs.items()
                          if k in ['target_column', 'feature_columns', 'n_estimators',
                                   'max_depth', 'max_features', 'optimize_params', 'n_jobs']}
        return random_forest(self.data, **filtered_kwargs)

    def _run_svm(self):
        self.status.emit("Entrenando SVM...")
        self.progress.emit(30)
        # Filtrar parámetros para SVM
        filtered_kwargs = {k: v for k, v in self.kwargs.items()
                          if k in ['target_column', 'feature_columns', 'kernel',
                                   'C', 'gamma', 'optimize_params']}
        return svm_modelo(self.data, **filtered_kwargs)

    def _run_comparar_modelos(self):
        self.status.emit("Comparando modelos...")
        self.progress.emit(20)
        # Filtrar parámetros para comparación
        filtered_kwargs = {k: v for k, v in self.kwargs.items()
                          if k in ['target_column', 'feature_columns', 'modelos',
                                   'optimize_all', 'cv_folds', 'n_jobs']}
        return comparar_modelos_supervisado(self.data, **filtered_kwargs)

# ==================== WIDGET DE SELECCIÓN DE VARIABLES ====================

class VariableSelectionWidget(QWidget):
    """Widget para selección de variables"""
    variables_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.data = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Variable objetivo
        target_group = QGroupBox("🎯 Variable Objetivo")
        target_layout = QVBoxLayout()

        self.target_combo = QComboBox()
        self.target_combo.currentTextChanged.connect(self._on_target_changed)
        target_layout.addWidget(self.target_combo)

        self.target_info_label = QLabel("Selecciona una variable objetivo")
        self.target_info_label.setStyleSheet("color: #666; font-size: 11px;")
        target_layout.addWidget(self.target_info_label)

        target_group.setLayout(target_layout)
        layout.addWidget(target_group)

        # Variables predictoras
        features_group = QGroupBox("📊 Variables Predictoras")
        features_layout = QVBoxLayout()

        # Controles de selección
        controls_layout = QHBoxLayout()

        self.select_all_btn = QPushButton("Todas")
        self.select_all_btn.clicked.connect(self._select_all_features)
        controls_layout.addWidget(self.select_all_btn)

        self.select_none_btn = QPushButton("Ninguna")
        self.select_none_btn.clicked.connect(self._select_none_features)
        controls_layout.addWidget(self.select_none_btn)

        self.auto_select_btn = QPushButton("🤖 Auto")
        self.auto_select_btn.clicked.connect(self._auto_select_features)
        controls_layout.addWidget(self.auto_select_btn)

        features_layout.addLayout(controls_layout)

        # Lista de características
        self.features_list = QListWidget()
        self.features_list.setSelectionMode(QListWidget.MultiSelection)
        self.features_list.itemSelectionChanged.connect(self._on_features_changed)
        features_layout.addWidget(self.features_list)

        self.features_info_label = QLabel("0 variables seleccionadas")
        self.features_info_label.setStyleSheet("color: #666; font-size: 11px;")
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

        # Limpiar
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
        default_targets = ['WQI', 'Calidad', 'Quality', 'Target', 'y', 'Indice_Calidad']
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
            self.target_info_label.setText("Selecciona una variable objetivo")
            return

        target_data = self.data[target]
        n_unique = target_data.nunique()

        if target_data.dtype in [np.float64, np.float32, np.int64, np.int32]:
            info = f"Numérica: {n_unique} valores únicos"
            if n_unique <= 10:
                info += " → Clasificación sugerida"
            else:
                info += " → Regresión sugerida"
        else:
            info = f"Categórica: {n_unique} categorías → Clasificación"

        self.target_info_label.setText(info)

    def _update_features_info(self):
        """Actualizar información de características"""
        n_selected = len(self.get_selected_features())
        n_total = self.features_list.count()
        self.features_info_label.setText(f"{n_selected} de {n_total} variables seleccionadas")

    def _select_all_features(self):
        """Seleccionar todas"""
        for i in range(self.features_list.count()):
            self.features_list.item(i).setSelected(True)

    def _select_none_features(self):
        """Deseleccionar todas"""
        for i in range(self.features_list.count()):
            self.features_list.item(i).setSelected(False)

    def _auto_select_features(self):
        """Selección automática"""
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
                f"Se seleccionaron las {selected_count} variables con mayor correlación"
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

# ==================== WIDGET DE RESULTADOS ====================

class ResultsVisualizationWidget(QWidget):
    """Widget para visualización de resultados"""

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

        # Canvas para matplotlib
        self.figure = Figure(figsize=(10, 8))
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
            info_text += f"Variable objetivo: {results['target_column']}\n"

        if 'feature_columns' in results:
            info_text += f"Características: {len(results['feature_columns'])}\n"

        self.model_info_text.setText(info_text)

        # Parámetros
        params = results.get('parametros', {})
        self.params_table.setRowCount(len(params))

        for i, (param, value) in enumerate(params.items()):
            self.params_table.setItem(i, 0, QTableWidgetItem(param))
            self.params_table.setItem(i, 1, QTableWidgetItem(str(value)))

        self.params_table.resizeColumnsToContents()

    def _update_visualization(self):
        """Actualizar visualización"""
        if not self.current_results:
            return

        try:
            self.figure.clear()
            self.figure = generar_visualizaciones_ml(
                self.current_results,
                figsize=(12, 10)
            )
            self.canvas.figure = self.figure
            self.canvas.draw()
        except Exception as e:
            print(f"Error en visualización: {e}")

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

# ==================== VENTANA PRINCIPAL ====================

class SupervisadoWindow(QWidget, ThemedWidget):
    """Ventana principal para ML Supervisado"""

    def __init__(self):
        QWidget.__init__(self)
        ThemedWidget.__init__(self)

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
            else:
                print("⚠️ SupervisadoWindow: DataManager no disponible")
        else:
            print("⚠️ SupervisadoWindow: DataManager no importado")

        self.setup_ui()

        # Verificar datos al inicio
        self.check_data_availability()
        print("✅ SupervisadoWindow: Inicialización completada")

    def setup_ui(self):
        """Configurar interfaz de usuario"""
        self.setWindowTitle("🤖 Machine Learning Supervisado")
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

        # Contenido principal con splitter
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
        print(f"🔔 SupervisadoWindow: Recibida notificación '{event_type}'")

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
            self.reg_simple_btn, self.reg_multiple_btn, self.tree_btn,
            self.forest_btn, self.svm_btn, self.compare_btn
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

        title = QLabel("🤖 Machine Learning Supervisado")
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

        self.history_btn = QPushButton("📜 Historial")
        self.history_btn.clicked.connect(self.show_history)
        layout.addWidget(self.history_btn)

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

        # Opciones de análisis
        options_group = QGroupBox("⚙️ Opciones de Análisis")
        options_layout = QVBoxLayout()

        # Opciones básicas
        self.optimize_checkbox = QCheckBox("Optimizar hiperparámetros automáticamente")
        self.optimize_checkbox.setChecked(True)
        options_layout.addWidget(self.optimize_checkbox)

        self.cv_folds_spin = QSpinBox()
        self.cv_folds_spin.setRange(3, 10)
        self.cv_folds_spin.setValue(5)
        self.cv_folds_spin.setPrefix("CV Folds: ")
        options_layout.addWidget(self.cv_folds_spin)

        self.test_size_spin = QDoubleSpinBox()
        self.test_size_spin.setRange(0.1, 0.5)
        self.test_size_spin.setValue(0.2)
        self.test_size_spin.setSingleStep(0.05)
        self.test_size_spin.setPrefix("Test size: ")
        self.test_size_spin.setSuffix("%")
        self.test_size_spin.setDecimals(0)
        self.test_size_spin.setValue(20)
        options_layout.addWidget(self.test_size_spin)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Botones de análisis
        analysis_group = QGroupBox("🚀 Ejecutar Análisis")
        analysis_layout = QVBoxLayout()

        # Grid de modelos
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

        self.svm_btn = QPushButton("🔷 SVM")
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
                padding: 12px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #6c757d;
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

        # Obtener configuración básica
        target = self.variable_selection.get_target_variable()
        features = self.variable_selection.get_selected_features()

        # Configuración base común
        base_kwargs = {
            'target_column': target,
            'feature_columns': features,
            'optimize_params': self.optimize_checkbox.isChecked()
        }

        # Configuración específica por tipo de análisis
        if analysis_type == 'regresion_simple':
            if len(features) != 1:
                QMessageBox.warning(
                    self, "Error",
                    "La regresión simple requiere exactamente una variable predictora"
                )
                return
            kwargs = {
                'x_column': features[0],
                'y_column': target,
                'optimize_params': self.optimize_checkbox.isChecked()
            }
        elif analysis_type == 'regresion_multiple':
            kwargs = {
                **base_kwargs,
                'regularization': 'none',  # Por defecto sin regularización
                'alpha': 1.0
            }
        elif analysis_type == 'arbol_decision':
            kwargs = {
                **base_kwargs,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            }
        elif analysis_type == 'random_forest':
            kwargs = {
                **base_kwargs,
                'n_estimators': 100,
                'max_depth': None,
                'max_features': 'sqrt',
                'n_jobs': -1
            }
        elif analysis_type == 'svm':
            kwargs = {
                **base_kwargs,
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale'
            }
        else:
            kwargs = base_kwargs

        # Mostrar progreso
        self.show_progress(True)
        self.log(f"🚀 Iniciando análisis: {analysis_type}")

        # Crear worker
        self.current_worker = MLAnalysisWorker(analysis_type, self.current_data, **kwargs)

        # Conectar señales
        self.current_worker.progress.connect(self.progress_bar.setValue)
        self.current_worker.status.connect(self.log)
        self.current_worker.finished.connect(self.on_analysis_finished)
        self.current_worker.error.connect(self.on_analysis_error)
        self.current_worker.log.connect(self.log)

        # Iniciar
        self.current_worker.start()

    def compare_models(self):
        """Comparar múltiples modelos"""
        if not self.validate_selection():
            return

        if not ML_AVAILABLE:
            QMessageBox.critical(self, "Error", "Funcionalidad ML no disponible")
            return

        # Diálogo de selección
        dialog = QDialog(self)
        dialog.setWindowTitle("Seleccionar Modelos")
        dialog.setModal(True)
        dialog.resize(400, 300)

        layout = QVBoxLayout()

        layout.addWidget(QLabel("Selecciona los modelos a comparar:"))

        # Checkboxes
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

        # Botones
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
                "Selecciona una variable objetivo y al menos una variable predictora"
            )
            return False

        if ML_AVAILABLE:
            # Verificar calidad de datos
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

    def show_history(self):
        """Mostrar historial de análisis"""
        if not self.analysis_history:
            QMessageBox.information(self, "Historial", "No hay análisis previos")
            return

        history_text = "📜 Historial de Análisis:\n\n"
        for i, entry in enumerate(reversed(self.analysis_history[-10:])):
            history_text += f"{i+1}. {entry['timestamp'].strftime('%H:%M:%S')} - "
            history_text += f"{entry['type']} (objetivo: {entry['target']})\n"

        QMessageBox.information(self, "Historial", history_text)

    def show_help(self):
        """Mostrar ayuda"""
        help_dialog = QDialog(self)
        help_dialog.setWindowTitle("Ayuda - Machine Learning Supervisado")
        help_dialog.setModal(True)
        help_dialog.resize(600, 500)

        layout = QVBoxLayout()

        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
        <h2>🤖 Machine Learning Supervisado</h2>
        
        <h3>📌 Pasos Básicos:</h3>
        <ol>
        <li>Asegúrate de que los datos estén cargados desde el módulo "Cargar Datos"</li>
        <li>Selecciona la variable objetivo (lo que quieres predecir)</li>
        <li>Selecciona las variables predictoras</li>
        <li>Configura las opciones de análisis</li>
        <li>Ejecuta el análisis deseado</li>
        </ol>

        <h3>📊 Tipos de Problemas:</h3>
        <ul>
        <li><b>Regresión:</b> Predecir valores numéricos continuos</li>
        <li><b>Clasificación:</b> Predecir categorías o clases</li>
        </ul>

        <h3>🔧 Modelos Disponibles:</h3>
        <ul>
        <li><b>Regresión Simple:</b> Relación entre dos variables</li>
        <li><b>Regresión Múltiple:</b> Múltiples predictores</li>
        <li><b>Árbol de Decisión:</b> Reglas de decisión</li>
        <li><b>Random Forest:</b> Conjunto de árboles</li>
        <li><b>SVM:</b> Support Vector Machine</li>
        </ul>

        <h3>💡 Consejos:</h3>
        <ul>
        <li>Usa "Selección Automática" para elegir variables relevantes</li>
        <li>Activa "Optimizar hiperparámetros" para mejores resultados</li>
        <li>Compara múltiples modelos para encontrar el mejor</li>
        <li>Revisa las métricas para evaluar el rendimiento</li>
        </ul>
        """)

        layout.addWidget(help_text)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(help_dialog.accept)
        layout.addWidget(buttons)

        help_dialog.setLayout(layout)
        help_dialog.exec_()

    def apply_styles(self):
        """Aplicar estilos CSS"""
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
                background-color: #f8f9fa;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #ffffff;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #495057;
            }
            
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
                min-height: 30px;
            }
            
            QPushButton:hover {
                background-color: #0056b3;
            }
            
            QPushButton:disabled {
                background-color: #6c757d;
                color: #adb5bd;
            }
            
            QComboBox, QSpinBox, QDoubleSpinBox {
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 6px;
                background-color: white;
                min-height: 25px;
            }
            
            QListWidget {
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
                selection-background-color: #007bff;
            }
            
            QTableWidget {
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
                gridline-color: #dee2e6;
                selection-background-color: #007bff;
            }
            
            QTableWidget::item {
                padding: 6px;
                border-bottom: 1px solid #dee2e6;
            }
            
            QTableWidget QHeaderView::section {
                background-color: #e9ecef;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
            
            QTextEdit {
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
                padding: 8px;
            }
            
            QProgressBar {
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: #e9ecef;
                text-align: center;
                font-weight: bold;
            }
            
            QProgressBar::chunk {
                background-color: #007bff;
                border-radius: 3px;
            }
            
            QFrame[frameShape="6"] {
                border: 1px solid #dee2e6;
                border-radius: 8px;
                background-color: white;
                margin: 2px;
            }
        """)

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

    # Crear datos de prueba si no hay DataManager
    if not DATA_MANAGER_AVAILABLE:
        print("⚠️ Ejecutando en modo de prueba sin DataManager")

    window = SupervisadoWindow()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()