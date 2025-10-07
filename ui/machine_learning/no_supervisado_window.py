"""
no_supervisado_window.py - Compatible con PyInstaller
Ventana principal para Machine Learning No Supervisado
Versión optimizada para empaquetado, sin dependencias problemáticas
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
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QColor

from ui.machine_learning.parallel_plotting import ParallelPlotter

# Importar gestor de datos con manejo de errores robusto
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

# Importar funciones ML No Supervisado optimizadas
ML_AVAILABLE = True
try:
    from .ml_functions_no_supervisado import (
        generar_datos_agua_realistas,
        kmeans_optimizado_completo,
        dbscan_optimizado,
        analisis_exploratorio_completo,
        pca_completo_avanzado,
        clustering_jerarquico_completo,
        generar_visualizaciones_ml_no_supervisado, plot_dendrogram_manual_mejorado, aplicar_escalado, manual_pca
)

    # Importar matplotlib con configuración optimizada
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

    # Crear mocks para evitar errores
    try:
        import matplotlib
        matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
    except ImportError:
        # Mocks básicos si matplotlib no está disponible
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

            def add_gridspec(self, *args, **kwargs):
                return MockGridSpec()

            def suptitle(self, *args, **kwargs):
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

            def scatter(self, *args, **kwargs):
                return None

            def plot(self, *args, **kwargs):
                return []

            def bar(self, *args, **kwargs):
                return []

            def barh(self, *args, **kwargs):
                return []

            def pie(self, *args, **kwargs):
                pass

            def imshow(self, *args, **kwargs):
                return None

            def set_xlabel(self, *args, **kwargs):
                pass

            def set_ylabel(self, *args, **kwargs):
                pass

            def legend(self, *args, **kwargs):
                pass

            def grid(self, *args, **kwargs):
                pass

            def axhline(self, *args, **kwargs):
                pass

            def set_xticks(self, *args, **kwargs):
                pass

            def set_yticks(self, *args, **kwargs):
                pass

            def set_xticklabels(self, *args, **kwargs):
                pass

            def set_yticklabels(self, *args, **kwargs):
                pass

            @property
            def transAxes(self):
                return None

        class MockGridSpec:
            def __init__(self, *args, **kwargs):
                pass

        plt = type('MockPlt', (), {
            'cm': type('MockCm', (), {
                'tab10': lambda x: ['blue'] * len(x) if hasattr(x, '__len__') else ['blue'],
                'Spectral': lambda x: ['blue'] * len(x) if hasattr(x, '__len__') else ['blue']
            })(),
            'colorbar': lambda *args, **kwargs: None,
            'close': lambda *args, **kwargs: None
        })()

    # Crear funciones mock para las funciones ML
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


class ThreadMonitorWidget(QWidget):
    """Widget para monitorear hilos en tiempo real"""

    def __init__(self, max_threads=12):
        super().__init__()
        self.max_threads = max_threads
        self.thread_states = [0] * max_threads  # 0: inactivo, 1: activo
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Título
        title = QLabel("Monitor de Hilos de Procesamiento")
        title.setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 11px;")
        layout.addWidget(title)

        # Grid para los hilos
        self.threads_layout = QGridLayout()
        self.threads_layout.setSpacing(3)

        self.thread_indicators = []
        cols = 6  # 6 columnas

        for i in range(self.max_threads):
            row = i // cols
            col = i % cols

            # Contenedor para cada hilo
            thread_frame = QFrame()
            thread_frame.setFixedSize(70, 35)
            thread_layout = QVBoxLayout()
            thread_layout.setContentsMargins(2, 2, 2, 2)
            thread_layout.setSpacing(1)

            # Label del número
            label = QLabel(f"Hilo {i + 1}")
            label.setStyleSheet("font-size: 8px; color: #666;")
            label.setAlignment(Qt.AlignCenter)

            # Indicador visual
            indicator = QLabel()
            indicator.setFixedSize(60, 10)
            indicator.setStyleSheet("""
                background-color: #95a5a6;
                border-radius: 3px;
            """)

            thread_layout.addWidget(label)
            thread_layout.addWidget(indicator)
            thread_frame.setLayout(thread_layout)

            self.threads_layout.addWidget(thread_frame, row, col)
            self.thread_indicators.append((indicator, label))

        layout.addLayout(self.threads_layout)

        # Estadísticas
        self.stats_label = QLabel("Hilos activos: 0/12 | Eficiencia: 0%")
        self.stats_label.setStyleSheet("font-size: 9px; color: #34495e;")
        layout.addWidget(self.stats_label)

        self.setLayout(layout)
        self.setMaximumHeight(120)

    def update_thread_state(self, thread_id, active):
        """Actualizar estado de un hilo específico"""
        if 0 <= thread_id < self.max_threads:
            self.thread_states[thread_id] = 1 if active else 0
            self._update_display()

    def update_active_threads(self, active_list):
        """Actualizar lista de hilos activos"""
        self.thread_states = [0] * self.max_threads
        for thread_id in active_list:
            if 0 <= thread_id < self.max_threads:
                self.thread_states[thread_id] = 1
        self._update_display()

    def _update_display(self):
        """Actualizar visualización"""
        active_count = sum(self.thread_states)
        efficiency = (active_count / self.max_threads) * 100 if self.max_threads > 0 else 0

        for i, (indicator, label) in enumerate(self.thread_indicators):
            if self.thread_states[i] == 1:
                # Hilo activo - verde pulsante
                indicator.setStyleSheet("""
                    background-color: #27ae60;
                    border-radius: 3px;
                    border: 1px solid #229954;
                """)
                label.setStyleSheet("font-size: 8px; color: #27ae60; font-weight: bold;")
            else:
                # Hilo inactivo - gris
                indicator.setStyleSheet("""
                    background-color: #95a5a6;
                    border-radius: 3px;
                """)
                label.setStyleSheet("font-size: 8px; color: #666;")

        self.stats_label.setText(
            f"Hilos activos: {active_count}/{self.max_threads} | "
            f"Eficiencia: {efficiency:.0f}%"
        )

    def reset(self):
        """Resetear todos los hilos a inactivo"""
        self.thread_states = [0] * self.max_threads
        self._update_display()

# ==================== WORKER THREAD PARA ML NO SUPERVISADO ====================

class MLNoSupervisadoWorker(QThread):
    """Worker thread para análisis ML No Supervisado optimizado"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    log = pyqtSignal(str)
    thread_activity = pyqtSignal(list)  # NUEVA SEÑAL para monitor de hilos

    def __init__(self, analysis_type: str, data: pd.DataFrame, **kwargs):
        super().__init__()
        self.analysis_type = analysis_type
        self.data = data.copy()
        self.kwargs = kwargs
        self._is_cancelled = False

    def run(self):
        """Ejecutar análisis con manejo robusto de errores"""
        try:
            if not ML_AVAILABLE:
                raise ImportError("Librerías de ML No Supervisado no disponibles")

            self.log.emit(f"Iniciando {self.analysis_type}")
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
                self.log.emit("Análisis cancelado")
                return

            self.progress.emit(100)
            self.status.emit("Análisis completado")
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))
            self.log.emit(f"Error: {str(e)}")
            print(f"Error en worker: {e}")
            print(traceback.format_exc())

    def cancel(self):
        """Cancelar análisis"""
        self._is_cancelled = True

    def _run_clustering_jerarquico(self):
        """Ejecutar clustering jerárquico con monitor de hilos"""
        try:
            self.status.emit("Ejecutando clustering jerárquico optimizado...")
            self.progress.emit(20)

            from .ml_functions_no_supervisado import (
                ParallelHierarchicalClustering,
                aplicar_escalado,
                analizar_clusters_manual,
                manual_silhouette_score,
                get_clusters_from_linkage
            )

            variables = self.kwargs.get('variables', [])
            X = self.data[variables].dropna()

            self.log.emit(f"Preparados: {X.shape[0]} muestras, {X.shape[1]} variables")

            X_scaled, scaler_info = aplicar_escalado(X, self.kwargs.get('escalado', 'standard'))

            # Crear instancia
            parallel_cluster = ParallelHierarchicalClustering()

            # Callbacks básicos
            def emit_progress(value):
                if not self._is_cancelled:
                    try:
                        self.progress.emit(20 + int(value * 0.6))
                    except:
                        pass

            def emit_status(message):
                if not self._is_cancelled:
                    try:
                        self.status.emit(message)
                        self.log.emit(message)
                    except:
                        pass

            def emit_thread_activity(active_threads):
                if not self._is_cancelled:
                    try:
                        self.thread_activity.emit(active_threads)
                    except:
                        pass

            # Conectar callbacks
            parallel_cluster.set_callbacks(emit_progress, emit_status, emit_thread_activity)

            self.log.emit(f"Usando {parallel_cluster.max_workers} hilos...")

            # Ejecutar
            linkage_matrix = parallel_cluster.hierarchical_clustering_optimized(
                X_scaled.values,
                method=self.kwargs.get('hierarchical_method', 'ward'),
                metric=self.kwargs.get('hierarchical_metric', 'euclidean')
            )

            if self._is_cancelled:
                return None

            self.progress.emit(80)

            # Evaluar K
            k_range = range(2, min(self.kwargs.get('hierarchical_max_clusters', 10) + 1, len(X)))
            resultados_por_k = {}

            for i, k in enumerate(k_range):
                if self._is_cancelled:
                    return None

                labels = get_clusters_from_linkage(linkage_matrix, k)

                if len(set(labels)) > 1:
                    silhouette = manual_silhouette_score(X_scaled.values, np.array(labels))
                else:
                    silhouette = 0.0

                cluster_analysis = analizar_clusters_manual(X, labels, variables)

                resultados_por_k[k] = {
                    'labels': labels,
                    'silhouette_score': silhouette,
                    'cluster_stats': cluster_analysis
                }

                progress = 80 + int((i + 1) / len(k_range) * 15)
                self.progress.emit(progress)

            mejor_k = max(resultados_por_k.keys(),
                          key=lambda k: resultados_por_k[k]['silhouette_score'])

            self.progress.emit(95)
            self.log.emit(f"K óptimo: {mejor_k}")

            # Limpiar monitor
            try:
                self.thread_activity.emit([])
            except:
                pass

            return {
                'tipo': 'clustering_jerarquico_completo',
                'variables_utilizadas': variables,
                'metodo_escalado': self.kwargs.get('escalado', 'standard'),
                'linkage_matrix': linkage_matrix.tolist(),
                'resultados_por_k': resultados_por_k,
                'mejor_configuracion': {
                    'metodo': self.kwargs.get('hierarchical_method', 'ward'),
                    'metrica': self.kwargs.get('hierarchical_metric', 'euclidean'),
                    'n_clusters_sugeridos': mejor_k,
                    'silhouette_score': resultados_por_k[mejor_k]['silhouette_score'],
                    'labels': resultados_por_k[mejor_k]['labels']
                },
                'datos_originales': X,
                'scaler_info': scaler_info,
                'sample_labels': [f"S{i}" for i in range(len(X))],
                'recomendaciones': [
                    f"Mejor: {self.kwargs.get('hierarchical_method', 'ward')}-"
                    f"{self.kwargs.get('hierarchical_metric', 'euclidean')} con {mejor_k} clusters",
                    f"Silhouette: {resultados_por_k[mejor_k]['silhouette_score']:.3f}",
                    f"Hilos: {parallel_cluster.max_workers}",
                ]
            }

        except Exception as e:
            self.log.emit(f"Error crítico: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _run_kmeans_optimizado(self):
        """Ejecutar K-Means optimizado"""
        self.status.emit("Ejecutando K-Means optimizado...")
        self.progress.emit(30)

        kwargs = {
            'variables': self.kwargs.get('variables', []),
            'k_range': self.kwargs.get('kmeans_k_range', range(2, 9)),
            'escalado': self.kwargs.get('escalado', 'standard'),
            'random_state': self.kwargs.get('random_state', 42),
            'verbose': self.kwargs.get('verbose', True)
        }

        resultado = kmeans_optimizado_completo(self.data, **kwargs)
        return resultado

    def _run_dbscan(self):
        """Ejecutar DBSCAN"""
        self.status.emit("Ejecutando DBSCAN...")
        self.progress.emit(30)

        kwargs = {
            'variables': self.kwargs.get('variables', []),
            'optimizar_parametros': self.kwargs.get('dbscan_optimize', True),
            'escalado': self.kwargs.get('escalado', 'standard'),
            'verbose': self.kwargs.get('verbose', True)
        }

        resultado = dbscan_optimizado(self.data, **kwargs)
        return resultado

    def _run_pca_avanzado(self):
        """Ejecutar PCA avanzado"""
        self.status.emit("Ejecutando PCA avanzado...")
        self.progress.emit(30)

        kwargs = {
            'variables': self.kwargs.get('variables', []),
            'explicar_varianza_objetivo': self.kwargs.get('pca_variance_threshold', 0.95),
            'escalado': self.kwargs.get('escalado', 'standard'),
            'verbose': self.kwargs.get('verbose', True)
        }

        resultado = pca_completo_avanzado(self.data, **kwargs)
        return resultado

    def _run_analisis_exploratorio(self):
        """Ejecutar análisis exploratorio"""
        self.status.emit("Ejecutando análisis exploratorio...")
        self.progress.emit(30)

        kwargs = {
            'variables': self.kwargs.get('variables', []),
            'escalado': self.kwargs.get('escalado', 'standard'),
            'handle_outliers': self.kwargs.get('handle_outliers', True),
            'verbose': self.kwargs.get('verbose', True)
        }

        resultado = analisis_exploratorio_completo(self.data, **kwargs)
        return resultado

# ==================== WIDGET DE SELECCIÓN DE VARIABLES ====================

class VariableSelectionWidget(QWidget):
    """Widget para selección de variables optimizado"""
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

        # Obtener columnas numéricas, excluyendo las no relevantes
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

        # Seleccionar variables con menos del 20% de valores faltantes y varianza suficiente
        selected_count = 0

        for i in range(self.variables_list.count()):
            item = self.variables_list.item(i)
            col_name = item.data(Qt.UserRole)

            col_data = self.data[col_name]
            missing_pct = (col_data.isnull().sum() / len(col_data)) * 100

            # Verificar varianza
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
    """Widget para configuración de análisis optimizado"""

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

        # DBSCAN
        self.dbscan_optimize = QCheckBox("Optimizar parámetros DBSCAN automáticamente")
        self.dbscan_optimize.setChecked(True)
        self.dbscan_optimize.setMinimumHeight(30)
        clustering_layout.addRow("", self.dbscan_optimize)

        # Clustering Jerárquico
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

            # Preprocesamiento
            'scaling_method': self.scaling_method.currentText(),
            'handle_outliers': self.handle_outliers.isChecked(),

            # General
            'random_state': self.random_state.value(),
            'verbose': self.verbose_output.isChecked()
        }

# ==================== WIDGET DE RESULTADOS ====================
class ResultsVisualizationWidget(QWidget):
    """Widget para visualización de resultados con graficación paralela optimizada"""

    def __init__(self):
        super().__init__()
        self.current_results = None
        self.current_analysis_type = None

        # Sistema de graficación paralela
        self.plotter = ParallelPlotter(max_workers=4)
        self.plot_results = {}

        self.setup_ui()
        self._setup_plotter_callbacks()

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

        # Barra de progreso para graficación paralela
        self.plot_progress_bar = QProgressBar()
        self.plot_progress_bar.setVisible(False)
        self.plot_progress_bar.setFormat("Generando gráficos: %p%")
        layout.addWidget(self.plot_progress_bar)

        # Monitor de hilos para graficación
        self.plot_thread_monitor = ThreadMonitorWidget(max_threads=4)
        self.plot_thread_monitor.setVisible(False)
        layout.addWidget(self.plot_thread_monitor)

        # Toolbar
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)

        self.setLayout(layout)

    def _setup_plotter_callbacks(self):
        """Configurar callbacks del sistema de graficación paralela"""

        def on_plot_progress(progress):
            self.plot_progress_bar.setValue(int(progress))

        def on_plot_status(msg):
            # Opcional: mostrar en un label de estado
            pass

        def on_plot_threads(active_threads):
            self.plot_thread_monitor.update_active_threads(active_threads)

        def on_plot_complete(task_id, result, error):
            if error:
                print(f"Error en tarea de graficación {task_id}: {error}")
            else:
                self.plot_results[task_id] = result

        self.plotter.set_callbacks(
            progress_callback=on_plot_progress,
            status_callback=on_plot_status,
            thread_monitor_callback=on_plot_threads,
            task_complete_callback=on_plot_complete
        )

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
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        # Selector de gráficos generados
        plot_selector_layout = QHBoxLayout()

        plot_selector_label = QLabel("Gráfico:")
        plot_selector_layout.addWidget(plot_selector_label)

        self.plot_selector = QComboBox()
        self.plot_selector.currentIndexChanged.connect(self._on_plot_selected)
        plot_selector_layout.addWidget(self.plot_selector)

        plot_selector_layout.addStretch()
        layout.addLayout(plot_selector_layout)

        # Controles
        controls_layout = QHBoxLayout()

        self.regenerate_parallel_btn = QPushButton("🚀 Regenerar en Paralelo")
        self.regenerate_parallel_btn.clicked.connect(self._regenerate_plots_parallel)
        self.regenerate_parallel_btn.setEnabled(False)
        controls_layout.addWidget(self.regenerate_parallel_btn)

        self.save_fig_btn = QPushButton("💾 Guardar Gráfico")
        self.save_fig_btn.clicked.connect(self._save_figure)
        controls_layout.addWidget(self.save_fig_btn)

        self.save_all_btn = QPushButton("💾 Guardar Todos")
        self.save_all_btn.clicked.connect(self._save_all_figures)
        self.save_all_btn.setEnabled(False)
        controls_layout.addWidget(self.save_all_btn)

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
            # Generar visualizaciones en paralelo
            self._update_visualization_parallel()

        self.status_label.setText(f"✅ {analysis_type} completado")
        self.status_label.setStyleSheet("color: green;")

        # Habilitar botones
        self.export_results_btn.setEnabled(True)
        self.generate_report_btn.setEnabled(True)
        self.regenerate_parallel_btn.setEnabled(True)

    def _update_visualization_parallel(self):
        """Actualizar visualización usando procesamiento paralelo"""
        if not self.current_results or not ML_AVAILABLE:
            return

        # Mostrar controles de progreso
        self.plot_progress_bar.setVisible(True)
        self.plot_thread_monitor.setVisible(True)
        self.plot_thread_monitor.reset()

        # Limpiar tareas previas
        self.plotter.clear_tasks()
        self.plot_results.clear()
        self.plot_selector.clear()

        tipo = self.current_results.get('tipo', '')

        try:
            # Agregar tareas según el tipo de análisis
            if tipo == 'kmeans_optimizado':
                self._add_kmeans_plots()
            elif tipo == 'dbscan_optimizado':
                self._add_dbscan_plots()
            elif tipo == 'pca_completo_avanzado':
                self._add_pca_plots()
            elif tipo == 'clustering_jerarquico_completo':
                self._add_hierarchical_plots()
            elif tipo == 'analisis_exploratorio_completo':
                self._add_exploratory_plots()
            else:
                # Fallback a visualización tradicional
                self._update_visualization_traditional()
                return

            # Ejecutar tareas en paralelo
            results = self.plotter.execute_all()

            # Actualizar selector de gráficos
            for task_id in sorted(results.keys()):
                task = self.plotter.tasks[task_id]
                self.plot_selector.addItem(task.description, task_id)

            # Mostrar primer gráfico
            if results:
                first_fig = list(results.values())[0]
                self.canvas.figure = first_fig
                self.canvas.draw()

            self.save_all_btn.setEnabled(len(results) > 0)

        except Exception as e:
            print(f"Error en visualización paralela: {e}")
            import traceback
            traceback.print_exc()
            self._update_visualization_traditional()

        finally:
            # Ocultar controles de progreso
            self.plot_progress_bar.setVisible(False)
            self.plot_thread_monitor.setVisible(False)

    def _add_kmeans_plots(self):
        """Agregar tareas de graficación para K-Means"""
        # Gráfico 1: Evaluación de K
        self.plotter.add_task(
            self._plot_kmeans_evaluation,
            self.current_results,
            {'figsize': (8, 6)},
            "K-Means: Evaluación K"
        )

        # Gráfico 2: Distribución clusters
        self.plotter.add_task(
            self._plot_cluster_distribution,
            self.current_results,
            {'figsize': (8, 6)},
            "K-Means: Distribución"
        )

        # Gráfico 3: Clusters en PCA
        self.plotter.add_task(
            self._plot_clusters_pca,
            self.current_results,
            {'figsize': (10, 8)},
            "K-Means: Vista PCA"
        )

    def _add_dbscan_plots(self):
        """Agregar tareas de graficación para DBSCAN"""
        self.plotter.add_task(
            self._plot_dbscan_info,
            self.current_results,
            {'figsize': (8, 6)},
            "DBSCAN: Información"
        )

        self.plotter.add_task(
            self._plot_cluster_distribution,
            self.current_results,
            {'figsize': (8, 6)},
            "DBSCAN: Distribución"
        )

        self.plotter.add_task(
            self._plot_clusters_pca,
            self.current_results,
            {'figsize': (10, 8)},
            "DBSCAN: Vista PCA"
        )

    def _add_pca_plots(self):
        """Agregar tareas de graficación para PCA"""
        self.plotter.add_task(
            self._plot_pca_variance,
            self.current_results,
            {'figsize': (8, 6)},
            "PCA: Varianza Explicada"
        )

        self.plotter.add_task(
            self._plot_pca_cumulative,
            self.current_results,
            {'figsize': (8, 6)},
            "PCA: Varianza Acumulada"
        )

        self.plotter.add_task(
            self._plot_pca_loadings,
            self.current_results,
            {'figsize': (10, 6)},
            "PCA: Loadings"
        )

        self.plotter.add_task(
            self._plot_pca_scree,
            self.current_results,
            {'figsize': (8, 6)},
            "PCA: Scree Plot"
        )

    def _add_hierarchical_plots(self):
        """Agregar tareas de graficación para clustering jerárquico"""
        self.plotter.add_task(
            self._plot_hierarchical_evaluation,
            self.current_results,
            {'figsize': (8, 6)},
            "Jerárquico: Evaluación K"
        )

        self.plotter.add_task(
            self._plot_dendrogram,
            self.current_results,
            {'figsize': (12, 6)},
            "Jerárquico: Dendrograma"
        )

        self.plotter.add_task(
            self._plot_clusters_pca,
            self.current_results,
            {'figsize': (10, 8)},
            "Jerárquico: Vista PCA"
        )

    def _add_exploratory_plots(self):
        """Agregar tareas de graficación para análisis exploratorio"""
        self.plotter.add_task(
            self._plot_correlation_matrix,
            self.current_results,
            {'figsize': (10, 8)},
            "Exploratorio: Correlaciones"
        )

        self.plotter.add_task(
            self._plot_outliers_distribution,
            self.current_results,
            {'figsize': (8, 6)},
            "Exploratorio: Outliers"
        )

        self.plotter.add_task(
            self._plot_data_quality,
            self.current_results,
            {'figsize': (8, 8)},
            "Exploratorio: Calidad Datos"
        )

    # ==================== FUNCIONES DE GRAFICACIÓN ====================

    def _plot_kmeans_evaluation(self, fig, data, config, progress_callback):
        """Gráfico de evaluación de K para K-Means"""
        if progress_callback:
            progress_callback(10)

        ax = fig.add_subplot(111)

        k_vals = list(data['resultados_por_k'].keys())
        silhouette_vals = [data['resultados_por_k'][k]['silhouette_score'] for k in k_vals]

        if progress_callback:
            progress_callback(50)

        ax.plot(k_vals, silhouette_vals, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Número de Clusters (K)')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Evaluación de K óptimo')
        ax.grid(True, alpha=0.3)

        # Marcar K óptimo
        k_opt = data.get('recomendacion_k')
        if k_opt in data['resultados_por_k']:
            best_score = data['resultados_por_k'][k_opt]['silhouette_score']
            ax.plot(k_opt, best_score, 'ro', markersize=12, label=f'K óptimo = {k_opt}')
            ax.legend()

        fig.tight_layout()

        if progress_callback:
            progress_callback(100)

        return fig

    def _plot_cluster_distribution(self, fig, data, config, progress_callback):
        """Gráfico de distribución por cluster"""
        if progress_callback:
            progress_callback(10)

        ax = fig.add_subplot(111)

        tipo = data.get('tipo', '')

        if tipo == 'kmeans_optimizado':
            k_opt = data.get('recomendacion_k')
            if k_opt and k_opt in data['resultados_por_k']:
                labels = data['resultados_por_k'][k_opt]['labels']
        elif tipo == 'dbscan_optimizado':
            labels = data['mejor_configuracion']['labels']
        else:
            labels = []

        if progress_callback:
            progress_callback(50)

        if labels:
            unique_labels = np.unique(labels)
            tamaños = [labels.count(label) for label in unique_labels]

            bars = ax.bar(range(len(unique_labels)), tamaños, alpha=0.7)
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Número de Puntos')
            ax.set_title('Distribución por Cluster')
            ax.set_xticks(range(len(unique_labels)))
            ax.set_xticklabels([f'C{label}' if label != -1 else 'Outliers'
                                for label in unique_labels])

            # Añadir valores
            for bar, tamaño in zip(bars, tamaños):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                        f'{tamaño}', ha='center', va='bottom')

        fig.tight_layout()

        if progress_callback:
            progress_callback(100)

        return fig

    def _plot_clusters_pca(self, fig, data, config, progress_callback):
        """Gráfico de clusters en espacio PCA"""
        if progress_callback:
            progress_callback(10)

        ax = fig.add_subplot(111)

        try:
            datos = data['datos_originales']

            # Obtener labels
            tipo = data.get('tipo', '')
            if tipo == 'kmeans_optimizado':
                k_opt = data.get('recomendacion_k')
                labels = data['resultados_por_k'][k_opt]['labels'] if k_opt else []
            elif tipo == 'dbscan_optimizado':
                labels = data['mejor_configuracion']['labels']
            elif tipo == 'clustering_jerarquico_completo':
                labels = data['mejor_configuracion']['labels']
            else:
                labels = [0] * len(datos)

            if progress_callback:
                progress_callback(40)

            # PCA para 2D
            X_scaled, _ = aplicar_escalado(datos, 'standard')
            pca_result = manual_pca(X_scaled.values, n_components=2)
            datos_2d = pca_result['X_transformed']

            if progress_callback:
                progress_callback(70)

            # Graficar
            unique_labels = set(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                mask = np.array(labels) == label

                if label == -1:
                    ax.scatter(datos_2d[mask, 0], datos_2d[mask, 1],
                               c='black', marker='x', s=100, alpha=0.8, label='Outliers')
                else:
                    ax.scatter(datos_2d[mask, 0], datos_2d[mask, 1],
                               c=[color], label=f'Cluster {label}',
                               s=60, alpha=0.7, edgecolors='black', linewidth=0.5)

            var_exp = pca_result['explained_variance_ratio']
            ax.set_xlabel(f'PC1 ({var_exp[0] * 100:.1f}%)')
            ax.set_ylabel(f'PC2 ({var_exp[1] * 100:.1f}%)')
            ax.set_title('Clusters en Espacio PCA')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}',
                    ha='center', va='center', transform=ax.transAxes)

        fig.tight_layout()

        if progress_callback:
            progress_callback(100)

        return fig

    def _plot_dbscan_info(self, fig, data, config, progress_callback):
        """Información de DBSCAN"""
        if progress_callback:
            progress_callback(50)

        ax = fig.add_subplot(111)

        mejor_config = data['mejor_configuracion']

        info_text = f"Parámetros DBSCAN:\n\n"
        info_text += f"Eps: {mejor_config['eps']:.3f}\n"
        info_text += f"Min Samples: {mejor_config['min_samples']}\n\n"
        info_text += f"Resultados:\n"
        info_text += f"Clusters: {mejor_config['n_clusters']}\n"
        info_text += f"Outliers: {mejor_config['n_noise']}\n"
        info_text += f"Silhouette: {mejor_config['silhouette_score']:.3f}"

        ax.text(0.05, 0.95, info_text, fontsize=10, va='top', ha='left',
                transform=ax.transAxes, family='monospace')
        ax.set_title('Configuración DBSCAN')
        ax.axis('off')

        fig.tight_layout()

        if progress_callback:
            progress_callback(100)

        return fig

    def _plot_pca_variance(self, fig, data, config, progress_callback):
        """Varianza explicada por componente"""
        if progress_callback:
            progress_callback(20)

        ax = fig.add_subplot(111)

        linear_result = data['resultados_por_metodo']['linear']
        analisis = linear_result['analisis']
        var_exp = analisis['varianza_explicada']

        if progress_callback:
            progress_callback(60)

        x = range(1, len(var_exp) + 1)
        bars = ax.bar(x, [v * 100 for v in var_exp], alpha=0.7, color='steelblue')

        ax.set_xlabel('Componente Principal')
        ax.set_ylabel('Varianza Explicada (%)')
        ax.set_title('Varianza por Componente')
        ax.grid(True, alpha=0.3)

        for i, (bar, val) in enumerate(zip(bars, var_exp)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                    f'{val * 100:.1f}%', ha='center', va='bottom', fontsize=9)

        fig.tight_layout()

        if progress_callback:
            progress_callback(100)

        return fig

    def _plot_pca_cumulative(self, fig, data, config, progress_callback):
        """Varianza acumulada"""
        if progress_callback:
            progress_callback(20)

        ax = fig.add_subplot(111)

        linear_result = data['resultados_por_metodo']['linear']
        analisis = linear_result['analisis']
        var_acum = analisis['varianza_acumulada']

        if progress_callback:
            progress_callback(60)

        x = range(1, len(var_acum) + 1)
        ax.plot(x, [v * 100 for v in var_acum], 'o-', linewidth=2, markersize=8, color='darkred')
        ax.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='95%')
        ax.axhline(y=85, color='orange', linestyle='--', alpha=0.7, label='85%')

        ax.set_xlabel('Componente Principal')
        ax.set_ylabel('Varianza Acumulada (%)')
        ax.set_title('Varianza Acumulada')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

        fig.tight_layout()

        if progress_callback:
            progress_callback(100)

        return fig

    def _plot_pca_loadings(self, fig, data, config, progress_callback):
        """Loadings de variables"""
        if progress_callback:
            progress_callback(20)

        ax = fig.add_subplot(111)

        linear_result = data['resultados_por_metodo']['linear']
        analisis = linear_result['analisis']
        componentes_info = analisis['componentes_info']

        if progress_callback:
            progress_callback(40)

        if len(componentes_info) >= 2:
            pc1_info = componentes_info[0]
            pc2_info = componentes_info[1]
            var_exp = analisis['varianza_explicada']

            top_vars_pc1 = pc1_info['top_variables'][:5]
            top_vars_pc2 = pc2_info['top_variables'][:5]

            all_vars = {}
            for var in top_vars_pc1:
                all_vars[var['variable']] = [var['loading'], 0]
            for var in top_vars_pc2:
                if var['variable'] in all_vars:
                    all_vars[var['variable']][1] = var['loading']
                else:
                    all_vars[var['variable']] = [0, var['loading']]

            if progress_callback:
                progress_callback(70)

            if all_vars:
                variables = list(all_vars.keys())
                pc1_loadings = [all_vars[var][0] for var in variables]
                pc2_loadings = [all_vars[var][1] for var in variables]

                x_pos = range(len(variables))
                width = 0.35

                ax.bar([x - width / 2 for x in x_pos], pc1_loadings, width,
                       label=f'PC1 ({var_exp[0] * 100:.1f}%)', alpha=0.8, color='steelblue')
                ax.bar([x + width / 2 for x in x_pos], pc2_loadings, width,
                       label=f'PC2 ({var_exp[1] * 100:.1f}%)', alpha=0.8, color='coral')

                ax.set_xlabel('Variables')
                ax.set_ylabel('Loading')
                ax.set_title('Loadings de Variables en PC1 y PC2')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(variables, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='black', linewidth=0.8)

        fig.tight_layout()

        if progress_callback:
            progress_callback(100)

        return fig

    def _plot_pca_scree(self, fig, data, config, progress_callback):
        """Scree plot"""
        if progress_callback:
            progress_callback(20)

        ax = fig.add_subplot(111)

        linear_result = data['resultados_por_metodo']['linear']
        analisis = linear_result['analisis']
        eigenvalues = analisis['eigenvalues']

        if progress_callback:
            progress_callback(60)

        x = range(1, len(eigenvalues) + 1)
        ax.plot(x, eigenvalues, 'o-', linewidth=2, markersize=8, color='purple')

        ax.set_xlabel('Componente Principal')
        ax.set_ylabel('Eigenvalue')
        ax.set_title('Scree Plot - Eigenvalues')
        ax.grid(True, alpha=0.3)

        if max(eigenvalues) > 1:
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Kaiser criterion (λ=1)')
            ax.legend()

        fig.tight_layout()

        if progress_callback:
            progress_callback(100)

        return fig

    def _plot_hierarchical_evaluation(self, fig, data, config, progress_callback):
        """Evaluación de K para jerárquico"""
        if progress_callback:
            progress_callback(20)

        ax = fig.add_subplot(111)

        resultados_por_k = data.get('resultados_por_k', {})

        if resultados_por_k:
            k_vals = list(resultados_por_k.keys())
            silhouette_vals = [resultados_por_k[k]['silhouette_score'] for k in k_vals]

            if progress_callback:
                progress_callback(60)

            ax.plot(k_vals, silhouette_vals, 'o-', linewidth=2, markersize=8, color='darkgreen')
            ax.set_xlabel('Número de Clusters (K)')
            ax.set_ylabel('Silhouette Score')
            ax.set_title('Evaluación de K óptimo')
            ax.grid(True, alpha=0.3)

            k_opt = data['mejor_configuracion'].get('n_clusters_sugeridos')
            if k_opt and k_opt in resultados_por_k:
                best_score = resultados_por_k[k_opt]['silhouette_score']
                ax.plot(k_opt, best_score, 'ro', markersize=12, label=f'K óptimo = {k_opt}')
                ax.legend()

        fig.tight_layout()

        if progress_callback:
            progress_callback(100)

        return fig

    def _plot_dendrogram(self, fig, data, config, progress_callback):
        """Dendrograma"""
        if progress_callback:
            progress_callback(10)

        ax = fig.add_subplot(111)

        linkage_matrix = data.get('linkage_matrix')
        sample_labels = data.get('sample_labels', [])

        if progress_callback:
            progress_callback(50)

        if linkage_matrix and len(linkage_matrix) > 0:
            try:
                linkage_np = np.array(linkage_matrix)

                if progress_callback:
                    progress_callback(70)

                # Usar función de dendrograma mejorado
                plot_dendrogram_manual_mejorado(linkage_np, labels=sample_labels,
                                               ax=ax, max_display=30)

            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)[:100]}',
                       ha='center', va='center', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='mistyrose'))
                ax.set_title('Error en Dendrograma')
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'No hay matriz de linkage disponible',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Dendrograma no disponible')
            ax.axis('off')

        fig.tight_layout()

        if progress_callback:
            progress_callback(100)

        return fig

    def _plot_correlation_matrix(self, fig, data, config, progress_callback):
        """Matriz de correlación"""
        if progress_callback:
            progress_callback(20)

        ax = fig.add_subplot(111)

        correlaciones = data.get('correlaciones', {})

        if 'matriz_pearson' in correlaciones:
            df_corr = pd.DataFrame(correlaciones['matriz_pearson'])

            # Limitar variables
            if len(df_corr.columns) > 8:
                df_corr = df_corr.iloc[:8, :8]

            if progress_callback:
                progress_callback(60)

            im = ax.imshow(df_corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            ax.set_xticks(range(len(df_corr.columns)))
            ax.set_yticks(range(len(df_corr.index)))
            ax.set_xticklabels(df_corr.columns, rotation=45, ha='right')
            ax.set_yticklabels(df_corr.index)
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title('Matriz de Correlaciones')

        fig.tight_layout()

        if progress_callback:
            progress_callback(100)

        return fig

    def _plot_outliers_distribution(self, fig, data, config, progress_callback):
        """Distribución de outliers"""
        if progress_callback:
            progress_callback(20)

        ax = fig.add_subplot(111)

        outliers = data.get('outliers', {})

        if outliers:
            metodos = []
            cantidades = []

            for metodo, outlier_data in outliers.items():
                if metodo != 'consenso' and isinstance(outlier_data, dict) and 'total' in outlier_data:
                    metodos.append(metodo.replace('_', ' ').title())
                    cantidades.append(outlier_data['total'])

            if progress_callback:
                progress_callback(60)

            if metodos:
                bars = ax.bar(metodos, cantidades, alpha=0.7, edgecolor='black')
                ax.set_ylabel('Número de Outliers')
                ax.set_title('Outliers por Método')
                ax.tick_params(axis='x', rotation=45)

                for bar, val in zip(bars, cantidades):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                           f'{val}', ha='center', va='bottom')

        fig.tight_layout()

        if progress_callback:
            progress_callback(100)

        return fig

    def _plot_data_quality(self, fig, data, config, progress_callback):
        """Calidad de datos (gauge)"""
        if progress_callback:
            progress_callback(20)

        ax = fig.add_subplot(111)

        calidad = data.get('calidad_datos', {})

        if calidad:
            quality_score = calidad.get('quality_score', 0)
            calificacion = calidad.get('calificacion', 'N/A')

            if progress_callback:
                progress_callback(60)

            # Gráfico de gauge
            categories = ['Excelente', 'Buena', 'Regular', 'Deficiente']
            colors = ['green', 'yellow', 'orange', 'red']
            values = [25, 25, 25, 25]

            wedges, texts = ax.pie(values, labels=categories, colors=colors,
                                  startangle=90, counterclock=False)

            # Aguja
            angle = 90 - (quality_score / 100) * 360
            ax.annotate('', xy=(0.7 * np.cos(np.radians(angle)),
                               0.7 * np.sin(np.radians(angle))),
                       xytext=(0, 0), arrowprops=dict(arrowstyle='->', lw=3, color='black'))

            ax.set_title(f'Calidad de Datos: {quality_score:.1f}/100\n({calificacion})')

        fig.tight_layout()

        if progress_callback:
            progress_callback(100)

        return fig

    def _update_visualization_traditional(self):
        """Visualización tradicional (fallback)"""
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
            print(f"Error en visualización tradicional: {e}")
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Error generando visualización:\n{str(e)[:100]}',
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='mistyrose'))
            ax.set_title('Error en Visualización')
            ax.axis('off')
            self.canvas.draw_idle()

    def _on_plot_selected(self, index):
        """Cuando se selecciona un gráfico del selector"""
        if index < 0:
            return

        task_id = self.plot_selector.itemData(index)

        if task_id in self.plot_results:
            fig = self.plot_results[task_id]
            self.canvas.figure = fig
            self.canvas.draw()

    def _regenerate_plots_parallel(self):
        """Regenerar todos los gráficos en paralelo"""
        if not self.current_results:
            return

        reply = QMessageBox.question(
            self, 'Regenerar Gráficos',
            '¿Desea regenerar todos los gráficos en paralelo?\n\n'
            'Esto puede tardar unos segundos.',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply == QMessageBox.Yes:
            self._update_visualization_parallel()

    def _save_all_figures(self):
        """Guardar todos los gráficos generados"""
        if not self.plot_results:
            QMessageBox.warning(self, "Sin gráficos", "No hay gráficos para guardar.")
            return

        folder = QFileDialog.getExistingDirectory(
            self, "Seleccionar Carpeta para Guardar Gráficos"
        )

        if folder:
            try:
                saved = 0
                for task_id, fig in self.plot_results.items():
                    task = self.plotter.tasks[task_id]
                    filename = f"{task.description.replace(':', '_').replace(' ', '_')}.png"
                    filepath = os.path.join(folder, filename)

                    fig.savefig(filepath, dpi=300, bbox_inches='tight')
                    saved += 1

                QMessageBox.information(
                    self, "Éxito",
                    f"Se guardaron {saved} gráficos en:\n{folder}"
                )

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error guardando gráficos:\n{e}")

    def _update_summary(self, results: dict, analysis_type: str):
        """Actualizar resumen"""
        summary = f"📊 Resumen - {analysis_type.replace('_', ' ').title()}\n"
        summary += "=" * 50 + "\n\n"

        summary += f"🔍 Tipo de análisis: {results.get('tipo', 'N/A')}\n"
        summary += f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        if 'variables_utilizadas' in results:
            summary += f"📈 Variables analizadas: {len(results['variables_utilizadas'])}\n"
            summary += f"📝 Variables: {', '.join(results['variables_utilizadas'][:5])}"
            if len(results['variables_utilizadas']) > 5:
                summary += f" (y {len(results['variables_utilizadas']) - 5} más)"
            summary += "\n\n"

        if results.get('tipo') == 'kmeans_optimizado':
            k_optimo = results.get('recomendacion_k', 'N/A')
            summary += f"🎯 K óptimo recomendado: {k_optimo}\n"

        elif results.get('tipo') == 'dbscan_optimizado':
            config = results.get('mejor_configuracion', {})
            summary += f"🎯 Clusters encontrados: {config.get('n_clusters', 'N/A')}\n"
            summary += f"🔍 Outliers detectados: {config.get('n_noise', 'N/A')}\n"

        elif results.get('tipo') == 'pca_completo_avanzado':
            if 'linear' in results.get('resultados_por_metodo', {}):
                linear_result = results['resultados_por_metodo']['linear']
                n_comp = linear_result.get('componentes_recomendados', 'N/A')
                summary += f"📊 Componentes recomendados: {n_comp}\n"

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

    def _show_error(self, error_msg):
        """Mostrar error"""
        self.summary_text.setText(f"❌ Error en el análisis:\n\n{error_msg}")
        self.status_label.setText("❌ Error en análisis")
        self.status_label.setStyleSheet("color: red;")

        self.metrics_table.setRowCount(0)
        self.details_text.setText(f"Error: {error_msg}")

        self.export_results_btn.setEnabled(False)
        self.generate_report_btn.setEnabled(False)
        self.regenerate_parallel_btn.setEnabled(False)

    def _save_figure(self):
        """Guardar figura actual"""
        if not hasattr(self, 'canvas') or not hasattr(self.canvas, 'figure'):
            QMessageBox.warning(self, "Sin figura", "No hay figura para guardar.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Figura", "", "Imagen PNG (*.png);;Imagen JPEG (*.jpg)"
        )
        if file_path:
            try:
                self.canvas.figure.savefig(file_path, dpi=300, bbox_inches='tight')
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
                f.write(f"Tipo de análisis: {self.current_results.get('tipo', 'N/A')}\n")
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
    """Ventana principal para ML No Supervisado optimizada"""

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

        # Crear scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(400)

        # Widget de contenido
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
        clustering_label.setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 13px;")
        analysis_layout.addWidget(clustering_label)

        self.kmeans_btn = QPushButton("🔹 K-Means Optimizado")
        self.kmeans_btn.setMinimumHeight(40)
        self.kmeans_btn.setToolTip("Clustering con optimización automática del número de clusters")
        self.kmeans_btn.clicked.connect(lambda: self.run_analysis('kmeans_optimizado'))
        analysis_layout.addWidget(self.kmeans_btn)

        self.hierarchical_btn = QPushButton("🔸 Clustering Jerárquico")
        self.hierarchical_btn.setMinimumHeight(40)
        self.hierarchical_btn.setToolTip("Clustering basado en jerarquías")
        self.hierarchical_btn.clicked.connect(lambda: self.run_analysis('clustering_jerarquico'))
        analysis_layout.addWidget(self.hierarchical_btn)

        self.dbscan_btn = QPushButton("🔺 DBSCAN")
        self.dbscan_btn.setMinimumHeight(40)
        self.dbscan_btn.setToolTip("Clustering basado en densidad con detección de outliers")
        self.dbscan_btn.clicked.connect(lambda: self.run_analysis('dbscan'))
        analysis_layout.addWidget(self.dbscan_btn)

        # PCA
        pca_label = QLabel("📊 Reducción Dimensional")
        pca_label.setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 13px;")
        analysis_layout.addWidget(pca_label)

        self.pca_btn = QPushButton("🔹 PCA Avanzado")
        self.pca_btn.setMinimumHeight(40)
        self.pca_btn.setToolTip("Análisis de Componentes Principales")
        self.pca_btn.clicked.connect(lambda: self.run_analysis('pca_avanzado'))
        analysis_layout.addWidget(self.pca_btn)

        # Análisis exploratorio
        exp_label = QLabel("🔍 Análisis Exploratorio")
        exp_label.setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 13px;")
        analysis_layout.addWidget(exp_label)

        self.exploratory_btn = QPushButton("🔹 Análisis Completo")
        self.exploratory_btn.setMinimumHeight(40)
        self.exploratory_btn.setToolTip("Análisis exploratorio completo")
        self.exploratory_btn.clicked.connect(lambda: self.run_analysis('analisis_exploratorio'))
        analysis_layout.addWidget(self.exploratory_btn)

        # Botón de cancelar
        self.cancel_btn = QPushButton("❌ Cancelar Análisis")
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_analysis)
        self.cancel_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold;")
        analysis_layout.addWidget(self.cancel_btn)

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
                "Los datos incluyen parámetros de calidad del agua realistas."
            )

        except Exception as e:
            self.log(f"❌ Error generando datos demo: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error generando datos demo:\n{str(e)}")

    # ==================== EJECUCIÓN DE ANÁLISIS ====================

    def run_analysis(self, analysis_type: str):
        """Ejecutar análisis específico con monitor de hilos"""
        if not self.validate_selection():
            return

        if not ML_AVAILABLE:
            QMessageBox.critical(
                self, "Error",
                "Las librerías de Machine Learning no están disponibles."
            )
            return

        # Obtener configuración
        variables = self.variable_selection.get_selected_variables()
        config = self.configuration.get_config()

        # Configurar kwargs base
        kwargs = {
            'variables': variables,
            'escalado': config['scaling_method'],
            'verbose': config['verbose'],
            'random_state': config['random_state']
        }

        # Añadir configuración específica
        if analysis_type == 'kmeans_optimizado':
            kwargs['kmeans_k_range'] = config['kmeans_k_range']
        elif analysis_type == 'clustering_jerarquico':
            kwargs['hierarchical_method'] = config['hierarchical_method']
            kwargs['hierarchical_metric'] = config['hierarchical_metric']
            kwargs['hierarchical_max_clusters'] = config['hierarchical_max_clusters']
        elif analysis_type == 'dbscan':
            kwargs['dbscan_optimize'] = config['dbscan_optimize']
        elif analysis_type == 'pca_avanzado':
            kwargs['pca_variance_threshold'] = config['pca_variance_threshold']
        elif analysis_type == 'analisis_exploratorio':
            kwargs['handle_outliers'] = config['handle_outliers']

        # Mostrar progreso
        self.show_progress(True)

        # Mostrar monitor de hilos solo para clustering jerárquico
        if analysis_type == 'clustering_jerarquico' and hasattr(self.thread_monitor, 'reset'):
            try:
                self.thread_monitor.setVisible(True)
                self.thread_monitor.reset()
            except Exception as e:
                print(f"Error en thread_monitor: {e}")

        self.log(f"Iniciando análisis: {analysis_type}")
        self.log(f"Variables seleccionadas: {len(variables)}")

        # Crear worker
        self.current_worker = MLNoSupervisadoWorker(analysis_type, self.current_data, **kwargs)

        # Conectar señales existentes
        self.current_worker.progress.connect(self.progress_bar.setValue)
        self.current_worker.status.connect(self.log)
        self.current_worker.finished.connect(self.on_analysis_finished)
        self.current_worker.error.connect(self.on_analysis_error)
        self.current_worker.log.connect(self.log)

        # Conectar señal del monitor de hilos (solo para clustering jerárquico)
        if analysis_type == 'clustering_jerarquico':
            if hasattr(self.current_worker, 'thread_activity') and hasattr(self.thread_monitor,
                                                                           'update_active_threads'):
                try:
                    self.current_worker.thread_activity.connect(self.thread_monitor.update_active_threads)
                except Exception as e:
                    print(f"Error conectando thread_activity: {e}")

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

    @pyqtSlot(str)
    def on_analysis_error(self, error_msg: str):
        """Cuando ocurre un error"""
        self.show_progress(False)
        self.log(f"❌ Error: {error_msg}")
        QMessageBox.critical(self, "Error en Análisis",
                             f"Error durante el análisis:\n\n{error_msg}")

    def on_variables_changed(self):
        """Cuando cambian las variables seleccionadas"""
        n_selected = len(self.variable_selection.get_selected_variables())
        if n_selected > 0:
            self.log(f"📊 {n_selected} variables seleccionadas")

    # ==================== UTILIDADES ====================

    def show_progress(self, show: bool):
        """Mostrar/ocultar progreso y monitor de hilos"""
        self.progress_bar.setVisible(show)
        self.cancel_btn.setVisible(show)

        if show:
            self.progress_bar.setValue(0)
            # Intentar resetear monitor solo si existe
            if hasattr(self.thread_monitor, 'reset'):
                try:
                    self.thread_monitor.reset()
                except:
                    pass
        else:
            # Ocultar monitor cuando termine
            if hasattr(self.thread_monitor, 'setVisible'):
                try:
                    self.thread_monitor.setVisible(False)
                except:
                    pass

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
        help_dialog.resize(800, 600)

        layout = QVBoxLayout()

        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
            <h2>Machine Learning No Supervisado</h2>

            <h3>Técnicas disponibles:</h3>
            <ul>
            <li><b>K-Means:</b> Agrupa datos en K clusters esféricos</li>
            <li><b>DBSCAN:</b> Detecta clusters de densidad variable y outliers</li>
            <li><b>Clustering Jerárquico:</b> Crea jerarquías de grupos</li>
            <li><b>PCA:</b> Reduce dimensionalidad manteniendo varianza</li>
            <li><b>Análisis Exploratorio:</b> Examina correlaciones y distribuciones</li>
            </ul>

            <h3>Flujo recomendado:</h3>
            <ol>
            <li>Cargar datos desde el módulo "Cargar Datos" o usar Demo</li>
            <li>Seleccionar variables (usar botón "Auto" para selección inteligente)</li>
            <li>Configurar parámetros según necesidades</li>
            <li>Ejecutar "Análisis Exploratorio" primero</li>
            <li>Aplicar técnicas de clustering o PCA</li>
            <li>Revisar resultados en las pestañas de visualización</li>
            </ol>

            <h3>Consejos:</h3>
            <ul>
            <li>Standard scaling es recomendado para la mayoría de casos</li>
            <li>Evita variables con más del 50% de valores faltantes</li>
            <li>K-Means funciona bien con clusters esféricos</li>
            <li>DBSCAN es ideal cuando no conoces el número de clusters</li>
            <li>PCA es útil cuando tienes muchas variables correlacionadas</li>
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