"""
wqi_calculator.py - Módulo COMPLETO e INDEPENDIENTE para cálculo del Índice de Calidad del Agua
Sistema avanzado para calcular WQI usando múltiples métodos y ecuaciones estándar
Incluye: NSF WQI, CCME WQI, WQI personalizado, análisis temporal y comparaciones
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox,
    QTextEdit, QSplitter, QMessageBox, QFileDialog, QCheckBox, QFrame,
    QTabWidget, QScrollArea, QGridLayout, QProgressBar, QSlider,
    QFormLayout, QDateEdit, QLineEdit, QApplication
)
from PyQt5.QtCore import Qt, pyqtSignal, QDate, QThread
from PyQt5.QtGui import QFont, QColor, QPixmap, QPainter

# Importar sistema de temas
try:
    from darkmode import ThemedWidget
except ImportError:
    class ThemedWidget:
        def __init__(self):
            pass

# ==================== ECUACIONES WQI ESTÁNDAR ====================

class WQICalculationEngine:
    """Motor de cálculo para diferentes métodos de WQI"""

    # Parámetros estándar NSF WQI
    NSF_PARAMETERS = {
        'pH': {'weight': 0.11, 'range': (0, 14), 'optimal': (6.5, 8.5)},
        'Oxigeno_Disuelto': {'weight': 0.17, 'range': (0, 20), 'optimal': (7, 14)},
        'DBO5': {'weight': 0.11, 'range': (0, 30), 'optimal': (0, 3)},
        'Coliformes_Fecales': {'weight': 0.17, 'range': (0, 100000), 'optimal': (0, 20)},
        'Temperatura': {'weight': 0.10, 'range': (0, 40), 'optimal': (15, 25)},
        'Fosforo_Total': {'weight': 0.10, 'range': (0, 10), 'optimal': (0, 0.1)},
        'Nitrato': {'weight': 0.10, 'range': (0, 100), 'optimal': (0, 10)},
        'Turbiedad': {'weight': 0.08, 'range': (0, 100), 'optimal': (0, 5)},
        'Solidos_Totales': {'weight': 0.08, 'range': (0, 500), 'optimal': (0, 100)}
    }

    # Clasificación de calidad estándar
    QUALITY_CLASSIFICATION = {
        (90, 100): {'label': 'Excelente', 'color': '#2E7D32', 'description': 'Agua de calidad excelente'},
        (70, 89): {'label': 'Buena', 'color': '#388E3C', 'description': 'Agua de buena calidad'},
        (50, 69): {'label': 'Regular', 'color': '#FFA000', 'description': 'Agua de calidad regular'},
        (25, 49): {'label': 'Deficiente', 'color': '#F57C00', 'description': 'Agua de calidad deficiente'},
        (0, 24): {'label': 'Muy Deficiente', 'color': '#D32F2F', 'description': 'Agua de muy mala calidad'}
    }

    @staticmethod
    def calculate_nsf_wqi(data_row, parameters=None, weights=None):
        """
        Calcular WQI usando el método NSF (National Sanitation Foundation)
        Fórmula: WQI = Π(Qi^Wi) donde Qi es el índice del parámetro i y Wi es su peso
        """
        if parameters is None:
            parameters = WQICalculationEngine.NSF_PARAMETERS

        if weights is None:
            weights = {param: config['weight'] for param, config in parameters.items()}

        # Normalizar pesos para que sumen 1
        total_weight = sum(weights.values())
        normalized_weights = {param: weight/total_weight for param, weight in weights.items()}

        wqi_product = 1.0
        used_parameters = []

        for param, weight in normalized_weights.items():
            if param in data_row and pd.notna(data_row[param]):
                qi = WQICalculationEngine._calculate_parameter_index(param, data_row[param], parameters)
                wqi_product *= (qi ** weight)
                used_parameters.append({
                    'parameter': param,
                    'value': data_row[param],
                    'qi': qi,
                    'weight': weight
                })

        return {
            'wqi': wqi_product,
            'used_parameters': used_parameters,
            'method': 'NSF'
        }

    @staticmethod
    def calculate_ccme_wqi(data_row, parameters=None, objectives=None):
        """
        Calcular WQI usando el método CCME (Canadian Council of Ministers of Environment)
        Fórmula: WQI = 100 - (√(F1² + F2² + F3²) / 1.732)
        """
        if parameters is None:
            parameters = list(WQICalculationEngine.NSF_PARAMETERS.keys())

        if objectives is None:
            objectives = {param: config['optimal'][1] for param, config in
                         WQICalculationEngine.NSF_PARAMETERS.items()}

        # F1: Alcance - Porcentaje de parámetros que no cumplen objetivos
        failed_tests = 0
        total_tests = 0

        # F2: Frecuencia - Porcentaje de pruebas individuales que no cumplen
        # F3: Amplitud - Cantidad por la cual los valores fallan
        excursions = []

        for param in parameters:
            if param in data_row and pd.notna(data_row[param]) and param in objectives:
                total_tests += 1
                value = data_row[param]
                objective = objectives[param]

                if value > objective:
                    failed_tests += 1
                    excursion = (value - objective) / objective
                    excursions.append(excursion)

        if total_tests == 0:
            return {'wqi': 0, 'method': 'CCME', 'error': 'No parameters available'}

        # Calcular factores CCME
        F1 = (failed_tests / total_tests) * 100
        F2 = F1  # Simplificado para una sola muestra

        if excursions:
            nse = sum(excursions) / len(excursions)  # Normalized Sum of Excursions
            F3 = (nse / (nse + 1)) * 100
        else:
            F3 = 0

        # Calcular WQI CCME
        wqi_ccme = 100 - (np.sqrt(F1**2 + F2**2 + F3**2) / 1.732)
        wqi_ccme = max(0, wqi_ccme)

        return {
            'wqi': wqi_ccme,
            'F1': F1,
            'F2': F2,
            'F3': F3,
            'failed_tests': failed_tests,
            'total_tests': total_tests,
            'method': 'CCME'
        }

    @staticmethod
    def calculate_weighted_arithmetic_wqi(data_row, parameters=None, weights=None):
        """
        Calcular WQI usando promedio aritmético ponderado
        Fórmula: WQI = Σ(Wi × Qi) donde Wi es el peso y Qi el índice normalizado
        """
        if parameters is None:
            parameters = WQICalculationEngine.NSF_PARAMETERS

        if weights is None:
            weights = {param: config['weight'] for param, config in parameters.items()}

        # Normalizar pesos
        total_weight = sum(weights.values())
        normalized_weights = {param: weight/total_weight for param, weight in weights.items()}

        weighted_sum = 0
        total_weight_used = 0
        used_parameters = []

        for param, weight in normalized_weights.items():
            if param in data_row and pd.notna(data_row[param]):
                qi = WQICalculationEngine._calculate_parameter_index(param, data_row[param], parameters)
                weighted_sum += weight * qi
                total_weight_used += weight
                used_parameters.append({
                    'parameter': param,
                    'value': data_row[param],
                    'qi': qi,
                    'weight': weight
                })

        if total_weight_used == 0:
            return {'wqi': 0, 'method': 'Weighted Arithmetic', 'error': 'No parameters available'}

        wqi = weighted_sum / total_weight_used * 100

        return {
            'wqi': wqi,
            'used_parameters': used_parameters,
            'method': 'Weighted Arithmetic'
        }

    @staticmethod
    def _calculate_parameter_index(parameter, value, parameters_config):
        """
        Calcular índice normalizado (0-1) para un parámetro específico
        """
        if parameter not in parameters_config:
            return 0.5  # Valor neutro si no está configurado

        config = parameters_config[parameter]
        param_range = config['range']
        optimal = config['optimal']

        # Normalización específica por tipo de parámetro
        if parameter == 'pH':
            return WQICalculationEngine._ph_index(value)
        elif parameter == 'Oxigeno_Disuelto':
            return WQICalculationEngine._oxygen_index(value)
        elif parameter in ['DBO5', 'Coliformes_Fecales', 'Turbiedad']:
            return WQICalculationEngine._pollutant_index(value, optimal[1])
        elif parameter == 'Temperatura':
            return WQICalculationEngine._temperature_index(value, optimal)
        else:
            return WQICalculationEngine._generic_index(value, optimal, param_range)

    @staticmethod
    def _ph_index(ph_value):
        """Índice específico para pH"""
        if ph_value <= 2:
            return 0.02
        elif ph_value <= 3:
            return 0.13 + 0.11 * ph_value + 0.024 * ph_value**2
        elif ph_value <= 4:
            return -25 + 17.2 * ph_value - 2.42 * ph_value**2
        elif ph_value <= 6.2:
            return -657.2 + 197.38 * ph_value - 12.9167 * ph_value**2
        elif ph_value <= 7:
            return -427.8 + 142.05 * ph_value - 9.695 * ph_value**2
        elif ph_value <= 8:
            return 216 - 16 * ph_value
        elif ph_value <= 8.5:
            return 1415823 * np.exp(-1.1507 * ph_value)
        elif ph_value <= 9:
            return 288 - 27 * ph_value
        elif ph_value <= 10:
            return 633 - 106.5 * ph_value + 4.5 * ph_value**2
        elif ph_value <= 12:
            return 633 - 106.5 * ph_value + 4.5 * ph_value**2
        else:
            return 0.03

        # Normalizar a 0-1
        return max(0, min(1, ph_value / 100))

    @staticmethod
    def _oxygen_index(do_value, temperature=20):
        """Índice específico para oxígeno disuelto"""
        # Saturación teórica a temperatura dada
        do_sat = 14.652 - 0.41022 * temperature + 0.007991 * temperature**2 - 0.000077774 * temperature**3

        percent_sat = (do_value / do_sat) * 100

        if percent_sat >= 140:
            return 0.50
        elif percent_sat >= 100:
            return 0.029 * percent_sat - 0.00025 * percent_sat**2 + 0.0000005609 * percent_sat**3 + 0.03
        elif percent_sat >= 85:
            return 0.037745 * percent_sat**0.704889 + 0.03
        elif percent_sat >= 50:
            return -0.01166 * percent_sat + 0.0058 * percent_sat**2 - 0.00003803435 * percent_sat**3 + 0.03
        else:
            return 0.0034 * percent_sat + 0.00008095 * percent_sat**2 + 0.0000135252 * percent_sat**3 + 0.03

        return max(0, min(1, percent_sat / 100))

    @staticmethod
    def _pollutant_index(value, max_acceptable):
        """Índice para contaminantes (menor es mejor)"""
        if value <= 0:
            return 1.0
        elif value >= max_acceptable * 10:
            return 0.0
        else:
            return max(0, 1 - (value / max_acceptable))

    @staticmethod
    def _temperature_index(temp_value, optimal_range):
        """Índice para temperatura"""
        optimal_min, optimal_max = optimal_range
        optimal_center = (optimal_min + optimal_max) / 2

        if optimal_min <= temp_value <= optimal_max:
            return 1.0
        else:
            deviation = min(abs(temp_value - optimal_min), abs(temp_value - optimal_max))
            return max(0, 1 - (deviation / optimal_center))

    @staticmethod
    def _generic_index(value, optimal_range, param_range):
        """Índice genérico para otros parámetros"""
        optimal_min, optimal_max = optimal_range
        range_min, range_max = param_range

        if optimal_min <= value <= optimal_max:
            return 1.0
        elif value < optimal_min:
            if value <= range_min:
                return 0.0
            return (value - range_min) / (optimal_min - range_min)
        else:  # value > optimal_max
            if value >= range_max:
                return 0.0
            return 1 - ((value - optimal_max) / (range_max - optimal_max))

    @staticmethod
    def classify_water_quality(wqi_value):
        """Clasificar calidad del agua según valor WQI"""
        for (min_val, max_val), classification in WQICalculationEngine.QUALITY_CLASSIFICATION.items():
            if min_val <= wqi_value <= max_val:
                return classification

        return {'label': 'No Clasificado', 'color': '#666666', 'description': 'Calidad no determinada'}

# ==================== MOTOR DE ANÁLISIS TEMPORAL ====================

class TemporalAnalysisEngine:
    """Motor para análisis temporal de WQI"""

    @staticmethod
    def analyze_temporal_trends(df, date_column='Fecha', wqi_column='WQI'):
        """Analizar tendencias temporales del WQI"""
        if date_column not in df.columns or wqi_column not in df.columns:
            return {'error': 'Columnas requeridas no encontradas'}

        # Asegurar que la fecha es datetime
        df_temp = df.copy()
        df_temp[date_column] = pd.to_datetime(df_temp[date_column])
        df_temp = df_temp.sort_values(date_column)

        # Análisis por períodos
        analysis = {}

        # Tendencia general
        if len(df_temp) >= 3:
            from scipy import stats
            x = np.arange(len(df_temp))
            y = df_temp[wqi_column].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            analysis['tendencia_general'] = {
                'pendiente': float(slope),
                'correlacion': float(r_value),
                'p_valor': float(p_value),
                'significativa': p_value < 0.05,
                'direccion': 'Mejorando' if slope > 0 else 'Empeorando' if slope < 0 else 'Estable'
            }

        # Análisis estacional
        df_temp['mes'] = df_temp[date_column].dt.month
        analisis_mensual = df_temp.groupby('mes')[wqi_column].agg(['mean', 'std', 'count']).round(2)

        analysis['variacion_estacional'] = {
            'por_mes': analisis_mensual.to_dict(),
            'mejor_mes': int(analisis_mensual['mean'].idxmax()),
            'peor_mes': int(analisis_mensual['mean'].idxmin()),
            'coeficiente_variacion': float(analisis_mensual['mean'].std() / analisis_mensual['mean'].mean())
        }

        # Eventos extremos
        q1 = df_temp[wqi_column].quantile(0.25)
        q3 = df_temp[wqi_column].quantile(0.75)
        iqr = q3 - q1

        outliers_bajos = df_temp[df_temp[wqi_column] < (q1 - 1.5 * iqr)]
        outliers_altos = df_temp[df_temp[wqi_column] > (q3 + 1.5 * iqr)]

        analysis['eventos_extremos'] = {
            'calidad_muy_baja': {
                'count': len(outliers_bajos),
                'fechas': outliers_bajos[date_column].dt.strftime('%Y-%m-%d').tolist(),
                'valores': outliers_bajos[wqi_column].tolist()
            },
            'calidad_muy_alta': {
                'count': len(outliers_altos),
                'fechas': outliers_altos[date_column].dt.strftime('%Y-%m-%d').tolist(),
                'valores': outliers_altos[wqi_column].tolist()
            }
        }

        return analysis

    @staticmethod
    def generate_forecast(df, date_column='Fecha', wqi_column='WQI', periods=30):
        """Generar pronóstico simple de WQI"""
        try:
            df_temp = df.copy()
            df_temp[date_column] = pd.to_datetime(df_temp[date_column])
            df_temp = df_temp.sort_values(date_column).reset_index(drop=True)

            # Modelo de suavizado exponencial simple
            from scipy.optimize import minimize_scalar

            def mse_alpha(alpha):
                forecast = [df_temp[wqi_column].iloc[0]]
                for i in range(1, len(df_temp)):
                    forecast.append(alpha * df_temp[wqi_column].iloc[i-1] + (1-alpha) * forecast[-1])

                mse = np.mean([(actual - pred)**2 for actual, pred in
                              zip(df_temp[wqi_column].iloc[1:], forecast[1:])])
                return mse

            # Optimizar alpha
            result = minimize_scalar(mse_alpha, bounds=(0.1, 0.9), method='bounded')
            optimal_alpha = result.x

            # Generar pronóstico
            last_value = df_temp[wqi_column].iloc[-1]
            trend = df_temp[wqi_column].diff().mean()

            forecast_dates = pd.date_range(
                start=df_temp[date_column].iloc[-1] + timedelta(days=1),
                periods=periods
            )

            forecast_values = []
            current_value = last_value

            for i in range(periods):
                current_value = optimal_alpha * current_value + (1-optimal_alpha) * (current_value + trend)
                forecast_values.append(current_value)

            return {
                'fechas': forecast_dates.strftime('%Y-%m-%d').tolist(),
                'valores_pronosticados': forecast_values,
                'parametros': {
                    'alpha': optimal_alpha,
                    'tendencia': float(trend),
                    'ultimo_valor': float(last_value)
                }
            }

        except Exception as e:
            return {'error': f'Error en pronóstico: {str(e)}'}

# ==================== WORKER THREAD PARA CÁLCULOS ====================

class WQICalculationWorker(QThread):
    """Worker thread para cálculos intensivos de WQI"""

    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    calculation_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, data, method, parameters, weights=None):
        super().__init__()
        self.data = data
        self.method = method
        self.parameters = parameters
        self.weights = weights

    def run(self):
        try:
            self.status_updated.emit("Iniciando cálculos de WQI...")
            self.progress_updated.emit(10)

            results = []
            total_rows = len(self.data)

            for index, row in self.data.iterrows():
                self.status_updated.emit(f"Procesando muestra {index + 1} de {total_rows}")

                if self.method == 'NSF':
                    result = WQICalculationEngine.calculate_nsf_wqi(row, self.parameters, self.weights)
                elif self.method == 'CCME':
                    result = WQICalculationEngine.calculate_ccme_wqi(row, self.parameters)
                elif self.method == 'Weighted_Arithmetic':
                    result = WQICalculationEngine.calculate_weighted_arithmetic_wqi(row, self.parameters, self.weights)
                else:
                    result = {'wqi': 0, 'error': f'Método {self.method} no reconocido'}

                # Agregar información adicional
                result['index'] = index
                result['classification'] = WQICalculationEngine.classify_water_quality(result.get('wqi', 0))
                results.append(result)

                # Actualizar progreso
                progress = 10 + int((index + 1) / total_rows * 80)
                self.progress_updated.emit(progress)

            # Análisis estadístico
            self.status_updated.emit("Generando análisis estadístico...")
            self.progress_updated.emit(90)

            wqi_values = [r['wqi'] for r in results if 'wqi' in r]

            if wqi_values:
                statistics = {
                    'count': len(wqi_values),
                    'mean': float(np.mean(wqi_values)),
                    'median': float(np.median(wqi_values)),
                    'std': float(np.std(wqi_values)),
                    'min': float(np.min(wqi_values)),
                    'max': float(np.max(wqi_values)),
                    'q25': float(np.percentile(wqi_values, 25)),
                    'q75': float(np.percentile(wqi_values, 75))
                }

                # Distribución por calidad
                quality_distribution = {}
                for result in results:
                    if 'classification' in result:
                        label = result['classification']['label']
                        quality_distribution[label] = quality_distribution.get(label, 0) + 1
            else:
                statistics = {}
                quality_distribution = {}

            self.progress_updated.emit(100)
            self.status_updated.emit("Cálculos completados")

            final_result = {
                'results': results,
                'statistics': statistics,
                'quality_distribution': quality_distribution,
                'method': self.method,
                'total_samples': total_rows
            }

            self.calculation_finished.emit(final_result)

        except Exception as e:
            self.error_occurred.emit(str(e))

# ==================== VENTANA PRINCIPAL WQI ====================

class WQICalculatorAdvanced(QWidget, ThemedWidget):
    """Calculadora avanzada de WQI - Módulo independiente"""

    def __init__(self):
        QWidget.__init__(self)
        ThemedWidget.__init__(self)

        self.data = None
        self.current_results = None
        self.calculation_worker = None

        self.setup_ui()
        self.apply_styles()

        # Cargar configuración por defecto
        self.load_default_configuration()

    def setup_ui(self):
        """Configurar interfaz de usuario principal"""
        self.setWindowTitle("💧 Calculadora Avanzada del Índice de Calidad del Agua (WQI)")
        self.setMinimumSize(1400, 900)

        main_layout = QVBoxLayout()

        # Header principal
        header = self.create_main_header()
        main_layout.addWidget(header)

        # Tabs principales
        self.main_tabs = QTabWidget()

        # Tab 1: Configuración y Cálculo
        self.setup_calculation_tab()

        # Tab 2: Resultados y Análisis
        self.setup_results_tab()

        # Tab 3: Análisis Temporal
        self.setup_temporal_tab()

        # Tab 4: Comparaciones
        self.setup_comparison_tab()

        main_layout.addWidget(self.main_tabs)

        # Footer con controles principales
        footer = self.create_main_footer()
        main_layout.addWidget(footer)

        self.setLayout(main_layout)

    def create_main_header(self):
        """Crear header principal"""
        header_frame = QFrame()
        header_frame.setObjectName("mainHeaderFrame")

        header_layout = QVBoxLayout()

        # Título principal
        title = QLabel("💧 Calculadora Avanzada del Índice de Calidad del Agua")
        title.setObjectName("mainTitle")
        title.setAlignment(Qt.AlignCenter)

        # Subtítulo con fórmulas
        subtitle = QLabel("WQI = Σ(wi × Ci) | Métodos: NSF, CCME, Aritmético Ponderado")
        subtitle.setObjectName("formulaSubtitle")
        subtitle.setAlignment(Qt.AlignCenter)

        # Información del dataset
        self.dataset_info = QLabel("Dataset: No cargado")
        self.dataset_info.setObjectName("datasetInfo")
        self.dataset_info.setAlignment(Qt.AlignCenter)

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        header_layout.addWidget(self.dataset_info)

        header_frame.setLayout(header_layout)
        return header_frame

    def setup_calculation_tab(self):
        """Configurar tab de cálculo principal"""
        calc_widget = QWidget()
        calc_layout = QHBoxLayout()

        # Panel izquierdo - Configuración
        left_panel = self.create_configuration_panel()
        calc_layout.addWidget(left_panel, 1)

        # Panel derecho - Vista previa y controles
        right_panel = self.create_preview_panel()
        calc_layout.addWidget(right_panel, 2)

        calc_widget.setLayout(calc_layout)
        self.main_tabs.addTab(calc_widget, "⚙️ Configuración y Cálculo")

    def setup_results_tab(self):
        """Configurar tab de resultados"""
        results_widget = QWidget()
        results_layout = QVBoxLayout()

        # Estadísticas principales
        stats_frame = self.create_statistics_frame()
        results_layout.addWidget(stats_frame)

        # Splitter para tablas y gráficos
        results_splitter = QSplitter(Qt.Horizontal)

        # Tabla de resultados
        self.results_table = QTableWidget()
        self.results_table.setObjectName("resultsTable")
        results_splitter.addWidget(self.results_table)

        # Panel de gráficos
        graphs_panel = self.create_graphs_panel()
        results_splitter.addWidget(graphs_panel)

        results_splitter.setSizes([600, 600])
        results_layout.addWidget(results_splitter)

        results_widget.setLayout(results_layout)
        self.main_tabs.addTab(results_widget, "📊 Resultados y Análisis")

    def setup_temporal_tab(self):
        """Configurar tab de análisis temporal"""
        temporal_widget = QWidget()
        temporal_layout = QVBoxLayout()

        # Controles temporales
        temporal_controls = self.create_temporal_controls()
        temporal_layout.addWidget(temporal_controls)

        # Área de resultados temporales
        self.temporal_results = QTextEdit()
        self.temporal_results.setObjectName("temporalResults")
        self.temporal_results.setPlaceholderText("Los análisis temporales aparecerán aquí...")
        temporal_layout.addWidget(self.temporal_results)

        temporal_widget.setLayout(temporal_layout)
        self.main_tabs.addTab(temporal_widget, "📈 Análisis Temporal")

    def setup_comparison_tab(self):
        """Configurar tab de comparaciones"""
        comparison_widget = QWidget()
        comparison_layout = QVBoxLayout()

        # Controles de comparación
        comparison_controls = self.create_comparison_controls()
        comparison_layout.addWidget(comparison_controls)

        # Resultados de comparación
        self.comparison_results = QTextEdit()
        self.comparison_results.setObjectName("comparisonResults")
        self.comparison_results.setPlaceholderText("Las comparaciones entre métodos aparecerán aquí...")
        comparison_layout.addWidget(self.comparison_results)

        comparison_widget.setLayout(comparison_layout)
        self.main_tabs.addTab(comparison_widget, "⚖️ Comparaciones")

    def create_configuration_panel(self):
        """Crear panel de configuración"""
        config_frame = QGroupBox("⚙️ Configuración de Parámetros WQI")
        config_layout = QVBoxLayout()

        # Método de cálculo
        method_group = QGroupBox("Método de Cálculo")
        method_layout = QVBoxLayout()

        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "NSF - National Sanitation Foundation",
            "CCME - Canadian Council of Ministers",
            "Aritmético Ponderado"
        ])
        self.method_combo.currentTextChanged.connect(self.on_method_changed)
        method_layout.addWidget(self.method_combo)

        method_group.setLayout(method_layout)
        config_layout.addWidget(method_group)

        # Tabla de parámetros
        params_group = QGroupBox("Parámetros y Pesos")
        params_layout = QVBoxLayout()

        self.params_table = QTableWidget()
        self.params_table.setColumnCount(5)
        self.params_table.setHorizontalHeaderLabels([
            "Incluir", "Parámetro", "Peso", "Rango Aceptable", "Valor Óptimo"
        ])
        self.setup_parameters_table()
        params_layout.addWidget(self.params_table)

        # Botones de configuración
        buttons_layout = QHBoxLayout()

        normalize_btn = QPushButton("⚖️ Normalizar Pesos")
        normalize_btn.clicked.connect(self.normalize_weights)
        buttons_layout.addWidget(normalize_btn)

        reset_btn = QPushButton("🔄 Restaurar Defecto")
        reset_btn.clicked.connect(self.load_default_configuration)
        buttons_layout.addWidget(reset_btn)

        save_config_btn = QPushButton("💾 Guardar Config")
        save_config_btn.clicked.connect(self.save_configuration)
        buttons_layout.addWidget(save_config_btn)

        params_layout.addLayout(buttons_layout)
        params_group.setLayout(params_layout)
        config_layout.addWidget(params_group)

        config_frame.setLayout(config_layout)
        return config_frame

    def create_preview_panel(self):
        """Crear panel de vista previa"""
        preview_frame = QGroupBox("📊 Vista Previa y Control")
        preview_layout = QVBoxLayout()

        # Carga de datos
        data_group = QGroupBox("Cargar Datos")
        data_layout = QVBoxLayout()

        data_buttons = QHBoxLayout()

        load_csv_btn = QPushButton("📄 Cargar CSV")
        load_csv_btn.clicked.connect(self.load_csv_data)
        data_buttons.addWidget(load_csv_btn)

        load_excel_btn = QPushButton("📊 Cargar Excel")
        load_excel_btn.clicked.connect(self.load_excel_data)
        data_buttons.addWidget(load_excel_btn)

        generate_demo_btn = QPushButton("🎯 Datos Demo")
        generate_demo_btn.clicked.connect(self.generate_demo_data)
        data_buttons.addWidget(generate_demo_btn)

        data_layout.addLayout(data_buttons)
        data_group.setLayout(data_layout)
        preview_layout.addWidget(data_group)

        # Vista previa de datos
        preview_group = QGroupBox("Vista Previa")
        preview_group_layout = QVBoxLayout()

        self.preview_table = QTableWidget()
        self.preview_table.setMaximumHeight(200)
        preview_group_layout.addWidget(self.preview_table)

        preview_group.setLayout(preview_group_layout)
        preview_layout.addWidget(preview_group)

        # Progreso y estado
        progress_group = QGroupBox("Estado del Cálculo")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("✅ Sistema listo")
        self.status_label.setObjectName("statusLabel")
        progress_layout.addWidget(self.status_label)

        progress_group.setLayout(progress_layout)
        preview_layout.addWidget(progress_group)

        preview_frame.setLayout(preview_layout)
        return preview_frame

    def create_statistics_frame(self):
        """Crear frame de estadísticas"""
        stats_frame = QFrame()
        stats_frame.setObjectName("statisticsFrame")

        stats_layout = QHBoxLayout()

        # Cards de estadísticas
        self.stats_cards = {}

        stats_info = [
            ("Muestras", "📊", "count"),
            ("WQI Promedio", "📈", "mean"),
            ("Desv. Estándar", "📏", "std"),
            ("Calidad Modal", "🏆", "mode")
        ]

        for title, icon, key in stats_info:
            card = self.create_stat_card(title, icon, "---")
            self.stats_cards[key] = card
            stats_layout.addWidget(card)

        stats_frame.setLayout(stats_layout)
        return stats_frame

    def create_stat_card(self, title, icon, value):
        """Crear card de estadística"""
        card = QFrame()
        card.setObjectName("statCard")

        card_layout = QVBoxLayout()

        icon_label = QLabel(icon)
        icon_label.setObjectName("statIcon")
        icon_label.setAlignment(Qt.AlignCenter)

        title_label = QLabel(title)
        title_label.setObjectName("statTitle")
        title_label.setAlignment(Qt.AlignCenter)

        value_label = QLabel(str(value))
        value_label.setObjectName("statValue")
        value_label.setAlignment(Qt.AlignCenter)

        card_layout.addWidget(icon_label)
        card_layout.addWidget(title_label)
        card_layout.addWidget(value_label)

        card.setLayout(card_layout)
        return card

    def create_graphs_panel(self):
        """Crear panel de gráficos"""
        graphs_frame = QGroupBox("📈 Visualizaciones")
        graphs_layout = QVBoxLayout()

        # Aquí se añadirían los gráficos con matplotlib
        graphs_placeholder = QLabel("📊 Los gráficos de WQI aparecerán aquí")
        graphs_placeholder.setAlignment(Qt.AlignCenter)
        graphs_placeholder.setObjectName("graphsPlaceholder")

        graphs_layout.addWidget(graphs_placeholder)
        graphs_frame.setLayout(graphs_layout)

        return graphs_frame

    def create_temporal_controls(self):
        """Crear controles temporales"""
        controls_frame = QGroupBox("⏱️ Controles de Análisis Temporal")
        controls_layout = QGridLayout()

        # Columna de fecha
        controls_layout.addWidget(QLabel("Columna de Fecha:"), 0, 0)
        self.date_column_combo = QComboBox()
        controls_layout.addWidget(self.date_column_combo, 0, 1)

        # Período de análisis
        controls_layout.addWidget(QLabel("Desde:"), 1, 0)
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate().addYears(-1))
        controls_layout.addWidget(self.start_date, 1, 1)

        controls_layout.addWidget(QLabel("Hasta:"), 1, 2)
        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        controls_layout.addWidget(self.end_date, 1, 3)

        # Botones de análisis
        analyze_trends_btn = QPushButton("📈 Analizar Tendencias")
        analyze_trends_btn.clicked.connect(self.analyze_temporal_trends)
        controls_layout.addWidget(analyze_trends_btn, 2, 0, 1, 2)

        forecast_btn = QPushButton("🔮 Generar Pronóstico")
        forecast_btn.clicked.connect(self.generate_forecast)
        controls_layout.addWidget(forecast_btn, 2, 2, 1, 2)

        controls_frame.setLayout(controls_layout)
        return controls_frame

    def create_comparison_controls(self):
        """Crear controles de comparación"""
        controls_frame = QGroupBox("⚖️ Controles de Comparación")
        controls_layout = QVBoxLayout()

        # Métodos a comparar
        methods_layout = QHBoxLayout()

        self.compare_nsf = QCheckBox("NSF WQI")
        self.compare_nsf.setChecked(True)
        methods_layout.addWidget(self.compare_nsf)

        self.compare_ccme = QCheckBox("CCME WQI")
        self.compare_ccme.setChecked(True)
        methods_layout.addWidget(self.compare_ccme)

        self.compare_arithmetic = QCheckBox("Aritmético")
        self.compare_arithmetic.setChecked(True)
        methods_layout.addWidget(self.compare_arithmetic)

        controls_layout.addLayout(methods_layout)

        # Botón de comparación
        compare_btn = QPushButton("🔍 Ejecutar Comparación")
        compare_btn.clicked.connect(self.run_comparison)
        controls_layout.addWidget(compare_btn)

        controls_frame.setLayout(controls_layout)
        return controls_frame

    def create_main_footer(self):
        """Crear footer principal"""
        footer_frame = QFrame()
        footer_frame.setObjectName("mainFooter")

        footer_layout = QHBoxLayout()

        # Información
        info_label = QLabel("💡 Seleccione parámetros y método de cálculo")
        info_label.setObjectName("footerInfo")
        footer_layout.addWidget(info_label)

        footer_layout.addStretch()

        # Botones principales
        self.calculate_btn = QPushButton("🧮 Calcular WQI")
        self.calculate_btn.setObjectName("calculateButton")
        self.calculate_btn.clicked.connect(self.calculate_wqi)
        self.calculate_btn.setEnabled(False)
        footer_layout.addWidget(self.calculate_btn)

        export_btn = QPushButton("📤 Exportar Resultados")
        export_btn.setObjectName("exportButton")
        export_btn.clicked.connect(self.export_results)
        footer_layout.addWidget(export_btn)

        help_btn = QPushButton("❓ Ayuda")
        help_btn.clicked.connect(self.show_help)
        footer_layout.addWidget(help_btn)

        footer_frame.setLayout(footer_layout)
        return footer_frame

    # ==================== MÉTODOS DE CONFIGURACIÓN ====================

    def setup_parameters_table(self):
        """Configurar tabla de parámetros"""
        parameters = WQICalculationEngine.NSF_PARAMETERS

        self.params_table.setRowCount(len(parameters))

        for i, (param, config) in enumerate(parameters.items()):
            # Checkbox incluir
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            self.params_table.setCellWidget(i, 0, checkbox)

            # Nombre del parámetro
            self.params_table.setItem(i, 1, QTableWidgetItem(param))

            # Peso
            weight_spin = QDoubleSpinBox()
            weight_spin.setRange(0.01, 1.0)
            weight_spin.setSingleStep(0.01)
            weight_spin.setDecimals(3)
            weight_spin.setValue(config['weight'])
            self.params_table.setCellWidget(i, 2, weight_spin)

            # Rango aceptable
            range_text = f"{config['range'][0]} - {config['range'][1]}"
            self.params_table.setItem(i, 3, QTableWidgetItem(range_text))

            # Valor óptimo
            optimal_text = f"{config['optimal'][0]} - {config['optimal'][1]}"
            self.params_table.setItem(i, 4, QTableWidgetItem(optimal_text))

        self.params_table.resizeColumnsToContents()

    def load_default_configuration(self):
        """Cargar configuración por defecto"""
        self.setup_parameters_table()
        self.method_combo.setCurrentIndex(0)
        self.status_label.setText("✅ Configuración por defecto cargada")

    def normalize_weights(self):
        """Normalizar pesos para que sumen 1"""
        total_weight = 0
        active_params = []

        for i in range(self.params_table.rowCount()):
            checkbox = self.params_table.cellWidget(i, 0)
            if checkbox.isChecked():
                weight_spin = self.params_table.cellWidget(i, 2)
                total_weight += weight_spin.value()
                active_params.append((i, weight_spin))

        if total_weight > 0:
            for i, weight_spin in active_params:
                normalized = weight_spin.value() / total_weight
                weight_spin.setValue(normalized)

            self.status_label.setText("✅ Pesos normalizados correctamente")
        else:
            QMessageBox.warning(self, "Error", "No hay parámetros activos para normalizar")

    def save_configuration(self):
        """Guardar configuración actual"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Configuración WQI", "config_wqi.json", "JSON (*.json)"
        )

        if file_path:
            config = self.get_current_configuration()

            try:
                import json
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)

                QMessageBox.information(self, "Éxito", "Configuración guardada correctamente")
                self.status_label.setText("✅ Configuración guardada")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al guardar: {str(e)}")

    def get_current_configuration(self):
        """Obtener configuración actual"""
        config = {
            'method': self.method_combo.currentText(),
            'parameters': {}
        }

        for i in range(self.params_table.rowCount()):
            checkbox = self.params_table.cellWidget(i, 0)
            param_name = self.params_table.item(i, 1).text()
            weight_spin = self.params_table.cellWidget(i, 2)

            config['parameters'][param_name] = {
                'included': checkbox.isChecked(),
                'weight': weight_spin.value()
            }

        return config

    # ==================== MÉTODOS DE DATOS ====================

    def load_csv_data(self):
        """Cargar datos desde CSV"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Cargar datos CSV", "", "CSV files (*.csv)"
        )

        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.update_data_info()
                self.update_preview_table()
                self.update_date_columns()
                self.status_label.setText("✅ Datos CSV cargados correctamente")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al cargar CSV: {str(e)}")

    def load_excel_data(self):
        """Cargar datos desde Excel"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Cargar datos Excel", "", "Excel files (*.xlsx *.xls)"
        )

        if file_path:
            try:
                self.data = pd.read_excel(file_path)
                self.update_data_info()
                self.update_preview_table()
                self.update_date_columns()
                self.status_label.setText("✅ Datos Excel cargados correctamente")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al cargar Excel: {str(e)}")

    def generate_demo_data(self):
        """Generar datos de demostración"""
        try:
            np.random.seed(42)
            n_samples = 200

            # Generar datos sintéticos de calidad del agua
            demo_data = {
                'Fecha': pd.date_range('2024-01-01', periods=n_samples, freq='D'),
                'Estacion': np.random.choice(['Norte', 'Sur', 'Centro'], n_samples),
                'pH': np.clip(np.random.normal(7.2, 0.8, n_samples), 5.5, 9.0),
                'Oxigeno_Disuelto': np.clip(np.random.normal(8.5, 1.5, n_samples), 3.0, 15.0),
                'DBO5': np.clip(np.random.exponential(3.0, n_samples), 0.5, 25.0),
                'Coliformes_Fecales': np.clip(np.random.exponential(50, n_samples), 1, 10000),
                'Temperatura': np.clip(np.random.normal(22, 4, n_samples), 10, 35),
                'Fosforo_Total': np.clip(np.random.exponential(0.5, n_samples), 0.01, 5.0),
                'Nitrato': np.clip(np.random.exponential(8, n_samples), 0.1, 50),
                'Turbiedad': np.clip(np.random.exponential(2.0, n_samples), 0.1, 50),
                'Solidos_Totales': np.clip(np.random.normal(150, 50, n_samples), 10, 500)
            }

            self.data = pd.DataFrame(demo_data)

            self.update_data_info()
            self.update_preview_table()
            self.update_date_columns()
            self.status_label.setText("✅ Datos de demostración generados")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generando datos demo: {str(e)}")

    def update_data_info(self):
        """Actualizar información del dataset"""
        if self.data is not None:
            info = f"Dataset: {len(self.data)} muestras × {len(self.data.columns)} variables"
            self.dataset_info.setText(info)
            self.calculate_btn.setEnabled(True)
        else:
            self.dataset_info.setText("Dataset: No cargado")
            self.calculate_btn.setEnabled(False)

    def update_preview_table(self):
        """Actualizar tabla de vista previa"""
        if self.data is not None:
            # Mostrar solo las primeras 10 filas
            preview_data = self.data.head(10)

            self.preview_table.setRowCount(len(preview_data))
            self.preview_table.setColumnCount(len(preview_data.columns))
            self.preview_table.setHorizontalHeaderLabels(preview_data.columns.tolist())

            for i in range(len(preview_data)):
                for j, col in enumerate(preview_data.columns):
                    value = preview_data.iloc[i, j]
                    if pd.isna(value):
                        item_text = "N/A"
                    elif isinstance(value, float):
                        item_text = f"{value:.3f}"
                    else:
                        item_text = str(value)

                    self.preview_table.setItem(i, j, QTableWidgetItem(item_text))

            self.preview_table.resizeColumnsToContents()

    def update_date_columns(self):
        """Actualizar opciones de columnas de fecha"""
        if self.data is not None:
            date_candidates = []

            for col in self.data.columns:
                if 'fecha' in col.lower() or 'date' in col.lower() or 'time' in col.lower():
                    date_candidates.append(col)
                elif self.data[col].dtype == 'datetime64[ns]':
                    date_candidates.append(col)

            self.date_column_combo.clear()
            self.date_column_combo.addItems(date_candidates)

    # ==================== MÉTODOS DE CÁLCULO ====================

    def on_method_changed(self):
        """Manejar cambio de método"""
        method = self.method_combo.currentText()
        self.status_label.setText(f"Método seleccionado: {method.split(' - ')[0]}")

    def calculate_wqi(self):
        """Calcular WQI principal"""
        if self.data is None:
            QMessageBox.warning(self, "Sin datos", "Cargue datos primero")
            return

        # Obtener configuración actual
        method_text = self.method_combo.currentText()
        method = method_text.split(' - ')[0]

        parameters = {}
        weights = {}

        for i in range(self.params_table.rowCount()):
            checkbox = self.params_table.cellWidget(i, 0)
            if checkbox.isChecked():
                param_name = self.params_table.item(i, 1).text()
                weight_spin = self.params_table.cellWidget(i, 2)

                if param_name in WQICalculationEngine.NSF_PARAMETERS:
                    parameters[param_name] = WQICalculationEngine.NSF_PARAMETERS[param_name]
                    weights[param_name] = weight_spin.value()

        if not parameters:
            QMessageBox.warning(self, "Sin parámetros", "Seleccione al menos un parámetro")
            return

        # Verificar que los parámetros existen en los datos
        missing_params = []
        for param in parameters.keys():
            if param not in self.data.columns:
                missing_params.append(param)

        if missing_params:
            QMessageBox.warning(
                self, "Parámetros faltantes",
                f"Los siguientes parámetros no están en los datos:\n{', '.join(missing_params)}"
            )
            return

        # Iniciar cálculo en worker thread
        self.start_calculation(method, parameters, weights)

    def start_calculation(self, method, parameters, weights):
        """Iniciar cálculo en worker thread"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.calculate_btn.setEnabled(False)

        # Limpiar worker anterior
        if self.calculation_worker:
            self.calculation_worker.deleteLater()

        self.calculation_worker = WQICalculationWorker(self.data, method, parameters, weights)
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

        # Actualizar resultados
        self.update_results_display()
        self.update_statistics_cards()

        # Cambiar a tab de resultados
        self.main_tabs.setCurrentIndex(1)

        self.status_label.setText("✅ Cálculos WQI completados")

        QMessageBox.information(
            self, "Cálculo Completado",
            f"WQI calculado para {results['total_samples']} muestras\n"
            f"Método: {results['method']}"
        )

    def on_calculation_error(self, error_message):
        """Manejar error en cálculo"""
        self.progress_bar.setVisible(False)
        self.calculate_btn.setEnabled(True)
        self.status_label.setText("❌ Error en cálculo")

        QMessageBox.critical(self, "Error de Cálculo", error_message)

    def update_results_display(self):
        """Actualizar display de resultados"""
        if not self.current_results:
            return

        results = self.current_results['results']

        # Actualizar tabla de resultados
        self.results_table.setRowCount(len(results))
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels([
            "Índice", "WQI", "Clasificación", "Método"
        ])

        for i, result in enumerate(results):
            self.results_table.setItem(i, 0, QTableWidgetItem(str(result.get('index', i))))

            wqi_value = result.get('wqi', 0)
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{wqi_value:.2f}"))

            classification = result.get('classification', {})
            class_label = classification.get('label', 'N/A')
            self.results_table.setItem(i, 2, QTableWidgetItem(class_label))

            method = result.get('method', 'N/A')
            self.results_table.setItem(i, 3, QTableWidgetItem(method))

            # Colorear fila según calidad
            color = classification.get('color', '#FFFFFF')
            for j in range(4):
                item = self.results_table.item(i, j)
                if item:
                    item.setBackground(QColor(color))

        self.results_table.resizeColumnsToContents()

    def update_statistics_cards(self):
        """Actualizar cards de estadísticas"""
        if not self.current_results:
            return

        stats = self.current_results['statistics']
        quality_dist = self.current_results['quality_distribution']

        # Actualizar valores en cards
        if 'count' in self.stats_cards:
            count_label = self.stats_cards['count'].layout().itemAt(2).widget()
            count_label.setText(str(stats.get('count', 0)))

        if 'mean' in self.stats_cards:
            mean_label = self.stats_cards['mean'].layout().itemAt(2).widget()
            mean_label.setText(f"{stats.get('mean', 0):.2f}")

        if 'std' in self.stats_cards:
            std_label = self.stats_cards['std'].layout().itemAt(2).widget()
            std_label.setText(f"{stats.get('std', 0):.2f}")

        if 'mode' in self.stats_cards and quality_dist:
            modal_quality = max(quality_dist.keys(), key=lambda k: quality_dist[k])
            mode_label = self.stats_cards['mode'].layout().itemAt(2).widget()
            mode_label.setText(modal_quality)

    # ==================== MÉTODOS DE ANÁLISIS ====================

    def analyze_temporal_trends(self):
        """Analizar tendencias temporales"""
        if self.data is None or self.current_results is None:
            QMessageBox.warning(self, "Sin datos", "Calcule WQI primero")
            return

        date_column = self.date_column_combo.currentText()
        if not date_column:
            QMessageBox.warning(self, "Sin columna de fecha", "Seleccione una columna de fecha")
            return

        # Crear DataFrame temporal con resultados
        wqi_values = [r['wqi'] for r in self.current_results['results']]
        temp_df = self.data.copy()
        temp_df['WQI'] = wqi_values

        # Realizar análisis temporal
        temporal_analysis = TemporalAnalysisEngine.analyze_temporal_trends(
            temp_df, date_column, 'WQI'
        )

        if 'error' in temporal_analysis:
            QMessageBox.critical(self, "Error", temporal_analysis['error'])
            return

        # Mostrar resultados
        self.display_temporal_analysis(temporal_analysis)

    def display_temporal_analysis(self, analysis):
        """Mostrar análisis temporal"""
        results_text = "=== 📈 ANÁLISIS TEMPORAL DEL WQI ===\n\n"

        # Tendencia general
        if 'tendencia_general' in analysis:
            trend = analysis['tendencia_general']
            results_text += f"🔍 TENDENCIA GENERAL:\n"
            results_text += f"   Dirección: {trend['direccion']}\n"
            results_text += f"   Pendiente: {trend['pendiente']:.4f}\n"
            results_text += f"   Correlación: {trend['correlacion']:.3f}\n"
            results_text += f"   Significativa: {'Sí' if trend['significativa'] else 'No'}\n\n"

        # Variación estacional
        if 'variacion_estacional' in analysis:
            seasonal = analysis['variacion_estacional']
            results_text += f"📅 VARIACIÓN ESTACIONAL:\n"
            results_text += f"   Mejor mes: {seasonal['mejor_mes']}\n"
            results_text += f"   Peor mes: {seasonal['peor_mes']}\n"
            results_text += f"   Coef. Variación: {seasonal['coeficiente_variacion']:.3f}\n\n"

        # Eventos extremos
        if 'eventos_extremos' in analysis:
            events = analysis['eventos_extremos']
            results_text += f"⚠️ EVENTOS EXTREMOS:\n"
            results_text += f"   Calidad muy baja: {events['calidad_muy_baja']['count']} eventos\n"
            results_text += f"   Calidad muy alta: {events['calidad_muy_alta']['count']} eventos\n"

        self.temporal_results.setText(results_text)

    def generate_forecast(self):
        """Generar pronóstico de WQI"""
        if self.data is None or self.current_results is None:
            QMessageBox.warning(self, "Sin datos", "Calcule WQI primero")
            return

        date_column = self.date_column_combo.currentText()
        if not date_column:
            QMessageBox.warning(self, "Sin columna de fecha", "Seleccione una columna de fecha")
            return

        # Crear DataFrame temporal con resultados
        wqi_values = [r['wqi'] for r in self.current_results['results']]
        temp_df = self.data.copy()
        temp_df['WQI'] = wqi_values

        # Generar pronóstico
        forecast = TemporalAnalysisEngine.generate_forecast(
            temp_df, date_column, 'WQI', periods=30
        )

        if 'error' in forecast:
            QMessageBox.critical(self, "Error", forecast['error'])
            return

        # Mostrar pronóstico
        self.display_forecast(forecast)

    def display_forecast(self, forecast):
        """Mostrar pronóstico"""
        current_text = self.temporal_results.toPlainText()

        forecast_text = "\n\n=== 🔮 PRONÓSTICO WQI (30 días) ===\n\n"

        if 'parametros' in forecast:
            params = forecast['parametros']
            forecast_text += f"📊 PARÁMETROS DEL MODELO:\n"
            forecast_text += f"   Alpha: {params['alpha']:.3f}\n"
            forecast_text += f"   Tendencia: {params['tendencia']:.3f}\n"
            forecast_text += f"   Último valor: {params['ultimo_valor']:.2f}\n\n"

        forecast_text += f"📅 VALORES PRONOSTICADOS (próximos 7 días):\n"
        for i in range(min(7, len(forecast['fechas']))):
            fecha = forecast['fechas'][i]
            valor = forecast['valores_pronosticados'][i]
            forecast_text += f"   {fecha}: {valor:.2f}\n"

        if len(forecast['fechas']) > 7:
            forecast_text += f"   ... y {len(forecast['fechas']) - 7} días más\n"

        self.temporal_results.setText(current_text + forecast_text)

    def run_comparison(self):
        """Ejecutar comparación de métodos"""
        if self.data is None:
            QMessageBox.warning(self, "Sin datos", "Cargue datos primero")
            return

        # Obtener métodos seleccionados
        methods_to_compare = []
        if self.compare_nsf.isChecked():
            methods_to_compare.append('NSF')
        if self.compare_ccme.isChecked():
            methods_to_compare.append('CCME')
        if self.compare_arithmetic.isChecked():
            methods_to_compare.append('Weighted_Arithmetic')

        if len(methods_to_compare) < 2:
            QMessageBox.warning(self, "Pocos métodos", "Seleccione al menos 2 métodos para comparar")
            return

        # Ejecutar comparación
        self.execute_comparison(methods_to_compare)

    def execute_comparison(self, methods):
        """Ejecutar comparación entre métodos"""
        comparison_results = {}

        # Obtener configuración de parámetros
        parameters = {}
        weights = {}

        for i in range(self.params_table.rowCount()):
            checkbox = self.params_table.cellWidget(i, 0)
            if checkbox.isChecked():
                param_name = self.params_table.item(i, 1).text()
                weight_spin = self.params_table.cellWidget(i, 2)

                if param_name in WQICalculationEngine.NSF_PARAMETERS:
                    parameters[param_name] = WQICalculationEngine.NSF_PARAMETERS[param_name]
                    weights[param_name] = weight_spin.value()

        # Calcular WQI para cada método
        for method in methods:
            method_results = []

            for _, row in self.data.iterrows():
                if method == 'NSF':
                    result = WQICalculationEngine.calculate_nsf_wqi(row, parameters, weights)
                elif method == 'CCME':
                    result = WQICalculationEngine.calculate_ccme_wqi(row, parameters)
                elif method == 'Weighted_Arithmetic':
                    result = WQICalculationEngine.calculate_weighted_arithmetic_wqi(row, parameters, weights)

                method_results.append(result.get('wqi', 0))

            comparison_results[method] = method_results

        # Mostrar comparación
        self.display_comparison(comparison_results)

    def display_comparison(self, results):
        """Mostrar resultados de comparación"""
        comparison_text = "=== ⚖️ COMPARACIÓN DE MÉTODOS WQI ===\n\n"

        # Estadísticas por método
        for method, values in results.items():
            comparison_text += f"📊 {method}:\n"
            comparison_text += f"   Media: {np.mean(values):.2f}\n"
            comparison_text += f"   Desv. Std: {np.std(values):.2f}\n"
            comparison_text += f"   Min: {np.min(values):.2f}\n"
            comparison_text += f"   Max: {np.max(values):.2f}\n\n"

        # Correlaciones entre métodos
        if len(results) >= 2:
            comparison_text += "🔗 CORRELACIONES ENTRE MÉTODOS:\n"
            methods_list = list(results.keys())

            for i in range(len(methods_list)):
                for j in range(i + 1, len(methods_list)):
                    method1, method2 = methods_list[i], methods_list[j]
                    correlation = np.corrcoef(results[method1], results[method2])[0, 1]
                    comparison_text += f"   {method1} vs {method2}: {correlation:.3f}\n"

        self.comparison_results.setText(comparison_text)

    # ==================== MÉTODOS DE EXPORTACIÓN ====================

    def export_results(self):
        """Exportar resultados"""
        if not self.current_results:
            QMessageBox.warning(self, "Sin resultados", "Calcule WQI primero")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Exportar Resultados WQI",
            f"resultados_wqi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            "Excel (*.xlsx);;CSV (*.csv);;JSON (*.json)"
        )

        if file_path:
            try:
                if file_path.endswith('.xlsx'):
                    self.export_to_excel(file_path)
                elif file_path.endswith('.csv'):
                    self.export_to_csv(file_path)
                elif file_path.endswith('.json'):
                    self.export_to_json(file_path)

                QMessageBox.information(self, "Éxito", "Resultados exportados correctamente")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al exportar: {str(e)}")

    def export_to_excel(self, file_path):
        """Exportar a Excel"""
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Hoja de resultados principales
            results_df = self.create_results_dataframe()
            results_df.to_excel(writer, sheet_name='Resultados_WQI', index=False)

            # Hoja de estadísticas
            stats_df = self.create_statistics_dataframe()
            stats_df.to_excel(writer, sheet_name='Estadisticas', index=False)

            # Hoja de configuración
            config_df = self.create_configuration_dataframe()
            config_df.to_excel(writer, sheet_name='Configuracion', index=False)

    def export_to_csv(self, file_path):
        """Exportar a CSV"""
        results_df = self.create_results_dataframe()
        results_df.to_csv(file_path, index=False, encoding='utf-8')

    def export_to_json(self, file_path):
        """Exportar a JSON"""
        export_data = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'method': self.current_results['method'],
                'total_samples': self.current_results['total_samples']
            },
            'results': self.current_results['results'],
            'statistics': self.current_results['statistics'],
            'quality_distribution': self.current_results['quality_distribution'],
            'configuration': self.get_current_configuration()
        }

        import json
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

    def create_results_dataframe(self):
        """Crear DataFrame de resultados"""
        results = self.current_results['results']

        data = []
        for result in results:
            row = {
                'Indice': result.get('index', ''),
                'WQI': result.get('wqi', 0),
                'Clasificacion': result.get('classification', {}).get('label', ''),
                'Metodo': result.get('method', ''),
                'Color_Calidad': result.get('classification', {}).get('color', ''),
                'Descripcion': result.get('classification', {}).get('description', '')
            }
            data.append(row)

        return pd.DataFrame(data)

    def create_statistics_dataframe(self):
        """Crear DataFrame de estadísticas"""
        stats = self.current_results['statistics']
        quality_dist = self.current_results['quality_distribution']

        # Estadísticas descriptivas
        stats_data = []
        for key, value in stats.items():
            stats_data.append({'Estadistica': key, 'Valor': value})

        stats_df = pd.DataFrame(stats_data)

        return stats_df

    def create_configuration_dataframe(self):
        """Crear DataFrame de configuración"""
        config = self.get_current_configuration()

        config_data = []
        config_data.append({'Parametro': 'Metodo', 'Valor': config['method']})

        for param, settings in config['parameters'].items():
            config_data.append({
                'Parametro': f'{param}_incluido',
                'Valor': settings['included']
            })
            config_data.append({
                'Parametro': f'{param}_peso',
                'Valor': settings['weight']
            })

        return pd.DataFrame(config_data)

    # ==================== MÉTODOS DE AYUDA ====================

    def show_help(self):
        """Mostrar ayuda completa"""
        help_text = """
💧 CALCULADORA AVANZADA WQI - GUÍA COMPLETA

🎯 PROPÓSITO:
El Índice de Calidad del Agua (WQI) combina múltiples parámetros 
fisicoquímicos en un valor único entre 0 y 100 que indica la 
calidad general del agua.

📊 MÉTODOS DISPONIBLES:

1. NSF (National Sanitation Foundation):
   • Fórmula: WQI = Π(Qi^Wi)
   • Método más utilizado mundialmente
   • Usa 9 parámetros estándar

2. CCME (Canadian Council of Ministers):
   • Fórmula: WQI = 100 - (√(F1² + F2² + F3²) / 1.732)
   • Considera excesos sobre objetivos de calidad
   • Robusto para diferentes tipos de agua

3. Aritmético Ponderado:
   • Fórmula: WQI = Σ(Wi × Qi)
   • Promedio ponderado simple
   • Más fácil de interpretar

🔧 PARÁMETROS PRINCIPALES:
• pH: Acidez/alcalinidad (óptimo: 6.5-8.5)
• Oxígeno Disuelto: Disponibilidad para vida acuática (>5 mg/L)
• DBO5: Demanda bioquímica de oxígeno (<3 mg/L)
• Coliformes Fecales: Contaminación bacteriana (<20 UFC/100mL)
• Temperatura: Rango térmico adecuado (15-25°C)
• Fósforo Total: Nutriente limitante (<0.1 mg/L)
• Nitratos: Contaminación por fertilizantes (<10 mg/L)
• Turbiedad: Claridad del agua (<5 NTU)
• Sólidos Totales: Contenido mineral (<100 mg/L)

📈 INTERPRETACIÓN WQI:
• 90-100: Excelente - Agua de alta calidad
• 70-89: Buena - Calidad aceptable
• 50-69: Regular - Calidad moderada
• 25-49: Deficiente - Calidad pobre
• 0-24: Muy Deficiente - Calidad muy mala

🚀 FLUJO DE TRABAJO:
1. Cargar datos (CSV, Excel o generar demo)
2. Configurar parámetros y pesos
3. Seleccionar método de cálculo
4. Ejecutar cálculo WQI
5. Analizar resultados y tendencias
6. Comparar métodos (opcional)
7. Exportar resultados

💡 CONSEJOS:
• Use normalización de pesos para métodos ponderados
• Para análisis temporal, asegúrese de tener columna de fecha
• Compare métodos para validar resultados
• Exporte configuración para reproducibilidad

🔍 ANÁLISIS TEMPORAL:
• Tendencias: Identifica si la calidad mejora o empeora
• Estacionalidad: Detecta patrones por mes/estación
• Pronóstico: Predice valores futuros usando tendencias
• Eventos extremos: Identifica outliers temporales

⚖️ COMPARACIONES:
• Ejecute múltiples métodos simultáneamente
• Compare correlaciones entre métodos
• Identifique diferencias en clasificaciones
• Valide consistencia de resultados
"""

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("💧 Ayuda - Calculadora WQI")
        msg_box.setText(help_text)
        msg_box.setStyleSheet("QMessageBox { min-width: 600px; }")
        msg_box.exec_()

    def apply_styles(self):
        """Aplicar estilos CSS"""
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
                background-color: #f8f9fa;
            }
            
            #mainHeaderFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e3f2fd, stop:1 #bbdefb);
                border: 2px solid #2196f3;
                border-radius: 10px;
                padding: 15px;
                margin: 5px;
            }
            
            #mainTitle {
                font-size: 22px;
                font-weight: bold;
                color: #0d47a1;
                margin: 10px 0;
            }
            
            #formulaSubtitle {
                font-size: 14px;
                color: #1565c0;
                font-style: italic;
                margin: 5px 0;
            }
            
            #datasetInfo {
                font-size: 13px;
                color: #1976d2;
                font-weight: 500;
                margin: 5px 0;
            }
            
            #calculateButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4caf50, stop:1 #388e3c);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 16px;
                min-width: 180px;
            }
            
            #calculateButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #66bb6a, stop:1 #4caf50);
            }
            
            #calculateButton:disabled {
                background: #bdbdbd;
                color: #757575;
            }
            
            #exportButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ff9800, stop:1 #f57c00);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 20px;
                font-weight: bold;
                font-size: 14px;
            }
            
            #exportButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffb74d, stop:1 #ff9800);
            }
            
            #statisticsFrame {
                background: #ffffff;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 10px;
                margin: 5px;
            }
            
            #statCard {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff, stop:1 #f5f5f5);
                border: 2px solid #e0e0e0;
                border-radius: 12px;
                padding: 15px;
                margin: 5px;
                min-width: 150px;
                min-height: 100px;
            }
            
            #statCard:hover {
                border-color: #2196f3;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f3f9ff, stop:1 #e8f4fd);
            }
            
            #statIcon {
                font-size: 32px;
                margin: 5px 0;
            }
            
            #statTitle {
                font-size: 12px;
                font-weight: bold;
                color: #424242;
                margin: 5px 0;
            }
            
            #statValue {
                font-size: 18px;
                font-weight: bold;
                color: #1976d2;
                margin: 5px 0;
            }
            
            #resultsTable {
                background: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                gridline-color: #f0f0f0;
                selection-background-color: #e3f2fd;
            }
            
            #resultsTable::item {
                padding: 8px;
                border-bottom: 1px solid #f0f0f0;
            }
            
            #resultsTable QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f5f5f5, stop:1 #e0e0e0);
                padding: 10px;
                border: 1px solid #d0d0d0;
                font-weight: bold;
                font-size: 12px;
            }
            
            #graphsPlaceholder {
                font-size: 16px;
                color: #757575;
                background: #fafafa;
                border: 2px dashed #e0e0e0;
                border-radius: 10px;
                padding: 40px;
                margin: 10px;
            }
            
            #temporalResults, #comparisonResults {
                background: #ffffff;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                padding: 15px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                line-height: 1.4;
            }
            
            #statusLabel {
                font-size: 13px;
                font-weight: 500;
                color: #2e7d32;
                padding: 5px 10px;
                background: #e8f5e8;
                border-radius: 15px;
                border: 1px solid #c8e6c9;
            }
            
            #footerInfo {
                font-size: 14px;
                color: #666;
                font-style: italic;
            }
            
            #mainFooter {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #fafafa, stop:1 #f0f0f0);
                border-top: 2px solid #e0e0e0;
                border-radius: 8px;
                padding: 15px;
                margin: 5px;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 10px;
                background: #ffffff;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px;
                color: #424242;
                background: #ffffff;
                font-size: 13px;
            }
            
            QTabWidget::pane {
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                background: #ffffff;
                margin: 2px;
            }
            
            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f5f5f5, stop:1 #e0e0e0);
                border: 2px solid #d0d0d0;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 12px 20px;
                margin-right: 2px;
                font-weight: bold;
                font-size: 13px;
                min-width: 120px;
            }
            
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff, stop:1 #f8f9fa);
                border-color: #2196f3;
                border-bottom: 2px solid #ffffff;
                color: #1976d2;
            }
            
            QTabBar::tab:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f0f4ff, stop:1 #e1ecff);
            }
            
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f5f5f5, stop:1 #e0e0e0);
                border: 2px solid #d0d0d0;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 12px;
                color: #424242;
            }
            
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e8f4fd, stop:1 #d1e7dd);
                border-color: #2196f3;
                color: #1976d2;
            }
            
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #d1e7dd, stop:1 #c8e6c9);
            }
            
            QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit, QLineEdit {
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                padding: 6px 10px;
                background: #ffffff;
                font-size: 12px;
                min-height: 20px;
            }
            
            QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, 
            QDateEdit:focus, QLineEdit:focus {
                border-color: #2196f3;
            }
            
            QProgressBar {
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                background: #f5f5f5;
                text-align: center;
                font-weight: bold;
                font-size: 11px;
                min-height: 20px;
            }
            
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4caf50, stop:1 #388e3c);
                border-radius: 6px;
                margin: 1px;
            }
            
            QCheckBox {
                spacing: 8px;
                font-size: 12px;
                color: #424242;
            }
            
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #d0d0d0;
                border-radius: 4px;
                background: #ffffff;
            }
            
            QCheckBox::indicator:checked {
                background: #2196f3;
                border-color: #1976d2;
                image: none;
            }
            
            QCheckBox::indicator:checked:after {
                content: "✓";
                color: white;
                font-weight: bold;
            }
        """)

# ==================== FUNCIÓN PRINCIPAL ====================

def main():
    """Función principal para ejecutar la calculadora WQI"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setApplicationName("Calculadora WQI Avanzada")
    app.setApplicationVersion("2.0")

    # Crear y mostrar ventana principal
    calculator = WQICalculatorAdvanced()
    calculator.show()

    # Mensaje de bienvenida