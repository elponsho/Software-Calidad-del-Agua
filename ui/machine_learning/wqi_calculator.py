"""
wqi_calculator.py - Motor COMPLETO e INDEPENDIENTE para cálculo del Índice de Calidad del Agua
VERSIÓN CORREGIDA con mapeo automático y fórmulas NSF oficiales
Incluye: NSF WQI corregido, CCME WQI, análisis temporal y comparaciones
Compatible con datos IDEAM y otros estándares
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# ==================== MAPEO AUTOMÁTICO DE PARÁMETROS ====================

class WQIParameterMapper:
    """Mapeo automático de parámetros WQI - CORREGIDO para datos IDEAM"""

    # Mapeo actualizado basado en el CSV TODOS_2000.csv
    PARAMETER_MAPPINGS = {
        'pH': ['pH', 'ph', 'PH'],
        'Oxigeno_Disuelto': ['DO', 'DissolvedOxygen', 'O2', 'OD'],
        'DBO5': ['BOD5', 'DBO5', 'BOD', 'DBO'],
        'Coliformes_Fecales': ['FC', 'FecalColiforms', 'CF', 'TC'],
        'Temperatura': ['WT', 'WaterTemp', 'Temperatura', 'Temperature', 'ET'],
        'Fosforo_Total': ['TP', 'TotalPhosphorus', 'P_Total', 'PT'],
        'Nitrato': ['NO3', 'Nitrate', 'Nitrato', 'NO3-N'],
        'Turbiedad': ['TBD', 'Turbidity', 'Turb', 'NTU'],
        'Solidos_Totales': ['TS', 'TotalSolids', 'TSS', 'ST']
    }

    @classmethod
    def map_parameters(cls, data_columns):
        """Mapea automáticamente las columnas a parámetros estándar"""
        mapping = {}

        for standard_name, possible_names in cls.PARAMETER_MAPPINGS.items():
            for possible_name in possible_names:
                # Buscar coincidencia exacta
                if possible_name in data_columns:
                    mapping[standard_name] = possible_name
                    break
                # Buscar sin importar mayúsculas/minúsculas
                for col in data_columns:
                    if col.upper() == possible_name.upper():
                        mapping[standard_name] = col
                        break
                if standard_name in mapping:
                    break

        return mapping

# ==================== ECUACIONES NSF CORREGIDAS ====================

class NSFWQICalculatorFixed:
    """Calculadora NSF WQI con fórmulas oficiales corregidas"""

    @staticmethod
    def calculate_ph_subindex(ph_value):
        """Cálculo correcto del subíndice de pH según NSF"""
        if pd.isna(ph_value) or ph_value <= 0:
            return 0

        # Fórmulas oficiales NSF para pH
        if ph_value <= 2:
            return 2
        elif ph_value <= 3:
            return 13 + 11.5 * (ph_value - 2)
        elif ph_value <= 4:
            return 25 + 17.2 * (ph_value - 3) - 2.42 * (ph_value - 3)**2
        elif ph_value <= 6.2:
            return 50 + 20 * (ph_value - 4) - 5 * (ph_value - 4)**2
        elif ph_value <= 7:
            return 80 + 25 * (ph_value - 6.2)
        elif ph_value <= 8:
            return 100 - 12.5 * (ph_value - 7)**2
        elif ph_value <= 8.5:
            return 90 - 20 * (ph_value - 8)
        elif ph_value <= 9:
            return 80 - 30 * (ph_value - 8.5)
        elif ph_value <= 10:
            return 65 - 15 * (ph_value - 9)
        elif ph_value <= 12:
            return 25 - 5 * (ph_value - 10)
        else:
            return 3

    @staticmethod
    def calculate_do_subindex(do_value, temp_value=20):
        """Cálculo correcto del subíndice de Oxígeno Disuelto"""
        if pd.isna(do_value) or do_value < 0:
            return 0

        # Calcular saturación de oxígeno según temperatura
        if pd.isna(temp_value):
            temp_value = 20  # Temperatura por defecto

        # Fórmula de saturación de oxígeno
        do_sat = 14.652 - 0.41022 * temp_value + 0.007991 * temp_value**2 - 0.000077774 * temp_value**3

        percent_sat = min(150, (do_value / do_sat) * 100)

        # Fórmulas NSF para oxígeno disuelto
        if percent_sat >= 140:
            return 50
        elif percent_sat >= 120:
            return 50 + 2.5 * (140 - percent_sat)
        elif percent_sat >= 100:
            return 100 - 2.5 * (percent_sat - 100)
        elif percent_sat >= 85:
            return 95 + 0.33 * (percent_sat - 85)
        elif percent_sat >= 50:
            return 25 + 2 * (percent_sat - 50)
        else:
            return max(0, percent_sat / 2)

    @staticmethod
    def calculate_bod5_subindex(bod5_value):
        """Cálculo correcto del subíndice de DBO5"""
        if pd.isna(bod5_value) or bod5_value < 0:
            return 100  # Valor máximo si no hay contaminante

        # Fórmula NSF para DBO5 (curva decreciente)
        if bod5_value <= 1:
            return 100 - 10 * bod5_value
        elif bod5_value <= 2:
            return 90 - 15 * (bod5_value - 1)
        elif bod5_value <= 3:
            return 75 - 10 * (bod5_value - 2)
        elif bod5_value <= 5:
            return 65 - 12.5 * (bod5_value - 3)
        elif bod5_value <= 10:
            return 40 - 6 * (bod5_value - 5)
        elif bod5_value <= 15:
            return 10 - 1.5 * (bod5_value - 10)
        elif bod5_value <= 20:
            return 2.5 - 0.4 * (bod5_value - 15)
        else:
            return max(0, 0.5 - 0.025 * (bod5_value - 20))

    @staticmethod
    def calculate_fc_subindex(fc_value):
        """Cálculo correcto del subíndice de Coliformes Fecales"""
        if pd.isna(fc_value) or fc_value <= 0:
            return 100

        # Fórmula NSF para coliformes fecales (escala logarítmica)
        if fc_value <= 1:
            return 100
        elif fc_value <= 2:
            return 95
        elif fc_value <= 10:
            return 90 - 10 * np.log10(fc_value/2)
        elif fc_value <= 20:
            return 70 - 8 * np.log10(fc_value/10)
        elif fc_value <= 100:
            return 60 - 12 * np.log10(fc_value/20)
        elif fc_value <= 1000:
            return 40 - 15 * np.log10(fc_value/100)
        elif fc_value <= 10000:
            return 20 - 10 * np.log10(fc_value/1000)
        else:
            return max(0, 10 - 5 * np.log10(fc_value/10000))

    @staticmethod
    def calculate_temperature_subindex(temp_value):
        """Cálculo correcto del subíndice de Temperatura"""
        if pd.isna(temp_value):
            return 90  # Valor alto por defecto

        # Fórmula NSF para temperatura (óptimo alrededor de 15-20°C)
        if temp_value <= 5:
            return max(0, 50 + 8 * temp_value)
        elif temp_value <= 15:
            return 90 + 1 * (temp_value - 5)
        elif temp_value <= 20:
            return 100
        elif temp_value <= 25:
            return 100 - 2 * (temp_value - 20)
        elif temp_value <= 30:
            return 90 - 4 * (temp_value - 25)
        elif temp_value <= 35:
            return 70 - 6 * (temp_value - 30)
        else:
            return max(0, 40 - 4 * (temp_value - 35))

    @staticmethod
    def calculate_phosphorus_subindex(tp_value):
        """Cálculo correcto del subíndice de Fósforo Total"""
        if pd.isna(tp_value) or tp_value < 0:
            return 100

        # Fórmula NSF para fósforo total (mg/L)
        if tp_value <= 0.01:
            return 100
        elif tp_value <= 0.05:
            return 100 - 400 * (tp_value - 0.01)
        elif tp_value <= 0.1:
            return 84 - 300 * (tp_value - 0.05)
        elif tp_value <= 0.5:
            return 69 - 50 * (tp_value - 0.1)
        elif tp_value <= 1:
            return 49 - 30 * (tp_value - 0.5)
        elif tp_value <= 5:
            return 34 - 6 * (tp_value - 1)
        else:
            return max(0, 10 - 2 * (tp_value - 5))

    @staticmethod
    def calculate_nitrate_subindex(no3_value):
        """Cálculo correcto del subíndice de Nitrato"""
        if pd.isna(no3_value) or no3_value < 0:
            return 100

        # Fórmula NSF para nitrato (mg/L)
        if no3_value <= 1:
            return 100
        elif no3_value <= 5:
            return 100 - 8 * (no3_value - 1)
        elif no3_value <= 10:
            return 68 - 6 * (no3_value - 5)
        elif no3_value <= 25:
            return 38 - 1.5 * (no3_value - 10)
        elif no3_value <= 50:
            return 15.5 - 0.6 * (no3_value - 25)
        else:
            return max(0, 0.5 - 0.01 * (no3_value - 50))

    @staticmethod
    def calculate_turbidity_subindex(tbd_value):
        """Cálculo correcto del subíndice de Turbiedad"""
        if pd.isna(tbd_value) or tbd_value < 0:
            return 100

        # Fórmula NSF para turbiedad (NTU)
        if tbd_value <= 1:
            return 100
        elif tbd_value <= 5:
            return 100 - 20 * (tbd_value - 1)
        elif tbd_value <= 10:
            return 20 - 3 * (tbd_value - 5)
        elif tbd_value <= 25:
            return 5 - 0.2 * (tbd_value - 10)
        elif tbd_value <= 100:
            return max(0, 2 - 0.025 * (tbd_value - 25))
        else:
            return 0

    @staticmethod
    def calculate_solids_subindex(ts_value):
        """Cálculo correcto del subíndice de Sólidos Totales"""
        if pd.isna(ts_value) or ts_value < 0:
            return 100

        # Fórmula NSF para sólidos totales (mg/L)
        if ts_value <= 50:
            return 100
        elif ts_value <= 150:
            return 100 - 40 * (ts_value - 50) / 100
        elif ts_value <= 300:
            return 60 - 30 * (ts_value - 150) / 150
        elif ts_value <= 500:
            return 30 - 25 * (ts_value - 300) / 200
        else:
            return max(0, 5 - 5 * (ts_value - 500) / 500)

# ==================== MOTOR DE CÁLCULO WQI CORREGIDO ====================

class WQICalculationEngine:
    """Motor de cálculo WQI - VERSIÓN CORREGIDA con mapeo automático"""

    # Parámetros estándar NSF WQI (mantener compatibilidad)
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
        Calcular NSF WQI usando fórmulas corregidas con mapeo automático
        """
        # Mapear parámetros automáticamente si no se especifican
        if isinstance(data_row, pd.Series):
            parameter_mapping = WQIParameterMapper.map_parameters(data_row.index.tolist())
        else:
            parameter_mapping = WQIParameterMapper.map_parameters(list(data_row.keys()))

        if parameters is None:
            parameters = WQICalculationEngine.NSF_PARAMETERS

        # Pesos estándar NSF
        default_weights = {param: config['weight'] for param, config in parameters.items()}
        if weights is None:
            weights = default_weights

        # Calcular subíndices para parámetros disponibles
        subindices = {}
        used_weights = {}

        for standard_param, data_column in parameter_mapping.items():
            if data_column in data_row.index and standard_param in weights:
                value = data_row[data_column]

                if pd.notna(value):
                    # Calcular subíndice usando las fórmulas corregidas
                    if standard_param == 'pH':
                        qi = NSFWQICalculatorFixed.calculate_ph_subindex(value)
                    elif standard_param == 'Oxigeno_Disuelto':
                        temp_col = parameter_mapping.get('Temperatura')
                        temp_val = data_row.get(temp_col, 20) if temp_col else 20
                        qi = NSFWQICalculatorFixed.calculate_do_subindex(value, temp_val)
                    elif standard_param == 'DBO5':
                        qi = NSFWQICalculatorFixed.calculate_bod5_subindex(value)
                    elif standard_param == 'Coliformes_Fecales':
                        qi = NSFWQICalculatorFixed.calculate_fc_subindex(value)
                    elif standard_param == 'Temperatura':
                        qi = NSFWQICalculatorFixed.calculate_temperature_subindex(value)
                    elif standard_param == 'Fosforo_Total':
                        qi = NSFWQICalculatorFixed.calculate_phosphorus_subindex(value)
                    elif standard_param == 'Nitrato':
                        qi = NSFWQICalculatorFixed.calculate_nitrate_subindex(value)
                    elif standard_param == 'Turbiedad':
                        qi = NSFWQICalculatorFixed.calculate_turbidity_subindex(value)
                    elif standard_param == 'Solidos_Totales':
                        qi = NSFWQICalculatorFixed.calculate_solids_subindex(value)
                    else:
                        qi = 50  # Valor neutro

                    if not pd.isna(qi) and qi >= 0:
                        subindices[standard_param] = qi
                        used_weights[standard_param] = weights[standard_param]

        if not subindices:
            return {
                'wqi': 0,
                'method': 'NSF',
                'error': 'No se pudieron calcular subíndices',
                'parameter_mapping': parameter_mapping,
                'available_params': list(parameter_mapping.keys())
            }

        # Normalizar pesos
        total_weight = sum(used_weights.values())
        if total_weight == 0:
            return {'wqi': 0, 'method': 'NSF', 'error': 'Pesos totales = 0'}

        normalized_weights = {param: weight/total_weight for param, weight in used_weights.items()}

        # Calcular WQI usando producto ponderado: WQI = Π(Qi^Wi)
        wqi_product = 1.0
        used_parameters = []

        for param, qi in subindices.items():
            if param in normalized_weights:
                weight = normalized_weights[param]
                wqi_product *= (qi ** weight)
                used_parameters.append({
                    'parameter': param,
                    'mapped_column': parameter_mapping.get(param, param),
                    'value': data_row[parameter_mapping[param]] if param in parameter_mapping else 'N/A',
                    'qi': qi,
                    'weight': weight
                })

        return {
            'wqi': wqi_product,
            'method': 'NSF',
            'subindices': subindices,
            'weights_used': normalized_weights,
            'used_parameters': used_parameters,
            'parameter_mapping': parameter_mapping
        }

    @staticmethod
    def calculate_ccme_wqi(data_row, parameters=None, objectives=None):
        """
        Calcular WQI usando el método CCME (Canadian Council of Ministers of Environment)
        """
        # Mapear parámetros automáticamente
        if isinstance(data_row, pd.Series):
            parameter_mapping = WQIParameterMapper.map_parameters(data_row.index.tolist())
        else:
            parameter_mapping = WQIParameterMapper.map_parameters(list(data_row.keys()))

        if parameters is None:
            parameters = list(WQICalculationEngine.NSF_PARAMETERS.keys())

        if objectives is None:
            objectives = {param: config['optimal'][1] for param, config in
                         WQICalculationEngine.NSF_PARAMETERS.items()}

        # Variables CCME
        failed_tests = 0
        total_tests = 0
        excursions = []

        for standard_param in parameters:
            if standard_param in parameter_mapping:
                data_column = parameter_mapping[standard_param]
                if data_column in data_row.index and pd.notna(data_row[data_column]) and standard_param in objectives:
                    total_tests += 1
                    value = data_row[data_column]
                    objective = objectives[standard_param]

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
            nse = sum(excursions) / len(excursions)
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
            'method': 'CCME',
            'parameter_mapping': parameter_mapping
        }

    @staticmethod
    def calculate_weighted_arithmetic_wqi(data_row, parameters=None, weights=None):
        """
        Calcular WQI usando promedio aritmético ponderado
        """
        # Usar el cálculo NSF pero con promedio aritmético en lugar de producto
        nsf_result = WQICalculationEngine.calculate_nsf_wqi(data_row, parameters, weights)

        if 'error' in nsf_result:
            nsf_result['method'] = 'Weighted Arithmetic'
            return nsf_result

        # Convertir a promedio aritmético ponderado
        if 'subindices' in nsf_result and 'weights_used' in nsf_result:
            weighted_sum = 0
            total_weight = 0

            for param, qi in nsf_result['subindices'].items():
                if param in nsf_result['weights_used']:
                    weight = nsf_result['weights_used'][param]
                    weighted_sum += weight * qi
                    total_weight += weight

            if total_weight > 0:
                wqi_arithmetic = weighted_sum / total_weight
            else:
                wqi_arithmetic = 0

            nsf_result['wqi'] = wqi_arithmetic
            nsf_result['method'] = 'Weighted Arithmetic'

        return nsf_result

    @staticmethod
    def classify_water_quality(wqi_value):
        """Clasificar calidad del agua según valor WQI"""
        if pd.isna(wqi_value):
            return {'label': 'No Calculado', 'color': '#666666', 'description': 'Sin datos'}

        for (min_val, max_val), classification in WQICalculationEngine.QUALITY_CLASSIFICATION.items():
            if min_val <= wqi_value <= max_val:
                return classification

        return {'label': 'No Clasificado', 'color': '#666666', 'description': 'Calidad no determinada'}

    # Mantener métodos originales para compatibilidad con interfaz existente
    @staticmethod
    def _calculate_parameter_index(parameter, value, parameters_config):
        """Método de compatibilidad - redirige a las nuevas funciones"""
        if parameter == 'pH':
            return NSFWQICalculatorFixed.calculate_ph_subindex(value) / 100
        elif parameter == 'Oxigeno_Disuelto':
            return NSFWQICalculatorFixed.calculate_do_subindex(value) / 100
        elif parameter == 'DBO5':
            return NSFWQICalculatorFixed.calculate_bod5_subindex(value) / 100
        elif parameter == 'Coliformes_Fecales':
            return NSFWQICalculatorFixed.calculate_fc_subindex(value) / 100
        elif parameter == 'Temperatura':
            return NSFWQICalculatorFixed.calculate_temperature_subindex(value) / 100
        elif parameter == 'Fosforo_Total':
            return NSFWQICalculatorFixed.calculate_phosphorus_subindex(value) / 100
        elif parameter == 'Nitrato':
            return NSFWQICalculatorFixed.calculate_nitrate_subindex(value) / 100
        elif parameter == 'Turbiedad':
            return NSFWQICalculatorFixed.calculate_turbidity_subindex(value) / 100
        elif parameter == 'Solidos_Totales':
            return NSFWQICalculatorFixed.calculate_solids_subindex(value) / 100
        else:
            return 0.5  # Valor neutro

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
            try:
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
            except ImportError:
                # Si scipy no está disponible, análisis simple
                y = df_temp[wqi_column].values
                x = np.arange(len(y))
                slope = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else 0

                analysis['tendencia_general'] = {
                    'pendiente': float(slope),
                    'correlacion': 0.0,
                    'p_valor': 1.0,
                    'significativa': False,
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

            # Modelo simple de pronóstico basado en tendencia
            y = df_temp[wqi_column].values
            x = np.arange(len(y))

            # Calcular tendencia lineal simple
            if len(y) > 1:
                slope = np.polyfit(x, y, 1)[0]
                last_value = y[-1]
            else:
                slope = 0
                last_value = y[0] if len(y) > 0 else 50

            # Generar fechas futuras
            last_date = df_temp[date_column].iloc[-1]
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=periods
            )

            # Generar valores pronosticados
            forecast_values = []
            for i in range(periods):
                predicted_value = last_value + slope * (i + 1)
                # Limitar valores entre 0 y 100
                predicted_value = max(0, min(100, predicted_value))
                forecast_values.append(predicted_value)

            return {
                'fechas': forecast_dates.strftime('%Y-%m-%d').tolist(),
                'valores_pronosticados': forecast_values,
                'parametros': {
                    'pendiente': float(slope),
                    'ultimo_valor': float(last_value),
                    'metodo': 'tendencia_lineal'
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

            wqi_values = [r['wqi'] for r in results if 'wqi' in r and not pd.isna(r['wqi']) and 'error' not in r]

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
                    if 'classification' in result and 'error' not in result:
                        label = result['classification']['label']
                        quality_distribution[label] = quality_distribution.get(label, 0) + 1
            else:
                statistics = {}
                quality_distribution = {}

            self.progress_updated.emit(100)
            self.status_updated.emit("Cálculos completados")

            # Mostrar mapeo de parámetros en los resultados
            parameter_mapping = {}
            if results and 'parameter_mapping' in results[0]:
                parameter_mapping = results[0]['parameter_mapping']

            final_result = {
                'results': results,
                'statistics': statistics,
                'quality_distribution': quality_distribution,
                'parameter_mapping': parameter_mapping,
                'method': self.method,
                'total_samples': total_rows,
                'successful_calculations': len(wqi_values)
            }

            self.calculation_finished.emit(final_result)

        except Exception as e:
            self.error_occurred.emit(str(e))

# ==================== FUNCIONES DE UTILIDAD ====================

def test_wqi_with_ideam_data(csv_file_path):
    """Función de prueba específica para datos IDEAM"""

    print("=== PRUEBA WQI CON DATOS IDEAM ===")

    try:
        # Cargar datos
        data = pd.read_csv(csv_file_path)
        print(f"✅ Datos cargados: {len(data)} filas, {len(data.columns)} columnas")

        # Mostrar columnas disponibles
        print(f"📋 Columnas disponibles: {list(data.columns)}")

        # Mapear parámetros
        parameter_mapping = WQIParameterMapper.map_parameters(data.columns.tolist())
        print(f"\n🔗 Mapeo de parámetros encontrado:")
        for standard, actual in parameter_mapping.items():
            print(f"  {standard} → {actual}")

        if not parameter_mapping:
            print("❌ No se encontraron parámetros WQI mapeables")
            return

        # Tomar muestra para prueba
        sample_size = min(10, len(data))
        sample_data = data.head(sample_size)

        print(f"\n🧮 Calculando WQI para {sample_size} muestras...")

        # Calcular WQI para cada fila
        results = []
        for idx, row in sample_data.iterrows():
            result = WQICalculationEngine.calculate_nsf_wqi(row)
            results.append(result)

            if 'error' not in result:
                wqi = result['wqi']
                classification = WQICalculationEngine.classify_water_quality(wqi)
                print(f"  Muestra {idx}: WQI = {wqi:.2f} ({classification['label']})")
            else:
                print(f"  Muestra {idx}: Error - {result['error']}")

        # Estadísticas generales
        valid_wqi = [r['wqi'] for r in results if 'error' not in r]
        if valid_wqi:
            print(f"\n📊 Estadísticas (n={len(valid_wqi)}):")
            print(f"  Media: {np.mean(valid_wqi):.2f}")
            print(f"  Rango: {np.min(valid_wqi):.2f} - {np.max(valid_wqi):.2f}")
            print(f"  Desv. Std: {np.std(valid_wqi):.2f}")

        # Comparar con WQI existente si está disponible
        if 'WQI_NSF_9V' in data.columns:
            original_wqi = sample_data['WQI_NSF_9V'].dropna()
            if len(original_wqi) > 0 and len(valid_wqi) > 0:
                print(f"\n🔍 Comparación con WQI_NSF_9V original:")
                min_len = min(len(original_wqi), len(valid_wqi))
                for i in range(min_len):
                    orig = original_wqi.iloc[i]
                    calc = valid_wqi[i]
                    diff = abs(orig - calc)
                    print(f"  Muestra {i}: Original={orig:.2f}, Calculado={calc:.2f}, Diff={diff:.2f}")

        return results

    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {csv_file_path}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

def quick_wqi_test():
    """Prueba rápida con datos sintéticos"""

    print("=== PRUEBA RÁPIDA WQI ===")

    # Crear datos de prueba que imitan el formato IDEAM
    test_data = pd.DataFrame({
        'pH': [7.2, 6.8, 8.1, 7.5, 6.9],
        'DO': [8.5, 6.2, 9.1, 7.8, 8.0],
        'BOD5': [2.1, 4.5, 1.8, 3.2, 2.8],
        'FC': [12, 45, 8, 23, 18],
        'WT': [22, 18, 25, 20, 21],
        'TP': [0.08, 0.15, 0.05, 0.12, 0.09],
        'NO3': [2.5, 8.2, 1.8, 4.7, 3.1],
        'TBD': [3.2, 6.8, 2.1, 4.5, 3.8],
        'TS': [120, 180, 95, 140, 110]
    })

    print(f"📊 Datos de prueba creados: {len(test_data)} muestras")

    # Mostrar mapeo
    parameter_mapping = WQIParameterMapper.map_parameters(test_data.columns.tolist())
    print(f"\n🔗 Mapeo de parámetros:")
    for standard, actual in parameter_mapping.items():
        print(f"  {standard} → {actual}")

    # Calcular WQI
    print(f"\n🧮 Resultados NSF WQI:")
    print("Muestra | WQI   | Calidad     | Subíndices principales")
    print("--------|-------|-------------|----------------------")

    for idx, row in test_data.iterrows():
        result = WQICalculationEngine.calculate_nsf_wqi(row)

        if 'error' not in result:
            wqi = result['wqi']
            classification = WQICalculationEngine.classify_water_quality(wqi)

            # Mostrar algunos subíndices importantes
            subindices = result.get('subindices', {})
            key_params = ['pH', 'Oxigeno_Disuelto', 'DBO5']
            sub_str = ', '.join([f"{p}:{subindices.get(p, 0):.0f}" for p in key_params if p in subindices])

            print(f"   {idx:2d}   | {wqi:5.1f} | {classification['label']:11} | {sub_str}")
        else:
            print(f"   {idx:2d}   | ERROR | {result['error']}")

    return test_data

# ==================== FUNCIÓN PRINCIPAL ====================

def main():
    """Función principal para pruebas y demostración"""

    print("💧 CALCULADORA WQI - VERSIÓN CORREGIDA")
    print("=" * 50)

    # Prueba rápida
    quick_wqi_test()

    # Si existe el archivo IDEAM, probarlo también
    ideam_file = "TODOS_2000.csv"
    try:
        import os
        if os.path.exists(ideam_file):
            print("\n" + "=" * 50)
            test_wqi_with_ideam_data(ideam_file)
    except:
        print(f"\n📝 Para probar con datos IDEAM, coloca el archivo '{ideam_file}' en el directorio actual")

    print("\n✅ Pruebas completadas. El módulo está listo para usar con wqi_window.py")

if __name__ == "__main__":
    main()