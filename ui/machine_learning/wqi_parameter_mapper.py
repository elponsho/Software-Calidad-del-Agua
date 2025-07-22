"""
wqi_parameter_mapper.py - Mapeo automático de nombres de parámetros para WQI
Permite usar diferentes nombres de columnas en los datos
"""


class WQIParameterMapper:
    """Clase para mapear nombres de parámetros de diferentes fuentes"""

    # Diccionario de mapeo: nombre_estandar -> posibles_nombres_en_datos
    PARAMETER_MAPPINGS = {
        'pH': ['pH', 'ph', 'PH'],
        'Oxigeno_Disuelto': ['DO', 'Oxigeno_Disuelto', 'DissolvedOxygen', 'O2', 'OD'],
        'DBO5': ['BOD5', 'DBO5', 'BOD', 'DBO'],
        'Coliformes_Fecales': ['FC', 'Coliformes_Fecales', 'FecalColiforms', 'CF', 'TC'],
        'Temperatura': ['WT', 'Temperatura', 'Temperature', 'Temp', 'T'],
        'Fosforo_Total': ['TP', 'Fosforo_Total', 'TotalPhosphorus', 'P_Total', 'PT'],
        'Nitrato': ['NO3', 'Nitrato', 'Nitrate', 'NO3-N'],
        'Turbiedad': ['TBD', 'Turbiedad', 'Turbidity', 'Turb', 'NTU'],
        'Solidos_Totales': ['TSS', 'Solidos_Totales', 'TotalSolids', 'TS', 'ST']
    }

    @classmethod
    def map_parameters(cls, data_columns):
        """
        Mapea las columnas disponibles en los datos a los nombres estándar

        Args:
            data_columns: Lista de nombres de columnas en el DataFrame

        Returns:
            dict: Diccionario {nombre_estandar: nombre_en_datos}
        """
        mapping = {}

        for standard_name, possible_names in cls.PARAMETER_MAPPINGS.items():
            for col in data_columns:
                if col in possible_names:
                    mapping[standard_name] = col
                    break

        return mapping

    @classmethod
    def get_available_parameters(cls, data_columns):
        """
        Obtiene los parámetros estándar disponibles en los datos

        Args:
            data_columns: Lista de nombres de columnas en el DataFrame

        Returns:
            list: Lista de nombres estándar de parámetros disponibles
        """
        mapping = cls.map_parameters(data_columns)
        return list(mapping.keys())

    @classmethod
    def prepare_data_row(cls, row, parameter_mapping):
        """
        Prepara una fila de datos mapeando los nombres de columnas

        Args:
            row: Fila del DataFrame
            parameter_mapping: Diccionario de mapeo

        Returns:
            dict: Fila con nombres estándar
        """
        mapped_row = {}

        for standard_name, data_name in parameter_mapping.items():
            if data_name in row.index:
                mapped_row[standard_name] = row[data_name]

        return mapped_row


# Actualización del WQICalculationEngine para usar el mapper
def calculate_wqi_with_mapping(data, method='NSF', custom_weights=None):
    """
    Calcula WQI con mapeo automático de parámetros

    Args:
        data: DataFrame con los datos
        method: Método de cálculo ('NSF', 'CCME', 'Weighted_Arithmetic')
        custom_weights: Pesos personalizados (opcional)

    Returns:
        list: Lista de resultados WQI para cada fila
    """
    from ui.machine_learning.wqi_calculator import WQICalculationEngine

    # Mapear parámetros
    parameter_mapping = WQIParameterMapper.map_parameters(data.columns.tolist())
    available_params = list(parameter_mapping.keys())

    # Filtrar parámetros NSF disponibles
    parameters = {}
    weights = {}

    for param in available_params:
        if param in WQICalculationEngine.NSF_PARAMETERS:
            parameters[param] = WQICalculationEngine.NSF_PARAMETERS[param]
            if custom_weights and param in custom_weights:
                weights[param] = custom_weights[param]
            else:
                weights[param] = WQICalculationEngine.NSF_PARAMETERS[param]['weight']

    # Calcular WQI para cada fila
    results = []

    for idx, row in data.iterrows():
        # Mapear la fila
        mapped_row = {}
        for standard_name, data_name in parameter_mapping.items():
            if data_name in row.index:
                mapped_row[standard_name] = row[data_name]

        # Calcular WQI según método
        if method == 'NSF':
            result = WQICalculationEngine.calculate_nsf_wqi(mapped_row, parameters, weights)
        elif method == 'CCME':
            result = WQICalculationEngine.calculate_ccme_wqi(mapped_row, parameters)
        else:
            result = WQICalculationEngine.calculate_weighted_arithmetic_wqi(mapped_row, parameters, weights)

        result['index'] = idx
        result['classification'] = WQICalculationEngine.classify_water_quality(result.get('wqi', 0))
        results.append(result)

    return results, parameter_mapping