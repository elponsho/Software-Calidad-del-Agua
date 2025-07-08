"""
__init__.py - Archivo de inicialización para el módulo machine_learning
"""

# Hacer que la carpeta machine_learning sea un paquete Python
__version__ = "1.0.0"
__author__ = "Control de Calidad del Agua"

# Importaciones opcionales para facilitar el acceso
try:
    from .ml_functions_supervisado import (
        regresion_multiple_proceso,
        svm_proceso,
        random_forest_proceso,
        regresion_lineal_proceso,
        generar_datos_agua_optimizado
    )
    ML_FUNCTIONS_AVAILABLE = True
except ImportError:
    ML_FUNCTIONS_AVAILABLE = False

# Hacer disponibles las funciones a nivel del paquete
__all__ = [
    'regresion_multiple_proceso',
    'svm_proceso',
    'random_forest_proceso',
    'regresion_lineal_proceso',
    'generar_datos_agua_optimizado',
    'ML_FUNCTIONS_AVAILABLE'
]