"""
__init__.py - Inicialización del paquete Machine Learning
Facilita las importaciones entre módulos
"""

# Importaciones de los módulos principales
try:
    from .data_manager import DataManagerSingleton, get_data_manager, has_shared_data, get_shared_data
except ImportError:
    pass

try:
    from .ml_functions_supervisado import (
        regresion_lineal_simple,
        regresion_lineal_multiple,
        arbol_decision,
        random_forest,
        svm_modelo,
        comparar_modelos_supervisado,
        preparar_datos_supervisado_optimizado,
        generar_visualizaciones_ml,
        exportar_modelo,
        cargar_modelo
    )
except ImportError:
    pass

try:
    from .ml_functions_no_supervisado import (
        clustering_jerarquico_completo,
        kmeans_optimizado_completo,
        dbscan_optimizado,
        pca_completo_avanzado,
        analisis_exploratorio_completo,
        generar_datos_agua_realistas
    )
except ImportError:
    pass

try:
    from .supervisado_window import SupervisadoWindow
except ImportError:
    pass

try:
    from .no_supervisado_window import NoSupervisadoWindow
except ImportError:
    pass

try:
    from .segmentacion_ml import SegmentacionML
except ImportError:
    pass

# Definir qué se exporta cuando se hace "from machine_learning import *"
__all__ = [
    'DataManagerSingleton',
    'get_data_manager',
    'has_shared_data',
    'get_shared_data',
    'SupervisadoWindow',
    'NoSupervisadoWindow',
    'SegmentacionML',
    'regresion_lineal_simple',
    'regresion_lineal_multiple',
    'arbol_decision',
    'random_forest',
    'svm_modelo',
    'comparar_modelos_supervisado',
    'preparar_datos_supervisado_optimizado',
    'generar_visualizaciones_ml',
    'exportar_modelo',
    'cargar_modelo',
    'clustering_jerarquico_completo',
    'kmeans_optimizado_completo',
    'dbscan_optimizado',
    'pca_completo_avanzado',
    'analisis_exploratorio_completo',
    'generar_datos_agua_realistas'
]

# Versión del paquete
__version__ = '2.1.0'