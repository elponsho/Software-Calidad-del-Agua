"""
data_manager.py - Gestor centralizado de datos para el sistema
Implementa el patrón Singleton para compartir datos entre módulos
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import pickle
import os
import traceback
from PyQt5.QtCore import pyqtSignal, QObject


class DataSignals(QObject):
    """Señales para notificar cambios en los datos"""
    data_changed = pyqtSignal()
    data_cleared = pyqtSignal()
    data_modified = pyqtSignal(str)  # Tipo de modificación

class DataManagerSingleton:
    """
    Singleton para gestión centralizada de datos
    Permite compartir datos entre diferentes módulos del sistema
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataManagerSingleton, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.data: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}
        self.observers: List[Any] = []
        self.history: List[Dict[str, Any]] = []
        self.current_file: Optional[str] = None
        self.data_modifications: List[Dict[str, Any]] = []
        self.signals = DataSignals()  # Añadir señales

    def _notify_observers(self, event: str) -> None:
        """Notificar a todos los observadores - VERSIÓN CORREGIDA"""
        try:
            print(f"DataManager: Notificando evento '{event}' a {len(self.observers)} observadores")

            # 1. Emitir señales Qt
            if event == 'data_changed':
                self.signals.data_changed.emit()
            elif event == 'data_cleared':
                self.signals.data_cleared.emit()
            elif event == 'data_modified':
                self.signals.data_modified.emit('data_modified')

            # 2. Notificar a observadores del patrón Observer tradicional
            for i, observer in enumerate(self.observers):
                try:
                    print(f"  Notificando observador {i}: {type(observer).__name__}")

                    # Llamar al método 'update' del observador
                    if hasattr(observer, 'update'):
                        observer.update(event)
                        print(f"    ✅ Notificado exitosamente")
                    else:
                        print(f"    ⚠️ Observador {type(observer).__name__} no tiene método 'update'")

                except Exception as e:
                    print(f"    ❌ Error al notificar observador {type(observer).__name__}: {e}")
                    traceback.print_exc()

        except Exception as e:
            print(f"Error general en _notify_observers: {e}")
            traceback.print_exc()

    # ==================== GESTIÓN DE DATOS ====================

    def set_data(self, data: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None,
                 source: str = "unknown") -> None:
        """
        Establecer nuevos datos en el sistema

        Args:
            data: DataFrame con los datos
            metadata: Metadatos opcionales
            source: Fuente de los datos (archivo, api, generado, etc.)
        """
        print(f"DataManager: Estableciendo datos desde '{source}'")
        print(f"  Shape: {data.shape}")
        print(f"  Columnas: {list(data.columns)}")

        self.data = data.copy()  # Copiar para evitar modificaciones externas

        # Actualizar metadata
        self.metadata = {
            'source': source,
            'loaded_at': datetime.now().isoformat(),
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
            'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(data.select_dtypes(include=['object', 'category']).columns),
            'missing_values': data.isnull().sum().to_dict(),
            'description': data.describe().to_dict() if not data.empty else {}
        }

        if metadata:
            self.metadata.update(metadata)

        # Añadir a historial
        self.history.append({
            'timestamp': datetime.now(),
            'action': 'data_loaded',
            'source': source,
            'shape': data.shape
        })

        # Limpiar modificaciones anteriores
        self.data_modifications = []

        print(f"DataManager: Datos establecidos, notificando cambios...")
        # Notificar a observadores
        self._notify_observers('data_changed')

    def get_data(self) -> Optional[pd.DataFrame]:
        """
        Obtener los datos actuales

        Returns:
            DataFrame actual o None si no hay datos
        """
        if self.data is not None:
            return self.data.copy()  # Retornar copia para evitar modificaciones
        return None

    def has_data(self) -> bool:
        """Verificar si hay datos cargados"""
        return self.data is not None and not self.data.empty

    def clear_data(self) -> None:
        """Limpiar todos los datos"""
        print("DataManager: Limpiando datos...")

        self.data = None
        self.metadata = {}
        self.current_file = None
        self.data_modifications = []

        self.history.append({
            'timestamp': datetime.now(),
            'action': 'data_cleared'
        })

        self._notify_observers('data_cleared')

    # ==================== OPERACIONES CON DATOS ====================

    def filter_data(self, condition: str) -> pd.DataFrame:
        """
        Filtrar datos según condición

        Args:
            condition: Condición de filtrado (e.g., "pH > 7")

        Returns:
            DataFrame filtrado
        """
        if not self.has_data():
            raise ValueError("No hay datos cargados")

        try:
            filtered = self.data.query(condition)

            self.data_modifications.append({
                'timestamp': datetime.now(),
                'action': 'filter',
                'condition': condition,
                'original_shape': self.data.shape,
                'filtered_shape': filtered.shape
            })

            return filtered
        except Exception as e:
            raise ValueError(f"Error al filtrar datos: {str(e)}")

    def get_column_stats(self, column: str) -> Dict[str, Any]:
        """
        Obtener estadísticas de una columna

        Args:
            column: Nombre de la columna

        Returns:
            Diccionario con estadísticas
        """
        if not self.has_data():
            raise ValueError("No hay datos cargados")

        if column not in self.data.columns:
            raise ValueError(f"Columna '{column}' no encontrada")

        col_data = self.data[column]

        stats = {
            'name': column,
            'dtype': str(col_data.dtype),
            'count': len(col_data),
            'missing': col_data.isnull().sum(),
            'missing_pct': (col_data.isnull().sum() / len(col_data)) * 100,
            'unique': col_data.nunique(),
            'memory_usage': col_data.memory_usage(deep=True) / 1024  # KB
        }

        if col_data.dtype in [np.float64, np.float32, np.int64, np.int32]:
            stats.update({
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'q25': float(col_data.quantile(0.25)),
                'median': float(col_data.median()),
                'q75': float(col_data.quantile(0.75)),
                'max': float(col_data.max()),
                'skewness': float(col_data.skew()),
                'kurtosis': float(col_data.kurtosis())
            })
        elif col_data.dtype == 'object' or col_data.dtype.name == 'category':
            value_counts = col_data.value_counts()
            stats.update({
                'top_values': value_counts.head(10).to_dict(),
                'mode': col_data.mode()[0] if not col_data.mode().empty else None
            })

        return stats

    def add_column(self, name: str, data: pd.Series) -> None:
        """
        Añadir nueva columna a los datos

        Args:
            name: Nombre de la columna
            data: Serie con los datos
        """
        if not self.has_data():
            raise ValueError("No hay datos cargados")

        if len(data) != len(self.data):
            raise ValueError("La longitud de los datos no coincide")

        self.data[name] = data

        self.data_modifications.append({
            'timestamp': datetime.now(),
            'action': 'add_column',
            'column_name': name,
            'dtype': str(data.dtype)
        })

        # Actualizar metadata
        self.metadata['columns'] = list(self.data.columns)
        if data.dtype in [np.float64, np.float32, np.int64, np.int32]:
            self.metadata['numeric_columns'].append(name)
        else:
            self.metadata['categorical_columns'].append(name)

        self._notify_observers('data_modified')

    def remove_column(self, name: str) -> None:
        """
        Eliminar columna de los datos

        Args:
            name: Nombre de la columna
        """
        if not self.has_data():
            raise ValueError("No hay datos cargados")

        if name not in self.data.columns:
            raise ValueError(f"Columna '{name}' no encontrada")

        self.data = self.data.drop(columns=[name])

        self.data_modifications.append({
            'timestamp': datetime.now(),
            'action': 'remove_column',
            'column_name': name
        })

        # Actualizar metadata
        self.metadata['columns'] = list(self.data.columns)
        if name in self.metadata.get('numeric_columns', []):
            self.metadata['numeric_columns'].remove(name)
        if name in self.metadata.get('categorical_columns', []):
            self.metadata['categorical_columns'].remove(name)

        self._notify_observers('data_modified')

    # ==================== GESTIÓN DE ARCHIVOS ====================

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Cargar datos desde archivo

        Args:
            filepath: Ruta del archivo

        Returns:
            DataFrame cargado
        """
        ext = os.path.splitext(filepath)[1].lower()

        try:
            if ext == '.csv':
                data = pd.read_csv(filepath)
            elif ext in ['.xlsx', '.xls']:
                data = pd.read_excel(filepath)
            elif ext == '.json':
                data = pd.read_json(filepath)
            elif ext == '.pkl':
                data = pd.read_pickle(filepath)
            else:
                raise ValueError(f"Formato de archivo no soportado: {ext}")

            self.current_file = filepath
            self.set_data(data, source=filepath)

            return data

        except Exception as e:
            raise ValueError(f"Error al cargar archivo: {str(e)}")

    def save_data(self, filepath: str) -> None:
        """
        Guardar datos en archivo

        Args:
            filepath: Ruta del archivo
        """
        if not self.has_data():
            raise ValueError("No hay datos para guardar")

        ext = os.path.splitext(filepath)[1].lower()

        try:
            if ext == '.csv':
                self.data.to_csv(filepath, index=False)
            elif ext == '.xlsx':
                self.data.to_excel(filepath, index=False)
            elif ext == '.json':
                self.data.to_json(filepath, orient='records', indent=2)
            elif ext == '.pkl':
                self.data.to_pickle(filepath)
            else:
                raise ValueError(f"Formato de archivo no soportado: {ext}")

            self.history.append({
                'timestamp': datetime.now(),
                'action': 'data_saved',
                'filepath': filepath
            })

        except Exception as e:
            raise ValueError(f"Error al guardar archivo: {str(e)}")

    # ==================== DATOS DE EJEMPLO ====================

    def generate_demo_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generar datos de demostración para calidad del agua

        Args:
            n_samples: Número de muestras a generar

        Returns:
            DataFrame con datos de ejemplo
        """
        np.random.seed(42)

        # Generar datos sintéticos realistas
        data = {
            'Estacion': [f"EST_{str(i + 1).zfill(3)}" for i in np.random.choice(50, n_samples)],
            'Fecha': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
            'pH': np.round(np.random.normal(7.2, 0.8, n_samples), 2),
            'Temperatura': np.round(np.random.normal(22, 5, n_samples), 1),
            'Oxigeno_Disuelto': np.round(np.random.normal(8.5, 1.5, n_samples), 2),
            'Turbidez': np.round(np.random.exponential(2.0, n_samples), 2),
            'Conductividad': np.round(np.random.normal(250, 80, n_samples), 1),
            'DBO5': np.round(np.random.exponential(3, n_samples), 2),
            'Coliformes_Totales': np.random.poisson(50, n_samples),
            'Nitratos': np.round(np.random.exponential(2, n_samples), 2),
            'Fosforo_Total': np.round(np.random.exponential(0.1, n_samples), 3),
            'Solidos_Suspendidos': np.round(np.random.exponential(10, n_samples), 1)
        }

        # Crear DataFrame
        df = pd.DataFrame(data)

        # Añadir algunas correlaciones realistas
        df.loc[df['Temperatura'] > 27, 'Oxigeno_Disuelto'] *= 0.8
        df.loc[df['pH'] < 6.5, 'Turbidez'] *= 1.3
        df.loc[df['Turbidez'] > 5, 'Solidos_Suspendidos'] *= 1.5

        # Calcular WQI simplificado
        df['WQI'] = self._calculate_wqi(df)

        # Clasificación de calidad
        df['Calidad'] = pd.cut(
            df['WQI'],
            bins=[0, 40, 60, 80, 100],
            labels=['Mala', 'Regular', 'Buena', 'Excelente']
        )

        # Añadir algunos valores faltantes aleatorios
        missing_cols = ['Nitratos', 'Fosforo_Total', 'Coliformes_Totales']
        for col in missing_cols:
            missing_idx = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
            df.loc[missing_idx, col] = np.nan

        self.set_data(df, source='demo_data')

        return df

    def _calculate_wqi(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcular índice de calidad del agua simplificado

        Args:
            df: DataFrame con parámetros

        Returns:
            Serie con valores WQI
        """
        # Pesos para cada parámetro (simplificado)
        weights = {
            'pH': 0.2,
            'Oxigeno_Disuelto': 0.25,
            'Turbidez': 0.15,
            'DBO5': 0.2,
            'Coliformes_Totales': 0.2
        }

        # Normalizar cada parámetro a escala 0-100
        scores = pd.DataFrame()

        # pH (óptimo en 7.0)
        scores['pH'] = 100 * np.exp(-0.5 * ((df['pH'] - 7.0) / 1.5) ** 2)

        # Oxígeno disuelto (mayor es mejor)
        scores['Oxigeno_Disuelto'] = np.clip((df['Oxigeno_Disuelto'] / 10) * 100, 0, 100)

        # Turbidez (menor es mejor)
        scores['Turbidez'] = np.clip(100 - (df['Turbidez'] * 10), 0, 100)

        # DBO5 (menor es mejor)
        scores['DBO5'] = np.clip(100 - (df['DBO5'] * 15), 0, 100)

        # Coliformes (menor es mejor)
        scores['Coliformes_Totales'] = np.clip(100 - (df['Coliformes_Totales'] / 10), 0, 100)

        # Calcular WQI ponderado
        wqi = pd.Series(0, index=df.index)
        for param, weight in weights.items():
            wqi += scores[param] * weight

        return np.round(wqi, 1)

    # ==================== PATRÓN OBSERVER ====================

    def add_observer(self, observer: Any):
        """Añadir observador al sistema"""
        if observer not in self.observers:
            self.observers.append(observer)
            print(f"DataManager: Observador {type(observer).__name__} añadido. Total: {len(self.observers)}")

    def remove_observer(self, observer: Any):
        """Remover observador del sistema"""
        if observer in self.observers:
            self.observers.remove(observer)
            print(f"DataManager: Observador {type(observer).__name__} removido. Total: {len(self.observers)}")

    # ==================== INFORMACIÓN Y ESTADO ====================

    def get_info(self) -> Dict[str, Any]:
        """
        Obtener información completa del estado actual

        Returns:
            Diccionario con información
        """
        info = {
            'has_data': self.has_data(),
            'metadata': self.metadata,
            'current_file': self.current_file,
            'modifications_count': len(self.data_modifications),
            'observers_count': len(self.observers),
            'history_count': len(self.history)
        }

        if self.has_data():
            info.update({
                'shape': self.data.shape,
                'columns': list(self.data.columns),
                'memory_usage_mb': self.data.memory_usage(deep=True).sum() / (1024 * 1024)
            })

        return info

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener historial de acciones

        Args:
            limit: Número máximo de entradas

        Returns:
            Lista con historial
        """
        return self.history[-limit:][::-1]  # Más recientes primero

    def export_session(self, filepath: str) -> None:
        """
        Exportar sesión completa (datos + metadata + historial)

        Args:
            filepath: Ruta del archivo
        """
        session_data = {
            'data': self.data.to_dict() if self.has_data() else None,
            'metadata': self.metadata,
            'history': self.history,
            'modifications': self.data_modifications,
            'exported_at': datetime.now().isoformat()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, default=str)

    def import_session(self, filepath: str) -> None:
        """
        Importar sesión desde archivo

        Args:
            filepath: Ruta del archivo
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            session_data = json.load(f)

        if session_data.get('data'):
            self.data = pd.DataFrame(session_data['data'])

        self.metadata = session_data.get('metadata', {})
        self.history = session_data.get('history', [])
        self.data_modifications = session_data.get('modifications', [])

        self._notify_observers('session_imported')


# ==================== FUNCIONES DE UTILIDAD ====================

def get_data_manager() -> DataManagerSingleton:
    """
    Obtener instancia del data manager

    Returns:
        Instancia única del DataManagerSingleton
    """
    return DataManagerSingleton()


def has_shared_data() -> bool:
    """
    Verificar si hay datos compartidos disponibles

    Returns:
        True si hay datos, False en caso contrario
    """
    dm = get_data_manager()
    return dm.has_data()


def get_shared_data() -> Optional[pd.DataFrame]:
    """
    Obtener datos compartidos

    Returns:
        DataFrame o None
    """
    dm = get_data_manager()
    return dm.get_data()


def set_shared_data(data: pd.DataFrame, source: str = "unknown") -> None:
    """
    Establecer datos compartidos

    Args:
        data: DataFrame a compartir
        source: Fuente de los datos
    """
    dm = get_data_manager()
    dm.set_data(data, source=source)