"""
data_cache.py - Sistema de cache para datos ML
Integrado con el sistema de temas existente
"""

import pickle
import os
import time
import json
from datetime import datetime, timedelta


class DataCache:
    """Sistema de cache simple para resultados ML"""

    def __init__(self, cache_dir=None, max_age_hours=24):
        if cache_dir is None:
            # Crear directorio cache en la raíz del proyecto
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            cache_dir = os.path.join(project_root, "cache")

        self.cache_dir = cache_dir
        self.max_age = timedelta(hours=max_age_hours)
        self.memory_cache = {}
        self.cache_info_file = os.path.join(cache_dir, "cache_info.json")

        # Crear directorio de cache si no existe
        if not os.path.exists(cache_dir):
            try:
                os.makedirs(cache_dir)
                print(f"📁 Directorio de cache creado: {cache_dir}")
            except Exception as e:
                print(f"⚠️ Error creando directorio cache: {e}")

    def _get_cache_path(self, key):
        """Obtener ruta del archivo de cache"""
        safe_key = "".join(c for c in key if c.isalnum() or c in "_-")
        return os.path.join(self.cache_dir, f"{safe_key}.cache")

    def _update_cache_info(self, key, action="access"):
        """Actualizar información del cache"""
        try:
            info = {}
            if os.path.exists(self.cache_info_file):
                with open(self.cache_info_file, 'r') as f:
                    info = json.load(f)

            if key not in info:
                info[key] = {}

            info[key]['last_' + action] = datetime.now().isoformat()
            if action == "create":
                info[key]['created'] = datetime.now().isoformat()
                info[key]['access_count'] = 0
            elif action == "access":
                info[key]['access_count'] = info[key].get('access_count', 0) + 1

            with open(self.cache_info_file, 'w') as f:
                json.dump(info, f, indent=2)

        except Exception as e:
            print(f"⚠️ Error actualizando info de cache: {e}")

    def get(self, key):
        """Obtener valor del cache"""
        # Primero intentar cache en memoria
        if key in self.memory_cache:
            self._update_cache_info(key, "access")
            return self.memory_cache[key]

        # Luego intentar cache en disco
        cache_path = self._get_cache_path(key)

        if os.path.exists(cache_path):
            try:
                # Verificar edad del archivo
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
                if datetime.now() - file_time < self.max_age:
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                        # Guardar en memoria para acceso rápido
                        self.memory_cache[key] = data
                        self._update_cache_info(key, "access")
                        return data
                else:
                    # Archivo muy antiguo, eliminarlo
                    os.remove(cache_path)
                    print(f"🗑️ Cache expirado eliminado: {key}")
            except Exception as e:
                print(f"⚠️ Error leyendo cache {key}: {e}")
                try:
                    os.remove(cache_path)
                except:
                    pass

        return None

    def set(self, key, value):
        """Guardar valor en cache"""
        # Guardar en memoria
        self.memory_cache[key] = value

        # Guardar en disco
        try:
            cache_path = self._get_cache_path(key)
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)

            self._update_cache_info(key, "create")
            print(f"💾 Cache guardado: {key}")

        except Exception as e:
            print(f"⚠️ Error guardando en cache {key}: {e}")

    def clear(self):
        """Limpiar todo el cache"""
        # Limpiar memoria
        memory_count = len(self.memory_cache)
        self.memory_cache.clear()

        # Limpiar archivos
        disk_count = 0
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.cache'):
                    os.remove(os.path.join(self.cache_dir, filename))
                    disk_count += 1

            # Limpiar archivo de info
            if os.path.exists(self.cache_info_file):
                os.remove(self.cache_info_file)

            print(f"🗑️ Cache limpiado: {memory_count} memoria + {disk_count} archivos")

        except Exception as e:
            print(f"⚠️ Error limpiando cache: {e}")

    def remove(self, key):
        """Eliminar una entrada específica del cache"""
        # Eliminar de memoria
        if key in self.memory_cache:
            del self.memory_cache[key]

        # Eliminar archivo
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                print(f"🗑️ Cache eliminado: {key}")
            except Exception as e:
                print(f"⚠️ Error eliminando cache {key}: {e}")

    def get_cache_info(self):
        """Obtener información detallada del cache"""
        memory_keys = len(self.memory_cache)
        memory_size = 0

        # Calcular tamaño aproximado en memoria
        try:
            import sys
            for key, value in self.memory_cache.items():
                memory_size += sys.getsizeof(value)
        except:
            memory_size = 0

        disk_files = 0
        total_size = 0
        try:
            if os.path.exists(self.cache_dir):
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.cache'):
                        disk_files += 1
                        file_path = os.path.join(self.cache_dir, filename)
                        total_size += os.path.getsize(file_path)
        except:
            pass

        # Leer información detallada
        detailed_info = {}
        try:
            if os.path.exists(self.cache_info_file):
                with open(self.cache_info_file, 'r') as f:
                    detailed_info = json.load(f)
        except:
            pass

        return {
            'memory_entries': memory_keys,
            'memory_size_mb': round(memory_size / (1024 * 1024), 2),
            'disk_files': disk_files,
            'disk_size_mb': round(total_size / (1024 * 1024), 2),
            'total_size_mb': round((memory_size + total_size) / (1024 * 1024), 2),
            'cache_dir': self.cache_dir,
            'max_age_hours': self.max_age.total_seconds() / 3600,
            'detailed_info': detailed_info
        }

    def print_cache_status(self):
        """Imprimir estado del cache"""
        info = self.get_cache_info()
        print(f"\n📊 Estado del Cache:")
        print(f"   💾 Memoria: {info['memory_entries']} entradas ({info['memory_size_mb']} MB)")
        print(f"   💿 Disco: {info['disk_files']} archivos ({info['disk_size_mb']} MB)")
        print(f"   📁 Total: {info['total_size_mb']} MB")
        print(f"   ⏰ Duración máxima: {info['max_age_hours']} horas")
        print(f"   📂 Directorio: {info['cache_dir']}")

        if info['detailed_info']:
            print(f"   📝 Entradas más usadas:")
            sorted_entries = sorted(
                info['detailed_info'].items(),
                key=lambda x: x[1].get('access_count', 0),
                reverse=True
            )[:3]
            for key, details in sorted_entries:
                count = details.get('access_count', 0)
                print(f"      • {key}: {count} accesos")


# Instancia global del cache para uso en toda la aplicación
_global_cache = None

def get_global_cache():
    """Obtener instancia global del cache"""
    global _global_cache
    if _global_cache is None:
        _global_cache = DataCache()
    return _global_cache