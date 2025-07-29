import sys
import os


# CONFIGURACIÓN DE PATHS ABSOLUTOS PARA EJECUTABLE
def setup_paths():
    """Configura los paths absolutos para el proyecto"""
    # Detectar si estamos en un ejecutable de PyInstaller
    if getattr(sys, 'frozen', False):
        # Si está empaquetado (PyInstaller)
        base_dir = sys._MEIPASS
        print(f"🔧 Modo ejecutable detectado. Base dir: {base_dir}")
    else:
        # Si está en desarrollo
        base_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"🔧 Modo desarrollo detectado. Base dir: {base_dir}")

    # Agregar el directorio base al path si no está ya
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
        print(f"✅ Agregado al path: {base_dir}")

    # Agregar subdirectorios específicos al path
    subdirs = ['ui', 'ml', 'darkmode', 'temp_graphs']
    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.exists(subdir_path) and subdir_path not in sys.path:
            sys.path.insert(0, subdir_path)
            print(f"✅ Agregado al path: {subdir}")
        elif os.path.exists(subdir_path):
            print(f"⚠️ Ya en path: {subdir}")
        else:
            print(f"❌ No encontrado: {subdir}")

    return base_dir


# Configurar paths ANTES de cualquier import
BASE_DIR = setup_paths()

# Ahora podemos hacer imports absolutos
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget

# Importar las pantallas principales (siempre requeridas)
try:
    from ui.cargar_datos import CargaDatos

    print("✅ CargaDatos importado correctamente")
except ImportError as e:
    print(f"❌ Error crítico importando CargaDatos: {e}")
    sys.exit(1)  # Sin carga de datos no puede funcionar

try:
    from ui.menu_principal import MenuPrincipal

    print("✅ MenuPrincipal importado correctamente")
except ImportError as e:
    print(f"❌ Error crítico importando MenuPrincipal: {e}")
    sys.exit(1)  # Sin menú principal no puede funcionar

# Importar las demás pantallas con manejo de errores (opcionales)
try:
    from ui.preprocesamiento import Preprocesamiento

    print("✅ Preprocesamiento importado correctamente")
except ImportError as e:
    print(f"❌ Error importando Preprocesamiento: {e}")
    Preprocesamiento = None

try:
    from ui.analisis_bivariado import AnalisisBivariado

    print("✅ AnalisisBivariado importado correctamente")
except ImportError as e:
    print(f"❌ Error importando AnalisisBivariado: {e}")
    AnalisisBivariado = None

try:
    from ui.machine_learning.segmentacion_ml import SegmentacionML

    print("✅ SegmentacionML importado correctamente")
except ImportError as e:
    print(f"❌ Error importando SegmentacionML: {e}")
    SegmentacionML = None

try:
    from ui.deep_learning import DeepLearningLightweight as DeepLearning

    print("✅ DeepLearning importado correctamente")
except ImportError as e:
    print(f"❌ Error importando DeepLearning: {e}")
    DeepLearning = None

# Importar la nueva ventana WQI
try:
    from ui.machine_learning.wqi_window import WQIWindow

    print("✅ WQIWindow importado correctamente")
except ImportError as e:
    print(f"❌ Error importando WQIWindow: {e}")
    WQIWindow = None

# Importar sistema de temas
try:
    # from darkmode.ui_theme_manager import ThemedWidget  # COMENTADA
    class ThemedWidget:
        def __init__(self):
            pass

        def apply_theme(self):
            pass
    print("✅ ThemedWidget importado correctamente")
except ImportError as e:
    print(f"❌ Error importando ThemedWidget: {e}")


    # Clase fallback para temas
    class ThemedWidget:
        def __init__(self):
            pass

        def apply_theme(self):
            pass

# Importar la pantalla de carga
try:
    from ui.pantalla_carga import PantallaCarga

    print("✅ PantallaCarga importado correctamente")
except ImportError as e:
    print(f"❌ Error importando PantallaCarga: {e}")
    PantallaCarga = None


class VentanaPrincipal(QMainWindow, ThemedWidget):
    """Ventana principal que hereda de ThemedWidget para soporte de temas"""

    def __init__(self):
        # Llamar a los constructores de ambas clases padre
        QMainWindow.__init__(self)
        ThemedWidget.__init__(self)

        self.setWindowTitle("Sistema de Análisis de Calidad del Agua")
        self.setGeometry(100, 100, 1200, 900)

        # Configurar el stack
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Crear instancias de pantallas con índices
        self.screen_indices = {}
        self.current_index = 0

        print("\n🔧 INICIALIZANDO PANTALLAS...")

        # Pantalla de carga de datos (siempre necesaria)
        try:
            self.pantalla_carga = CargaDatos()
            self.stack.addWidget(self.pantalla_carga)
            self.screen_indices['carga'] = self.current_index
            self.current_index += 1
            print(f"✅ Pantalla carga agregada en índice: {self.screen_indices['carga']}")
        except Exception as e:
            print(f"❌ Error crítico creando pantalla de carga: {e}")
            sys.exit(1)

        # Menú principal (siempre necesario)
        try:
            self.pantalla_menu = MenuPrincipal()
            self.stack.addWidget(self.pantalla_menu)
            self.screen_indices['menu'] = self.current_index
            self.current_index += 1
            print(f"✅ Pantalla menú agregada en índice: {self.screen_indices['menu']}")
        except Exception as e:
            print(f"❌ Error crítico creando menú principal: {e}")
            sys.exit(1)

        # Pantallas opcionales con manejo robusto de errores

        # Preprocesamiento
        if Preprocesamiento:
            try:
                self.pantalla_prepro = Preprocesamiento()
                self.stack.addWidget(self.pantalla_prepro)
                self.screen_indices['prepro'] = self.current_index
                self.current_index += 1
                print(f"✅ Pantalla preprocesamiento agregada en índice: {self.screen_indices['prepro']}")
            except Exception as e:
                print(f"❌ Error creando pantalla preprocesamiento: {e}")
                self.pantalla_prepro = None
        else:
            self.pantalla_prepro = None
            print("❌ Pantalla preprocesamiento no disponible (import falló)")

        # Análisis Bivariado
        if AnalisisBivariado:
            try:
                self.pantalla_bivariado = AnalisisBivariado()
                self.stack.addWidget(self.pantalla_bivariado)
                self.screen_indices['bivariado'] = self.current_index
                self.current_index += 1
                print(f"✅ Pantalla bivariado agregada en índice: {self.screen_indices['bivariado']}")
            except Exception as e:
                print(f"❌ Error creando pantalla bivariado: {e}")
                self.pantalla_bivariado = None
        else:
            self.pantalla_bivariado = None
            print("❌ Pantalla bivariado no disponible (import falló)")

        # Machine Learning
        if SegmentacionML:
            try:
                self.pantalla_ml = SegmentacionML()
                self.stack.addWidget(self.pantalla_ml)
                self.screen_indices['ml'] = self.current_index
                self.current_index += 1
                print(f"✅ Pantalla ML agregada en índice: {self.screen_indices['ml']}")
            except Exception as e:
                print(f"❌ Error creando pantalla ML: {e}")
                self.pantalla_ml = None
        else:
            self.pantalla_ml = None
            print("❌ Pantalla ML no disponible (import falló)")

        # Deep Learning
        if DeepLearning:
            try:
                self.pantalla_dl = DeepLearning()
                self.stack.addWidget(self.pantalla_dl)
                self.screen_indices['dl'] = self.current_index
                self.current_index += 1
                print(f"✅ Pantalla DL agregada en índice: {self.screen_indices['dl']}")
            except Exception as e:
                print(f"❌ Error creando pantalla DL: {e}")
                self.pantalla_dl = None
        else:
            self.pantalla_dl = None
            print("❌ Pantalla DL no disponible (import falló)")

        # WQI Window
        if WQIWindow:
            try:
                self.pantalla_wqi = WQIWindow()
                self.stack.addWidget(self.pantalla_wqi)
                self.screen_indices['wqi'] = self.current_index
                self.current_index += 1
                print(f"✅ Pantalla WQI agregada en índice: {self.screen_indices['wqi']}")
            except Exception as e:
                print(f"❌ Error creando pantalla WQI: {e}")
                self.pantalla_wqi = None
        else:
            self.pantalla_wqi = None
            print("❌ Pantalla WQI no disponible (import falló)")

        print(f"\n📋 RESUMEN DE PANTALLAS:")
        print(f"  - Pantallas disponibles: {list(self.screen_indices.keys())}")
        print(f"  - Total pantallas cargadas: {len(self.screen_indices)}")
        print(f"  - Preprocesamiento disponible: {'✅' if self.pantalla_prepro else '❌'}")
        print(f"  - Bivariado disponible: {'✅' if self.pantalla_bivariado else '❌'}")
        print(f"  - ML disponible: {'✅' if self.pantalla_ml else '❌'}")
        print(f"  - DL disponible: {'✅' if self.pantalla_dl else '❌'}")
        print(f"  - WQI disponible: {'✅' if self.pantalla_wqi else '❌'}")

        # Configurar las conexiones de navegación
        self.setup_navigation()

        # Debug de navegación
        self.debug_navigation_setup()

        # Aplicar tema inicial
        try:
            self.apply_theme()
            print("✅ Tema aplicado correctamente")
        except Exception as e:
            print(f"⚠️ Error aplicando tema: {e}")

    def setup_navigation(self):
        """Configurar todas las conexiones de navegación con manejo de errores"""
        print("\n🔗 CONFIGURANDO NAVEGACIÓN...")

        # De carga de datos → menú
        def ir_a_menu():
            try:
                if hasattr(self.pantalla_carga, 'df') and self.pantalla_carga.df is not None:
                    print("🔄 Transfiriendo datos a otras pantallas...")

                    # Compartir datos con menú principal si tiene la capacidad
                    if hasattr(self.pantalla_menu, 'df'):
                        self.pantalla_menu.df = self.pantalla_carga.df
                        print("✅ Datos transferidos a menú principal")

                    # Transferir a preprocesamiento
                    if self.pantalla_prepro and hasattr(self.pantalla_prepro, 'cargar_dataframe'):
                        self.pantalla_prepro.cargar_dataframe(self.pantalla_carga.df)
                        print("✅ Datos transferidos a preprocesamiento")

                    # Transferir a análisis bivariado
                    if self.pantalla_bivariado and hasattr(self.pantalla_bivariado, 'cargar_dataframe'):
                        self.pantalla_bivariado.cargar_dataframe(self.pantalla_carga.df)
                        print("✅ Datos transferidos a análisis bivariado")

                    # Transferir a Deep Learning
                    if self.pantalla_dl and hasattr(self.pantalla_dl, 'cargar_dataframe'):
                        print("🔄 Transfiriendo datos a Deep Learning...")
                        self.pantalla_dl.cargar_dataframe(self.pantalla_carga.df)
                        print("✅ Datos transferidos a Deep Learning")

                    # Transferir a Machine Learning
                    if self.pantalla_ml and hasattr(self.pantalla_ml, 'cargar_dataframe'):
                        self.pantalla_ml.cargar_dataframe(self.pantalla_carga.df)
                        print("✅ Datos transferidos a Machine Learning")

                    # Transferir a WQI Window
                    if self.pantalla_wqi and hasattr(self.pantalla_wqi, 'cargar_dataframe'):
                        self.pantalla_wqi.cargar_dataframe(self.pantalla_carga.df)
                        print("✅ Datos transferidos a WQI Window")

                self.stack.setCurrentIndex(self.screen_indices['menu'])
                print("✅ Navegación a menú completada")
            except Exception as e:
                print(f"❌ Error al ir al menú: {e}")
                # Intentar ir al menú de todos modos
                try:
                    self.stack.setCurrentIndex(self.screen_indices['menu'])
                except:
                    print("❌ Error crítico: No se puede navegar al menú")

        # Conectar botón de carga
        if hasattr(self.pantalla_carga, 'btn_cargar'):
            try:
                self.pantalla_carga.btn_cargar.clicked.connect(ir_a_menu)
                print("✅ Conectado btn_cargar de carga de datos")
            except Exception as e:
                print(f"❌ Error conectando btn_cargar: {e}")

        # CONEXIONES DESDE EL MENÚ PRINCIPAL
        print("🔗 Configurando conexiones desde menú principal...")

        # Función para navegar a preprocesamiento con debug
        def ir_a_preprocesamiento():
            print("🔗 FUNCIÓN ir_a_preprocesamiento() ejecutada")
            if self.pantalla_prepro and 'prepro' in self.screen_indices:
                print(f"  - Navegando a preprocesamiento (índice: {self.screen_indices['prepro']})")
                self.stack.setCurrentIndex(self.screen_indices['prepro'])
                print("  - ✅ Navegación completada")
            else:
                print("  - ❌ Pantalla preprocesamiento no disponible")
                print(f"  - pantalla_prepro existe: {self.pantalla_prepro is not None}")
                print(f"  - 'prepro' en screen_indices: {'prepro' in self.screen_indices}")

        try:
            # Verificar que el menú tiene las señales
            print(f"  - Verificando señales del menú:")
            attrs_to_check = [
                'abrir_preprocesamiento', 'abrir_carga_datos', 'abrir_machine_learning',
                'abrir_deep_learning', 'abrir_wqi'
            ]

            for attr in attrs_to_check:
                has_attr = hasattr(self.pantalla_menu, attr)
                print(f"    - {attr}: {has_attr}")

            # Conexión a carga de datos
            if hasattr(self.pantalla_menu, 'abrir_carga_datos'):
                self.pantalla_menu.abrir_carga_datos.connect(
                    lambda: self.safe_navigate('carga')
                )
                print("✅ Conectado abrir_carga_datos")

            # CONEXIÓN A PREPROCESAMIENTO - CON DEBUG DETALLADO
            if hasattr(self.pantalla_menu, 'abrir_preprocesamiento'):
                print("  - Conectando señal abrir_preprocesamiento...")
                self.pantalla_menu.abrir_preprocesamiento.connect(ir_a_preprocesamiento)
                print("  - ✅ Señal abrir_preprocesamiento conectada")
            else:
                print("  - ❌ Señal abrir_preprocesamiento NO EXISTE en pantalla_menu")

            # Conexión a ML
            if self.pantalla_ml and 'ml' in self.screen_indices:
                if hasattr(self.pantalla_menu, 'abrir_machine_learning'):
                    self.pantalla_menu.abrir_machine_learning.connect(
                        lambda: self.safe_navigate('ml')
                    )
                    print("✅ Conectado abrir_machine_learning")

            # Conexión a DL
            if self.pantalla_dl and 'dl' in self.screen_indices:
                if hasattr(self.pantalla_menu, 'abrir_deep_learning'):
                    self.pantalla_menu.abrir_deep_learning.connect(
                        lambda: self.safe_navigate('dl')
                    )
                    print("✅ Conectado abrir_deep_learning")

            # Conexión a WQI
            if self.pantalla_wqi and 'wqi' in self.screen_indices:
                if hasattr(self.pantalla_menu, 'abrir_wqi'):
                    self.pantalla_menu.abrir_wqi.connect(
                        lambda: self.safe_navigate('wqi')
                    )
                    print("✅ Conectado abrir_wqi")

        except Exception as e:
            print(f"❌ Error en conexiones del menú principal: {e}")

        # Configurar navegación de regreso para cada pantalla
        self.setup_return_navigation()

    def setup_return_navigation(self):
        """Configurar navegación de regreso para todas las pantallas"""
        print("🔗 Configurando navegación de regreso...")

        # Navegación desde preprocesamiento
        if self.pantalla_prepro:
            try:
                # Cambiar a bivariado
                if hasattr(self.pantalla_prepro, 'cambiar_a_bivariado') and self.pantalla_bivariado:
                    self.pantalla_prepro.cambiar_a_bivariado.connect(
                        lambda: self.safe_navigate('bivariado')
                    )
                    print("✅ Conectado cambiar_a_bivariado desde preprocesamiento")

                # btn_regresar es un QPushButton en preprocesamiento
                if hasattr(self.pantalla_prepro, 'btn_regresar'):
                    btn = getattr(self.pantalla_prepro, 'btn_regresar')
                    if hasattr(btn, 'clicked'):
                        btn.clicked.connect(lambda: self.safe_navigate('menu'))
                        print("✅ Conectado btn_regresar de preprocesamiento (QPushButton)")
                    else:
                        print("⚠️  btn_regresar de preprocesamiento no es un QPushButton")

            except Exception as e:
                print(f"❌ Error conectando navegación de preprocesamiento: {e}")

        # Navegación desde análisis bivariado
        if self.pantalla_bivariado:
            try:
                if hasattr(self.pantalla_bivariado, 'btn_regresar'):
                    btn = getattr(self.pantalla_bivariado, 'btn_regresar')
                    if hasattr(btn, 'clicked'):
                        btn.clicked.connect(lambda: self.safe_navigate('menu'))
                        print("✅ Conectado btn_regresar de bivariado (QPushButton)")
                    else:
                        print("⚠️  btn_regresar de bivariado no es un QPushButton")

            except Exception as e:
                print(f"❌ Error conectando navegación de bivariado: {e}")

        # Navegación desde ML
        if self.pantalla_ml:
            try:
                if hasattr(self.pantalla_ml, 'exit_button'):
                    self.pantalla_ml.exit_button.clicked.connect(
                        lambda: self.safe_navigate('menu')
                    )
                    print("✅ Conectado exit_button de ML")
            except Exception as e:
                print(f"❌ Error conectando exit_button de ML: {e}")

        # Navegación desde Deep Learning
        if self.pantalla_dl:
            try:
                if hasattr(self.pantalla_dl, 'btn_regresar'):
                    signal = getattr(self.pantalla_dl, 'btn_regresar')
                    if hasattr(signal, 'connect') and hasattr(signal, 'emit'):
                        # Es una señal, conectar directamente
                        signal.connect(lambda: self.safe_navigate('menu'))
                        print("✅ Conectado btn_regresar de Deep Learning (pyqtSignal)")
                    elif hasattr(signal, 'clicked'):
                        # Es un botón, usar .clicked
                        signal.clicked.connect(lambda: self.safe_navigate('menu'))
                        print("✅ Conectado btn_regresar de Deep Learning (QPushButton)")
                    else:
                        print(f"⚠️  btn_regresar de Deep Learning tipo desconocido: {type(signal)}")
                else:
                    print("⚠️  Deep Learning no tiene btn_regresar")

            except Exception as e:
                print(f"❌ Error conectando navegación de Deep Learning: {e}")

        # Navegación desde WQI
        if self.pantalla_wqi:
            try:
                if hasattr(self.pantalla_wqi, 'regresar_menu'):
                    self.pantalla_wqi.regresar_menu.connect(
                        lambda: self.safe_navigate('menu')
                    )
                    print("✅ Conectado regresar_menu de WQI")
            except Exception as e:
                print(f"❌ Error conectando navegación de WQI: {e}")

    def safe_navigate(self, screen_name):
        """Navegación segura a una pantalla específica con manejo de errores"""
        try:
            if screen_name in self.screen_indices:
                self.stack.setCurrentIndex(self.screen_indices[screen_name])
                print(f"✅ Navegado a: {screen_name}")
            else:
                print(f"❌ Pantalla no encontrada: {screen_name}")
                # Volver al menú por defecto
                if 'menu' in self.screen_indices:
                    self.stack.setCurrentIndex(self.screen_indices['menu'])
                    print("🔄 Regresando al menú por seguridad")
        except Exception as e:
            print(f"❌ Error navegando a {screen_name}: {e}")
            # Intentar volver al menú como último recurso
            try:
                if 'menu' in self.screen_indices:
                    self.stack.setCurrentIndex(self.screen_indices['menu'])
                    print("🔄 Regresando al menú por error")
            except Exception as fatal_error:
                print(f"❌ Error crítico de navegación: {fatal_error}")

    def debug_navigation_setup(self):
        """Debug de configuración de navegación"""
        print("\n🔍 DEBUG: Verificando configuración de navegación")
        print(f"  - Pantallas disponibles: {list(self.screen_indices.keys())}")
        print(f"  - Preprocesamiento disponible: {'✅' if self.pantalla_prepro else '❌'}")

        if self.pantalla_prepro:
            print(f"  - Tipo de pantalla_prepro: {type(self.pantalla_prepro)}")

        # Verificar señales del menú
        if hasattr(self.pantalla_menu, 'abrir_preprocesamiento'):
            signal = getattr(self.pantalla_menu, 'abrir_preprocesamiento')
            print(f"  - Señal abrir_preprocesamiento: {type(signal)}")
        else:
            print("  - ❌ Señal abrir_preprocesamiento no encontrada")

        print("🔍 Fin debug navegación\n")

    def go_to_screen(self, screen_name):
        """Método auxiliar para ir a una pantalla específica (compatible con código anterior)"""
        self.safe_navigate(screen_name)

    def mostrar_ventana_principal(self):
        """Mostrar la ventana principal después de la carga"""
        try:
            self.showMaximized()
            print("✅ Ventana principal mostrada en pantalla completa")
        except Exception as e:
            print(f"❌ Error mostrando ventana principal: {e}")
            # Intentar mostrar normal si falla el maximizado
            try:
                self.show()
                print("✅ Ventana principal mostrada en modo normal")
            except Exception as fallback_error:
                print(f"❌ Error crítico mostrando ventana: {fallback_error}")

    def closeEvent(self, event):
        """Manejar cierre de la aplicación con limpieza de recursos"""
        print("🔄 Cerrando aplicación...")
        try:
            # Limpiar recursos de Deep Learning si es necesario
            if self.pantalla_dl and hasattr(self.pantalla_dl, 'training_thread'):
                if self.pantalla_dl.training_thread and self.pantalla_dl.training_thread.isRunning():
                    print("🔄 Terminando hilo de entrenamiento de Deep Learning...")
                    self.pantalla_dl.training_thread.terminate()
                    self.pantalla_dl.training_thread.wait()
                    print("✅ Hilo de Deep Learning terminado")

            # Limpiar recursos de Machine Learning si es necesario
            if self.pantalla_ml and hasattr(self.pantalla_ml, 'cleanup'):
                print("🔄 Limpiando recursos de Machine Learning...")
                self.pantalla_ml.cleanup()

            # Limpiar tema si es necesario
            if hasattr(self, 'theme_manager'):
                print("🔄 Limpiando gestor de temas...")
                self.theme_manager.remove_observer(self)

            print("✅ Recursos limpiados correctamente")
            event.accept()
        except Exception as e:
            print(f"⚠️ Error durante limpieza de recursos: {e}")
            # Aceptar el cierre de todos modos
            event.accept()


def main():
    """Función principal con manejo robusto de errores"""
    print("=" * 60)
    print("🚀 INICIANDO SISTEMA DE ANÁLISIS DE CALIDAD DEL AGUA")
    print("=" * 60)
    print(f"📁 Directorio base: {BASE_DIR}")
    print(f"🐍 Python: {sys.version}")
    print(f"📊 PyQt5 disponible: {'✅' if 'PyQt5' in sys.modules else '❌'}")

    try:
        app = QApplication(sys.argv)
        app.setApplicationName("Sistema de Análisis de Calidad del Agua")
        app.setApplicationVersion("1.0.0")
        print("✅ Aplicación PyQt5 creada")

        # Si existe pantalla de carga, usarla
        if PantallaCarga:
            print("🔄 Iniciando con pantalla de carga...")

            # Crear y mostrar la pantalla de carga
            splash = PantallaCarga()
            splash.show()
            print("✅ Pantalla de carga mostrada")

            # Crear la ventana principal
            ventana = VentanaPrincipal()
            print("✅ Ventana principal creada")

            # Conectar la señal de carga completada
            def mostrar_app():
                try:
                    splash.close()
                    ventana.mostrar_ventana_principal()
                    print("✅ Transición de splash a ventana principal completada")
                except Exception as e:
                    print(f"❌ Error en transición: {e}")
                    # Mostrar ventana principal de todos modos
                    ventana.show()

            if hasattr(splash, 'carga_completada'):
                splash.carga_completada.connect(mostrar_app)
                print("✅ Señal de carga completada conectada")
            else:
                # Si no hay señal, mostrar directamente después de un tiempo
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(2000, mostrar_app)
                print("⚠️ Sin señal de carga, usando timer de 2 segundos")
        else:
            print("🔄 Iniciando sin pantalla de carga...")
            # Sin pantalla de carga, mostrar directamente
            ventana = VentanaPrincipal()
            ventana.mostrar_ventana_principal()
            print("✅ Ventana principal mostrada directamente")

        print("🚀 Aplicación iniciada correctamente")
        print("=" * 60)

        # Ejecutar la aplicación
        exit_code = app.exec_()
        print(f"✅ Aplicación cerrada con código: {exit_code}")
        return exit_code

    except Exception as e:
        print(f"❌ Error crítico en main(): {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)