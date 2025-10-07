# ==================== CONFIGURACIÓN PARA PYINSTALLER ====================
# DEBE IR AL PRINCIPIO DE main.py, ANTES DE CUALQUIER OTRA IMPORTACIÓN
import sys
import os
import warnings


# 1. Fix crítico para PyInstaller y distutils
def fix_pyinstaller_compatibility():
    """Resolver problemas específicos de PyInstaller"""

    # Fix para distutils que causa el error que viste
    if 'distutils' in sys.modules:
        del sys.modules['distutils']

    # Configurar variables de entorno antes de importar librerías ML
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    # Suprimir warnings problemáticos
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', message='.*distutils.*')
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    # Configurar path para PyInstaller
    if hasattr(sys, '_MEIPASS'):
        bundle_dir = sys._MEIPASS
        print(f"🔧 PyInstaller bundle detectado: {bundle_dir}")
    else:
        bundle_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"🔧 Modo desarrollo detectado: {bundle_dir}")

    # Añadir directorios del proyecto al path
    project_dirs = ['ui', 'ui/machine_learning', 'ml', 'data', 'temp_graphs', 'darkmode']
    for dirname in project_dirs:
        full_path = os.path.join(bundle_dir, dirname)
        if os.path.exists(full_path) and full_path not in sys.path:
            sys.path.insert(0, full_path)
            print(f"✅ Agregado al path PyInstaller: {dirname}")

    return bundle_dir


# Ejecutar fix ANTES de cualquier importación ML
bundle_dir = fix_pyinstaller_compatibility()

# 2. Configurar matplotlib ANTES de importar PyQt5
try:
    import matplotlib

    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt

    plt.ioff()
    print("✅ Matplotlib configurado para PyInstaller")
except ImportError as e:
    print(f"⚠️ Matplotlib no disponible: {e}")

# ==================== IMPORTACIONES ORIGINALES MEJORADAS ====================

try:
    import import_patch

    import_patch.initialize_ml_environment()
    print("✅ Import patch inicializado")
except Exception as e:
    print(f"Warning - Import patch: {e}")

# Al inicio de main.py, después de los imports básicos
try:
    # Intentar parche del data_manager antes de importar otras UI
    exec(open('data_manager_patch.py').read())
    print("✅ Data manager patch aplicado")
except FileNotFoundError:
    print("⚠️ data_manager_patch.py no encontrado")
except Exception as e:
    print(f"⚠️ Error en data_manager_patch: {e}")


# CONFIGURACIÓN DE PATHS ABSOLUTOS PARA EJECUTABLE (MEJORADA)
def setup_paths():
    """Configura los paths absolutos para el proyecto - versión PyInstaller"""
    # Detectar si estamos en un ejecutable de PyInstaller
    if getattr(sys, 'frozen', False):
        # Si está empaquetado (PyInstaller)
        base_dir = sys._MEIPASS
        is_executable = True
        print(f"🔧 Modo ejecutable PyInstaller detectado. Base dir: {base_dir}")
    else:
        # Si está en desarrollo
        base_dir = os.path.dirname(os.path.abspath(__file__))
        is_executable = False
        print(f"🔧 Modo desarrollo detectado. Base dir: {base_dir}")

    # Agregar el directorio base al path si no está ya
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
        print(f"✅ Agregado al path: {base_dir}")

    # Agregar subdirectorios específicos al path
    subdirs = ['ui', 'ui/machine_learning', 'ml', 'data', 'temp_graphs', 'darkmode']
    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.exists(subdir_path) and subdir_path not in sys.path:
            sys.path.insert(0, subdir_path)
            print(f"✅ Agregado al path: {subdir}")
        elif os.path.exists(subdir_path):
            print(f"⚠️ Ya en path: {subdir}")
        else:
            print(f"❌ No encontrado: {subdir}")

    return base_dir, is_executable


# Configurar paths ANTES de cualquier import
BASE_DIR, IS_EXECUTABLE = setup_paths()


# FUNCIÓN PARA IMPORTACIONES SEGURAS (MEJORADA)
def safe_import(module_name, fallback_action=None, required=False):
    """Importar módulos de manera segura con manejo de errores para PyInstaller"""
    try:
        if '.' in module_name:
            # Para imports con puntos (ej: ui.menu_principal)
            parts = module_name.split('.')
            module = __import__(module_name, fromlist=[parts[-1]])
        else:
            module = __import__(module_name)

        print(f"✅ {module_name} importado correctamente")
        return module
    except ImportError as e:
        if required:
            print(f"❌ Error crítico importando {module_name}: {e}")
            if fallback_action:
                return fallback_action()
            sys.exit(1)
        else:
            print(f"❌ Error importando {module_name}: {e}")
            if fallback_action:
                return fallback_action()
            return None
    except Exception as e:
        print(f"❌ Error inesperado importando {module_name}: {e}")
        if required:
            sys.exit(1)
        return None


# VERIFICAR DEPENDENCIAS CRÍTICAS (MEJORADA)
def check_critical_dependencies():
    """Verificar que las dependencias críticas estén disponibles para PyInstaller"""
    missing_critical = []

    # Verificar PyQt5
    try:
        import PyQt5
        print("✅ PyQt5 disponible")
    except ImportError:
        missing_critical.append('PyQt5')

    # Verificar pandas (si es necesario)
    try:
        import pandas
        print("✅ Pandas disponible")
    except ImportError:
        print("⚠️ Pandas no disponible - algunas funciones pueden fallar")

    # Verificar numpy (si es necesario)
    try:
        import numpy
        print("✅ Numpy disponible")
    except ImportError:
        print("⚠️ Numpy no disponible - algunas funciones pueden fallar")

    # Verificar sklearn para ML
    try:
        import sklearn
        print(f"✅ Sklearn disponible (versión: {sklearn.__version__})")
    except ImportError:
        print("⚠️ Sklearn no disponible - funcionalidad ML limitada")

    if missing_critical:
        print(f"❌ Dependencias críticas faltantes: {missing_critical}")
        sys.exit(1)

    return True


# Verificar dependencias antes de continuar
check_critical_dependencies()

# Ahora podemos hacer imports absolutos
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QMessageBox

# IMPORTACIONES CON MANEJO SEGURO PARA PYINSTALLER

# Importar DataManager (si existe)
try:
    from data.data_manager import DataManager

    print("✅ DataManager importado correctamente")
    DATA_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error cargando DataManager: {e}")
    DataManager = None
    DATA_MANAGER_AVAILABLE = False

# Verificar scipy (necesario para ML)
try:
    import scipy

    print("✅ Scipy disponible")
    SCIPY_AVAILABLE = True
except ImportError:
    print("❌ Error cargando scipy - funcionalidad ML limitada")
    SCIPY_AVAILABLE = False

# Verificar ML modules - CORREGIR RUTAS
ml_supervisado = safe_import('ui.machine_learning.ml_functions_supervisado',
                             lambda: print("✅ Módulo Supervisado disponible"))
ml_no_supervisado = safe_import('ui.machine_learning.ml_functions_no_supervisado',
                                lambda: print("✅ Módulo No Supervisado disponible"))

# Importar las pantallas principales (siempre requeridas)
CargaDatos = safe_import('ui.cargar_datos', required=True)
MenuPrincipal = safe_import('ui.menu_principal', required=True)

# Extraer clases específicas
if MenuPrincipal:
    try:
        from ui.menu_principal import MenuPrincipal as MenuPrincipalClass

        # Usar la misma clase para ambas
        MenuPrincipalConNavegacion = MenuPrincipalClass
        print("✅ MenuPrincipal clases importadas correctamente")
    except ImportError as e:
        print(f"❌ Error importando clases de MenuPrincipal: {e}")
        sys.exit(1)

if CargaDatos:
    try:
        from ui.cargar_datos import CargaDatos as CargaDatosClass

        print("✅ CargaDatos clase importada correctamente")
    except ImportError as e:
        print(f"❌ Error importando clase CargaDatos: {e}")
        sys.exit(1)

# Importar las demás pantallas con manejo de errores (opcionales)
Preprocesamiento = safe_import('ui.preprocesamiento')
AnalisisBivariado = safe_import('ui.analisis_bivariado')

# Importar segmentación ML compatible
SegmentacionML = safe_import('ui.machine_learning.segmentacion_ml')
if not SegmentacionML:
    # Fallback a la versión original
    SegmentacionML = safe_import('ui.machine_learning.segmentacion_ml')

# Importar Deep Learning desde deep_learning.py
DeepLearning = safe_import('ui.deep_learning')
if not DeepLearning:
    # Fallback si está en el directorio principal
    DeepLearning = safe_import('deep_learning')
print(f"ℹ️ DeepLearning estado: {'✅ Disponible' if DeepLearning else '❌ No disponible'}")

# Importar la ventana WQI
WQIWindow = safe_import('ui.machine_learning.wqi_window')

# Importar la pantalla de carga
PantallaCarga = safe_import('ui.pantalla_carga')


# FUNCIONES PARA VERIFICAR DEPENDENCIAS DE DEEP LEARNING (MEJORADA)
def check_deep_learning_dependencies():
    """Verificar que todas las dependencias de Deep Learning estén disponibles para PyInstaller"""
    missing_deps = []

    # Verificar dependencias básicas
    deps = ['numpy', 'pandas', 'matplotlib']
    for dep in deps:
        try:
            __import__(dep)
            print(f"✅ {dep} disponible para Deep Learning")
        except ImportError:
            missing_deps.append(dep)
            print(f"❌ {dep} no disponible")

    # Verificar dependencias opcionales de ML
    optional_deps = ['sklearn', 'scipy', 'seaborn']
    for dep in optional_deps:
        try:
            __import__(dep)
            print(f"✅ {dep} disponible (opcional)")
        except ImportError:
            print(f"⚠️ {dep} no disponible (opcional)")

    if missing_deps:
        print(f"❌ Dependencias faltantes para Deep Learning: {missing_deps}")
        return False, missing_deps
    else:
        print("✅ Todas las dependencias básicas de Deep Learning están disponibles")
        return True, []


class VentanaPrincipal(QMainWindow):
    """Ventana principal de la aplicación - Compatible con PyInstaller"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sistema de Análisis de Calidad del Agua v1.0")
        self.setGeometry(100, 100, 1200, 900)

        # Configurar el stack
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Crear instancias de pantallas con índices
        self.screen_indices = {}
        self.current_index = 0

        # Inicializar DataManager si está disponible
        if DATA_MANAGER_AVAILABLE:
            self.data_manager = DataManager()
            print("✅ DataManager inicializado")
        else:
            self.data_manager = None
            print("⚠️ DataManager no disponible")

        print("\n🔧 INICIALIZANDO PANTALLAS (PyInstaller compatible)...")

        # Pantalla de carga de datos (siempre necesaria)
        try:
            self.pantalla_carga = CargaDatosClass()

            # Conectar con DataManager si está disponible
            if self.data_manager and hasattr(self.pantalla_carga, 'data_manager'):
                self.pantalla_carga.data_manager = self.data_manager
                print("✅ DataManager conectado con CargaDatos")

            self.stack.addWidget(self.pantalla_carga)
            self.screen_indices['carga'] = self.current_index
            self.current_index += 1
            print(f"✅ Pantalla carga agregada en índice: {self.screen_indices['carga']}")
        except Exception as e:
            print(f"❌ Error crítico creando pantalla de carga: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # MENÚ PRINCIPAL CON NAVEGACIÓN
        try:
            self.pantalla_menu = MenuPrincipalClass()

            # Configurar DataManager en el menú si está disponible
            if self.data_manager and hasattr(self.pantalla_menu, 'data_manager'):
                self.pantalla_menu.data_manager = self.data_manager
                print("✅ DataManager conectado con MenuPrincipal")

            self.stack.addWidget(self.pantalla_menu)
            self.screen_indices['menu'] = self.current_index
            self.current_index += 1
            print(f"✅ Pantalla menú agregada en índice: {self.screen_indices['menu']}")

            # CONECTAR NAVEGACIÓN DEL MENÚ
            print("🔗 Conectando navegación del menú...")
            self.pantalla_menu.abrir_preprocesamiento.connect(lambda: self.safe_navigate('prepro'))
            self.pantalla_menu.abrir_carga_datos.connect(lambda: self.safe_navigate('carga'))
            self.pantalla_menu.abrir_machine_learning.connect(lambda: self.safe_navigate('ml'))
            self.pantalla_menu.abrir_wqi.connect(lambda: self.safe_navigate('wqi'))
            self.pantalla_menu.abrir_deep_learning.connect(self.manejar_deep_learning)
            print("✅ Navegación del menú conectada")

        except Exception as e:
            print(f"❌ Error crítico creando menú principal: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # Pantallas opcionales con manejo robusto de errores para PyInstaller

        # Preprocesamiento
        if Preprocesamiento:
            try:
                if hasattr(Preprocesamiento, 'Preprocesamiento'):
                    self.pantalla_prepro = Preprocesamiento.Preprocesamiento()
                else:
                    self.pantalla_prepro = Preprocesamiento()

                # Conectar DataManager si está disponible
                if self.data_manager and hasattr(self.pantalla_prepro, 'data_manager'):
                    self.pantalla_prepro.data_manager = self.data_manager

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
                if hasattr(AnalisisBivariado, 'AnalisisBivariado'):
                    self.pantalla_bivariado = AnalisisBivariado.AnalisisBivariado()
                else:
                    self.pantalla_bivariado = AnalisisBivariado()

                # Conectar DataManager si está disponible
                if self.data_manager and hasattr(self.pantalla_bivariado, 'data_manager'):
                    self.pantalla_bivariado.data_manager = self.data_manager

                self.stack.addWidget(self.pantalla_bivariado)
                self.screen_indices['bivariado'] = self.current_index
                self.current_index += 1
                print(f"✅ Pantalla bivariado agregada en índice: {self.screen_indices['bivariado']}")
            except Exception as e:
                print(f"❌ Error creando pantalla bivariado: {e}")
                import traceback
                traceback.print_exc()
                self.pantalla_bivariado = None
        else:
            self.pantalla_bivariado = None
            print("❌ Pantalla bivariado no disponible (import falló)")

        # Machine Learning
        if SegmentacionML:
            try:
                if hasattr(SegmentacionML, 'SegmentacionML'):
                    self.pantalla_ml = SegmentacionML.SegmentacionML()
                else:
                    self.pantalla_ml = SegmentacionML()

                # Conectar DataManager si está disponible
                if self.data_manager and hasattr(self.pantalla_ml, 'data_manager'):
                    self.pantalla_ml.data_manager = self.data_manager

                self.stack.addWidget(self.pantalla_ml)
                self.screen_indices['ml'] = self.current_index
                self.current_index += 1
                print(f"✅ Pantalla ML agregada en índice: {self.screen_indices['ml']}")
            except Exception as e:
                print(f"❌ Error creando pantalla ML: {e}")
                import traceback
                traceback.print_exc()
                self.pantalla_ml = None
        else:
            self.pantalla_ml = None
            print("❌ Pantalla ML no disponible (import falló)")

        # WQI Window
        if WQIWindow:
            try:
                if hasattr(WQIWindow, 'WQIWindow'):
                    self.pantalla_wqi = WQIWindow.WQIWindow()
                else:
                    self.pantalla_wqi = WQIWindow()

                # Conectar DataManager si está disponible
                if self.data_manager and hasattr(self.pantalla_wqi, 'data_manager'):
                    self.pantalla_wqi.data_manager = self.data_manager

                self.stack.addWidget(self.pantalla_wqi)
                self.screen_indices['wqi'] = self.current_index
                self.current_index += 1
                print(f"✅ Pantalla WQI agregada en índice: {self.screen_indices['wqi']}")
            except Exception as e:
                print(f"❌ Error creando pantalla WQI: {e}")
                import traceback
                traceback.print_exc()
                self.pantalla_wqi = None
        else:
            self.pantalla_wqi = None
            print("❌ Pantalla WQI no disponible (import falló)")

        print(f"\n📋 RESUMEN DE PANTALLAS (PyInstaller):")
        print(f"  - Pantallas disponibles: {list(self.screen_indices.keys())}")
        print(f"  - Total pantallas cargadas: {len(self.screen_indices)}")
        print(f"  - Preprocesamiento disponible: {'✅' if self.pantalla_prepro else '❌'}")
        print(f"  - Bivariado disponible: {'✅' if self.pantalla_bivariado else '❌'}")
        print(f"  - ML disponible: {'✅' if self.pantalla_ml else '❌'}")
        print(f"  - DL disponible: {'✅' if DeepLearning else '❌'}")
        print(f"  - WQI disponible: {'✅' if self.pantalla_wqi else '❌'}")
        print(f"  - Ejecutable PyInstaller: {'✅' if IS_EXECUTABLE else '❌'}")

        # Configurar las conexiones de navegación
        self.setup_navigation()

    def manejar_deep_learning(self):
        """Manejar la navegación a Deep Learning - Compatible con PyInstaller"""
        print("🧠 Intentando abrir Deep Learning (PyInstaller)...")
        try:
            # Verificar si ya existe
            if 'deep_learning' in self.screen_indices:
                self.safe_navigate('deep_learning')
                return

            # Crear dinámicamente usando tu archivo
            deep_learning_widget = self.abrir_deep_learning_seguro()
            if deep_learning_widget:
                # Conectar navegación de regreso
                if hasattr(deep_learning_widget, 'btn_regresar'):
                    deep_learning_widget.btn_regresar.connect(lambda: self.safe_navigate('menu'))
                    print("✅ Navegación de regreso conectada")

                # Conectar DataManager si está disponible
                if self.data_manager and hasattr(deep_learning_widget, 'data_manager'):
                    deep_learning_widget.data_manager = self.data_manager
                    print("✅ DataManager conectado a Deep Learning")

                # Agregar al stack
                self.stack.addWidget(deep_learning_widget)
                new_index = self.stack.count() - 1
                self.screen_indices['deep_learning'] = new_index

                # Navegar
                self.stack.setCurrentIndex(new_index)
                print("✅ Deep Learning cargado y navegado exitosamente")
            else:
                print("❌ No se pudo cargar Deep Learning")
                QMessageBox.critical(self, "Error",
                                     "No se pudo cargar el módulo de Deep Learning.\n"
                                     "Verifique que el archivo deep_learning.py esté disponible.")

        except Exception as e:
            print(f"❌ Error manejando Deep Learning: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error de Deep Learning",
                                 f"Error al cargar Deep Learning:\n{str(e)}")

    def abrir_deep_learning_seguro(self):
        """Abrir Deep Learning usando deep_learning.py - Compatible con PyInstaller"""
        print("🧠 Cargando Deep Learning desde deep_learning.py (PyInstaller)...")

        # Verificar dependencias básicas para PyInstaller
        try:
            import numpy
            import pandas
            import matplotlib
            print("✅ Dependencias básicas disponibles en PyInstaller")
        except ImportError as e:
            QMessageBox.critical(self, "Dependencias Faltantes",
                                 f"Faltan dependencias básicas para Deep Learning:\n{e}\n\n"
                                 "El ejecutable no incluye las librerías necesarias.")
            return None

        try:
            # Usar tu archivo deep_learning.py directamente
            if DeepLearning and hasattr(DeepLearning, 'DeepLearningLightweight'):
                deep_learning_widget = DeepLearning.DeepLearningLightweight()
                print("✅ Deep Learning importado desde deep_learning.py (PyInstaller)")
                return deep_learning_widget
            else:
                print("❌ Clase DeepLearningLightweight no encontrada en deep_learning.py")
                return None

        except Exception as e:
            print(f"❌ Error cargando Deep Learning en PyInstaller: {e}")
            import traceback
            traceback.print_exc()
            return None

    def setup_navigation(self):
        """Configurar todas las conexiones de navegación con manejo de errores para PyInstaller"""
        print("\n🔗 CONFIGURANDO NAVEGACIÓN (PyInstaller)...")

        # De carga de datos → menú
        def ir_a_menu():
            try:
                # Transferir datos usando DataManager si está disponible
                if self.data_manager and hasattr(self.pantalla_carga, 'df') and self.pantalla_carga.df is not None:
                    print("📊 Datos registrados en DataManager: {}".format(self.pantalla_carga.df.shape))
                    self.data_manager.set_data(self.pantalla_carga.df, source='carga_manual')
                    print("📡 Observadores notificados")

                # Método tradicional de transferencia (fallback)
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
                import traceback
                traceback.print_exc()
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

        # Configurar navegación de regreso para pantallas existentes
        self.setup_return_navigation()

    def setup_return_navigation(self):
        """Configurar navegación de regreso para todas las pantallas - PyInstaller"""
        print("🔗 Configurando navegación de regreso (PyInstaller)...")

        # Navegación desde preprocesamiento
        if self.pantalla_prepro:
            try:
                if hasattr(self.pantalla_prepro, 'cambiar_a_bivariado') and self.pantalla_bivariado:
                    self.pantalla_prepro.cambiar_a_bivariado.connect(
                        lambda: self.safe_navigate('bivariado')
                    )
                    print("✅ Conectado cambiar_a_bivariado desde preprocesamiento")

                if hasattr(self.pantalla_prepro, 'btn_regresar'):
                    btn = getattr(self.pantalla_prepro, 'btn_regresar')
                    if hasattr(btn, 'clicked'):
                        btn.clicked.connect(lambda: self.safe_navigate('menu'))
                        print("✅ Conectado btn_regresar de preprocesamiento")

            except Exception as e:
                print(f"❌ Error conectando navegación de preprocesamiento: {e}")

        # Navegación desde análisis bivariado
        if self.pantalla_bivariado:
            try:
                if hasattr(self.pantalla_bivariado, 'btn_regresar'):
                    btn = getattr(self.pantalla_bivariado, 'btn_regresar')
                    if hasattr(btn, 'clicked'):
                        btn.clicked.connect(lambda: self.safe_navigate('menu'))
                        print("✅ Conectado btn_regresar de bivariado")

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
        """Navegación segura a una pantalla específica con manejo de errores para PyInstaller"""
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

    def go_to_screen(self, screen_name):
        """Método auxiliar para ir a una pantalla específica"""
        self.safe_navigate(screen_name)

    def mostrar_ventana_principal(self):
        """Mostrar la ventana principal después de la carga - PyInstaller compatible"""
        try:
            self.showMaximized()
            print("✅ Ventana principal mostrada en pantalla completa (PyInstaller)")
        except Exception as e:
            print(f"❌ Error mostrando ventana principal: {e}")
            try:
                self.show()
                print("✅ Ventana principal mostrada en modo normal (PyInstaller)")
            except Exception as fallback_error:
                print(f"❌ Error crítico mostrando ventana: {fallback_error}")

    def closeEvent(self, event):
        """Manejar cierre de la aplicación con limpieza de recursos - PyInstaller compatible"""
        print("🔄 Cerrando aplicación (PyInstaller)...")
        try:
            # Limpiar recursos de Deep Learning si es necesario
            if hasattr(self, 'deep_learning_widget') and self.deep_learning_widget:
                if hasattr(self.deep_learning_widget, 'training_thread'):
                    if self.deep_learning_widget.training_thread and self.deep_learning_widget.training_thread.isRunning():
                        print("🔄 Terminando hilo de entrenamiento de Deep Learning...")
                        self.deep_learning_widget.training_thread.terminate()
                        self.deep_learning_widget.training_thread.wait()
                        print("✅ Hilo de Deep Learning terminado")

            # Limpiar recursos de Machine Learning si es necesario
            if self.pantalla_ml and hasattr(self.pantalla_ml, 'cleanup'):
                print("🔄 Limpiando recursos de Machine Learning...")
                self.pantalla_ml.cleanup()

            # Limpiar DataManager si es necesario
            if self.data_manager and hasattr(self.data_manager, 'clear_data'):
                print("🔄 Limpiando DataManager...")
                self.data_manager.clear_data()

            # Limpiar matplotlib si está disponible
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
                print("✅ Gráficos matplotlib cerrados")
            except:
                pass

            print("✅ Recursos limpiados correctamente (PyInstaller)")
            event.accept()
        except Exception as e:
            print(f"⚠️ Error durante limpieza de recursos: {e}")
            # Aceptar el cierre de todos modos
            event.accept()


def verificar_entorno():
    """Verificar el entorno de ejecución y mostrar información de debug - PyInstaller"""
    print(f"🔧 Modo ejecutable PyInstaller: {'✅' if IS_EXECUTABLE else '❌'}")
    print(f"📁 Directorio base: {BASE_DIR}")
    print(f"📁 Bundle dir: {bundle_dir}")
    print(f"🐍 Python: {sys.version}")
    print(f"📊 PyQt5 disponible: {'✅' if 'PyQt5.QtWidgets' in sys.modules else '❌'}")
    print(f"📊 DataManager disponible: {'✅' if DATA_MANAGER_AVAILABLE else '❌'}")
    print(f"🧮 Scipy disponible: {'✅' if SCIPY_AVAILABLE else '❌'}")

    # Verificar imports críticos
    critical_modules = [
        'ui.cargar_datos',
        'ui.menu_principal',
    ]

    for module in critical_modules:
        try:
            __import__(module)
            print(f"✅ {module} verificado")
        except ImportError as e:
            print(f"❌ {module} fallo: {e}")

    # Verificar imports opcionales
    optional_modules = [
        'ui.preprocesamiento',
        'ui.analisis_bivariado',
        'ui.machine_learning.segmentacion_ml',
        'ui.machine_learning.wqi_window',
    ]

    for module in optional_modules:
        try:
            __import__(module)
            print(f"✅ {module} disponible")
        except ImportError:
            print(f"⚠️ {module} no disponible")

    # Información específica de PyInstaller
    if IS_EXECUTABLE:
        print(f"📦 PyInstaller _MEIPASS: {sys._MEIPASS}")
        print(f"📦 Executable path: {sys.executable}")
        print(f"📦 Frozen: {getattr(sys, 'frozen', False)}")


def test_sklearn_import():
    """Test de sklearn compatible con PyInstaller"""
    try:
        import sklearn
        print(f"✅ sklearn version: {sklearn.__version__}")

        # Test imports críticos
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        print("✅ Imports básicos de sklearn funcionan (PyInstaller)")

        # Test extensiones Cython (puede fallar en PyInstaller)
        try:
            from sklearn.tree._tree import Tree
            print("✅ Extensiones Cython básicas funcionan (PyInstaller)")
        except ImportError as e:
            print(f"⚠️ Extensiones Cython no disponibles en PyInstaller: {e}")

        return True
    except Exception as e:
        print(f"❌ Error en sklearn (PyInstaller): {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Función principal con manejo robusto de errores - Compatible con PyInstaller"""
    print("=" * 60)
    print("🚀 INICIANDO SISTEMA DE ANÁLISIS DE CALIDAD DEL AGUA")
    print("🔧 VERSION COMPATIBLE CON PYINSTALLER")
    print("=" * 60)

    # Verificar entorno
    verificar_entorno()

    try:
        app = QApplication(sys.argv)
        app.setApplicationName("Sistema de Análisis de Calidad del Agua")
        app.setApplicationVersion("1.0.0 PyInstaller")
        print("✅ Aplicación PyQt5 creada (PyInstaller compatible)")

        # Configurar estilo para PyInstaller
        try:
            app.setStyle('Fusion')  # Estilo que funciona bien en PyInstaller
            print("✅ Estilo Fusion aplicado")
        except Exception as e:
            print(f"⚠️ No se pudo aplicar estilo: {e}")

        # Si existe pantalla de carga, usarla
        if PantallaCarga:
            print("🔄 Iniciando con pantalla de carga (PyInstaller)...")

            # Crear y mostrar la pantalla de carga
            try:
                if hasattr(PantallaCarga, 'PantallaCarga'):
                    splash = PantallaCarga.PantallaCarga()
                else:
                    splash = PantallaCarga()
                splash.show()
                print("✅ Pantalla de carga mostrada (PyInstaller)")
            except Exception as e:
                print(f"❌ Error creando pantalla de carga: {e}")
                splash = None

            # Crear la ventana principal
            ventana = VentanaPrincipal()
            print("✅ Ventana principal creada (PyInstaller)")

            # Conectar la señal de carga completada
            def mostrar_app():
                try:
                    if splash:
                        splash.close()
                    ventana.mostrar_ventana_principal()
                    print("✅ Transición de splash a ventana principal completada (PyInstaller)")
                except Exception as e:
                    print(f"❌ Error en transición: {e}")
                    # Mostrar ventana principal de todos modos
                    ventana.show()

            if splash and hasattr(splash, 'carga_completada'):
                splash.carga_completada.connect(mostrar_app)
                print("✅ Señal de carga completada conectada")
            else:
                # Si no hay señal, mostrar directamente después de un tiempo
                try:
                    from PyQt5.QtCore import QTimer
                    QTimer.singleShot(2000, mostrar_app)
                    print("⚠️ Sin señal de carga, usando timer de 2 segundos")
                except ImportError:
                    # Si no hay QTimer, mostrar inmediatamente
                    mostrar_app()
                    print("⚠️ Sin QTimer, mostrando inmediatamente")
        else:
            print("🔄 Iniciando sin pantalla de carga (PyInstaller)...")
            # Sin pantalla de carga, mostrar directamente
            ventana = VentanaPrincipal()
            ventana.mostrar_ventana_principal()
            print("✅ Ventana principal mostrada directamente (PyInstaller)")

        print("🚀 Aplicación iniciada correctamente (PyInstaller)")
        print("=" * 60)

        # Ejecutar la aplicación
        exit_code = app.exec_()
        print(f"✅ Aplicación cerrada con código: {exit_code}")
        return exit_code

    except Exception as e:
        print(f"❌ Error crítico en main() (PyInstaller): {e}")
        import traceback
        traceback.print_exc()

        # Intentar mostrar un mensaje de error al usuario si es posible
        try:
            if 'app' in locals():
                error_dialog = QMessageBox()
                error_dialog.setIcon(QMessageBox.Critical)
                error_dialog.setWindowTitle("Error Crítico")
                error_dialog.setText(f"Error crítico en la aplicación:\n\n{str(e)}")
                error_dialog.setInformativeText("La aplicación se cerrará. Revise el log para más detalles.")
                error_dialog.exec_()
        except:
            # Si ni siquiera se puede mostrar el diálogo, simplemente continuar
            pass

        return 1


# ==================== PUNTO DE ENTRADA PARA PYINSTALLER ====================
if __name__ == "__main__":
    # Test de sklearn antes de iniciar la aplicación
    print("🧪 Verificando compatibilidad sklearn...")
    sklearn_ok = test_sklearn_import()

    if not sklearn_ok and IS_EXECUTABLE:
        print("⚠️ sklearn tiene problemas en PyInstaller, pero continuando...")

    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🔄 Aplicación interrumpida por el usuario")
        sys.exit(0)
    except Exception as fatal_error:
        print(f"❌ Error fatal no manejado (PyInstaller): {fatal_error}")
        import traceback

        traceback.print_exc()

        # En PyInstaller, intentar mostrar error en archivo log
        if IS_EXECUTABLE:
            try:
                log_file = os.path.join(os.path.dirname(sys.executable), 'error_log.txt')
                with open(log_file, 'w') as f:
                    f.write(f"Error fatal: {fatal_error}\n")
                    f.write(traceback.format_exc())
                print(f"📄 Error guardado en: {log_file}")
            except:
                pass

        sys.exit(1)