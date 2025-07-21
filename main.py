import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget

# Importar las pantallas
from ui.cargar_datos import CargaDatos
from ui.menu_principal import MenuPrincipal

# Importar las demás pantallas con manejo de errores
try:
    from ui.preprocesamiento import Preprocesamiento
except ImportError:
    print("Advertencia: No se pudo importar Preprocesamiento")
    Preprocesamiento = None

try:
    from ui.analisis_bivariado import AnalisisBivariado
except ImportError:
    print("Advertencia: No se pudo importar AnalisisBivariado")
    AnalisisBivariado = None

try:
    from ui.machine_learning.segmentacion_ml import SegmentacionML
except ImportError:
    print("Advertencia: No se pudo importar SegmentacionML")
    SegmentacionML = None

try:
    from ui.deep_learning import DeepLearning
except ImportError:
    print("Advertencia: No se pudo importar DeepLearning")
    DeepLearning = None

# Importar la nueva ventana WQI
try:
    from ui.machine_learning.wqi_window import WQIWindow
except ImportError:
    print("Advertencia: No se pudo importar WQIWindow")
    WQIWindow = None

# Importar sistema de temas
try:
    from darkmode.theme_manager import ThemedWidget
except ImportError:
    print("Advertencia: No se pudo importar ThemedWidget")


    class ThemedWidget:
        def __init__(self):
            pass

        def apply_theme(self):
            pass

# Importar la pantalla de carga
try:
    from ui.pantalla_carga import PantallaCarga
except ImportError:
    print("Advertencia: No se pudo importar PantallaCarga")
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

        # Pantalla de carga de datos (siempre necesaria)
        self.pantalla_carga = CargaDatos()
        self.stack.addWidget(self.pantalla_carga)
        self.screen_indices['carga'] = self.current_index
        self.current_index += 1

        # Menú principal (siempre necesario)
        self.pantalla_menu = MenuPrincipal()
        self.stack.addWidget(self.pantalla_menu)
        self.screen_indices['menu'] = self.current_index
        self.current_index += 1

        # Pantallas opcionales
        if Preprocesamiento:
            self.pantalla_prepro = Preprocesamiento()
            self.stack.addWidget(self.pantalla_prepro)
            self.screen_indices['prepro'] = self.current_index
            self.current_index += 1
        else:
            self.pantalla_prepro = None

        if AnalisisBivariado:
            self.pantalla_bivariado = AnalisisBivariado()
            self.stack.addWidget(self.pantalla_bivariado)
            self.screen_indices['bivariado'] = self.current_index
            self.current_index += 1
        else:
            self.pantalla_bivariado = None

        if SegmentacionML:
            self.pantalla_ml = SegmentacionML()
            self.stack.addWidget(self.pantalla_ml)
            self.screen_indices['ml'] = self.current_index
            self.current_index += 1
        else:
            self.pantalla_ml = None

        if DeepLearning:
            self.pantalla_dl = DeepLearning()
            self.stack.addWidget(self.pantalla_dl)
            self.screen_indices['dl'] = self.current_index
            self.current_index += 1
        else:
            self.pantalla_dl = None

        if WQIWindow:
            self.pantalla_wqi = WQIWindow()
            self.stack.addWidget(self.pantalla_wqi)
            self.screen_indices['wqi'] = self.current_index
            self.current_index += 1
        else:
            self.pantalla_wqi = None

        # Configurar las conexiones de navegación
        self.setup_navigation()

        # Aplicar tema inicial
        try:
            self.apply_theme()
        except:
            pass

    def setup_navigation(self):
        """Configurar todas las conexiones de navegación con manejo de errores"""

        # De carga de datos → menú
        def ir_a_menu():
            try:
                if hasattr(self.pantalla_carga, 'df') and self.pantalla_carga.df is not None:
                    # Compartir datos con otras pantallas si existen
                    if hasattr(self.pantalla_menu, 'df'):
                        self.pantalla_menu.df = self.pantalla_carga.df

                    if self.pantalla_prepro and hasattr(self.pantalla_prepro, 'cargar_dataframe'):
                        self.pantalla_prepro.cargar_dataframe(self.pantalla_carga.df)

                    if self.pantalla_bivariado and hasattr(self.pantalla_bivariado, 'cargar_dataframe'):
                        self.pantalla_bivariado.cargar_dataframe(self.pantalla_carga.df)

                self.stack.setCurrentIndex(self.screen_indices['menu'])
            except Exception as e:
                print(f"Error al ir al menú: {e}")
                self.stack.setCurrentIndex(self.screen_indices['menu'])

        # Conectar botón de carga
        if hasattr(self.pantalla_carga, 'btn_cargar'):
            try:
                self.pantalla_carga.btn_cargar.clicked.connect(ir_a_menu)
            except Exception as e:
                print(f"Error conectando btn_cargar: {e}")

        # Conexiones desde el menú principal
        if hasattr(self.pantalla_menu, 'abrir_carga_datos'):
            self.pantalla_menu.abrir_carga_datos.connect(
                lambda: self.stack.setCurrentIndex(self.screen_indices['carga'])
            )

        # Conexión a preprocesamiento
        if self.pantalla_prepro and 'prepro' in self.screen_indices:
            if hasattr(self.pantalla_menu, 'btn_prepro'):
                try:
                    self.pantalla_menu.btn_prepro.clicked.connect(
                        lambda: self.stack.setCurrentIndex(self.screen_indices['prepro'])
                    )
                except Exception as e:
                    print(f"Error conectando btn_prepro: {e}")

        # Conexión a ML
        if self.pantalla_ml and 'ml' in self.screen_indices:
            if hasattr(self.pantalla_menu, 'abrir_machine_learning'):
                self.pantalla_menu.abrir_machine_learning.connect(
                    lambda: self.stack.setCurrentIndex(self.screen_indices['ml'])
                )

        # Conexión a DL
        if self.pantalla_dl and 'dl' in self.screen_indices:
            if hasattr(self.pantalla_menu, 'abrir_deep_learning'):
                self.pantalla_menu.abrir_deep_learning.connect(
                    lambda: self.stack.setCurrentIndex(self.screen_indices['dl'])
                )

        # Conexión a WQI
        if self.pantalla_wqi and 'wqi' in self.screen_indices:
            if hasattr(self.pantalla_menu, 'abrir_wqi'):
                self.pantalla_menu.abrir_wqi.connect(
                    lambda: self.stack.setCurrentIndex(self.screen_indices['wqi'])
                )

        # Navegación desde preprocesamiento
        if self.pantalla_prepro:
            # Cambiar a bivariado
            if hasattr(self.pantalla_prepro, 'cambiar_a_bivariado') and self.pantalla_bivariado:
                self.pantalla_prepro.cambiar_a_bivariado.connect(
                    lambda: self.stack.setCurrentIndex(
                        self.screen_indices.get('bivariado', self.screen_indices['menu']))
                )

            # Regresar al menú - manejo más robusto
            if hasattr(self.pantalla_prepro, 'btn_regresar'):
                try:
                    # Verificar que btn_regresar sea realmente un QPushButton
                    btn = getattr(self.pantalla_prepro, 'btn_regresar', None)
                    if btn and hasattr(btn, 'clicked'):
                        btn.clicked.connect(lambda: self.stack.setCurrentIndex(self.screen_indices['menu']))
                except Exception as e:
                    print(f"Error conectando btn_regresar de preprocesamiento: {e}")

        # Navegación desde análisis bivariado
        if self.pantalla_bivariado:
            if hasattr(self.pantalla_bivariado, 'btn_regresar'):
                try:
                    btn = getattr(self.pantalla_bivariado, 'btn_regresar', None)
                    if btn and hasattr(btn, 'clicked'):
                        btn.clicked.connect(lambda: self.stack.setCurrentIndex(self.screen_indices['menu']))
                except Exception as e:
                    print(f"Error conectando btn_regresar de bivariado: {e}")

        # Navegación desde ML
        if self.pantalla_ml:
            if hasattr(self.pantalla_ml, 'exit_button'):
                try:
                    self.pantalla_ml.exit_button.clicked.connect(
                        lambda: self.stack.setCurrentIndex(self.screen_indices['menu'])
                    )
                except Exception as e:
                    print(f"Error conectando exit_button de ML: {e}")

        # Navegación desde DL
        if self.pantalla_dl:
            if hasattr(self.pantalla_dl, 'btn_regresar'):
                try:
                    btn = getattr(self.pantalla_dl, 'btn_regresar', None)
                    if btn and hasattr(btn, 'clicked'):
                        btn.clicked.connect(lambda: self.stack.setCurrentIndex(self.screen_indices['menu']))
                except Exception as e:
                    print(f"Error conectando btn_regresar de DL: {e}")

        # Navegación desde WQI
        if self.pantalla_wqi:
            if hasattr(self.pantalla_wqi, 'regresar_menu'):
                self.pantalla_wqi.regresar_menu.connect(
                    lambda: self.stack.setCurrentIndex(self.screen_indices['menu'])
                )

    def mostrar_ventana_principal(self):
        """Mostrar la ventana principal después de la carga"""
        self.showMaximized()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Si existe pantalla de carga, usarla
    if PantallaCarga:
        # Crear y mostrar la pantalla de carga
        splash = PantallaCarga()
        splash.show()

        # Crear la ventana principal
        ventana = VentanaPrincipal()


        # Conectar la señal de carga completada
        def mostrar_app():
            splash.close()
            ventana.mostrar_ventana_principal()


        if hasattr(splash, 'carga_completada'):
            splash.carga_completada.connect(mostrar_app)
        else:
            # Si no hay señal, mostrar directamente después de un tiempo
            from PyQt5.QtCore import QTimer

            QTimer.singleShot(2000, mostrar_app)
    else:
        # Sin pantalla de carga, mostrar directamente
        ventana = VentanaPrincipal()
        ventana.mostrar_ventana_principal()

    sys.exit(app.exec_())