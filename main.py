import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PyQt5.QtCore import QTimer

# Importar las pantallas
from ui.cargar_datos import CargaDatos
from ui.menu_principal import MenuPrincipal
from ui.preprocesamiento import Preprocesamiento
from ui.analisis_bivariado import AnalisisBivariado
from ui.segmentacion_ml import SegmentacionML
from ui.deep_learning import DeepLearning

# Importar sistema de temas
from darkmode import ThemeManager, ThemedWidget

# Importar la pantalla de carga
from ui.pantalla_carga import PantallaCarga


class VentanaPrincipal(QMainWindow, ThemedWidget):
    """Ventana principal que hereda de ThemedWidget para soporte de temas"""

    def __init__(self):
        # Llamar a los constructores de ambas clases padre
        QMainWindow.__init__(self)
        ThemedWidget.__init__(self)

        self.setWindowTitle("App ML con Qt")
        self.setGeometry(100, 100, 1000, 900)

        # Configurar el stack
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Instancias de pantallas (ahora todas heredan de ThemedWidget)
        self.pantalla_carga = CargaDatos()  # 0
        self.pantalla_menu = MenuPrincipal()  # 1
        self.pantalla_prepro = Preprocesamiento()  # 2
        self.pantalla_bivariado = AnalisisBivariado()  # 3
        self.pantalla_ml = SegmentacionML()  # 4
        self.pantalla_dl = DeepLearning()  # 5

        # Agregarlas al stack
        self.stack.addWidget(self.pantalla_carga)
        self.stack.addWidget(self.pantalla_menu)
        self.stack.addWidget(self.pantalla_prepro)
        self.stack.addWidget(self.pantalla_bivariado)
        self.stack.addWidget(self.pantalla_ml)
        self.stack.addWidget(self.pantalla_dl)

        # Configurar las conexiones de navegación
        self.setup_navigation()

        # Aplicar tema inicial
        self.apply_theme()

    def setup_navigation(self):
        """Configurar todas las conexiones de navegación"""

        # De carga de datos → menú
        def ir_a_menu():
            self.pantalla_menu.df = self.pantalla_carga.df
            self.pantalla_prepro.cargar_dataframe(self.pantalla_carga.df)
            self.pantalla_bivariado.cargar_dataframe(self.pantalla_carga.df)
            self.stack.setCurrentIndex(1)

        self.pantalla_carga.btn_cargar.clicked.connect(ir_a_menu)

        # Desde menú principal
        self.pantalla_menu.btn_prepro.clicked.connect(lambda: self.stack.setCurrentIndex(2))
        self.pantalla_prepro.cambiar_a_bivariado.connect(lambda: self.stack.setCurrentIndex(3))
        self.pantalla_menu.btn_ml.clicked.connect(lambda: self.stack.setCurrentIndex(4))
        self.pantalla_menu.btn_dl.clicked.connect(lambda: self.stack.setCurrentIndex(5))

        # Desde preprocesamiento → menú
        self.pantalla_prepro.btn_regresar.clicked.connect(lambda: self.stack.setCurrentIndex(1))

        # Desde análisis bivariado → menú
        self.pantalla_bivariado.btn_regresar.clicked.connect(lambda: self.stack.setCurrentIndex(1))

        # Desde ML → menú
        self.pantalla_ml.btn_regresar.clicked.connect(lambda: self.stack.setCurrentIndex(1))

        # Desde DL → menú
        self.pantalla_dl.btn_regresar.clicked.connect(lambda: self.stack.setCurrentIndex(1))

    def mostrar_ventana_principal(self):
        """Mostrar la ventana principal después de la carga"""
        self.showMaximized()

    # Las funciones apply_theme, apply_light_theme y apply_dark_theme
    # son heredadas automáticamente de ThemedWidget


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Crear y mostrar la pantalla de carga
    splash = PantallaCarga()
    splash.show()

    # Crear la ventana principal
    ventana = VentanaPrincipal()


    # Conectar la señal de carga completada
    def mostrar_app():
        splash.close()
        ventana.mostrar_ventana_principal()


    splash.carga_completada.connect(mostrar_app)

    sys.exit(app.exec_())