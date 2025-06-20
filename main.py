import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget

from ui.cargar_datos import CargaDatos
from ui.menu_principal import MenuPrincipal
from ui.preprocesamiento import Preprocesamiento
from ui.analisis_bivariado import AnalisisBivariado
from ui.segmentacion_ml import SegmentacionML
from ui.deep_learning import DeepLearning

class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("App ML con Qt")
        self.setGeometry(100, 100, 1000, 900)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Instancias de pantallas
        self.pantalla_carga = CargaDatos()               # 0
        self.pantalla_menu = MenuPrincipal()             # 1
        self.pantalla_prepro = Preprocesamiento()        # 2
        self.pantalla_bivariado = AnalisisBivariado()    # 3
        self.pantalla_ml = SegmentacionML()              # 4
        self.pantalla_dl = DeepLearning()                # 5

        # Agregarlas al stack
        self.stack.addWidget(self.pantalla_carga)
        self.stack.addWidget(self.pantalla_menu)
        self.stack.addWidget(self.pantalla_prepro)
        self.stack.addWidget(self.pantalla_bivariado)
        self.stack.addWidget(self.pantalla_ml)
        self.stack.addWidget(self.pantalla_dl)

        # === CONEXIONES DE NAVEGACIÓN ===

        # De carga de datos (carga los datos que se hayan subido en excel o csv y realiza el procesamiento) → menú
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = VentanaPrincipal()
    ventana.showMaximized()
    sys.exit(app.exec_())
