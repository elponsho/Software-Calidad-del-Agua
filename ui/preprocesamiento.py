from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit
from PyQt5.QtCore import pyqtSignal
from ml.resumen_estadistico import resumen_univariable

class Preprocesamiento(QWidget):
    cambiar_a_bivariado = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Preprocesamiento")
        self.df = None  # Aquí recibiremos el DataFrame

        layout = QVBoxLayout()
        opciones_layout = QHBoxLayout()

        self.btn_univ = QPushButton("Univariable")
        self.btn_biv = QPushButton("Bivariable")
        opciones_layout.addWidget(self.btn_univ)
        opciones_layout.addWidget(self.btn_biv)

        self.txt_datos = QTextEdit("Datos")
        self.txt_datos.setReadOnly(True)
        self.btn_regresar = QPushButton("Regresar")

        layout.addLayout(opciones_layout)
        layout.addWidget(self.txt_datos)
        layout.addWidget(self.btn_regresar)
        self.setLayout(layout)

        self.btn_biv.clicked.connect(self.cambiar_a_bivariado.emit)
        self.btn_univ.clicked.connect(self.mostrar_resumen)

    def mostrar_resumen(self):
        if self.df is not None:
            resumen = resumen_univariable(self.df.select_dtypes(include='number'))
            self.txt_datos.setText(str(resumen))
        else:
            self.txt_datos.setText("No hay datos cargados.")
