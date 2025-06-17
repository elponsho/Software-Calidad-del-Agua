from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QGroupBox

class MenuPrincipal(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Menú Principal")

        layout = QVBoxLayout()

        titulo = QLabel("📘 Selecciona una opción de análisis:")
        titulo.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(titulo)

        grupo_opciones = QGroupBox("📊 Opciones disponibles")
        opciones_layout = QVBoxLayout()

        self.btn_prepro = QPushButton("1️⃣ Preprocesamiento")
        self.btn_ml = QPushButton("2️⃣ Machine Learning")
        self.btn_dl = QPushButton("3️⃣ Deep Learning")

        for boton in [self.btn_prepro, self.btn_ml, self.btn_dl]:
            boton.setMinimumHeight(40)
            opciones_layout.addWidget(boton)

        grupo_opciones.setLayout(opciones_layout)
        layout.addWidget(grupo_opciones)

        self.setLayout(layout)
