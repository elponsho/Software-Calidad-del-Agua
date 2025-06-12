from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton

class MenuPrincipal(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Menú principal")

        layout = QVBoxLayout()
        self.btn_preprocesamiento = QPushButton("Preprocesamiento")
        self.btn_ml = QPushButton("Machine Learning")
        self.btn_dl = QPushButton("Deep Learning")

        layout.addWidget(self.btn_preprocesamiento)
        layout.addWidget(self.btn_ml)
        layout.addWidget(self.btn_dl)
        self.setLayout(layout)
