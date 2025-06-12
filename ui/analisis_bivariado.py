from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel

class AnalisisBivariado(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Análisis Bivariado")

        layout = QVBoxLayout()
        self.label = QLabel("Selecciona un análisis:")
        self.btn_pearson = QPushButton("Correlación de Pearson")
        self.btn_spearman = QPushButton("Correlación de Spearman")
        self.btn_dispersion = QPushButton("Diagrama de Dispersión")
        self.btn_serie_tiempo = QPushButton("Serie de Tiempo")

        self.output = QTextEdit("Resultados...")
        self.btn_regresar = QPushButton("Regresar")

        layout.addWidget(self.label)
        layout.addWidget(self.btn_pearson)
        layout.addWidget(self.btn_spearman)
        layout.addWidget(self.btn_dispersion)
        layout.addWidget(self.btn_serie_tiempo)
        layout.addWidget(self.output)
        layout.addWidget(self.btn_regresar)
        self.setLayout(layout)
