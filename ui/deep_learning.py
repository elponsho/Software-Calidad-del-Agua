from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel

class DeepLearning(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deep Learning")

        layout = QVBoxLayout()
        self.label = QLabel("Redes neuronales disponibles:")
        self.btn_cnn = QPushButton("Convolucional (CNN)")
        self.btn_rnn = QPushButton("Recurrente (RNN)")

        self.output = QTextEdit("Resultados...")
        self.btn_regresar = QPushButton("Regresar")

        layout.addWidget(self.label)
        layout.addWidget(self.btn_cnn)
        layout.addWidget(self.btn_rnn)
        layout.addWidget(self.output)
        layout.addWidget(self.btn_regresar)
        self.setLayout(layout)
