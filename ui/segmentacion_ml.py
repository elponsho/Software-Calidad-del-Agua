from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel

class SegmentacionML(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Segmentación - Machine Learning")

        layout = QVBoxLayout()
        self.label = QLabel("Modelos disponibles:")
        self.btn_normalizar = QPushButton("Normalizar Datos")
        self.btn_pca = QPushButton("Análisis de Componentes Principales (PCA)")
        self.btn_cluster = QPushButton("Clustering Jerárquico")
        self.btn_arbol = QPushButton("Árbol de Decisión")
        self.btn_rf = QPushButton("Random Forest")
        self.btn_svm = QPushButton("SVM")
        self.btn_hiperparam = QPushButton("Búsqueda de Hiperparámetros")

        self.output = QTextEdit("Resultados...")
        self.btn_regresar = QPushButton("Regresar")

        layout.addWidget(self.label)
        layout.addWidget(self.btn_normalizar)
        layout.addWidget(self.btn_pca)
        layout.addWidget(self.btn_cluster)
        layout.addWidget(self.btn_arbol)
        layout.addWidget(self.btn_rf)
        layout.addWidget(self.btn_svm)
        layout.addWidget(self.btn_hiperparam)
        layout.addWidget(self.output)
        layout.addWidget(self.btn_regresar)
        self.setLayout(layout)
