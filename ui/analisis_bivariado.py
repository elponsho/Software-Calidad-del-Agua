from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel, QComboBox
from PyQt5.QtGui import QPixmap
from ml.correlaciones import correlacion_pearson, correlacion_spearman
from ml.visualizaciones import diagrama_dispersion, serie_tiempo, obtener_ruta_imagen

class AnalisisBivariado(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Análisis Bivariado")
        self.df = None

        layout = QVBoxLayout()
        self.label = QLabel("Selecciona un análisis:")

        self.btn_pearson = QPushButton("Correlación de Pearson")
        self.btn_spearman = QPushButton("Correlación de Spearman")
        self.btn_dispersion = QPushButton("Diagrama de Dispersión")
        self.btn_serie_tiempo = QPushButton("Serie de Tiempo")

        self.combo_x = QComboBox()
        self.combo_y = QComboBox()

        self.label_grafica = QLabel()
        self.output = QTextEdit("Resultados...")
        self.output.setReadOnly(True)
        self.btn_regresar = QPushButton("Regresar")

        layout.addWidget(self.label)
        layout.addWidget(self.btn_pearson)
        layout.addWidget(self.btn_spearman)
        layout.addWidget(self.combo_x)
        layout.addWidget(self.combo_y)
        layout.addWidget(self.btn_dispersion)
        layout.addWidget(self.btn_serie_tiempo)
        layout.addWidget(self.label_grafica)
        layout.addWidget(self.output)
        layout.addWidget(self.btn_regresar)

        self.setLayout(layout)

        # Conexiones
        self.btn_pearson.clicked.connect(self.mostrar_pearson)
        self.btn_spearman.clicked.connect(self.mostrar_spearman)
        self.btn_dispersion.clicked.connect(self.mostrar_dispersion)
        self.btn_serie_tiempo.clicked.connect(self.mostrar_serie_tiempo)

    def cargar_dataframe(self, df):
        self.df = df
        columnas = df.select_dtypes(include='number').columns.tolist()
        self.combo_x.clear()
        self.combo_y.clear()
        self.combo_x.addItems(columnas)
        self.combo_y.addItems(columnas)

    def mostrar_pearson(self):
        if self.df is not None:
            resultado = correlacion_pearson(self.df.select_dtypes(include='number'))
            self.output.setText(str(resultado))

    def mostrar_spearman(self):
        if self.df is not None:
            resultado = correlacion_spearman(self.df.select_dtypes(include='number'))
            self.output.setText(str(resultado))

    def mostrar_dispersion(self):
        if self.df is not None:
            x = self.combo_x.currentText()
            y = self.combo_y.currentText()
            diagrama_dispersion(self.df, x, y)
            self.label_grafica.setPixmap(QPixmap(obtener_ruta_imagen()))

    def mostrar_serie_tiempo(self):
        if self.df is not None:
            y = self.combo_y.currentText()
            if "fecha" in self.df.columns:
                serie_tiempo(self.df, "fecha", y)
                self.label_grafica.setPixmap(QPixmap(obtener_ruta_imagen()))
            else:
                self.output.setText("No hay columna 'fecha' en los datos.")
