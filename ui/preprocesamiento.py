from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QTableWidget, QTableWidgetItem, QGroupBox
from PyQt5.QtGui import QPixmap
from ml.resumen_estadistico import resumen_univariable
from ml.visualizaciones import generar_boxplot, generar_histograma, generar_densidad, obtener_ruta_imagen

class Preprocesamiento(QWidget):
    cambiar_a_bivariado = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Análisis Univariable")
        self.df = None

        layout = QVBoxLayout()

        # Botones principales
        botones = QHBoxLayout()
        self.btn_univ = QPushButton("📊 Resumen")
        self.btn_box = QPushButton("📦 Boxplot")
        self.btn_hist = QPushButton("📈 Histograma")
        self.btn_densidad = QPushButton("🌫️ Densidad")
        self.btn_bivariable = QPushButton("➡️ Bivariable")

        for btn in [self.btn_univ, self.btn_box, self.btn_hist, self.btn_densidad, self.btn_bivariable]:
            btn.setMinimumHeight(32)
            botones.addWidget(btn)

        layout.addLayout(botones)

        # Selección de variable
        self.combo_columnas = QComboBox()
        self.combo_columnas.setMinimumHeight(30)
        layout.addWidget(self.combo_columnas)

        # Gráfica
        self.label_grafica = QLabel()
        self.label_grafica.setMinimumHeight(250)
        self.label_grafica.setScaledContents(True)
        layout.addWidget(self.label_grafica)

        # Resumen estadístico
        self.tabla_datos = QTableWidget()
        layout.addWidget(self.tabla_datos)

        # Botón regresar
        self.btn_regresar = QPushButton("⬅️ Regresar al Menú")
        layout.addWidget(self.btn_regresar)

        self.setLayout(layout)

        # Acciones
        self.btn_univ.clicked.connect(self.mostrar_resumen)
        self.btn_box.clicked.connect(self.mostrar_boxplot)
        self.btn_hist.clicked.connect(self.mostrar_histograma)
        self.btn_densidad.clicked.connect(self.mostrar_densidad)
        self.btn_bivariable.clicked.connect(self.cambiar_a_bivariado.emit)


    def cargar_dataframe(self, df):
        self.df = df
        columnas = df.select_dtypes(include='number').columns.tolist()
        self.combo_columnas.clear()
        self.combo_columnas.addItems(columnas)

    def mostrar_resumen(self):
        if self.df is not None:
            resumen = resumen_univariable(self.df)
            self.mostrar_en_tabla(resumen)

    def mostrar_en_tabla(self, resumen):
        self.tabla_datos.setRowCount(len(resumen))
        self.tabla_datos.setColumnCount(len(resumen.columns))
        self.tabla_datos.setHorizontalHeaderLabels(resumen.columns.tolist())
        self.tabla_datos.setVerticalHeaderLabels(resumen.index.tolist())

        for i, row in enumerate(resumen.values):
            for j, val in enumerate(row):
                self.tabla_datos.setItem(i, j, QTableWidgetItem(str(val)))

    def mostrar_boxplot(self):
        col = self.combo_columnas.currentText()
        generar_boxplot(self.df, col)
        self.label_grafica.setPixmap(QPixmap(obtener_ruta_imagen()))

    def mostrar_histograma(self):
        col = self.combo_columnas.currentText()
        generar_histograma(self.df, col)
        self.label_grafica.setPixmap(QPixmap(obtener_ruta_imagen()))

    def mostrar_densidad(self):
        col = self.combo_columnas.currentText()
        generar_densidad(self.df, col)
        self.label_grafica.setPixmap(QPixmap(obtener_ruta_imagen()))
