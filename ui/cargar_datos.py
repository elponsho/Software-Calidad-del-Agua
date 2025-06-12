from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QFileDialog, QLabel
from ml.cargar_datos import leer_archivo_csv, leer_archivo_excel

class CargaDatos(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Carga de datos")
        self.df = None  # Aquí se guarda el DataFrame

        layout = QVBoxLayout()
        botones_layout = QHBoxLayout()

        self.btn_excel = QPushButton("Excel")
        self.btn_csv = QPushButton("CSV")
        self.btn_api = QPushButton("API")  # pendiente implementar

        botones_layout.addWidget(self.btn_excel)
        botones_layout.addWidget(self.btn_csv)
        botones_layout.addWidget(self.btn_api)

        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.label_info = QLabel("Vista previa:")

        self.btn_cargar = QPushButton("Cargar")
        self.btn_cargar.setEnabled(False)

        layout.addLayout(botones_layout)
        layout.addWidget(self.label_info)
        layout.addWidget(self.preview)
        layout.addWidget(self.btn_cargar)
        self.setLayout(layout)

        # Conexiones
        self.btn_csv.clicked.connect(self.cargar_csv)
        self.btn_excel.clicked.connect(self.cargar_excel)

    def cargar_csv(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo CSV", "", "CSV Files (*.csv)")
        if ruta:
            self.df = leer_archivo_csv(ruta)
            if self.df is not None:
                self.preview.setText(str(self.df.head()))
                self.btn_cargar.setEnabled(True)  # ← Activar botón
            else:
                self.preview.setText("Error al leer archivo CSV")
                self.btn_cargar.setEnabled(False)

    def cargar_excel(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo Excel", "", "Excel Files (*.xlsx *.xls)")
        if ruta:
            self.df = leer_archivo_excel(ruta)
            if self.df is not None:
                self.preview.setText(str(self.df.head()))
                self.btn_cargar.setEnabled(True)  # ← Activar botón
            else:
                self.preview.setText("Error al leer archivo Excel")
                self.btn_cargar.setEnabled(False)
