from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QGroupBox
import pandas as pd

class CargaDatos(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Carga de Datos")
        self.df = None

        layout = QVBoxLayout()

        # Botones de carga
        botones_layout = QHBoxLayout()
        self.btn_excel = QPushButton("📂 Cargar Excel")
        self.btn_csv = QPushButton("📂 Cargar CSV")
        self.btn_api = QPushButton("🌐 Cargar desde API")
        botones_layout.addWidget(self.btn_excel)
        botones_layout.addWidget(self.btn_csv)
        botones_layout.addWidget(self.btn_api)

        # Vista previa
        group_vista = QGroupBox("📊 Vista previa de los datos")
        vista_layout = QVBoxLayout()
        self.tabla_preview = QTableWidget()
        vista_layout.addWidget(self.tabla_preview)
        group_vista.setLayout(vista_layout)

        # Botón de continuar
        self.btn_cargar = QPushButton("➡️ Continuar al menú")
        self.btn_cargar.setEnabled(False)

        layout.addLayout(botones_layout)
        layout.addWidget(group_vista)
        layout.addWidget(self.btn_cargar)
        self.setLayout(layout)

        self.btn_excel.clicked.connect(self.cargar_excel)
        self.btn_csv.clicked.connect(self.cargar_csv)

    def cargar_excel(self):
        archivo, _ = QFileDialog.getOpenFileName(self, "Selecciona un archivo Excel", "", "Archivos Excel (*.xlsx *.xls)")
        if archivo:
            try:
                df = pd.read_excel(archivo)
                self.actualizar_tabla(df)
            except Exception as e:
                print("Error al leer Excel:", e)

    def cargar_csv(self):
        archivo, _ = QFileDialog.getOpenFileName(self, "Selecciona un archivo CSV", "", "Archivos CSV (*.csv)")
        if archivo:
            try:
                df = pd.read_csv(archivo)
                self.actualizar_tabla(df)
            except Exception as e:
                print("Error al leer CSV:", e)

    def actualizar_tabla(self, df):
        df.rename(columns={"Sampling_date": "fecha"}, inplace=True)
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        self.df = df
        self.btn_cargar.setEnabled(True)

        self.tabla_preview.setRowCount(len(df.head(10)))
        self.tabla_preview.setColumnCount(len(df.columns))
        self.tabla_preview.setHorizontalHeaderLabels(df.columns.tolist())

        for i, row in enumerate(df.head(10).values):
            for j, val in enumerate(row):
                self.tabla_preview.setItem(i, j, QTableWidgetItem(str(val)))
