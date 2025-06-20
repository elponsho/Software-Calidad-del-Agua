from ui.cargar_datos import CargaDatos

# En tu menú principal:
def abrir_carga_datos(self):
    self.carga_window = CargaDatos()
    self.carga_window.data_loaded_signal.connect(self.on_datos_cargados)
    self.carga_window.show()

def on_datos_cargados(self, df):
    print(f"Datos recibidos: {df.shape}")
    # Aquí procesas los datos cargados