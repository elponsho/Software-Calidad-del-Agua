from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QComboBox, QGroupBox, QSizePolicy, QTabWidget,
    QFrame, QGridLayout, QSplitter, QScrollArea, QSpacerItem,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, pyqtSignal
from ml.correlaciones import correlacion_pearson, correlacion_spearman
from ml.visualizaciones import diagrama_dispersion, serie_tiempo, obtener_ruta_imagen


class AnalisisBivariado(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Análisis Bivariado - Relaciones entre Variables")
        self.df = None
        self.setMinimumSize(1600, 1000)
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f7fa;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
            }

            /* Título principal */
            .main-title {
                font-size: 28px;
                font-weight: bold;
                color: white;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #4f46e5, stop:1 #7c3aed);
                padding: 25px;
                border-radius: 12px;
                margin-bottom: 20px;
                text-align: center;
            }

            /* Contenedores principales */
            .main-container {
                background-color: white;
                border: 1px solid #d1d5db;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
            }

            /* Botones de acción */
            .action-button {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px;
                font-size: 16px;
                font-weight: 600;
                margin: 5px;
                min-height: 50px;
            }

            .action-button:hover {
                background-color: #2563eb;
                transform: translateY(-1px);
            }

            .viz-button {
                background-color: #10b981;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px;
                font-size: 16px;
                font-weight: 600;
                margin: 5px;
                min-height: 50px;
            }

            .viz-button:hover {
                background-color: #059669;
                transform: translateY(-1px);
            }

            /* ComboBox mejorados */
            QComboBox {
                background-color: white;
                border: 2px solid #d1d5db;
                border-radius: 6px;
                padding: 12px;
                font-size: 15px;
                min-height: 30px;
                margin: 5px 0;
            }

            QComboBox:focus {
                border-color: #3b82f6;
            }

            /* Labels */
            .section-title {
                font-size: 18px;
                font-weight: bold;
                color: #374151;
                margin-bottom: 15px;
                padding-bottom: 8px;
                border-bottom: 2px solid #e5e7eb;
            }

            .variable-label {
                font-size: 15px;
                font-weight: 600;
                color: #4b5563;
                margin: 10px 0 5px 0;
            }

            /* Pestañas */
            QTabWidget::pane {
                border: 1px solid #d1d5db;
                background-color: white;
                border-radius: 8px;
                padding: 10px;
            }

            QTabBar::tab {
                background: #f3f4f6;
                border: 1px solid #d1d5db;
                padding: 12px 24px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: 600;
                color: #4b5563;
                min-width: 120px;
            }

            QTabBar::tab:selected {
                background: white;
                border-bottom-color: white;
                color: #1f2937;
            }

            QTabBar::tab:hover {
                background: #e5e7eb;
            }

            /* Tablas */
            QTableWidget {
                background-color: white;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                gridline-color: #f3f4f6;
                font-size: 13px;
                selection-background-color: #dbeafe;
            }

            QTableWidget::item {
                padding: 12px 8px;
                border-bottom: 1px solid #f3f4f6;
                text-align: center;
            }

            QHeaderView::section {
                background: #f8fafc;
                border: none;
                border-bottom: 2px solid #e5e7eb;
                border-right: 1px solid #e5e7eb;
                padding: 12px 8px;
                font-weight: bold;
                color: #374151;
                text-align: center;
            }

            /* Área de imagen mejorada */
            .image-scroll {
                background-color: white;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 10px;
            }

            .image-placeholder {
                background-color: #f9fafb;
                border: 2px dashed #d1d5db;
                border-radius: 8px;
                color: #6b7280;
                font-size: 16px;
                text-align: center;
                padding: 40px;
                min-height: 400px;
            }

            /* Botones de navegación */
            .nav-button {
                background-color: #ef4444;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 20px;
                font-size: 15px;
                font-weight: 600;
            }

            .nav-button:hover {
                background-color: #dc2626;
            }

            .help-button {
                background-color: #f59e0b;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 20px;
                font-size: 15px;
                font-weight: 600;
            }

            .help-button:hover {
                background-color: #d97706;
            }
        """)

        # Layout principal
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Título principal
        self.create_header(main_layout)

        # Panel de controles
        self.create_controls(main_layout)

        # Área de resultados con pestañas
        self.create_tabbed_results(main_layout)

        # Navegación
        self.create_navigation(main_layout)

        self.setLayout(main_layout)

    def create_header(self, layout):
        title = QLabel("🔗 Análisis Bivariado")
        title.setProperty("class", "main-title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

    def create_controls(self, layout):
        # Contenedor para controles
        controls_container = QFrame()
        controls_container.setProperty("class", "main-container")

        controls_layout = QVBoxLayout()

        # Título
        controls_title = QLabel("⚙️ Panel de Control")
        controls_title.setProperty("class", "section-title")
        controls_layout.addWidget(controls_title)

        # Layout horizontal para las secciones
        sections_layout = QHBoxLayout()
        sections_layout.setSpacing(20)

        # SECCIÓN 1: Correlaciones
        corr_group = QGroupBox("📊 Análisis de Correlación")
        corr_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 16px;
                color: #374151;
                border: 2px solid #e5e7eb;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
                background-color: #f5f7fa;
            }
        """)

        corr_layout = QVBoxLayout()
        corr_layout.setSpacing(10)

        self.btn_pearson = QPushButton("📈 Correlación de Pearson")
        self.btn_pearson.setProperty("class", "action-button")
        self.btn_pearson.clicked.connect(self.mostrar_pearson)

        self.btn_spearman = QPushButton("📊 Correlación de Spearman")
        self.btn_spearman.setProperty("class", "action-button")
        self.btn_spearman.clicked.connect(self.mostrar_spearman)

        corr_layout.addWidget(self.btn_pearson)
        corr_layout.addWidget(self.btn_spearman)
        corr_layout.addStretch()

        corr_group.setLayout(corr_layout)

        # SECCIÓN 2: Variables
        vars_group = QGroupBox("🎯 Selección de Variables")
        vars_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 16px;
                color: #374151;
                border: 2px solid #e5e7eb;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
                background-color: #f5f7fa;
            }
        """)

        vars_layout = QVBoxLayout()
        vars_layout.setSpacing(8)

        x_label = QLabel("Variable X:")
        x_label.setProperty("class", "variable-label")
        self.combo_x = QComboBox()

        y_label = QLabel("Variable Y:")
        y_label.setProperty("class", "variable-label")
        self.combo_y = QComboBox()

        vars_layout.addWidget(x_label)
        vars_layout.addWidget(self.combo_x)
        vars_layout.addWidget(y_label)
        vars_layout.addWidget(self.combo_y)
        vars_layout.addStretch()

        vars_group.setLayout(vars_layout)

        # SECCIÓN 3: Visualizaciones
        viz_group = QGroupBox("📈 Visualizaciones")
        viz_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 16px;
                color: #374151;
                border: 2px solid #e5e7eb;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
                background-color: #f5f7fa;
            }
        """)

        viz_layout = QVBoxLayout()
        viz_layout.setSpacing(10)

        self.btn_dispersion = QPushButton("🎯 Diagrama de Dispersión")
        self.btn_dispersion.setProperty("class", "viz-button")
        self.btn_dispersion.clicked.connect(self.mostrar_dispersion)

        self.btn_serie_tiempo = QPushButton("📅 Serie de Tiempo")
        self.btn_serie_tiempo.setProperty("class", "viz-button")
        self.btn_serie_tiempo.clicked.connect(self.mostrar_serie_tiempo)

        viz_layout.addWidget(self.btn_dispersion)
        viz_layout.addWidget(self.btn_serie_tiempo)
        viz_layout.addStretch()

        viz_group.setLayout(viz_layout)

        # Agregar secciones
        sections_layout.addWidget(corr_group, 1)
        sections_layout.addWidget(vars_group, 1)
        sections_layout.addWidget(viz_group, 1)

        controls_layout.addLayout(sections_layout)
        controls_container.setLayout(controls_layout)
        layout.addWidget(controls_container)

    def create_tabbed_results(self, layout):
        # Contenedor principal para pestañas
        tabs_container = QFrame()
        tabs_container.setProperty("class", "main-container")

        tabs_layout = QVBoxLayout()

        # Título del área
        tabs_title = QLabel("📊 Resultados y Visualizaciones")
        tabs_title.setProperty("class", "section-title")
        tabs_layout.addWidget(tabs_title)

        # Widget de pestañas
        self.tabs = QTabWidget()

        # Pestaña 1: Tabla de Correlaciones
        self.create_correlation_tab()

        # Pestaña 2: Gráficos
        self.create_graphics_tab()

        # Pestaña 3: Interpretación
        self.create_interpretation_tab()

        tabs_layout.addWidget(self.tabs)
        tabs_container.setLayout(tabs_layout)
        layout.addWidget(tabs_container)

    def create_correlation_tab(self):
        """Crea la pestaña de correlaciones con tabla profesional"""
        corr_widget = QWidget()
        corr_layout = QVBoxLayout()

        # Información del análisis
        self.info_label = QLabel("Seleccione un tipo de correlación para ver los resultados en tabla")
        self.info_label.setStyleSheet("""
            background-color: #f0f9ff;
            border: 1px solid #0ea5e9;
            border-radius: 6px;
            padding: 12px;
            color: #0c4a6e;
            font-weight: 600;
            margin-bottom: 15px;
        """)
        corr_layout.addWidget(self.info_label)

        # Tabla de correlaciones
        self.correlation_table = QTableWidget()
        self.correlation_table.setMinimumHeight(400)
        self.correlation_table.setSortingEnabled(True)

        # Configurar headers para que se ajusten
        header = self.correlation_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        vertical_header = self.correlation_table.verticalHeader()
        vertical_header.setSectionResizeMode(QHeaderView.ResizeToContents)

        corr_layout.addWidget(self.correlation_table)

        corr_widget.setLayout(corr_layout)
        self.tabs.addTab(corr_widget, "📊 Correlaciones")

    def create_graphics_tab(self):
        """Crea la pestaña de gráficos con scroll para imagen completa"""
        graph_widget = QWidget()
        graph_layout = QVBoxLayout()

        # Información del gráfico
        self.graph_info_label = QLabel("Seleccione variables y genere gráficos para verlos aquí")
        self.graph_info_label.setStyleSheet("""
            background-color: #f0fdf4;
            border: 1px solid #22c55e;
            border-radius: 6px;
            padding: 12px;
            color: #15803d;
            font-weight: 600;
            margin-bottom: 15px;
        """)
        graph_layout.addWidget(self.graph_info_label)

        # Área de scroll para imagen completa
        self.scroll_area = QScrollArea()
        self.scroll_area.setProperty("class", "image-scroll")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(500)

        # Label para la imagen dentro del scroll
        self.label_grafica = QLabel()
        self.label_grafica.setProperty("class", "image-placeholder")
        self.label_grafica.setText("""
🎯 ÁREA DE VISUALIZACIONES

Para generar gráficos:

1. Seleccione las variables X e Y en el panel de control
2. Haga clic en el tipo de visualización deseado:
   • Diagrama de Dispersión: Muestra relación entre dos variables
   • Serie de Tiempo: Análisis temporal (requiere columna 'fecha')

3. El gráfico aparecerá aquí en tamaño completo
4. Use las barras de desplazamiento si el gráfico es grande

¡Los gráficos se mostrarán en alta resolución para mejor análisis!
        """)
        self.label_grafica.setAlignment(Qt.AlignCenter)
        self.label_grafica.setWordWrap(True)
        self.label_grafica.setMinimumSize(800, 600)  # Tamaño mínimo generoso

        self.scroll_area.setWidget(self.label_grafica)
        graph_layout.addWidget(self.scroll_area)

        graph_widget.setLayout(graph_layout)
        self.tabs.addTab(graph_widget, "📈 Gráficos")

    def create_interpretation_tab(self):
        """Crea la pestaña de interpretación"""
        interp_widget = QWidget()
        interp_layout = QVBoxLayout()

        # Título
        interp_title = QLabel("🧠 Interpretación Automática")
        interp_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #374151; margin-bottom: 15px;")
        interp_layout.addWidget(interp_title)

        # Área de interpretación
        self.interpretation_area = QTextEdit()
        self.interpretation_area.setReadOnly(True)
        self.interpretation_area.setMinimumHeight(400)
        self.interpretation_area.setText("""
🧠 INTERPRETACIÓN AUTOMÁTICA

Esta sección proporcionará automáticamente:

✅ Análisis detallado de correlaciones encontradas
✅ Identificación de relaciones fuertes, moderadas y débiles
✅ Interpretación específica para calidad del agua
✅ Recomendaciones técnicas basadas en los resultados
✅ Alertas sobre correlaciones inusuales o problemáticas

Para activar la interpretación automática:
1. Ejecute un análisis de correlación (Pearson o Spearman)
2. La interpretación aparecerá aquí automáticamente
3. Use esta información para tomar decisiones técnicas

¡Comience ejecutando un análisis de correlación!
        """)

        interp_layout.addWidget(self.interpretation_area)

        interp_widget.setLayout(interp_layout)
        self.tabs.addTab(interp_widget, "🧠 Interpretación")

    def create_navigation(self, layout):
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(15)

        self.btn_regresar = QPushButton("⬅️ Volver al Menú Principal")
        self.btn_regresar.setProperty("class", "nav-button")

        nav_layout.addWidget(self.btn_regresar)
        nav_layout.addStretch()

        btn_ayuda = QPushButton("❓ Guía Completa")
        btn_ayuda.setProperty("class", "help-button")
        btn_ayuda.clicked.connect(self.mostrar_ayuda)

        nav_layout.addWidget(btn_ayuda)
        layout.addLayout(nav_layout)

    def cargar_dataframe(self, df):
        """Carga el dataframe y actualiza los controles"""
        self.df = df
        columnas = df.select_dtypes(include='number').columns.tolist()

        # Actualizar ComboBoxes
        self.combo_x.clear()
        self.combo_y.clear()
        self.combo_x.addItems(columnas)
        self.combo_y.addItems(columnas)

        # Actualizar mensaje informativo
        filas, cols = df.shape
        self.info_label.setText(
            f"✅ Datos cargados: {filas:,} muestras, {len(columnas)} variables numéricas - Listo para análisis")
        self.graph_info_label.setText(
            f"✅ {len(columnas)} variables disponibles para visualización - Seleccione X e Y para generar gráficos")

    def mostrar_pearson(self):
        """Ejecuta correlación de Pearson y muestra en tabla"""
        if self.df is not None:
            try:
                resultado = correlacion_pearson(self.df.select_dtypes(include='number'))
                self.mostrar_tabla_correlacion("Correlación de Pearson", resultado)
                self.generar_interpretacion(resultado, "Pearson", "lineal")
                self.tabs.setCurrentIndex(0)  # Ir a pestaña de correlaciones
            except Exception as e:
                self.info_label.setText(f"❌ Error en Pearson: {str(e)}")
        else:
            self.info_label.setText("⚠️ No hay datos cargados")

    def mostrar_spearman(self):
        """Ejecuta correlación de Spearman y muestra en tabla"""
        if self.df is not None:
            try:
                resultado = correlacion_spearman(self.df.select_dtypes(include='number'))
                self.mostrar_tabla_correlacion("Correlación de Spearman", resultado)
                self.generar_interpretacion(resultado, "Spearman", "monótona")
                self.tabs.setCurrentIndex(0)  # Ir a pestaña de correlaciones
            except Exception as e:
                self.info_label.setText(f"❌ Error en Spearman: {str(e)}")
        else:
            self.info_label.setText("⚠️ No hay datos cargados")

    def mostrar_tabla_correlacion(self, tipo, resultado):
        """Muestra los resultados de correlación en una tabla profesional"""
        from datetime import datetime

        # Actualizar información
        self.info_label.setText(
            f"📊 {tipo} - Ejecutado: {datetime.now().strftime('%H:%M:%S')} - {len(resultado.columns)} variables")

        # Configurar tabla
        num_vars = len(resultado.columns)
        self.correlation_table.setRowCount(num_vars)
        self.correlation_table.setColumnCount(num_vars)

        # Headers
        variables = resultado.columns.tolist()
        self.correlation_table.setHorizontalHeaderLabels(variables)
        self.correlation_table.setVerticalHeaderLabels(variables)

        # Llenar datos con formato y colores
        for i in range(num_vars):
            for j in range(num_vars):
                valor = resultado.iloc[i, j]

                # Formatear valor
                if i == j:
                    texto = "1.000"
                else:
                    texto = f"{valor:.3f}"

                item = QTableWidgetItem(texto)
                item.setTextAlignment(Qt.AlignCenter)

                # Colorear según fuerza de correlación
                if i == j:
                    # Diagonal principal
                    item.setBackground(Qt.lightGray)
                elif abs(valor) >= 0.7:
                    # Correlación fuerte
                    if valor > 0:
                        item.setBackground(Qt.green)
                        item.setForeground(Qt.white)
                    else:
                        item.setBackground(Qt.red)
                        item.setForeground(Qt.white)
                elif abs(valor) >= 0.3:
                    # Correlación moderada
                    if valor > 0:
                        item.setBackground(Qt.yellow)
                    else:
                        item.setBackground(Qt.magenta)
                        item.setForeground(Qt.white)

                self.correlation_table.setItem(i, j, item)

        # Ajustar tamaño de columnas
        self.correlation_table.resizeColumnsToContents()

    def mostrar_dispersion(self):
        """Genera diagrama de dispersión con imagen completa"""
        if self.df is None:
            self.mostrar_mensaje_grafico("⚠️ No hay datos cargados")
            return

        x = self.combo_x.currentText()
        y = self.combo_y.currentText()

        if not x or not y:
            self.mostrar_mensaje_grafico("⚠️ Seleccione variables X e Y")
            return

        if x == y:
            self.mostrar_mensaje_grafico("⚠️ Seleccione variables diferentes")
            return

        try:
            self.graph_info_label.setText(f"🎯 Generando dispersión: {x} vs {y}...")

            diagrama_dispersion(self.df, x, y)
            pixmap = QPixmap(obtener_ruta_imagen())

            if not pixmap.isNull():
                # Mostrar imagen en tamaño original (sin escalar)
                self.label_grafica.setPixmap(pixmap)
                self.label_grafica.resize(pixmap.size())
                self.label_grafica.setText("")

                # Actualizar información
                self.graph_info_label.setText(f"✅ Diagrama de dispersión: {x} vs {y} - Use scroll para navegar")

                # Cambiar a pestaña de gráficos
                self.tabs.setCurrentIndex(1)
            else:
                self.mostrar_mensaje_grafico("❌ Error al generar gráfico")

        except Exception as e:
            self.mostrar_mensaje_grafico(f"❌ Error: {str(e)}")

    def mostrar_serie_tiempo(self):
        """Genera serie de tiempo con imagen completa"""
        if self.df is None:
            self.mostrar_mensaje_grafico("⚠️ No hay datos cargados")
            return

        if "fecha" not in self.df.columns:
            self.mostrar_mensaje_grafico("⚠️ No hay columna 'fecha'")
            return

        y = self.combo_y.currentText()
        if not y:
            self.mostrar_mensaje_grafico("⚠️ Seleccione variable Y")
            return

        try:
            self.graph_info_label.setText(f"📅 Generando serie temporal: {y}...")

            serie_tiempo(self.df, "fecha", y)
            pixmap = QPixmap(obtener_ruta_imagen())

            if not pixmap.isNull():
                # Mostrar imagen en tamaño original
                self.label_grafica.setPixmap(pixmap)
                self.label_grafica.resize(pixmap.size())
                self.label_grafica.setText("")

                # Actualizar información
                self.graph_info_label.setText(f"✅ Serie temporal: {y} - Use scroll para navegar")

                # Cambiar a pestaña de gráficos
                self.tabs.setCurrentIndex(1)
            else:
                self.mostrar_mensaje_grafico("❌ Error al generar serie")

        except Exception as e:
            self.mostrar_mensaje_grafico(f"❌ Error: {str(e)}")

    def mostrar_mensaje_grafico(self, mensaje):
        """Muestra mensaje en área de gráficos"""
        self.label_grafica.clear()
        self.label_grafica.setText(mensaje)
        self.label_grafica.setAlignment(Qt.AlignCenter)
        self.label_grafica.resize(800, 600)
        self.graph_info_label.setText(mensaje)

    def generar_interpretacion(self, matriz_corr, tipo_analisis, tipo_relacion):
        """Genera interpretación automática detallada"""
        try:
            interpretacion = []
            interpretacion.append(f"🔍 INTERPRETACIÓN AUTOMÁTICA - {tipo_analisis.upper()}")
            interpretacion.append("=" * 70)
            interpretacion.append("")

            # Análisis de correlaciones
            correlaciones_fuertes = []
            correlaciones_moderadas = []
            correlaciones_debiles = []

            variables = list(matriz_corr.columns)

            for i in range(len(variables)):
                for j in range(i + 1, len(variables)):
                    var1 = variables[i]
                    var2 = variables[j]
                    corr_val = matriz_corr.iloc[i, j]

                    if abs(corr_val) >= 0.7:
                        correlaciones_fuertes.append((var1, var2, corr_val))
                    elif abs(corr_val) >= 0.3:
                        correlaciones_moderadas.append((var1, var2, corr_val))
                    else:
                        correlaciones_debiles.append((var1, var2, corr_val))

            # Resumen ejecutivo
            interpretacion.append("📋 RESUMEN EJECUTIVO:")
            interpretacion.append("")
            interpretacion.append(f"• Correlaciones fuertes (|r| ≥ 0.7): {len(correlaciones_fuertes)}")
            interpretacion.append(f"• Correlaciones moderadas (0.3 ≤ |r| < 0.7): {len(correlaciones_moderadas)}")
            interpretacion.append(f"• Correlaciones débiles (|r| < 0.3): {len(correlaciones_debiles)}")
            interpretacion.append("")

            # Correlaciones fuertes
            if correlaciones_fuertes:
                interpretacion.append("🔴 CORRELACIONES FUERTES (ALTA SIGNIFICANCIA):")
                interpretacion.append("")
                for var1, var2, corr in sorted(correlaciones_fuertes, key=lambda x: abs(x[2]), reverse=True):
                    direccion = "POSITIVA" if corr > 0 else "NEGATIVA"
                    interpretacion.append(f"   • {var1} ↔ {var2}")
                    interpretacion.append(f"     Correlación: {corr:.3f} ({direccion})")
                    interpretacion.append(f"     Interpretación: Relación {tipo_relacion} muy fuerte")

                    # Interpretaciones específicas para calidad del agua
                    self.agregar_interpretacion_especifica(interpretacion, var1, var2, corr)
                    interpretacion.append("")

            # Correlaciones moderadas (mostrar solo las más relevantes)
            if correlaciones_moderadas:
                interpretacion.append("🟡 CORRELACIONES MODERADAS (SIGNIFICANCIA MEDIA):")
                interpretacion.append("")
                # Mostrar las 5 más fuertes
                for var1, var2, corr in sorted(correlaciones_moderadas, key=lambda x: abs(x[2]), reverse=True)[:5]:
                    direccion = "positiva" if corr > 0 else "negativa"
                    interpretacion.append(f"   • {var1} ↔ {var2}: r = {corr:.3f} ({direccion})")

                if len(correlaciones_moderadas) > 5:
                    interpretacion.append(
                        f"   ... y {len(correlaciones_moderadas) - 5} correlaciones moderadas adicionales")
                interpretacion.append("")

            # Análisis específico para calidad del agua
            interpretacion.append("🌊 ANÁLISIS ESPECÍFICO PARA CALIDAD DEL AGUA:")
            interpretacion.append("")

            variables_agua = self.identificar_variables_agua(variables)
            if variables_agua:
                interpretacion.append("Variables de calidad del agua identificadas:")
                for categoria, vars_encontradas in variables_agua.items():
                    if vars_encontradas:
                        interpretacion.append(f"   • {categoria}: {', '.join(vars_encontradas)}")
                interpretacion.append("")

            # Recomendaciones
            interpretacion.append("📊 RECOMENDACIONES TÉCNICAS:")
            interpretacion.append("")

            if correlaciones_fuertes:
                interpretacion.append("• INVESTIGAR correlaciones fuertes encontradas:")
                interpretacion.append("  - Verificar si son relaciones causales o espurias")
                interpretacion.append("  - Considerar variables de confusión")
                interpretacion.append("  - Validar con conocimiento del dominio")
                interpretacion.append("")

            if len(correlaciones_debiles) > len(correlaciones_fuertes) + len(correlaciones_moderadas):
                interpretacion.append("• VARIABLES INDEPENDIENTES predominantes:")
                interpretacion.append("  - Mayoría de variables son independientes")
                interpretacion.append("  - Buscar factores externos no medidos")
                interpretacion.append("  - Considerar análisis multivariado")
                interpretacion.append("")

            # Alertas y warnings
            interpretacion.append("⚠️ ALERTAS Y VERIFICACIONES:")
            interpretacion.append("")

            # Verificar correlaciones inusuales
            correlaciones_inusuales = self.detectar_correlaciones_inusuales(
                correlaciones_fuertes + correlaciones_moderadas)
            if correlaciones_inusuales:
                interpretacion.append("🚨 Correlaciones que requieren verificación:")
                for alerta in correlaciones_inusuales:
                    interpretacion.append(f"   • {alerta}")
                interpretacion.append("")

            interpretacion.append("🎯 PRÓXIMOS PASOS SUGERIDOS:")
            interpretacion.append("")
            interpretacion.append("1. Generar gráficos de dispersión para correlaciones > 0.5")
            interpretacion.append("2. Realizar análisis de regresión para variables altamente correlacionadas")
            interpretacion.append("3. Investigar outliers que puedan afectar correlaciones")
            interpretacion.append("4. Comparar con estándares de calidad del agua vigentes")
            interpretacion.append("5. Implementar monitoreo continuo de variables críticas")
            interpretacion.append("")

            interpretacion.append("📈 NOTA METODOLÓGICA:")
            interpretacion.append(f"• Análisis realizado: {tipo_analisis}")
            interpretacion.append(f"• Tipo de relación detectada: {tipo_relacion}")
            interpretacion.append("• Los resultados deben validarse con conocimiento técnico del dominio")
            interpretacion.append("• Correlación no implica causalidad")

            self.interpretation_area.setText("\n".join(interpretacion))

        except Exception as e:
            self.interpretation_area.setText(f"❌ Error al generar interpretación: {str(e)}")

    def agregar_interpretacion_especifica(self, interpretacion, var1, var2, corr):
        """Agrega interpretaciones específicas para pares de variables de calidad del agua"""
        var1_lower = var1.lower()
        var2_lower = var2.lower()

        # pH relacionado
        if ('ph' in var1_lower and 'oxigeno' in var2_lower) or ('ph' in var2_lower and 'oxigeno' in var1_lower):
            if corr > 0:
                interpretacion.append("     ✅ Relación pH-Oxígeno positiva: Indica equilibrio químico saludable")
            else:
                interpretacion.append("     ⚠️ Relación pH-Oxígeno negativa: Posible proceso de acidificación")

        elif ('ph' in var1_lower and 'conductividad' in var2_lower) or (
                'ph' in var2_lower and 'conductividad' in var1_lower):
            interpretacion.append("     🔍 Relación pH-Conductividad: Evaluar contenido iónico del agua")

        # Temperatura relacionada
        elif ('temperatura' in var1_lower and 'oxigeno' in var2_lower) or (
                'temperatura' in var2_lower and 'oxigeno' in var1_lower):
            if corr < 0:
                interpretacion.append(
                    "     ✅ Relación Temperatura-Oxígeno negativa: Comportamiento normal (solubilidad)")
            else:
                interpretacion.append("     🚨 Relación Temperatura-Oxígeno positiva: Investigar fuentes de oxigenación")

        # Turbidez relacionada
        elif ('turbidez' in var1_lower and 'oxigeno' in var2_lower) or (
                'turbidez' in var2_lower and 'oxigeno' in var1_lower):
            if corr < 0:
                interpretacion.append(
                    "     ✅ Relación Turbidez-Oxígeno negativa: Esperada (partículas consumen oxígeno)")
            else:
                interpretacion.append("     ⚠️ Relación Turbidez-Oxígeno positiva: Revisar fuentes de turbidez")

        # Conductividad relacionada
        elif ('conductividad' in var1_lower and ('solidos' in var2_lower or 'tds' in var2_lower)) or \
                ('conductividad' in var2_lower and ('solidos' in var1_lower or 'tds' in var1_lower)):
            if corr > 0.8:
                interpretacion.append("     ✅ Relación Conductividad-Sólidos muy fuerte: Correlación esperada")
            else:
                interpretacion.append("     🔍 Relación Conductividad-Sólidos débil: Verificar calibración de equipos")

    def identificar_variables_agua(self, variables):
        """Identifica y categoriza variables relacionadas con calidad del agua"""
        categorias = {
            'Físicas': [],
            'Químicas': [],
            'Biológicas': [],
            'Iones': []
        }

        for var in variables:
            var_lower = var.lower()

            # Variables físicas
            if any(term in var_lower for term in ['temperatura', 'turbidez', 'color', 'olor', 'sabor']):
                categorias['Físicas'].append(var)

            # Variables químicas
            elif any(term in var_lower for term in ['ph', 'oxigeno', 'dbo', 'dqo', 'conductividad', 'alcalinidad']):
                categorias['Químicas'].append(var)

            # Variables biológicas
            elif any(term in var_lower for term in ['coliform', 'bacteria', 'algas', 'microorg']):
                categorias['Biológicas'].append(var)

            # Iones
            elif any(term in var_lower for term in ['nitrato', 'fosfato', 'sulfato', 'cloruro', 'hierro', 'manganese']):
                categorias['Iones'].append(var)

        return categorias

    def detectar_correlaciones_inusuales(self, correlaciones):
        """Detecta correlaciones que podrían ser inusuales o problemáticas"""
        alertas = []

        for var1, var2, corr in correlaciones:
            var1_lower = var1.lower()
            var2_lower = var2.lower()

            # pH muy alto con metales
            if (('ph' in var1_lower and any(
                    metal in var2_lower for metal in ['hierro', 'plomo', 'cadmio', 'mercurio'])) or
                    ('ph' in var2_lower and any(
                        metal in var1_lower for metal in ['hierro', 'plomo', 'cadmio', 'mercurio']))):
                if corr > 0:
                    alertas.append(f"{var1}-{var2}: Correlación positiva pH-metal inusual")

            # Oxígeno con contaminantes
            elif (('oxigeno' in var1_lower and any(cont in var2_lower for cont in ['coliform', 'dbo', 'dqo'])) or
                  ('oxigeno' in var2_lower and any(cont in var1_lower for cont in ['coliform', 'dbo', 'dqo']))):
                if corr > 0:
                    alertas.append(f"{var1}-{var2}: Oxígeno correlacionado positivamente con contaminantes")

            # Correlaciones extremadamente altas (posible redundancia)
            elif abs(corr) > 0.95:
                alertas.append(f"{var1}-{var2}: Correlación extrema ({corr:.3f}) - posible redundancia")

        return alertas

    def mostrar_ayuda(self):
        """Muestra guía completa de interpretación"""
        ayuda_texto = """
🔍 GUÍA COMPLETA - ANÁLISIS BIVARIADO

════════════════════════════════════════════════════════════════════

📋 NAVEGACIÓN POR PESTAÑAS

📊 PESTAÑA CORRELACIONES:
   • Tabla interactiva con código de colores
   • Verde: Correlaciones positivas fuertes
   • Rojo: Correlaciones negativas fuertes
   • Amarillo: Correlaciones positivas moderadas
   • Magenta: Correlaciones negativas moderadas
   • Gris: Diagonal principal (autocorrelación = 1)

📈 PESTAÑA GRÁFICOS:
   • Visualizaciones en tamaño completo
   • Área de scroll para imágenes grandes
   • Calidad HD para análisis detallado

🧠 PESTAÑA INTERPRETACIÓN:
   • Análisis automático de resultados
   • Recomendaciones específicas para calidad del agua
   • Alertas sobre correlaciones inusuales

════════════════════════════════════════════════════════════════════

📊 INTERPRETACIÓN DE CORRELACIONES

🔢 RANGOS DE INTERPRETACIÓN:
   • |r| = 0.0 - 0.1: Sin correlación
   • |r| = 0.1 - 0.3: Correlación débil
   • |r| = 0.3 - 0.5: Correlación moderada-baja
   • |r| = 0.5 - 0.7: Correlación moderada-alta
   • |r| = 0.7 - 0.9: Correlación fuerte
   • |r| = 0.9 - 1.0: Correlación muy fuerte

🎯 SIGNIFICANCIA PRÁCTICA:
   • r < 0.3: Relación prácticamente irrelevante
   • 0.3 ≤ r < 0.7: Relación moderada, investigar
   • r ≥ 0.7: Relación fuerte, alta importancia

════════════════════════════════════════════════════════════════════

🌊 CORRELACIONES TÍPICAS EN CALIDAD DEL AGUA

✅ CORRELACIONES POSITIVAS ESPERADAS:
   • Conductividad ↔ Sólidos Totales (0.85-0.95)
   • pH ↔ Alcalinidad (0.6-0.8)
   • Turbidez ↔ Sólidos Suspendidos (0.7-0.9)
   • DBO ↔ DQO (0.6-0.8)
   • Nitratos ↔ Fosfatos (contaminación agrícola)

✅ CORRELACIONES NEGATIVAS ESPERADAS:
   • Temperatura ↔ Oxígeno Disuelto (-0.7 a -0.9)
   • pH ↔ Hierro (-0.4 a -0.7)
   • Oxígeno ↔ DBO (-0.5 a -0.8)
   • pH ↔ Acidez (-0.8 a -0.95)

🚨 CORRELACIONES DE ALERTA:
   • Coliformes ↔ cualquier variable positiva alta
   • Metales pesados ↔ pH positiva
   • Oxígeno ↔ contaminantes positiva

════════════════════════════════════════════════════════════════════

🔧 USO PRÁCTICO DE RESULTADOS

📊 CORRELACIONES FUERTES (|r| ≥ 0.7):
   • Implementar monitoreo conjunto
   • Una variable puede predecir la otra
   • Considerar para modelos predictivos
   • Evaluar redundancia en monitoreo

📈 CORRELACIONES MODERADAS (0.3 ≤ |r| < 0.7):
   • Relación significativa pero no determinante
   • Considerar en análisis multivariado
   • Útil para identificar tendencias

📉 CORRELACIONES DÉBILES (|r| < 0.3):
   • Variables prácticamente independientes
   • Monitoreo independiente necesario
   • Buscar factores externos

════════════════════════════════════════════════════════════════════

⚠️ PRECAUCIONES Y LIMITACIONES

🚫 CORRELACIÓN ≠ CAUSALIDAD:
   • Una correlación NO implica que una variable cause la otra
   • Pueden existir variables ocultas (confounding)
   • Validar con conocimiento técnico del dominio

📊 FACTORES QUE AFECTAN CORRELACIONES:
   • Valores atípicos (outliers)
   • Transformaciones no lineales
   • Heteroscedasticidad
   • Autocorrelación temporal

🔍 VALIDACIÓN NECESARIA:
   • Verificar normalidad para Pearson
   • Examinar gráficos de dispersión
   • Considerar transformaciones de datos
   • Evaluar estacionalidad en series temporales

════════════════════════════════════════════════════════════════════

🎯 FLUJO DE TRABAJO RECOMENDADO

1. 📊 ANÁLISIS EXPLORATORIO:
   • Comience con Spearman (más robusto)
   • Identifique correlaciones > 0.5
   • Revise la tabla de colores

2. 📈 VISUALIZACIÓN:
   • Genere gráficos de dispersión para correlaciones altas
   • Examine patrones y outliers
   • Verifique linealidad

3. 🧠 INTERPRETACIÓN:
   • Lea la interpretación automática
   • Compare con conocimiento técnico
   • Identifique correlaciones problemáticas

4. 🔧 ACCIÓN:
   • Implemente recomendaciones
   • Ajuste protocolos de monitoreo
   • Planifique investigaciones adicionales

════════════════════════════════════════════════════════════════════

✨ ¡Use esta guía como referencia constante durante su análisis!
        """

        self.interpretation_area.setText(ayuda_texto)
        self.tabs.setCurrentIndex(2)  # Ir a pestaña de interpretación