from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QComboBox, QGroupBox, QSizePolicy, QTabWidget,
    QFrame, QGridLayout, QSplitter, QScrollArea, QSpacerItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QDateEdit
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QDate
from ml.correlaciones import correlacion_pearson, correlacion_spearman
from ml.visualizaciones import diagrama_dispersion, serie_tiempo, obtener_ruta_imagen
import pandas as pd


class AnalisisBivariado(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Análisis Bivariado - Relaciones entre Variables")
        self.df = None
        self.columna_fecha = None  # Para almacenar la columna de fecha detectada
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

            .viz-button:disabled {
                background-color: #9ca3af;
                color: #6b7280;
            }

            .time-button {
                background-color: #f59e0b;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px;
                font-size: 16px;
                font-weight: 600;
                margin: 5px;
                min-height: 50px;
            }

            .time-button:hover {
                background-color: #d97706;
                transform: translateY(-1px);
            }

            .time-button:disabled {
                background-color: #9ca3af;
                color: #6b7280;
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

            /* DateEdit mejorados */
            QDateEdit {
                background-color: white;
                border: 2px solid #d1d5db;
                border-radius: 6px;
                padding: 12px;
                font-size: 15px;
                min-height: 30px;
                margin: 5px 0;
            }

            QDateEdit:focus {
                border-color: #f59e0b;
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

            .date-info {
                font-size: 13px;
                color: #059669;
                font-weight: 600;
                background-color: #ecfdf5;
                padding: 8px;
                border-radius: 4px;
                margin: 5px 0;
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
                border: 2px solid #e5e7eb;
                border-radius: 12px;
                padding: 20px;
            }

            .image-placeholder {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #f8fafc, stop:1 #e2e8f0);
                border: 3px dashed #94a3b8;
                border-radius: 12px;
                color: #475569;
                font-size: 16px;
                text-align: center;
                padding: 40px;
                min-height: 600px;
                line-height: 1.8;
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

            /* Scroll areas específicas */
            QScrollArea {
                border: 1px solid #d1d5db;
                border-radius: 8px;
                background-color: white;
            }

            QScrollBar:vertical {
                background-color: #f3f4f6;
                width: 14px;
                border-radius: 7px;
            }

            QScrollBar::handle:vertical {
                background-color: #9ca3af;
                border-radius: 7px;
                min-height: 25px;
                margin: 2px;
            }

            QScrollBar::handle:vertical:hover {
                background-color: #6b7280;
            }

            QScrollBar:horizontal {
                background-color: #f3f4f6;
                height: 14px;
                border-radius: 7px;
            }

            QScrollBar::handle:horizontal {
                background-color: #9ca3af;
                border-radius: 7px;
                min-width: 25px;
                margin: 2px;
            }

            QScrollBar::handle:horizontal:hover {
                background-color: #6b7280;
            }
        """)

        # Layout principal con scroll
        main_scroll = QScrollArea()
        main_scroll.setWidgetResizable(True)
        main_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        main_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        main_widget = QWidget()
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

        main_widget.setLayout(main_layout)
        main_scroll.setWidget(main_widget)

        # Layout final
        final_layout = QVBoxLayout()
        final_layout.setContentsMargins(0, 0, 0, 0)
        final_layout.addWidget(main_scroll)
        self.setLayout(final_layout)

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

        # SECCIÓN 2: Diagrama de Dispersión (Variables X e Y)
        scatter_group = QGroupBox("🎯 Diagrama de Dispersión")
        scatter_group.setStyleSheet("""
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

        scatter_layout = QVBoxLayout()
        scatter_layout.setSpacing(8)

        x_label = QLabel("Variable X (Horizontal):")
        x_label.setProperty("class", "variable-label")
        self.combo_x = QComboBox()

        y_label = QLabel("Variable Y (Vertical):")
        y_label.setProperty("class", "variable-label")
        self.combo_y = QComboBox()

        self.btn_dispersion = QPushButton("🎯 Generar Dispersión")
        self.btn_dispersion.setProperty("class", "viz-button")
        self.btn_dispersion.clicked.connect(self.mostrar_dispersion)

        scatter_layout.addWidget(x_label)
        scatter_layout.addWidget(self.combo_x)
        scatter_layout.addWidget(y_label)
        scatter_layout.addWidget(self.combo_y)
        scatter_layout.addWidget(self.btn_dispersion)
        scatter_layout.addStretch()

        scatter_group.setLayout(scatter_layout)

        # SECCIÓN 3: Serie de Tiempo
        time_group = QGroupBox("📅 Serie de Tiempo")
        time_group.setStyleSheet("""
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

        time_layout = QVBoxLayout()
        time_layout.setSpacing(8)

        # Variable para serie de tiempo
        var_time_label = QLabel("Variable a Analizar:")
        var_time_label.setProperty("class", "variable-label")
        self.combo_time_var = QComboBox()

        # Label para mostrar información de fecha detectada
        self.date_info_label = QLabel("📅 No se detectó columna de fecha")
        self.date_info_label.setProperty("class", "date-info")
        self.date_info_label.setVisible(False)

        # Rango de fechas
        date_range_label = QLabel("Período de Análisis:")
        date_range_label.setProperty("class", "variable-label")

        # Layout horizontal para fechas
        date_layout = QHBoxLayout()
        date_layout.setSpacing(10)

        from_label = QLabel("Desde:")
        from_label.setStyleSheet("font-size: 13px; color: #6b7280; font-weight: 600;")
        self.date_from = QDateEdit()
        self.date_from.setCalendarPopup(True)
        self.date_from.setDate(QDate.currentDate().addYears(-1))

        to_label = QLabel("Hasta:")
        to_label.setStyleSheet("font-size: 13px; color: #6b7280; font-weight: 600;")
        self.date_to = QDateEdit()
        self.date_to.setCalendarPopup(True)
        self.date_to.setDate(QDate.currentDate())

        date_layout.addWidget(from_label)
        date_layout.addWidget(self.date_from, 1)
        date_layout.addWidget(to_label)
        date_layout.addWidget(self.date_to, 1)

        self.btn_serie_tiempo = QPushButton("📈 Generar Serie Temporal")
        self.btn_serie_tiempo.setProperty("class", "time-button")
        self.btn_serie_tiempo.clicked.connect(self.mostrar_serie_tiempo)

        time_layout.addWidget(var_time_label)
        time_layout.addWidget(self.combo_time_var)
        time_layout.addWidget(self.date_info_label)
        time_layout.addWidget(date_range_label)
        time_layout.addLayout(date_layout)
        time_layout.addWidget(self.btn_serie_tiempo)
        time_layout.addStretch()

        time_group.setLayout(time_layout)

        # Agregar secciones
        sections_layout.addWidget(corr_group, 1)
        sections_layout.addWidget(scatter_group, 1)
        sections_layout.addWidget(time_group, 1)

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

        # Scroll area para la tabla
        table_scroll = QScrollArea()
        table_scroll.setWidgetResizable(True)
        table_scroll.setMinimumHeight(450)

        # Tabla de correlaciones
        self.correlation_table = QTableWidget()
        self.correlation_table.setMinimumHeight(400)
        self.correlation_table.setSortingEnabled(True)

        # Configurar headers para que se ajusten
        header = self.correlation_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        vertical_header = self.correlation_table.verticalHeader()
        vertical_header.setSectionResizeMode(QHeaderView.ResizeToContents)

        table_scroll.setWidget(self.correlation_table)
        corr_layout.addWidget(table_scroll)

        corr_widget.setLayout(corr_layout)
        self.tabs.addTab(corr_widget, "📊 Correlaciones")

    def create_graphics_tab(self):
        """Crea la pestaña de gráficos con visualización optimizada y completa"""
        graph_widget = QWidget()
        graph_layout = QVBoxLayout()
        graph_layout.setContentsMargins(10, 10, 10, 10)
        graph_layout.setSpacing(10)

        # Información del gráfico
        self.graph_info_label = QLabel("Configure las variables y genere visualizaciones para verlas aquí")
        self.graph_info_label.setStyleSheet("""
            background-color: #f0fdf4;
            border: 1px solid #22c55e;
            border-radius: 8px;
            padding: 15px;
            color: #15803d;
            font-weight: 600;
            font-size: 14px;
        """)
        graph_layout.addWidget(self.graph_info_label)

        # Área de scroll optimizada para gráficos completos
        self.scroll_area = QScrollArea()
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: white;
                border: 2px solid #e5e7eb;
                border-radius: 12px;
                padding: 5px;
            }
            QScrollBar:vertical {
                background-color: #f3f4f6;
                width: 16px;
                border-radius: 8px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background-color: #9ca3af;
                border-radius: 8px;
                min-height: 30px;
                margin: 2px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #6b7280;
            }
            QScrollBar:horizontal {
                background-color: #f3f4f6;
                height: 16px;
                border-radius: 8px;
                margin: 2px;
            }
            QScrollBar::handle:horizontal {
                background-color: #9ca3af;
                border-radius: 8px;
                min-width: 30px;
                margin: 2px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #6b7280;
            }
        """)

        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(700)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Widget contenedor principal para el gráfico
        self.image_container = QWidget()
        self.image_container.setStyleSheet("background-color: white;")

        # Layout centrado para el contenedor
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(20, 20, 20, 20)
        container_layout.setSpacing(0)
        container_layout.setAlignment(Qt.AlignCenter)

        # Label para la imagen con configuración optimizada
        self.label_grafica = QLabel()
        self.label_grafica.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #f8fafc, stop:1 #e2e8f0);
                border: 3px dashed #94a3b8;
                border-radius: 15px;
                color: #475569;
                font-size: 16px;
                font-weight: 600;
                padding: 30px;
                line-height: 1.8;
            }
        """)

        self.label_grafica.setText("""
🎨 ÁREA DE VISUALIZACIÓN PROFESIONAL

┌─ CONFIGURACIÓN DE ANÁLISIS ─────────────────────────────────┐
│                                                             │
│  📊 DIAGRAMA DE DISPERSIÓN:                                 │
│     • Seleccione Variable X (eje horizontal)               │
│     • Seleccione Variable Y (eje vertical)                 │
│     • Presione "Generar Dispersión"                        │
│     • Ideal para relaciones entre dos variables numéricas  │
│                                                             │
│  📈 SERIE DE TIEMPO:                                        │
│     • Seleccione una Variable a Analizar                   │
│     • Configure el período (Desde - Hasta)                 │
│     • Presione "Generar Serie Temporal"                    │
│     • Automático cuando se detecta columna de fecha        │
│                                                             │
└─────────────────────────────────────────────────────────────┘

🔍 CARACTERÍSTICAS DE VISUALIZACIÓN:

✨ Gráficos Completamente Visibles
✨ Escala Automática y Centrado
✨ Navegación Optimizada con Scroll
✨ Resolución HD para Análisis Detallado
✨ Información Dimensional en Tiempo Real

🎯 FLUJO DE TRABAJO RECOMENDADO:

1️⃣ Configure las variables en el Panel de Control
2️⃣ Genere la visualización deseada
3️⃣ La gráfica aparecerá completamente visible
4️⃣ Use scroll si la gráfica es muy grande
5️⃣ Analice patrones, tendencias y correlaciones

¡Los gráficos aparecerán aquí completamente visibles!
        """)

        self.label_grafica.setAlignment(Qt.AlignCenter)
        self.label_grafica.setWordWrap(True)
        self.label_grafica.setMinimumSize(1000, 700)
        self.label_grafica.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_grafica.setScaledContents(False)  # Importante: no escalar contenido automáticamente

        container_layout.addWidget(self.label_grafica)
        self.image_container.setLayout(container_layout)
        self.scroll_area.setWidget(self.image_container)

        graph_layout.addWidget(self.scroll_area)
        graph_widget.setLayout(graph_layout)
        self.tabs.addTab(graph_widget, "📈 Visualizaciones")

    def create_interpretation_tab(self):
        """Crea la pestaña de interpretación con scroll"""
        interp_widget = QWidget()
        interp_layout = QVBoxLayout()

        # Título
        interp_title = QLabel("🧠 Interpretación Automática")
        interp_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #374151; margin-bottom: 15px;")
        interp_layout.addWidget(interp_title)

        # Área de interpretación con scroll
        self.interpretation_area = QTextEdit()
        self.interpretation_area.setReadOnly(True)
        self.interpretation_area.setMinimumHeight(500)
        self.interpretation_area.setText("""
🧠 INTERPRETACIÓN AUTOMÁTICA AVANZADA

Esta sección proporcionará automáticamente:

📊 ANÁLISIS DE CORRELACIONES:
   ✅ Clasificación automática por fuerza (débil, moderada, fuerte)
   ✅ Identificación de relaciones positivas y negativas
   ✅ Interpretación específica para calidad del agua
   ✅ Detección de correlaciones inusuales o problemáticas

📈 ANÁLISIS DE VISUALIZACIONES:
   ✅ Identificación de patrones en dispersión
   ✅ Detección de outliers y valores atípicos
   ✅ Análisis de tendencias temporales
   ✅ Evaluación de estacionalidad en series de tiempo

🌊 CONTEXTO DE CALIDAD DEL AGUA:
   ✅ Interpretación de variables físicas, químicas y biológicas
   ✅ Alertas sobre valores fuera de rango normal
   ✅ Recomendaciones técnicas específicas
   ✅ Correlaciones esperadas vs. inusuales

🎯 RECOMENDACIONES TÉCNICAS:
   ✅ Próximos pasos de análisis
   ✅ Variables que requieren monitoreo especial
   ✅ Investigaciones adicionales sugeridas
   ✅ Protocolos de calidad recomendados

Para activar la interpretación automática:
1. Ejecute un análisis de correlación (Pearson o Spearman)
2. Genere visualizaciones para variables de interés
3. La interpretación aparecerá aquí automáticamente
4. Use esta información para tomar decisiones técnicas

¡Comience ejecutando un análisis para ver la interpretación completa!
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

    def detectar_columna_fecha(self, df):
        """Detecta automáticamente columnas que podrían contener fechas"""
        posibles_nombres = [
            'fecha', 'date', 'sampling_date', 'sample_date', 'fecha_muestreo',
            'timestamp', 'time', 'datetime', 'fecha_muestra', 'sampling_datetime'
        ]

        # Buscar por nombre de columna
        for col in df.columns:
            if col.lower() in posibles_nombres:
                return col

        # Buscar columnas que contengan palabras clave
        for col in df.columns:
            col_lower = col.lower()
            if any(palabra in col_lower for palabra in ['fecha', 'date', 'time', 'sampling']):
                return col

        # Buscar por tipo de dato (si ya están convertidas a datetime)
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col

        # Buscar columnas que se puedan convertir a fecha
        for col in df.columns:
            if df[col].dtype == 'object':  # Solo string/object columns
                try:
                    # Intentar convertir una muestra pequeña
                    sample = df[col].dropna().head(3)
                    pd.to_datetime(sample)
                    return col
                except:
                    continue

        return None

    def cargar_dataframe(self, df):
        """Carga el dataframe y actualiza los controles"""
        self.df = df

        # Detectar columna de fecha
        self.columna_fecha = self.detectar_columna_fecha(df)

        # Obtener solo columnas numéricas
        columnas = df.select_dtypes(include='number').columns.tolist()

        # Actualizar ComboBoxes para dispersión
        self.combo_x.clear()
        self.combo_y.clear()
        self.combo_x.addItems(columnas)
        self.combo_y.addItems(columnas)

        # Actualizar ComboBox para serie de tiempo
        self.combo_time_var.clear()
        self.combo_time_var.addItems(columnas)

        # Actualizar información de fecha y configurar fechas
        if self.columna_fecha:
            self.date_info_label.setText(f"📅 Columna de fecha detectada: {self.columna_fecha}")
            self.date_info_label.setVisible(True)
            self.btn_serie_tiempo.setEnabled(True)
            self.btn_serie_tiempo.setToolTip(f"Serie de tiempo usando columna: {self.columna_fecha}")

            # Configurar rango de fechas basado en los datos
            try:
                fechas = pd.to_datetime(df[self.columna_fecha])
                fecha_min = fechas.min()
                fecha_max = fechas.max()

                # Convertir a QDate
                q_fecha_min = QDate(fecha_min.year, fecha_min.month, fecha_min.day)
                q_fecha_max = QDate(fecha_max.year, fecha_max.month, fecha_max.day)

                # Configurar los QDateEdit
                self.date_from.setDateRange(q_fecha_min, q_fecha_max)
                self.date_to.setDateRange(q_fecha_min, q_fecha_max)
                self.date_from.setDate(q_fecha_min)
                self.date_to.setDate(q_fecha_max)

            except Exception as e:
                print(f"Error configurando fechas: {e}")

        else:
            self.date_info_label.setText("📅 No se detectó columna de fecha - Serie de tiempo no disponible")
            self.date_info_label.setVisible(True)
            self.btn_serie_tiempo.setEnabled(False)
            self.btn_serie_tiempo.setToolTip("No hay columna de fecha disponible")

        # Actualizar mensaje informativo
        filas, cols = df.shape
        fecha_info = f" | Fecha: {self.columna_fecha}" if self.columna_fecha else " | Sin fecha"
        self.info_label.setText(
            f"✅ Datos cargados: {filas:,} muestras, {len(columnas)} variables numéricas{fecha_info} - Listo para análisis")
        self.graph_info_label.setText(
            f"✅ {len(columnas)} variables disponibles - Configure las secciones correspondientes para generar visualizaciones")

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
        """Genera diagrama de dispersión con visualización completa y optimizada"""
        if self.df is None:
            self.mostrar_mensaje_grafico("⚠️ No hay datos cargados")
            return

        x = self.combo_x.currentText()
        y = self.combo_y.currentText()

        if not x or not y:
            self.mostrar_mensaje_grafico("⚠️ Seleccione variables X e Y para dispersión")
            return

        if x == y:
            self.mostrar_mensaje_grafico("⚠️ Seleccione variables diferentes para X e Y")
            return

        try:
            self.graph_info_label.setText(f"🎯 Generando dispersión: {x} vs {y}...")

            diagrama_dispersion(self.df, x, y)
            pixmap = QPixmap(obtener_ruta_imagen())

            if not pixmap.isNull():
                # Calcular tamaño óptimo para visualización completa
                max_width = 1200
                max_height = 800

                # Escalar pixmap si es necesario para que se vea completo
                if pixmap.width() > max_width or pixmap.height() > max_height:
                    pixmap = pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                # Configurar el label para mostrar imagen completa
                self.label_grafica.setPixmap(pixmap)
                self.label_grafica.setText("")
                self.label_grafica.setAlignment(Qt.AlignCenter)

                # Ajustar tamaño del label para acomodar la imagen con margen
                margin = 40
                label_width = pixmap.width() + margin
                label_height = pixmap.height() + margin

                self.label_grafica.setFixedSize(label_width, label_height)

                # Ajustar contenedor para asegurar visibilidad completa
                container_width = max(label_width + 60, 1000)
                container_height = max(label_height + 60, 700)
                self.image_container.setMinimumSize(container_width, container_height)

                # Actualizar información con detalles de optimización
                original_info = f"Original: {QPixmap(obtener_ruta_imagen()).width()}×{QPixmap(obtener_ruta_imagen()).height()}px"
                display_info = f"Mostrado: {pixmap.width()}×{pixmap.height()}px"

                self.graph_info_label.setText(
                    f"✅ Dispersión: {x} vs {y} | {original_info} | {display_info} | Visualización optimizada")

                # Cambiar a pestaña de gráficos
                self.tabs.setCurrentIndex(1)
            else:
                self.mostrar_mensaje_grafico("❌ Error al cargar el gráfico generado")

        except Exception as e:
            self.mostrar_mensaje_grafico(f"❌ Error al generar dispersión: {str(e)}")

    def mostrar_serie_tiempo(self):
        """Genera serie de tiempo con visualización completa y optimizada"""
        if self.df is None:
            self.mostrar_mensaje_grafico("⚠️ No hay datos cargados")
            return

        if not self.columna_fecha:
            self.mostrar_mensaje_grafico("⚠️ No hay columna de fecha detectada")
            return

        variable = self.combo_time_var.currentText()
        if not variable:
            self.mostrar_mensaje_grafico("⚠️ Seleccione variable para la serie temporal")
            return

        try:
            self.graph_info_label.setText(f"📅 Generando serie temporal: {variable} por {self.columna_fecha}...")

            # Obtener fechas seleccionadas
            fecha_desde = self.date_from.date().toPyDate()
            fecha_hasta = self.date_to.date().toPyDate()

            # Filtrar DataFrame por rango de fechas
            df_filtrado = self.df.copy()
            df_filtrado[self.columna_fecha] = pd.to_datetime(df_filtrado[self.columna_fecha])

            mask = (df_filtrado[self.columna_fecha].dt.date >= fecha_desde) & \
                   (df_filtrado[self.columna_fecha].dt.date <= fecha_hasta)
            df_filtrado = df_filtrado[mask]

            if df_filtrado.empty:
                self.mostrar_mensaje_grafico("⚠️ No hay datos en el rango de fechas seleccionado")
                return

            # Generar serie temporal con datos filtrados
            serie_tiempo(df_filtrado, self.columna_fecha, variable)
            pixmap = QPixmap(obtener_ruta_imagen())

            if not pixmap.isNull():
                # Calcular tamaño óptimo para visualización completa
                max_width = 1200
                max_height = 800

                # Escalar pixmap si es necesario para que se vea completo
                if pixmap.width() > max_width or pixmap.height() > max_height:
                    pixmap = pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                # Configurar el label para mostrar imagen completa
                self.label_grafica.setPixmap(pixmap)
                self.label_grafica.setText("")
                self.label_grafica.setAlignment(Qt.AlignCenter)

                # Ajustar tamaño del label para acomodar la imagen con margen
                margin = 40
                label_width = pixmap.width() + margin
                label_height = pixmap.height() + margin

                self.label_grafica.setFixedSize(label_width, label_height)

                # Ajustar contenedor para asegurar visibilidad completa
                container_width = max(label_width + 60, 1000)
                container_height = max(label_height + 60, 700)
                self.image_container.setMinimumSize(container_width, container_height)

                # Actualizar información con detalles completos
                original_info = f"Original: {QPixmap(obtener_ruta_imagen()).width()}×{QPixmap(obtener_ruta_imagen()).height()}px"
                display_info = f"Mostrado: {pixmap.width()}×{pixmap.height()}px"

                self.graph_info_label.setText(
                    f"✅ Serie Temporal: {variable} | Período: {fecha_desde} a {fecha_hasta} | "
                    f"Puntos: {len(df_filtrado)} | {original_info} | {display_info} | Visualización optimizada")

                # Cambiar a pestaña de gráficos
                self.tabs.setCurrentIndex(1)
            else:
                self.mostrar_mensaje_grafico("❌ Error al cargar la serie temporal generada")

        except Exception as e:
            self.mostrar_mensaje_grafico(f"❌ Error al generar serie temporal: {str(e)}")

    def mostrar_mensaje_grafico(self, mensaje):
        """Muestra mensaje en área de gráficos y restaura configuración por defecto"""
        self.label_grafica.clear()
        self.label_grafica.setPixmap(QPixmap())  # Limpiar cualquier imagen

        # Configurar mensaje con estilo mejorado
        self.label_grafica.setText(f"""
🎨 ÁREA DE VISUALIZACIÓN

{mensaje}

┌─ INSTRUCCIONES ──────────────────────────────────────────────┐
│                                                              │
│  📊 Para generar gráficos:                                   │
│     1. Configure las variables en el Panel de Control       │
│     2. Presione el botón correspondiente                     │
│     3. La gráfica aparecerá completamente visible aquí      │
│                                                              │
│  🔍 Características:                                         │
│     • Visualización completa y centrada                     │
│     • Escala automática para máxima claridad                │
│     • Scroll disponible para gráficos grandes               │
│     • Información detallada en tiempo real                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘

✨ Las gráficas se optimizan automáticamente para máxima claridad
        """)

        self.label_grafica.setAlignment(Qt.AlignCenter)
        self.label_grafica.setWordWrap(True)

        # Restaurar tamaño por defecto
        self.label_grafica.setFixedSize(1000, 700)
        self.image_container.setMinimumSize(1000, 700)

        # Actualizar información
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

            # Información sobre fecha si está disponible
            if self.columna_fecha:
                interpretacion.append(f"📅 ANÁLISIS TEMPORAL DISPONIBLE:")
                interpretacion.append(f"   • Columna de fecha detectada: {self.columna_fecha}")
                interpretacion.append("   • Use 'Serie de Tiempo' para análisis temporal")
                interpretacion.append("   • Configure períodos específicos para análisis dirigido")
                interpretacion.append("   • Identifique tendencias estacionales y patrones temporales")
                interpretacion.append("")

            # Recomendaciones mejoradas
            interpretacion.append("📊 RECOMENDACIONES TÉCNICAS:")
            interpretacion.append("")

            if correlaciones_fuertes:
                interpretacion.append("• INVESTIGAR correlaciones fuertes encontradas:")
                interpretacion.append("  - Generar diagramas de dispersión para visualizar")
                interpretacion.append("  - Verificar si son relaciones causales o espurias")
                interpretacion.append("  - Considerar variables de confusión")
                interpretacion.append("  - Validar con conocimiento del dominio")
                interpretacion.append("")

            if self.columna_fecha:
                interpretacion.append("• ANÁLISIS TEMPORAL recomendado:")
                interpretacion.append("  - Generar series temporales para variables críticas")
                interpretacion.append("  - Usar filtros de fecha para períodos específicos")
                interpretacion.append("  - Identificar tendencias estacionales")
                interpretacion.append("  - Comparar períodos antes/después de eventos")
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
            if self.columna_fecha:
                interpretacion.append("6. Analizar series temporales por períodos específicos")
                interpretacion.append("7. Identificar patrones estacionales y tendencias")
            interpretacion.append("")

            interpretacion.append("📈 NOTA METODOLÓGICA:")
            interpretacion.append(f"• Análisis realizado: {tipo_analisis}")
            interpretacion.append(f"• Tipo de relación detectada: {tipo_relacion}")
            if self.columna_fecha:
                interpretacion.append(f"• Columna temporal: {self.columna_fecha}")
                interpretacion.append("• Filtrado temporal disponible para análisis dirigido")
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
            if any(term in var_lower for term in
                   ['temperatura', 'turbidez', 'color', 'olor', 'sabor', 'wt', 'et', 'tbd']):
                categorias['Físicas'].append(var)

            # Variables químicas
            elif any(term in var_lower for term in
                     ['ph', 'oxigeno', 'dbo', 'dqo', 'conductividad', 'alcalinidad', 'do', 'bod', 'cod', 'alc']):
                categorias['Químicas'].append(var)

            # Variables biológicas
            elif any(term in var_lower for term in ['coliform', 'bacteria', 'algas', 'microorg', 'fc', 'tc']):
                categorias['Biológicas'].append(var)

            # Iones y nutrientes
            elif any(term in var_lower for term in
                     ['nitrato', 'fosfato', 'sulfato', 'cloruro', 'hierro', 'manganese', 'no3', 'no2', 'nh3', 'tp',
                      'tn', 'tkn']):
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
            elif (('oxigeno' in var1_lower or 'do' in var1_lower) and any(
                    cont in var2_lower for cont in ['coliform', 'dbo', 'dqo', 'fc', 'tc', 'bod', 'cod'])) or \
                    (('oxigeno' in var2_lower or 'do' in var2_lower) and any(
                        cont in var1_lower for cont in ['coliform', 'dbo', 'dqo', 'fc', 'tc', 'bod', 'cod'])):
                if corr > 0:
                    alertas.append(f"{var1}-{var2}: Oxígeno correlacionado positivamente con contaminantes")

            # Correlaciones extremadamente altas (posible redundancia)
            elif abs(corr) > 0.95:
                alertas.append(f"{var1}-{var2}: Correlación extrema ({corr:.3f}) - posible redundancia")

        return alertas

    def mostrar_ayuda(self):
        """Muestra guía completa de interpretación"""
        ayuda_texto = """
🔍 GUÍA COMPLETA - ANÁLISIS BIVARIADO REORGANIZADO

════════════════════════════════════════════════════════════════════

🆕 NUEVAS CARACTERÍSTICAS

📱 INTERFAZ REORGANIZADA:
   • Separación clara entre Dispersión y Serie de Tiempo
   • Variables X e Y específicas para dispersión
   • Variable única para series temporales
   • Filtrado por períodos de tiempo personalizables

📅 CONTROL TEMPORAL AVANZADO:
   • Detección automática de columnas de fecha
   • Configuración de rango basada en datos reales
   • Filtrado preciso por períodos específicos
   • Análisis dirigido por ventanas temporales

🖼️ VISUALIZACIÓN MEJORADA:
   • Gráficos en resolución completa optimizada
   • Información detallada de dimensiones
   • Scroll bidireccional mejorado
   • Padding automático para mejor visualización

════════════════════════════════════════════════════════════════════

📋 NAVEGACIÓN POR SECCIONES

📊 ANÁLISIS DE CORRELACIÓN:
   • Pearson: Para relaciones lineales
   • Spearman: Para relaciones monótonas
   • Tabla interactiva con código de colores
   • Interpretación automática específica

🎯 DIAGRAMA DE DISPERSIÓN:
   • Variable X: Eje horizontal
   • Variable Y: Eje vertical
   • Ideal para visualizar correlaciones
   • Identificación de patrones y outliers

📈 SERIE DE TIEMPO:
   • Variable única para análisis temporal
   • Período configurable (Desde - Hasta)
   • Filtrado automático por fechas
   • Análisis de tendencias estacionales

════════════════════════════════════════════════════════════════════

📊 PESTAÑAS DE RESULTADOS

📊 CORRELACIONES:
   • Tabla profesional con colores intuitivos
   • Verde: Correlaciones positivas fuertes (≥0.7)
   • Rojo: Correlaciones negativas fuertes (≤-0.7)
   • Amarillo: Correlaciones positivas moderadas (0.3-0.7)
   • Magenta: Correlaciones negativas moderadas (-0.7 a -0.3)
   • Scroll independiente para tablas grandes

📈 VISUALIZACIONES:
   • Gráficos en resolución HD completa
   • Información de dimensiones en tiempo real
   • Navegación fluida con scroll optimizado
   • Área expandible según contenido

🧠 INTERPRETACIÓN:
   • Análisis automático de correlaciones
   • Detección de variables de calidad del agua
   • Recomendaciones técnicas específicas
   • Alertas sobre correlaciones inusuales

════════════════════════════════════════════════════════════════════

📅 ANÁLISIS TEMPORAL MEJORADO

🔍 CONFIGURACIÓN DE PERÍODOS:
   • Fechas configuradas automáticamente según datos
   • Rango completo disponible desde datos cargados
   • Selección precisa de períodos de interés
   • Validación automática de rangos

📈 CASOS DE USO TEMPORAL:
   • Análisis estacional (verano vs invierno)
   • Eventos específicos (antes/después)
   • Tendencias a largo plazo
   • Comparación entre períodos

⚠️ VALIDACIONES TEMPORALES:
   • Verificación de datos en el rango seleccionado
   • Alerta si no hay datos en el período
   • Conteo de puntos disponibles
   • Información de cobertura temporal

════════════════════════════════════════════════════════════════════

🌊 CORRELACIONES EN CALIDAD DEL AGUA

✅ VARIABLES FÍSICAS RECONOCIDAS:
   • WT (Water Temperature), ET (Environmental Temp)
   • TBD (Turbidity), TSS (Total Suspended Solids)
   • TS (Total Solids), Color, Olor

✅ VARIABLES QUÍMICAS RECONOCIDAS:
   • pH, DO (Dissolved Oxygen)
   • BOD5, COD (Demanda bioquímica/química de oxígeno)
   • ALC (Alkalinity), CTD (Conductivity)

✅ VARIABLES BIOLÓGICAS RECONOCIDAS:
   • FC (Fecal Coliforms), TC (Total Coliforms)
   • Indicadores microbiológicos diversos

✅ NUTRIENTES E IONES RECONOCIDOS:
   • NO3 (Nitratos), NO2 (Nitritos), N_NH3 (Amonio)
   • TP (Fósforo Total), TN (Nitrógeno Total)
   • TKN (Nitrógeno Kjeldahl Total)

✅ ÍNDICES DE CALIDAD:
   • WQI_IDEAM_6V, WQI_IDEAM_7V
   • WQI_NSF_9V
   • Classifications automáticas

════════════════════════════════════════════════════════════════════

🎯 FLUJO DE TRABAJO OPTIMIZADO

1. 📊 ANÁLISIS EXPLORATORIO:
   • Comience con correlaciones Spearman (más robusta)
   • Identifique variables con correlaciones significativas
   • Use la tabla de colores para interpretación rápida

2. 🎯 VISUALIZACIÓN DE DISPERSIÓN:
   • Seleccione variables X e Y correlacionadas
   • Genere diagramas para correlaciones > 0.5
   • Identifique patrones, outliers y tendencias

3. 📈 ANÁLISIS TEMPORAL:
   • Seleccione variable crítica para monitoreo
   • Configure período de interés específico
   • Analice tendencias y patrones estacionales
   • Compare períodos antes/después de eventos

4. 🧠 INTERPRETACIÓN INTEGRAL:
   • Lea el análisis automático completo
   • Compare con conocimiento del dominio
   • Identifique alertas y recomendaciones
   • Planifique acciones correctivas

5. 🔧 IMPLEMENTACIÓN:
   • Siga recomendaciones específicas generadas
   • Ajuste protocolos de monitoreo
   • Implemente vigilancia continua
   • Documente hallazgos y acciones

════════════════════════════════════════════════════════════════════

💡 CONSEJOS PRÁCTICOS

🔍 SELECCIÓN DE VARIABLES:
   • Para dispersión: Elija variables con correlación moderada-fuerte
   • Para series temporales: Priorice variables críticas de calidad
   • Considere variables regulatorias importantes
   • Incluya indicadores de alerta temprana

📅 CONFIGURACIÓN TEMPORAL:
   • Use períodos completos para análisis general
   • Filtre por estaciones para análisis estacional
   • Compare períodos antes/después de intervenciones
   • Identifique ventanas de mayor variabilidad

🎨 INTERPRETACIÓN VISUAL:
   • Busque patrones lineales en dispersión
   • Identifique outliers y valores atípicos
   • Observe tendencias y ciclos en series temporales
   • Compare múltiples variables críticas

⚡ OPTIMIZACIÓN DE RENDIMIENTO:
   • Los gráficos se generan en resolución HD
   • Use scroll para navegar gráficos grandes
   • La información dimensional aparece en tiempo real
   • Los filtros temporales mejoran el rendimiento

════════════════════════════════════════════════════════════════════

🚨 ALERTAS Y RECOMENDACIONES

⚠️ CORRELACIONES INUSUALES:
   • pH-metales con correlación positiva
   • Oxígeno-contaminantes con correlación positiva
   • Correlaciones extremas (>0.95) que sugieren redundancia

✅ CORRELACIONES ESPERADAS:
   • Temperatura-Oxígeno: Negativa (normal)
   • Conductividad-Sólidos: Positiva fuerte
   • pH-Alcalinidad: Positiva moderada
   • Turbidez-Oxígeno: Negativa (partículas consumen O2)

📊 MONITOREO RECOMENDADO:
   • Variables con correlaciones > 0.7: Monitoreо conjunto
   • Variables independientes: Monitoreo individual
   • Tendencias temporales significativas: Seguimiento continuo
   • Outliers recurrentes: Investigación de causas

════════════════════════════════════════════════════════════════════

✨ ¡La interfaz reorganizada permite análisis más intuitivo y dirigido!

Use cada sección para su propósito específico:
• Correlaciones → Exploración de relaciones
• Dispersión → Visualización de dos variables
• Serie Temporal → Análisis de una variable en el tiempo

¡Combine los tres enfoques para un análisis completo!
        """

        self.interpretation_area.setText(ayuda_texto)
        self.tabs.setCurrentIndex(2)  # Ir a pestaña de interpretación