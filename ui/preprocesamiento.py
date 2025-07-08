from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QComboBox, QTableWidget, QTableWidgetItem,
                             QGroupBox, QTabWidget, QScrollArea, QGridLayout,
                             QFrame, QTextEdit, QSplitter, QSizePolicy)
from PyQt5.QtGui import QPixmap, QFont
from ml.resumen_estadistico import resumen_univariable
from ml.visualizaciones import generar_boxplot, generar_histograma, generar_densidad, obtener_ruta_imagen


class Preprocesamiento(QWidget):
    cambiar_a_bivariado = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Análisis de Datos - Preprocesamiento")
        self.df = None
        self.setMinimumSize(1400, 900)
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
                font-size: 32px;
                font-weight: bold;
                color: white;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #667eea, stop:1 #764ba2);
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                margin-bottom: 25px;
            }

            /* Contenedores principales */
            .main-container {
                background-color: white;
                border: 1px solid #d1d5db;
                border-radius: 12px;
                padding: 25px;
                margin: 15px 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }

            /* Panel de control superior */
            .control-panel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #ffecd2, stop:1 #fcb69f);
                border: 2px solid #f59e0b;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
            }

            /* Botones principales */
            .primary-button {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4f46e5, stop:1 #7c3aed);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 15px 25px;
                font-size: 16px;
                font-weight: 700;
                min-height: 50px;
                margin: 5px;
            }

            .primary-button:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3730a3, stop:1 #581c87);
                transform: translateY(-2px);
            }

            .secondary-button {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #10b981, stop:1 #059669);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 15px 25px;
                font-size: 16px;
                font-weight: 700;
                min-height: 50px;
                margin: 5px;
            }

            .secondary-button:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #047857, stop:1 #065f46);
                transform: translateY(-2px);
            }

            .danger-button {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ef4444, stop:1 #dc2626);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 15px 25px;
                font-size: 16px;
                font-weight: 700;
                min-height: 50px;
                margin: 5px;
            }

            .danger-button:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #dc2626, stop:1 #b91c1c);
                transform: translateY(-2px);
            }

            /* ComboBox mejorado */
            QComboBox {
                background-color: white;
                border: 3px solid #d1d5db;
                border-radius: 8px;
                padding: 12px 15px;
                font-size: 16px;
                font-weight: 600;
                min-height: 25px;
                min-width: 250px;
            }

            QComboBox:focus {
                border-color: #4f46e5;
            }

            QComboBox::drop-down {
                border: none;
                width: 35px;
            }

            QComboBox::down-arrow {
                image: none;
                border-left: 6px solid transparent;
                border-right: 6px solid transparent;
                border-top: 6px solid #4a5568;
            }

            /* Pestañas modernas */
            QTabWidget::pane {
                border: 2px solid #e5e7eb;
                background-color: white;
                border-radius: 12px;
                padding: 15px;
            }

            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f3f4f6, stop:1 #e5e7eb);
                border: 2px solid #d1d5db;
                padding: 15px 30px;
                margin-right: 3px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                font-weight: 700;
                font-size: 15px;
                color: #4b5563;
                min-width: 150px;
            }

            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 white, stop:1 #f8fafc);
                border-bottom-color: white;
                color: #1f2937;
            }

            QTabBar::tab:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e5e7eb, stop:1 #d1d5db);
            }

            /* Contenedores de gráficos mejorados */
            .graph-container {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 white, stop:1 #fafbfc);
                border: 2px solid #e5e7eb;
                border-radius: 15px;
                padding: 20px;
                margin: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }

            .graph-title {
                font-size: 18px;
                font-weight: bold;
                color: #1f2937;
                text-align: center;
                margin-bottom: 15px;
                padding: 10px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ddd6fe, stop:1 #c7d2fe);
                border-radius: 8px;
            }

            /* Área de estadísticas */
            .stats-section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f0f9ff, stop:1 #e0f2fe);
                border: 2px solid #0ea5e9;
                border-radius: 12px;
                padding: 25px;
            }

            .section-title {
                font-size: 22px;
                font-weight: bold;
                color: #1e40af;
                margin-bottom: 20px;
                text-align: center;
                padding: 12px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #dbeafe, stop:1 #bfdbfe);
                border-radius: 8px;
            }

            /* Tabla mejorada */
            QTableWidget {
                background-color: white;
                border: 2px solid #e5e7eb;
                border-radius: 10px;
                gridline-color: #f3f4f6;
                font-size: 14px;
                selection-background-color: #dbeafe;
            }

            QTableWidget::item {
                padding: 15px 10px;
                border-bottom: 1px solid #f3f4f6;
                text-align: center;
            }

            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8fafc, stop:1 #e2e8f0);
                border: none;
                border-bottom: 3px solid #e5e7eb;
                border-right: 1px solid #e5e7eb;
                padding: 15px 10px;
                font-weight: bold;
                font-size: 14px;
                color: #374151;
                text-align: center;
            }

            /* Área de recomendaciones */
            .recommendations-area {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f0fdf4, stop:1 #dcfce7);
                border: 2px solid #22c55e;
                border-radius: 12px;
                padding: 25px;
                font-size: 14px;
                line-height: 1.6;
            }

            /* Placeholder para gráficos */
            .graph-placeholder {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8fafc, stop:1 #e2e8f0);
                border: 3px dashed #94a3b8;
                border-radius: 12px;
                color: #475569;
                font-size: 15px;
                font-weight: 600;
                text-align: center;
                padding: 30px;
                line-height: 1.8;
            }

            /* ScrollArea mejorada */
            QScrollArea {
                border: none;
                background-color: transparent;
            }

            QScrollBar:vertical {
                background-color: #f3f4f6;
                width: 14px;
                border-radius: 7px;
                margin: 2px;
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
        """)

        # Layout principal con scroll
        main_scroll = QScrollArea()
        main_scroll.setWidgetResizable(True)
        main_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        main_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(20)

        # Header principal
        self.create_header(main_layout)

        # Panel de control
        self.create_control_panel(main_layout)

        # Área de contenido con pestañas
        self.create_content_area(main_layout)

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
        title = QLabel("🔍 Análisis Exploratorio de Datos")
        title.setProperty("class", "main-title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

    def create_control_panel(self, layout):
        control_container = QFrame()
        control_container.setProperty("class", "control-panel")

        control_layout = QHBoxLayout()
        control_layout.setSpacing(20)

        # Sección de variable
        var_section = QVBoxLayout()
        var_label = QLabel("🎯 Variable a Analizar:")
        var_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #92400e;
            margin-bottom: 10px;
        """)

        self.combo_columnas = QComboBox()
        self.combo_columnas.currentTextChanged.connect(self.on_variable_changed)

        var_section.addWidget(var_label)
        var_section.addWidget(self.combo_columnas)

        # Sección de botones
        buttons_section = QVBoxLayout()
        buttons_section.setSpacing(15)

        self.btn_generar_todo = QPushButton("🚀 Generar Análisis Completo")
        self.btn_generar_todo.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4f46e5, stop:1 #7c3aed);
                color: white;
                border: none;
                border-radius: 12px;
                padding: 20px 25px;
                font-size: 18px;
                font-weight: 700;
                min-height: 30px;
                min-width: 280px;
                text-align: center;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3730a3, stop:1 #581c87);
                transform: translateY(-3px);
                box-shadow: 0 8px 15px rgba(79, 70, 229, 0.4);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #312e81, stop:1 #4c1d95);
                transform: translateY(-1px);
            }
        """)
        self.btn_generar_todo.clicked.connect(self.generar_analisis_completo)

        self.btn_limpiar = QPushButton("🗑️ Limpiar Resultados")
        self.btn_limpiar.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f59e0b, stop:1 #d97706);
                color: white;
                border: none;
                border-radius: 12px;
                padding: 15px 25px;
                font-size: 16px;
                font-weight: 700;
                min-height: 25px;
                min-width: 280px;
                text-align: center;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #d97706, stop:1 #b45309);
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(245, 158, 11, 0.3);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #b45309, stop:1 #92400e);
                transform: translateY(0px);
            }
        """)
        self.btn_limpiar.clicked.connect(self.limpiar_resultados)

        buttons_section.addWidget(self.btn_generar_todo)
        buttons_section.addWidget(self.btn_limpiar)

        # Layout horizontal
        control_layout.addLayout(var_section, 1)
        control_layout.addStretch()
        control_layout.addLayout(buttons_section, 1)

        control_container.setLayout(control_layout)
        layout.addWidget(control_container)

    def create_content_area(self, layout):
        content_container = QFrame()
        content_container.setProperty("class", "main-container")

        content_layout = QVBoxLayout()

        # Título del área
        content_title = QLabel("📊 Resultados del Análisis")
        content_title.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #374151;
            text-align: center;
            margin-bottom: 20px;
            padding: 15px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #f3f4f6, stop:1 #e5e7eb);
            border-radius: 10px;
        """)
        content_layout.addWidget(content_title)

        # Pestañas
        self.tabs = QTabWidget()

        # Pestaña 1: Visualizaciones
        self.create_visualizations_tab()

        # Pestaña 2: Estadísticas
        self.create_statistics_tab()

        # Pestaña 3: Recomendaciones
        self.create_recommendations_tab()

        content_layout.addWidget(self.tabs)
        content_container.setLayout(content_layout)
        layout.addWidget(content_container)

    def create_visualizations_tab(self):
        viz_widget = QWidget()
        viz_layout = QVBoxLayout()
        viz_layout.setContentsMargins(15, 15, 15, 15)
        viz_layout.setSpacing(15)

        # Título de la sección
        section_title = QLabel("📈 Visualizaciones Estadísticas")
        section_title.setProperty("class", "section-title")
        viz_layout.addWidget(section_title)

        # Container para los gráficos en grid
        graphs_container = QHBoxLayout()
        graphs_container.setSpacing(20)

        # Boxplot
        self.boxplot_container = self.create_graph_container("📦 Diagrama de Caja", "boxplot")
        graphs_container.addWidget(self.boxplot_container)

        # Histograma
        self.histogram_container = self.create_graph_container("📊 Histograma", "histogram")
        graphs_container.addWidget(self.histogram_container)

        # Densidad
        self.density_container = self.create_graph_container("🌊 Gráfico de Densidad", "density")
        graphs_container.addWidget(self.density_container)

        viz_layout.addLayout(graphs_container)
        viz_widget.setLayout(viz_layout)
        self.tabs.addTab(viz_widget, "📈 Visualizaciones")

    def create_graph_container(self, title, graph_type):
        container = QFrame()
        container.setProperty("class", "graph-container")
        container.setMinimumHeight(400)
        container.setMinimumWidth(350)

        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Título del gráfico
        graph_title = QLabel(title)
        graph_title.setProperty("class", "graph-title")
        layout.addWidget(graph_title)

        # Label para la imagen
        graph_label = QLabel(f"""
🎨 Área de Visualización

Haga clic en:
'🚀 Generar Análisis Completo'

Para ver el {title.lower()}

✨ Visualización automática
📊 Calidad HD optimizada
🔍 Análisis profesional
        """)
        graph_label.setProperty("class", "graph-placeholder")
        graph_label.setMinimumHeight(280)
        graph_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        setattr(self, f"label_{graph_type}", graph_label)
        layout.addWidget(graph_label)

        container.setLayout(layout)
        return container

    def create_statistics_tab(self):
        stats_widget = QWidget()
        stats_layout = QVBoxLayout()
        stats_layout.setContentsMargins(15, 15, 15, 15)
        stats_layout.setSpacing(15)

        # Contenedor con estilo
        stats_container = QFrame()
        stats_container.setProperty("class", "stats-section")

        container_layout = QVBoxLayout()

        # Título
        title = QLabel("📋 Resumen Estadístico Detallado")
        title.setProperty("class", "section-title")
        container_layout.addWidget(title)

        # Información introductoria
        info_label = QLabel("""
📊 Este resumen incluye todas las medidas estadísticas importantes:
• Medidas de tendencia central (media, mediana, moda)
• Medidas de dispersión (desviación estándar, varianza)
• Medidas de posición (cuartiles, percentiles)
• Información sobre distribución y valores atípicos
        """)
        info_label.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            color: #374151;
            font-size: 14px;
        """)
        container_layout.addWidget(info_label)

        # Tabla de estadísticas
        self.tabla_datos = QTableWidget()
        self.tabla_datos.setMinimumHeight(400)
        container_layout.addWidget(self.tabla_datos)

        stats_container.setLayout(container_layout)
        stats_layout.addWidget(stats_container)
        stats_widget.setLayout(stats_layout)
        self.tabs.addTab(stats_widget, "📋 Estadísticas")

    def create_recommendations_tab(self):
        rec_widget = QWidget()
        rec_layout = QVBoxLayout()
        rec_layout.setContentsMargins(15, 15, 15, 15)
        rec_layout.setSpacing(15)

        # Título
        title = QLabel("💡 Recomendaciones Inteligentes")
        title.setProperty("class", "section-title")
        rec_layout.addWidget(title)

        # Información introductoria
        intro_label = QLabel("""
🧠 Sistema de Recomendaciones Automáticas:
• Análisis inteligente de la distribución de datos
• Detección automática de valores atípicos y patrones
• Recomendaciones específicas para calidad del agua
• Sugerencias de próximos pasos de análisis
        """)
        intro_label.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #ecfdf5, stop:1 #d1fae5);
            border: 2px solid #34d399;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
            color: #065f46;
            font-size: 14px;
            font-weight: 600;
        """)
        rec_layout.addWidget(intro_label)

        # Área de recomendaciones
        self.recommendations_area = QTextEdit()
        self.recommendations_area.setProperty("class", "recommendations-area")
        self.recommendations_area.setReadOnly(True)
        self.recommendations_area.setMinimumHeight(400)
        self.recommendations_area.setText("""
🎯 BIENVENIDO AL SISTEMA DE RECOMENDACIONES INTELIGENTES

Para comenzar:

1️⃣ Seleccione una variable numérica del dropdown superior
2️⃣ Haga clic en "🚀 Generar Análisis Completo"
3️⃣ El sistema analizará automáticamente sus datos

Recibirá recomendaciones sobre:
✅ Calidad y distribución de los datos
✅ Detección de valores atípicos
✅ Interpretación específica para calidad del agua
✅ Próximos pasos recomendados

¡Comience seleccionando una variable para análisis!
        """)

        rec_layout.addWidget(self.recommendations_area)
        rec_widget.setLayout(rec_layout)
        self.tabs.addTab(rec_widget, "💡 Recomendaciones")

    def create_navigation(self, layout):
        nav_container = QFrame()
        nav_container.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #f8fafc, stop:1 #e2e8f0);
                border: 2px solid #cbd5e0;
                border-radius: 15px;
                padding: 25px;
                margin-top: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
        """)

        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(30)

        # Sección izquierda - Botón de regreso
        left_section = QVBoxLayout()
        left_section.setSpacing(8)

        back_info = QLabel("🏠 Navegación Principal")
        back_info.setStyleSheet("""
            color: #dc2626;
            font-weight: 700;
            font-size: 14px;
            margin-bottom: 5px;
        """)

        self.btn_regresar = QPushButton("⬅️ Volver al Menú Principal")
        self.btn_regresar.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ef4444, stop:1 #dc2626);
                color: white;
                border: none;
                border-radius: 12px;
                padding: 18px 30px;
                font-size: 16px;
                font-weight: 700;
                min-height: 25px;
                min-width: 220px;
                text-align: center;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #dc2626, stop:1 #b91c1c);
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(220, 38, 38, 0.3);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #b91c1c, stop:1 #991b1b);
                transform: translateY(0px);
            }
        """)

        left_section.addWidget(back_info)
        left_section.addWidget(self.btn_regresar)

        # Divisor visual
        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setFrameShadow(QFrame.Sunken)
        divider.setStyleSheet("""
            QFrame {
                background-color: #cbd5e0;
                max-width: 2px;
                margin: 10px 0;
            }
        """)

        # Sección derecha - Botón de continuación
        right_section = QVBoxLayout()
        right_section.setSpacing(8)

        continue_info = QLabel("🔗 Continuar Análisis")
        continue_info.setStyleSheet("""
            color: #059669;
            font-weight: 700;
            font-size: 14px;
            margin-bottom: 5px;
        """)

        continue_description = QLabel("Explore relaciones entre variables")
        continue_description.setStyleSheet("""
            color: #6b7280;
            font-size: 13px;
            font-style: italic;
            margin-bottom: 10px;
        """)

        self.btn_bivariable = QPushButton("➡️ Ir a Análisis Bivariado")
        self.btn_bivariable.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #10b981, stop:1 #059669);
                color: white;
                border: none;
                border-radius: 12px;
                padding: 18px 30px;
                font-size: 16px;
                font-weight: 700;
                min-height: 25px;
                min-width: 220px;
                text-align: center;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #047857, stop:1 #065f46);
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(16, 185, 129, 0.3);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #065f46, stop:1 #064e3b);
                transform: translateY(0px);
            }
        """)
        self.btn_bivariable.clicked.connect(self.cambiar_a_bivariado.emit)

        right_section.addWidget(continue_info)
        right_section.addWidget(continue_description)
        right_section.addWidget(self.btn_bivariable)

        # Agregar secciones al layout principal
        nav_layout.addLayout(left_section)
        nav_layout.addWidget(divider)
        nav_layout.addLayout(right_section)

        nav_container.setLayout(nav_layout)
        layout.addWidget(nav_container)

    def cargar_dataframe(self, df):
        self.df = df
        columnas = df.select_dtypes(include='number').columns.tolist()
        self.combo_columnas.clear()
        self.combo_columnas.addItems(columnas)

    def on_variable_changed(self):
        """Se ejecuta cuando cambia la variable seleccionada"""
        if self.df is not None:
            self.limpiar_graficos()

    def generar_analisis_completo(self):
        """Genera todas las visualizaciones y estadísticas de una vez"""
        if self.df is None:
            return

        col = self.combo_columnas.currentText()
        if not col:
            return

        # Generar estadísticas
        self.mostrar_resumen()

        # Generar gráficos
        self.mostrar_boxplot()
        self.mostrar_histograma()
        self.mostrar_densidad()

        # Generar recomendaciones
        self.generar_recomendaciones()

        # Cambiar a la pestaña de visualizaciones para mostrar resultados
        self.tabs.setCurrentIndex(0)

    def limpiar_resultados(self):
        """Limpia todos los resultados"""
        self.limpiar_graficos()
        self.tabla_datos.clear()
        self.recommendations_area.setText("""
🎯 SISTEMA DE RECOMENDACIONES LISTO

Seleccione una variable y genere el análisis completo para ver las recomendaciones automáticas.

El sistema está preparado para analizar sus datos y proporcionar insights valiosos.
        """)

    def limpiar_graficos(self):
        """Limpia solo los gráficos"""
        placeholders = {
            'boxplot': """🎨 Diagrama de Caja

Seleccione variable y genere análisis

📦 Muestra distribución
📊 Identifica outliers
📈 Visualiza cuartiles""",
            'histogram': """🎨 Histograma

Seleccione variable y genere análisis

📊 Muestra frecuencias
📈 Identifica distribución
🔍 Revela patrones""",
            'density': """🎨 Gráfico de Densidad

Seleccione variable y genere análisis

🌊 Muestra densidad
📈 Curva suavizada
📊 Distribución continua"""
        }

        for graph_type, placeholder_text in placeholders.items():
            label = getattr(self, f"label_{graph_type}")
            label.clear()
            label.setText(placeholder_text)
            label.setProperty("class", "graph-placeholder")

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
                item = QTableWidgetItem(str(round(val, 4) if isinstance(val, float) else val))
                item.setTextAlignment(Qt.AlignCenter)
                self.tabla_datos.setItem(i, j, item)

        # Ajustar columnas
        self.tabla_datos.resizeColumnsToContents()

    def mostrar_boxplot(self):
        col = self.combo_columnas.currentText()
        if col:
            generar_boxplot(self.df, col)
            pixmap = QPixmap(obtener_ruta_imagen())
            if not pixmap.isNull():
                # Escalar imagen para ajustarse al contenedor
                scaled_pixmap = pixmap.scaled(
                    self.label_boxplot.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.label_boxplot.setPixmap(scaled_pixmap)
                self.label_boxplot.setText("")

    def mostrar_histograma(self):
        col = self.combo_columnas.currentText()
        if col:
            generar_histograma(self.df, col)
            pixmap = QPixmap(obtener_ruta_imagen())
            if not pixmap.isNull():
                # Escalar imagen para ajustarse al contenedor
                scaled_pixmap = pixmap.scaled(
                    self.label_histogram.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.label_histogram.setPixmap(scaled_pixmap)
                self.label_histogram.setText("")

    def mostrar_densidad(self):
        col = self.combo_columnas.currentText()
        if col:
            generar_densidad(self.df, col)
            pixmap = QPixmap(obtener_ruta_imagen())
            if not pixmap.isNull():
                # Escalar imagen para ajustarse al contenedor
                scaled_pixmap = pixmap.scaled(
                    self.label_density.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.label_density.setPixmap(scaled_pixmap)
                self.label_density.setText("")

    def generar_recomendaciones(self):
        """Genera recomendaciones automáticas basadas en los datos"""
        if self.df is None:
            return

        col = self.combo_columnas.currentText()
        if not col:
            return

        try:
            serie = self.df[col].dropna()

            # Estadísticas básicas
            media = serie.mean()
            mediana = serie.median()
            std = serie.std()
            q1 = serie.quantile(0.25)
            q3 = serie.quantile(0.75)
            iqr = q3 - q1

            # Detección de outliers
            outliers_lower = serie < (q1 - 1.5 * iqr)
            outliers_upper = serie > (q3 + 1.5 * iqr)
            total_outliers = outliers_lower.sum() + outliers_upper.sum()

            # Generar recomendaciones con formato mejorado
            recomendaciones = []

            recomendaciones.append(f"🔍 ANÁLISIS INTELIGENTE DE LA VARIABLE: {col.upper()}")
            recomendaciones.append("═" * 70)
            recomendaciones.append("")

            # Resumen ejecutivo
            recomendaciones.append("📋 RESUMEN EJECUTIVO:")
            recomendaciones.append(f"   • Variable analizada: {col}")
            recomendaciones.append(f"   • Número de observaciones: {len(serie):,}")
            recomendaciones.append(f"   • Valores válidos: {len(serie):,} ({len(serie) / len(self.df) * 100:.1f}%)")
            recomendaciones.append(f"   • Media: {media:.4f}")
            recomendaciones.append(f"   • Mediana: {mediana:.4f}")
            recomendaciones.append(f"   • Desviación estándar: {std:.4f}")
            recomendaciones.append("")

            # Análisis de distribución mejorado
            asimetria = abs(media - mediana) / std if std > 0 else 0
            recomendaciones.append("📊 ANÁLISIS DE DISTRIBUCIÓN:")

            if asimetria < 0.1:
                recomendaciones.append(
                    "   ✅ DISTRIBUCIÓN NORMAL: Los datos siguen una distribución aproximadamente normal")
                recomendaciones.append("      → Excelente para análisis estadísticos paramétricos")
                recomendaciones.append("      → Intervalos de confianza y pruebas t son apropiadas")
                recomendaciones.append("      → Los datos son simétricos y bien distribuidos")
            elif asimetria < 0.5:
                recomendaciones.append("   ⚠️  LIGERA ASIMETRÍA: Los datos muestran una asimetría moderada")
                recomendaciones.append("      → Considerar transformaciones logarítmicas si es necesario")
                recomendaciones.append("      → Métodos robustos pueden ser más apropiados")
                recomendaciones.append("      → Verificar si la asimetría es significativa")
            else:
                recomendaciones.append("   🚨 DISTRIBUCIÓN ASIMÉTRICA: Los datos muestran fuerte asimetría")
                recomendaciones.append("      → Recomendado: Usar métodos no paramétricos")
                recomendaciones.append("      → Considerar transformaciones (log, sqrt, Box-Cox)")
                recomendaciones.append("      → La mediana es más representativa que la media")

            recomendaciones.append("")

            # Análisis de outliers mejorado
            outlier_percentage = (total_outliers / len(serie)) * 100
            recomendaciones.append("🔍 ANÁLISIS DE VALORES ATÍPICOS:")

            if total_outliers == 0:
                recomendaciones.append("   ✅ SIN OUTLIERS: No se detectaron valores atípicos")
                recomendaciones.append("      → Excelente calidad de datos")
                recomendaciones.append("      → Los datos son consistentes y confiables")
                recomendaciones.append("      → No se requiere limpieza adicional")
            elif outlier_percentage <= 2:
                recomendaciones.append(
                    f"   ✅ OUTLIERS MÍNIMOS: {total_outliers} outliers detectados ({outlier_percentage:.1f}%)")
                recomendaciones.append("      → Cantidad normal para datos ambientales")
                recomendaciones.append("      → Revisar si representan eventos reales o errores")
                recomendaciones.append("      → Mantener en análisis a menos que sean errores evidentes")
            elif outlier_percentage <= 5:
                recomendaciones.append(
                    f"   ⚠️  OUTLIERS MODERADOS: {total_outliers} outliers detectados ({outlier_percentage:.1f}%)")
                recomendaciones.append("      → Investigar causas específicas")
                recomendaciones.append("      → Pueden representar eventos extremos reales")
                recomendaciones.append("      → Considerar análisis separado de outliers")
            else:
                recomendaciones.append(
                    f"   🚨 OUTLIERS ABUNDANTES: {total_outliers} outliers detectados ({outlier_percentage:.1f}%)")
                recomendaciones.append("      → Requiere investigación exhaustiva")
                recomendaciones.append("      → Posibles problemas en recolección de datos")
                recomendaciones.append("      → Considerar métodos robustos para análisis")

            recomendaciones.append("")

            # Análisis de variabilidad mejorado
            cv = (std / media) * 100 if media != 0 else 0
            recomendaciones.append("📈 ANÁLISIS DE VARIABILIDAD:")

            if cv < 10:
                recomendaciones.append(f"   ✅ BAJA VARIABILIDAD: CV = {cv:.1f}% (Excelente)")
                recomendaciones.append("      → Datos muy consistentes y precisos")
                recomendaciones.append("      → Alta confiabilidad en las mediciones")
                recomendaciones.append("      → Proceso de medición muy controlado")
            elif cv < 20:
                recomendaciones.append(f"   ✅ VARIABILIDAD ACEPTABLE: CV = {cv:.1f}% (Bueno)")
                recomendaciones.append("      → Variabilidad normal para datos ambientales")
                recomendaciones.append("      → Proceso de medición controlado")
                recomendaciones.append("      → No se requieren ajustes inmediatos")
            elif cv < 35:
                recomendaciones.append(f"   ⚠️  VARIABILIDAD MODERADA: CV = {cv:.1f}% (Aceptable)")
                recomendaciones.append("      → Revisar protocolos de medición")
                recomendaciones.append("      → Considerar factores externos que afecten variabilidad")
                recomendaciones.append("      → Aumentar frecuencia de calibraciones")
            else:
                recomendaciones.append(f"   🚨 ALTA VARIABILIDAD: CV = {cv:.1f}% (Crítico)")
                recomendaciones.append("      → Urgente: Revisar todo el proceso de medición")
                recomendaciones.append("      → Posibles problemas en equipos o procedimientos")
                recomendaciones.append("      → Implementar controles de calidad más estrictos")

            recomendaciones.append("")

            # Recomendaciones específicas para calidad del agua mejoradas
            recomendaciones.append("🌊 RECOMENDACIONES ESPECÍFICAS PARA CALIDAD DEL AGUA:")
            recomendaciones.append("")

            variable_lower = col.lower()

            if 'ph' in variable_lower:
                if 6.5 <= media <= 8.5:
                    recomendaciones.append("   ✅ pH: EXCELENTE - Dentro del rango óptimo (6.5-8.5)")
                    recomendaciones.append("      → Cumple estándares de agua potable")
                    recomendaciones.append("      → Mantener monitoreo rutinario")
                elif 6.0 <= media <= 9.0:
                    recomendaciones.append("   ⚠️  pH: ACEPTABLE - En rango amplio pero vigilar")
                    recomendaciones.append("      → Monitorear tendencias")
                    recomendaciones.append("      → Evaluar sistemas de neutralización")
                else:
                    recomendaciones.append("   🚨 pH: CRÍTICO - Fuera de rangos seguros")
                    recomendaciones.append("      → ACCIÓN INMEDIATA: Implementar corrección de pH")
                    recomendaciones.append("      → Evaluar procesos de tratamiento")

            elif any(x in variable_lower for x in ['oxigeno', 'do', 'od']):
                if media >= 6:
                    recomendaciones.append("   ✅ OXÍGENO DISUELTO: EXCELENTE (≥6 mg/L)")
                    recomendaciones.append("      → Ecosistema acuático saludable")
                    recomendaciones.append("      → Mantener condiciones actuales")
                elif media >= 4:
                    recomendaciones.append("   ⚠️  OXÍGENO DISUELTO: ACEPTABLE (4-6 mg/L)")
                    recomendaciones.append("      → Monitorear carga orgánica")
                    recomendaciones.append("      → Evaluar aireación si es necesario")
                else:
                    recomendaciones.append("   🚨 OXÍGENO DISUELTO: CRÍTICO (<4 mg/L)")
                    recomendaciones.append("      → URGENTE: Riesgo para vida acuática")
                    recomendaciones.append("      → Investigar fuentes de contaminación orgánica")

            elif any(x in variable_lower for x in ['turbidez', 'turbiedad', 'tbd']):
                if media <= 1:
                    recomendaciones.append("   ✅ TURBIDEZ: EXCELENTE (≤1 NTU)")
                    recomendaciones.append("      → Agua cristalina, excelente calidad")
                elif media <= 4:
                    recomendaciones.append("   ✅ TURBIDEZ: BUENA (1-4 NTU)")
                    recomendaciones.append("      → Dentro de estándares de agua potable")
                elif media <= 10:
                    recomendaciones.append("   ⚠️  TURBIDEZ: MODERADA (4-10 NTU)")
                    recomendaciones.append("      → Evaluar sistemas de filtración")
                else:
                    recomendaciones.append("   🚨 TURBIDEZ: ALTA (>10 NTU)")
                    recomendaciones.append("      → Revisar procesos de clarificación")

            elif any(x in variable_lower for x in ['conductividad', 'ctd']):
                if media <= 300:
                    recomendaciones.append("   ✅ CONDUCTIVIDAD: BAJA - Agua de baja mineralización")
                elif media <= 1000:
                    recomendaciones.append("   ✅ CONDUCTIVIDAD: NORMAL - Mineralización adecuada")
                else:
                    recomendaciones.append("   ⚠️  CONDUCTIVIDAD: ALTA - Evaluar contenido iónico")

            elif any(x in variable_lower for x in ['coliform', 'fc', 'tc']):
                if media == 0:
                    recomendaciones.append("   ✅ COLIFORMES: AUSENTES - Excelente calidad microbiológica")
                elif media <= 10:
                    recomendaciones.append("   ⚠️  COLIFORMES: DETECTABLES - Monitorear desinfección")
                else:
                    recomendaciones.append("   🚨 COLIFORMES: ELEVADOS - Riesgo sanitario alto")

            elif any(x in variable_lower for x in ['temperatura', 'temp', 'wt']):
                if 15 <= media <= 25:
                    recomendaciones.append("   ✅ TEMPERATURA: ÓPTIMA para ecosistemas acuáticos")
                elif media > 30:
                    recomendaciones.append("   🚨 TEMPERATURA: ELEVADA - Riesgo para vida acuática")
                elif media < 10:
                    recomendaciones.append("   ⚠️  TEMPERATURA: BAJA - Monitorear ecosistema")

            recomendaciones.append("")

            # Próximos pasos mejorados
            recomendaciones.append("🎯 PRÓXIMOS PASOS RECOMENDADOS:")
            recomendaciones.append("")
            recomendaciones.append("   1️⃣ ANÁLISIS BIVARIADO:")
            recomendaciones.append("      • Investigar correlaciones con otras variables")
            recomendaciones.append("      • Generar gráficos de dispersión")
            recomendaciones.append("      • Identificar relaciones causa-efecto")
            recomendaciones.append("")
            recomendaciones.append("   2️⃣ ANÁLISIS TEMPORAL:")
            recomendaciones.append("      • Evaluar tendencias en el tiempo")
            recomendaciones.append("      • Identificar patrones estacionales")
            recomendaciones.append("      • Detectar cambios significativos")
            recomendaciones.append("")
            recomendaciones.append("   3️⃣ MONITOREO CONTINUO:")
            recomendaciones.append("      • Establecer frecuencia de muestreo óptima")
            recomendaciones.append("      • Implementar alertas automáticas")
            recomendaciones.append("      • Crear indicadores de calidad")
            recomendaciones.append("")
            recomendaciones.append("   4️⃣ VALIDACIÓN Y CONTROL:")
            recomendaciones.append("      • Comparar con normativas vigentes")
            recomendaciones.append("      • Verificar calibración de equipos")
            recomendaciones.append("      • Documentar procedimientos")

            recomendaciones.append("")
            recomendaciones.append("💡 NOTA: Use el botón '➡️ Ir a Análisis Bivariado' para continuar")
            recomendaciones.append("    con el análisis de relaciones entre variables.")

            self.recommendations_area.setText("\n".join(recomendaciones))

        except Exception as e:
            self.recommendations_area.setText(f"""
🚨 ERROR AL GENERAR RECOMENDACIONES

Ha ocurrido un error durante el análisis: {str(e)}

🔧 SOLUCIONES SUGERIDAS:
• Verifique que la variable seleccionada contenga datos numéricos
• Asegúrese de que hay suficientes datos para el análisis
• Intente seleccionar una variable diferente
• Contacte al administrador si el problema persiste

💡 El análisis estadístico y las visualizaciones pueden estar disponibles
   en las otras pestañas incluso si las recomendaciones fallan.
            """)