from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QComboBox, QTableWidget, QTableWidgetItem,
                             QGroupBox, QTabWidget, QScrollArea, QGridLayout,
                             QFrame, QTextEdit, QSplitter)
from PyQt5.QtGui import QPixmap, QFont
from ml.resumen_estadistico import resumen_univariable
from ml.visualizaciones import generar_boxplot, generar_histograma, generar_densidad, obtener_ruta_imagen


class Preprocesamiento(QWidget):
    cambiar_a_bivariado = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Análisis de Datos - Preprocesamiento")
        self.df = None
        self.setMinimumSize(1200, 800)
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #f8fafc;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
            }

            QTabWidget::pane {
                border: 1px solid #e2e8f0;
                background-color: white;
                border-radius: 8px;
            }

            QTabWidget::tab-bar {
                alignment: left;
            }

            QTabBar::tab {
                background: #f1f5f9;
                border: 1px solid #e2e8f0;
                padding: 12px 24px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: 600;
                color: #4a5568;
            }

            QTabBar::tab:selected {
                background: white;
                border-bottom-color: white;
                color: #2d3748;
            }

            QTabBar::tab:hover {
                background: #e2e8f0;
            }

            .header-title {
                font-size: 24px;
                font-weight: bold;
                color: #1a365d;
                margin: 15px 0;
                background: none;
            }

            .control-button {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4299e1, stop:1 #3182ce);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 20px;
                font-weight: 600;
                font-size: 14px;
                min-height: 20px;
            }

            .control-button:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3182ce, stop:1 #2c5282);
            }

            .control-button:pressed {
                background: #2c5282;
            }

            .secondary-button {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #68d391, stop:1 #48bb78);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 20px;
                font-weight: 600;
                font-size: 14px;
            }

            .secondary-button:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #48bb78, stop:1 #38a169);
            }

            .danger-button {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #fc8181, stop:1 #f56565);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 20px;
                font-weight: 600;
            }

            .danger-button:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f56565, stop:1 #e53e3e);
            }

            QComboBox {
                background-color: white;
                border: 2px solid #e2e8f0;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 14px;
                min-height: 20px;
            }

            QComboBox:focus {
                border-color: #4299e1;
            }

            QComboBox::drop-down {
                border: none;
                width: 30px;
            }

            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #4a5568;
            }

            .stats-group {
                background-color: white;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 20px;
                margin: 10px 0;
            }

            .stats-title {
                font-size: 18px;
                font-weight: bold;
                color: #2d3748;
                margin-bottom: 15px;
                background: none;
            }

            QTableWidget {
                background-color: white;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                gridline-color: #f1f5f9;
                font-size: 13px;
            }

            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #f1f5f9;
            }

            QHeaderView::section {
                background: #f8fafc;
                border: none;
                border-bottom: 2px solid #e2e8f0;
                padding: 12px 8px;
                font-weight: bold;
                color: #4a5568;
            }

            .graph-container {
                background-color: white;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 5px;
            }

            .graph-title {
                font-size: 16px;
                font-weight: bold;
                color: #2d3748;
                text-align: center;
                margin-bottom: 10px;
                background: none;
            }

            .recommendations-box {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f0fff4, stop:1 #f7fafc);
                border: 1px solid #68d391;
                border-radius: 8px;
                padding: 20px;
                margin: 15px 0;
            }

            .recommendations-title {
                font-size: 16px;
                font-weight: bold;
                color: #22543d;
                margin-bottom: 10px;
                background: none;
            }

            .recommendations-text {
                color: #2f855a;
                line-height: 1.6;
                background: none;
            }
        """)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Header
        self.create_header(main_layout)

        # Controles principales
        self.create_controls(main_layout)

        # Pestañas principales
        self.create_tabs(main_layout)

        # Botones de navegación
        self.create_navigation(main_layout)

        self.setLayout(main_layout)

    def create_header(self, layout):
        header_layout = QHBoxLayout()

        title = QLabel("🔍 Análisis Exploratorio de Datos")
        title.setProperty("class", "header-title")
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Selector de variable
        var_layout = QHBoxLayout()
        var_label = QLabel("Variable a analizar:")
        var_label.setStyleSheet("font-weight: bold; color: #4a5568; background: none; margin-right: 10px;")
        self.combo_columnas = QComboBox()
        self.combo_columnas.setMinimumWidth(200)
        self.combo_columnas.currentTextChanged.connect(self.on_variable_changed)

        var_layout.addWidget(var_label)
        var_layout.addWidget(self.combo_columnas)
        header_layout.addLayout(var_layout)

        layout.addLayout(header_layout)

    def create_controls(self, layout):
        controls_frame = QFrame()
        controls_frame.setStyleSheet(
            "background-color: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px;")

        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(15)

        # Botones de análisis
        self.btn_generar_todo = QPushButton("🚀 Generar Análisis Completo")
        self.btn_generar_todo.setProperty("class", "control-button")
        self.btn_generar_todo.clicked.connect(self.generar_analisis_completo)

        self.btn_limpiar = QPushButton("🗑️ Limpiar Resultados")
        self.btn_limpiar.setProperty("class", "danger-button")
        self.btn_limpiar.clicked.connect(self.limpiar_resultados)

        controls_layout.addWidget(self.btn_generar_todo)
        controls_layout.addWidget(self.btn_limpiar)
        controls_layout.addStretch()

        controls_frame.setLayout(controls_layout)
        layout.addWidget(controls_frame)

    def create_tabs(self, layout):
        self.tabs = QTabWidget()

        # Pestaña 1: Visualizaciones
        self.create_visualizations_tab()

        # Pestaña 2: Resumen Estadístico
        self.create_statistics_tab()

        # Pestaña 3: Recomendaciones
        self.create_recommendations_tab()

        layout.addWidget(self.tabs)

    def create_visualizations_tab(self):
        viz_widget = QWidget()
        viz_layout = QVBoxLayout()

        # Título de la sección
        section_title = QLabel("📊 Visualizaciones Estadísticas")
        section_title.setProperty("class", "stats-title")
        viz_layout.addWidget(section_title)

        # Container para los gráficos
        graphs_container = QHBoxLayout()
        graphs_container.setSpacing(15)

        # Boxplot
        self.boxplot_container = self.create_graph_container("📦 Diagrama de Caja", "boxplot")
        graphs_container.addWidget(self.boxplot_container)

        # Histograma
        self.histogram_container = self.create_graph_container("📈 Histograma", "histogram")
        graphs_container.addWidget(self.histogram_container)

        # Densidad
        self.density_container = self.create_graph_container("🌊 Densidad", "density")
        graphs_container.addWidget(self.density_container)

        viz_layout.addLayout(graphs_container)
        viz_widget.setLayout(viz_layout)
        self.tabs.addTab(viz_widget, "📊 Visualizaciones")

    def create_graph_container(self, title, graph_type):
        container = QFrame()
        container.setProperty("class", "graph-container")
        container.setMinimumHeight(350)

        layout = QVBoxLayout()

        # Título del gráfico
        graph_title = QLabel(title)
        graph_title.setProperty("class", "graph-title")
        layout.addWidget(graph_title)

        # Label para la imagen
        graph_label = QLabel("Haga clic en 'Generar Análisis Completo' para ver el gráfico")
        graph_label.setAlignment(Qt.AlignCenter)
        graph_label.setMinimumHeight(250)
        graph_label.setStyleSheet("""
            background-color: #f8fafc;
            border: 2px dashed #cbd5e0;
            border-radius: 6px;
            color: #a0aec0;
            font-style: italic;
        """)

        setattr(self, f"label_{graph_type}", graph_label)
        layout.addWidget(graph_label)

        container.setLayout(layout)
        return container

    def create_statistics_tab(self):
        stats_widget = QWidget()
        stats_layout = QVBoxLayout()

        # Título
        title = QLabel("📋 Resumen Estadístico Detallado")
        title.setProperty("class", "stats-title")
        stats_layout.addWidget(title)

        # Tabla de estadísticas
        self.tabla_datos = QTableWidget()
        self.tabla_datos.setMinimumHeight(400)
        stats_layout.addWidget(self.tabla_datos)

        stats_widget.setLayout(stats_layout)
        self.tabs.addTab(stats_widget, "📋 Estadísticas")

    def create_recommendations_tab(self):
        rec_widget = QWidget()
        rec_layout = QVBoxLayout()

        # Título
        title = QLabel("💡 Recomendaciones y Observaciones")
        title.setProperty("class", "stats-title")
        rec_layout.addWidget(title)

        # Área de recomendaciones
        self.recommendations_area = QTextEdit()
        self.recommendations_area.setProperty("class", "recommendations-box")
        self.recommendations_area.setReadOnly(True)
        self.recommendations_area.setMinimumHeight(300)
        self.recommendations_area.setText("Genere el análisis completo para ver las recomendaciones automáticas...")

        rec_layout.addWidget(self.recommendations_area)

        rec_widget.setLayout(rec_layout)
        self.tabs.addTab(rec_widget, "💡 Recomendaciones")

    def create_navigation(self, layout):
        nav_layout = QHBoxLayout()

        self.btn_regresar = QPushButton("⬅️ Volver al Menú Principal")
        self.btn_regresar.setProperty("class", "danger-button")

        self.btn_bivariable = QPushButton("➡️ Ir a Análisis Bivariado")
        self.btn_bivariable.setProperty("class", "secondary-button")
        self.btn_bivariable.clicked.connect(self.cambiar_a_bivariado.emit)

        nav_layout.addWidget(self.btn_regresar)
        nav_layout.addStretch()
        nav_layout.addWidget(self.btn_bivariable)

        layout.addLayout(nav_layout)

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

    def limpiar_resultados(self):
        """Limpia todos los resultados"""
        self.limpiar_graficos()
        self.tabla_datos.clear()
        self.recommendations_area.setText("Genere el análisis completo para ver las recomendaciones automáticas...")

    def limpiar_graficos(self):
        """Limpia solo los gráficos"""
        for graph_type in ['boxplot', 'histogram', 'density']:
            label = getattr(self, f"label_{graph_type}")
            label.clear()
            label.setText("Haga clic en 'Generar Análisis Completo' para ver el gráfico")

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
                self.label_boxplot.setPixmap(pixmap.scaled(
                    self.label_boxplot.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def mostrar_histograma(self):
        col = self.combo_columnas.currentText()
        if col:
            generar_histograma(self.df, col)
            pixmap = QPixmap(obtener_ruta_imagen())
            if not pixmap.isNull():
                self.label_histogram.setPixmap(pixmap.scaled(
                    self.label_histogram.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def mostrar_densidad(self):
        col = self.combo_columnas.currentText()
        if col:
            generar_densidad(self.df, col)
            pixmap = QPixmap(obtener_ruta_imagen())
            if not pixmap.isNull():
                self.label_density.setPixmap(pixmap.scaled(
                    self.label_density.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

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

            # Generar recomendaciones
            recomendaciones = []

            recomendaciones.append(f"📊 ANÁLISIS DE LA VARIABLE: {col.upper()}")
            recomendaciones.append("=" * 50)
            recomendaciones.append("")

            # Análisis de distribución
            if abs(media - mediana) / std < 0.1:
                recomendaciones.append("✅ DISTRIBUCIÓN: Los datos muestran una distribución aproximadamente normal.")
                recomendaciones.append("   → Recomendación: Se pueden usar métodos estadísticos paramétricos.")
            else:
                recomendaciones.append("⚠️  DISTRIBUCIÓN: Los datos muestran asimetría significativa.")
                recomendaciones.append("   → Recomendación: Considerar transformaciones o métodos no paramétricos.")

            recomendaciones.append("")

            # Análisis de outliers
            if total_outliers == 0:
                recomendaciones.append("✅ VALORES ATÍPICOS: No se detectaron outliers significativos.")
                recomendaciones.append("   → Los datos son consistentes y confiables.")
            elif total_outliers <= len(serie) * 0.05:  # Menos del 5%
                recomendaciones.append(
                    f"⚠️  VALORES ATÍPICOS: Se detectaron {total_outliers} outliers ({total_outliers / len(serie) * 100:.1f}% de los datos).")
                recomendaciones.append("   → Cantidad normal, revisar si son errores de medición o valores reales.")
            else:
                recomendaciones.append(
                    f"🚨 VALORES ATÍPICOS: Se detectaron {total_outliers} outliers ({total_outliers / len(serie) * 100:.1f}% de los datos).")
                recomendaciones.append("   → Cantidad elevada, requiere investigación detallada.")

            recomendaciones.append("")

            # Análisis de variabilidad
            cv = (std / media) * 100 if media != 0 else 0
            if cv < 15:
                recomendaciones.append("✅ VARIABILIDAD: Baja variabilidad en los datos (CV < 15%).")
                recomendaciones.append("   → Los datos son consistentes y precisos.")
            elif cv < 30:
                recomendaciones.append("⚠️  VARIABILIDAD: Variabilidad moderada en los datos (15% ≤ CV < 30%).")
                recomendaciones.append("   → Variabilidad normal para datos ambientales.")
            else:
                recomendaciones.append("🚨 VARIABILIDAD: Alta variabilidad en los datos (CV ≥ 30%).")
                recomendaciones.append("   → Revisar posibles factores que afecten las mediciones.")

            recomendaciones.append("")

            # Recomendaciones específicas para calidad del agua
            recomendaciones.append("🌊 RECOMENDACIONES ESPECÍFICAS PARA CALIDAD DEL AGUA:")
            recomendaciones.append("")

            if 'ph' in col.lower() or 'pH' in col:
                if 6.5 <= media <= 8.5:
                    recomendaciones.append("✅ pH: Dentro del rango aceptable para agua potable (6.5-8.5).")
                else:
                    recomendaciones.append("⚠️  pH: Fuera del rango recomendado para agua potable.")
                    recomendaciones.append("   → Implementar medidas de corrección de pH.")

            elif 'oxigeno' in col.lower() or 'od' in col.lower():
                if media >= 5:
                    recomendaciones.append("✅ Oxígeno Disuelto: Niveles adecuados para vida acuática (≥5 mg/L).")
                else:
                    recomendaciones.append("🚨 Oxígeno Disuelto: Niveles críticos para vida acuática (<5 mg/L).")
                    recomendaciones.append("   → Investigar fuentes de contaminación orgánica.")

            elif 'turbidez' in col.lower() or 'turbiedad' in col.lower():
                if media <= 4:
                    recomendaciones.append("✅ Turbidez: Dentro de los límites aceptables (≤4 NTU).")
                else:
                    recomendaciones.append("⚠️  Turbidez: Elevada, puede indicar contaminación.")
                    recomendaciones.append("   → Evaluar sistemas de filtración y sedimentación.")

            elif 'coliform' in col.lower():
                if media == 0:
                    recomendaciones.append("✅ Coliformes: Ausentes, excelente calidad microbiológica.")
                else:
                    recomendaciones.append("🚨 Coliformes: Presentes, riesgo sanitario.")
                    recomendaciones.append("   → Implementar desinfección inmediata.")

            recomendaciones.append("")
            recomendaciones.append("📋 PRÓXIMOS PASOS RECOMENDADOS:")
            recomendaciones.append("• Realizar análisis bivariado para identificar correlaciones")
            recomendaciones.append("• Analizar tendencias temporales si hay datos de fecha")
            recomendaciones.append("• Comparar con normativas locales de calidad del agua")
            recomendaciones.append("• Implementar monitoreo continuo en puntos críticos")

            self.recommendations_area.setText("\n".join(recomendaciones))

        except Exception as e:
            self.recommendations_area.setText(f"Error al generar recomendaciones: {str(e)}")