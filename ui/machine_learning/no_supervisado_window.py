"""
no_supervisado_window.py - Ventana de Aprendizaje No Supervisado
Clustering, PCA y análisis exploratorio
"""

import sys
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFrame, QApplication, QMessageBox,
                             QTabWidget, QGridLayout, QGroupBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# Importar sistema de temas
try:
    from darkmode.theme_manager import ThemedWidget, ThemeManager
except ImportError:
    # Fallback si no existe darkmode
    class ThemedWidget:
        def __init__(self):
            pass
    class ThemeManager:
        @staticmethod
        def toggle_theme():
            pass
        @staticmethod
        def is_dark_theme():
            return False


class NoSupervisadoWindow(QWidget, ThemedWidget):
    """Ventana para análisis de aprendizaje no supervisado"""

    def __init__(self):
        QWidget.__init__(self)
        ThemedWidget.__init__(self)
        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):
        """Configurar la interfaz de usuario"""
        self.setWindowTitle("🔍 Aprendizaje No Supervisado - Calidad del Agua")
        self.setMinimumSize(1000, 700)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(25, 25, 25, 25)

        # Header
        header_layout = self.create_header()
        main_layout.addLayout(header_layout)

        # Título principal
        title_label = QLabel("🔍 Aprendizaje No Supervisado")
        title_label.setObjectName("mainTitle")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # Subtítulo
        subtitle = QLabel("Descubrimiento de patrones y agrupamiento en datos de calidad del agua")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle)

        # Tabs principales
        self.create_tabs(main_layout)

        # Footer con botones
        footer_layout = self.create_footer()
        main_layout.addLayout(footer_layout)

        self.setLayout(main_layout)

    def create_header(self):
        """Crear header con botón de tema"""
        header_layout = QHBoxLayout()

        # Logo
        logo_label = QLabel("🔍 No Supervisado")
        logo_label.setObjectName("logoLabel")

        # Spacer
        header_layout.addWidget(logo_label)
        header_layout.addStretch()

        # Botón de tema
        self.theme_button = QPushButton("🌙")
        self.theme_button.setObjectName("themeButton")
        self.theme_button.setFixedSize(40, 40)
        self.theme_button.clicked.connect(self.toggle_theme)

        header_layout.addWidget(self.theme_button)

        return header_layout

    def create_tabs(self, parent_layout):
        """Crear pestañas principales"""
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("mainTabs")

        # Tab 1: Clustering
        clustering_tab = self.create_clustering_tab()
        self.tab_widget.addTab(clustering_tab, "🎯 Clustering")

        # Tab 2: PCA
        pca_tab = self.create_pca_tab()
        self.tab_widget.addTab(pca_tab, "📊 PCA")

        # Tab 3: Análisis Exploratorio
        exploratory_tab = self.create_exploratory_tab()
        self.tab_widget.addTab(exploratory_tab, "🔍 Exploratorio")

        parent_layout.addWidget(self.tab_widget)

    def create_clustering_tab(self):
        """Crear pestaña de clustering"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Información
        info_frame = QFrame()
        info_frame.setObjectName("infoFrame")
        info_layout = QVBoxLayout()

        info_title = QLabel("🎯 Análisis de Clustering")
        info_title.setObjectName("sectionTitle")
        info_layout.addWidget(info_title)

        info_text = QLabel(
            "Descubre grupos naturales en tus datos de calidad del agua.\n"
            "Identifica patrones de contaminación y estaciones similares."
        )
        info_text.setObjectName("infoText")
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)

        info_frame.setLayout(info_layout)
        layout.addWidget(info_frame)

        # Opciones de clustering
        options_group = QGroupBox("Algoritmos de Clustering")
        options_layout = QGridLayout()

        # K-Means
        kmeans_btn = QPushButton("🎯 K-Means\nAgrupar por similitud")
        kmeans_btn.setObjectName("algorithmButton")
        kmeans_btn.clicked.connect(self.show_kmeans_info)

        # Clustering Jerárquico
        hierarchical_btn = QPushButton("🌳 Jerárquico\nDendrogramas")
        hierarchical_btn.setObjectName("algorithmButton")
        hierarchical_btn.clicked.connect(self.show_hierarchical_info)

        # DBSCAN
        dbscan_btn = QPushButton("🔍 DBSCAN\nDetección de anomalías")
        dbscan_btn.setObjectName("algorithmButton")
        dbscan_btn.clicked.connect(self.show_dbscan_info)

        options_layout.addWidget(kmeans_btn, 0, 0)
        options_layout.addWidget(hierarchical_btn, 0, 1)
        options_layout.addWidget(dbscan_btn, 0, 2)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def create_pca_tab(self):
        """Crear pestaña de PCA"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Información
        info_frame = QFrame()
        info_frame.setObjectName("infoFrame")
        info_layout = QVBoxLayout()

        info_title = QLabel("📊 Análisis de Componentes Principales (PCA)")
        info_title.setObjectName("sectionTitle")
        info_layout.addWidget(info_title)

        info_text = QLabel(
            "Reduce la dimensionalidad de tus datos manteniendo la información más importante.\n"
            "Identifica las variables más influyentes en la calidad del agua."
        )
        info_text.setObjectName("infoText")
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)

        info_frame.setLayout(info_layout)
        layout.addWidget(info_frame)

        # Opciones de PCA
        options_group = QGroupBox("Análisis de Componentes")
        options_layout = QGridLayout()

        # PCA Estándar
        pca_btn = QPushButton("📊 PCA Básico\nComponentes principales")
        pca_btn.setObjectName("algorithmButton")
        pca_btn.clicked.connect(self.show_pca_info)

        # Análisis de Varianza
        variance_btn = QPushButton("📈 Varianza Explicada\nImportancia de componentes")
        variance_btn.setObjectName("algorithmButton")
        variance_btn.clicked.connect(self.show_variance_info)

        # Biplot
        biplot_btn = QPushButton("🎯 Biplot\nVisualización 2D/3D")
        biplot_btn.setObjectName("algorithmButton")
        biplot_btn.clicked.connect(self.show_biplot_info)

        options_layout.addWidget(pca_btn, 0, 0)
        options_layout.addWidget(variance_btn, 0, 1)
        options_layout.addWidget(biplot_btn, 0, 2)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def create_exploratory_tab(self):
        """Crear pestaña de análisis exploratorio"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Información
        info_frame = QFrame()
        info_frame.setObjectName("infoFrame")
        info_layout = QVBoxLayout()

        info_title = QLabel("🔍 Análisis Exploratorio de Datos")
        info_title.setObjectName("sectionTitle")
        info_layout.addWidget(info_title)

        info_text = QLabel(
            "Explora relaciones ocultas y patrones en tus datos de calidad del agua.\n"
            "Descubre correlaciones y distribuciones importantes."
        )
        info_text.setObjectName("infoText")
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)

        info_frame.setLayout(info_layout)
        layout.addWidget(info_frame)

        # Opciones de análisis exploratorio
        options_group = QGroupBox("Herramientas Exploratorias")
        options_layout = QGridLayout()

        # Matriz de correlación
        corr_btn = QPushButton("🔗 Matriz Correlación\nRelaciones entre variables")
        corr_btn.setObjectName("algorithmButton")
        corr_btn.clicked.connect(self.show_correlation_info)

        # Distribuciones
        dist_btn = QPushButton("📊 Distribuciones\nHistogramas y boxplots")
        dist_btn.setObjectName("algorithmButton")
        dist_btn.clicked.connect(self.show_distribution_info)

        # Outliers
        outlier_btn = QPushButton("⚠️ Detección Outliers\nValores atípicos")
        outlier_btn.setObjectName("algorithmButton")
        outlier_btn.clicked.connect(self.show_outlier_info)

        options_layout.addWidget(corr_btn, 0, 0)
        options_layout.addWidget(dist_btn, 0, 1)
        options_layout.addWidget(outlier_btn, 0, 2)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def create_footer(self):
        """Crear footer con controles"""
        footer_layout = QHBoxLayout()

        # Botón de ayuda
        help_btn = QPushButton("❓ Ayuda")
        help_btn.setObjectName("secondaryButton")
        help_btn.clicked.connect(self.show_help)

        # Botón de configuración
        config_btn = QPushButton("⚙️ Config")
        config_btn.setObjectName("secondaryButton")
        config_btn.clicked.connect(self.show_config)

        # Spacer
        footer_layout.addWidget(help_btn)
        footer_layout.addWidget(config_btn)
        footer_layout.addStretch()

        # Botón cerrar
        close_btn = QPushButton("❌ Cerrar")
        close_btn.setObjectName("dangerButton")
        close_btn.clicked.connect(self.close)

        footer_layout.addWidget(close_btn)

        return footer_layout

    def apply_styles(self):
        """Aplicar estilos CSS"""
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
                background-color: #f8f9fa;
            }

            #logoLabel {
                font-size: 16px;
                font-weight: bold;
                color: #495057;
            }

            #themeButton {
                border: 2px solid #dee2e6;
                border-radius: 20px;
                background: #ffffff;
                font-size: 16px;
            }

            #themeButton:hover {
                background: #e9ecef;
                border-color: #adb5bd;
            }

            #mainTitle {
                font-size: 32px;
                font-weight: bold;
                color: #2c3e50;
                margin: 15px 0;
            }

            #subtitle {
                font-size: 16px;
                color: #6c757d;
                margin-bottom: 20px;
            }

            #mainTabs {
                background: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 10px;
            }

            #mainTabs::pane {
                border: 1px solid #dee2e6;
                border-radius: 8px;
                background: #ffffff;
                margin-top: 5px;
            }

            #mainTabs::tab-bar {
                alignment: center;
            }

            QTabBar::tab {
                background: #f8f9fa;
                color: #495057;
                padding: 12px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: 600;
                font-size: 14px;
                min-width: 120px;
                border: 1px solid #dee2e6;
            }

            QTabBar::tab:selected {
                background: #007bff;
                color: white;
                border-color: #007bff;
            }

            QTabBar::tab:hover:!selected {
                background: #e9ecef;
                color: #007bff;
            }

            #infoFrame {
                background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                border: 1px solid #2196f3;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
            }

            #sectionTitle {
                font-size: 20px;
                font-weight: bold;
                color: #1565c0;
                margin-bottom: 10px;
            }

            #infoText {
                font-size: 14px;
                color: #424242;
                line-height: 1.5;
            }

            QGroupBox {
                font-weight: bold;
                font-size: 16px;
                color: #2c3e50;
                border: 2px solid #dee2e6;
                border-radius: 10px;
                margin-top: 20px;
                padding-top: 15px;
                background: #ffffff;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px;
                background: #ffffff;
                color: #2c3e50;
            }

            #algorithmButton {
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                border: 2px solid #28a745;
                border-radius: 12px;
                padding: 20px;
                font-size: 14px;
                font-weight: 600;
                color: #155724;
                text-align: center;
                min-height: 80px;
                max-width: 200px;
            }

            #algorithmButton:hover {
                background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                border-color: #20c997;
                transform: translateY(-2px);
            }

            #algorithmButton:pressed {
                background: linear-gradient(135deg, #c3e6cb 0%, #b1dfbb 100%);
                transform: translateY(0px);
            }

            #secondaryButton {
                background: #6c757d;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
                font-size: 13px;
            }

            #secondaryButton:hover {
                background: #5a6268;
            }

            #dangerButton {
                background: #dc3545;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
                font-size: 13px;
            }

            #dangerButton:hover {
                background: #c82333;
            }
        """)

    def toggle_theme(self):
        """Alternar tema"""
        try:
            theme_manager = ThemeManager()
            theme_manager.toggle_theme()

            if theme_manager.is_dark_theme():
                self.theme_button.setText("☀️")
            else:
                self.theme_button.setText("🌙")
        except Exception:
            # Fallback simple
            if self.theme_button.text() == "🌙":
                self.theme_button.setText("☀️")
            else:
                self.theme_button.setText("🌙")

    # Métodos de información para cada algoritmo
    def show_kmeans_info(self):
        """Mostrar información sobre K-Means"""
        QMessageBox.information(
            self,
            "🎯 K-Means Clustering",
            "<h3>K-Means Clustering</h3>"
            "<p><b>¿Qué hace?</b><br>"
            "Agrupa las muestras de agua en K grupos basándose en similitudes "
            "en los parámetros de calidad.</p>"
            
            "<p><b>Casos de uso:</b></p>"
            "<ul>"
            "<li>• Identificar tipos de agua similares</li>"
            "<li>• Agrupar estaciones de monitoreo</li>"
            "<li>• Clasificar niveles de contaminación</li>"
            "</ul>"
            
            "<p><b>Parámetros:</b> Número de clusters (K), algoritmo de inicialización</p>"
            
            "<p><i>⚠️ Módulo en desarrollo. Funcionalidad completa próximamente.</i></p>"
        )

    def show_hierarchical_info(self):
        """Mostrar información sobre clustering jerárquico"""
        QMessageBox.information(
            self,
            "🌳 Clustering Jerárquico",
            "<h3>Clustering Jerárquico</h3>"
            "<p><b>¿Qué hace?</b><br>"
            "Crea un árbol (dendrograma) que muestra las relaciones jerárquicas "
            "entre muestras de agua.</p>"
            
            "<p><b>Casos de uso:</b></p>"
            "<ul>"
            "<li>• Visualizar relaciones entre estaciones</li>"
            "<li>• Encontrar grupos naturales</li>"
            "<li>• Análisis de similitud temporal</li>"
            "</ul>"
            
            "<p><b>Ventajas:</b> No necesita especificar número de clusters previamente</p>"
            
            "<p><i>⚠️ Módulo en desarrollo. Funcionalidad completa próximamente.</i></p>"
        )

    def show_dbscan_info(self):
        """Mostrar información sobre DBSCAN"""
        QMessageBox.information(
            self,
            "🔍 DBSCAN",
            "<h3>DBSCAN (Density-Based Clustering)</h3>"
            "<p><b>¿Qué hace?</b><br>"
            "Encuentra grupos basándose en densidad y detecta automáticamente "
            "valores atípicos (outliers).</p>"
            
            "<p><b>Casos de uso:</b></p>"
            "<ul>"
            "<li>• Detectar contaminación anómala</li>"
            "<li>• Identificar eventos extraordinarios</li>"
            "<li>• Filtrar datos erróneos</li>"
            "</ul>"
            
            "<p><b>Ventajas:</b> Detecta automáticamente outliers y no requiere "
            "especificar número de clusters</p>"
            
            "<p><i>⚠️ Módulo en desarrollo. Funcionalidad completa próximamente.</i></p>"
        )

    def show_pca_info(self):
        """Mostrar información sobre PCA"""
        QMessageBox.information(
            self,
            "📊 PCA - Análisis de Componentes Principales",
            "<h3>PCA (Principal Component Analysis)</h3>"
            "<p><b>¿Qué hace?</b><br>"
            "Reduce la dimensionalidad manteniendo la máxima varianza posible. "
            "Identifica las combinaciones de variables más importantes.</p>"
            
            "<p><b>Casos de uso:</b></p>"
            "<ul>"
            "<li>• Simplificar datasets complejos</li>"
            "<li>• Identificar variables más influyentes</li>"
            "<li>• Visualizar datos en 2D/3D</li>"
            "<li>• Preprocesamiento para otros algoritmos</li>"
            "</ul>"
            
            "<p><b>Interpretación:</b> Los primeros componentes explican la mayor "
            "variabilidad en los datos de calidad del agua.</p>"
            
            "<p><i>⚠️ Módulo en desarrollo. Funcionalidad completa próximamente.</i></p>"
        )

    def show_variance_info(self):
        """Mostrar información sobre análisis de varianza"""
        QMessageBox.information(
            self,
            "📈 Análisis de Varianza Explicada",
            "<h3>Varianza Explicada por Componentes</h3>"
            "<p><b>¿Qué muestra?</b><br>"
            "El porcentaje de información (varianza) que captura cada "
            "componente principal.</p>"
            
            "<p><b>Interpretación:</b></p>"
            "<ul>"
            "<li>• Componente 1: Mayor variabilidad</li>"
            "<li>• Componente 2: Segunda mayor variabilidad</li>"
            "<li>• Y así sucesivamente...</li>"
            "</ul>"
            
            "<p><b>Regla práctica:</b> Mantener componentes que expliquen "
            "al menos 80-90% de la varianza total.</p>"
            
            "<p><i>⚠️ Módulo en desarrollo. Funcionalidad completa próximamente.</i></p>"
        )

    def show_biplot_info(self):
        """Mostrar información sobre biplot"""
        QMessageBox.information(
            self,
            "🎯 Biplot",
            "<h3>Biplot - Visualización PCA</h3>"
            "<p><b>¿Qué muestra?</b><br>"
            "Combina la visualización de las muestras y las variables "
            "en el mismo gráfico.</p>"
            
            "<p><b>Elementos:</b></p>"
            "<ul>"
            "<li>• Puntos: Muestras de agua individuales</li>"
            "<li>• Flechas: Variables (pH, turbidez, etc.)</li>"
            "<li>• Longitud de flecha: Importancia de la variable</li>"
            "<li>• Ángulo entre flechas: Correlación</li>"
            "</ul>"
            
            "<p><b>Interpretación:</b> Flechas en la misma dirección indican "
            "variables correlacionadas positivamente.</p>"
            
            "<p><i>⚠️ Módulo en desarrollo. Funcionalidad completa próximamente.</i></p>"
        )

    def show_correlation_info(self):
        """Mostrar información sobre matriz de correlación"""
        QMessageBox.information(
            self,
            "🔗 Matriz de Correlación",
            "<h3>Matriz de Correlación</h3>"
            "<p><b>¿Qué muestra?</b><br>"
            "Las relaciones lineales entre todas las variables de "
            "calidad del agua.</p>"
            
            "<p><b>Valores:</b></p>"
            "<ul>"
            "<li>• +1: Correlación positiva perfecta</li>"
            "<li>• 0: Sin correlación lineal</li>"
            "<li>• -1: Correlación negativa perfecta</li>"
            "</ul>"
            
            "<p><b>Interpretación para calidad del agua:</b></p>"
            "<ul>"
            "<li>• pH vs Oxígeno: Generalmente positiva</li>"
            "<li>• Turbidez vs Coliformes: Positiva</li>"
            "<li>• Temperatura vs Oxígeno: Negativa</li>"
            "</ul>"
            
            "<p><i>⚠️ Módulo en desarrollo. Funcionalidad completa próximamente.</i></p>"
        )

    def show_distribution_info(self):
        """Mostrar información sobre distribuciones"""
        QMessageBox.information(
            self,
            "📊 Análisis de Distribuciones",
            "<h3>Distribuciones de Variables</h3>"
            "<p><b>¿Qué analiza?</b><br>"
            "La forma en que se distribuyen los valores de cada "
            "parámetro de calidad del agua.</p>"
            
            "<p><b>Visualizaciones:</b></p>"
            "<ul>"
            "<li>• Histogramas: Frecuencia de valores</li>"
            "<li>• Box plots: Mediana, cuartiles, outliers</li>"
            "<li>• Curvas de densidad: Distribución suavizada</li>"
            "</ul>"
            
            "<p><b>Información útil:</b></p>"
            "<ul>"
            "<li>• Valores típicos vs excepcionales</li>"
            "<li>• Simetría de los datos</li>"
            "<li>• Presencia de múltiples picos</li>"
            "</ul>"
            
            "<p><i>⚠️ Módulo en desarrollo. Funcionalidad completa próximamente.</i></p>"
        )

    def show_outlier_info(self):
        """Mostrar información sobre detección de outliers"""
        QMessageBox.information(
            self,
            "⚠️ Detección de Outliers",
            "<h3>Detección de Valores Atípicos</h3>"
            "<p><b>¿Qué son los outliers?</b><br>"
            "Valores que se desvían significativamente del patrón "
            "normal en los datos de calidad del agua.</p>"
            
            "<p><b>Métodos de detección:</b></p>"
            "<ul>"
            "<li>• IQR (Rango Intercuartílico)</li>"
            "<li>• Z-Score (Desviaciones estándar)</li>"
            "<li>• Isolation Forest</li>"
            "<li>• DBSCAN clustering</li>"
            "</ul>"
            
            "<p><b>Causas posibles:</b></p>"
            "<ul>"
            "<li>• Eventos de contaminación</li>"
            "<li>• Errores de medición</li>"
            "<li>• Condiciones ambientales extremas</li>"
            "</ul>"
            
            "<p><i>⚠️ Módulo en desarrollo. Funcionalidad completa próximamente.</i></p>"
        )

    def show_help(self):
        """Mostrar ayuda general"""
        QMessageBox.information(
            self,
            "❓ Ayuda - Aprendizaje No Supervisado",
            "<h3>💡 Guía de Uso</h3>"
            
            "<p><b>🎯 Clustering:</b><br>"
            "Usa K-Means para grupos definidos, Jerárquico para explorar "
            "relaciones, DBSCAN para detectar anomalías.</p>"
            
            "<p><b>📊 PCA:</b><br>"
            "Ideal para reducir dimensionalidad y entender qué variables "
            "son más importantes en tu dataset.</p>"
            
            "<p><b>🔍 Análisis Exploratorio:</b><br>"
            "Comienza siempre aquí para entender la estructura de tus datos "
            "antes de aplicar algoritmos más complejos.</p>"
            
            "<p><b>🔄 Flujo recomendado:</b></p>"
            "<ol>"
            "<li>1. Análisis exploratorio y correlaciones</li>"
            "<li>2. Detección y tratamiento de outliers</li>"
            "<li>3. PCA para reducir dimensionalidad</li>"
            "<li>4. Clustering para encontrar grupos</li>"
            "</ol>"
            
            "<p><i>💡 Tip: Cada análisis proporciona insights únicos. "
            "Combina varios métodos para obtener una visión completa.</i></p>"
        )

    def show_config(self):
        """Mostrar configuración"""
        QMessageBox.information(
            self,
            "⚙️ Configuración - No Supervisado",
            "<h3>🔧 Configuración del Módulo</h3>"
            
            "<p><b>📦 Dependencias requeridas:</b></p>"
            "<ul>"
            "<li>• scikit-learn (clustering, PCA)</li>"
            "<li>• matplotlib (visualizaciones)</li>"
            "<li>• seaborn (gráficos estadísticos)</li>"
            "<li>• pandas & numpy (manipulación datos)</li>"
            "</ul>"
            
            "<p><b>🎛️ Parámetros configurables:</b></p>"
            "<ul>"
            "<li>• Número de clusters (K-Means)</li>"
            "<li>• Número de componentes (PCA)</li>"
            "<li>• Umbrales de outliers</li>"
            "<li>• Métricas de distancia</li>"
            "</ul>"
            
            "<p><b>💾 Exportación:</b><br>"
            "Los resultados se pueden exportar en formatos CSV, "
            "Excel y PNG (gráficos).</p>"
            
            "<p><b>🚀 Estado:</b> Módulo en desarrollo activo</p>"
        )

    def closeEvent(self, event):
        """Manejar evento de cierre"""
        if hasattr(self, 'theme_manager'):
            self.theme_manager.remove_observer(self)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NoSupervisadoWindow()
    window.show()
    sys.exit(app.exec_())