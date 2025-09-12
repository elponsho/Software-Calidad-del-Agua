"""
segmentacion_ml.py - Sistema ML para Análisis de Calidad del Agua
Ventana principal con navegación a módulos Supervisado y No Supervisado
Versión actualizada con módulo No Supervisado integrado
"""

import sys
import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                             QHBoxLayout, QFrame, QApplication, QMessageBox,
                             QGridLayout, QScrollArea, QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette

import traceback

# Importar Data Manager
try:
    from .data_manager import DataManagerSingleton, get_data_manager
    DATA_MANAGER_AVAILABLE = True
except ImportError:
    DATA_MANAGER_AVAILABLE = False
    print("⚠️ DataManager no disponible")

# Importar ventana de supervisado optimizada
try:
    from .supervisado_window import SupervisadoWindow
    SUPERVISADO_AVAILABLE = True
    print("✅ Módulo Supervisado disponible")
except ImportError:
    SUPERVISADO_AVAILABLE = False
    print("⚠️ Módulo Supervisado no disponible")

# Importar ventana de no supervisado CORREGIDA
try:
    from .no_supervisado_window import NoSupervisadoWindow
    NO_SUPERVISADO_AVAILABLE = True
    print("✅ Módulo No Supervisado disponible")
except ImportError as e:
    NO_SUPERVISADO_AVAILABLE = False
    print(f"⚠️ Módulo No Supervisado no disponible: {e}")


class ModernButton(QFrame):
    """Botón moderno personalizado con efectos hover"""
    clicked = pyqtSignal()

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setup_ui()
        self.add_shadow_effect()

    def setup_ui(self):
        self.setObjectName("modernButton")
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumSize(320, 200)
        self.setMaximumSize(400, 250)

        # Layout principal
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(30, 25, 30, 25)

        # Contenedor del icono
        icon_container = QFrame()
        icon_container.setObjectName("iconContainer")
        icon_container.setFixedSize(80, 80)
        icon_layout = QVBoxLayout(icon_container)

        icon_label = QLabel(self.config["icon"])
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet(f"""
            font-size: 40px;
            color: {self.config['color']};
        """)
        icon_layout.addWidget(icon_label)

        # Título
        title_label = QLabel(self.config["title"])
        title_label.setObjectName("buttonTitle")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setWordWrap(True)

        # Descripción
        desc_label = QLabel(self.config["description"])
        desc_label.setObjectName("buttonDescription")
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setWordWrap(True)

        # Características
        features_label = QLabel(self.config["features"])
        features_label.setObjectName("buttonFeatures")
        features_label.setAlignment(Qt.AlignCenter)
        features_label.setWordWrap(True)

        # Estado del módulo
        status_label = QLabel(self.config.get("status", ""))
        status_label.setObjectName("buttonStatus")
        status_label.setAlignment(Qt.AlignCenter)

        # Agregar widgets
        layout.addWidget(icon_container, alignment=Qt.AlignCenter)
        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        layout.addWidget(features_label)
        layout.addWidget(status_label)
        layout.addStretch()

        # Estilo base
        self.setStyleSheet(f"""
            #modernButton {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #ffffff, stop: 1 #f8f9fa);
                border: 2px solid #e9ecef;
                border-radius: 15px;
            }}
            
            #modernButton:hover {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #ffffff, stop: 1 {self.config['color']}20);
                border: 2px solid {self.config['color']};
            }}
            
            #iconContainer {{
                background: {self.config['color']}15;
                border-radius: 40px;
            }}
            
            #buttonTitle {{
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
                margin-top: 10px;
            }}
            
            #buttonDescription {{
                font-size: 14px;
                color: #6c757d;
                margin-top: 5px;
            }}
            
            #buttonFeatures {{
                font-size: 12px;
                color: #868e96;
                font-style: italic;
                margin-top: 10px;
            }}
            
            #buttonStatus {{
                font-size: 11px;
                font-weight: bold;
                margin-top: 5px;
            }}
        """)

    def add_shadow_effect(self):
        """Agregar efecto de sombra"""
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(5)
        shadow.setColor(QColor(0, 0, 0, 30))
        self.setGraphicsEffect(shadow)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()

    def enterEvent(self, event):
        """Efecto al pasar el mouse"""
        shadow = self.graphicsEffect()
        shadow.setBlurRadius(25)
        shadow.setYOffset(8)
        shadow.setColor(QColor(0, 0, 0, 50))

    def leaveEvent(self, event):
        """Efecto al quitar el mouse"""
        shadow = self.graphicsEffect()
        shadow.setBlurRadius(20)
        shadow.setYOffset(5)
        shadow.setColor(QColor(0, 0, 0, 30))


class SegmentacionML(QWidget):
    """Ventana principal del sistema ML - Diseño moderno y optimizado"""

    # Señales
    ventana_cerrada = pyqtSignal()
    regresar_menu = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.supervisado_window = None
        self.no_supervisado_window = None
        self.data_manager = get_data_manager() if DATA_MANAGER_AVAILABLE else None
        self.init_ui()
        self.apply_theme()

        # Verificar datos al iniciar
        self.check_data_status()

    def init_ui(self):
        """Inicializar interfaz de usuario"""
        self.setWindowTitle("🧠 Sistema Machine Learning - Análisis de Calidad del Agua")
        self.setMinimumSize(1000, 700)

        # Layout principal con scroll
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #f5f7fa;
            }
            QScrollBar:vertical {
                background: #f5f7fa;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #cbd5e0;
                border-radius: 5px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: #a0aec0;
            }
        """)

        # Widget contenedor
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(0)
        container_layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = self.create_header()
        container_layout.addWidget(header)

        # Contenido principal
        content = self.create_main_content()
        container_layout.addWidget(content)

        # Footer
        footer = self.create_footer()
        container_layout.addWidget(footer)

        scroll_area.setWidget(container)
        main_layout.addWidget(scroll_area)

    def create_header(self):
        """Crear header moderno"""
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(180)

        layout = QVBoxLayout(header)
        layout.setSpacing(10)
        layout.setContentsMargins(40, 30, 40, 30)

        # Título principal con gradiente
        title_container = QFrame()
        title_container.setObjectName("titleContainer")
        title_layout = QVBoxLayout(title_container)

        main_title = QLabel("Sistema de Machine Learning")
        main_title.setObjectName("mainTitle")
        main_title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("Análisis Avanzado de Calidad del Agua con IA")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignCenter)

        title_layout.addWidget(main_title)
        title_layout.addWidget(subtitle)

        # Barra de estado
        status_bar = self.create_status_bar()

        layout.addWidget(title_container)
        layout.addWidget(status_bar)

        return header

    def create_status_bar(self):
        """Crear barra de estado del sistema"""
        status_frame = QFrame()
        status_frame.setObjectName("statusBar")
        status_frame.setMaximumHeight(40)

        layout = QHBoxLayout(status_frame)
        layout.setContentsMargins(20, 5, 20, 5)

        # Verificar dependencias y datos
        deps_ok = True  # Asumimos que las dependencias están disponibles

        # Estado de datos
        if self.data_manager and self.data_manager.has_data():
            data_status = "📊 Datos cargados"
            data_color = "✅"
            try:
                data_info = self.data_manager.get_info()
                data_shape = f"({data_info['shape'][0]} × {data_info['shape'][1]})"
            except:
                data_shape = ""
        else:
            data_status = "📊 Sin datos"
            data_color = "❌"
            data_shape = ""

        # Estado de módulos
        supervisado_status = "✅" if SUPERVISADO_AVAILABLE else "❌"
        no_supervisado_status = "✅" if NO_SUPERVISADO_AVAILABLE else "❌"

        # Icono de estado general
        status_icon = QLabel("✅" if deps_ok else "⚠️")
        status_text = QLabel("Sistema listo" if deps_ok else "Faltan dependencias")
        status_text.setObjectName("statusText")

        # Estado de datos
        data_icon = QLabel(data_color)
        self.data_status_label = QLabel(f"{data_status} {data_shape}")
        self.data_status_label.setObjectName("statusText")

        # Estado de módulos
        modules_label = QLabel(f"📊 Supervisado {supervisado_status} | 🔍 No Supervisado {no_supervisado_status}")
        modules_label.setObjectName("statusText")

        layout.addWidget(status_icon)
        layout.addWidget(status_text)
        layout.addWidget(QLabel(" | "))
        layout.addWidget(data_icon)
        layout.addWidget(self.data_status_label)
        layout.addWidget(QLabel(" | "))
        layout.addWidget(modules_label)
        layout.addStretch()

        # Información del sistema
        import multiprocessing
        python_info = QLabel(f"🐍 Python {sys.version.split()[0]}")
        python_info.setObjectName("systemInfo")
        layout.addWidget(python_info)

        return status_frame

    def create_main_content(self):
        """Crear contenido principal con los dos módulos"""
        content = QFrame()
        content.setObjectName("mainContent")

        layout = QVBoxLayout(content)
        layout.setSpacing(30)
        layout.setContentsMargins(40, 40, 40, 40)

        # Descripción
        desc_label = QLabel("Selecciona el tipo de análisis que deseas realizar:")
        desc_label.setObjectName("description")
        desc_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc_label)

        # Contenedor de botones
        buttons_container = QFrame()
        buttons_layout = QHBoxLayout(buttons_container)
        buttons_layout.setSpacing(30)
        buttons_layout.setContentsMargins(0, 0, 0, 0)

        # Configuración de módulos
        modules = [
            {
                "title": "Aprendizaje Supervisado",
                "icon": "🎯",
                "description": "Predicción y clasificación con datos etiquetados",
                "features": "Regresión • SVM • Random Forest • Redes Neuronales",
                "color": "#3498db",
                "status": "✅ Disponible" if SUPERVISADO_AVAILABLE else "❌ No disponible",
                "action": self.open_supervisado
            },
            {
                "title": "Aprendizaje No Supervisado",
                "icon": "🔍",
                "description": "Descubrimiento de patrones sin etiquetas",
                "features": "K-Means • DBSCAN • PCA • Clustering Jerárquico",
                "color": "#9b59b6",
                "status": "✅ Disponible" if NO_SUPERVISADO_AVAILABLE else "❌ No disponible",
                "action": self.open_no_supervisado
            }
        ]

        # Crear botones
        for module_config in modules:
            button = ModernButton(module_config)
            button.clicked.connect(module_config["action"])
            buttons_layout.addWidget(button)

        layout.addWidget(buttons_container)

        # Información adicional
        info_frame = self.create_info_section()
        layout.addWidget(info_frame)

        return content

    def create_info_section(self):
        """Crear sección de información adicional"""
        info_frame = QFrame()
        info_frame.setObjectName("infoSection")
        info_frame.setMaximumHeight(150)

        layout = QGridLayout(info_frame)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 20, 30, 20)

        # Características del sistema
        features = [
            ("🚀", "Alto Rendimiento", "Procesamiento paralelo optimizado"),
            ("📊", "Visualizaciones", "Gráficos interactivos y reportes"),
            ("🔧", "Personalizable", "Parámetros ajustables para cada modelo"),
            ("💾", "Exportación", "Guarda resultados en múltiples formatos")
        ]

        for i, (icon, title, desc) in enumerate(features):
            feature_widget = self.create_feature_widget(icon, title, desc)
            row = i // 2
            col = i % 2
            layout.addWidget(feature_widget, row, col)

        return info_frame

    def create_feature_widget(self, icon, title, description):
        """Crear widget de característica"""
        widget = QFrame()
        widget.setObjectName("featureWidget")

        layout = QHBoxLayout(widget)
        layout.setSpacing(15)

        # Icono
        icon_label = QLabel(icon)
        icon_label.setObjectName("featureIcon")
        icon_label.setFixedSize(40, 40)
        icon_label.setAlignment(Qt.AlignCenter)

        # Texto
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)

        title_label = QLabel(title)
        title_label.setObjectName("featureTitle")

        desc_label = QLabel(description)
        desc_label.setObjectName("featureDesc")

        text_layout.addWidget(title_label)
        text_layout.addWidget(desc_label)

        layout.addWidget(icon_label)
        layout.addLayout(text_layout)
        layout.addStretch()

        return widget

    def create_footer(self):
        """Crear footer con controles"""
        footer = QFrame()
        footer.setObjectName("footer")
        footer.setFixedHeight(80)

        layout = QHBoxLayout(footer)
        layout.setContentsMargins(40, 20, 40, 20)

        # Botones de acción
        help_btn = QPushButton("❓ Ayuda")
        help_btn.setObjectName("footerButton")
        help_btn.clicked.connect(self.show_help)

        about_btn = QPushButton("ℹ️ Acerca de")
        about_btn.setObjectName("footerButton")
        about_btn.clicked.connect(self.show_about)

        # Botón actualizar datos
        self.refresh_btn = QPushButton("🔄 Actualizar Datos")
        self.refresh_btn.setObjectName("footerButton")
        self.refresh_btn.clicked.connect(self.check_data_status)

        # Botón de salir
        self.exit_button = QPushButton("← Regresar al Menú")
        self.exit_button.setObjectName("exitButton")
        self.exit_button.clicked.connect(self.on_regresar_menu)

        layout.addWidget(help_btn)
        layout.addWidget(about_btn)
        layout.addWidget(self.refresh_btn)
        layout.addStretch()
        layout.addWidget(self.exit_button)

        return footer

    def apply_theme(self):
        """Aplicar tema moderno"""
        self.setStyleSheet("""
            /* Estilos generales */
            QWidget {
                font-family: 'Segoe UI', -apple-system, Arial, sans-serif;
                color: #2d3748;
            }
            
            /* Header */
            #header {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #667eea, stop: 1 #764ba2);
            }
            
            #titleContainer {
                background: transparent;
            }
            
            #mainTitle {
                font-size: 36px;
                font-weight: 700;
                color: white;
                margin-bottom: 5px;
            }
            
            #subtitle {
                font-size: 18px;
                color: rgba(255, 255, 255, 0.9);
                font-weight: 300;
            }
            
            /* Status Bar */
            #statusBar {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            #statusText {
                color: white;
                font-weight: 500;
            }
            
            #systemInfo {
                color: rgba(255, 255, 255, 0.8);
                font-size: 12px;
            }
            
            /* Contenido principal */
            #mainContent {
                background: transparent;
            }
            
            #description {
                font-size: 16px;
                color: #4a5568;
                margin-bottom: 20px;
            }
            
            /* Sección de información */
            #infoSection {
                background: white;
                border-radius: 15px;
                border: 1px solid #e2e8f0;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            }
            
            #featureWidget {
                background: transparent;
            }
            
            #featureIcon {
                background: #f7fafc;
                border-radius: 10px;
                font-size: 20px;
            }
            
            #featureTitle {
                font-weight: 600;
                font-size: 14px;
                color: #2d3748;
            }
            
            #featureDesc {
                font-size: 12px;
                color: #718096;
            }
            
            /* Footer */
            #footer {
                background: white;
                border-top: 1px solid #e2e8f0;
            }
            
            #footerButton {
                background: #e2e8f0;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                color: #4a5568;
                font-weight: 500;
            }
            
            #footerButton:hover {
                background: #cbd5e0;
                color: #2d3748;
            }
            
            #exitButton {
                background: #48bb78;
                border: none;
                border-radius: 8px;
                padding: 10px 25px;
                color: white;
                font-weight: 600;
                font-size: 14px;
            }
            
            #exitButton:hover {
                background: #38a169;
            }
        """)

    def check_data_status(self):
        """Verificar y actualizar estado de los datos"""
        if self.data_manager and self.data_manager.has_data():
            try:
                data_info = self.data_manager.get_info()
                self.data_status_label.setText(
                    f"📊 Datos cargados ({data_info['shape'][0]:,} × {data_info['shape'][1]})"
                )
            except:
                self.data_status_label.setText("📊 Datos cargados (info no disponible)")
        else:
            self.data_status_label.setText("📊 Sin datos cargados")

    def open_supervisado(self):
        """Abrir módulo de aprendizaje supervisado"""
        # Verificar disponibilidad del módulo
        if not SUPERVISADO_AVAILABLE:
            QMessageBox.warning(
                self,
                "Módulo no disponible",
                "El módulo de Aprendizaje Supervisado no está instalado.\n\n"
                "Asegúrate de que el archivo 'supervisado_window.py' esté en el directorio correcto."
            )
            return

        # Verificar datos cargados
        if not self.data_manager or not self.data_manager.has_data():
            reply = QMessageBox.question(
                self,
                "Sin datos cargados",
                "No hay datos cargados en el sistema.\n\n"
                "¿Deseas cargar datos de demostración para explorar el módulo?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Yes:
                try:
                    self.data_manager.generate_demo_data(n_samples=500)
                    self.check_data_status()
                    QMessageBox.information(
                        self,
                        "Datos de demostración",
                        "Se han generado datos de demostración con 500 muestras.\n"
                        "Ahora puedes explorar el módulo de ML Supervisado."
                    )
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "Error",
                        f"Error al generar datos de demostración:\n{str(e)}"
                    )
                    return
            else:
                return

        # Crear o mostrar ventana supervisado
        try:
            if self.supervisado_window is None:
                self.supervisado_window = SupervisadoWindow()

                # Conectar señal de cierre para limpieza
                self.supervisado_window.destroyed.connect(
                    lambda: setattr(self, 'supervisado_window', None)
                )

            self.supervisado_window.show()
            self.supervisado_window.raise_()
            self.supervisado_window.activateWindow()

        except Exception as e:
            print(f"Error abriendo SupervisadoWindow:\n{traceback.format_exc()}")
            QMessageBox.critical(
                self,
                "Error al abrir módulo",
                f"No se pudo abrir el módulo de aprendizaje supervisado:\n{str(e)}"
            )

    def open_no_supervisado(self):
        """Abrir módulo de aprendizaje no supervisado"""
        print("🔍 Intentando abrir módulo No Supervisado...")

        # Verificar disponibilidad del módulo
        if not NO_SUPERVISADO_AVAILABLE:
            QMessageBox.warning(
                self,
                "Módulo no disponible",
                "El módulo de Aprendizaje No Supervisado no está disponible.\n\n"
                "Verifica que:\n"
                "• El archivo 'no_supervisado_window.py' esté en el directorio correcto\n"
                "• Las librerías scikit-learn, matplotlib y numpy estén instaladas\n\n"
                "Instala con: pip install scikit-learn matplotlib numpy pandas"
            )
            return

        # Verificar datos
        if not self.data_manager or not self.data_manager.has_data():
            reply = QMessageBox.question(
                self,
                "Sin datos cargados",
                "No hay datos cargados en el sistema.\n\n"
                "¿Deseas cargar datos de demostración para explorar el módulo?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Yes:
                try:
                    self.data_manager.generate_demo_data(n_samples=300)
                    self.check_data_status()
                    QMessageBox.information(
                        self,
                        "Datos de demostración",
                        "Se han generado datos de demostración con 300 muestras.\n"
                        "Incluyen parámetros de calidad del agua con patrones para clustering.\n\n"
                        "Ahora puedes explorar el módulo de ML No Supervisado."
                    )
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "Error",
                        f"Error al generar datos de demostración:\n{str(e)}"
                    )
                    return
            else:
                return

        # Crear o mostrar ventana
        try:
            print("✅ Creando ventana NoSupervisadoWindow...")

            if self.no_supervisado_window is None:
                self.no_supervisado_window = NoSupervisadoWindow()
                print("✅ Ventana NoSupervisadoWindow creada")

                # Conectar señal de cierre para limpieza
                self.no_supervisado_window.destroyed.connect(
                    lambda: setattr(self, 'no_supervisado_window', None)
                )

            print("📱 Mostrando ventana NoSupervisadoWindow...")
            self.no_supervisado_window.show()
            self.no_supervisado_window.raise_()
            self.no_supervisado_window.activateWindow()
            print("✅ Ventana NoSupervisadoWindow mostrada correctamente")

        except Exception as e:
            error_msg = f"Error abriendo NoSupervisadoWindow:\n{traceback.format_exc()}"
            print(error_msg)
            QMessageBox.critical(
                self,
                "Error al abrir módulo",
                f"No se pudo abrir el módulo de Aprendizaje No Supervisado:\n\n{str(e)}\n\n"
                "Detalles técnicos:\n"
                "• Verifica que las dependencias estén instaladas\n"
                "• Revisa que el archivo no_supervisado_window.py esté presente\n"
                "• Consulta la consola para más detalles del error\n\n"
                "Instala dependencias con:\n"
                "pip install scikit-learn matplotlib numpy pandas"
            )

    def show_help(self):
        """Mostrar ayuda del sistema"""
        help_text = """
        <h2>🧠 Sistema de Machine Learning</h2>
        
        <h3>🎯 Aprendizaje Supervisado</h3>
        <p>Utiliza datos etiquetados para entrenar modelos que pueden:</p>
        <ul>
            <li>• <b>Regresión:</b> Predecir valores continuos (pH, temperatura, etc.)</li>
            <li>• <b>Clasificación:</b> Categorizar calidad del agua (Buena, Regular, Mala)</li>
        </ul>
        
        <p><b>Algoritmos disponibles:</b></p>
        <ul>
            <li>• Regresión Lineal (Simple y Múltiple)</li>
            <li>• Árboles de Decisión</li>
            <li>• Random Forest</li>
            <li>• Support Vector Machines (SVM)</li>
        </ul>
        
        <h3>🔍 Aprendizaje No Supervisado</h3>
        <p>Descubre patrones ocultos en datos sin etiquetas:</p>
        <ul>
            <li>• <b>K-Means:</b> Clustering optimizado con selección automática de K</li>
            <li>• <b>Clustering Jerárquico:</b> Dendrogramas y múltiples métricas</li>
            <li>• <b>DBSCAN:</b> Clustering basado en densidad, robusto a outliers</li>
            <li>• <b>PCA Avanzado:</b> Reducción dimensional lineal y no lineal</li>
            <li>• <b>Análisis Exploratorio:</b> Correlaciones, outliers, distribuciones</li>
        </ul>
        
        <h3>📊 Flujo de trabajo recomendado</h3>
        <ol>
            <li>1. Carga tus datos desde el módulo de Cargar Datos</li>
            <li>2. Explora con Análisis No Supervisado para entender patrones</li>
            <li>3. Usa Supervisado para predicción/clasificación</li>
            <li>4. Configura parámetros según tus necesidades</li>
            <li>5. Visualiza y exporta los resultados</li>
        </ol>
        
        <h3>💡 Casos de uso para calidad del agua:</h3>
        <ul>
            <li>• <b>Clustering:</b> Agrupar estaciones de monitoreo similares</li>
            <li>• <b>Predicción:</b> Estimar índices de calidad futuros</li>
            <li>• <b>Detección de anomalías:</b> Identificar contaminación</li>
            <li>• <b>Reducción dimensional:</b> Visualizar datos complejos</li>
        </ul>
        
        <p><b>💡 Consejo:</b> Si no tienes datos, usa "Demo" para generar 
        datos sintéticos realistas y explorar las funcionalidades.</p>
        """

        msg = QMessageBox(self)
        msg.setWindowTitle("❓ Ayuda - Sistema ML")
        msg.setTextFormat(Qt.RichText)
        msg.setText(help_text)
        msg.setIcon(QMessageBox.Information)
        msg.exec_()

    def show_about(self):
        """Mostrar información del sistema"""
        # Verificar versiones de librerías
        versions_info = []

        try:
            import numpy
            versions_info.append(f"• NumPy {numpy.__version__}")
        except:
            versions_info.append("• NumPy (no instalado)")

        try:
            import pandas
            versions_info.append(f"• Pandas {pandas.__version__}")
        except:
            versions_info.append("• Pandas (no instalado)")

        try:
            import sklearn
            versions_info.append(f"• Scikit-learn {sklearn.__version__}")
        except:
            versions_info.append("• Scikit-learn (no instalado)")

        try:
            import matplotlib
            versions_info.append(f"• Matplotlib {matplotlib.__version__}")
        except:
            versions_info.append("• Matplotlib (no instalado)")

        versions_text = "<br>".join(versions_info)

        # Estado de módulos
        supervisado_estado = "✅ Disponible" if SUPERVISADO_AVAILABLE else "❌ No disponible"
        no_supervisado_estado = "✅ Disponible" if NO_SUPERVISADO_AVAILABLE else "❌ No disponible"

        about_text = f"""
        <h2>💧 Sistema ML - Calidad del Agua</h2>
        <p><b>Versión:</b> 2.1</p>
        <p><b>Desarrollado para:</b> Análisis avanzado de calidad del agua</p>
        
        <h3>🛠️ Tecnologías:</h3>
        <ul>
            <li>• Python {sys.version.split()[0]}</li>
            <li>• PyQt5 - Interfaz gráfica</li>
            {versions_text}
        </ul>
        
        <h3>📊 Estado de Módulos:</h3>
        <ul>
            <li>• Aprendizaje Supervisado: {supervisado_estado}</li>
            <li>• Aprendizaje No Supervisado: {no_supervisado_estado}</li>
            <li>• Gestión de Datos: {"✅ Disponible" if DATA_MANAGER_AVAILABLE else "❌ No disponible"}</li>
        </ul>
        
        <h3>✨ Características principales:</h3>
        <ul>
            <li>• Procesamiento paralelo optimizado</li>
            <li>• Gestión centralizada de datos</li>
            <li>• Visualizaciones interactivas avanzadas</li>
            <li>• Algoritmos de clustering optimizados</li>
            <li>• PCA lineal y no lineal</li>
            <li>• Detección automática de outliers</li>
            <li>• Exportación de modelos y resultados</li>
        </ul>
        
        <h3>🔍 Novedades en No Supervisado:</h3>
        <ul>
            <li>• K-Means con optimización automática de K</li>
            <li>• Clustering jerárquico con múltiples métricas</li>
            <li>• DBSCAN con búsqueda automática de parámetros</li>
            <li>• PCA avanzado con análisis de contribuciones</li>
            <li>• Análisis exploratorio completo</li>
        </ul>
        
        <p><i>Sistema optimizado para análisis eficiente de grandes volúmenes de datos de calidad del agua</i></p>
        """

        msg = QMessageBox(self)
        msg.setWindowTitle("ℹ️ Acerca de")
        msg.setTextFormat(Qt.RichText)
        msg.setText(about_text)
        msg.setIcon(QMessageBox.Information)
        msg.exec_()

    def on_regresar_menu(self):
        """Handler para regresar al menú principal"""
        # Cerrar ventanas secundarias si están abiertas
        if self.supervisado_window:
            self.supervisado_window.close()
        if self.no_supervisado_window:
            self.no_supervisado_window.close()

        # Emitir señal para regresar
        self.regresar_menu.emit()

        # Cerrar esta ventana
        self.close()

    def closeEvent(self, event):
        """Manejar cierre de la ventana"""
        # Cerrar ventanas secundarias
        for window in [self.supervisado_window, self.no_supervisado_window]:
            if window:
                window.close()

        # Emitir señal de cierre
        self.ventana_cerrada.emit()
        event.accept()


# ==================== FUNCIÓN PRINCIPAL ====================

def main():
    """Función principal para ejecutar el sistema ML"""
    app = QApplication(sys.argv)

    # Configurar estilos de la aplicación
    app.setStyle("Fusion")
    app.setApplicationName("Sistema ML - Calidad del Agua")

    # Configurar paleta moderna
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(245, 247, 250))
    palette.setColor(QPalette.WindowText, QColor(45, 55, 72))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(248, 249, 250))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(45, 55, 72))
    palette.setColor(QPalette.Text, QColor(45, 55, 72))
    palette.setColor(QPalette.Button, QColor(255, 255, 255))
    palette.setColor(QPalette.ButtonText, QColor(45, 55, 72))
    palette.setColor(QPalette.BrightText, QColor(255, 255, 255))
    palette.setColor(QPalette.Link, QColor(59, 130, 246))
    palette.setColor(QPalette.Highlight, QColor(59, 130, 246))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))

    app.setPalette(palette)

    # Crear y mostrar ventana
    window = SegmentacionML()
    window.show()

    # Mensaje de bienvenida
    QTimer.singleShot(500, lambda: print("""
    ========================================
    🧠 Sistema ML - Calidad del Agua v2.1
    ========================================
    ✅ Sistema iniciado correctamente
    
    Módulos disponibles:
    - Aprendizaje Supervisado: {}
    - Aprendizaje No Supervisado: {}
    - Gestión de Datos: {}
    
    Nuevas características:
    - Clustering optimizado automático
    - PCA avanzado con kernels
    - Análisis exploratorio completo
    - Detección inteligente de outliers
    ========================================
    """.format(
        "✅ Disponible" if SUPERVISADO_AVAILABLE else "❌ No disponible",
        "✅ Disponible" if NO_SUPERVISADO_AVAILABLE else "❌ No disponible",
        "✅ Disponible" if DATA_MANAGER_AVAILABLE else "❌ No disponible"
    )))

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()