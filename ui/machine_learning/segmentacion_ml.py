"""
segmentacion_ml.py - VERSIÓN CORREGIDA SIN DEEP LEARNING
Sistema ML para Análisis de Calidad del Agua - Compatible con PyInstaller
Solo módulos Supervisado y No Supervisado
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
    print("✅ DataManager disponible")
except ImportError:
    try:
        from .data_manager import get_data_manager
        DATA_MANAGER_AVAILABLE = True
        print("✅ DataManager disponible (fallback)")
    except ImportError:
        DATA_MANAGER_AVAILABLE = False
        print("⚠️ DataManager no disponible")

# Importar ventana de supervisado compatible
try:
    from .supervisado_window import SupervisadoWindowCompatible
    SUPERVISADO_AVAILABLE = True
    print("✅ Módulo Supervisado Compatible disponible")
except ImportError:
    try:
        # Fallback a la versión original
        from .supervisado_window import SupervisadoWindow as SupervisadoWindowCompatible
        SUPERVISADO_AVAILABLE = True
        print("✅ Módulo Supervisado Original disponible")
    except ImportError:
        SUPERVISADO_AVAILABLE = False
        print("⚠️ Módulo Supervisado no disponible")

# Importar ventana de no supervisado - CORREGIDO
print("Intentando cargar módulo No Supervisado...")
try:
    from .no_supervisado_window import NoSupervisadoWindow
    NO_SUPERVISADO_AVAILABLE = True
    print("✅ Módulo No Supervisado cargado exitosamente")
except ImportError as e:
    NO_SUPERVISADO_AVAILABLE = False
    print(f"❌ Error cargando módulo No Supervisado: {e}")
    print("Traceback completo:")
    traceback.print_exc()

# Verificación final de módulos
print(f"\n=== ESTADO FINAL DE MÓDULOS ===")
print(f"DataManager: {'✅' if DATA_MANAGER_AVAILABLE else '❌'}")
print(f"Supervisado: {'✅' if SUPERVISADO_AVAILABLE else '❌'}")
print(f"No Supervisado: {'✅' if NO_SUPERVISADO_AVAILABLE else '❌'}")
print(f"================================\n")


class ModernButtonCompatible(QFrame):
    """Botón moderno compatible con PyInstaller"""
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
        if shadow:
            shadow.setBlurRadius(25)
            shadow.setYOffset(8)
            shadow.setColor(QColor(0, 0, 0, 50))

    def leaveEvent(self, event):
        """Efecto al quitar el mouse"""
        shadow = self.graphicsEffect()
        if shadow:
            shadow.setBlurRadius(20)
            shadow.setYOffset(5)
            shadow.setColor(QColor(0, 0, 0, 30))


class SegmentacionMLCompatible(QWidget):
    """Ventana principal del sistema ML - Compatible con PyInstaller"""

    # Señales
    ventana_cerrada = pyqtSignal()
    regresar_menu = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.supervisado_window = None
        self.no_supervisado_window = None

        # Inicializar data manager de forma segura
        if DATA_MANAGER_AVAILABLE:
            try:
                self.data_manager = get_data_manager()
                print("✅ DataManager inicializado")
            except Exception as e:
                self.data_manager = None
                print(f"⚠️ Error inicializando DataManager: {e}")
        else:
            self.data_manager = None

        self.init_ui()
        self.apply_theme()

        # Verificar datos al iniciar
        self.check_data_status()

    def init_ui(self):
        """Inicializar interfaz de usuario compatible"""
        self.setWindowTitle("🧠 Sistema Machine Learning - Compatible PyInstaller")
        self.setMinimumSize(1000, 700)

        # Layout principal con scroll
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

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

        # Título principal
        title_container = QFrame()
        title_container.setObjectName("titleContainer")
        title_layout = QVBoxLayout(title_container)

        main_title = QLabel("Sistema de Machine Learning Compatible")
        main_title.setObjectName("mainTitle")
        main_title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("Análisis Avanzado de Calidad del Agua - PyInstaller Ready")
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

        # Estado general
        deps_ok = SUPERVISADO_AVAILABLE or NO_SUPERVISADO_AVAILABLE
        status_icon = QLabel("✅" if deps_ok else "⚠️")
        status_text = QLabel("Sistema Compatible PyInstaller" if deps_ok else "Módulos no disponibles")
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

        return status_frame

    def create_main_content(self):
        """Crear contenido principal con módulos disponibles"""
        content = QFrame()
        content.setObjectName("mainContent")

        main_layout = QVBoxLayout(content)
        main_layout.setSpacing(30)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setAlignment(Qt.AlignCenter)  # 🔹 Centrar verticalmente todo el contenido

        # Descripción
        desc_label = QLabel("Selecciona el tipo de análisis que deseas realizar:")
        desc_label.setObjectName("description")
        desc_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(desc_label)

        # Contenedor de botones
        buttons_container = QFrame()
        buttons_layout = QHBoxLayout(buttons_container)
        buttons_layout.setSpacing(20)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setAlignment(Qt.AlignCenter)  # 🔹 Centrar horizontalmente los botones

        # Configuración de módulos
        modules = []

        # Módulo Supervisado
        if SUPERVISADO_AVAILABLE:
            modules.append({
                "title": "Aprendizaje Supervisado",
                "icon": "🎯",
                "description": "Predicción y clasificación con datos etiquetados",
                "features": "Regresión • SVM • Árboles • Implementación NumPy",
                "color": "#3498db",
                "status": "✅ Compatible PyInstaller",
                "action": self.open_supervisado
            })

        # Módulo No Supervisado
        if NO_SUPERVISADO_AVAILABLE:
            modules.append({
                "title": "Aprendizaje No Supervisado",
                "icon": "🔍",
                "description": "Descubrimiento de patrones sin etiquetas",
                "features": "K-Means • DBSCAN • PCA • Clustering",
                "color": "#9b59b6",
                "status": "✅ Gráficas Corregidas",
                "action": self.open_no_supervisado
            })

        # Si no hay módulos disponibles
        if not modules:
            modules.append({
                "title": "No hay módulos disponibles",
                "icon": "⚠️",
                "description": "Verifica la instalación de dependencias",
                "features": "pip install scikit-learn matplotlib numpy pandas",
                "color": "#95a5a6",
                "status": "❌ Error de instalación",
                "action": lambda: QMessageBox.warning(
                    self, "Error",
                    "No hay módulos disponibles.\n\n"
                    "Instala las dependencias:\n"
                    "pip install scikit-learn matplotlib numpy pandas"
                )
            })

        # Crear botones
        for module_config in modules:
            button = ModernButtonCompatible(module_config)
            button.clicked.connect(module_config["action"])
            buttons_layout.addWidget(button)

        main_layout.addWidget(buttons_container, alignment=Qt.AlignCenter)

        # Información adicional
        info_frame = self.create_info_section()
        main_layout.addWidget(info_frame, alignment=Qt.AlignCenter)

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
            ("🚀", "Alto Rendimiento", "Implementaciones optimizadas en NumPy"),
            ("📊", "Visualizaciones", "Gráficos detallados con PyInstaller"),
            ("🔧", "Sin Dependencias", "Evita librerías problemáticas"),
            ("💾", "Ejecutables", "Genera .exe sin conflictos")
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
        """Aplicar tema moderno compatible"""
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
                font-size: 32px;
                font-weight: 700;
                color: white;
                margin-bottom: 5px;
            }
            
            #subtitle {
                font-size: 16px;
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
        """Abrir módulo de aprendizaje supervisado compatible"""
        if not SUPERVISADO_AVAILABLE:
            QMessageBox.warning(
                self,
                "Módulo no disponible",
                "El módulo de Aprendizaje Supervisado no está instalado.\n\n"
                "Verifica que el archivo 'supervisado_window.py' esté disponible."
            )
            return

        # Verificar datos cargados
        if not self.data_manager or not self.data_manager.has_data():
            reply = QMessageBox.question(
                self,
                "Sin datos cargados",
                "No hay datos cargados en el sistema.\n\n"
                "¿Deseas cargar datos de demostración?",
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
                        "Ahora puedes explorar el módulo ML Supervisado Compatible."
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

        # Crear o mostrar ventana supervisado compatible
        try:
            if self.supervisado_window is None:
                self.supervisado_window = SupervisadoWindowCompatible()
                self.supervisado_window.destroyed.connect(
                    lambda: setattr(self, 'supervisado_window', None)
                )

            self.supervisado_window.show()
            self.supervisado_window.raise_()
            self.supervisado_window.activateWindow()

        except Exception as e:
            print(f"Error abriendo SupervisadoWindowCompatible:\n{traceback.format_exc()}")
            QMessageBox.critical(
                self,
                "Error al abrir módulo",
                f"No se pudo abrir el módulo supervisado compatible:\n{str(e)}"
            )

    def open_no_supervisado(self):
        """Abrir módulo de aprendizaje no supervisado"""
        if not NO_SUPERVISADO_AVAILABLE:
            QMessageBox.warning(
                self,
                "Módulo no disponible",
                "El módulo de Aprendizaje No Supervisado no está disponible.\n\n"
                "Verifica las dependencias:\n"
                "pip install scikit-learn matplotlib numpy pandas"
            )
            return

        # Verificar datos
        if not self.data_manager or not self.data_manager.has_data():
            reply = QMessageBox.question(
                self,
                "Sin datos cargados",
                "¿Deseas cargar datos de demostración?",
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
                        "Datos de demostración generados para clustering y PCA.\n"
                        "Incluye gráficas detalladas corregidas."
                    )
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Error: {str(e)}")
                    return
            else:
                return

        try:
            if self.no_supervisado_window is None:
                self.no_supervisado_window = NoSupervisadoWindow()
                self.no_supervisado_window.destroyed.connect(
                    lambda: setattr(self, 'no_supervisado_window', None)
                )

            self.no_supervisado_window.show()
            self.no_supervisado_window.raise_()
            self.no_supervisado_window.activateWindow()

        except Exception as e:
            print(f"Error abriendo NoSupervisadoWindow:\n{traceback.format_exc()}")
            QMessageBox.critical(
                self,
                "Error al abrir módulo",
                f"Error en módulo No Supervisado:\n{str(e)}"
            )

    def show_help(self):
        """Mostrar ayuda del sistema compatible"""
        help_text = """
        <h2>🧠 Sistema ML Compatible con PyInstaller</h2>
        
        <h3>✅ Características de compatibilidad:</h3>
        <ul>
            <li>• <b>Módulo Supervisado:</b> Implementaciones propias en NumPy</li>
            <li>• <b>Módulo No Supervisado:</b> Con gráficas detalladas corregidas</li>
            <li>• <b>Dependencias mínimas:</b> Solo librerías esenciales</li>
            <li>• <b>Ejecutables optimizados:</b> Sin conflictos</li>
        </ul>
        
        <h3>🎯 Aprendizaje Supervisado:</h3>
        <ul>
            <li>• <b>Regresión Lineal:</b> Implementación NumPy pura</li>
            <li>• <b>Árboles de Decisión:</b> Algoritmo CART</li>
            <li>• <b>SVM:</b> Descenso por gradiente</li>
            <li>• <b>Visualizaciones:</b> Compatibles con PyInstaller</li>
        </ul>
        
        <h3>🔍 Aprendizaje No Supervisado:</h3>
        <ul>
            <li>• <b>K-Means:</b> Con evaluación de K óptimo</li>
            <li>• <b>DBSCAN:</b> Detección de outliers</li>
            <li>• <b>PCA:</b> Análisis de componentes principales</li>
            <li>• <b>Gráficas:</b> Scatter plots reales con datos</li>
        </ul>
        
        <h3>📦 Generar ejecutable:</h3>
        <p><code>pyinstaller CalidadAgua_MINIMAL_ML.spec</code></p>
        
        <h3>💡 Ventajas:</h3>
        <ul>
            <li>• Gráficas detalladas con puntos de datos reales</li>
            <li>• Proyecciones PCA automáticas para visualización</li>
            <li>• Detección visual de outliers y clusters</li>
            <li>• Ejecutables pequeños y rápidos</li>
        </ul>
        
        <p><b>Nota:</b> Sistema optimizado para generar ejecutables estables 
        con visualizaciones de Machine Learning detalladas.</p>
        """

        msg = QMessageBox(self)
        msg.setWindowTitle("Ayuda - Sistema Compatible")
        msg.setTextFormat(Qt.RichText)
        msg.setText(help_text)
        msg.setIcon(QMessageBox.Information)
        msg.exec_()

    def show_about(self):
        """Mostrar información del sistema compatible"""
        # Verificar qué está disponible
        supervisado_estado = "Compatible" if SUPERVISADO_AVAILABLE else "No disponible"
        no_supervisado_estado = "Disponible" if NO_SUPERVISADO_AVAILABLE else "No disponible"

        about_text = f"""
        <h2>Sistema ML Compatible - PyInstaller Ready</h2>
        <p><b>Versión:</b> 3.0 - Compatible</p>
        <p><b>Objetivo:</b> Generar ejecutables sin conflictos</p>
        
        <h3>Tecnologías Core:</h3>
        <ul>
            <li>• Python {sys.version.split()[0]}</li>
            <li>• PyQt5 - Interfaz gráfica</li>
            <li>• NumPy - Cálculos matriciales</li>
            <li>• Pandas - Manejo de datos</li>
            <li>• Matplotlib - Visualizaciones</li>
            <li>• Scikit-learn - ML No Supervisado</li>
        </ul>
        
        <h3>Estado de Módulos:</h3>
        <ul>
            <li>• Supervisado Compatible: {supervisado_estado}</li>
            <li>• No Supervisado: {no_supervisado_estado}</li>
        </ul>
        
        <h3>Optimizaciones PyInstaller:</h3>
        <ul>
            <li>• Dependencias controladas y optimizadas</li>
            <li>• Gráficas corregidas para ejecutables</li>
            <li>• Manejo de memoria optimizado</li>
            <li>• Compatible con Windows/Linux/Mac</li>
        </ul>
        
        <h3>Generación de ejecutable:</h3>
        <ul>
            <li>• Tamaño estimado: 200-300 MB</li>
            <li>• Tiempo de carga: 5-10 segundos</li>
            <li>• Sin instalación de dependencias</li>
            <li>• Visualizaciones ML completamente funcionales</li>
        </ul>
        
        <p><i>Versión optimizada para distribución como ejecutable independiente</i></p>
        """

        msg = QMessageBox(self)
        msg.setWindowTitle("Acerca de - Compatible")
        msg.setTextFormat(Qt.RichText)
        msg.setText(about_text)
        msg.setIcon(QMessageBox.Information)
        msg.exec_()

    def on_regresar_menu(self):
        """Handler para regresar al menú principal"""
        # Cerrar ventanas secundarias
        for window in [self.supervisado_window, self.no_supervisado_window]:
            if window:
                window.close()

        self.regresar_menu.emit()
        self.close()

    def closeEvent(self, event):
        """Manejar cierre de la ventana"""
        # Cerrar ventanas secundarias
        for window in [self.supervisado_window, self.no_supervisado_window]:
            if window:
                window.close()

        self.ventana_cerrada.emit()
        event.accept()


# ==================== USAR LA CLASE COMPATIBLE ====================

# Asignar la clase compatible como la clase principal
SegmentacionML = SegmentacionMLCompatible

# ==================== FUNCIÓN PRINCIPAL ====================

def main():
    """Función principal para ejecutar el sistema ML compatible"""
    app = QApplication(sys.argv)

    # Configurar estilos de la aplicación
    app.setStyle("Fusion")
    app.setApplicationName("Sistema ML Compatible - PyInstaller")

    # Crear y mostrar ventana
    window = SegmentacionMLCompatible()
    window.show()

    # Mensaje de bienvenida
    QTimer.singleShot(500, lambda: print("""
    ==========================================
    Sistema ML Compatible v3.0 - PyInstaller
    ==========================================
    Sistema iniciado correctamente
    
    Módulos compatibles:
    - Supervisado Compatible: {}
    - No Supervisado: {}
    
    Características PyInstaller:
    - Implementaciones NumPy optimizadas
    - Gráficas detalladas corregidas
    - Dependencias mínimas controladas
    - Ejecutables estables y funcionales
    ==========================================
    """.format(
        "Disponible" if SUPERVISADO_AVAILABLE else "No disponible",
        "Disponible" if NO_SUPERVISADO_AVAILABLE else "No disponible"
    )))

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()