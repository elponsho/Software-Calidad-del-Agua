"""
segmentacion_ml.py - Ventana Principal del Sistema ML
Sistema de Machine Learning para Análisis de Calidad del Agua
"""

import sys
import multiprocessing
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                             QHBoxLayout, QFrame, QApplication, QMessageBox,
                             QGridLayout, QSpacerItem, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

# Importar sistema de temas
try:
    from ...darkmode.theme_manager import ThemedWidget, ThemeManager
except ImportError:
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

# Importar ventanas específicas
try:
    from .supervisado_window import SupervisadoWindow
except ImportError:
    try:
        from supervisado_window import SupervisadoWindow
    except ImportError:
        SupervisadoWindow = None

try:
    from .no_supervisado_window import NoSupervisadoWindow
except ImportError:
    try:
        from no_supervisado_window import NoSupervisadoWindow
    except ImportError:
        NoSupervisadoWindow = None

try:
    from .analisis_complementarios import AnalisisComplementarios
except ImportError:
    try:
        from analisis_complementarios import AnalisisComplementarios
    except ImportError:
        AnalisisComplementarios = None

# Verificar dependencias
try:
    import numpy as np
    import pandas as pd
    import sklearn
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


# Versión simplificada de ventana de análisis (en caso de que falten los archivos)
class SimpleAnalysisWindow(QWidget):
    """Ventana simple de análisis como fallback"""

    def __init__(self, title, analysis_type):
        super().__init__()
        self.setWindowTitle(title)
        self.setMinimumSize(800, 600)

        layout = QVBoxLayout()

        # Header
        header = QHBoxLayout()
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        header.addWidget(title_label)
        header.addStretch()

        close_btn = QPushButton("✕")
        close_btn.setStyleSheet("""
            background: #e74c3c; color: white; border: none; 
            border-radius: 15px; font-weight: bold; font-size: 14px;
        """)
        close_btn.setFixedSize(30, 30)
        close_btn.clicked.connect(self.close)
        header.addWidget(close_btn)

        layout.addLayout(header)

        # Contenido
        content_frame = QFrame()
        content_frame.setStyleSheet("""
            background: #f8f9fa; border: 1px solid #dee2e6; 
            border-radius: 10px; padding: 20px; margin: 10px;
        """)

        content_layout = QVBoxLayout(content_frame)

        info_label = QLabel(f"""
        <h2>🚧 Módulo en Desarrollo</h2>
        <p><strong>Tipo de Análisis:</strong> {analysis_type}</p>
        <p><strong>Estado:</strong> Los archivos específicos para este módulo no están disponibles.</p>
        
        <h3>📋 Funcionalidades Planeadas:</h3>
        <ul>
            <li>• Análisis avanzados de machine learning</li>
            <li>• Visualizaciones interactivas</li>
            <li>• Reportes detallados</li>
            <li>• Exportación de resultados</li>
        </ul>
        
        <h3>🔧 Para habilitar este módulo:</h3>
        <ol>
            <li>Asegúrate de que todos los archivos estén en la carpeta machine_learning</li>
            <li>Verifica las dependencias de Python</li>
            <li>Reinstala si es necesario</li>
        </ol>
        
        <p style="color: #666; font-style: italic;">
        Este es un placeholder temporal. La funcionalidad completa estará disponible pronto.
        </p>
        """)

        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-size: 13px; line-height: 1.5;")
        content_layout.addWidget(info_label)

        layout.addWidget(content_frame)
        self.setLayout(layout)


class SegmentacionML(QWidget, ThemedWidget):
    """Ventana principal del sistema ML con navegación a submódulos"""

    # Señal para notificar cuando se cierra la ventana
    ventana_cerrada = pyqtSignal()

    def __init__(self):
        QWidget.__init__(self)
        ThemedWidget.__init__(self)

        self.supervisado_window = None
        self.no_supervisado_window = None
        self.complementarios_window = None

        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):
        """Configurar la interfaz principal"""
        self.setWindowTitle("💧 Sistema Machine Learning - Calidad del Agua")
        self.setMinimumSize(1000, 700)
        self.setMaximumSize(1200, 800)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(25)
        main_layout.setContentsMargins(30, 30, 30, 30)

        # Header con tema
        header_layout = self.create_header()
        main_layout.addLayout(header_layout)

        # Título principal
        title_layout = self.create_title_section()
        main_layout.addLayout(title_layout)

        # Información del sistema
        info_section = self.create_info_section()
        main_layout.addWidget(info_section)

        # Sección principal de botones
        buttons_section = self.create_main_buttons()
        main_layout.addWidget(buttons_section)

        # Spacer
        main_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Footer con controles
        footer_layout = self.create_footer()
        main_layout.addLayout(footer_layout)

        self.setLayout(main_layout)

    def create_header(self):
        """Crear header con botón de tema"""
        header_layout = QHBoxLayout()

        # Logo/Título pequeño
        logo_label = QLabel("🧠 ML System")
        logo_label.setObjectName("logoLabel")

        # Spacer
        header_layout.addWidget(logo_label)
        header_layout.addStretch()

        # Botón de tema
        self.theme_button = QPushButton("🌙")
        self.theme_button.setObjectName("themeButton")
        self.theme_button.setFixedSize(45, 45)
        self.theme_button.clicked.connect(self.toggle_theme)
        self.theme_button.setToolTip("Cambiar tema claro/oscuro")

        header_layout.addWidget(self.theme_button)

        return header_layout

    def create_title_section(self):
        """Crear sección de título principal"""
        title_layout = QVBoxLayout()
        title_layout.setSpacing(15)
        title_layout.setAlignment(Qt.AlignCenter)

        # Título principal
        main_title = QLabel("💧 Sistema Machine Learning")
        main_title.setObjectName("mainTitle")
        main_title.setAlignment(Qt.AlignCenter)

        # Subtítulo
        subtitle = QLabel("Análisis Avanzado de Calidad del Agua")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignCenter)

        # Descripción
        description = QLabel("Selecciona el tipo de análisis que deseas realizar")
        description.setObjectName("description")
        description.setAlignment(Qt.AlignCenter)

        title_layout.addWidget(main_title)
        title_layout.addWidget(subtitle)
        title_layout.addWidget(description)

        return title_layout

    def create_info_section(self):
        """Crear sección de información del sistema"""
        info_frame = QFrame()
        info_frame.setObjectName("infoFrame")
        info_frame.setMaximumHeight(120)

        info_layout = QVBoxLayout(info_frame)
        info_layout.setSpacing(10)
        info_layout.setContentsMargins(20, 15, 20, 15)

        # Estado del sistema
        if DEPS_AVAILABLE:
            status_text = "✅ Sistema listo - Todas las dependencias disponibles"
            status_color = "#4CAF50"
        else:
            status_text = "⚠️ Advertencia - Faltan dependencias de ML"
            status_color = "#FF9800"

        # Verificar si las ventanas están disponibles
        windows_available = all([SupervisadoWindow, NoSupervisadoWindow, AnalisisComplementarios])

        if not windows_available:
            status_text += " | Módulos específicos no disponibles"
            status_color = "#FF9800"

        status_label = QLabel(status_text)
        status_label.setObjectName("statusText")
        status_label.setAlignment(Qt.AlignCenter)
        status_label.setStyleSheet(f"color: {status_color}; font-weight: bold;")

        # Información técnica
        cpu_count = multiprocessing.cpu_count()
        tech_info = QLabel(f"🖥️ CPUs: {cpu_count} | 🚀 Procesamiento paralelo | 📊 Visualizaciones avanzadas")
        tech_info.setObjectName("techInfo")
        tech_info.setAlignment(Qt.AlignCenter)

        info_layout.addWidget(status_label)
        info_layout.addWidget(tech_info)

        return info_frame

    def create_main_buttons(self):
        """Crear sección principal de botones"""
        buttons_frame = QFrame()
        buttons_frame.setObjectName("buttonsFrame")

        main_layout = QVBoxLayout(buttons_frame)
        main_layout.setSpacing(30)
        main_layout.setContentsMargins(40, 30, 40, 30)

        # Grid de botones principales
        grid_layout = QGridLayout()
        grid_layout.setSpacing(25)

        # Definir botones principales
        main_buttons = [
            {
                "title": "🎯 Aprendizaje Supervisado",
                "subtitle": "Predicción y Clasificación",
                "description": "Regresión Multiple • SVM • Random Forest",
                "icon": "🎯",
                "action": self.open_supervisado,
                "color": "#4CAF50"
            },
            {
                "title": "🔍 Aprendizaje No Supervisado",
                "subtitle": "Patrones y Agrupamiento",
                "description": "Clustering • PCA • Análisis Exploratorio",
                "icon": "🔍",
                "action": self.open_no_supervisado,
                "color": "#2196F3"
            },
            {
                "title": "📊 Análisis Complementarios",
                "subtitle": "Evaluación y Optimización",
                "description": "Calidad • Comparación • Optimización",
                "icon": "📊",
                "action": self.open_complementarios,
                "color": "#FF9800"
            }
        ]

        # Crear botones
        for i, button_config in enumerate(main_buttons):
            button_widget = self.create_main_button(button_config)

            if i < 2:  # Primeros dos en fila superior
                grid_layout.addWidget(button_widget, 0, i)
            else:  # Tercero centrado en fila inferior
                grid_layout.addWidget(button_widget, 1, 0, 1, 2)

        main_layout.addLayout(grid_layout)
        return buttons_frame

    def create_main_button(self, config):
        """Crear un botón principal personalizado"""
        button_frame = QFrame()
        button_frame.setObjectName("mainButtonFrame")
        button_frame.setMinimumHeight(180)
        button_frame.setMaximumHeight(200)
        button_frame.setMinimumWidth(300)
        button_frame.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(button_frame)
        layout.setSpacing(12)
        layout.setContentsMargins(25, 20, 25, 20)
        layout.setAlignment(Qt.AlignCenter)

        # Icono grande
        icon_label = QLabel(config["icon"])
        icon_label.setObjectName("buttonIcon")
        icon_label.setAlignment(Qt.AlignCenter)

        # Título
        title_label = QLabel(config["title"])
        title_label.setObjectName("buttonTitle")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setWordWrap(True)

        # Subtítulo
        subtitle_label = QLabel(config["subtitle"])
        subtitle_label.setObjectName("buttonSubtitle")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setWordWrap(True)

        # Descripción
        desc_label = QLabel(config["description"])
        desc_label.setObjectName("buttonDescription")
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setWordWrap(True)

        # Botón de acción
        action_button = QPushButton("Acceder →")
        action_button.setObjectName("actionButton")
        action_button.clicked.connect(config["action"])
        action_button.setMinimumHeight(35)

        layout.addWidget(icon_label)
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addWidget(desc_label)
        layout.addWidget(action_button)

        # Efecto hover
        button_frame.mousePressEvent = lambda event: config["action"]()

        return button_frame

    def create_footer(self):
        """Crear footer con controles adicionales"""
        footer_layout = QHBoxLayout()
        footer_layout.setSpacing(15)

        # Botón de ayuda
        help_button = QPushButton("❓ Ayuda")
        help_button.setObjectName("secondaryButton")
        help_button.clicked.connect(self.show_help)
        help_button.setMaximumWidth(100)

        # Botón de configuración
        config_button = QPushButton("⚙️ Config")
        config_button.setObjectName("secondaryButton")
        config_button.clicked.connect(self.show_config)
        config_button.setMaximumWidth(100)

        # Spacer
        footer_layout.addWidget(help_button)
        footer_layout.addWidget(config_button)
        footer_layout.addStretch()

        # Botón de salir (IMPORTANTE: Este es el que main.py necesita)
        self.exit_button = QPushButton("← Regresar")
        self.exit_button.setObjectName("exitButton")
        self.exit_button.setMaximumWidth(120)

        footer_layout.addWidget(self.exit_button)

        return footer_layout

    def apply_styles(self):
        """Aplicar estilos CSS personalizados"""
        styles = """
        QWidget {
            font-family: 'Segoe UI', Arial, sans-serif;
        }

        #logoLabel {
            font-size: 16px;
            font-weight: bold;
            color: #666;
        }

        #themeButton {
            border: 2px solid #ddd;
            border-radius: 22px;
            background: #f5f5f5;
            font-size: 18px;
        }

        #themeButton:hover {
            background: #e0e0e0;
            border-color: #999;
        }

        #mainTitle {
            font-size: 36px;
            font-weight: bold;
            color: #2c3e50;
            margin: 10px 0;
        }

        #subtitle {
            font-size: 20px;
            color: #34495e;
            font-weight: 500;
        }

        #description {
            font-size: 14px;
            color: #7f8c8d;
            margin-top: 5px;
        }

        #infoFrame {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border: 1px solid #dee2e6;
            border-radius: 10px;
        }

        #statusText {
            font-size: 16px;
            font-weight: bold;
        }

        #techInfo {
            font-size: 12px;
            color: #6c757d;
        }

        #buttonsFrame {
            background: #ffffff;
            border: 1px solid #e9ecef;
            border-radius: 15px;
        }

        #mainButtonFrame {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border: 2px solid #e9ecef;
            border-radius: 15px;
            margin: 5px;
        }

        #mainButtonFrame:hover {
            border-color: #007bff;
            background: linear-gradient(135deg, #f8f9ff 0%, #e6f3ff 100%);
            transform: translateY(-2px);
        }

        #buttonIcon {
            font-size: 48px;
            margin-bottom: 5px;
        }

        #buttonTitle {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
        }

        #buttonSubtitle {
            font-size: 14px;
            color: #6c757d;
            font-weight: 500;
        }

        #buttonDescription {
            font-size: 12px;
            color: #868e96;
            margin-top: 5px;
        }

        #actionButton {
            background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
            color: white;
            border: none;
            border-radius: 6px;
            font-weight: bold;
            font-size: 13px;
            margin-top: 8px;
        }

        #actionButton:hover {
            background: linear-gradient(135deg, #0056b3 0%, #004085 100%);
        }

        #secondaryButton {
            background: #6c757d;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 15px;
            font-size: 12px;
        }

        #secondaryButton:hover {
            background: #5a6268;
        }

        #exitButton {
            background: #28a745;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 15px;
            font-weight: bold;
        }

        #exitButton:hover {
            background: #218838;
        }
        """

        self.setStyleSheet(styles)

    def toggle_theme(self):
        """Alternar tema claro/oscuro"""
        try:
            theme_manager = ThemeManager()
            theme_manager.toggle_theme()

            if theme_manager.is_dark_theme():
                self.theme_button.setText("☀️")
            else:
                self.theme_button.setText("🌙")
        except Exception as e:
            # Si no hay sistema de temas, simplemente cambiar el texto
            if self.theme_button.text() == "🌙":
                self.theme_button.setText("☀️")
            else:
                self.theme_button.setText("🌙")

    def open_supervisado(self):
        """Abrir ventana de aprendizaje supervisado"""
        if not DEPS_AVAILABLE:
            self.show_dependencies_warning()
            return

        if SupervisadoWindow is None:
            # Usar versión simplificada si no está disponible el módulo específico
            if self.supervisado_window is None:
                self.supervisado_window = SimpleAnalysisWindow(
                    "🎯 Aprendizaje Supervisado - ML Calidad del Agua",
                    "Aprendizaje Supervisado"
                )
        else:
            if self.supervisado_window is None:
                self.supervisado_window = SupervisadoWindow()

        self.supervisado_window.show()
        self.supervisado_window.raise_()
        self.supervisado_window.activateWindow()

    def open_no_supervisado(self):
        """Abrir ventana de aprendizaje no supervisado"""
        if not DEPS_AVAILABLE:
            self.show_dependencies_warning()
            return

        if NoSupervisadoWindow is None:
            # Usar versión simplificada si no está disponible el módulo específico
            if self.no_supervisado_window is None:
                self.no_supervisado_window = SimpleAnalysisWindow(
                    "🔍 Aprendizaje No Supervisado - ML Calidad del Agua",
                    "Aprendizaje No Supervisado"
                )
        else:
            if self.no_supervisado_window is None:
                self.no_supervisado_window = NoSupervisadoWindow()

        self.no_supervisado_window.show()
        self.no_supervisado_window.raise_()
        self.no_supervisado_window.activateWindow()

    def open_complementarios(self):
        """Abrir ventana de análisis complementarios"""
        if not DEPS_AVAILABLE:
            self.show_dependencies_warning()
            return

        if AnalisisComplementarios is None:
            # Usar versión simplificada si no está disponible el módulo específico
            if self.complementarios_window is None:
                self.complementarios_window = SimpleAnalysisWindow(
                    "📊 Análisis Complementarios - ML Calidad del Agua",
                    "Análisis Complementarios"
                )
        else:
            if self.complementarios_window is None:
                self.complementarios_window = AnalisisComplementarios()

        self.complementarios_window.show()
        self.complementarios_window.raise_()
        self.complementarios_window.activateWindow()

    def show_dependencies_warning(self):
        """Mostrar advertencia de dependencias faltantes"""
        QMessageBox.warning(
            self,
            "Dependencias Faltantes",
            "Faltan las siguientes dependencias para ML:\n\n"
            "• scikit-learn\n"
            "• pandas\n" 
            "• numpy\n"
            "• matplotlib\n"
            "• scipy\n\n"
            "Instala con: pip install scikit-learn pandas numpy matplotlib scipy"
        )

    def show_help(self):
        """Mostrar ayuda del sistema"""
        windows_status = "disponibles" if all([SupervisadoWindow, NoSupervisadoWindow, AnalisisComplementarios]) else "en desarrollo"

        QMessageBox.information(
            self,
            "💡 Ayuda del Sistema",
            f"🎯 <b>Aprendizaje Supervisado:</b>\n"
            f"   • Regresión múltiple y lineal\n"
            f"   • SVM (Support Vector Machines)\n"
            f"   • Random Forest\n\n"
            f"🔍 <b>Aprendizaje No Supervisado:</b>\n"
            f"   • Clustering (K-Means + Jerárquico)\n"
            f"   • PCA (Análisis de Componentes)\n\n"
            f"📊 <b>Análisis Complementarios:</b>\n"
            f"   • Evaluación de calidad\n"
            f"   • Optimización de parámetros\n"
            f"   • Comparación de algoritmos\n\n"
            f"🔧 <b>Estado de módulos:</b> {windows_status}\n"
            f"📦 <b>Dependencias ML:</b> {'✅ Instaladas' if DEPS_AVAILABLE else '❌ Faltantes'}"
        )

    def show_config(self):
        """Mostrar configuración del sistema"""
        modules_info = []
        for name, module in [("Supervisado", SupervisadoWindow),
                           ("No Supervisado", NoSupervisadoWindow),
                           ("Complementarios", AnalisisComplementarios)]:
            status = "✅ Disponible" if module else "❌ No disponible"
            modules_info.append(f"{name}: {status}")

        QMessageBox.information(
            self,
            "⚙️ Configuración",
            f"<b>Sistema Machine Learning</b>\n\n"
            f"🖥️ CPUs disponibles: {multiprocessing.cpu_count()}\n"
            f"🐍 Python: {sys.version.split()[0]}\n"
            f"📦 Dependencias ML: {'✅ OK' if DEPS_AVAILABLE else '❌ Faltantes'}\n"
            f"🎨 Tema: {'Oscuro' if hasattr(self, 'theme_button') and self.theme_button.text() == '☀️' else 'Claro'}\n\n"
            f"<b>📁 Módulos:</b>\n" + "\n".join(modules_info) + "\n\n"
            f"Para instalar dependencias:\n"
            f"pip install scikit-learn pandas numpy matplotlib scipy"
        )

    def closeEvent(self, event):
        """Manejar cierre de la aplicación"""
        # Cerrar ventanas secundarias si están abiertas
        for window in [self.supervisado_window, self.no_supervisado_window, self.complementarios_window]:
            if window:
                window.close()

        # Emitir señal de cierre
        self.ventana_cerrada.emit()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Configurar multiprocesamiento para Windows
    if sys.platform.startswith('win'):
        multiprocessing.set_start_method('spawn', force=True)

    window = SegmentacionML()
    window.show()

    sys.exit(app.exec_())