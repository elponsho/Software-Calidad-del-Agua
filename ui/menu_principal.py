from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFrame, QGridLayout, QSpacerItem, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QIcon

# Importar sistema de temas
try:
    from darkmode.theme_manager import ThemedWidget, ThemeManager
except ImportError:
    try:
        from darkmode import ThemedWidget, ThemeManager
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


class MenuPrincipal(QWidget, ThemedWidget):
    """Menú principal del sistema con soporte para temas"""

    # Señales para navegación
    abrir_carga_datos = pyqtSignal()
    abrir_machine_learning = pyqtSignal()
    abrir_deep_learning = pyqtSignal()
    abrir_wqi = pyqtSignal()  # Señal para WQI

    def __init__(self):
        QWidget.__init__(self)
        ThemedWidget.__init__(self)
        self.setWindowTitle("Sistema de Análisis de Calidad del Agua")
        self.setMinimumSize(900, 650)
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(40, 30, 40, 30)

        # Header con título, subtítulo y botón de tema
        self.create_header(main_layout)

        # Espaciador
        main_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Fixed))

        # Cards de opciones
        self.create_option_cards(main_layout)

        # Espaciador flexible
        main_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Footer
        self.create_footer(main_layout)

        self.setLayout(main_layout)

    def create_header(self, layout):
        """Crear header con título y botón de tema"""
        header_frame = QFrame()
        header_frame.setObjectName("headerFrame")

        header_layout = QVBoxLayout()

        # Fila superior con botón de tema
        top_row = QHBoxLayout()
        top_row.addStretch()

        # Botón de tema
        self.theme_button = QPushButton("🌙")
        self.theme_button.setObjectName("themeButton")
        self.theme_button.setFixedSize(45, 45)
        self.theme_button.clicked.connect(self.toggle_theme)
        self.theme_button.setToolTip("Cambiar tema claro/oscuro")
        top_row.addWidget(self.theme_button)

        header_layout.addLayout(top_row)

        # Título principal
        title = QLabel("Sistema de Análisis de Calidad del Agua")
        title.setObjectName("mainTitle")
        title.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title)

        # Subtítulo
        subtitle = QLabel("Herramienta profesional para el análisis estadístico y visualización de datos ambientales")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        header_layout.addWidget(subtitle)

        header_frame.setLayout(header_layout)
        layout.addWidget(header_frame)

    def create_option_cards(self, layout):
        # Container para las cards
        cards_layout = QGridLayout()
        cards_layout.setSpacing(20)

        # Card 1: Preprocesamiento
        self.btn_prepro = self.create_card(
            "📊",
            "Preprocesamiento de Datos",
            "Carga y análisis exploratorio de datos.\nGeneración de estadísticas descriptivas,\nvisualizaciones básicas y resúmenes.",
            self.abrir_carga_datos.emit
        )

        # Card 2: Machine Learning
        self.btn_ml = self.create_card(
            "🤖",
            "Análisis con Machine Learning",
            "Aplicación de algoritmos de aprendizaje\nautomático para clasificación y predicción\nde calidad del agua.",
            self.abrir_machine_learning.emit
        )

        # Card 3: Deep Learning
        self.btn_dl = self.create_card(
            "🧠",
            "Análisis con Deep Learning",
            "Técnicas avanzadas de redes neuronales\npara análisis complejos y predicciones\nde alta precisión.",
            self.abrir_deep_learning.emit
        )

        # Card 4: Ecuación WQI
        self.btn_wqi = self.create_card(
            "💧",
            "Índice de Calidad del Agua (WQI)",
            "Cálculo del índice WQI mediante\nponderación de parámetros fisicoquímicos\ny bacteriológicos del agua.",
            self.abrir_wqi.emit
        )

        # Ajustar el layout para 2x2
        cards_layout.addWidget(self.btn_prepro, 0, 0)
        cards_layout.addWidget(self.btn_ml, 0, 1)
        cards_layout.addWidget(self.btn_dl, 1, 0)
        cards_layout.addWidget(self.btn_wqi, 1, 1)

        layout.addLayout(cards_layout)

    def create_card(self, icon, title, description, action):
        card = QPushButton()
        card.setObjectName("cardButton")
        card.clicked.connect(action)

        # Layout interno de la card
        card_layout = QVBoxLayout()
        card_layout.setAlignment(Qt.AlignTop)

        # Icono
        icon_label = QLabel(icon)
        icon_label.setObjectName("cardIcon")
        icon_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(icon_label)

        # Título
        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setWordWrap(True)
        card_layout.addWidget(title_label)

        # Descripción
        desc_label = QLabel(description)
        desc_label.setObjectName("cardDescription")
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setWordWrap(True)
        card_layout.addWidget(desc_label)

        # Widget container para el layout
        container = QWidget()
        container.setLayout(card_layout)

        # Layout del botón
        button_layout = QVBoxLayout()
        button_layout.addWidget(container)
        card.setLayout(button_layout)

        return card

    def create_footer(self, layout):
        footer = QLabel("© 2025 Sistema de Análisis de Calidad del Agua - Versión 1.0")
        footer.setObjectName("footerText")
        footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer)

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

    def apply_light_theme(self):
        """Aplicar tema claro personalizado"""
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8f9fa, stop:1 #e9ecef);
                font-family: 'Segoe UI', Arial, sans-serif;
            }

            #headerFrame {
                background: rgba(255, 255, 255, 0.8);
                border: 1px solid #dee2e6;
                border-radius: 15px;
                margin-bottom: 20px;
                padding: 20px;
            }

            #themeButton {
                background: #f8f9fa;
                border: 2px solid #dee2e6;
                border-radius: 22px;
                color: #495057;
                font-size: 18px;
                font-weight: bold;
            }

            #themeButton:hover {
                background: #e9ecef;
                border-color: #adb5bd;
            }

            #mainTitle {
                font-size: 28px;
                font-weight: bold;
                color: #1a365d;
                margin: 20px 0;
                text-align: center;
                background: none;
            }

            #subtitle {
                font-size: 16px;
                color: #4a5568;
                text-align: center;
                margin-bottom: 30px;
                background: none;
            }

            #cardButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff, stop:1 #f1f5f9);
                border: 2px solid #e2e8f0;
                border-radius: 15px;
                padding: 25px;
                margin: 10px;
                min-height: 150px;
                font-size: 16px;
                font-weight: 600;
                color: #2d3748;
                text-align: left;
            }

            #cardButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e6fffa, stop:1 #b2f5ea);
                border-color: #38b2ac;
                transform: translateY(-2px);
            }

            #cardButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #b2f5ea, stop:1 #81e6d9);
            }

            #cardTitle {
                font-size: 18px;
                font-weight: bold;
                color: #1a365d;
                margin-bottom: 8px;
                background: none;
            }

            #cardDescription {
                font-size: 14px;
                color: #4a5568;
                line-height: 1.4;
                background: none;
            }

            #cardIcon {
                font-size: 48px;
                margin-bottom: 15px;
                background: none;
            }

            #footerText {
                font-size: 12px;
                color: #718096;
                text-align: center;
                margin-top: 30px;
                background: none;
            }
        """)

    def apply_dark_theme(self):
        """Aplicar tema oscuro personalizado"""
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2d3748, stop:1 #1a202c);
                font-family: 'Segoe UI', Arial, sans-serif;
            }

            #headerFrame {
                background: rgba(45, 55, 72, 0.8);
                border: 1px solid #4a5568;
                border-radius: 15px;
                margin-bottom: 20px;
                padding: 20px;
            }

            #themeButton {
                background: #4a5568;
                border: 2px solid #718096;
                border-radius: 22px;
                color: #f7fafc;
                font-size: 18px;
                font-weight: bold;
            }

            #themeButton:hover {
                background: #718096;
                border-color: #a0aec0;
            }

            #mainTitle {
                font-size: 28px;
                font-weight: bold;
                color: #f7fafc;
                margin: 20px 0;
                text-align: center;
                background: none;
            }

            #subtitle {
                font-size: 16px;
                color: #cbd5e0;
                text-align: center;
                margin-bottom: 30px;
                background: none;
            }

            #cardButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4a5568, stop:1 #2d3748);
                border: 2px solid #718096;
                border-radius: 15px;
                padding: 25px;
                margin: 10px;
                min-height: 150px;
                font-size: 16px;
                font-weight: 600;
                color: #f7fafc;
                text-align: left;
            }

            #cardButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #68d391, stop:1 #48bb78);
                border-color: #68d391;
                transform: translateY(-2px);
                color: #1a202c;
            }

            #cardButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #48bb78, stop:1 #38a169);
            }

            #cardTitle {
                font-size: 18px;
                font-weight: bold;
                color: #f7fafc;
                margin-bottom: 8px;
                background: none;
            }

            #cardDescription {
                font-size: 14px;
                color: #cbd5e0;
                line-height: 1.4;
                background: none;
            }

            #cardIcon {
                font-size: 48px;
                margin-bottom: 15px;
                background: none;
            }

            #footerText {
                font-size: 12px;
                color: #a0aec0;
                text-align: center;
                margin-top: 30px;
                background: none;
            }
        """)

    def closeEvent(self, event):
        """Manejar evento de cierre"""
        if hasattr(self, 'theme_manager'):
            self.theme_manager.remove_observer(self)
        event.accept()