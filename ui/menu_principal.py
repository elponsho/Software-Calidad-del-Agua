from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFrame, QGridLayout, QSpacerItem, QSizePolicy)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon


class MenuPrincipal(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Análisis de Calidad del Agua")
        self.setMinimumSize(900, 650)
        self.setup_ui()

    def setup_ui(self):
        # Estilo moderno y profesional
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8f9fa, stop:1 #e9ecef);
                font-family: 'Segoe UI', Arial, sans-serif;
            }

            .main-title {
                font-size: 28px;
                font-weight: bold;
                color: #1a365d;
                margin: 20px 0;
                text-align: center;
                background: none;
            }

            .subtitle {
                font-size: 16px;
                color: #4a5568;
                text-align: center;
                margin-bottom: 30px;
                background: none;
            }

            .card-button {
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

            .card-button:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e6fffa, stop:1 #b2f5ea);
                border-color: #38b2ac;
                transform: translateY(-2px);
            }

            .card-button:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #b2f5ea, stop:1 #81e6d9);
            }

            .card-title {
                font-size: 18px;
                font-weight: bold;
                color: #1a365d;
                margin-bottom: 8px;
                background: none;
            }

            .card-description {
                font-size: 14px;
                color: #4a5568;
                line-height: 1.4;
                background: none;
            }

            .card-icon {
                font-size: 48px;
                margin-bottom: 15px;
                background: none;
            }

            .footer-text {
                font-size: 12px;
                color: #718096;
                text-align: center;
                margin-top: 30px;
                background: none;
            }
        """)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(40, 30, 40, 30)

        # Header con título y subtítulo
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
        # Título principal
        title = QLabel("Sistema de Análisis de Calidad del Agua")
        title.setProperty("class", "main-title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Subtítulo
        subtitle = QLabel("Herramienta profesional para el análisis estadístico y visualización de datos ambientales")
        subtitle.setProperty("class", "subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

    def create_option_cards(self, layout):
        # Container para las cards
        cards_layout = QGridLayout()
        cards_layout.setSpacing(20)

        # Card 1: Preprocesamiento
        self.btn_prepro = self.create_card(
            "📊",
            "Preprocesamiento de Datos",
            "Carga y análisis exploratorio de datos.\nGeneración de estadísticas descriptivas,\nvisualizaciones básicas y resúmenes."
        )

        # Card 2: Machine Learning
        self.btn_ml = self.create_card(
            "🤖",
            "Análisis con Machine Learning",
            "Aplicación de algoritmos de aprendizaje\nautomático para clasificación y predicción\nde calidad del agua."
        )

        # Card 3: Deep Learning
        self.btn_dl = self.create_card(
            "🧠",
            "Análisis con Deep Learning",
            "Técnicas avanzadas de redes neuronales\npara análisis complejos y predicciones\nde alta precisión."
        )

        cards_layout.addWidget(self.btn_prepro, 0, 0)
        cards_layout.addWidget(self.btn_ml, 0, 1)
        cards_layout.addWidget(self.btn_dl, 0, 2)

        layout.addLayout(cards_layout)

    def create_card(self, icon, title, description):
        card = QPushButton()
        card.setProperty("class", "card-button")

        # Layout interno de la card
        card_layout = QVBoxLayout()
        card_layout.setAlignment(Qt.AlignTop)

        # Icono
        icon_label = QLabel(icon)
        icon_label.setProperty("class", "card-icon")
        icon_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(icon_label)

        # Título
        title_label = QLabel(title)
        title_label.setProperty("class", "card-title")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setWordWrap(True)
        card_layout.addWidget(title_label)

        # Descripción
        desc_label = QLabel(description)
        desc_label.setProperty("class", "card-description")
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
        footer.setProperty("class", "footer-text")
        footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer)