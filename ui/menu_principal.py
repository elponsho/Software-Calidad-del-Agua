from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFrame, QGridLayout, QSpacerItem, QSizePolicy,
                             QScrollArea, QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QRect, QEasingCurve
from PyQt5.QtGui import QFont, QIcon, QPainter, QPalette


class AnimatedCard(QPushButton):
    """Tarjeta animada con efectos de hover y click"""

    def __init__(self, icon, title, description, action):
        super().__init__()
        self.action = action
        self.setObjectName("cardButton")
        self.clicked.connect(action)

        # Configurar tamaño fijo para evitar textos cortados
        self.setFixedSize(380, 260)

        # Efecto de sombra
        self.shadow_effect = QGraphicsDropShadowEffect()
        self.shadow_effect.setBlurRadius(15)
        self.shadow_effect.setColor(Qt.lightGray)
        self.shadow_effect.setOffset(0, 5)
        self.setGraphicsEffect(self.shadow_effect)

        # Layout interno
        self.setup_layout(icon, title, description)

        # Estado de hover
        self.is_hovered = False

    def setup_layout(self, icon, title, description):
        """Configurar el layout interno de la card"""
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(15)
        layout.setContentsMargins(25, 30, 25, 30)

        # Icono con contenedor
        icon_container = QFrame()
        icon_container.setObjectName("iconContainer")
        icon_container.setFixedSize(80, 80)

        icon_layout = QVBoxLayout()
        icon_layout.setAlignment(Qt.AlignCenter)
        icon_layout.setContentsMargins(0, 0, 0, 0)

        self.icon_label = QLabel(icon)
        self.icon_label.setObjectName("cardIcon")
        self.icon_label.setAlignment(Qt.AlignCenter)
        icon_layout.addWidget(self.icon_label)

        icon_container.setLayout(icon_layout)
        layout.addWidget(icon_container, 0, Qt.AlignCenter)

        # Título
        self.title_label = QLabel(title)
        self.title_label.setObjectName("cardTitle")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setWordWrap(True)
        layout.addWidget(self.title_label)

        # Descripción
        self.desc_label = QLabel(description)
        self.desc_label.setObjectName("cardDescription")
        self.desc_label.setAlignment(Qt.AlignCenter)
        self.desc_label.setWordWrap(True)
        layout.addWidget(self.desc_label)

        # Spacer para empujar contenido hacia arriba
        layout.addStretch()

        self.setLayout(layout)

    def enterEvent(self, event):
        """Evento cuando el mouse entra en la card"""
        self.is_hovered = True
        self.shadow_effect.setBlurRadius(20)
        self.shadow_effect.setOffset(0, 8)
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Evento cuando el mouse sale de la card"""
        self.is_hovered = False
        self.shadow_effect.setBlurRadius(15)
        self.shadow_effect.setOffset(0, 5)
        self.update()
        super().leaveEvent(event)


class ModernHeader(QFrame):
    """Header moderno con gradiente y mejor tipografía"""

    def __init__(self):
        super().__init__()
        self.setObjectName("modernHeader")
        self.setFixedHeight(140)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(10)
        layout.setContentsMargins(40, 30, 40, 30)

        # Título principal con icono
        title_container = QHBoxLayout()
        title_container.setAlignment(Qt.AlignCenter)
        title_container.setSpacing(15)

        # Icono del sistema
        system_icon = QLabel("💧")
        system_icon.setObjectName("systemIcon")
        title_container.addWidget(system_icon)

        # Título
        title = QLabel("Sistema de Análisis de Calidad del Agua")
        title.setObjectName("modernTitle")
        title_container.addWidget(title)

        layout.addLayout(title_container)

        # Subtítulo mejorado
        subtitle = QLabel(
            "Plataforma integral para análisis estadístico, machine learning y visualización de datos ambientales")
        subtitle.setObjectName("modernSubtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        self.setLayout(layout)


class StatsCard(QFrame):
    """Tarjeta de estadísticas rápidas"""

    def __init__(self, icon, title, subtitle):
        super().__init__()
        self.setObjectName("statsCard")
        self.setFixedSize(120, 80)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(5)
        layout.setContentsMargins(10, 10, 10, 10)

        # Icono
        icon_label = QLabel(icon)
        icon_label.setObjectName("statsIcon")
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)

        # Título
        title_label = QLabel(title)
        title_label.setObjectName("statsTitle")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Subtítulo
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("statsSubtitle")
        subtitle_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle_label)

        self.setLayout(layout)


class MenuPrincipal(QWidget):
    """Menú principal con diseño UX moderno"""

    # Señales para navegación
    abrir_preprocesamiento = pyqtSignal()
    abrir_carga_datos = pyqtSignal()
    abrir_machine_learning = pyqtSignal()
    abrir_deep_learning = pyqtSignal()
    abrir_wqi = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Análisis de Calidad del Agua")
        self.setMinimumSize(1000, 750)
        self.setup_ui()
        self.apply_modern_styles()

    def setup_ui(self):
        # Scroll area para contenido
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setObjectName("mainScroll")

        # Widget principal
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(30)
        main_layout.setContentsMargins(40, 30, 40, 30)

        # Header moderno
        self.header = ModernHeader()
        main_layout.addWidget(self.header)

        # Botón de cargar datos
        self.create_data_button_section(main_layout)

        # Tarjetas de estadísticas rápidas
        self.create_stats_section(main_layout)

        # Espaciador
        main_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Fixed))

        # Sección principal de módulos
        self.create_modules_section(main_layout)

        # Espaciador flexible
        main_layout.addSpacerItem(QSpacerItem(20, 30, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Footer moderno
        self.create_modern_footer(main_layout)

        main_widget.setLayout(main_layout)
        scroll.setWidget(main_widget)

        # Layout principal
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        self.setLayout(layout)

    def create_data_button_section(self, layout):
        """Crear sección del botón de cargar datos"""
        data_container = QFrame()
        data_container.setObjectName("dataContainer")

        data_layout = QHBoxLayout()
        data_layout.setAlignment(Qt.AlignCenter)
        data_layout.setSpacing(15)
        data_layout.setContentsMargins(20, 15, 20, 15)

        # Información sobre datos
        data_info = QLabel("📂 Antes de usar los módulos de análisis")
        data_info.setObjectName("dataInfo")
        data_layout.addWidget(data_info)

        # Botón principal de cargar datos
        load_data_btn = QPushButton("Cargar Datos")
        load_data_btn.setObjectName("loadDataButton")
        load_data_btn.setMinimumHeight(45)
        load_data_btn.setMinimumWidth(150)
        load_data_btn.clicked.connect(self.abrir_carga_datos.emit)
        data_layout.addWidget(load_data_btn)

        # Información adicional
        data_hint = QLabel("• CSV, Excel, JSON")
        data_hint.setObjectName("dataHint")
        data_layout.addWidget(data_hint)

        data_container.setLayout(data_layout)
        layout.addWidget(data_container)

    def create_stats_section(self, layout):
        """Crear sección de estadísticas rápidas"""
        stats_container = QFrame()
        stats_container.setObjectName("statsContainer")

        stats_layout = QHBoxLayout()
        stats_layout.setAlignment(Qt.AlignCenter)
        stats_layout.setSpacing(20)
        stats_layout.setContentsMargins(20, 15, 20, 15)

        # Tarjetas de stats
        stats_cards = [
            ("🎯", "5", "Módulos"),
            ("📊", "∞", "Análisis"),
            ("🚀", "Pro", "Versión"),
            ("💡", "AI", "Powered")
        ]

        for icon, title, subtitle in stats_cards:
            card = StatsCard(icon, title, subtitle)
            stats_layout.addWidget(card)

        stats_container.setLayout(stats_layout)
        layout.addWidget(stats_container)

    def create_modules_section(self, layout):
        """Crear sección principal de módulos"""
        # Título de sección
        section_title = QLabel("Selecciona un Módulo de Análisis")
        section_title.setObjectName("sectionTitle")
        section_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(section_title)

        # Contenedor de cards
        cards_container = QFrame()
        cards_container.setObjectName("cardsContainer")

        cards_layout = QGridLayout()
        cards_layout.setSpacing(30)
        cards_layout.setContentsMargins(20, 20, 20, 20)

        # Crear las cards principales
        cards_data = [
            {
                "icon": "📊",
                "title": "Preprocesamiento de Datos",
                "description": "Carga, limpieza y análisis exploratorio de datos. Incluye estadísticas descriptivas, visualizaciones interactivas y detección de valores atípicos.",
                "action": self.abrir_preprocesamiento.emit
            },
            {
                "icon": "🤖",
                "title": "Machine Learning",
                "description": "Algoritmos de aprendizaje automático para clasificación y predicción. Incluye SVM, Random Forest, regresión y validación cruzada.",
                "action": self.abrir_machine_learning.emit
            },
            {
                "icon": "🧠",
                "title": "Deep Learning",
                "description": "Redes neuronales profundas para análisis avanzados. Incluye CNN, RNN y modelos de predicción de alta precisión.",
                "action": self.abrir_deep_learning.emit
            },
            {
                "icon": "💧",
                "title": "Índice de Calidad WQI",
                "description": "Cálculo automático del Índice de Calidad del Agua mediante ponderación de parámetros fisicoquímicos y bacteriológicos.",
                "action": self.abrir_wqi.emit
            }
        ]

        # Crear y posicionar las cards
        for i, card_data in enumerate(cards_data):
            card = AnimatedCard(
                card_data["icon"],
                card_data["title"],
                card_data["description"],
                card_data["action"]
            )

            row = i // 2
            col = i % 2
            cards_layout.addWidget(card, row, col, Qt.AlignCenter)

        # Configurar stretch
        cards_layout.setColumnStretch(0, 1)
        cards_layout.setColumnStretch(1, 1)

        cards_container.setLayout(cards_layout)
        layout.addWidget(cards_container)

    def create_modern_footer(self, layout):
        """Crear footer moderno"""
        footer_container = QFrame()
        footer_container.setObjectName("modernFooter")

        footer_layout = QVBoxLayout()
        footer_layout.setAlignment(Qt.AlignCenter)
        footer_layout.setSpacing(8)
        footer_layout.setContentsMargins(20, 15, 20, 15)

        # Información principal
        footer_main = QLabel("Sistema de Análisis de Calidad del Agua")
        footer_main.setObjectName("footerMain")
        footer_main.setAlignment(Qt.AlignCenter)
        footer_layout.addWidget(footer_main)

        # Información secundaria
        footer_info = QLabel("Versión 1.0 • © 2025 • Desarrollado con PyQt5 y Python")
        footer_info.setObjectName("footerInfo")
        footer_info.setAlignment(Qt.AlignCenter)
        footer_layout.addWidget(footer_info)

        footer_container.setLayout(footer_layout)
        layout.addWidget(footer_container)

    def apply_modern_styles(self):
        """Aplicar estilos modernos con mejor UX"""
        self.setStyleSheet("""
            /* CONFIGURACIÓN GENERAL */
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f7fafc, stop:0.5 #edf2f7, stop:1 #e2e8f0);
                font-family: 'Segoe UI', 'SF Pro Display', 'Inter', system-ui, sans-serif;
                color: #2d3748;
            }

            /* SCROLL AREA */
            #mainScroll {
                border: none;
                background: transparent;
            }

            QScrollBar:vertical {
                background: rgba(203, 213, 224, 0.3);
                width: 8px;
                border-radius: 4px;
                margin: 0px;
            }

            QScrollBar::handle:vertical {
                background: rgba(113, 128, 150, 0.5);
                border-radius: 4px;
                min-height: 20px;
            }

            QScrollBar::handle:vertical:hover {
                background: rgba(113, 128, 150, 0.8);
            }

            /* HEADER MODERNO */
            #modernHeader {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #667eea, stop:0.5 #764ba2, stop:1 #f093fb);
                border: none;
                border-radius: 20px;
                color: white;
            }

            #systemIcon {
                font-size: 42px;
                margin: 0px;
                padding: 0px;
            }

            #modernTitle {
                font-size: 32px;
                font-weight: 700;
                color: white;
                margin: 0px;
                padding: 0px;
                text-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }

            #modernSubtitle {
                font-size: 16px;
                color: rgba(255, 255, 255, 0.9);
                margin: 0px;
                padding: 0px 20px;
                line-height: 1.5;
                font-weight: 400;
            }

            /* SECCIÓN DE CARGAR DATOS */
            #dataContainer {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e3f2fd, stop:0.5 #f3e5f5, stop:1 #fce4ec);
                border: 2px solid rgba(33, 150, 243, 0.3);
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }

            #dataInfo {
                font-size: 16px;
                font-weight: 600;
                color: #1565c0;
                margin: 0px;
            }

            #loadDataButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2196f3, stop:1 #1976d2);
                border: none;
                border-radius: 12px;
                color: white;
                font-size: 16px;
                font-weight: 600;
                padding: 0px 25px;
                text-shadow: 0 1px 2px rgba(0,0,0,0.1);
            }

            #loadDataButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1976d2, stop:1 #1565c0);
                transform: translateY(-2px);
            }

            #loadDataButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1565c0, stop:1 #0d47a1);
                transform: translateY(0px);
            }

            #dataHint {
                font-size: 14px;
                color: #1976d2;
                font-weight: 500;
                margin: 0px;
            }

            /* ESTADÍSTICAS */
            #statsContainer {
                background: rgba(255, 255, 255, 0.8);
                border: 1px solid rgba(203, 213, 224, 0.5);
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }

            #statsCard {
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(203, 213, 224, 0.3);
                border-radius: 12px;
            }

            #statsIcon {
                font-size: 20px;
                margin: 0px;
            }

            #statsTitle {
                font-size: 14px;
                font-weight: 600;
                color: #2d3748;
                margin: 0px;
            }

            #statsSubtitle {
                font-size: 10px;
                color: #718096;
                margin: 0px;
            }

            /* TÍTULO DE SECCIÓN */
            #sectionTitle {
                font-size: 24px;
                font-weight: 600;
                color: #2d3748;
                margin: 20px 0px;
                padding: 0px;
            }

            /* CONTENEDOR DE CARDS */
            #cardsContainer {
                background: transparent;
                border: none;
            }

            /* CARDS PRINCIPALES */
            #cardButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff, stop:1 #f7fafc);
                border: 2px solid rgba(226, 232, 240, 0.8);
                border-radius: 20px;
                padding: 0px;
                margin: 0px;
                text-align: left;
                font-weight: 500;
            }

            #cardButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e6fffa, stop:1 #b2f5ea);
                border-color: #38b2ac;
                transform: translateY(-4px);
            }

            #cardButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #b2f5ea, stop:1 #81e6d9);
                transform: translateY(-2px);
            }

            /* CONTENIDO DE CARDS */
            #iconContainer {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #667eea, stop:1 #764ba2);
                border: none;
                border-radius: 40px;
            }

            #cardIcon {
                font-size: 36px;
                color: white;
                margin: 0px;
                padding: 0px;
                text-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }

            #cardTitle {
                font-size: 18px;
                font-weight: 600;
                color: #1a365d;
                margin: 0px;
                padding: 0px 10px;
                line-height: 1.3;
                text-align: center;
            }

            #cardDescription {
                font-size: 13px;
                color: #4a5568;
                line-height: 1.4;
                margin: 0px;
                padding: 0px 5px;
                text-align: center;
                font-weight: 400;
            }

            /* FOOTER MODERNO */
            #modernFooter {
                background: rgba(255, 255, 255, 0.6);
                border: 1px solid rgba(203, 213, 224, 0.3);
                border-radius: 12px;
                backdrop-filter: blur(10px);
            }

            #footerMain {
                font-size: 14px;
                font-weight: 600;
                color: #2d3748;
                margin: 0px;
            }

            #footerInfo {
                font-size: 12px;
                color: #718096;
                margin: 0px;
                font-weight: 400;
            }

            /* EFECTOS DE TRANSICIÓN */
            #cardButton {
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }

            #statsCard:hover {
                transform: translateY(-2px);
                transition: transform 0.2s ease;
            }

            #loadDataButton {
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
        """)

    def closeEvent(self, event):
        """Manejar evento de cierre"""
        event.accept()


# Para ejecutar standalone
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Configurar aplicación
    app.setApplicationName("Sistema de Análisis de Calidad del Agua")
    app.setApplicationVersion("1.0")

    window = MenuPrincipal()
    window.show()

    # Conectar señales de prueba
    def test_signal(signal_name):
        print(f"🚀 Módulo seleccionado: {signal_name}")

    window.abrir_preprocesamiento.connect(lambda: test_signal("Preprocesamiento de Datos"))
    window.abrir_carga_datos.connect(lambda: test_signal("Cargar Datos"))
    window.abrir_machine_learning.connect(lambda: test_signal("Machine Learning"))
    window.abrir_deep_learning.connect(lambda: test_signal("Deep Learning"))
    window.abrir_wqi.connect(lambda: test_signal("Índice WQI"))

    sys.exit(app.exec_())