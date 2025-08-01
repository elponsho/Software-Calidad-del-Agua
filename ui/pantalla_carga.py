import sys
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QProgressBar, QGraphicsDropShadowEffect, QFrame)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtGui import QPixmap, QFont, QPainter, QBrush, QColor, QPen


class AnimatedProgressBar(QProgressBar):
    """Barra de progreso animada personalizada"""

    def __init__(self):
        super().__init__()
        self.setRange(0, 100)
        self.setValue(0)
        self.setTextVisible(False)
        self.setFixedHeight(8)

        # Animación de valor
        self.animation = QPropertyAnimation(self, b"value")
        self.animation.setDuration(6000)  # 6 segundos
        self.animation.setEasingCurve(QEasingCurve.OutCubic)

    def start_animation(self):
        """Iniciar animación de 0 a 100"""
        self.animation.setStartValue(0)
        self.animation.setEndValue(100)
        self.animation.start()


class LogoContainer(QLabel):
    """Contenedor de logo estático centrado"""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(150, 150)

        # Cargar logo
        self.load_logo()

        # Efecto de sombra suave (sin animación)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(102, 126, 234, 80))
        shadow.setOffset(0, 5)
        self.setGraphicsEffect(shadow)

    def load_logo(self):
        """Cargar logo o mostrar emoji de respaldo"""
        logo_path = "img/LogoCalidadAgua.png"

        # Verificar si existe el archivo de logo
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            if not pixmap.isNull():
                # Redimensionar manteniendo proporción
                scaled_pixmap = pixmap.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.setPixmap(scaled_pixmap)

                # Estilo para imagen
                self.setStyleSheet("""
                    QLabel {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                            stop:0 rgba(255, 255, 255, 0.9),
                            stop:1 rgba(247, 250, 252, 0.9));
                        border: 3px solid rgba(102, 126, 234, 0.3);
                        border-radius: 75px;
                        padding: 15px;
                    }
                """)
                return

        # Si no se puede cargar la imagen, usar emoji
        self.setText("💧")
        self.setStyleSheet("""
            QLabel {
                font-size: 72px;
                color: white;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #667eea, stop:1 #764ba2);
                border: 3px solid rgba(255, 255, 255, 0.3);
                border-radius: 75px;
                padding: 15px;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }
        """)


class FloatingDots(QLabel):
    """Puntos flotantes animados"""

    def __init__(self):
        super().__init__()
        self.setFixedSize(100, 30)
        self.dots = "●●●"
        self.current_dots = ""
        self.dot_count = 0

        # Timer para animación de puntos
        self.dot_timer = QTimer()
        self.dot_timer.timeout.connect(self.update_dots)
        self.dot_timer.start(300)  # Cada 300ms

        self.setStyleSheet("""
            QLabel {
                font-size: 20px;
                color: #667eea;
                font-weight: bold;
            }
        """)
        self.setAlignment(Qt.AlignCenter)

    def update_dots(self):
        """Actualizar animación de puntos"""
        self.dot_count = (self.dot_count + 1) % 4
        self.current_dots = "●" * self.dot_count
        self.setText(self.current_dots)


class ModernCard(QFrame):
    """Tarjeta moderna con glassmorphism"""

    def __init__(self):
        super().__init__()
        self.setObjectName("modernCard")
        self.setFixedSize(450, 400)  # Altura aumentada

        # Efecto de sombra
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 10)
        self.setGraphicsEffect(shadow)


class PantallaCarga(QWidget):
    """Pantalla de carga moderna con animaciones"""

    carga_completada = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.iniciar_carga()

    def init_ui(self):
        """Configurar interfaz de usuario moderna"""
        # Configuración de ventana
        self.setWindowTitle("Cargando...")
        self.setFixedSize(600, 550)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        # Centrar ventana
        self.center()

        # Layout principal
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(75, 75, 75, 75)
        main_layout.setAlignment(Qt.AlignCenter)

        # Tarjeta contenedora
        self.card = ModernCard()
        self.card.setFixedSize(450, 400)  # Aumentar altura para acomodar logo
        card_layout = QVBoxLayout()
        card_layout.setAlignment(Qt.AlignCenter)
        card_layout.setSpacing(20)
        card_layout.setContentsMargins(40, 30, 40, 30)

        # Logo principal (centrado y prominente)
        self.logo = LogoContainer()
        card_layout.addWidget(self.logo, 0, Qt.AlignCenter)

        # Título principal
        title = QLabel("Sistema de Análisis")
        title.setObjectName("mainTitle")
        title.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(title)

        # Subtítulo
        subtitle = QLabel("Calidad del Agua")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(subtitle)

        # Espaciador
        card_layout.addSpacing(15)

        # Barra de progreso moderna
        self.progress_bar = AnimatedProgressBar()
        card_layout.addWidget(self.progress_bar)

        # Espaciador pequeño
        card_layout.addSpacing(8)

        # Container para estado y puntos
        status_container = QHBoxLayout()
        status_container.setAlignment(Qt.AlignCenter)
        status_container.setSpacing(10)

        # Texto de estado
        self.status_label = QLabel("Iniciando")
        self.status_label.setObjectName("statusLabel")
        status_container.addWidget(self.status_label)

        # Puntos animados
        self.dots = FloatingDots()
        status_container.addWidget(self.dots)

        card_layout.addLayout(status_container)

        # Información adicional
        info_label = QLabel("Preparando herramientas de análisis avanzado")
        info_label.setObjectName("infoLabel")
        info_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(info_label)

        self.card.setLayout(card_layout)
        main_layout.addWidget(self.card)

        self.setLayout(main_layout)
        self.apply_modern_styles()

    def center(self):
        """Centrar ventana en la pantalla"""
        screen = QApplication.desktop().screenGeometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2
        )

    def iniciar_carga(self):
        """Iniciar proceso de carga de 6 segundos"""
        # Estados de carga (más detallados para 6 segundos)
        self.estados = [
            "Iniciando sistema",
            "Cargando módulos",
            "Configurando ML",
            "Preparando análisis",
            "Inicializando Deep Learning",
            "Cargando interfaz",
            "Optimizando rendimiento",
            "Configurando herramientas",
            "Finalizando"
        ]
        self.estado_actual = 0

        # Timer para cambiar estados (cada ~670ms para 9 estados en 6 segundos)
        self.estado_timer = QTimer()
        self.estado_timer.timeout.connect(self.cambiar_estado)
        self.estado_timer.start(670)

        # Iniciar solo animación de la barra de progreso (NO del logo)
        self.progress_bar.start_animation()

        # Timer para finalizar (6 segundos)
        self.finish_timer = QTimer()
        self.finish_timer.timeout.connect(self.finalizar_carga)
        self.finish_timer.setSingleShot(True)
        self.finish_timer.start(6000)  # 6 segundos exactos

        # Animación de entrada de la tarjeta
        self.animate_entrance()

    def animate_entrance(self):
        """Animar entrada de la tarjeta"""
        self.card_animation = QPropertyAnimation(self.card, b"geometry")
        self.card_animation.setDuration(600)
        self.card_animation.setEasingCurve(QEasingCurve.OutBack)

        # Posición inicial (fuera de pantalla arriba)
        start_rect = QRect(75, -400, 450, 400)
        # Posición final (centrada)
        end_rect = QRect(75, 75, 450, 400)

        self.card_animation.setStartValue(start_rect)
        self.card_animation.setEndValue(end_rect)
        self.card_animation.start()

    def cambiar_estado(self):
        """Cambiar texto de estado"""
        if self.estado_actual < len(self.estados) - 1:
            self.estado_actual += 1
            self.status_label.setText(self.estados[self.estado_actual])

    def finalizar_carga(self):
        """Finalizar carga con animación de salida"""
        self.estado_timer.stop()
        self.dots.dot_timer.stop()
        # NO detenemos animación del logo porque ya no la tiene

        # Cambiar a estado final
        self.status_label.setText("¡Listo!")
        self.dots.setText("✓")
        self.dots.setStyleSheet("""
            QLabel {
                font-size: 20px;
                color: #48bb78;
                font-weight: bold;
            }
        """)

        # Animación de salida
        self.exit_animation = QPropertyAnimation(self.card, b"geometry")
        self.exit_animation.setDuration(400)
        self.exit_animation.setEasingCurve(QEasingCurve.InBack)

        # Posición actual
        current_rect = self.card.geometry()
        # Posición final (fuera de pantalla abajo)
        end_rect = QRect(75, 550, 450, 400)

        self.exit_animation.setStartValue(current_rect)
        self.exit_animation.setEndValue(end_rect)
        self.exit_animation.finished.connect(self.emit_completed)
        self.exit_animation.start()

    def emit_completed(self):
        """Emitir señal de completado y cerrar"""
        self.carga_completada.emit()
        self.close()

    def apply_modern_styles(self):
        """Aplicar estilos modernos"""
        self.setStyleSheet("""
            /* FONDO PRINCIPAL */
            QWidget {
                background: transparent;
                font-family: 'Segoe UI', 'SF Pro Display', system-ui, sans-serif;
            }

            /* TARJETA PRINCIPAL */
            #modernCard {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 255, 255, 0.95),
                    stop:1 rgba(247, 250, 252, 0.95));
                border: 1px solid rgba(226, 232, 240, 0.8);
                border-radius: 25px;
                backdrop-filter: blur(20px);
            }

            /* TÍTULO PRINCIPAL */
            #mainTitle {
                font-size: 28px;
                font-weight: 700;
                color: #1a365d;
                margin: 0px;
                padding: 0px;
                text-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }

            /* SUBTÍTULO */
            #subtitle {
                font-size: 20px;
                font-weight: 400;
                color: #4a5568;
                margin: 0px;
                padding: 0px;
            }

            /* TEXTO DE ESTADO */
            #statusLabel {
                font-size: 16px;
                font-weight: 500;
                color: #667eea;
                margin: 0px;
                padding: 0px;
            }

            /* INFORMACIÓN ADICIONAL */
            #infoLabel {
                font-size: 12px;
                color: #718096;
                margin: 0px;
                padding: 0px;
                font-weight: 400;
            }

            /* BARRA DE PROGRESO */
            QProgressBar {
                background: rgba(226, 232, 240, 0.5);
                border: none;
                border-radius: 4px;
                text-align: center;
            }

            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:0.5 #764ba2, stop:1 #f093fb);
                border-radius: 4px;
                transition: width 0.3s ease;
            }
        """)

    def paintEvent(self, event):
        """Pintar fondo con gradiente"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Gradiente de fondo
        gradient = QBrush(QColor(0, 0, 0, 100))
        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())


# Para ejecutar standalone
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Configurar aplicación
    app.setApplicationName("Sistema de Análisis de Calidad del Agua")

    splash = PantallaCarga()
    splash.show()


    def on_loading_finished():
        print("🚀 Carga completada en 6 segundos - Listo para mostrar aplicación principal")
        app.quit()


    splash.carga_completada.connect(on_loading_finished)

    sys.exit(app.exec_())