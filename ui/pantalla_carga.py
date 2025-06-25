import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QProgressBar
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont


class PantallaCarga(QWidget):
    carga_completada = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.initUI()
        self.iniciar_carga()

    def initUI(self):
        # Configuración de la ventana
        self.setWindowTitle("Cargando...")
        self.setFixedSize(500, 300)  # Tamaño rectangular
        self.setWindowFlags(Qt.FramelessWindowHint)  # Sin borde
        self.setAttribute(Qt.WA_TranslucentBackground, False)

        # Centrar la ventana en la pantalla
        self.center()

        # Layout principal
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        # Logo
        self.logo_label = QLabel()
        pixmap = QPixmap("img/LogoCalidadAgua.png")

        # Verificar si la imagen existe
        if pixmap.isNull():
            # Si no existe la imagen, mostrar texto alternativo
            self.logo_label.setText("LOGO CALIDAD AGUA")
            self.logo_label.setAlignment(Qt.AlignCenter)
            self.logo_label.setStyleSheet("""
                QLabel {
                    color: #2c3e50;
                    font-size: 24px;
                    font-weight: bold;
                    background-color: #ecf0f1;
                    border: 2px solid #3498db;
                    padding: 20px;
                    border-radius: 10px;
                }
            """)
        else:
            # Redimensionar la imagen manteniendo la proporción
            scaled_pixmap = pixmap.scaled(200, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logo_label.setPixmap(scaled_pixmap)
            self.logo_label.setAlignment(Qt.AlignCenter)

        # Título de la aplicación
        self.title_label = QLabel("Sistema de Análisis de Calidad del Agua")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.title_label.setStyleSheet("color: #2c3e50; margin: 10px;")

        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
                color: #2c3e50;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
        """)

        # Texto de estado
        self.status_label = QLabel("Iniciando aplicación...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #7f8c8d; font-size: 12px;")

        # Agregar widgets al layout
        layout.addWidget(self.logo_label)
        layout.addWidget(self.title_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        self.setLayout(layout)

        # Estilo de fondo
        self.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                border: 2px solid #3498db;
                border-radius: 10px;
            }
        """)

    def center(self):
        """Centrar la ventana en la pantalla"""
        screen = QApplication.desktop().screenGeometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2
        )

    def iniciar_carga(self):
        """Simular proceso de carga"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_progreso)
        self.timer.start(50)  # Actualizar cada 50ms

        self.progreso = 0
        self.estados = [
            "Iniciando aplicación...",
            "Cargando módulos...",
            "Configurando interfaz...",
            "Preparando herramientas de análisis...",
            "Inicializando machine learning...",
            "Configurando deep learning...",
            "Finalizando configuración...",
            "¡Listo!"
        ]
        self.estado_actual = 0

    def actualizar_progreso(self):
        """Actualizar la barra de progreso y el texto de estado"""
        self.progreso += 2
        self.progress_bar.setValue(self.progreso)

        # Cambiar el texto de estado según el progreso
        if self.progreso >= len(self.estados) * 12.5:
            nuevo_estado = min(self.estado_actual + 1, len(self.estados) - 1)
            if nuevo_estado != self.estado_actual:
                self.estado_actual = nuevo_estado
                self.status_label.setText(self.estados[self.estado_actual])

        # Cuando se complete la carga
        if self.progreso >= 100:
            self.timer.stop()
            QTimer.singleShot(500, self.finalizar_carga)  # Esperar 500ms antes de cerrar

    def finalizar_carga(self):
        """Emitir señal de carga completada y cerrar"""
        self.carga_completada.emit()
        self.close()