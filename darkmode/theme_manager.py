from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal, QObject


class ThemeManager(QObject):
    """Singleton para manejar los temas de la aplicación"""
    _instance = None
    _observers = []
    _is_dark = True  # Tema por defecto

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            super(ThemeManager, cls._instance).__init__()
        return cls._instance

    def add_observer(self, widget):
        """Agregar un widget que debe ser notificado cuando cambie el tema"""
        if widget not in self._observers:
            self._observers.append(widget)

    def remove_observer(self, widget):
        """Remover un widget de las notificaciones"""
        if widget in self._observers:
            self._observers.remove(widget)

    def set_dark_theme(self, is_dark=True):
        """Cambiar el tema y notificar a todos los observadores"""
        self._is_dark = is_dark
        self._notify_observers()

    def toggle_theme(self):
        """Alternar entre tema claro y oscuro"""
        self.set_dark_theme(not self._is_dark)

    def is_dark_theme(self):
        """Verificar si el tema actual es oscuro"""
        return self._is_dark

    def _notify_observers(self):
        """Notificar a todos los widgets observadores sobre el cambio de tema"""
        for widget in self._observers[:]:  # Copia la lista para evitar problemas de modificación
            if hasattr(widget, 'apply_theme'):
                widget.apply_theme()


class ThemedWidget:
    """Mixin class para widgets que soportan temas"""

    def __init__(self):
        # No llamar super().__init__() aquí para evitar conflictos
        self.theme_manager = ThemeManager()
        self.theme_manager.add_observer(self)

        # Aplicar tema inicial después de que el widget esté completamente inicializado
        if hasattr(self, 'show'):  # Verificar que es un widget válido
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, self.apply_theme)

    def closeEvent(self, event):
        """Limpiar observer al cerrar"""
        self.theme_manager.remove_observer(self)
        super().closeEvent(event)

    def apply_theme(self):
        """Aplicar el tema actual - debe ser implementado por las subclases"""
        if self.theme_manager.is_dark_theme():
            self.apply_dark_theme()
        else:
            self.apply_light_theme()

    def apply_light_theme(self):
        """Aplicar tema claro - implementación por defecto"""
        self.setStyleSheet(self.get_light_theme_style())

    def apply_dark_theme(self):
        """Aplicar tema oscuro - implementación por defecto"""
        self.setStyleSheet(self.get_dark_theme_style())

    def get_light_theme_style(self):
        """Obtener estilos del tema claro"""
        return """
            /* Estilos globales - Tema Claro */
            * {
                background-color: #f8fafb;
                color: #2c3e50;
                font-family: 'Segoe UI', 'Inter', 'Roboto', sans-serif;
            }

            QWidget {
                background-color: #f8fafb;
                color: #2c3e50;
            }

            QMainWindow {
                background-color: #f8fafb;
                color: #2c3e50;
            }

            QFrame {
                background-color: #f8fafb;
                color: #2c3e50;
                border: none;
            }

            QStackedWidget {
                background-color: #f8fafb;
                color: #2c3e50;
            }

            /* Header Frame */
            #headerFrame {
                background-color: #e3f2fd;
                border: 2px solid #2196f3;
                border-radius: 10px;
                margin-bottom: 10px;
                color: #2c3e50;
            }

            #mainTitle {
                font-size: 28px;
                font-weight: 600;
                color: #1976d2;
                letter-spacing: 0.5px;
                background-color: transparent;
            }

            /* Dark Mode Button */
            #darkModeButton {
                background-color: #2196f3;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
                font-size: 16px;
            }

            #darkModeButton:hover {
                background-color: #1976d2;
                color: white;
            }

            /* Status Frame */
            #statusFrame {
                background-color: white;
                border: 2px solid #e1f5fe;
                border-radius: 12px;
                box-shadow: 0 2px 10px rgba(33, 150, 243, 0.1);
                color: #2c3e50;
            }

            /* Control Frame */
            #controlFrame {
                background-color: white;
                border: 2px solid #e1f5fe;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(33, 150, 243, 0.1);
                color: #2c3e50;
            }

            /* Preview Frame */
            #previewFrame {
                background-color: white;
                border: 2px solid #e1f5fe;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(33, 150, 243, 0.1);
                color: #2c3e50;
            }

            /* Footer Frame */
            #footerFrame {
                background-color: white;
                border: 2px solid #e1f5fe;
                border-radius: 12px;
                box-shadow: 0 2px 10px rgba(33, 150, 243, 0.1);
                color: #2c3e50;
            }

            /* Labels generales */
            QLabel {
                background-color: transparent;
                color: #2c3e50;
            }

            /* Group Boxes */
            QGroupBox {
                font-weight: 600;
                font-size: 14px;
                color: #1976d2;
                border: 2px solid #bbdefb;
                border-radius: 10px;
                margin-top: 15px;
                padding-top: 20px;
                background-color: #f8fbff;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px;
                background-color: #f8fbff;
                color: #1976d2;
                font-size: 15px;
                font-weight: 600;
            }

            /* Buttons - CSV */
            #csvButton {
                background-color: #4caf50;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 15px;
                font-size: 14px;
                font-weight: 600;
            }

            #csvButton:hover {
                background-color: #45a049;
                color: white;
            }

            /* Buttons - Excel */
            #excelButton {
                background-color: #2196f3;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 15px;
                font-size: 14px;
                font-weight: 600;
            }

            #excelButton:hover {
                background-color: #1976d2;
                color: white;
            }

            /* Buttons - Sample */
            #sampleButton {
                background-color: #ff9800;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 15px;
                font-size: 14px;
                font-weight: 600;
            }

            #sampleButton:hover {
                background-color: #f57c00;
                color: white;
            }

            /* API Button (disabled) */
            #apiButton {
                background-color: #e0e0e0;
                color: #9e9e9e;
                border: none;
                border-radius: 10px;
                padding: 15px;
                font-size: 14px;
                font-weight: 600;
            }

            /* Botones generales */
            QPushButton {
                background-color: #2196f3;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: 600;
            }

            QPushButton:hover {
                background-color: #1976d2;
                color: white;
            }

            QPushButton:disabled {
                background-color: #e0e0e0;
                color: #9e9e9e;
            }

            
            #primaryButton:hover {
                background-color: #1976d2;
                color: white;
            }

            #primaryButton:disabled {
                background-color: #e0e0e0;
                color: #9e9e9e;
            }

            

            #secondaryButton:hover {
                background-color: #5a6268;
                color: white;
            }

            #secondaryButton:disabled {
                background-color: #e0e0e0;
                color: #9e9e9e;
            }


            #dangerButton:hover {
                background-color: #d32f2f;
                color: white;
            }

            /* Tab Widget */
            QTabWidget {
                background-color: white;
                color: #2c3e50;
            }

            QTabWidget::pane {
                border: 2px solid #bbdefb;
                background-color: white;
                border-radius: 8px;
                margin-top: 5px;
            }

            QTabBar::tab {
                background-color: #f8fbff;
                color: #1976d2;
                padding: 12px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: 600;
                font-size: 13px;
                min-width: 120px;
                border: 2px solid #e3f2fd;
            }

            QTabBar::tab:selected {
                background-color: #2196f3;
                color: white;
                border-color: #2196f3;
            }

            QTabBar::tab:hover:!selected {
                background-color: #e3f2fd;
                border-color: #2196f3;
                color: #1976d2;
            }

            /* Table Widget */
            QTableWidget {
                gridline-color: #e3f2fd;
                background-color: white;
                alternate-background-color: #f8fbff;
                border: 2px solid #bbdefb;
                border-radius: 8px;
                selection-background-color: #e3f2fd;
                font-size: 12px;
                color: #2c3e50;
            }

            QHeaderView::section {
                background-color: #2196f3;
                color: white;
                padding: 8px;
                border: none;
                font-weight: 600;
                font-size: 12px;
                border-right: 1px solid #1976d2;
            }

            QTableWidget::item {
                padding: 4px;
                border-bottom: 1px solid #e3f2fd;
                background-color: transparent;
                color: #2c3e50;
            }

            QTableWidget::item:selected {
                background-color: #bbdefb;
                color: #1976d2;
            }

            /* Text Edit and Scroll Area */
            QTextEdit, QScrollArea {
                border: 2px solid #e3f2fd;
                border-radius: 8px;
                background-color: white;
                padding: 10px;
                color: #2c3e50;
            }

            QScrollBar:vertical {
                background-color: #f8fbff;
                width: 12px;
                border-radius: 6px;
                border: 1px solid #e3f2fd;
            }

            QScrollBar::handle:vertical {
                background-color: #2196f3;
                border-radius: 6px;
                min-height: 20px;
            }

            QScrollBar::handle:vertical:hover {
                background-color: #1976d2;
            }

            /* ComboBox */
            QComboBox {
                border: 2px solid #e3f2fd;
                border-radius: 8px;
                padding: 8px;
                background-color: white;
                color: #2c3e50;
                font-size: 13px;
            }

            QComboBox:hover {
                border-color: #2196f3;
                color: #2c3e50;
            }

            QComboBox::drop-down {
                border: none;
                background-color: #2196f3;
                border-radius: 4px;
                width: 20px;
            }

            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid white;
            }

            QComboBox QAbstractItemView {
                border: 2px solid #e3f2fd;
                background-color: white;
                selection-background-color: #e3f2fd;
                color: #2c3e50;
            }

            /* CheckBox */
            QCheckBox {
                color: #2c3e50;
                font-size: 13px;
                font-weight: 500;
                background-color: transparent;
            }

            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #e3f2fd;
                border-radius: 4px;
                background-color: white;
            }

            QCheckBox::indicator:checked {
                background-color: #2196f3;
                border-color: #2196f3;
            }

            QCheckBox::indicator:hover {
                border-color: #2196f3;
            }

            /* RadioButton */
            QRadioButton {
                color: #2c3e50;
                font-size: 13px;
                font-weight: 500;
                background-color: transparent;
            }

            QRadioButton::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #e3f2fd;
                border-radius: 9px;
                background-color: white;
            }

            QRadioButton::indicator:checked {
                background-color: #2196f3;
                border-color: #2196f3;
            }

            QRadioButton::indicator:hover {
                border-color: #2196f3;
            }

            /* LineEdit */
            QLineEdit {
                border: 2px solid #e3f2fd;
                border-radius: 8px;
                padding: 8px;
                background-color: white;
                color: #2c3e50;
                font-size: 13px;
            }

            QLineEdit:focus {
                border-color: #2196f3;
                color: #2c3e50;
            }

            /* Progress Bar */
            QProgressBar {
                border: none;
                border-radius: 4px;
                background-color: #e3f2fd;
                text-align: center;
                font-weight: 600;
                color: #1976d2;
            }

            QProgressBar::chunk {
                background-color: #2196f3;
                border-radius: 4px;
            }

            /* Estilos específicos para elementos de la app ML */
            #title {
                font-size: 24px;
                font-weight: bold;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4a90e2, stop:0.5 #357abd, stop:1 #1e5f99);
                color: white;
                padding: 18px;
                border-radius: 10px;
                margin-bottom: 8px;
            }

            #subtitle {
                font-size: 13px;
                color: #7f8c8d;
                font-style: italic;
                margin-bottom: 15px;
                padding: 5px;
            }

            #systemInfo {
                font-size: 11px;
                color: #27ae60;
                font-weight: bold;
                background-color: #e8f5e8;
                padding: 8px 12px;
                border-radius: 6px;
                border-left: 3px solid #27ae60;
            }

            #statusLabel {
                background-color: #ffffff;
                color: #2c3e50;
                padding: 10px 16px;
                border-radius: 6px;
                border-left: 4px solid #4a90e2;
                font-weight: bold;
                font-size: 12px;
            }

            #infoFrame {
                background-color: #e8f4fd;
                border: 2px solid #b3d9ff;
                border-radius: 6px;
                padding: 12px;
                margin: 5px;
                color: #2c3e50;
            }

            #infoTitle {
                font-weight: bold;
                color: #2980b9;
                font-size: 13px;
                margin-bottom: 6px;
            }

            #infoText {
                color: #34495e;
                font-size: 11px;
                line-height: 1.4;
            }

            #buttonFrame {
                background-color: #ffffff;
                border: 2px solid #e1e8ed;
                border-radius: 8px;
                padding: 10px;
                margin: 3px;
                color: #2c3e50;
            }

            #buttonFrame:hover {
                border-color: #4a90e2;
                background-color: #f8fcff;
                box-shadow: 0 2px 8px rgba(74, 144, 226, 0.1);
            }

            #analysisBtn {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4a90e2, stop:1 #357abd);
                color: white;
                border: none;
                padding: 10px 12px;
                font-size: 12px;
                font-weight: bold;
                border-radius: 6px;
            }

            #analysisBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5ba0f2, stop:1 #4a90e2);
                color: white;
            }

            #analysisBtn:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }

            #descLabel {
                color: #5a6c7d;
                font-size: 10px;
                font-weight: normal;
                line-height: 1.3;
            }

            #statusIndicator {
                color: #27ae60;
                font-size: 16px;
                font-weight: bold;
            }

            #clearBtn, #memoryBtn {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 6px 12px;
                font-weight: bold;
                border-radius: 4px;
                font-size: 11px;
                min-width: 100px;
            }

            #clearBtn:hover, #memoryBtn:hover {
                background-color: #5a6268;
                color: white;
            }

            #memoryBtn {
                background-color: #28a745;
            }

            #memoryBtn:hover {
                background-color: #218838;
                color: white;
            }

            #backBtn {
                background-color: #6f7d8c;
                color: white;
                border: none;
                padding: 10px 16px;
                font-weight: bold;
                border-radius: 6px;
                font-size: 12px;
            }

            #backBtn:hover {
                background-color: #5a6c7d;
                color: white;
            }

            #resumenContent, #recomendacionesContent {
                background-color: white;
                padding: 12px;
                border-radius: 6px;
                font-size: 12px;
                line-height: 1.4;
                color: #2c3e50;
            }
        """

    def get_dark_theme_style(self):
        """Obtener estilos del tema oscuro"""
        return """
            /* Estilos globales - Tema Oscuro */
            * {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: 'Segoe UI', 'Inter', 'Roboto', sans-serif;
            }

            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }

            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
            }

            QFrame {
                background-color: #1e1e1e;
                color: #ffffff;
                border: none;
            }

            QStackedWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }

            /* Header Frame */
            #headerFrame {
                background-color: #2d2d2d;
                border: 2px solid #4a9eff;
                border-radius: 10px;
                margin-bottom: 10px;
                color: #ffffff;
            }

            #mainTitle {
                font-size: 28px;
                font-weight: 600;
                color: #4a9eff;
                letter-spacing: 0.5px;
                background-color: transparent;
            }

            /* Dark Mode Button */
            #darkModeButton {
                background-color: #4a9eff;
                color: #1e1e1e;
                border: none;
                border-radius: 8px;
                padding: 10px;
                font-size: 16px;
            }

            #darkModeButton:hover {
                background-color: #357abd;
                color: #1e1e1e;
            }

            /* Status Frame */
            #statusFrame {
                background-color: #2d2d2d;
                border: 2px solid #3a3a3a;
                border-radius: 12px;
                box-shadow: 0 2px 10px rgba(74, 158, 255, 0.2);
                color: #ffffff;
            }

            /* Control Frame */
            #controlFrame {
                background-color: #2d2d2d;
                border: 2px solid #3a3a3a;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(74, 158, 255, 0.2);
                color: #ffffff;
            }

            /* Preview Frame */
            #previewFrame {
                background-color: #2d2d2d;
                border: 2px solid #3a3a3a;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(74, 158, 255, 0.2);
                color: #ffffff;
            }

            /* Footer Frame */
            #footerFrame {
                background-color: #2d2d2d;
                border: 2px solid #3a3a3a;
                border-radius: 12px;
                box-shadow: 0 2px 10px rgba(74, 158, 255, 0.2);
                color: #ffffff;
            }

            /* Labels generales */
            QLabel {
                background-color: transparent;
                color: #ffffff;
            }

            /* Group Boxes */
            QGroupBox {
                font-weight: 600;
                font-size: 14px;
                color: #4a9eff;
                border: 2px solid #444444;
                border-radius: 10px;
                margin-top: 15px;
                padding-top: 20px;
                background-color: #252525;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px;
                background-color: #252525;
                color: #4a9eff;
                font-size: 15px;
                font-weight: 600;
            }

            /* Buttons - CSV */
            #csvButton {
                background-color: #66bb6a;
                color: #1e1e1e;
                border: none;
                border-radius: 10px;
                padding: 15px;
                font-size: 14px;
                font-weight: 600;
            }

            #csvButton:hover {
                background-color: #5cb85c;
                color: #1e1e1e;
            }

            /* Buttons - Excel */
            #excelButton {
                background-color: #4a9eff;
                color: #1e1e1e;
                border: none;
                border-radius: 10px;
                padding: 15px;
                font-size: 14px;
                font-weight: 600;
            }

            #excelButton:hover {
                background-color: #357abd;
                color: #1e1e1e;
            }

            /* Buttons - Sample */
            #sampleButton {
                background-color: #ffb74d;
                color: #1e1e1e;
                border: none;
                border-radius: 10px;
                padding: 15px;
                font-size: 14px;
                font-weight: 600;
            }

            #sampleButton:hover {
                background-color: #ffa726;
                color: #1e1e1e;
            }

            /* API Button (disabled) */
            #apiButton {
                background-color: #404040;
                color: #808080;
                border: none;
                border-radius: 10px;
                padding: 15px;
                font-size: 14px;
                font-weight: 600;
            }

            /* Botones generales */
            QPushButton {
                background-color: #4a9eff;
                color: #1e1e1e;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: 600;
            }

            QPushButton:hover {
                background-color: #357abd;
                color: #1e1e1e;
            }

            QPushButton:disabled {
                background-color: #404040;
                color: #808080;
            }

            

            #primaryButton:hover {
                background-color: #357abd;
                color: #1e1e1e;
            }

            #primaryButton:disabled {
                background-color: #404040;
                color: #808080;
            }

            

            #secondaryButton:hover {
                background-color: #757575;
                color: #1e1e1e;
            }

            #secondaryButton:disabled {
                background-color: #404040;
                color: #808080;
            }

        

            #dangerButton:hover {
                background-color: #e53e3e;
                color: #1e1e1e;
            }

            /* Tab Widget */
            QTabWidget {
                background-color: #2d2d2d;
                color: #ffffff;
            }

            QTabWidget::pane {
                border: 2px solid #444444;
                background-color: #2d2d2d;
                border-radius: 8px;
                margin-top: 5px;
            }

            QTabBar::tab {
                background-color: #252525;
                color: #4a9eff;
                padding: 12px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: 600;
                font-size: 13px;
                min-width: 120px;
                border: 2px solid #3a3a3a;
            }

            QTabBar::tab:selected {
                background-color: #4a9eff;
                color: #1e1e1e;
                border-color: #4a9eff;
            }

            QTabBar::tab:hover:!selected {
                background-color: #3a3a3a;
                border-color: #4a9eff;
                color: #4a9eff;
            }

            /* Table Widget */
            QTableWidget {
                gridline-color: #3a3a3a;
                background-color: #2d2d2d;
                alternate-background-color: #252525;
                border: 2px solid #444444;
                border-radius: 8px;
                selection-background-color: #3a3a3a;
                font-size: 12px;
                color: #ffffff;
            }

            QHeaderView::section {
                background-color: #4a9eff;
                color: #1e1e1e;
                padding: 8px;
                border: none;
                font-weight: 600;
                font-size: 12px;
                border-right: 1px solid #357abd;
            }

            QTableWidget::item {
                padding: 4px;
                border-bottom: 1px solid #3a3a3a;
                background-color: transparent;
                color: #ffffff;
            }

            QTableWidget::item:selected {
                background-color: #444444;
                color: #4a9eff;
            }

            /* Text Edit and Scroll Area */
            QTextEdit, QScrollArea {
                border: 2px solid #3a3a3a;
                border-radius: 8px;
                background-color: #2d2d2d;
                padding: 10px;
                color: #ffffff;
            }

            QScrollBar:vertical {
                background-color: #252525;
                width: 12px;
                border-radius: 6px;
                border: 1px solid #3a3a3a;
            }

            QScrollBar::handle:vertical {
                background-color: #4a9eff;
                border-radius: 6px;
                min-height: 20px;
            }

            QScrollBar::handle:vertical:hover {
                background-color: #357abd;
            }

            /* ComboBox */
            QComboBox {
                border: 2px solid #3a3a3a;
                border-radius: 8px;
                padding: 8px;
                background-color: #2d2d2d;
                color: #ffffff;
                font-size: 13px;
            }

            QComboBox:hover {
                border-color: #4a9eff;
                color: #ffffff;
            }

            QComboBox::drop-down {
                border: none;
                background-color: #4a9eff;
                border-radius: 4px;
                width: 20px;
            }

            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #1e1e1e;
            }

            QComboBox QAbstractItemView {
                border: 2px solid #3a3a3a;
                background-color: #2d2d2d;
                selection-background-color: #3a3a3a;
                color: #ffffff;
            }

            /* CheckBox */
            QCheckBox {
                color: #ffffff;
                font-size: 13px;
                font-weight: 500;
                background-color: transparent;
            }

            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #3a3a3a;
                border-radius: 4px;
                background-color: #2d2d2d;
            }

            QCheckBox::indicator:checked {
                background-color: #4a9eff;
                border-color: #4a9eff;
            }

            QCheckBox::indicator:hover {
                border-color: #4a9eff;
            }

            /* RadioButton */
            QRadioButton {
                color: #ffffff;
                font-size: 13px;
                font-weight: 500;
                background-color: transparent;
            }

            QRadioButton::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #3a3a3a;
                border-radius: 9px;
                background-color: #2d2d2d;
            }

            QRadioButton::indicator:checked {
                background-color: #4a9eff;
                border-color: #4a9eff;
            }

            QRadioButton::indicator:hover {
                border-color: #4a9eff;
            }

            /* LineEdit */
            QLineEdit {
                border: 2px solid #3a3a3a;
                border-radius: 8px;
                padding: 8px;
                background-color: #2d2d2d;
                color: #ffffff;
                font-size: 13px;
            }

            QLineEdit:focus {
                border-color: #4a9eff;
                color: #ffffff;
            }

            /* Progress Bar */
            QProgressBar {
                border: none;
                border-radius: 4px;
                background-color: #3a3a3a;
                text-align: center;
                font-weight: 600;
                color: #4a9eff;
            }

            QProgressBar::chunk {
                background-color: #4a9eff;
                border-radius: 4px;
            }

            /* Estilos específicos para elementos de la app ML */
            #title {
                font-size: 24px;
                font-weight: bold;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4a90e2, stop:0.5 #357abd, stop:1 #1e5f99);
                color: white;
                padding: 18px;
                border-radius: 10px;
                margin-bottom: 8px;
            }

            #subtitle {
                font-size: 13px;
                color: #b0b0b0;
                font-style: italic;
                margin-bottom: 15px;
                padding: 5px;
            }

            #systemInfo {
                font-size: 11px;
                color: #81c784;
                font-weight: bold;
                background-color: #2e4a2e;
                padding: 8px 12px;
                border-radius: 6px;
                border-left: 3px solid #81c784;
            }

            #statusLabel {
                background-color: #2d2d2d;
                color: #ffffff;
                padding: 10px 16px;
                border-radius: 6px;
                border-left: 4px solid #4a90e2;
                font-weight: bold;
                font-size: 12px;
            }

            #infoFrame {
                background-color: #3a3a3a;
                border: 2px solid #4a4a4a;
                border-radius: 6px;
                padding: 12px;
                margin: 5px;
                color: #ffffff;
            }

            #infoTitle {
                font-weight: bold;
                color: #4a9eff;
                font-size: 13px;
                margin-bottom: 6px;
            }

            #infoText {
                color: #e0e0e0;
                font-size: 11px;
                line-height: 1.4;
            }

            #buttonFrame {
                background-color: #2d2d2d;
                border: 2px solid #3a3a3a;
                border-radius: 8px;
                padding: 10px;
                margin: 3px;
                color: #ffffff;
            }

            #buttonFrame:hover {
                border-color: #4a90e2;
                background-color: #333333;
                box-shadow: 0 2px 8px rgba(74, 144, 226, 0.3);
            }

            #analysisBtn {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4a90e2, stop:1 #357abd);
                color: white;
                border: none;
                padding: 10px 12px;
                font-size: 12px;
                font-weight: bold;
                border-radius: 6px;
            }

            #analysisBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5ba0f2, stop:1 #4a90e2);
                color: white;
            }

            #analysisBtn:disabled {
                background-color: #404040;
                color: #808080;
            }

            #descLabel {
                color: #b0b0b0;
                font-size: 10px;
                font-weight: normal;
                line-height: 1.3;
            }

            #statusIndicator {
                color: #81c784;
                font-size: 16px;
                font-weight: bold;
            }

            #clearBtn, #memoryBtn {
                background-color: #8a8a8a;
                color: #1e1e1e;
                border: none;
                padding: 6px 12px;
                font-weight: bold;
                border-radius: 4px;
                font-size: 11px;
                min-width: 100px;
            }

            #clearBtn:hover, #memoryBtn:hover {
                background-color: #757575;
                color: #1e1e1e;
            }

            #memoryBtn {
                background-color: #66bb6a;
            }

            #memoryBtn:hover {
                background-color: #5cb85c;
                color: #1e1e1e;
            }

            #backBtn {
                background-color: #8a8a8a;
                color: #1e1e1e;
                border: none;
                padding: 10px 16px;
                font-weight: bold;
                border-radius: 6px;
                font-size: 12px;
            }

            #backBtn:hover {
                background-color: #757575;
                color: #1e1e1e;
            }

            #resumenContent, #recomendacionesContent {
                background-color: #2d2d2d;
                padding: 12px;
                border-radius: 6px;
                font-size: 12px;
                line-height: 1.4;
                color: #ffffff;
            }

            /* Utilty Frame */
            #utilityFrame {
                background-color: #2d2d2d;
                border-radius: 6px;
                padding: 8px;
                margin: 5px 0;
                color: #ffffff;
            }

            /* Progress Bar para ML */
            #progressBar {
                border: 2px solid #3a3a3a;
                border-radius: 6px;
                text-align: center;
                font-weight: bold;
                font-size: 11px;
                height: 22px;
                color: #4a9eff;
                background-color: #2d2d2d;
            }

            #progressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #66bb6a, stop:1 #81c784);
                border-radius: 4px;
            }

            /* Result Title */
            #resultTitle {
                font-size: 18px;
                font-weight: bold;
                color: #ffffff;
                padding: 12px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2d2d2d, stop:1 #3a3a3a);
                border-radius: 6px;
                margin-bottom: 8px;
            }
        """