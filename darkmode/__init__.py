"""
Módulo para manejo de temas oscuros y claros en la aplicación Qt.

Este módulo proporciona:
- ThemeManager: Singleton para gestionar el estado global del tema
- ThemedWidget: Clase base para widgets que soportan temas
"""

from .theme_manager import ThemeManager, ThemedWidget

__all__ = ['ThemeManager', 'ThemedWidget']