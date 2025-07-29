"""
resultados_visuales.py - Widget para mostrar resultados de análisis ML
Visualización unificada para todos los tipos de análisis
"""

import sys
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QTabWidget,
                             QScrollArea, QFrame)
from PyQt5.QtCore import Qt

# Importaciones de matplotlib con manejo de errores
try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import numpy as np
    import pandas as pd
    plt.style.use('default')
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: matplotlib not available: {e}")
    MATPLOTLIB_AVAILABLE = False

# from darkmode import ThemedWidget  # COMENTADA TEMPORALMENTE
class ThemedWidget:
    def __init__(self):
        pass
    def apply_theme(self):
        pass

class ResultadosVisuales(QWidget, ThemedWidget):
    """Widget para mostrar resultados de análisis ML con múltiples pestañas"""

    def __init__(self):
        QWidget.__init__(self)
        ThemedWidget.__init__(self)

        self.setup_ui()
        self.current_data = None
        self.current_analysis = None

    def setup_ui(self):
        """Configurar interfaz de usuario"""
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # Título de resultados
        self.titulo_resultado = QLabel("📊 Resultados del Análisis")
        self.titulo_resultado.setObjectName("resultTitle")
        self.titulo_resultado.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.titulo_resultado)

        # Tabs para diferentes vistas
        self.tabs = QTabWidget()
        self.tabs.setObjectName("resultTabs")

        # Tab 1: Resumen
        self.resumen_tab = QWidget()
        self.setup_resumen_tab()
        self.tabs.addTab(self.resumen_tab, "📋 Resumen")

        # Tab 2: Gráficos
        self.graficos_tab = QWidget()
        self.setup_graficos_tab()
        self.tabs.addTab(self.graficos_tab, "📊 Gráficos")

        # Tab 3: Métricas
        self.metricas_tab = QWidget()
        self.setup_metricas_tab()
        self.tabs.addTab(self.metricas_tab, "📈 Métricas")

        # Tab 4: Datos
        self.datos_tab = QWidget()
        self.setup_datos_tab()
        self.tabs.addTab(self.datos_tab, "📋 Datos")

        layout.addWidget(self.tabs)
        self.setLayout(layout)

        self.apply_styles()

    def setup_resumen_tab(self):
        """Configurar tab de resumen"""
        layout = QVBoxLayout()

        self.resumen_scroll = QScrollArea()
        self.resumen_scroll.setWidgetResizable(True)

        self.resumen_content = QLabel("Ejecuta un análisis para ver el resumen de resultados")
        self.resumen_content.setObjectName("resumenContent")
        self.resumen_content.setAlignment(Qt.AlignTop)
        self.resumen_content.setWordWrap(True)

        self.resumen_scroll.setWidget(self.resumen_content)
        layout.addWidget(self.resumen_scroll)

        self.resumen_tab.setLayout(layout)

    def setup_graficos_tab(self):
        """Configurar tab de gráficos"""
        layout = QVBoxLayout()

        if MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(12, 8), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)
        else:
            no_matplotlib_label = QLabel("⚠️ Matplotlib no disponible\nInstala con: pip install matplotlib")
            no_matplotlib_label.setAlignment(Qt.AlignCenter)
            no_matplotlib_label.setObjectName("warningLabel")
            layout.addWidget(no_matplotlib_label)

        self.graficos_tab.setLayout(layout)

    def setup_metricas_tab(self):
        """Configurar tab de métricas"""
        layout = QVBoxLayout()

        self.metricas_scroll = QScrollArea()
        self.metricas_scroll.setWidgetResizable(True)

        self.metricas_content = QLabel("Las métricas detalladas aparecerán aquí")
        self.metricas_content.setObjectName("metricasContent")
        self.metricas_content.setAlignment(Qt.AlignTop)
        self.metricas_content.setWordWrap(True)

        self.metricas_scroll.setWidget(self.metricas_content)
        layout.addWidget(self.metricas_scroll)

        self.metricas_tab.setLayout(layout)

    def setup_datos_tab(self):
        """Configurar tab de datos"""
        layout = QVBoxLayout()

        self.datos_scroll = QScrollArea()
        self.datos_scroll.setWidgetResizable(True)

        self.datos_content = QLabel("Los datos procesados aparecerán aquí")
        self.datos_content.setObjectName("datosContent")
        self.datos_content.setAlignment(Qt.AlignTop)
        self.datos_content.setWordWrap(True)

        self.datos_scroll.setWidget(self.datos_content)
        layout.addWidget(self.datos_scroll)

        self.datos_tab.setLayout(layout)

    def mostrar_resultados(self, resultados, tipo_analisis):
        """Mostrar resultados según el tipo de análisis"""
        try:
            self.current_data = resultados
            self.current_analysis = tipo_analisis

            print(f"📊 Mostrando resultados para: {tipo_analisis}")

            # Actualizar título
            analysis_names = {
                "regresion_multiple": "📈 Regresión Múltiple",
                "svm": "🎯 Support Vector Machine",
                "random_forest": "🌳 Random Forest",
                "regresion_lineal": "📏 Regresión Lineal",
                "clustering": "🔍 Clustering",
                "pca": "🎯 PCA",
                "calidad_agua": "🧪 Análisis de Calidad",
                "optimizar_sistema": "⚙️ Optimización",
                "comparar_metodos": "📊 Comparación"
            }

            self.titulo_resultado.setText(f"📊 Resultados: {analysis_names.get(tipo_analisis, tipo_analisis)}")

            # Mostrar en cada tab
            self.mostrar_resumen(resultados, tipo_analisis)
            self.mostrar_metricas(resultados, tipo_analisis)
            self.mostrar_datos(resultados, tipo_analisis)

            if MATPLOTLIB_AVAILABLE:
                self.crear_graficos(resultados, tipo_analisis)

        except Exception as e:
            error_msg = f"❌ Error al mostrar resultados: {str(e)}"
            print(error_msg)
            self.mostrar_error(error_msg)

    def mostrar_resumen(self, resultados, tipo_analisis):
        """Mostrar resumen según el tipo de análisis"""
        try:
            if tipo_analisis == "regresion_multiple":
                self.mostrar_resumen_regresion_multiple(resultados)
            elif tipo_analisis == "svm":
                self.mostrar_resumen_svm(resultados)
            elif tipo_analisis == "random_forest":
                self.mostrar_resumen_random_forest(resultados)
            elif tipo_analisis == "regresion_lineal":
                self.mostrar_resumen_regresion_lineal(resultados)
            elif tipo_analisis == "clustering":
                self.mostrar_resumen_clustering(resultados)
            elif tipo_analisis == "pca":
                self.mostrar_resumen_pca(resultados)
            else:
                self.resumen_content.setText(f"Resumen para {tipo_analisis} no implementado aún")

        except Exception as e:
            self.mostrar_error(f"Error en resumen: {str(e)}")

    def mostrar_resumen_regresion_multiple(self, resultados):
        """Mostrar resumen de regresión múltiple"""
        r2 = resultados.get('r2_score', 0)
        mse = resultados.get('mse', 0)
        mae = resultados.get('mae', 0)

        html = f"""
        <div style="font-family: Arial; padding: 20px;">
            <h2 style="color: #2c3e50; text-align: center;">📈 Regresión Múltiple</h2>
            
            <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h3 style="color: #2e7d32;">📊 Métricas Principales</h3>
                <p><strong>R² Score:</strong> {r2:.3f} ({r2*100:.1f}% de varianza explicada)</p>
                <p><strong>Error Cuadrático Medio (MSE):</strong> {mse:.3f}</p>
                <p><strong>Error Absoluto Medio (MAE):</strong> {mae:.3f}</p>
            </div>
            
            <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h3 style="color: #1565c0;">🎯 Ventajas del Ensemble</h3>
                <p><strong>Robustez:</strong> Reduce overfitting con múltiples árboles</p>
                <p><strong>Versatilidad:</strong> Maneja clasificación y regresión simultáneamente</p>
                <p><strong>Interpretabilidad:</strong> Proporciona importancia de características</p>
            </div>
        </div>
        """

        self.resumen_content.setText(html)

    def mostrar_resumen_regresion_lineal(self, resultados):
        """Mostrar resumen de regresión lineal"""
        r2_multiple = resultados.get('r2_multiple', 0)
        r2_ridge = resultados.get('r2_ridge', 0)
        r2_lasso = resultados.get('r2_lasso', 0)

        html = f"""
        <div style="font-family: Arial; padding: 20px;">
            <h2 style="color: #2c3e50; text-align: center;">📏 Regresión Lineal Completa</h2>
            
            <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h3 style="color: #2e7d32;">📊 Comparación de Métodos</h3>
                <p><strong>Regresión Múltiple:</strong> R² = {r2_multiple:.3f}</p>
                <p><strong>Ridge (L2):</strong> R² = {r2_ridge:.3f}</p>
                <p><strong>Lasso (L1):</strong> R² = {r2_lasso:.3f}</p>
            </div>
            
            <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h3 style="color: #1565c0;">🎯 Recomendación</h3>
                <p><strong>Mejor método:</strong> {'Ridge' if r2_ridge >= max(r2_multiple, r2_lasso) else 'Lasso' if r2_lasso >= r2_multiple else 'Múltiple'}</p>
                <p><strong>Regularización:</strong> {'Recomendada' if max(r2_ridge, r2_lasso) > r2_multiple else 'No necesaria'}</p>
            </div>
        </div>
        """

        self.resumen_content.setText(html)

    def mostrar_resumen_clustering(self, resultados):
        """Mostrar resumen de clustering"""
        best_k_kmeans = resultados.get('best_k_kmeans', 0)
        best_k_hier = resultados.get('best_k_hierarchical', 0)

        html = f"""
        <div style="font-family: Arial; padding: 20px;">
            <h2 style="color: #2c3e50; text-align: center;">🔍 Clustering Avanzado</h2>
            
            <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h3 style="color: #2e7d32;">📊 Configuración Óptima</h3>
                <p><strong>K-Means óptimo:</strong> {best_k_kmeans} clusters</p>
                <p><strong>Jerárquico óptimo:</strong> {best_k_hier} clusters</p>
                <p><strong>Evaluación:</strong> Silhouette Score</p>
            </div>
            
            <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h3 style="color: #1565c0;">🎯 Interpretación</h3>
                <p><strong>Grupos identificados:</strong> Las muestras se agrupan naturalmente en {best_k_kmeans} tipos</p>
                <p><strong>Consistencia:</strong> {'Alta' if abs(best_k_kmeans - best_k_hier) <= 1 else 'Media'} concordancia entre métodos</p>
            </div>
        </div>
        """

        self.resumen_content.setText(html)

    def mostrar_resumen_pca(self, resultados):
        """Mostrar resumen de PCA"""
        n_comp_85 = resultados.get('n_componentes_85', 0)
        varianza_explicada = resultados.get('varianza_explicada', [])

        html = f"""
        <div style="font-family: Arial; padding: 20px;">
            <h2 style="color: #2c3e50; text-align: center;">🎯 Análisis de Componentes Principales</h2>
            
            <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h3 style="color: #2e7d32;">📊 Reducción de Dimensionalidad</h3>
                <p><strong>Componentes para 85% varianza:</strong> {n_comp_85} de 4</p>
                <p><strong>Eficiencia:</strong> {(4-n_comp_85)/4*100:.1f}% reducción manteniendo información</p>
            </div>
            
            <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h3 style="color: #1565c0;">🎯 Componentes Principales</h3>
        """

        if varianza_explicada:
            for i, var in enumerate(varianza_explicada[:3]):
                html += f"<p><strong>PC{i+1}:</strong> {var*100:.1f}% de varianza explicada</p>"

        html += """
            </div>
        </div>
        """

        self.resumen_content.setText(html)

    def mostrar_metricas(self, resultados, tipo_analisis):
        """Mostrar métricas detalladas"""
        try:
            html = f"""
            <div style="font-family: Arial; padding: 20px;">
                <h2 style="color: #2c3e50;">📈 Métricas Detalladas</h2>
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                    <pre style="font-family: 'Courier New', monospace; font-size: 12px;">
            """

            # Mostrar métricas según el tipo
            for key, value in resultados.items():
                if key not in ['datos', 'scaled_data', 'linkage_matrix', 'datos_transformados_2d', 'datos_transformados_3d']:
                    if isinstance(value, (int, float)):
                        html += f"{key}: {value:.4f}\n"
                    elif isinstance(value, str):
                        html += f"{key}: {value}\n"
                    elif isinstance(value, list) and len(value) < 10:
                        html += f"{key}: {value}\n"

            html += """
                    </pre>
                </div>
            </div>
            """

            self.metricas_content.setText(html)

        except Exception as e:
            self.metricas_content.setText(f"Error al mostrar métricas: {str(e)}")

    def mostrar_datos(self, resultados, tipo_analisis):
        """Mostrar datos procesados"""
        try:
            html = f"""
            <div style="font-family: Arial; padding: 20px;">
                <h2 style="color: #2c3e50;">📋 Datos Procesados</h2>
            """

            # Mostrar ejemplos si existen
            if 'ejemplos' in resultados:
                ejemplos = resultados['ejemplos']
                html += """
                <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; margin: 15px 0;">
                    <h3 style="color: #2e7d32;">🔍 Ejemplos de Resultados</h3>
                    <table style="width: 100%; border-collapse: collapse;">
                """

                if ejemplos and len(ejemplos) > 0:
                    # Headers
                    html += "<tr style='background: #f0f0f0;'>"
                    for key in ejemplos[0].keys():
                        html += f"<th style='border: 1px solid #ddd; padding: 8px;'>{key}</th>"
                    html += "</tr>"

                    # Datos (máximo 5 filas)
                    for ejemplo in ejemplos[:5]:
                        html += "<tr>"
                        for value in ejemplo.values():
                            if isinstance(value, float):
                                formatted_value = f"{value:.3f}"
                            else:
                                formatted_value = str(value)
                            html += f"<td style='border: 1px solid #ddd; padding: 8px;'>{formatted_value}</td>"
                        html += "</tr>"

                html += "</table></div>"

            # Mostrar información de datasets
            if 'datos' in resultados:
                datos = resultados['datos']
                if isinstance(datos, list) and len(datos) > 0:
                    html += f"""
                    <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 15px 0;">
                        <h3 style="color: #1565c0;">📊 Información del Dataset</h3>
                        <p><strong>Total de muestras:</strong> {len(datos)}</p>
                        <p><strong>Primeras 3 muestras:</strong></p>
                        <pre style="font-size: 11px; background: #f8f9fa; padding: 10px; border-radius: 4px;">
                    """

                    for i, muestra in enumerate(datos[:3]):
                        html += f"Muestra {i+1}: {muestra}\n"

                    html += "</pre></div>"

            html += "</div>"
            self.datos_content.setText(html)

        except Exception as e:
            self.datos_content.setText(f"Error al mostrar datos: {str(e)}")

    def crear_graficos(self, resultados, tipo_analisis):
        """Crear gráficos según el tipo de análisis"""
        try:
            self.figure.clear()

            if tipo_analisis == "regresion_multiple":
                self.crear_graficos_regresion_multiple(resultados)
            elif tipo_analisis == "svm":
                self.crear_graficos_svm(resultados)
            elif tipo_analisis == "random_forest":
                self.crear_graficos_random_forest(resultados)
            elif tipo_analisis == "regresion_lineal":
                self.crear_graficos_regresion_lineal(resultados)
            elif tipo_analisis == "clustering":
                self.crear_graficos_clustering(resultados)
            elif tipo_analisis == "pca":
                self.crear_graficos_pca(resultados)
            else:
                self.crear_grafico_placeholder(tipo_analisis)

            self.canvas.draw()

        except Exception as e:
            print(f"Error creando gráficos: {e}")
            self.crear_grafico_error(str(e))

    def crear_graficos_regresion_multiple(self, resultados):
        """Crear gráficos para regresión múltiple"""
        ax1 = self.figure.add_subplot(2, 2, 1)
        ax2 = self.figure.add_subplot(2, 2, 2)
        ax3 = self.figure.add_subplot(2, 2, 3)
        ax4 = self.figure.add_subplot(2, 2, 4)

        # Gráfico 1: Coeficientes
        if 'coeficientes' in resultados:
            coeficientes = resultados['coeficientes']
            params = [c['Parámetro'] for c in coeficientes]
            coefs = [c['Coeficiente'] for c in coeficientes]

            colors = ['#4CAF50' if c > 0 else '#F44336' for c in coefs]
            ax1.bar(params, coefs, color=colors, alpha=0.7, edgecolor='black')
            ax1.set_title('📊 Coeficientes de Regresión')
            ax1.tick_params(axis='x', rotation=45)
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Gráfico 2: Métricas
        metricas = ['R²', 'MSE', 'MAE']
        valores = [
            resultados.get('r2_score', 0),
            resultados.get('mse', 0) / 100,  # Escalar para visualización
            resultados.get('mae', 0)
        ]

        ax2.bar(metricas, valores, color=['#2196F3', '#FF9800', '#9C27B0'], alpha=0.7)
        ax2.set_title('📈 Métricas de Evaluación')

        # Gráfico 3: Ejemplos - Predicho vs Real
        if 'ejemplos' in resultados:
            ejemplos = resultados['ejemplos']
            predichos = [e.get('Predicho', 0) for e in ejemplos]
            reales = [e.get('Real', 0) for e in ejemplos]

            ax3.scatter(reales, predichos, alpha=0.7, color='#4CAF50')

            # Línea diagonal
            if reales and predichos:
                min_val = min(min(reales), min(predichos))
                max_val = max(max(reales), max(predichos))
                ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)

            ax3.set_title('🎯 Predicción vs Real')
            ax3.set_xlabel('Valores Reales')
            ax3.set_ylabel('Valores Predichos')

        # Gráfico 4: Residuos
        if 'ejemplos' in resultados:
            ejemplos = resultados['ejemplos']
            residuos = [e.get('Error', 0) for e in ejemplos]
            indices = list(range(len(residuos)))

            ax4.scatter(indices, residuos, alpha=0.7, color='#FF5722')
            ax4.axhline(y=0, color='red', linestyle='--', alpha=0.8)
            ax4.set_title('📈 Residuos')
            ax4.set_xlabel('Índice de Muestra')
            ax4.set_ylabel('Error')

        self.figure.suptitle('📈 Análisis de Regresión Múltiple', fontsize=14, fontweight='bold')
        self.figure.tight_layout()

    def crear_graficos_svm(self, resultados):
        """Crear gráficos para SVM"""
        # Implementación básica
        ax = self.figure.add_subplot(1, 1, 1)

        precision = resultados.get('precision', 0)
        ax.pie([precision, 100-precision], labels=['Correcto', 'Incorrecto'],
               colors=['#4CAF50', '#F44336'], autopct='%1.1f%%', startangle=90)
        ax.set_title(f'🎯 Precisión SVM: {precision:.1f}%', fontweight='bold')

        self.figure.tight_layout()

    def crear_graficos_random_forest(self, resultados):
        """Crear gráficos para Random Forest"""
        ax1 = self.figure.add_subplot(1, 2, 1)
        ax2 = self.figure.add_subplot(1, 2, 2)

        # Gráfico 1: Importancia de características
        if 'importancias' in resultados:
            importancias = resultados['importancias']
            params = [i['Parámetro'] for i in importancias]
            valores = [i['Importancia'] * 100 for i in importancias]

            ax1.barh(params, valores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            ax1.set_title('🎯 Importancia de Características')
            ax1.set_xlabel('Importancia (%)')

        # Gráfico 2: Comparación de rendimiento
        tipos = ['Clasificación', 'Regresión (R²×100)']
        rendimientos = [
            resultados.get('precision_clasificacion', 0),
            resultados.get('r2_regresion', 0) * 100
        ]

        ax2.bar(tipos, rendimientos, color=['#4CAF50', '#2196F3'], alpha=0.7)
        ax2.set_title('📊 Rendimiento Dual')
        ax2.set_ylabel('Score (%)')

        self.figure.suptitle('🌳 Análisis Random Forest', fontsize=14, fontweight='bold')
        self.figure.tight_layout()

    def crear_graficos_regresion_lineal(self, resultados):
        """Crear gráficos para regresión lineal"""
        ax1 = self.figure.add_subplot(1, 2, 1)
        ax2 = self.figure.add_subplot(1, 2, 2)

        # Gráfico 1: R² por variable simple
        if 'regresiones_simples' in resultados:
            regresiones = resultados['regresiones_simples']
            params = list(regresiones.keys())
            r2_valores = [regresiones[p]['r2'] for p in params]

            colors = ['#4CAF50' if r > 0.5 else '#FF9800' if r > 0.2 else '#F44336' for r in r2_valores]
            ax1.bar(params, r2_valores, color=colors, alpha=0.7)
            ax1.set_title('📊 R² por Variable')
            ax1.set_ylabel('R² Score')
            ax1.tick_params(axis='x', rotation=45)

        # Gráfico 2: Comparación de métodos
        metodos = ['Múltiple', 'Ridge', 'Lasso']
        r2_metodos = [
            resultados.get('r2_multiple', 0),
            resultados.get('r2_ridge', 0),
            resultados.get('r2_lasso', 0)
        ]

        ax2.bar(metodos, r2_metodos, color=['#2196F3', '#FF9800', '#9C27B0'], alpha=0.7)
        ax2.set_title('🔍 Comparación de Métodos')
        ax2.set_ylabel('R² Score')

        self.figure.suptitle('📏 Análisis Regresión Lineal', fontsize=14, fontweight='bold')
        self.figure.tight_layout()

    def crear_graficos_clustering(self, resultados):
        """Crear gráficos para clustering"""
        ax1 = self.figure.add_subplot(2, 2, 1)
        ax2 = self.figure.add_subplot(2, 2, 2)
        ax3 = self.figure.add_subplot(2, 2, 3)
        ax4 = self.figure.add_subplot(2, 2, 4)

        # Silhouette Scores
        if 'kmeans_results' in resultados:
            kmeans_results = resultados['kmeans_results']
            k_values = list(kmeans_results.keys())
            silhouette_scores = [kmeans_results[k]['silhouette_score'] for k in k_values]

            ax1.bar(k_values, silhouette_scores, color='#4ECDC4', alpha=0.7)
            ax1.set_title('📊 Silhouette Score K-Means')
            ax1.set_xlabel('Número de Clusters')

        # Inertia (método del codo)
        if 'kmeans_results' in resultados:
            inertias = [kmeans_results[k]['inertia'] for k in k_values]
            ax2.plot(k_values, inertias, 'bo-', color='#FF6B6B')
            ax2.set_title('📈 Método del Codo')
            ax2.set_xlabel('Número de Clusters')
            ax2.set_ylabel('Inertia')

        # Distribución de clusters
        if 'analisis_clusters' in resultados:
            analisis = resultados['analisis_clusters']
            cluster_names = list(analisis.keys())
            cluster_sizes = [analisis[c]['tamaño'] for c in cluster_names]

            ax3.pie(cluster_sizes, labels=cluster_names, autopct='%1.1f%%', startangle=90)
            ax3.set_title('🥧 Distribución de Clusters')

        # Calidad por cluster
        if 'analisis_clusters' in resultados:
            cluster_quality = [analisis[c]['calidad_promedio'] for c in cluster_names]
            ax4.bar(range(len(cluster_names)), cluster_quality,
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(cluster_names)])
            ax4.set_title('📊 Calidad por Cluster')
            ax4.set_xticks(range(len(cluster_names)))
            ax4.set_xticklabels([f'C{i+1}' for i in range(len(cluster_names))])

        self.figure.suptitle('🔍 Análisis de Clustering', fontsize=14, fontweight='bold')
        self.figure.tight_layout()

    def crear_graficos_pca(self, resultados):
        """Crear gráficos para PCA"""
        ax1 = self.figure.add_subplot(2, 2, 1)
        ax2 = self.figure.add_subplot(2, 2, 2)
        ax3 = self.figure.add_subplot(2, 2, 3)
        ax4 = self.figure.add_subplot(2, 2, 4)

        # Varianza explicada
        if 'varianza_explicada' in resultados:
            varianza = [v * 100 for v in resultados['varianza_explicada']]
            componentes = [f'PC{i+1}' for i in range(len(varianza))]

            ax1.bar(componentes, varianza, color='#4CAF50', alpha=0.7)
            ax1.set_title('📊 Varianza Explicada')
            ax1.set_ylabel('Varianza (%)')

        # Varianza acumulada
        if 'varianza_acumulada' in resultados:
            var_acum = [v * 100 for v in resultados['varianza_acumulada']]
            ax2.plot(range(1, len(var_acum)+1), var_acum, 'bo-', color='#2196F3')
            ax2.axhline(y=85, color='red', linestyle='--', alpha=0.7, label='85% umbral')
            ax2.set_title('📈 Varianza Acumulada')
            ax2.set_xlabel('Número de Componentes')
            ax2.set_ylabel('Varianza Acumulada (%)')
            ax2.legend()

        # Contribuciones PC1
        if 'componentes_principales' in resultados and len(resultados['componentes_principales']) > 0:
            pc1 = resultados['componentes_principales'][0]
            contrib = pc1['contribuciones']
            params = list(contrib.keys())
            valores = list(contrib.values())

            colors = ['#4CAF50' if v > 0 else '#F44336' for v in valores]
            ax3.bar(params, valores, color=colors, alpha=0.7)
            ax3.set_title('🎯 Contribuciones PC1')
            ax3.tick_params(axis='x', rotation=45)

        # Visualización 2D si disponible
        if 'datos_transformados_2d' in resultados:
            datos_2d = np.array(resultados['datos_transformados_2d'])
            ax4.scatter(datos_2d[:, 0], datos_2d[:, 1], alpha=0.7, color='#9C27B0')
            ax4.set_title('🗺️ Datos en Espacio PCA')
            ax4.set_xlabel('PC1')
            ax4.set_ylabel('PC2')

        self.figure.suptitle('🎯 Análisis PCA', fontsize=14, fontweight='bold')
        self.figure.tight_layout()

    def crear_grafico_placeholder(self, tipo_analisis):
        """Crear gráfico placeholder para análisis no implementados"""
        ax = self.figure.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, f'Gráficos para {tipo_analisis}\npronto disponibles',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(f'📊 {tipo_analisis}')

    def crear_grafico_error(self, error_msg):
        """Crear gráfico de error"""
        self.figure.clear()
        ax = self.figure.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, f'❌ Error al crear gráficos:\n{error_msg}',
                ha='center', va='center', transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        ax.set_title('Error en Visualización')
        self.canvas.draw()

    def mostrar_error(self, mensaje):
        """Mostrar mensaje de error en todas las pestañas"""
        error_html = f"""
        <div style="font-family: Arial; padding: 20px; text-align: center;">
            <h2 style="color: #dc3545;">❌ Error</h2>
            <p style="color: #6c757d;">{mensaje}</p>
        </div>
        """

        self.resumen_content.setText(error_html)
        self.metricas_content.setText(error_html)
        self.datos_content.setText(error_html)

    def limpiar_resultados(self):
        """Limpiar todos los resultados"""
        self.current_data = None
        self.current_analysis = None

        self.titulo_resultado.setText("📊 Resultados del Análisis")
        self.resumen_content.setText("Ejecuta un análisis para ver el resumen de resultados")
        self.metricas_content.setText("Las métricas detalladas aparecerán aquí")
        self.datos_content.setText("Los datos procesados aparecerán aquí")

        if MATPLOTLIB_AVAILABLE:
            self.figure.clear()
            self.canvas.draw()

    def apply_styles(self):
        """Aplicar estilos CSS"""
        styles = """
        #resultTitle {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            margin: 10px 0;
        }
        
        #resultTabs {
            font-size: 12px;
        }
        
        #resumenContent, #metricasContent, #datosContent {
            font-size: 13px;
            line-height: 1.4;
            padding: 10px;
        }
        
        #warningLabel {
            font-size: 14px;
            color: #ff9800;
            font-weight: bold;
        }
        """

        self.setStyleSheet(styles)


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

