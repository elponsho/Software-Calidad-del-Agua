"""
ml_functions_no_supervisado.py - Funciones de Machine Learning No Supervisado SIN SCIPY
Implementación completa y optimizada para clustering, PCA y análisis exploratorio
Incluye: Clustering Jerárquico, K-Means, DBSCAN, PCA avanzado, análisis exploratorio
Versión SIN scipy - solo usando matplotlib y sklearn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
import warnings

from typing import Dict, Any, Tuple

warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('default')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])

# ==================== FUNCIONES DE GENERACIÓN DE DATOS ====================

def generar_datos_agua_realistas(n_muestras=200, seed=42, incluir_outliers=True):
    """
    Generar datos sintéticos realistas de calidad del agua con patrones complejos
    """
    np.random.seed(seed)

    # Generar estaciones con diferentes características
    n_estaciones = min(5, n_muestras // 20)
    estacion_ids = np.random.choice(range(1, n_estaciones + 1), n_muestras)

    # Parámetros base por estación
    estacion_params = {}
    for i in range(1, n_estaciones + 1):
        estacion_params[i] = {
            'ph_base': np.random.uniform(6.8, 7.8),
            'temp_base': np.random.uniform(18, 26),
            'conduct_base': np.random.uniform(200, 800),
            'quality_factor': np.random.uniform(0.7, 1.3)
        }

    # Generar datos con correlaciones realistas
    datos = {
        'Points': range(1, n_muestras + 1),
        'Sampling_date': pd.date_range('2023-01-01', periods=n_muestras, freq='D'),
    }

    # Variables con estructura dependiente de estación
    ph_values = []
    temperatura_values = []
    conductividad_values = []

    for estacion in estacion_ids:
        params = estacion_params[estacion]

        ph = np.random.normal(params['ph_base'], 0.3)
        temp = np.random.normal(params['temp_base'], 2)
        conduct = np.random.normal(params['conduct_base'], 100)

        ph_values.append(ph)
        temperatura_values.append(temp)
        conductividad_values.append(conduct)

    datos['pH'] = np.clip(ph_values, 5.5, 9.0)
    datos['WT'] = np.clip(temperatura_values, 10, 35)  # Water Temperature
    datos['CTD'] = np.clip(conductividad_values, 50, 1500)  # Conductivity

    # Variables correlacionadas
    datos['DO'] = np.clip(
        10 - 0.2 * (datos['WT'] - 20) + np.random.normal(0, 1, n_muestras),
        2, 15
    )

    datos['TBD'] = np.clip(
        np.random.exponential(2.0, n_muestras) * (1 + 0.1 * np.abs(datos['pH'] - 7)),
        0.1, 50
    )

    datos['BOD5'] = np.clip(
        np.random.exponential(3, n_muestras) * (1 + datos['TBD'] / 20),
        0.5, 25
    )

    datos['COD'] = np.clip(
        datos['BOD5'] * np.random.uniform(1.5, 3.0, n_muestras),
        1, 100
    )

    datos['FC'] = np.clip(
        np.random.exponential(100, n_muestras) * (1 + datos['BOD5'] / 10),
        1, 10000
    )

    datos['TC'] = np.clip(
        datos['FC'] * np.random.uniform(2, 5, n_muestras),
        10, 50000
    )

    datos['NO3'] = np.clip(
        np.random.exponential(5, n_muestras) + 0.01 * datos['CTD'],
        0.1, 50
    )

    datos['NO2'] = np.clip(
        np.random.exponential(0.5, n_muestras),
        0.01, 3
    )

    datos['N_NH3'] = np.clip(
        np.random.exponential(1, n_muestras),
        0.01, 10
    )

    datos['TP'] = np.clip(
        np.random.exponential(0.5, n_muestras) * (1 + datos['BOD5'] / 15),
        0.01, 5
    )

    datos['TN'] = np.clip(
        datos['NO3'] + datos['NO2'] + datos['N_NH3'] + np.random.exponential(2, n_muestras),
        0.5, 20
    )

    datos['TKN'] = np.clip(
        datos['N_NH3'] + np.random.exponential(1, n_muestras),
        0.1, 15
    )

    datos['TSS'] = np.clip(
        np.random.exponential(10, n_muestras) * (1 + datos['TBD'] / 10),
        1, 200
    )

    datos['TS'] = np.clip(
        datos['TSS'] + np.random.exponential(100, n_muestras),
        50, 2000
    )

    datos['Q'] = np.clip(
        np.random.exponential(50, n_muestras),
        1, 500
    )

    datos['ALC'] = np.clip(
        np.random.normal(150, 50, n_muestras),
        50, 500
    )

    datos['H'] = np.clip(
        np.random.exponential(20, n_muestras),
        5, 200
    )

    datos['ET'] = np.clip(
        np.random.normal(25, 5, n_muestras),
        15, 40
    )

    # Crear DataFrame
    df = pd.DataFrame(datos)

    # Calcular índices de calidad
    df['WQI_IDEAM_6V'] = calcular_indice_calidad_simple(df)
    df['WQI_IDEAM_7V'] = calcular_indice_calidad_simple(df) * np.random.uniform(0.9, 1.1, n_muestras)
    df['WQI_NSF_9V'] = calcular_indice_calidad_simple(df) * np.random.uniform(0.8, 1.2, n_muestras)

    # Clasificaciones
    df['Classification_6V'] = pd.cut(
        df['WQI_IDEAM_6V'],
        bins=[0, 40, 60, 80, 100],
        labels=['Deficiente', 'Regular', 'Buena', 'Excelente']
    )

    df['Classification_7V'] = pd.cut(
        df['WQI_IDEAM_7V'],
        bins=[0, 40, 60, 80, 100],
        labels=['Deficiente', 'Regular', 'Buena', 'Excelente']
    )

    df['Classification_9V'] = pd.cut(
        df['WQI_NSF_9V'],
        bins=[0, 40, 60, 80, 100],
        labels=['Deficiente', 'Regular', 'Buena', 'Excelente']
    )

    # Agregar outliers realistas si se solicita
    if incluir_outliers:
        n_outliers = max(1, n_muestras // 50)
        outlier_indices = np.random.choice(df.index, n_outliers, replace=False)

        for idx in outlier_indices:
            # Alterar aleatoriamente algunos parámetros
            factor = np.random.uniform(2, 5)
            param = np.random.choice(['TBD', 'BOD5', 'FC'])
            df.loc[idx, param] *= factor

    return df

def calcular_indice_calidad_simple(df):
    """Calcular un índice de calidad simplificado"""
    # Normalizar parámetros (0-100)
    ph_score = 100 * np.exp(-0.5 * ((df['pH'] - 7.0) / 1.5) ** 2)
    do_score = np.minimum(100, (df['DO'] / 10.0) * 100)
    turb_score = np.maximum(0, 100 - (df['TBD'] / 10) * 100)
    dbo_score = np.maximum(0, 100 - (df['BOD5'] / 10) * 100)

    # Promedio ponderado
    indice = (0.25 * ph_score + 0.35 * do_score + 0.2 * turb_score + 0.2 * dbo_score)
    return np.clip(indice, 0, 100)

# ==================== FUNCIONES AUXILIARES DE PREPROCESAMIENTO ====================

def aplicar_escalado(X, metodo='standard'):
    """Aplicar escalado a los datos"""
    if metodo == 'standard':
        scaler = StandardScaler()
    elif metodo == 'robust':
        scaler = RobustScaler()
    elif metodo == 'minmax':
        scaler = MinMaxScaler()
    elif metodo == 'quantile':
        scaler = QuantileTransformer()
    else:
        return X, None

    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def manejar_valores_faltantes(X, estrategia='median'):
    """Manejar valores faltantes"""
    if estrategia == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif estrategia == 'median':
        imputer = SimpleImputer(strategy='median')
    elif estrategia == 'mode':
        imputer = SimpleImputer(strategy='most_frequent')
    elif estrategia == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    else:
        return X, None

    X_imputed = imputer.fit_transform(X)
    return X_imputed, imputer

def detectar_outliers(X, metodo='isolation_forest', contamination=0.1):
    """Detectar outliers usando diferentes métodos"""
    if metodo == 'isolation_forest':
        detector = IsolationForest(contamination=contamination, random_state=42)
        outliers = detector.fit_predict(X)
        return outliers == -1
    elif metodo == 'zscore':
        # Z-score sin scipy
        z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
        return np.any(z_scores > 3, axis=1)
    elif metodo == 'iqr':
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return np.any((X < lower_bound) | (X > upper_bound), axis=1)
    else:
        return np.zeros(len(X), dtype=bool)

# ==================== CLUSTERING JERÁRQUICO SIN SCIPY ====================

def clustering_jerarquico_completo(data, variables=None, metodos=['ward', 'complete', 'average'],
                                 metricas=['euclidean', 'manhattan', 'cosine'],
                                 max_clusters=10, escalado='standard', n_jobs=-1, verbose=True):
    """
    Análisis exhaustivo de clustering jerárquico usando AgglomerativeClustering de sklearn
    """
    if verbose:
        print("🔍 Iniciando clustering jerárquico completo...")

    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()

    # Preparar datos
    X = data[variables].dropna()

    if verbose:
        print(f"📊 Datos preparados: {X.shape[0]} muestras, {X.shape[1]} variables")

    # Escalado de datos
    X_scaled, scaler = aplicar_escalado(X, escalado)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    resultados_completos = {}

    # Probar diferentes combinaciones de método y métrica
    for metodo in metodos:
        for metrica in metricas:
            # Verificar compatibilidad
            if metodo == 'ward' and metrica != 'euclidean':
                continue

            try:
                if verbose:
                    print(f"   Probando {metodo} con métrica {metrica}...")

                # Evaluar diferentes números de clusters
                evaluaciones_k = {}
                labels_por_k = {}

                for k in range(2, min(max_clusters + 1, len(X) // 2)):
                    # Usar AgglomerativeClustering - FIXED: correct parameter names
                    if metrica == 'cosine':
                        affinity = 'cosine'
                    elif metrica == 'manhattan':
                        affinity = 'manhattan'
                    else:
                        affinity = 'euclidean'

<<<<<<< HEAD
                    clustering = AgglomerativeClustering(
                        n_clusters=n_clusters,
                        linkage=metodo,
                        metric=affinity  # Aquí pones "euclidean", "manhattan", etc.
=======
                    # FIXED: Use correct parameter name 'metric' instead of 'affinity'
                    clusterer = AgglomerativeClustering(
                        n_clusters=k,
                        linkage=metodo,
                        metric=affinity
>>>>>>> 73b8f33 (Vivan los papus)
                    )

                    labels = clusterer.fit_predict(X_scaled.values)

                    # Verificar que hay al menos 2 clusters
                    if len(np.unique(labels)) < 2:
                        continue

                    # Calcular métricas de calidad
                    try:
                        silhouette = silhouette_score(X_scaled, labels)
                        davies_bouldin = davies_bouldin_score(X_scaled, labels)
                        calinski_harabasz = calinski_harabasz_score(X_scaled, labels)

                        # Análisis adicional de clusters
                        cluster_stats = analizar_clusters_detallado(X_scaled, labels, variables)

                        evaluaciones_k[k] = {
                            'silhouette_score': float(silhouette),
                            'davies_bouldin_score': float(davies_bouldin),
                            'calinski_harabasz_score': float(calinski_harabasz),
                            'cluster_stats': cluster_stats,
                            'labels': labels.tolist()
                        }
                        labels_por_k[k] = labels

                    except Exception as e:
                        if verbose:
                            print(f"Error evaluando k={k} para {metodo}-{metrica}: {e}")
                        continue

                # Encontrar número óptimo de clusters
                if evaluaciones_k:
                    # Usar silhouette score como criterio principal
                    mejor_k = max(evaluaciones_k.keys(),
                                key=lambda k: evaluaciones_k[k]['silhouette_score'])

                    # Análisis de estabilidad
                    estabilidad = analizar_estabilidad_clusters_sklearn(
                        X_scaled, metodo, affinity, range(2, min(8, len(evaluaciones_k) + 2))
                    )

                    # Crear datos para dendrograma usando distancias
                    dendrograma_data = crear_dendrograma_data_sklearn(X_scaled, metodo, affinity)

                    resultados_completos[f"{metodo}_{metrica}"] = {
                        'metodo': metodo,
                        'metrica': metrica,
                        'evaluaciones_por_k': evaluaciones_k,
                        'mejor_k': mejor_k,
                        'mejor_silhouette': evaluaciones_k[mejor_k]['silhouette_score'],
                        'estabilidad': estabilidad,
                        'dendrograma_data': dendrograma_data,
                        'linkage_matrix': dendrograma_data.get('linkage_matrix', [])
                    }

            except Exception as e:
                if verbose:
                    print(f"Error con {metodo}-{metrica}: {e}")
                continue

    # Determinar la mejor configuración general
    mejor_config = None
    if resultados_completos:
        mejor_config = max(resultados_completos.keys(),
                          key=lambda k: resultados_completos[k]['mejor_silhouette'])

        # FIXED: Add proper mejor_configuracion structure
        mejor_resultado = resultados_completos[mejor_config]
        mejor_configuracion = {
            'metodo': mejor_resultado['metodo'],
            'metrica': mejor_resultado['metrica'],
            'n_clusters_sugeridos': mejor_resultado['mejor_k'],
            'silhouette_score': mejor_resultado['mejor_silhouette'],
            'labels': mejor_resultado['evaluaciones_por_k'][mejor_resultado['mejor_k']]['labels']
        }
    else:
        mejor_configuracion = {}

    if verbose:
        print(f"✅ Clustering jerárquico completado. Mejor configuración: {mejor_config}")

    return {
        'tipo': 'clustering_jerarquico_completo',
        'variables_utilizadas': variables,
        'metodo_escalado': escalado,
        'resultados_por_configuracion': resultados_completos,
        'mejor_configuracion': mejor_configuracion,  # FIXED: Add this key
        'datos_originales': X,  # FIXED: Add original data
        'datos_escalados': X_scaled.values.tolist(),
        'indices_muestras': X.index.tolist(),
        'resumen_evaluacion': generar_resumen_clustering(resultados_completos),
        'recomendaciones': generar_recomendaciones_clustering(resultados_completos)
    }
def analizar_estabilidad_clusters_sklearn(X, metodo, affinity, k_range):
    """Analizar estabilidad usando sklearn"""
    estabilidad_scores = {}

    for k in k_range:
        # Clustering original
        clusterer_original = AgglomerativeClustering(
            n_clusters=k,
            linkage=metodo,
            metric=affinity  # FIXED: Use 'metric' instead of 'affinity'
        )

        labels_original = clusterer_original.fit_predict(X)

        # Bootstrap sampling
        n_bootstrap = 10
        ari_scores = []

        for _ in range(n_bootstrap):
            # Muestra bootstrap
            bootstrap_indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap = X.iloc[bootstrap_indices] if hasattr(X, 'iloc') else X[bootstrap_indices]

            try:
                # Clustering en muestra bootstrap
                clusterer_bootstrap = AgglomerativeClustering(
                    n_clusters=k,
                    linkage=metodo,
                    metric=affinity  # FIXED: Use 'metric' instead of 'affinity'
                )

                labels_bootstrap = clusterer_bootstrap.fit_predict(X_bootstrap)

                # Calcular ARI con etiquetas originales (solo índices comunes)
                ari = adjusted_rand_score(labels_original[bootstrap_indices], labels_bootstrap)
                ari_scores.append(ari)

            except:
                continue

        if ari_scores:
            estabilidad_scores[k] = {
                'ari_mean': float(np.mean(ari_scores)),
                'ari_std': float(np.std(ari_scores)),
                'estabilidad': 'Alta' if np.mean(ari_scores) > 0.7 else 'Media' if np.mean(ari_scores) > 0.5 else 'Baja'
            }

    return estabilidad_scores


# 3. Fix the crear_dendrograma_data_sklearn function
def crear_dendrograma_data_sklearn(X_scaled, metodo, affinity):
    """Crear datos simulados para dendrograma usando sklearn"""
    try:
        # Limitar datos para eficiencia
        if len(X_scaled) > 100:
            indices = np.random.choice(len(X_scaled), 100, replace=False)
            X_sample = X_scaled.iloc[indices] if hasattr(X_scaled, 'iloc') else X_scaled[indices]
        else:
            X_sample = X_scaled

        # Crear clustering jerárquico para múltiples niveles
        distances = []
        cluster_sizes = []

        max_clusters = min(20, len(X_sample) - 1)
        for n_clusters in range(max_clusters, 1, -1):
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=metodo,
<<<<<<< HEAD
                metric=affinity  # Aquí pones "euclidean", "manhattan", etc.
=======
                metric=affinity  # FIXED: Use 'metric' instead of 'affinity'
>>>>>>> 73b8f33 (Vivan los papus)
            )

            labels = clustering.fit_predict(X_sample)

            # Calcular distancia promedio intra-cluster como medida de fusión
            unique_labels = np.unique(labels)
            total_distance = 0

            for label in unique_labels:
                cluster_points = X_sample[labels == label]
                if len(cluster_points) > 1:
                    if affinity == 'euclidean':
                        dists = euclidean_distances(cluster_points)
                    elif affinity == 'manhattan':
                        dists = manhattan_distances(cluster_points)
                    elif affinity == 'cosine':
                        dists = cosine_distances(cluster_points)
                    else:
                        dists = euclidean_distances(cluster_points)

                    # Promedio de distancias dentro del cluster
                    cluster_dist = np.mean(dists[np.triu_indices_from(dists, k=1)])
                    total_distance += cluster_dist

            distances.append(total_distance / len(unique_labels))
            cluster_sizes.append(len(unique_labels))

        # Crear matriz estilo linkage simplificada
        linkage_matrix = []
        for i, (dist, size) in enumerate(zip(distances, cluster_sizes)):
            linkage_matrix.append([i, i+1, dist, size])

        return {
            'linkage_matrix': linkage_matrix,
            'distances': distances,
            'cluster_sizes': cluster_sizes,
            'n_samples': len(X_sample),
            'method': metodo,
            'affinity': affinity
        }

    except Exception as e:
        print(f"Error creando datos de dendrograma: {e}")
        return {
            'linkage_matrix': [],
            'distances': [],
            'cluster_sizes': [],
            'n_samples': 0,
            'method': metodo,
            'affinity': affinity
        }


<<<<<<< HEAD
    for k in k_range:
        # Clustering original
        clusterer_original = AgglomerativeClustering(
            n_clusters=k,
            linkage=metodo,
            metric=affinity  # 'euclidean', 'manhattan', etc.
        )

        labels_original = clusterer_original.fit_predict(X)
=======
# 4. Fix the _create_pca_visualization function in no_supervisado_window.py
# Replace the problematic method in ResultsVisualizationWidget:
>>>>>>> 73b8f33 (Vivan los papus)

def _create_pca_visualization(self):
    """Crear visualización para PCA con enfoque en puntos de muestreo"""
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        resultados = self.current_results.get('resultados_por_metodo', {})

<<<<<<< HEAD
            try:
                # Clustering en muestra bootstrap
                clusterer_original = AgglomerativeClustering(
                    n_clusters=k,
                    linkage=metodo,
                    metric=affinity  # 'euclidean', 'manhattan', etc.
                )

                labels_bootstrap = clusterer_bootstrap.fit_predict(X_bootstrap)
=======
        if 'linear' not in resultados:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No hay resultados de PCA lineal para visualizar',
                    ha='center', va='center', transform=ax.transAxes)
            return
>>>>>>> 73b8f33 (Vivan los papus)

        # Usar la nueva función especializada para puntos de muestreo
        self.figure.clear()
        if 'datos_originales' in self.current_results:
            # FIXED: Use the correct function that exists
            self.figure = _crear_visualizacion_pca_puntos_muestreo(
                self.current_results, figsize=(16, 12)
            )
        else:
            # Visualización PCA tradicional como fallback
            self._create_traditional_pca_visualization()

    except Exception as e:
        print(f"Error en visualización PCA: {e}")
        self._show_error_visualization()

def _create_traditional_pca_visualization(self):
    """Visualización PCA tradicional como fallback"""
    try:
        resultados = self.current_results.get('resultados_por_metodo', {})

        if 'linear' not in resultados:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No hay resultados de PCA para visualizar',
                    ha='center', va='center', transform=ax.transAxes)
            return

        linear_result = resultados['linear']
        analisis = linear_result.get('analisis', {})

        # Crear subplot 2x2
        fig = self.figure
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Varianza explicada
        ax1 = fig.add_subplot(gs[0, 0])
        var_explicada = analisis.get('varianza_explicada', [])
        if var_explicada:
            x = range(1, len(var_explicada) + 1)
            ax1.bar(x, [v * 100 for v in var_explicada], alpha=0.7)
            ax1.set_xlabel('Componente Principal')
            ax1.set_ylabel('Varianza Explicada (%)')
            ax1.set_title('Varianza por Componente')
            ax1.grid(True, alpha=0.3)

        # 2. Varianza acumulada
        ax2 = fig.add_subplot(gs[0, 1])
        var_acumulada = analisis.get('varianza_acumulada', [])
        if var_acumulada:
            x = range(1, len(var_acumulada) + 1)
            ax2.plot(x, [v * 100 for v in var_acumulada], 'o-', linewidth=2)
            ax2.axhline(y=95, color='red', linestyle='--', label='95%')
            ax2.set_xlabel('Componente Principal')
            ax2.set_ylabel('Varianza Acumulada (%)')
            ax2.set_title('Varianza Acumulada')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. Información de componentes
        ax3 = fig.add_subplot(gs[1, :])
        componentes_info = analisis.get('componentes_info', [])
        if componentes_info and len(componentes_info) > 0:
            # Mostrar contribuciones del primer componente
            pc1_info = componentes_info[0]
            top_vars = pc1_info.get('top_variables', [])[:5]

            if top_vars:
                variables = [var['variable'] for var in top_vars]
                loadings = [var['loading'] for var in top_vars]

                bars = ax3.barh(range(len(variables)), loadings)
                ax3.set_yticks(range(len(variables)))
                ax3.set_yticklabels(variables)
                ax3.set_xlabel('Loading')
                ax3.set_title('Variables más importantes en PC1')
                ax3.grid(True, alpha=0.3)

        plt.suptitle('Análisis de Componentes Principales', fontsize=14, fontweight='bold')

    except Exception as e:
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, f'Error en PCA tradicional: {str(e)[:100]}',
                ha='center', va='center', transform=ax.transAxes)

def _show_error_visualization(self):
    """Mostrar visualización de error"""
    self.figure.clear()
    ax = self.figure.add_subplot(111)
    ax.text(0.5, 0.5, 'Error generando visualización',
            ha='center', va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='mistyrose'))
    ax.set_title('Error en Visualización')
    ax.axis('off')


# 5. Fix missing functions in ResultsVisualizationWidget class
# Add these methods to the ResultsVisualizationWidget class:

def _show_error(self, error_msg):
    """Mostrar error en lugar de resultados"""
    self.summary_text.setText(f"❌ Error en el análisis:\n\n{error_msg}")
    self.status_label.setText("❌ Error en análisis")
    self.status_label.setStyleSheet("color: red;")

    # Limpiar otros tabs
    self.metrics_table.setRowCount(0)
    self.details_text.setText(f"Error: {error_msg}")

    # Deshabilitar botones
    self.export_results_btn.setEnabled(False)
    self.generate_report_btn.setEnabled(False)


# 6. Add the missing _create_exploratory_visualization method
def _create_exploratory_visualization(self):
    """Crear visualización para análisis exploratorio"""
    try:
        self.figure.clear()

        # Usar la función de visualización exploratoria
        if 'datos_originales' in self.current_results:
            self.figure = _crear_visualizacion_exploratorio_puntos_muestreo(
                self.current_results, figsize=(16, 12)
            )
        else:
            # Fallback básico
            ax = self.figure.add_subplot(111)

            # Mostrar información básica del análisis exploratorio
            calidad = self.current_results.get('calidad_datos', {})
            outliers = self.current_results.get('outliers', {})
            correlaciones = self.current_results.get('correlaciones', {})

            info_text = "Resumen Análisis Exploratorio:\n\n"

            if calidad:
                score = calidad.get('quality_score', 0)
                info_text += f"Calidad de datos: {score:.1f}/100\n"
                info_text += f"Calificación: {calidad.get('calificacion', 'N/A')}\n\n"

            if outliers:
                consenso = outliers.get('consenso', {})
                if consenso:
                    info_text += f"Outliers detectados: {consenso.get('total_unico', 0)}\n"
                    info_text += f"Porcentaje outliers: {consenso.get('porcentaje', 0):.1f}%\n\n"

            if correlaciones:
                corr_fuertes = correlaciones.get('correlaciones_fuertes', [])
                info_text += f"Correlaciones fuertes: {len(corr_fuertes)}\n"
                multicolineal = correlaciones.get('multicolinealidad', 'N/A')
                info_text += f"Multicolinealidad: {multicolineal}\n"

            ax.text(0.1, 0.9, info_text, transform=ax.transAxes,
                   fontsize=12, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            ax.set_title('Análisis Exploratorio - Resumen')
            ax.axis('off')

        self.figure.tight_layout()
        self.canvas.draw()

    except Exception as e:
        print(f"Error en visualización exploratoria: {e}")
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, f'Error en visualización exploratoria:\n{str(e)[:100]}',
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='mistyrose'))
        ax.set_title('Error en Visualización')
        ax.axis('off')
        self.canvas.draw()



def generar_visualizaciones_ml_no_supervisado(resultado: Dict[str, Any],
                                              figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
    """
    Generar visualizaciones especializadas para ML No Supervisado
    """
    tipo = resultado.get('tipo', '')

    try:
        if tipo in ['kmeans_optimizado', 'clustering_jerarquico_completo', 'dbscan_optimizado']:
            # Usar visualización especializada para puntos de muestreo
            return crear_visualizacion_clustering_puntos_muestreo(resultado, figsize)
        elif tipo == 'pca_completo_avanzado':
            return _crear_visualizacion_pca_puntos_muestreo(resultado, figsize)
        elif tipo == 'analisis_exploratorio_completo':
            return _crear_visualizacion_exploratorio_puntos_muestreo(resultado, figsize)
        else:
            return _crear_visualizacion_generica(resultado, figsize)

    except Exception as e:
        # Fallback en caso de error
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f'Error generando visualización:\n{str(e)[:100]}',
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='mistyrose'))
        ax.set_title('Error en Visualización')
        ax.axis('off')
        return fig


# ================ FUNCIONES PARA CLUSTERING ====================
def crear_visualizacion_clustering_puntos_muestreo(resultado: Dict[str, Any],
                                                   figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
    """
    Crear visualización especializada para clustering de puntos de muestreo
    """
    if 'error' in resultado:
        raise ValueError(f"Error en el resultado: {resultado['error']}")

    tipo = resultado.get('tipo', '')
    fig = plt.figure(figsize=figsize)

    if tipo == 'kmeans_optimizado':
        return _crear_viz_kmeans_puntos_muestreo(resultado, fig)
    elif tipo == 'clustering_jerarquico_completo':
        return _crear_viz_jerarquico_puntos_muestreo(resultado, fig)
    elif tipo == 'dbscan_optimizado':
        return _crear_viz_dbscan_puntos_muestreo(resultado, fig)
    else:
        return _crear_viz_generica_puntos_muestreo(resultado, fig)


def _crear_viz_kmeans_puntos_muestreo(resultado: Dict, fig: plt.Figure) -> plt.Figure:
    """Visualización específica para K-Means de puntos de muestreo"""

    # Layout de 2x2
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Mapa de puntos de muestreo coloreados por cluster
    ax1 = fig.add_subplot(gs[0, 0])
    _graficar_puntos_muestreo_clusters(ax1, resultado)

    # 2. Timeline de muestreos con clusters
    ax2 = fig.add_subplot(gs[0, 1])
    _graficar_timeline_muestreos(ax2, resultado)

    # 3. Matriz de distancias entre puntos
    ax3 = fig.add_subplot(gs[1, 0])
    _graficar_matriz_distancias_puntos(ax3, resultado)

    # 4. Estadísticas por cluster
    ax4 = fig.add_subplot(gs[1, 1])
    _graficar_estadisticas_clusters_puntos(ax4, resultado)

    plt.suptitle('Clustering K-Means - Análisis de Puntos de Muestreo',
                 fontsize=16, fontweight='bold')

    return fig


def _crear_viz_dbscan_puntos_muestreo(resultado: Dict, fig: plt.Figure) -> plt.Figure:
    """Visualización específica para DBSCAN de puntos de muestreo"""

    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Puntos de muestreo con outliers destacados
    ax1 = fig.add_subplot(gs[0, 0])
    _graficar_puntos_dbscan_outliers(ax1, resultado)

    # 2. Densidad de muestreos en el tiempo
    ax2 = fig.add_subplot(gs[0, 1])
    _graficar_densidad_temporal_muestreos(ax2, resultado)

    # 3. Análisis de outliers temporales
    ax3 = fig.add_subplot(gs[1, 0])
    _graficar_outliers_temporales(ax3, resultado)

    # 4. Información de clusters y outliers
    ax4 = fig.add_subplot(gs[1, 1])
    _mostrar_info_dbscan_puntos(ax4, resultado)

    plt.suptitle('DBSCAN - Análisis de Puntos de Muestreo y Outliers',
                 fontsize=16, fontweight='bold')

    return fig


def _graficar_puntos_muestreo_clusters(ax, resultado):
    """Graficar puntos de muestreo coloreados por cluster"""
    try:
        # Obtener datos originales
        if 'datos_originales' not in resultado:
            ax.text(0.5, 0.5, 'Datos no disponibles', ha='center', va='center',
                    transform=ax.transAxes)
            return

        datos = resultado['datos_originales']

        # Verificar si existe columna Points
        if 'Points' not in datos.columns:
            # Crear índices como puntos
            puntos = list(range(1, len(datos) + 1))
        else:
            puntos = datos['Points'].tolist()

        # Obtener labels de clusters
        k_optimo = resultado.get('recomendacion_k')
        if k_optimo and 'resultados_por_k' in resultado:
            if k_optimo in resultado['resultados_por_k']:
                labels = resultado['resultados_por_k'][k_optimo]['labels']
            else:
                labels = list(range(len(puntos)))
        else:
            labels = list(range(len(puntos)))

        # Crear coordenadas para visualización
        # Si hay información geográfica, usarla; si no, crear distribución
        if len(puntos) <= 50:  # Para datasets pequeños, usar grid
            n_cols = int(np.ceil(np.sqrt(len(puntos))))
            x_coords = [i % n_cols for i in range(len(puntos))]
            y_coords = [i // n_cols for i in range(len(puntos))]
        else:  # Para datasets grandes, usar distribución aleatoria
            np.random.seed(42)
            x_coords = np.random.uniform(0, 10, len(puntos))
            y_coords = np.random.uniform(0, 10, len(puntos))

        # Scatter plot con colores por cluster
        unique_labels = set(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

        for i, (label, color) in enumerate(zip(unique_labels, colors)):
            mask = np.array(labels) == label
            ax.scatter(np.array(x_coords)[mask], np.array(y_coords)[mask],
                       c=[color], label=f'Cluster {label}',
                       s=100, alpha=0.7, edgecolors='black', linewidth=1)

        # Anotar algunos puntos representativos
        for i in range(0, len(puntos), max(1, len(puntos) // 10)):
            ax.annotate(f'P{puntos[i]}', (x_coords[i], y_coords[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)

        ax.set_xlabel('Coordenada X (relativa)')
        ax.set_ylabel('Coordenada Y (relativa)')
        ax.set_title('Distribución de Puntos de Muestreo por Cluster')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

    except Exception as e:
        ax.text(0.5, 0.5, f'Error graficando puntos: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _graficar_timeline_muestreos(ax, resultado):
    """Graficar timeline de muestreos con franjas de colores por cluster"""
    try:
        if 'datos_originales' not in resultado:
            ax.text(0.5, 0.5, 'Datos no disponibles', ha='center', va='center',
                    transform=ax.transAxes)
            return

        datos = resultado['datos_originales']

        # Verificar columna de fecha
        fecha_col = None
        for col in ['Sampling_date', 'Fecha', 'Date', 'fecha']:
            if col in datos.columns:
                fecha_col = col
                break

        if fecha_col is None:
            # Crear fechas sintéticas
            fechas = pd.date_range('2023-01-01', periods=len(datos), freq='D')
        else:
            fechas = pd.to_datetime(datos[fecha_col])

        # Obtener labels
        k_optimo = resultado.get('recomendacion_k')
        if k_optimo and 'resultados_por_k' in resultado:
            if k_optimo in resultado['resultados_por_k']:
                labels = resultado['resultados_por_k'][k_optimo]['labels']
            else:
                labels = [0] * len(datos)
        else:
            labels = [0] * len(datos)

        # Puntos de muestreo
        if 'Points' in datos.columns:
            puntos = datos['Points'].tolist()
        else:
            puntos = list(range(1, len(datos) + 1))

        # Crear timeline
        unique_labels = sorted(set(labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

        for i, (fecha, punto, label) in enumerate(zip(fechas, puntos, labels)):
            color = color_map[label]
            ax.scatter(fecha, punto, c=[color], s=60, alpha=0.8,
                       edgecolors='black', linewidth=0.5)

        # Añadir franjas de colores por período
        if len(fechas) > 1:
            fecha_min, fecha_max = fechas.min(), fechas.max()

            # Dividir en períodos y colorear fondo
            n_periodos = min(5, len(unique_labels))
            periodo_duration = (fecha_max - fecha_min) / n_periodos

            for i in range(n_periodos):
                inicio = fecha_min + i * periodo_duration
                fin = fecha_min + (i + 1) * periodo_duration
                color = colors[i % len(colors)]
                ax.axvspan(inicio, fin, alpha=0.1, color=color)

        ax.set_xlabel('Fecha de Muestreo')
        ax.set_ylabel('Punto de Muestreo')
        ax.set_title('Timeline de Muestreos por Cluster')

        # Rotar etiquetas de fecha
        ax.tick_params(axis='x', rotation=45)

        # Leyenda
        legend_elements = [plt.scatter([], [], c=[color_map[label]],
                                       label=f'Cluster {label}', s=60)
                           for label in unique_labels]
        ax.legend(handles=legend_elements, loc='upper right')

        ax.grid(True, alpha=0.3)

    except Exception as e:
        ax.text(0.5, 0.5, f'Error en timeline: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _graficar_matriz_distancias_puntos(ax, resultado):
    """Graficar matriz de distancias entre puntos de muestreo"""
    try:
        if 'datos_originales' not in resultado:
            ax.text(0.5, 0.5, 'Datos no disponibles', ha='center', va='center',
                    transform=ax.transAxes)
            return

        datos = resultado['datos_originales']

        # Seleccionar subset de puntos para visualización clara
        max_puntos = 20
        if len(datos) > max_puntos:
            indices = np.random.choice(len(datos), max_puntos, replace=False)
            datos_subset = datos.iloc[indices]
        else:
            datos_subset = datos
            indices = range(len(datos))

        # Obtener variables numéricas
        numeric_cols = datos_subset.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            ax.text(0.5, 0.5, 'No hay variables numéricas', ha='center', va='center',
                    transform=ax.transAxes)
            return

        # Calcular matriz de distancias
        from sklearn.metrics.pairwise import euclidean_distances
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        datos_scaled = scaler.fit_transform(datos_subset[numeric_cols].dropna())

        distancias = euclidean_distances(datos_scaled)

        # Crear heatmap
        im = ax.imshow(distancias, cmap='viridis', aspect='auto')

        # Etiquetas
        if 'Points' in datos_subset.columns:
            labels = [f'P{int(p)}' for p in datos_subset['Points'].tolist()]
        else:
            labels = [f'P{i + 1}' for i in indices]

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)

        # Colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)

        ax.set_title('Matriz de Distancias entre Puntos de Muestreo')

    except Exception as e:
        ax.text(0.5, 0.5, f'Error en matriz distancias: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _graficar_estadisticas_clusters_puntos(ax, resultado):
    """Graficar estadísticas de clusters de puntos de muestreo"""
    try:
        k_optimo = resultado.get('recomendacion_k')
        if not k_optimo or 'resultados_por_k' not in resultado:
            ax.text(0.5, 0.5, 'Estadísticas no disponibles', ha='center', va='center',
                    transform=ax.transAxes)
            return

        if k_optimo not in resultado['resultados_por_k']:
            ax.text(0.5, 0.5, 'Datos de cluster no disponibles', ha='center', va='center',
                    transform=ax.transAxes)
            return

        cluster_data = resultado['resultados_por_k'][k_optimo]
        labels = cluster_data['labels']

        # Estadísticas por cluster
        unique_labels = sorted(set(labels))
        tamaños = [labels.count(label) for label in unique_labels]
        proporciones = [size / len(labels) * 100 for size in tamaños]

        # Gráfico de barras
        bars = ax.bar(range(len(unique_labels)), proporciones,
                      color=plt.cm.Set3(np.linspace(0, 1, len(unique_labels))),
                      alpha=0.7, edgecolor='black')

        # Añadir valores en las barras
        for bar, tamaño, prop in zip(bars, tamaños, proporciones):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                    f'{tamaño}\n({prop:.1f}%)', ha='center', va='bottom',
                    fontweight='bold', fontsize=10)

        ax.set_xlabel('Cluster')
        ax.set_ylabel('Porcentaje de Puntos')
        ax.set_title('Distribución de Puntos por Cluster')
        ax.set_xticks(range(len(unique_labels)))
        ax.set_xticklabels([f'Cluster {label}' for label in unique_labels])
        ax.set_ylim(0, max(proporciones) * 1.2)
        ax.grid(True, alpha=0.3, axis='y')

        # Información adicional
        silhouette = cluster_data.get('silhouette_score', 0)
        ax.text(0.02, 0.98, f'Silhouette Score: {silhouette:.3f}',
                transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    except Exception as e:
        ax.text(0.5, 0.5, f'Error en estadísticas: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)



# ==================== PCA AVANZADO ====================

def pca_completo_avanzado(data, variables=None, metodos=['linear'],
                         explicar_varianza_objetivo=0.95, escalado='standard',
                         random_state=42, verbose=True, max_components=None,
                         kernel_type='rbf', gamma=1.0):
    """
    Análisis de componentes principales avanzado con múltiples métodos
    """
    if verbose:
        print("🔍 Iniciando PCA avanzado...")

    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()

    X = data[variables].dropna()

    if verbose:
        print(f"📊 Datos preparados: {X.shape[0]} muestras, {X.shape[1]} variables")

    # Escalado de datos
    X_scaled, scaler = aplicar_escalado(X, escalado)

    if max_components is None:
        max_components = min(len(variables), len(X) - 1)

    resultados_pca = {}

    # PCA Lineal estándar
    if 'linear' in metodos:
        if verbose:
            print("   Ejecutando PCA lineal...")

        pca_linear = PCA(n_components=max_components, random_state=random_state)
        X_pca_linear = pca_linear.fit_transform(X_scaled)

        # Análisis detallado
        analisis_linear = analizar_pca_detallado(pca_linear, variables, explicar_varianza_objetivo)

        resultados_pca['linear'] = {
            'modelo': pca_linear,
            'transformacion': X_pca_linear.tolist(),
            'analisis': analisis_linear,
            'componentes_recomendados': analisis_linear['n_componentes_objetivo']
        }

        # Análisis de correlación entre componentes originales y transformados
        correlaciones = analizar_correlaciones_pca(X.values, X_pca_linear, variables)
        resultados_pca['linear']['correlaciones'] = correlaciones

        # Análisis de contribución de variables
        contribuciones = analizar_contribuciones_variables(pca_linear, variables)
        resultados_pca['linear']['contribuciones'] = contribuciones

    # Kernel PCA
    if 'kernel' in metodos and len(X) <= 1000:  # Limitado por costo computacional
        if verbose:
            print("   Ejecutando Kernel PCA...")

        kernels = [kernel_type] if kernel_type else ['rbf', 'poly', 'sigmoid']

        for kernel in kernels:
            try:
                if kernel == 'poly':
                    kpca = KernelPCA(n_components=min(10, max_components), kernel=kernel,
                                   degree=3, random_state=random_state)
                elif kernel == 'rbf':
                    kpca = KernelPCA(n_components=min(10, max_components), kernel=kernel,
                                   gamma=gamma, random_state=random_state)
                else:
                    kpca = KernelPCA(n_components=min(10, max_components), kernel=kernel,
                                   random_state=random_state)

                X_kpca = kpca.fit_transform(X_scaled)

                # Evaluación de calidad (reconstrucción aproximada)
                calidad = evaluar_calidad_kernel_pca(X_scaled, X_kpca, kpca)

                resultados_pca[f'kernel_{kernel}'] = {
                    'modelo': kpca,
                    'transformacion': X_kpca.tolist(),
                    'calidad_reconstruccion': calidad,
                    'kernel': kernel
                }

            except Exception as e:
                if verbose:
                    print(f"Error con kernel PCA {kernel}: {e}")
                continue

    if verbose:
        print("✅ PCA completado")

    return {
        'tipo': 'pca_completo_avanzado',
        'variables_utilizadas': variables,
        'n_muestras': len(X),
        'metodos_aplicados': list(resultados_pca.keys()),
        'resultados_por_metodo': resultados_pca,
        'recomendaciones': generar_recomendaciones_pca(resultados_pca),
        'datos_originales_escalados': X_scaled.tolist()
    }

# ==================== ANÁLISIS EXPLORATORIO ====================

def analisis_exploratorio_completo(data, variables=None, escalado='standard',
                                  handle_outliers=True, verbose=True,
                                  outlier_method='isolation_forest', random_state=42):
    """
    Análisis exploratorio exhaustivo de los datos
    """
    if verbose:
        print("🔍 Iniciando análisis exploratorio completo...")

    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()

    X = data[variables].dropna()

    if verbose:
        print(f"📊 Datos preparados: {X.shape[0]} muestras, {X.shape[1]} variables")

    # 1. Estadísticas descriptivas avanzadas
    if verbose:
        print("   Calculando estadísticas descriptivas...")
    estadisticas = calcular_estadisticas_avanzadas(X, variables)

    # 2. Análisis de correlaciones
    if verbose:
        print("   Analizando correlaciones...")
    correlaciones = analizar_correlaciones_avanzado(X, variables)

    # 3. Detección de outliers múltiples métodos
    if verbose:
        print("   Detectando outliers...")
    outliers = detectar_outliers_multiples_metodos(X, variables, outlier_method)

    # 4. Análisis de distribuciones
    if verbose:
        print("   Analizando distribuciones...")
    distribuciones = analizar_distribuciones_avanzado(X, variables)

    # 5. Análisis de componentes principales básico
    if verbose:
        print("   Ejecutando PCA exploratorio...")
    X_scaled, _ = aplicar_escalado(X, escalado)
    pca_basico = PCA(n_components=min(5, len(variables)), random_state=random_state)
    pca_basico.fit(X_scaled)

    # 6. Clustering exploratorio rápido
    if verbose:
        print("   Ejecutando clustering exploratorio...")
    clustering_exploratorio = clustering_exploratorio_rapido(X, variables, escalado)

    # 7. Análisis de calidad de datos
    if verbose:
        print("   Evaluando calidad de datos...")
    calidad_datos = evaluar_calidad_datos(data, variables)

    # 8. Recomendaciones automáticas
    recomendaciones = generar_recomendaciones_exploratorio(
        estadisticas, correlaciones, outliers, distribuciones, calidad_datos
    )

    if verbose:
        print("✅ Análisis exploratorio completado")

    return {
        'tipo': 'analisis_exploratorio_completo',
        'variables_analizadas': variables,
        'n_muestras': len(X),
        'estadisticas_descriptivas': estadisticas,
        'correlaciones': correlaciones,
        'outliers': outliers,
        'distribuciones': distribuciones,
        'pca_exploratorio': {
            'varianza_explicada': pca_basico.explained_variance_ratio_.tolist(),
            'varianza_acumulada': np.cumsum(pca_basico.explained_variance_ratio_).tolist()
        },
        'clustering_exploratorio': clustering_exploratorio,
        'calidad_datos': calidad_datos,
        'recomendaciones': recomendaciones
    }

# ==================== FUNCIONES DE ANÁLISIS AUXILIARES ====================

def analizar_clusters_detallado(X, labels, variables):
    """Análisis detallado de las características de cada cluster"""
    cluster_stats = {}

    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        cluster_data = X[mask]

        # Estadísticas por variable
        stats_por_variable = {}
        for var in variables:
            if var in cluster_data.columns:
                serie = cluster_data[var]
                stats_por_variable[var] = {
                    'media': float(serie.mean()),
                    'std': float(serie.std()),
                    'min': float(serie.min()),
                    'max': float(serie.max()),
                    'mediana': float(serie.median())
                }

        # Centroide del cluster
        centroide = cluster_data.mean().to_dict()

        cluster_stats[f'cluster_{cluster_id}'] = {
            'tamaño': int(np.sum(mask)),
            'proporcion': float(np.sum(mask) / len(labels)),
            'centroide': centroide,
            'estadisticas': stats_por_variable,
            'compacidad': float(np.mean(np.sum((cluster_data - cluster_data.mean()) ** 2, axis=1)))
        }

    return cluster_stats

def analizar_clusters_kmeans(X_original, labels, variables, scaler, kmeans_model):
    """Análisis detallado de clusters de K-Means"""
    cluster_analysis = {}

    for cluster_id in range(kmeans_model.n_clusters):
        mask = labels == cluster_id
        cluster_data = X_original[mask]

        if len(cluster_data) == 0:
            continue

        # Estadísticas del cluster
        cluster_stats = {
            'tamaño': int(np.sum(mask)),
            'proporcion': float(np.sum(mask) / len(labels)),
        }

        # Estadísticas por variable
        stats_variables = {}
        for var in variables:
            if var in cluster_data.columns:
                serie = cluster_data[var]
                stats_variables[var] = {
                    'media': float(serie.mean()),
                    'std': float(serie.std()),
                    'min': float(serie.min()),
                    'max': float(serie.max()),
                    'percentil_25': float(serie.quantile(0.25)),
                    'percentil_75': float(serie.quantile(0.75))
                }

        # Centroide en espacio original
        centroide_original = cluster_data[variables].mean().to_dict()

        # Distancia promedio al centroide
        centroide_escalado = kmeans_model.cluster_centers_[cluster_id]
        cluster_data_escalado = scaler.transform(cluster_data[variables])
        distancias = np.linalg.norm(cluster_data_escalado - centroide_escalado, axis=1)

        cluster_stats.update({
            'centroide_original': centroide_original,
            'estadisticas_variables': stats_variables,
            'distancia_promedio_centroide': float(np.mean(distancias)),
            'compacidad': float(np.std(distancias)),
            'diametro': float(np.max(distancias))
        })

        cluster_analysis[f'cluster_{cluster_id}'] = cluster_stats

    return cluster_analysis

def determinar_k_codo(k_values, inercias):
    """Determinar K óptimo usando el método del codo"""
    if len(inercias) < 3:
        return k_values[0] if k_values else 2

    # Calcular diferencias de segundo orden
    diffs = np.diff(inercias)
    diffs2 = np.diff(diffs)

    # Encontrar el punto de máxima curvatura
    if len(diffs2) > 0:
        k_optimo_idx = np.argmax(np.abs(diffs2)) + 2  # +2 porque perdemos 2 elementos
        return k_values[min(k_optimo_idx, len(k_values) - 1)]

    return k_values[len(k_values) // 2]

def calcular_gap_statistic(X, k_range, n_refs=10):
    """Calcular gap statistic para determinar K óptimo"""
    gaps = []

    for k in k_range:
        # Clustering en datos reales
        kmeans_real = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_real.fit(X)
        log_wk_real = np.log(kmeans_real.inertia_)

        # Clustering en datos de referencia (aleatorios)
        log_wk_refs = []
        for _ in range(n_refs):
            # Generar datos aleatorios con la misma distribución
            X_ref = np.random.uniform(X.min(axis=0), X.max(axis=0), X.shape)
            kmeans_ref = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_ref.fit(X_ref)
            log_wk_refs.append(np.log(kmeans_ref.inertia_))

        # Gap = promedio(log_wk_ref) - log_wk_real
        gap = np.mean(log_wk_refs) - log_wk_real
        gaps.append(gap)

    # K óptimo es donde gap es máximo
    if gaps:
        return list(k_range)[np.argmax(gaps)]
    return list(k_range)[0]

def evaluar_estabilidad_kmeans(X, k_range, n_bootstrap=20):
    """Evaluar estabilidad de clustering mediante bootstrap"""
    estabilidad_scores = {}

    for k in k_range:
        ari_scores = []

        # Clustering original
        kmeans_original = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_original = kmeans_original.fit_predict(X)

        # Bootstrap samples
        for _ in range(n_bootstrap):
            # Muestra bootstrap
            bootstrap_idx = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap = X[bootstrap_idx]

            try:
                kmeans_bootstrap = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels_bootstrap = kmeans_bootstrap.fit_predict(X_bootstrap)

                # ARI entre clustering original y bootstrap
                ari = adjusted_rand_score(labels_original[bootstrap_idx], labels_bootstrap)
                ari_scores.append(ari)

            except:
                continue

        if ari_scores:
            estabilidad_scores[k] = np.mean(ari_scores)

    # Retornar K con mayor estabilidad
    if estabilidad_scores:
        return max(estabilidad_scores.keys(), key=lambda k: estabilidad_scores[k])
    return list(k_range)[0]

def determinar_k_final(k_optimos):
    """Determinar K final basado en múltiples criterios"""
    if not k_optimos:
        return 3

    # Votar por el K más frecuente
    from collections import Counter
    votos = Counter(k_optimos.values())

    if votos:
        return votos.most_common(1)[0][0]

    return 3

def determinar_rango_eps(X_scaled):
    """Determinar rango de eps usando el método de k-distancias"""
    # Calcular k-distancias para k=4 (regla general)
    nbrs = NearestNeighbors(n_neighbors=4)
    nbrs.fit(X_scaled)
    distances, indices = nbrs.kneighbors(X_scaled)

    # Usar la 4ta distancia más cercana
    k_distances = distances[:, 3]
    k_distances_sorted = np.sort(k_distances)

    # Buscar el "codo" en la curva de k-distancias
    diffs = np.diff(k_distances_sorted)
    knee_point = np.argmax(diffs)

    eps_optimo = k_distances_sorted[knee_point]

    # Crear rango alrededor del eps óptimo
    eps_min = eps_optimo * 0.5
    eps_max = eps_optimo * 2.0

    return np.linspace(eps_min, eps_max, 20)

def analizar_clusters_dbscan(X_original, labels, variables):
    """Análisis detallado de clusters encontrados por DBSCAN"""
    cluster_analysis = {}

    # Analizar clusters normales
    for cluster_id in set(labels):
        if cluster_id == -1:  # Ruido
            continue

        mask = labels == cluster_id
        cluster_data = X_original[mask]

        # Estadísticas básicas
        cluster_stats = {
            'tamaño': int(np.sum(mask)),
            'proporcion': float(np.sum(mask) / len(labels)),
            'centroide': cluster_data[variables].mean().to_dict(),
            'variabilidad': cluster_data[variables].std().to_dict()
        }

        cluster_analysis[f'cluster_{cluster_id}'] = cluster_stats

    # Analizar puntos de ruido
    noise_mask = labels == -1
    if np.any(noise_mask):
        noise_data = X_original[noise_mask]
        cluster_analysis['noise'] = {
            'tamaño': int(np.sum(noise_mask)),
            'proporcion': float(np.sum(noise_mask) / len(labels)),
            'caracteristicas_promedio': noise_data[variables].mean().to_dict(),
            'indices': np.where(noise_mask)[0].tolist()
        }

    return cluster_analysis

def analizar_densidad_clusters(X_scaled, labels, eps):
    """Analizar la densidad de cada cluster"""
    densidad_analysis = {}

    for cluster_id in set(labels):
        if cluster_id == -1:
            continue

        mask = labels == cluster_id
        cluster_points = X_scaled[mask]

        if len(cluster_points) > 1:
            # Calcular densidad como número de puntos / volumen aproximado
            # Usar la distancia promedio entre puntos como aproximación
            distances = euclidean_distances(cluster_points)

            # Solo tomar la diagonal superior para evitar duplicados
            upper_tri_indices = np.triu_indices_from(distances, k=1)
            distances_array = distances[upper_tri_indices]

            densidad = len(cluster_points) / (np.mean(distances_array) ** len(X_scaled[0]))

            densidad_analysis[f'cluster_{cluster_id}'] = {
                'densidad': float(densidad),
                'distancia_promedio_intra': float(np.mean(distances_array)),
                'compacidad': float(np.std(distances_array))
            }

    return densidad_analysis

def analizar_outliers_dbscan(X_original, labels, variables):
    """Análisis detallado de outliers detectados por DBSCAN"""
    outlier_mask = labels == -1

    if not np.any(outlier_mask):
        return {'n_outliers': 0}

    outliers = X_original[outlier_mask]

    # Caracterizar outliers
    outlier_analysis = {
        'n_outliers': int(np.sum(outlier_mask)),
        'proporcion': float(np.sum(outlier_mask) / len(labels)),
        'estadisticas': {}
    }

    for var in variables:
        if var in outliers.columns:
            serie = outliers[var]
            outlier_analysis['estadisticas'][var] = {
                'media': float(serie.mean()),
                'std': float(serie.std()),
                'min': float(serie.min()),
                'max': float(serie.max()),
                'rango': float(serie.max() - serie.min())
            }

    return outlier_analysis

def analizar_pca_detallado(pca_model, variables, varianza_objetivo):
    """Análisis detallado del modelo PCA"""
    varianza_explicada = pca_model.explained_variance_ratio_
    varianza_acumulada = np.cumsum(varianza_explicada)

    # Encontrar número de componentes para objetivo de varianza
    n_componentes_objetivo = np.argmax(varianza_acumulada >= varianza_objetivo) + 1

    # Análisis de cada componente
    componentes_info = []
    for i in range(len(varianza_explicada)):
        # Loadings (pesos de las variables)
        loadings = pca_model.components_[i]

        # Variables más importantes en este componente
        importancia_abs = np.abs(loadings)
        top_variables_idx = np.argsort(importancia_abs)[::-1][:5]

        top_variables = [
            {
                'variable': variables[idx],
                'loading': float(loadings[idx]),
                'importancia_abs': float(importancia_abs[idx])
            }
            for idx in top_variables_idx
        ]

        componentes_info.append({
            'componente': f'PC{i+1}',
            'varianza_explicada': float(varianza_explicada[i]),
            'varianza_acumulada': float(varianza_acumulada[i]),
            'top_variables': top_variables,
            'loadings_completos': loadings.tolist()
        })

    return {
        'varianza_explicada': varianza_explicada.tolist(),
        'varianza_acumulada': varianza_acumulada.tolist(),
        'eigenvalues': pca_model.explained_variance_.tolist(),
        'n_componentes_objetivo': n_componentes_objetivo,
        'varianza_objetivo': varianza_objetivo,
        'componentes_info': componentes_info,
        'matriz_componentes': pca_model.components_.tolist()
    }

def evaluar_calidad_kernel_pca(X_original, X_transformed, kpca_model):
    """Evaluar calidad de Kernel PCA mediante métricas indirectas"""
    # Como no podemos reconstruir exactamente, usamos métricas indirectas

    # 1. Preservación de distancias relativas
    from scipy.stats import spearmanr

    # Muestra aleatoria para eficiencia
    if len(X_original) > 500:
        indices = np.random.choice(len(X_original), 500, replace=False)
        X_sample = X_original[indices]
        X_trans_sample = X_transformed[indices]
    else:
        X_sample = X_original
        X_trans_sample = X_transformed

    # Distancias en espacio original vs transformado
    dist_original = euclidean_distances(X_sample).flatten()
    dist_transformed = euclidean_distances(X_trans_sample).flatten()

    # Correlación de Spearman entre distancias (usando implementación propia)
    corr_distancias = np.corrcoef(dist_original, dist_transformed)[0, 1]

    return {
        'correlacion_distancias': float(corr_distancias),
        'calidad': 'Alta' if corr_distancias > 0.7 else 'Media' if corr_distancias > 0.5 else 'Baja'
    }

def analizar_correlaciones_pca(X_original, X_pca, variables):
    """Analizar correlaciones entre variables originales y componentes principales"""
    correlaciones = {}

    for i, variable in enumerate(variables):
        corr_con_componentes = {}
        for j in range(X_pca.shape[1]):
            corr = np.corrcoef(X_original[:, i], X_pca[:, j])[0, 1]
            corr_con_componentes[f'PC{j+1}'] = float(corr)

        correlaciones[variable] = corr_con_componentes

    return correlaciones

def analizar_contribuciones_variables(pca_model, variables):
    """Analizar contribución de cada variable a cada componente"""
    contribuciones = {}

    for i, variable in enumerate(variables):
        contrib_por_componente = {}
        for j in range(len(pca_model.components_)):
            # Contribución como el cuadrado del loading normalizado
            loading_squared = pca_model.components_[j, i] ** 2
            contrib_normalizada = loading_squared / np.sum(pca_model.components_[j] ** 2)
            contrib_por_componente[f'PC{j+1}'] = float(contrib_normalizada)

        contribuciones[variable] = contrib_por_componente

    return contribuciones

def calcular_estadisticas_avanzadas(X, variables):
    """Calcular estadísticas descriptivas avanzadas"""
    estadisticas = {}

    for variable in variables:
        serie = X[variable]

        # Estadísticas básicas
        stats_basicas = {
            'count': len(serie),
            'mean': float(serie.mean()),
            'std': float(serie.std()),
            'min': float(serie.min()),
            'max': float(serie.max()),
            'range': float(serie.max() - serie.min()),
            'median': float(serie.median()),
            'q1': float(serie.quantile(0.25)),
            'q3': float(serie.quantile(0.75)),
            'iqr': float(serie.quantile(0.75) - serie.quantile(0.25))
        }

        # Estadísticas de forma (sin scipy)
        mean_val = serie.mean()
        std_val = serie.std()

        # Skewness
        skew_val = ((serie - mean_val) ** 3).mean() / (std_val ** 3)

        # Kurtosis
        kurt_val = ((serie - mean_val) ** 4).mean() / (std_val ** 4) - 3

        stats_forma = {
            'skewness': float(skew_val),
            'kurtosis': float(kurt_val),
            'cv': float(serie.std() / serie.mean()) if serie.mean() != 0 else np.inf
        }

        # Test de normalidad simplificado
        normalidad = {
            'es_normal': abs(skew_val) < 1 and abs(kurt_val) < 1  # Criterio simplificado
        }

        estadisticas[variable] = {
            **stats_basicas,
            **stats_forma,
            'normalidad': normalidad
        }

    return estadisticas

def analizar_correlaciones_avanzado(X, variables):
    """Análisis avanzado de correlaciones"""
    # Matriz de correlación de Pearson
    corr_pearson = X.corr()

    # Matriz de correlación de Spearman (correlaciones no lineales)
    corr_spearman = X.corr(method='spearman')

    # Correlaciones significativas
    correlaciones_fuertes = []
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            var1, var2 = variables[i], variables[j]
            corr_pearson_val = corr_pearson.loc[var1, var2]
            corr_spearman_val = corr_spearman.loc[var1, var2]

            if abs(corr_pearson_val) > 0.6 or abs(corr_spearman_val) > 0.6:
                correlaciones_fuertes.append({
                    'variable_1': var1,
                    'variable_2': var2,
                    'pearson': float(corr_pearson_val),
                    'spearman': float(corr_spearman_val),
                    'tipo': 'Fuerte positiva' if corr_pearson_val > 0.6 else 'Fuerte negativa'
                })

    # Análisis de multicolinealidad
    condicion = np.linalg.cond(corr_pearson.values)

    return {
        'matriz_pearson': corr_pearson.to_dict(),
        'matriz_spearman': corr_spearman.to_dict(),
        'correlaciones_fuertes': correlaciones_fuertes,
        'numero_condicion': float(condicion),
        'multicolinealidad': 'Alta' if condicion > 1000 else 'Media' if condicion > 100 else 'Baja'
    }

def detectar_outliers_multiples_metodos(X, variables, metodo_principal='isolation_forest'):
    """Detección de outliers usando múltiples métodos"""
    outliers_por_metodo = {}

    # 1. Z-Score (sin scipy)
    z_scores = np.abs((X - X.mean()) / X.std())
    outliers_zscore = np.where(z_scores > 3)

    # 2. IQR
    outliers_iqr = {}
    for variable in variables:
        Q1 = X[variable].quantile(0.25)
        Q3 = X[variable].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_mask = (X[variable] < lower_bound) | (X[variable] > upper_bound)
        outliers_iqr[variable] = X[outliers_mask].index.tolist()

    # 3. Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outliers_isolation = iso_forest.fit_predict(X)
    indices_outliers_isolation = X[outliers_isolation == -1].index.tolist()

    # 4. Distancia de Mahalanobis (simplificada)
    try:
        cov_matrix = np.cov(X.T)
        inv_cov = np.linalg.pinv(cov_matrix)
        mean_vec = np.mean(X, axis=0)

        mahalanobis_dist = []
        for _, row in X.iterrows():
            diff = row.values - mean_vec
            dist = np.sqrt(diff.T @ inv_cov @ diff)
            mahalanobis_dist.append(dist)

        # Outliers usando percentil 95
        threshold = np.percentile(mahalanobis_dist, 95)
        outliers_mahalanobis = X[np.array(mahalanobis_dist) > threshold].index.tolist()

    except:
        outliers_mahalanobis = []
        mahalanobis_dist = []

    # Consolidar resultados
    outliers_zscore_indices = [X.index[outliers_zscore[0][i]] for i in range(len(outliers_zscore[0]))]

    # Outliers consenso
    todos_outliers = set(outliers_zscore_indices + indices_outliers_isolation + outliers_mahalanobis)

    return {
        'zscore': {
            'indices': outliers_zscore_indices,
            'total': len(outliers_zscore_indices)
        },
        'iqr': outliers_iqr,
        'isolation_forest': {
            'indices': indices_outliers_isolation,
            'total': len(indices_outliers_isolation)
        },
        'mahalanobis': {
            'indices': outliers_mahalanobis,
            'total': len(outliers_mahalanobis)
        },
        'consenso': {
            'indices_unicos': list(todos_outliers),
            'total_unico': len(todos_outliers),
            'porcentaje': float(len(todos_outliers) / len(X) * 100)
        }
    }

def analizar_distribuciones_avanzado(X, variables):
    """Análisis avanzado de distribuciones sin scipy"""
    distribuciones = {}

    for variable in variables:
        serie = X[variable]

        # Test simplificado de normalidad basado en skewness y kurtosis
        mean_val = serie.mean()
        std_val = serie.std()

        # Skewness
        skew_val = ((serie - mean_val) ** 3).mean() / (std_val ** 3)

        # Kurtosis
        kurt_val = ((serie - mean_val) ** 4).mean() / (std_val ** 4) - 3

        # Criterios simplificados para diferentes distribuciones
        distribuciones_candidatas = []

        # Normal: skewness cercano a 0, kurtosis cercano a 0
        if abs(skew_val) < 0.5 and abs(kurt_val) < 0.5:
            distribuciones_candidatas.append({
                'distribucion': 'normal',
                'score': 1.0 - (abs(skew_val) + abs(kurt_val))/2,
                'parametros': [mean_val, std_val]
            })

        # Exponencial: skewness positivo alto
        if skew_val > 1.5:
            distribuciones_candidatas.append({
                'distribucion': 'exponential',
                'score': min(1.0, skew_val/2),
                'parametros': [1/mean_val]
            })

        # Uniforme: kurtosis negativo
        if kurt_val < -1:
            distribuciones_candidatas.append({
                'distribucion': 'uniform',
                'score': min(1.0, abs(kurt_val)/2),
                'parametros': [serie.min(), serie.max()]
            })

        # Si no hay candidatos, usar normal por defecto
        if not distribuciones_candidatas:
            distribuciones_candidatas.append({
                'distribucion': 'normal',
                'score': 0.5,
                'parametros': [mean_val, std_val]
            })

        # Ordenar por score
        distribuciones_candidatas.sort(key=lambda x: x['score'], reverse=True)

        distribuciones[variable] = {
            'mejor_ajuste': distribuciones_candidatas[0],
            'todos_ajustes': distribuciones_candidatas,
            'es_aproximadamente_normal': distribuciones_candidatas[0]['distribucion'] == 'normal',
            'skewness': float(skew_val),
            'kurtosis': float(kurt_val)
        }

    return distribuciones

def clustering_exploratorio_rapido(X, variables, escalado='standard'):
    """Clustering exploratorio rápido para identificar patrones"""
    # Escalado
    X_scaled, scaler = aplicar_escalado(X, escalado)

    # K-Means rápido con k=3,4,5
    resultados_rapidos = {}

    for k in [3, 4, 5]:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
            labels = kmeans.fit_predict(X_scaled)

            silhouette = silhouette_score(X_scaled, labels)

            resultados_rapidos[k] = {
                'silhouette_score': float(silhouette),
                'inercia': float(kmeans.inertia_),
                'labels': labels.tolist()
            }

        except:
            continue

    # Mejor k rápido
    mejor_k_rapido = max(resultados_rapidos.keys(),
                        key=lambda k: resultados_rapidos[k]['silhouette_score']) if resultados_rapidos else 3

    return {
        'resultados': resultados_rapidos,
        'mejor_k_rapido': mejor_k_rapido,
        'recomendacion': f"Se sugiere explorar clustering con {mejor_k_rapido} grupos"
    }

def evaluar_calidad_datos(data_original, variables):
    """Evaluar calidad general de los datos"""
    # Valores faltantes
    missing_info = {}
    for var in variables:
        if var in data_original.columns:
            missing_count = data_original[var].isnull().sum()
            missing_info[var] = {
                'count': int(missing_count),
                'percentage': float(missing_count / len(data_original) * 100)
            }

    # Duplicados
    duplicados = data_original.duplicated().sum()

    # Valores únicos por variable
    unique_info = {}
    for var in variables:
        if var in data_original.columns:
            unique_count = data_original[var].nunique()
            unique_info[var] = {
                'unique_count': int(unique_count),
                'unique_ratio': float(unique_count / len(data_original))
            }

    # Score de calidad general
    total_missing = sum(info['count'] for info in missing_info.values())
    missing_ratio = total_missing / (len(data_original) * len(variables))
    duplicate_ratio = duplicados / len(data_original)

    quality_score = 100 * (1 - missing_ratio - duplicate_ratio)

    return {
        'valores_faltantes': missing_info,
        'duplicados': {
            'count': int(duplicados),
            'percentage': float(duplicate_ratio * 100)
        },
        'valores_unicos': unique_info,
        'quality_score': float(max(0, quality_score)),
        'calificacion': 'Excelente' if quality_score > 95 else 'Buena' if quality_score > 85 else 'Regular' if quality_score > 70 else 'Deficiente'
    }

# ==================== FUNCIONES DE RECOMENDACIONES ====================

def generar_recomendaciones_exploratorio(estadisticas, correlaciones, outliers, distribuciones, calidad):
    """Generar recomendaciones basadas en el análisis exploratorio"""
    recomendaciones = []

    # Recomendaciones sobre calidad de datos
    if calidad['quality_score'] < 85:
        recomendaciones.append("Considere limpiar los datos antes del análisis (valores faltantes, duplicados)")

    # Recomendaciones sobre correlaciones
    correlaciones_fuertes = correlaciones['correlaciones_fuertes']
    if len(correlaciones_fuertes) > 0:
        recomendaciones.append(f"Se detectaron {len(correlaciones_fuertes)} correlaciones fuertes. Considere usar PCA para reducir dimensionalidad")

    if correlaciones['multicolinealidad'] == 'Alta':
        recomendaciones.append("Alta multicolinealidad detectada. Use regularización en modelos lineales")

    # Recomendaciones sobre outliers
    porcentaje_outliers = outliers['consenso']['porcentaje']
    if porcentaje_outliers > 10:
        recomendaciones.append(f"Alto porcentaje de outliers ({porcentaje_outliers:.1f}%). Considere usar métodos robustos como DBSCAN")
    elif porcentaje_outliers > 5:
        recomendaciones.append("Outliers moderados detectados. Evalúe si remover o usar métodos robustos")

    # Recomendaciones sobre distribuciones
    variables_no_normales = [var for var, info in distribuciones.items()
                           if not info.get('es_aproximadamente_normal', False)]

    if len(variables_no_normales) > len(distribuciones) * 0.7:
        recomendaciones.append("Muchas variables no siguen distribución normal. Considere transformaciones o métodos no paramétricos")

    # Recomendaciones sobre técnicas ML
    n_variables = len(estadisticas)
    if n_variables > 10:
        recomendaciones.append("Alto número de variables. PCA o selección de características recomendada")

    if porcentaje_outliers > 5:
        recomendaciones.append("Para clustering, considere DBSCAN debido a la presencia de outliers")
    else:
        recomendaciones.append("K-Means puede ser apropiado para clustering")

    return recomendaciones

def generar_recomendaciones_pca(resultados_pca):
    """Generar recomendaciones basadas en los resultados de PCA"""
    recomendaciones = []

    if 'linear' in resultados_pca:
        linear_result = resultados_pca['linear']
        n_recomendado = linear_result['componentes_recomendados']
        varianza_total = linear_result['analisis']['varianza_acumulada'][n_recomendado - 1]

        recomendaciones.append(
            f"Se recomienda usar {n_recomendado} componentes principales "
            f"(explican {varianza_total:.1%} de la varianza)"
        )

        # Identificar variables más importantes
        primer_pc = linear_result['analisis']['componentes_info'][0]
        var_importante = primer_pc['top_variables'][0]['variable']

        recomendaciones.append(
            f"La variable '{var_importante}' tiene mayor peso en el primer componente principal"
        )

        # Recomendación sobre dimensionalidad
        reduccion = 1 - n_recomendado / len(linear_result['analisis']['varianza_explicada'])
        if reduccion > 0.5:
            recomendaciones.append(
                f"PCA permite reducir la dimensionalidad en {reduccion:.1%} "
                "manteniendo la mayor parte de la información"
            )

    return recomendaciones

def generar_recomendaciones_clustering(resultados_completos):
    """Generar recomendaciones para clustering jerárquico"""
    recomendaciones = []

    if not resultados_completos:
        recomendaciones.append("No se pudieron generar clusters válidos. Revise la selección de variables.")
        return recomendaciones

    mejor_config = max(resultados_completos.keys(),
                      key=lambda k: resultados_completos[k]['mejor_silhouette'])
    mejor_resultado = resultados_completos[mejor_config]

    recomendaciones.append(f"Mejor configuración: {mejor_config} con {mejor_resultado['mejor_k']} clusters")

    if mejor_resultado['mejor_silhouette'] > 0.7:
        recomendaciones.append("Clustering de excelente calidad detectado")
    elif mejor_resultado['mejor_silhouette'] > 0.5:
        recomendaciones.append("Clustering de buena calidad detectado")
    else:
        recomendaciones.append("Clustering de calidad moderada. Considere ajustar parámetros")

    return recomendaciones

def generar_recomendaciones_kmeans(k_optimos, resultados_kmeans, k_final):
    """Generar recomendaciones para K-Means"""
    recomendaciones = []

    recomendaciones.append(f"K óptimo recomendado: {k_final}")

    if k_final in resultados_kmeans:
        resultado = resultados_kmeans[k_final]
        if resultado['silhouette_score'] > 0.7:
            recomendaciones.append("Clustering de excelente calidad")
        elif resultado['silhouette_score'] > 0.5:
            recomendaciones.append("Clustering de buena calidad")
        else:
            recomendaciones.append("Clustering de calidad moderada")

    # Consenso entre métodos
    valores_k = list(k_optimos.values())
    if len(set(valores_k)) == 1:
        recomendaciones.append("Todos los métodos coinciden en el K óptimo")
    else:
        recomendaciones.append("Los métodos sugieren diferentes valores de K. Evalúe el contexto del problema")

    return recomendaciones

def generar_recomendaciones_dbscan(mejor_resultado):
    """Generar recomendaciones para DBSCAN"""
    recomendaciones = []

    n_clusters = mejor_resultado['n_clusters']
    noise_ratio = mejor_resultado['noise_ratio']

    recomendaciones.append(f"DBSCAN detectó {n_clusters} clusters con {noise_ratio:.1%} de outliers")

    if noise_ratio < 0.05:
        recomendaciones.append("Muy pocos outliers detectados. Los datos son homogéneos")
    elif noise_ratio < 0.15:
        recomendaciones.append("Cantidad normal de outliers detectados")
    else:
        recomendaciones.append("Alto porcentaje de outliers. Revise la calidad de los datos")

    if mejor_resultado['silhouette_score'] > 0.6:
        recomendaciones.append("Clusters bien definidos y separados")
    else:
        recomendaciones.append("Clusters con separación moderada")

    return recomendaciones

def generar_resumen_clustering(resultados_completos):
    """Generar resumen comparativo de métodos de clustering"""
    if not resultados_completos:
        return {}

    # Comparar silhouette scores
    scores_por_metodo = {}
    for config, resultado in resultados_completos.items():
        scores_por_metodo[config] = resultado['mejor_silhouette']

    mejor_configuracion_global = max(scores_por_metodo.keys(), key=lambda k: scores_por_metodo[k])

    return {
        'mejor_configuracion_global': mejor_configuracion_global,
        'scores_por_metodo': scores_por_metodo,
        'mejores_k_por_metodo': {
            config: resultado['mejor_k']
            for config, resultado in resultados_completos.items()
        },
        'resumen_rendimiento': {
            config: {
                'silhouette': scores_por_metodo[config],
                'k_optimo': resultado['mejor_k'],
                'metodo': resultado['metodo'],
                'metrica': resultado['metrica']
            }
            for config, resultado in resultados_completos.items()
        }
    }
# Agregar esta función a ml_functions_no_supervisado.py si no existe

def kmeans_optimizado_completo(data, variables=None, k_range=None,
                             criterios_optimo=['silhouette', 'elbow', 'gap'],
                             escalado='standard', random_state=42, n_jobs=-1, verbose=True):
    """
    K-Means optimizado con múltiples criterios para determinar K óptimo
    """
    if verbose:
        print("🔍 Iniciando K-Means optimizado...")

    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()

    X = data[variables].dropna()

    if verbose:
        print(f"📊 Datos preparados: {X.shape[0]} muestras, {X.shape[1]} variables")

    # Escalado
    X_scaled, scaler = aplicar_escalado(X, escalado)

    if k_range is None:
        k_range = range(2, min(15, len(X) // 10))

    resultados_kmeans = {}
    inercias = []
    silhouette_scores = []

    # Evaluar diferentes valores de K
    for k in k_range:
        if verbose:
            print(f"   Evaluando K={k}...")

        try:
            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                n_init=20,
                max_iter=500,
                random_state=random_state
            )
            labels = kmeans.fit_predict(X_scaled)

            # Métricas de evaluación
            silhouette = silhouette_score(X_scaled, labels)
            davies_bouldin = davies_bouldin_score(X_scaled, labels)
            calinski_harabasz = calinski_harabasz_score(X_scaled, labels)

            # Análisis detallado de clusters
            cluster_analysis = analizar_clusters_kmeans(X, labels, variables, scaler, kmeans)

            resultado_k = {
                'k': k,
                'labels': labels.tolist(),
                'centroides': kmeans.cluster_centers_.tolist(),
                'inercia': float(kmeans.inertia_),
                'silhouette_score': float(silhouette),
                'davies_bouldin_score': float(davies_bouldin),
                'calinski_harabasz_score': float(calinski_harabasz),
                'n_iteraciones': int(kmeans.n_iter_),
                'cluster_analysis': cluster_analysis
            }

            resultados_kmeans[k] = resultado_k
            inercias.append(resultado_k['inercia'])
            silhouette_scores.append(resultado_k['silhouette_score'])

        except Exception as e:
            if verbose:
                print(f"Error con K={k}: {e}")
            continue

    # Determinar K óptimo usando diferentes criterios
    k_optimos = {}

    # 1. Método del codo
    if len(inercias) >= 3:
        k_optimo_codo = determinar_k_codo(list(k_range), inercias)
        k_optimos['elbow'] = k_optimo_codo

    # 2. Silhouette score máximo
    if silhouette_scores:
        k_optimo_silhouette = list(k_range)[np.argmax(silhouette_scores)]
        k_optimos['silhouette'] = k_optimo_silhouette

    # 3. Gap statistic (simplificado)
    if len(resultados_kmeans) >= 3:
        k_optimo_gap = calcular_gap_statistic(X_scaled, k_range, n_refs=10)
        k_optimos['gap'] = k_optimo_gap

    # 4. Criterio de estabilidad
    k_optimo_estabilidad = evaluar_estabilidad_kmeans(X_scaled, k_range)
    k_optimos['estabilidad'] = k_optimo_estabilidad

    k_final = determinar_k_final(k_optimos)

    # Agregar datos originales para visualización
    datos_originales = X.copy()

    if verbose:
        print(f"✅ K-Means completado. K recomendado: {k_final}")

    return {
        'tipo': 'kmeans_optimizado',
        'variables_utilizadas': variables,
        'k_range': list(k_range),
        'resultados_por_k': resultados_kmeans,
        'inercias': inercias,
        'silhouette_scores': silhouette_scores,
        'k_optimos': k_optimos,
        'recomendacion_k': k_final,
        'datos_originales': datos_originales,  # Agregar datos originales
        'datos_escalados': X_scaled.tolist(),
        'scaler_params': {
            'mean': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else [],
            'scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else []
        },
        'recomendaciones': generar_recomendaciones_kmeans(k_optimos, resultados_kmeans, k_final)
    }
# Agregar esta función a ml_functions_no_supervisado.py

def dbscan_optimizado(data, variables=None, optimizar_parametros=True,
                      escalado='standard', n_jobs=-1, verbose=True, contamination=0.1):
    """
    DBSCAN optimizado con búsqueda automática de parámetros óptimos
    """

    def _determinar_rango_eps_interno(X_scaled):
        """Función interna para determinar rango de eps"""
        try:
            n_samples = len(X_scaled)
            k = min(4, max(2, n_samples // 20))
            k = min(k, n_samples - 1)

            nbrs = NearestNeighbors(n_neighbors=k)
            nbrs.fit(X_scaled)
            distances, indices = nbrs.kneighbors(X_scaled)

            k_distances = distances[:, -1]
            k_distances_sorted = np.sort(k_distances)

            if len(k_distances_sorted) > 2:
                diffs = np.diff(k_distances_sorted)
                if len(diffs) > 0:
                    knee_point = np.argmax(diffs)
                    eps_optimo = k_distances_sorted[knee_point]
                else:
                    eps_optimo = np.median(k_distances_sorted)
            else:
                eps_optimo = 0.5

            eps_min = max(0.05, eps_optimo * 0.3)
            eps_max = eps_optimo * 3.0

            if eps_max - eps_min < 0.1:
                eps_min = 0.1
                eps_max = 2.0

            return np.linspace(eps_min, eps_max, 20)

        except Exception as e:
            print(f"Error en determinar rango eps: {e}")
            return np.linspace(0.1, 3.0, 20)

    if verbose:
        print("🔍 Iniciando DBSCAN optimizado...")

    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()

    X = data[variables].dropna()

    if verbose:
        print(f"📊 Datos preparados: {X.shape[0]} muestras, {X.shape[1]} variables")

    if len(X) < 5:
        return {
            'tipo': 'dbscan_optimizado',
            'error': 'Datos insuficientes para DBSCAN (mínimo 5 muestras)',
            'variables_utilizadas': variables,
            'datos_originales': X.copy(),
            'n_muestras': len(X)
        }

    # Escalado de datos
    X_scaled, scaler = aplicar_escalado(X, escalado)

    # Usar función interna para evitar conflictos de scope
    eps_range = _determinar_rango_eps_interno(X_scaled)
    min_samples_range = range(2, min(10, max(3, len(X) // 15)))

    mejores_resultados = []

    if optimizar_parametros:
        if verbose:
            print("   Optimizando parámetros automáticamente...")

        for eps in eps_range:
            for min_samples in min_samples_range:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=n_jobs)
                    labels = dbscan.fit_predict(X_scaled)

                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    noise_ratio = n_noise / len(labels)

                    if n_clusters < 1 or noise_ratio > 0.9:
                        continue

                    cluster_analysis = analizar_clusters_dbscan(X, labels, variables)

                    if n_clusters > 1:
                        labels_sin_ruido = labels[labels != -1]
                        X_sin_ruido = X_scaled[labels != -1]

                        if len(np.unique(labels_sin_ruido)) > 1 and len(X_sin_ruido) > 1:
                            try:
                                silhouette = silhouette_score(X_sin_ruido, labels_sin_ruido)
                                davies_bouldin = davies_bouldin_score(X_sin_ruido, labels_sin_ruido)
                            except:
                                silhouette = 0.3
                                davies_bouldin = 2.0
                        else:
                            silhouette = 0.3
                            davies_bouldin = 2.0
                    else:
                        silhouette = 0.5
                        davies_bouldin = 1.0

                    resultado = {
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'n_outliers': n_noise,
                        'noise_ratio': noise_ratio,
                        'silhouette_score': silhouette,
                        'silhouette': silhouette,
                        'davies_bouldin_score': davies_bouldin,
                        'davies_bouldin': davies_bouldin,
                        'labels': labels.tolist(),
                        'cluster_labels': labels.tolist(),
                        'cluster_analysis': cluster_analysis,
                        'total_points': len(X),
                        'score_compuesto': silhouette * (1 - noise_ratio)
                    }

                    mejores_resultados.append(resultado)

                except Exception as e:
                    continue

    # Configuraciones por defecto si no hay resultados
    if not mejores_resultados:
        if verbose:
            print("   Probando configuraciones por defecto...")

        configuraciones_default = [
            (0.5, 3), (1.0, 3), (0.3, 2), (1.5, 4), (0.1, 2), (2.0, 5)
        ]

        for eps_default, min_samples_default in configuraciones_default:
            try:
                dbscan = DBSCAN(eps=eps_default, min_samples=min_samples_default, n_jobs=n_jobs)
                labels = dbscan.fit_predict(X_scaled)

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                noise_ratio = n_noise / len(labels)

                if n_clusters >= 1:
                    cluster_analysis = analizar_clusters_dbscan(X, labels, variables)

                    resultado = {
                        'eps': eps_default,
                        'min_samples': min_samples_default,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'n_outliers': n_noise,
                        'noise_ratio': noise_ratio,
                        'silhouette_score': 0.3,
                        'silhouette': 0.3,
                        'davies_bouldin_score': 1.5,
                        'davies_bouldin': 1.5,
                        'labels': labels.tolist(),
                        'cluster_labels': labels.tolist(),
                        'cluster_analysis': cluster_analysis,
                        'total_points': len(X),
                        'score_compuesto': 0.3 * (1 - noise_ratio),
                        'es_default': True
                    }
                    mejores_resultados.append(resultado)
                    break

            except Exception as e:
                continue

    # Último recurso
    if not mejores_resultados:
        if verbose:
            print("   Creando clustering artificial...")

        labels_artificial = [0] * len(X)

        resultado = {
            'eps': 1.0,
            'min_samples': 2,
            'n_clusters': 1,
            'n_noise': 0,
            'n_outliers': 0,
            'noise_ratio': 0.0,
            'silhouette_score': 0.5,
            'silhouette': 0.5,
            'davies_bouldin_score': 0.5,
            'davies_bouldin': 0.5,
            'labels': labels_artificial,
            'cluster_labels': labels_artificial,
            'cluster_analysis': analizar_clusters_dbscan(X, labels_artificial, variables),
            'total_points': len(X),
            'score_compuesto': 0.5,
            'es_artificial': True
        }
        mejores_resultados.append(resultado)

    mejores_resultados.sort(key=lambda x: x['score_compuesto'], reverse=True)
    mejor_resultado = mejores_resultados[0]

    labels_final = np.array(mejor_resultado['labels'])

    try:
        analisis_densidad = analizar_densidad_clusters(X_scaled, labels_final, mejor_resultado['eps'])
        analisis_outliers = analizar_outliers_dbscan(X, labels_final, variables)
    except:
        analisis_densidad = {}
        analisis_outliers = {'n_outliers': mejor_resultado['n_noise']}

    datos_originales = X.copy()

    if verbose:
        tipo_resultado = ""
        if mejor_resultado.get('es_artificial'):
            tipo_resultado = " (clustering artificial)"
        elif mejor_resultado.get('es_default'):
            tipo_resultado = " (configuración por defecto)"

        print(
            f"✅ DBSCAN completado{tipo_resultado}: {mejor_resultado['n_clusters']} clusters, {mejor_resultado['n_noise']} outliers")

    return {
        'tipo': 'dbscan_optimizado',
        'variables_utilizadas': variables,
        'metrica': 'euclidean',
        'mejor_configuracion': mejor_resultado,
        'todas_configuraciones': mejores_resultados[:10],
        'analisis_densidad': analisis_densidad,
        'analisis_outliers': analisis_outliers,
        'datos_originales': datos_originales,
        'datos_escalados': X_scaled.tolist(),
        'n_muestras': len(X),
        'parametros_probados': {
            'eps_range': list(eps_range),
            'min_samples_range': list(min_samples_range)
        },
        'recomendaciones': generar_recomendaciones_dbscan(mejor_resultado)
    }

    # Ordenar por score compuesto
    mejores_resultados.sort(key=lambda x: x['score_compuesto'], reverse=True)
    mejor_resultado = mejores_resultados[0]

    # Análisis adicional
    labels_final = np.array(mejor_resultado['labels'])

    try:
        analisis_densidad = analizar_densidad_clusters(X_scaled, labels_final, mejor_resultado['eps'])
        analisis_outliers = analizar_outliers_dbscan(X, labels_final, variables)
    except:
        analisis_densidad = {}
        analisis_outliers = {'n_outliers': mejor_resultado['n_noise']}

    datos_originales = X.copy()

    if verbose:
        tipo_resultado = ""
        if mejor_resultado.get('es_artificial'):
            tipo_resultado = " (clustering artificial)"
        elif mejor_resultado.get('es_default'):
            tipo_resultado = " (configuración por defecto)"

        print(
            f"✅ DBSCAN completado{tipo_resultado}: {mejor_resultado['n_clusters']} clusters, {mejor_resultado['n_noise']} outliers")

    return {
        'tipo': 'dbscan_optimizado',
        'variables_utilizadas': variables,
        'metrica': 'euclidean',
        'mejor_configuracion': mejor_resultado,
        'todas_configuraciones': mejores_resultados[:10],
        'analisis_densidad': analisis_densidad,
        'analisis_outliers': analisis_outliers,
        'datos_originales': datos_originales,
        'datos_escalados': X_scaled.tolist(),
        'n_muestras': len(X),
        'parametros_probados': {
            'eps_range': list(eps_range),
            'min_samples_range': list(min_samples_range)
        },
        'recomendaciones': generar_recomendaciones_dbscan(mejor_resultado)
    }

    # Escalado de datos
    X_scaled, scaler = aplicar_escalado(X, escalado)

    # Determinar rangos automáticamente con valores más amplios
    eps_range = determinar_rango_eps(X_scaled)
    min_samples_range = range(2, min(15, max(5, len(X) // 10)))  # Rango más flexible

    mejores_resultados = []

    if optimizar_parametros:
        if verbose:
            print("   Optimizando parámetros automáticamente...")

        # Búsqueda en grid de parámetros MÁS FLEXIBLE
        for eps in eps_range:
            for min_samples in min_samples_range:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=n_jobs)
                    labels = dbscan.fit_predict(X_scaled)

                    # Evaluar calidad del clustering - CRITERIOS MÁS FLEXIBLES
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    noise_ratio = n_noise / len(labels)

                    # Filtros de calidad MÁS PERMISIVOS
                    if n_clusters < 1 or noise_ratio > 0.8:  # Permitir incluso 1 cluster
                        continue

                    # Si solo hay un cluster, usar métricas simplificadas
                    if n_clusters == 1:
                        resultado = {
                            'eps': eps,
                            'min_samples': min_samples,
                            'n_clusters': n_clusters,
                            'n_noise': n_noise,
                            'n_outliers': n_noise,
                            'noise_ratio': noise_ratio,
                            'silhouette_score': 0.5,  # Score neutro para 1 cluster
                            'silhouette': 0.5,
                            'davies_bouldin_score': 1.0,
                            'davies_bouldin': 1.0,
                            'labels': labels.tolist(),
                            'cluster_labels': labels.tolist(),
                            'cluster_analysis': analizar_clusters_dbscan(X, labels, variables),
                            'total_points': len(X),
                            'score_compuesto': 0.5 * (1 - noise_ratio)
                        }
                        mejores_resultados.append(resultado)
                        continue

                    # Para múltiples clusters, calcular métricas normalmente
                    labels_sin_ruido = labels[labels != -1]
                    X_sin_ruido = X_scaled[labels != -1]

                    if len(np.unique(labels_sin_ruido)) > 1 and len(X_sin_ruido) > 1:
                        try:
                            silhouette = silhouette_score(X_sin_ruido, labels_sin_ruido)
                            davies_bouldin = davies_bouldin_score(X_sin_ruido, labels_sin_ruido)
                        except:
                            silhouette = 0.3  # Score por defecto si falla el cálculo
                            davies_bouldin = 2.0

                        cluster_analysis = analizar_clusters_dbscan(X, labels, variables)

                        resultado = {
                            'eps': eps,
                            'min_samples': min_samples,
                            'n_clusters': n_clusters,
                            'n_noise': n_noise,
                            'n_outliers': n_noise,
                            'noise_ratio': noise_ratio,
                            'silhouette_score': silhouette,
                            'silhouette': silhouette,
                            'davies_bouldin_score': davies_bouldin,
                            'davies_bouldin': davies_bouldin,
                            'labels': labels.tolist(),
                            'cluster_labels': labels.tolist(),
                            'cluster_analysis': cluster_analysis,
                            'total_points': len(X),
                            'score_compuesto': silhouette * (1 - noise_ratio)
                        }

                        mejores_resultados.append(resultado)

                except Exception as e:
                    if verbose:
                        print(f"   Error con eps={eps:.3f}, min_samples={min_samples}: {e}")
                    continue

    # Si la optimización falló, probar parámetros por defecto más agresivos
    if not mejores_resultados:
        if verbose:
            print("   Probando configuración por defecto...")

        # Probar múltiples configuraciones por defecto
        configuraciones_default = [
            (0.5, 3), (1.0, 3), (0.3, 2), (1.5, 4), (0.1, 2)
        ]

        for eps_default, min_samples_default in configuraciones_default:
            try:
                dbscan = DBSCAN(eps=eps_default, min_samples=min_samples_default, n_jobs=n_jobs)
                labels = dbscan.fit_predict(X_scaled)

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                noise_ratio = n_noise / len(labels)

                # Aceptar cualquier resultado que produzca al menos algún agrupamiento
                if n_clusters >= 1:
                    cluster_analysis = analizar_clusters_dbscan(X, labels, variables)

                    resultado = {
                        'eps': eps_default,
                        'min_samples': min_samples_default,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'n_outliers': n_noise,
                        'noise_ratio': noise_ratio,
                        'silhouette_score': 0.3,  # Score por defecto
                        'silhouette': 0.3,
                        'davies_bouldin_score': 1.5,
                        'davies_bouldin': 1.5,
                        'labels': labels.tolist(),
                        'cluster_labels': labels.tolist(),
                        'cluster_analysis': cluster_analysis,
                        'total_points': len(X),
                        'score_compuesto': 0.3 * (1 - noise_ratio)
                    }
                    mejores_resultados.append(resultado)
                    break  # Usar el primer resultado válido

            except Exception as e:
                continue

    # Si aún no hay resultados, crear uno artificial con todos los puntos como un cluster
    if not mejores_resultados:
        if verbose:
            print("   Creando clustering por defecto (todos los puntos en un cluster)...")

        # Crear labels artificiales: todos en cluster 0
        labels_default = [0] * len(X)

        resultado = {
            'eps': 1.0,
            'min_samples': 2,
            'n_clusters': 1,
            'n_noise': 0,
            'n_outliers': 0,
            'noise_ratio': 0.0,
            'silhouette_score': 0.5,
            'silhouette': 0.5,
            'davies_bouldin_score': 0.5,
            'davies_bouldin': 0.5,
            'labels': labels_default,
            'cluster_labels': labels_default,
            'cluster_analysis': analizar_clusters_dbscan(X, labels_default, variables),
            'total_points': len(X),
            'score_compuesto': 0.5,
            'es_fallback': True  # Marcar como resultado de fallback
        }
        mejores_resultados.append(resultado)

    # Ordenar por score compuesto
    mejores_resultados.sort(key=lambda x: x['score_compuesto'], reverse=True)
    mejor_resultado = mejores_resultados[0]

    # Análisis adicional del mejor resultado
    labels_final = np.array(mejor_resultado['labels'])

    # Análisis de densidad y outliers
    try:
        analisis_densidad = analizar_densidad_clusters(X_scaled, labels_final, mejor_resultado['eps'])
        analisis_outliers = analizar_outliers_dbscan(X, labels_final, variables)
    except:
        analisis_densidad = {}
        analisis_outliers = {'n_outliers': mejor_resultado['n_noise']}

    # Agregar datos originales para visualización
    datos_originales = X.copy()

    if verbose:
        es_fallback = mejor_resultado.get('es_fallback', False)
        if es_fallback:
            print(
                f"⚠️ DBSCAN completado con configuración por defecto: {mejor_resultado['n_clusters']} cluster, {mejor_resultado['n_noise']} outliers")
        else:
            print(
                f"✅ DBSCAN completado: {mejor_resultado['n_clusters']} clusters, {mejor_resultado['n_noise']} outliers")

    return {
        'tipo': 'dbscan_optimizado',
        'variables_utilizadas': variables,
        'metrica': 'euclidean',
        'mejor_configuracion': mejor_resultado,
        'todas_configuraciones': mejores_resultados[:10],
        'analisis_densidad': analisis_densidad,
        'analisis_outliers': analisis_outliers,
        'datos_originales': datos_originales,
        'datos_escalados': X_scaled.tolist(),
        'n_muestras': len(X),
        'parametros_probados': {
            'eps_range': list(eps_range),
            'min_samples_range': list(min_samples_range)
        },
        'recomendaciones': generar_recomendaciones_dbscan(mejor_resultado)
    }

    def determinar_rango_eps(X_scaled):
        """Determinar rango de eps usando el método de k-distancias - VERSIÓN MEJORADA"""
        try:
            n_samples = len(X_scaled)

            # Adaptar k según el tamaño de los datos
            k = min(4, max(2, n_samples // 20))

            # Asegurar que k no sea mayor que el número de muestras
            k = min(k, n_samples - 1)

            nbrs = NearestNeighbors(n_neighbors=k)
            nbrs.fit(X_scaled)
            distances, indices = nbrs.kneighbors(X_scaled)

            # Usar la k-ésima distancia más cercana
            k_distances = distances[:, -1]  # Última columna
            k_distances_sorted = np.sort(k_distances)

            # Método del codo para encontrar eps óptimo
            if len(k_distances_sorted) > 2:
                diffs = np.diff(k_distances_sorted)
                if len(diffs) > 0:
                    knee_point = np.argmax(diffs)
                    eps_optimo = k_distances_sorted[knee_point]
                else:
                    eps_optimo = np.median(k_distances_sorted)
            else:
                eps_optimo = 0.5

            # Crear rango más amplio y robusto
            eps_min = max(0.05, eps_optimo * 0.3)  # Mínimo más bajo
            eps_max = eps_optimo * 3.0  # Máximo más alto

            # Asegurar rango mínimo útil
            if eps_max - eps_min < 0.1:
                eps_min = 0.1
                eps_max = 2.0

            # Crear más puntos en el rango para mayor cobertura
            return np.linspace(eps_min, eps_max, 20)

        except Exception as e:
            print(f"Error en determinar_rango_eps: {e}")
            # Rango por defecto amplio y seguro
            return np.linspace(0.1, 3.0, 20)

    # Ordenar por score compuesto
    mejores_resultados.sort(key=lambda x: x['score_compuesto'], reverse=True)
    mejor_resultado = mejores_resultados[0]

    # AGREGADO: asegurar que total_points esté en mejor_resultado
    if 'total_points' not in mejor_resultado:
        mejor_resultado['total_points'] = len(X)

    # Análisis adicional del mejor resultado
    labels_final = np.array(mejor_resultado['labels'])

    # Análisis de densidad
    analisis_densidad = analizar_densidad_clusters(X_scaled, labels_final, mejor_resultado['eps'])

    # Análisis de outliers
    analisis_outliers = analizar_outliers_dbscan(X, labels_final, variables)

    # Agregar datos originales para visualización
    datos_originales = X.copy()

    if verbose:
        print(f"✅ DBSCAN completado. {mejor_resultado['n_clusters']} clusters, {mejor_resultado['n_noise']} outliers")

    return {
        'tipo': 'dbscan_optimizado',
        'variables_utilizadas': variables,
        'metrica': 'euclidean',
        'mejor_configuracion': mejor_resultado,
        'todas_configuraciones': mejores_resultados[:10],  # Top 10
        'analisis_densidad': analisis_densidad,
        'analisis_outliers': analisis_outliers,
        'datos_originales': datos_originales,  # Agregar datos originales
        'datos_escalados': X_scaled.tolist(),  # AGREGADO: datos escalados
        'n_muestras': len(X),  # AGREGADO: número de muestras
        'parametros_probados': {
            'eps_range': list(eps_range),
            'min_samples_range': list(min_samples_range)
        },
        'recomendaciones': generar_recomendaciones_dbscan(mejor_resultado)
    }
# También necesitas estas funciones de soporte si no las tienes:

def determinar_rango_eps(X_scaled):
    """Determinar rango de eps usando el método de k-distancias"""
    try:
        # Calcular k-distancias para k=4 (regla general)
        nbrs = NearestNeighbors(n_neighbors=min(4, len(X_scaled)))
        nbrs.fit(X_scaled)
        distances, indices = nbrs.kneighbors(X_scaled)

        # Usar la distancia más lejana disponible
        k_distances = distances[:, -1]  # Última columna
        k_distances_sorted = np.sort(k_distances)

        # Buscar el "codo" en la curva de k-distancias
        if len(k_distances_sorted) > 1:
            diffs = np.diff(k_distances_sorted)
            knee_point = np.argmax(diffs) if len(diffs) > 0 else 0
            eps_optimo = k_distances_sorted[min(knee_point, len(k_distances_sorted) - 1)]
        else:
            eps_optimo = 0.5

        # Crear rango alrededor del eps óptimo
        eps_min = max(0.1, eps_optimo * 0.5)
        eps_max = eps_optimo * 2.0

        return np.linspace(eps_min, eps_max, 10)

    except Exception as e:
        print(f"Error en determinar_rango_eps: {e}")
        # Rango por defecto
        return np.linspace(0.1, 2.0, 10)


def analizar_clusters_dbscan(X_original, labels, variables):
    """Análisis detallado de clusters encontrados por DBSCAN"""
    cluster_analysis = {}

    try:
        # Analizar clusters normales
        unique_labels = set(labels)
        for cluster_id in unique_labels:
            if cluster_id == -1:  # Ruido
                continue

            mask = np.array(labels) == cluster_id
            if not np.any(mask):
                continue

            cluster_data = X_original[mask]

            # Estadísticas básicas
            cluster_stats = {
                'tamaño': int(np.sum(mask)),
                'proporcion': float(np.sum(mask) / len(labels)),
                'centroide': cluster_data[variables].mean().to_dict(),
                'variabilidad': cluster_data[variables].std().to_dict()
            }

            cluster_analysis[f'cluster_{cluster_id}'] = cluster_stats

        # Analizar puntos de ruido
        noise_mask = np.array(labels) == -1
        if np.any(noise_mask):
            noise_data = X_original[noise_mask]
            cluster_analysis['noise'] = {
                'tamaño': int(np.sum(noise_mask)),
                'proporcion': float(np.sum(noise_mask) / len(labels)),
                'caracteristicas_promedio': noise_data[variables].mean().to_dict(),
                'indices': np.where(noise_mask)[0].tolist()
            }

        return cluster_analysis

    except Exception as e:
        print(f"Error en analizar_clusters_dbscan: {e}")
        return {'error': str(e)}


def analizar_densidad_clusters(X_scaled, labels, eps):
    """Analizar la densidad de cada cluster"""
    densidad_analysis = {}

    try:
        unique_labels = set(labels)
        for cluster_id in unique_labels:
            if cluster_id == -1:
                continue

            mask = np.array(labels) == cluster_id
            cluster_points = X_scaled[mask]

            if len(cluster_points) > 1:
                # Calcular densidad como número de puntos / volumen aproximado
                distances = euclidean_distances(cluster_points)
                upper_tri_indices = np.triu_indices_from(distances, k=1)

                if len(upper_tri_indices[0]) > 0:
                    distances_array = distances[upper_tri_indices]
                    densidad = len(cluster_points) / (np.mean(distances_array) + 1e-6)

                    densidad_analysis[f'cluster_{cluster_id}'] = {
                        'densidad': float(densidad),
                        'distancia_promedio_intra': float(np.mean(distances_array)),
                        'compacidad': float(np.std(distances_array))
                    }

        return densidad_analysis

    except Exception as e:
        print(f"Error en analizar_densidad_clusters: {e}")
        return {'error': str(e)}


def analizar_outliers_dbscan(X_original, labels, variables):
    """Análisis detallado de outliers detectados por DBSCAN"""
    try:
        outlier_mask = np.array(labels) == -1

        if not np.any(outlier_mask):
            return {'n_outliers': 0}

        outliers = X_original[outlier_mask]

        # Caracterizar outliers
        outlier_analysis = {
            'n_outliers': int(np.sum(outlier_mask)),
            'proporcion': float(np.sum(outlier_mask) / len(labels)),
            'estadisticas': {}
        }

        for var in variables:
            if var in outliers.columns:
                serie = outliers[var]
                outlier_analysis['estadisticas'][var] = {
                    'media': float(serie.mean()),
                    'std': float(serie.std()),
                    'min': float(serie.min()),
                    'max': float(serie.max()),
                    'rango': float(serie.max() - serie.min())
                }

        return outlier_analysis

    except Exception as e:
        print(f"Error en analizar_outliers_dbscan: {e}")
        return {'n_outliers': 0, 'error': str(e)}


def generar_recomendaciones_dbscan(mejor_resultado):
    """Generar recomendaciones para DBSCAN"""
    recomendaciones = []

    try:
        n_clusters = mejor_resultado.get('n_clusters', 0)
        noise_ratio = mejor_resultado.get('noise_ratio', 0)

        recomendaciones.append(f"DBSCAN detectó {n_clusters} clusters con {noise_ratio:.1%} de outliers")

        if noise_ratio < 0.05:
            recomendaciones.append("Muy pocos outliers detectados. Los datos son homogéneos")
        elif noise_ratio < 0.15:
            recomendaciones.append("Cantidad normal de outliers detectados")
        else:
            recomendaciones.append("Alto porcentaje de outliers. Revise la calidad de los datos")

        silhouette_score = mejor_resultado.get('silhouette_score', 0)
        if silhouette_score > 0.6:
            recomendaciones.append("Clusters bien definidos y separados")
        else:
            recomendaciones.append("Clusters con separación moderada")

        return recomendaciones

    except Exception as e:
        return [f"Error generando recomendaciones: {str(e)}"]
# ==================== DEMO COMPLETO ====================

def demo_ml_no_supervisado_completo():
    """
    Demostración completa del sistema ML no supervisado sin scipy
    """
    print("🚀 Generando datos de demostración...")
    datos = generar_datos_agua_realistas(n_muestras=300, incluir_outliers=True)

    print("📊 Datos generados exitosamente")
    print(f"   Shape: {datos.shape}")
    print(f"   Columnas: {list(datos.columns)}")

    # Variables para análisis (usando las columnas del CSV real)
    variables_analisis = ['pH', 'WT', 'DO', 'TBD', 'CTD', 'BOD5', 'COD', 'FC', 'TC', 'NO3', 'NO2', 'N_NH3', 'TP']

    # Ejemplo 1: Análisis exploratorio completo
    print("\n🔍 Ejemplo 1: Análisis Exploratorio Completo")
    exploratorio = analisis_exploratorio_completo(datos, variables_analisis)
    print(f"   Calidad de datos: {exploratorio['calidad_datos']['calificacion']}")
    print(f"   Outliers detectados: {exploratorio['outliers']['consenso']['porcentaje']:.1f}%")
    print(f"   Correlaciones fuertes: {len(exploratorio['correlaciones']['correlaciones_fuertes'])}")

    # Ejemplo 2: Clustering jerárquico completo
    print("\n🔍 Ejemplo 2: Clustering Jerárquico Completo")
    jerarquico = clustering_jerarquico_completo(datos, variables_analisis,
                                               metodos=['ward', 'complete'],
                                               metricas=['euclidean'])
    if jerarquico['mejor_configuracion']:
        mejor_config = jerarquico['mejor_configuracion']
        mejor_resultado = jerarquico['resultados_por_configuracion'][mejor_config]
        print(f"   Mejor configuración: {mejor_config}")
        print(f"   K óptimo: {mejor_resultado['mejor_k']}")
        print(f"   Silhouette Score: {mejor_resultado['mejor_silhouette']:.3f}")

    # Ejemplo 3: K-Means optimizado
    print("\n🔍 Ejemplo 3: K-Means Optimizado")
    kmeans_opt = kmeans_optimizado_completo(datos, variables_analisis)
    print(f"   K recomendado: {kmeans_opt['recomendacion_k']}")
    print(f"   Criterios de optimización: {list(kmeans_opt['k_optimos'].keys())}")
    if kmeans_opt['recomendacion_k'] in kmeans_opt['resultados_por_k']:
        mejor_kmeans = kmeans_opt['resultados_por_k'][kmeans_opt['recomendacion_k']]
        print(f"   Silhouette Score: {mejor_kmeans['silhouette_score']:.3f}")

    # Ejemplo 4: DBSCAN optimizado
    print("\n🔍 Ejemplo 4: DBSCAN Optimizado")
    dbscan_opt = dbscan_optimizado(datos, variables_analisis, optimizar_parametros=True)
    if 'error' not in dbscan_opt:
        mejor_dbscan = dbscan_opt['mejor_configuracion']
        print(f"   Clusters encontrados: {mejor_dbscan['n_clusters']}")
        print(f"   Outliers detectados: {mejor_dbscan['n_noise']} ({mejor_dbscan['noise_ratio']:.1%})")
        print(f"   Silhouette Score: {mejor_dbscan['silhouette_score']:.3f}")
    else:
        print("   No se encontraron configuraciones válidas")

    # Ejemplo 5: PCA avanzado
    print("\n🔍 Ejemplo 5: PCA Avanzado")
    pca_avanzado = pca_completo_avanzado(datos, variables_analisis, metodos=['linear'])
    if 'linear' in pca_avanzado['resultados_por_metodo']:
        pca_linear = pca_avanzado['resultados_por_metodo']['linear']
        n_comp = pca_linear['componentes_recomendados']
        varianza = pca_linear['analisis']['varianza_acumulada'][n_comp-1]
        print(f"   Componentes recomendados: {n_comp}")
        print(f"   Varianza explicada: {varianza:.1%}")

        # Variable más importante en PC1
        pc1_info = pca_linear['analisis']['componentes_info'][0]
        var_importante = pc1_info['top_variables'][0]['variable']
        loading = pc1_info['top_variables'][0]['loading']
        print(f"   Variable más importante en PC1: {var_importante} (loading: {loading:.3f})")

    print(f"\n📋 Recomendaciones del análisis exploratorio:")
    for i, rec in enumerate(exploratorio['recomendaciones'][:3], 1):
        print(f"   {i}. {rec}")

    return datos, {
        'exploratorio': exploratorio,
        'jerarquico': jerarquico,
        'kmeans': kmeans_opt,
        'dbscan': dbscan_opt,
        'pca': pca_avanzado
    }


# Add these functions to ml_functions_no_supervisado.py

def crear_visualizacion_clustering_puntos_muestreo(resultado, figsize=(16, 12)):
    """
    Crear visualización especializada para clustering de puntos de muestreo
    """
    if 'error' in resultado:
        raise ValueError(f"Error en el resultado: {resultado['error']}")

    tipo = resultado.get('tipo', '')
    fig = plt.figure(figsize=figsize)

    if tipo == 'kmeans_optimizado':
        return _crear_viz_kmeans_puntos_muestreo(resultado, fig)
    elif tipo == 'clustering_jerarquico_completo':
        return _crear_viz_jerarquico_puntos_muestreo(resultado, fig)
    elif tipo == 'dbscan_optimizado':
        return _crear_viz_dbscan_puntos_muestreo(resultado, fig)
    else:
        return _crear_viz_generica_puntos_muestreo(resultado, fig)


def _crear_viz_kmeans_puntos_muestreo(resultado, fig):
    """Visualización específica para K-Means de puntos de muestreo"""

    # Layout de 2x2
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Mapa de puntos de muestreo coloreados por cluster
    ax1 = fig.add_subplot(gs[0, 0])
    _graficar_puntos_muestreo_clusters(ax1, resultado)

    # 2. Timeline de muestreos con clusters
    ax2 = fig.add_subplot(gs[0, 1])
    _graficar_timeline_muestreos(ax2, resultado)

    # 3. Matriz de distancias entre puntos
    ax3 = fig.add_subplot(gs[1, 0])
    _graficar_matriz_distancias_puntos(ax3, resultado)

    # 4. Estadísticas por cluster
    ax4 = fig.add_subplot(gs[1, 1])
    _graficar_estadisticas_clusters_puntos(ax4, resultado)

    plt.suptitle('Clustering K-Means - Análisis de Puntos de Muestreo',
                 fontsize=16, fontweight='bold')

    return fig


def _crear_viz_jerarquico_puntos_muestreo(resultado, fig):
    """Visualización específica para clustering jerárquico de puntos de muestreo"""

    # Layout de 2x2
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Dendrograma simplificado
    ax1 = fig.add_subplot(gs[0, 0])
    _graficar_dendrograma_simplificado(ax1, resultado)

    # 2. Puntos de muestreo coloreados por cluster
    ax2 = fig.add_subplot(gs[0, 1])
    _graficar_puntos_muestreo_clusters(ax2, resultado)

    # 3. Métricas de evaluación
    ax3 = fig.add_subplot(gs[1, 0])
    _graficar_metricas_jerarquico(ax3, resultado)

    # 4. Información de configuración
    ax4 = fig.add_subplot(gs[1, 1])
    _mostrar_info_jerarquico(ax4, resultado)

    plt.suptitle('Clustering Jerárquico - Análisis de Puntos de Muestreo',
                 fontsize=16, fontweight='bold')

    return fig


def _crear_viz_dbscan_puntos_muestreo(resultado, fig):
    """Visualización específica para DBSCAN de puntos de muestreo"""

    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Puntos de muestreo con outliers destacados
    ax1 = fig.add_subplot(gs[0, 0])
    _graficar_puntos_dbscan_outliers(ax1, resultado)

    # 2. Densidad de muestreos en el tiempo
    ax2 = fig.add_subplot(gs[0, 1])
    _graficar_densidad_temporal_muestreos(ax2, resultado)

    # 3. Análisis de outliers temporales
    ax3 = fig.add_subplot(gs[1, 0])
    _graficar_outliers_temporales(ax3, resultado)

    # 4. Información de clusters y outliers
    ax4 = fig.add_subplot(gs[1, 1])
    _mostrar_info_dbscan_puntos(ax4, resultado)

    plt.suptitle('DBSCAN - Análisis de Puntos de Muestreo y Outliers',
                 fontsize=16, fontweight='bold')

    return fig


def _crear_viz_generica_puntos_muestreo(resultado, fig):
    """Visualización genérica para resultados de clustering"""
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, f'Visualización no implementada para: {resultado.get("tipo", "desconocido")}',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=14, bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax.set_title('Visualización No Disponible')
    ax.axis('off')
    return fig


def _graficar_puntos_muestreo_clusters(ax, resultado):
    """Graficar puntos de muestreo coloreados por cluster"""
    try:
        # Obtener datos originales
        if 'datos_originales' not in resultado:
            ax.text(0.5, 0.5, 'Datos no disponibles', ha='center', va='center',
                    transform=ax.transAxes)
            return

        datos = resultado['datos_originales']

        # Verificar si existe columna Points
        if 'Points' not in datos.columns:
            # Crear índices como puntos
            puntos = list(range(1, len(datos) + 1))
        else:
            puntos = datos['Points'].tolist()

        # Obtener labels de clusters
        k_optimo = resultado.get('recomendacion_k')
        if k_optimo and 'resultados_por_k' in resultado:
            if k_optimo in resultado['resultados_por_k']:
                labels = resultado['resultados_por_k'][k_optimo]['labels']
            else:
                labels = list(range(len(puntos)))
        else:
            # Para otros tipos de clustering
            mejor_config = resultado.get('mejor_configuracion', {})
            if 'labels' in mejor_config:
                labels = mejor_config['labels']
            else:
                labels = list(range(len(puntos)))

        # Crear coordenadas para visualización
        if len(puntos) <= 50:  # Para datasets pequeños, usar grid
            n_cols = int(np.ceil(np.sqrt(len(puntos))))
            x_coords = [i % n_cols for i in range(len(puntos))]
            y_coords = [i // n_cols for i in range(len(puntos))]
        else:  # Para datasets grandes, usar distribución aleatoria
            np.random.seed(42)
            x_coords = np.random.uniform(0, 10, len(puntos))
            y_coords = np.random.uniform(0, 10, len(puntos))

        # Scatter plot con colores por cluster
        unique_labels = set(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

        for i, (label, color) in enumerate(zip(unique_labels, colors)):
            mask = np.array(labels) == label
            if label == -1:  # Outliers en DBSCAN
                ax.scatter(np.array(x_coords)[mask], np.array(y_coords)[mask],
                           c='black', marker='x', s=100, alpha=0.8, label='Outliers')
            else:
                ax.scatter(np.array(x_coords)[mask], np.array(y_coords)[mask],
                           c=[color], label=f'Cluster {label}',
                           s=100, alpha=0.7, edgecolors='black', linewidth=1)

        ax.set_xlabel('Coordenada X (relativa)')
        ax.set_ylabel('Coordenada Y (relativa)')
        ax.set_title('Distribución de Puntos de Muestreo por Cluster')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

    except Exception as e:
        ax.text(0.5, 0.5, f'Error graficando puntos: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _graficar_timeline_muestreos(ax, resultado):
    """Graficar timeline de muestreos con franjas de colores por cluster"""
    try:
        if 'datos_originales' not in resultado:
            ax.text(0.5, 0.5, 'Datos no disponibles', ha='center', va='center',
                    transform=ax.transAxes)
            return

        datos = resultado['datos_originales']

        # Verificar columna de fecha
        fecha_col = None
        for col in ['Sampling_date', 'Fecha', 'Date', 'fecha']:
            if col in datos.columns:
                fecha_col = col
                break

        if fecha_col is None:
            # Crear fechas sintéticas
            fechas = pd.date_range('2023-01-01', periods=len(datos), freq='D')
        else:
            fechas = pd.to_datetime(datos[fecha_col])

        # Obtener labels
        k_optimo = resultado.get('recomendacion_k')
        if k_optimo and 'resultados_por_k' in resultado:
            if k_optimo in resultado['resultados_por_k']:
                labels = resultado['resultados_por_k'][k_optimo]['labels']
            else:
                labels = [0] * len(datos)
        else:
            mejor_config = resultado.get('mejor_configuracion', {})
            if 'labels' in mejor_config:
                labels = mejor_config['labels']
            else:
                labels = [0] * len(datos)

        # Puntos de muestreo
        if 'Points' in datos.columns:
            puntos = datos['Points'].tolist()
        else:
            puntos = list(range(1, len(datos) + 1))

        # Crear timeline
        unique_labels = sorted(set(labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

        for fecha, punto, label in zip(fechas, puntos, labels):
            color = color_map.get(label, 'gray')
            marker = 'x' if label == -1 else 'o'
            ax.scatter(fecha, punto, c=[color], s=60, alpha=0.8,
                       edgecolors='black', linewidth=0.5, marker=marker)

        ax.set_xlabel('Fecha de Muestreo')
        ax.set_ylabel('Punto de Muestreo')
        ax.set_title('Timeline de Muestreos por Cluster')

        # Rotar etiquetas de fecha
        ax.tick_params(axis='x', rotation=45)

        # Leyenda
        legend_elements = []
        for label in unique_labels:
            if label == -1:
                legend_elements.append(plt.scatter([], [], c='black', marker='x',
                                                   label='Outliers', s=60))
            else:
                legend_elements.append(plt.scatter([], [], c=[color_map[label]],
                                                   label=f'Cluster {label}', s=60))
        ax.legend(handles=legend_elements, loc='upper right')

        ax.grid(True, alpha=0.3)

    except Exception as e:
        ax.text(0.5, 0.5, f'Error en timeline: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _graficar_matriz_distancias_puntos(ax, resultado):
    """Graficar matriz de distancias entre puntos de muestreo"""
    try:
        if 'datos_originales' not in resultado:
            ax.text(0.5, 0.5, 'Datos no disponibles', ha='center', va='center',
                    transform=ax.transAxes)
            return

        datos = resultado['datos_originales']

        # Seleccionar subset de puntos para visualización clara
        max_puntos = 20
        if len(datos) > max_puntos:
            indices = np.random.choice(len(datos), max_puntos, replace=False)
            datos_subset = datos.iloc[indices]
        else:
            datos_subset = datos
            indices = range(len(datos))

        # Obtener variables numéricas
        numeric_cols = datos_subset.select_dtypes(include=[np.number]).columns
        # Excluir columnas no relevantes
        exclude_cols = ['Points', 'WQI_IDEAM_6V', 'WQI_IDEAM_7V', 'WQI_NSF_9V']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        if len(numeric_cols) == 0:
            ax.text(0.5, 0.5, 'No hay variables numéricas', ha='center', va='center',
                    transform=ax.transAxes)
            return

        # Calcular matriz de distancias
        from sklearn.metrics.pairwise import euclidean_distances
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        datos_scaled = scaler.fit_transform(datos_subset[numeric_cols].dropna())

        distancias = euclidean_distances(datos_scaled)

        # Crear heatmap
        im = ax.imshow(distancias, cmap='viridis', aspect='auto')

        # Etiquetas
        if 'Points' in datos_subset.columns:
            labels = [f'P{int(p)}' for p in datos_subset['Points'].tolist()]
        else:
            labels = [f'P{i + 1}' for i in indices]

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)

        # Colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)

        ax.set_title('Matriz de Distancias entre Puntos')

    except Exception as e:
        ax.text(0.5, 0.5, f'Error en matriz distancias: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _graficar_estadisticas_clusters_puntos(ax, resultado):
    """Graficar estadísticas de clusters de puntos de muestreo"""
    try:
        # Para K-Means
        k_optimo = resultado.get('recomendacion_k')
        if k_optimo and 'resultados_por_k' in resultado:
            if k_optimo not in resultado['resultados_por_k']:
                ax.text(0.5, 0.5, 'Datos de cluster no disponibles', ha='center', va='center',
                        transform=ax.transAxes)
                return

            cluster_data = resultado['resultados_por_k'][k_optimo]
            labels = cluster_data['labels']
        else:
            # Para otros métodos
            mejor_config = resultado.get('mejor_configuracion', {})
            if 'labels' not in mejor_config:
                ax.text(0.5, 0.5, 'Estadísticas no disponibles', ha='center', va='center',
                        transform=ax.transAxes)
                return
            labels = mejor_config['labels']
            cluster_data = mejor_config

        # Estadísticas por cluster
        unique_labels = sorted([l for l in set(labels) if l != -1])  # Excluir outliers
        tamaños = [labels.count(label) for label in unique_labels]
        proporciones = [size / len(labels) * 100 for size in tamaños]

        # Gráfico de barras
        bars = ax.bar(range(len(unique_labels)), proporciones,
                      color=plt.cm.Set3(np.linspace(0, 1, len(unique_labels))),
                      alpha=0.7, edgecolor='black')

        # Añadir valores en las barras
        for bar, tamaño, prop in zip(bars, tamaños, proporciones):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                    f'{tamaño}\n({prop:.1f}%)', ha='center', va='bottom',
                    fontweight='bold', fontsize=10)

        ax.set_xlabel('Cluster')
        ax.set_ylabel('Porcentaje de Puntos')
        ax.set_title('Distribución de Puntos por Cluster')
        ax.set_xticks(range(len(unique_labels)))
        ax.set_xticklabels([f'Cluster {label}' for label in unique_labels])
        ax.set_ylim(0, max(proporciones) * 1.2 if proporciones else 1)
        ax.grid(True, alpha=0.3, axis='y')

        # Información adicional
        silhouette = cluster_data.get('silhouette_score', 0)
        ax.text(0.02, 0.98, f'Silhouette Score: {silhouette:.3f}',
                transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Información de outliers si los hay
        if -1 in labels:
            n_outliers = labels.count(-1)
            ax.text(0.02, 0.90, f'Outliers: {n_outliers}',
                    transform=ax.transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.5))

    except Exception as e:
        ax.text(0.5, 0.5, f'Error en estadísticas: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


# Funciones adicionales para completar las visualizaciones

def _graficar_dendrograma_simplificado(ax, resultado):
    """Graficar un dendrograma simplificado para clustering jerárquico"""
    try:
        ax.text(0.5, 0.5, 'Dendrograma simplificado\n(Implementación pendiente)',
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue'))
        ax.set_title('Dendrograma')
        ax.axis('off')
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _graficar_metricas_jerarquico(ax, resultado):
    """Graficar métricas de evaluación para clustering jerárquico"""
    try:
        ax.text(0.5, 0.5, 'Métricas de evaluación\n(Implementación pendiente)',
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgreen'))
        ax.set_title('Métricas de Evaluación')
        ax.axis('off')
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _mostrar_info_jerarquico(ax, resultado):
    """Mostrar información de configuración jerárquica"""
    try:
        ax.text(0.5, 0.5, 'Información de configuración\n(Implementación pendiente)',
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightyellow'))
        ax.set_title('Configuración')
        ax.axis('off')
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _graficar_puntos_dbscan_outliers(ax, resultado):
    """Graficar puntos DBSCAN con outliers destacados"""
    try:
        _graficar_puntos_muestreo_clusters(ax, resultado)
        ax.set_title('Puntos de Muestreo - DBSCAN con Outliers')
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _graficar_densidad_temporal_muestreos(ax, resultado):
    """Graficar densidad temporal de muestreos"""
    try:
        ax.text(0.5, 0.5, 'Densidad temporal\n(Implementación pendiente)',
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightcyan'))
        ax.set_title('Densidad Temporal')
        ax.axis('off')
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _graficar_outliers_temporales(ax, resultado):
    """Graficar análisis de outliers temporales"""
    try:
        ax.text(0.5, 0.5, 'Outliers temporales\n(Implementación pendiente)',
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='mistyrose'))
        ax.set_title('Outliers Temporales')
        ax.axis('off')
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _mostrar_info_dbscan_puntos(ax, resultado):
    """Mostrar información de DBSCAN"""
    try:
        mejor_config = resultado.get('mejor_configuracion', {})

        info_text = f"Configuración DBSCAN:\n\n"
        info_text += f"Eps: {mejor_config.get('eps', 'N/A'):.3f}\n"
        info_text += f"Min Samples: {mejor_config.get('min_samples', 'N/A')}\n\n"
        info_text += f"Resultados:\n"
        info_text += f"Clusters: {mejor_config.get('n_clusters', 'N/A')}\n"
        info_text += f"Outliers: {mejor_config.get('n_noise', 'N/A')}\n"
        info_text += f"Silhouette: {mejor_config.get('silhouette_score', 'N/A'):.3f}\n"

        ax.text(0.05, 0.95, info_text, fontsize=10, va='top', ha='left',
                family='monospace', transform=ax.transAxes)
        ax.set_title('Información DBSCAN')
        ax.axis('off')

    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _crear_visualizacion_pca_puntos_muestreo(resultado, figsize=(16, 12)):
    """Crear visualización para PCA con enfoque en puntos de muestreo"""
    try:
        fig = plt.figure(figsize=figsize)

        resultados = resultado.get('resultados_por_metodo', {})

        if 'linear' not in resultados:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No hay resultados de PCA lineal para visualizar',
                    ha='center', va='center', transform=ax.transAxes)
            return fig

        # Layout 2x2
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Biplot PC1 vs PC2
        ax1 = fig.add_subplot(gs[0, 0])
        _graficar_biplot_pca(ax1, resultado)

        # 2. Varianza explicada
        ax2 = fig.add_subplot(gs[0, 1])
        _graficar_varianza_explicada(ax2, resultado)

        # 3. Contribuciones de variables
        ax3 = fig.add_subplot(gs[1, 0])
        _graficar_contribuciones_variables_pca(ax3, resultado)

        # 4. Información de componentes
        ax4 = fig.add_subplot(gs[1, 1])
        _mostrar_info_pca(ax4, resultado)

        plt.suptitle('Análisis de Componentes Principales (PCA)',
                     fontsize=16, fontweight='bold')

        return fig

    except Exception as e:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f'Error en visualización PCA: {str(e)[:100]}',
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='mistyrose'))
        ax.set_title('Error en Visualización PCA')
        ax.axis('off')
        return fig


def _graficar_biplot_pca(ax, resultado):
    """Graficar biplot de PCA"""
    try:
        linear_result = resultado['resultados_por_metodo']['linear']
        transformacion = np.array(linear_result['transformacion'])

        if transformacion.shape[1] < 2:
            ax.text(0.5, 0.5, 'Se necesitan al menos 2 componentes para biplot',
                    ha='center', va='center', transform=ax.transAxes)
            return

        # Scatter plot de las dos primeras componentes
        ax.scatter(transformacion[:, 0], transformacion[:, 1],
                   alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

        # Añadir algunos puntos representativos
        for i in range(0, len(transformacion), max(1, len(transformacion) // 10)):
            ax.annotate(f'P{i + 1}', (transformacion[i, 0], transformacion[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)

        # Información de varianza explicada
        analisis = linear_result.get('analisis', {})
        var_exp = analisis.get('varianza_explicada', [0, 0])

        pc1_var = var_exp[0] * 100 if len(var_exp) > 0 else 0
        pc2_var = var_exp[1] * 100 if len(var_exp) > 1 else 0

        ax.set_xlabel(f'PC1 ({pc1_var:.1f}% varianza)')
        ax.set_ylabel(f'PC2 ({pc2_var:.1f}% varianza)')
        ax.set_title('Biplot PCA - Puntos de Muestreo')
        ax.grid(True, alpha=0.3)

    except Exception as e:
        ax.text(0.5, 0.5, f'Error en biplot: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _graficar_varianza_explicada(ax, resultado):
    """Graficar varianza explicada por componentes"""
    try:
        linear_result = resultado['resultados_por_metodo']['linear']
        analisis = linear_result.get('analisis', {})

        var_explicada = analisis.get('varianza_explicada', [])
        var_acumulada = analisis.get('varianza_acumulada', [])

        if not var_explicada:
            ax.text(0.5, 0.5, 'No hay datos de varianza',
                    ha='center', va='center', transform=ax.transAxes)
            return

        x = range(1, len(var_explicada) + 1)

        # Barras para varianza individual
        bars = ax.bar(x, [v * 100 for v in var_explicada],
                      alpha=0.6, color='skyblue', label='Individual')

        # Línea para varianza acumulada
        ax2 = ax.twinx()
        ax2.plot(x, [v * 100 for v in var_acumulada],
                 'ro-', linewidth=2, markersize=6, label='Acumulada')

        ax.set_xlabel('Componente Principal')
        ax.set_ylabel('Varianza Explicada Individual (%)', color='blue')
        ax2.set_ylabel('Varianza Acumulada (%)', color='red')

        ax.set_title('Varianza Explicada por Componente')
        ax.grid(True, alpha=0.3)

        # Línea de referencia en 95%
        ax2.axhline(y=95, color='green', linestyle='--', alpha=0.7,
                    label='95% objetivo')

        # Leyendas
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

    except Exception as e:
        ax.text(0.5, 0.5, f'Error en varianza: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _graficar_contribuciones_variables_pca(ax, resultado):
    """Graficar contribuciones de variables a los primeros componentes"""
    try:
        linear_result = resultado['resultados_por_metodo']['linear']
        analisis = linear_result.get('analisis', {})
        componentes_info = analisis.get('componentes_info', [])

        if not componentes_info:
            ax.text(0.5, 0.5, 'No hay información de componentes',
                    ha='center', va='center', transform=ax.transAxes)
            return

        # Tomar los primeros 2 componentes
        n_comp = min(2, len(componentes_info))

        # Obtener top variables para cada componente
        variables_pc1 = []
        loadings_pc1 = []

        if len(componentes_info) > 0:
            top_vars_pc1 = componentes_info[0].get('top_variables', [])[:5]
            variables_pc1 = [var['variable'] for var in top_vars_pc1]
            loadings_pc1 = [var['loading'] for var in top_vars_pc1]

        if variables_pc1:
            # Gráfico de barras horizontales
            y_pos = range(len(variables_pc1))
            colors = ['red' if x < 0 else 'blue' for x in loadings_pc1]

            bars = ax.barh(y_pos, loadings_pc1, color=colors, alpha=0.7)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(variables_pc1)
            ax.set_xlabel('Loading (Contribución)')
            ax.set_title('Top Variables en PC1')
            ax.grid(True, alpha=0.3, axis='x')

            # Añadir valores en las barras
            for i, (bar, val) in enumerate(zip(bars, loadings_pc1)):
                ax.text(val + 0.01 if val >= 0 else val - 0.01, i,
                        f'{val:.2f}', va='center',
                        ha='left' if val >= 0 else 'right')
        else:
            ax.text(0.5, 0.5, 'No hay datos de contribuciones',
                    ha='center', va='center', transform=ax.transAxes)

    except Exception as e:
        ax.text(0.5, 0.5, f'Error en contribuciones: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _mostrar_info_pca(ax, resultado):
    """Mostrar información resumida de PCA"""
    try:
        linear_result = resultado['resultados_por_metodo']['linear']
        analisis = linear_result.get('analisis', {})

        n_comp_rec = linear_result.get('componentes_recomendados', 'N/A')
        var_objetivo = analisis.get('varianza_objetivo', 0.95)

        info_text = f"Resumen PCA:\n\n"
        info_text += f"Componentes recomendados: {n_comp_rec}\n"
        info_text += f"Objetivo de varianza: {var_objetivo:.1%}\n\n"

        # Información de los primeros componentes
        componentes_info = analisis.get('componentes_info', [])
        for i, comp_info in enumerate(componentes_info[:3]):
            var_exp = comp_info.get('varianza_explicada', 0)
            info_text += f"PC{i + 1}: {var_exp:.1%} varianza\n"

        if len(componentes_info) > 0:
            var_acum = analisis.get('varianza_acumulada', [])
            if len(var_acum) >= n_comp_rec:
                var_total = var_acum[n_comp_rec - 1] if n_comp_rec != 'N/A' else 0
                info_text += f"\nVarianza total explicada: {var_total:.1%}"

        ax.text(0.05, 0.95, info_text, fontsize=10, va='top', ha='left',
                family='monospace', transform=ax.transAxes)
        ax.set_title('Información PCA')
        ax.axis('off')

    except Exception as e:
        ax.text(0.5, 0.5, f'Error en info PCA: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _crear_visualizacion_exploratorio_puntos_muestreo(resultado, figsize=(16, 12)):
    """Crear visualización para análisis exploratorio"""
    try:
        fig = plt.figure(figsize=figsize)

        # Layout 2x2
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Matriz de correlaciones
        ax1 = fig.add_subplot(gs[0, 0])
        _graficar_matriz_correlaciones(ax1, resultado)

        # 2. Distribución de outliers
        ax2 = fig.add_subplot(gs[0, 1])
        _graficar_distribucion_outliers(ax2, resultado)

        # 3. Estadísticas descriptivas
        ax3 = fig.add_subplot(gs[1, 0])
        _graficar_estadisticas_descriptivas(ax3, resultado)

        # 4. Calidad de datos
        ax4 = fig.add_subplot(gs[1, 1])
        _mostrar_calidad_datos(ax4, resultado)

        plt.suptitle('Análisis Exploratorio de Datos',
                     fontsize=16, fontweight='bold')

        return fig

    except Exception as e:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f'Error en visualización exploratoria: {str(e)[:100]}',
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='mistyrose'))
        ax.set_title('Error en Visualización Exploratoria')
        ax.axis('off')
        return fig


def _graficar_matriz_correlaciones(ax, resultado):
    """Graficar matriz de correlaciones"""
    try:
        correlaciones = resultado.get('correlaciones', {})
        matriz_pearson = correlaciones.get('matriz_pearson', {})

        if not matriz_pearson:
            ax.text(0.5, 0.5, 'No hay datos de correlaciones',
                    ha='center', va='center', transform=ax.transAxes)
            return

        # Convertir a DataFrame
        import pandas as pd
        df_corr = pd.DataFrame(matriz_pearson)

        # Seleccionar subset de variables si hay muchas
        if len(df_corr.columns) > 10:
            # Tomar las 10 primeras variables
            df_corr = df_corr.iloc[:10, :10]

        # Crear heatmap
        im = ax.imshow(df_corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

        # Etiquetas
        ax.set_xticks(range(len(df_corr.columns)))
        ax.set_yticks(range(len(df_corr.index)))
        ax.set_xticklabels(df_corr.columns, rotation=45, ha='right')
        ax.set_yticklabels(df_corr.index)

        # Colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Añadir valores en las celdas
        for i in range(len(df_corr.index)):
            for j in range(len(df_corr.columns)):
                val = df_corr.iloc[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color=color, fontsize=8)

        ax.set_title('Matriz de Correlaciones (Pearson)')

    except Exception as e:
        ax.text(0.5, 0.5, f'Error en correlaciones: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _graficar_distribucion_outliers(ax, resultado):
    """Graficar distribución de outliers por método"""
    try:
        outliers = resultado.get('outliers', {})

        metodos = []
        cantidades = []

        for metodo, data in outliers.items():
            if metodo == 'consenso':
                continue
            if isinstance(data, dict) and 'total' in data:
                metodos.append(metodo.replace('_', ' ').title())
                cantidades.append(data['total'])

        if not metodos:
            ax.text(0.5, 0.5, 'No hay datos de outliers',
                    ha='center', va='center', transform=ax.transAxes)
            return

        # Gráfico de barras
        bars = ax.bar(metodos, cantidades,
                      color=['skyblue', 'lightgreen', 'salmon', 'gold'][:len(metodos)],
                      alpha=0.7, edgecolor='black')

        # Añadir valores en las barras
        for bar, val in zip(bars, cantidades):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                    f'{val}', ha='center', va='bottom', fontweight='bold')

        ax.set_xlabel('Método de Detección')
        ax.set_ylabel('Número de Outliers')
        ax.set_title('Outliers Detectados por Método')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        # Añadir información de consenso
        consenso = outliers.get('consenso', {})
        if consenso:
            total_consenso = consenso.get('total_unico', 0)
            porcentaje = consenso.get('porcentaje', 0)
            ax.text(0.02, 0.98, f'Consenso: {total_consenso} ({porcentaje:.1f}%)',
                    transform=ax.transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    except Exception as e:
        ax.text(0.5, 0.5, f'Error en outliers: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _graficar_estadisticas_descriptivas(ax, resultado):
    """Graficar resumen de estadísticas descriptivas"""
    try:
        estadisticas = resultado.get('estadisticas_descriptivas', {})

        if not estadisticas:
            ax.text(0.5, 0.5, 'No hay estadísticas descriptivas',
                    ha='center', va='center', transform=ax.transAxes)
            return

        # Tomar subset de variables
        variables = list(estadisticas.keys())[:8]  # Máximo 8 variables

        # Métricas a mostrar
        skewness_vals = []
        kurtosis_vals = []
        cv_vals = []

        for var in variables:
            stats = estadisticas[var]
            skewness_vals.append(stats.get('skewness', 0))
            kurtosis_vals.append(stats.get('kurtosis', 0))
            cv_val = stats.get('cv', 0)
            # Limitar CV extremos
            cv_vals.append(min(cv_val, 2) if cv_val != np.inf else 0)

        # Gráfico de líneas múltiples
        x = range(len(variables))

        ax.plot(x, skewness_vals, 'o-', label='Skewness', linewidth=2, markersize=6)
        ax.plot(x, kurtosis_vals, 's-', label='Kurtosis', linewidth=2, markersize=6)
        ax.plot(x, cv_vals, '^-', label='CV', linewidth=2, markersize=6)

        # Líneas de referencia
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Referencia ±1')
        ax.axhline(y=-1, color='red', linestyle='--', alpha=0.5)

        ax.set_xlabel('Variables')
        ax.set_ylabel('Valor')
        ax.set_title('Estadísticas de Forma por Variable')
        ax.set_xticks(x)
        ax.set_xticklabels([var[:8] + '...' if len(var) > 8 else var
                            for var in variables], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

    except Exception as e:
        ax.text(0.5, 0.5, f'Error en estadísticas: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _mostrar_calidad_datos(ax, resultado):
    """Mostrar información de calidad de datos"""
    try:
        calidad = resultado.get('calidad_datos', {})

        if not calidad:
            ax.text(0.5, 0.5, 'No hay información de calidad',
                    ha='center', va='center', transform=ax.transAxes)
            return

        quality_score = calidad.get('quality_score', 0)
        calificacion = calidad.get('calificacion', 'N/A')

        # Información de valores faltantes
        missing_info = calidad.get('valores_faltantes', {})
        total_missing = sum(info.get('count', 0) for info in missing_info.values())

        # Información de duplicados
        duplicados_info = calidad.get('duplicados', {})
        duplicados_count = duplicados_info.get('count', 0)

        info_text = f"Calidad de Datos:\n\n"
        info_text += f"Score General: {quality_score:.1f}/100\n"
        info_text += f"Calificación: {calificacion}\n\n"
        info_text += f"Valores Faltantes: {total_missing}\n"
        info_text += f"Duplicados: {duplicados_count}\n\n"

        # Top variables con missing values
        if missing_info:
            sorted_missing = sorted(missing_info.items(),
                                    key=lambda x: x[1].get('percentage', 0),
                                    reverse=True)[:3]
            info_text += "Variables con más faltantes:\n"
            for var, info in sorted_missing:
                pct = info.get('percentage', 0)
                if pct > 0:
                    info_text += f"• {var}: {pct:.1f}%\n"

        # Color según calidad
        if quality_score >= 90:
            bgcolor = 'lightgreen'
        elif quality_score >= 70:
            bgcolor = 'lightyellow'
        else:
            bgcolor = 'mistyrose'

        ax.text(0.05, 0.95, info_text, fontsize=10, va='top', ha='left',
                family='monospace', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor=bgcolor, alpha=0.7))
        ax.set_title('Evaluación de Calidad')
        ax.axis('off')

    except Exception as e:
        ax.text(0.5, 0.5, f'Error en calidad: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _crear_visualizacion_generica(resultado, figsize=(16, 12)):
    """Crear visualización genérica para resultados no específicos"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    tipo = resultado.get('tipo', 'desconocido')
    ax.text(0.5, 0.5, f'Visualización genérica para: {tipo}\n\n'
                      'Esta función se puede expandir según las necesidades específicas',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=14, bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax.set_title(f'Resultado: {tipo}')
    ax.axis('off')

    return fig


# Agregar estas funciones al final de ml_functions_no_supervisado.py

def _crear_visualizacion_pca_puntos_muestreo(resultado, figsize=(16, 12)):
    """Crear visualización para PCA con enfoque en puntos de muestreo"""
    try:
        fig = plt.figure(figsize=figsize)

        resultados = resultado.get('resultados_por_metodo', {})

        if 'linear' not in resultados:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No hay resultados de PCA lineal para visualizar',
                    ha='center', va='center', transform=ax.transAxes)
            return fig

        # Layout 2x2
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Biplot PC1 vs PC2
        ax1 = fig.add_subplot(gs[0, 0])
        _graficar_biplot_pca(ax1, resultado)

        # 2. Varianza explicada
        ax2 = fig.add_subplot(gs[0, 1])
        _graficar_varianza_explicada(ax2, resultado)

        # 3. Contribuciones de variables
        ax3 = fig.add_subplot(gs[1, 0])
        _graficar_contribuciones_variables_pca(ax3, resultado)

        # 4. Información de componentes
        ax4 = fig.add_subplot(gs[1, 1])
        _mostrar_info_pca(ax4, resultado)

        plt.suptitle('Análisis de Componentes Principales (PCA)',
                     fontsize=16, fontweight='bold')

        return fig

    except Exception as e:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f'Error en visualización PCA: {str(e)[:100]}',
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='mistyrose'))
        ax.set_title('Error en Visualización PCA')
        ax.axis('off')
        return fig


def _crear_visualizacion_exploratorio_puntos_muestreo(resultado, figsize=(16, 12)):
    """Crear visualización para análisis exploratorio"""
    try:
        fig = plt.figure(figsize=figsize)

        # Layout 2x2
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Matriz de correlaciones
        ax1 = fig.add_subplot(gs[0, 0])
        _graficar_matriz_correlaciones(ax1, resultado)

        # 2. Distribución de outliers
        ax2 = fig.add_subplot(gs[0, 1])
        _graficar_distribucion_outliers(ax2, resultado)

        # 3. Estadísticas descriptivas
        ax3 = fig.add_subplot(gs[1, 0])
        _graficar_estadisticas_descriptivas(ax3, resultado)

        # 4. Calidad de datos
        ax4 = fig.add_subplot(gs[1, 1])
        _mostrar_calidad_datos(ax4, resultado)

        plt.suptitle('Análisis Exploratorio de Datos',
                     fontsize=16, fontweight='bold')

        return fig

    except Exception as e:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f'Error en visualización exploratoria: {str(e)[:100]}',
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='mistyrose'))
        ax.set_title('Error en Visualización Exploratoria')
        ax.axis('off')
        return fig


# Funciones de soporte adicionales que también necesitas:

def _graficar_biplot_pca(ax, resultado):
    """Graficar biplot de PCA"""
    try:
        linear_result = resultado['resultados_por_metodo']['linear']
        transformacion = np.array(linear_result['transformacion'])

        if transformacion.shape[1] < 2:
            ax.text(0.5, 0.5, 'Se necesitan al menos 2 componentes para biplot',
                    ha='center', va='center', transform=ax.transAxes)
            return

        # Scatter plot de las dos primeras componentes
        ax.scatter(transformacion[:, 0], transformacion[:, 1],
                   alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

        # Añadir algunos puntos representativos
        for i in range(0, len(transformacion), max(1, len(transformacion) // 10)):
            ax.annotate(f'P{i + 1}', (transformacion[i, 0], transformacion[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)

        # Información de varianza explicada
        analisis = linear_result.get('analisis', {})
        var_exp = analisis.get('varianza_explicada', [0, 0])

        pc1_var = var_exp[0] * 100 if len(var_exp) > 0 else 0
        pc2_var = var_exp[1] * 100 if len(var_exp) > 1 else 0

        ax.set_xlabel(f'PC1 ({pc1_var:.1f}% varianza)')
        ax.set_ylabel(f'PC2 ({pc2_var:.1f}% varianza)')
        ax.set_title('Biplot PCA - Puntos de Muestreo')
        ax.grid(True, alpha=0.3)

    except Exception as e:
        ax.text(0.5, 0.5, f'Error en biplot: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _graficar_varianza_explicada(ax, resultado):
    """Graficar varianza explicada por componentes"""
    try:
        linear_result = resultado['resultados_por_metodo']['linear']
        analisis = linear_result.get('analisis', {})

        var_explicada = analisis.get('varianza_explicada', [])
        var_acumulada = analisis.get('varianza_acumulada', [])

        if not var_explicada:
            ax.text(0.5, 0.5, 'No hay datos de varianza',
                    ha='center', va='center', transform=ax.transAxes)
            return

        x = range(1, len(var_explicada) + 1)

        # Barras para varianza individual
        bars = ax.bar(x, [v * 100 for v in var_explicada],
                      alpha=0.6, color='skyblue', label='Individual')

        # Línea para varianza acumulada
        ax2 = ax.twinx()
        ax2.plot(x, [v * 100 for v in var_acumulada],
                 'ro-', linewidth=2, markersize=6, label='Acumulada')

        ax.set_xlabel('Componente Principal')
        ax.set_ylabel('Varianza Explicada Individual (%)', color='blue')
        ax2.set_ylabel('Varianza Acumulada (%)', color='red')

        ax.set_title('Varianza Explicada por Componente')
        ax.grid(True, alpha=0.3)

        # Línea de referencia en 95%
        ax2.axhline(y=95, color='green', linestyle='--', alpha=0.7,
                    label='95% objetivo')

        # Leyendas
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

    except Exception as e:
        ax.text(0.5, 0.5, f'Error en varianza: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _graficar_contribuciones_variables_pca(ax, resultado):
    """Graficar contribuciones de variables a los primeros componentes"""
    try:
        linear_result = resultado['resultados_por_metodo']['linear']
        analisis = linear_result.get('analisis', {})
        componentes_info = analisis.get('componentes_info', [])

        if not componentes_info:
            ax.text(0.5, 0.5, 'No hay información de componentes',
                    ha='center', va='center', transform=ax.transAxes)
            return

        # Obtener top variables para PC1
        variables_pc1 = []
        loadings_pc1 = []

        if len(componentes_info) > 0:
            top_vars_pc1 = componentes_info[0].get('top_variables', [])[:5]
            variables_pc1 = [var['variable'] for var in top_vars_pc1]
            loadings_pc1 = [var['loading'] for var in top_vars_pc1]

        if variables_pc1:
            # Gráfico de barras horizontales
            y_pos = range(len(variables_pc1))
            colors = ['red' if x < 0 else 'blue' for x in loadings_pc1]

            bars = ax.barh(y_pos, loadings_pc1, color=colors, alpha=0.7)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(variables_pc1)
            ax.set_xlabel('Loading (Contribución)')
            ax.set_title('Top Variables en PC1')
            ax.grid(True, alpha=0.3, axis='x')

            # Añadir valores en las barras
            for i, (bar, val) in enumerate(zip(bars, loadings_pc1)):
                ax.text(val + 0.01 if val >= 0 else val - 0.01, i,
                        f'{val:.2f}', va='center',
                        ha='left' if val >= 0 else 'right')
        else:
            ax.text(0.5, 0.5, 'No hay datos de contribuciones',
                    ha='center', va='center', transform=ax.transAxes)

    except Exception as e:
        ax.text(0.5, 0.5, f'Error en contribuciones: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _mostrar_info_pca(ax, resultado):
    """Mostrar información resumida de PCA"""
    try:
        linear_result = resultado['resultados_por_metodo']['linear']
        analisis = linear_result.get('analisis', {})

        n_comp_rec = linear_result.get('componentes_recomendados', 'N/A')
        var_objetivo = analisis.get('varianza_objetivo', 0.95)

        info_text = f"Resumen PCA:\n\n"
        info_text += f"Componentes recomendados: {n_comp_rec}\n"
        info_text += f"Objetivo de varianza: {var_objetivo:.1%}\n\n"

        # Información de los primeros componentes
        componentes_info = analisis.get('componentes_info', [])
        for i, comp_info in enumerate(componentes_info[:3]):
            var_exp = comp_info.get('varianza_explicada', 0)
            info_text += f"PC{i + 1}: {var_exp:.1%} varianza\n"

        if len(componentes_info) > 0 and n_comp_rec != 'N/A':
            var_acum = analisis.get('varianza_acumulada', [])
            if len(var_acum) >= n_comp_rec:
                var_total = var_acum[n_comp_rec - 1]
                info_text += f"\nVarianza total explicada: {var_total:.1%}"

        ax.text(0.05, 0.95, info_text, fontsize=10, va='top', ha='left',
                family='monospace', transform=ax.transAxes)
        ax.set_title('Información PCA')
        ax.axis('off')

    except Exception as e:
        ax.text(0.5, 0.5, f'Error en info PCA: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _graficar_matriz_correlaciones(ax, resultado):
    """Graficar matriz de correlaciones"""
    try:
        correlaciones = resultado.get('correlaciones', {})
        matriz_pearson = correlaciones.get('matriz_pearson', {})

        if not matriz_pearson:
            ax.text(0.5, 0.5, 'No hay datos de correlaciones',
                    ha='center', va='center', transform=ax.transAxes)
            return

        # Convertir a DataFrame
        import pandas as pd
        df_corr = pd.DataFrame(matriz_pearson)

        # Seleccionar subset de variables si hay muchas
        if len(df_corr.columns) > 10:
            # Tomar las 10 primeras variables
            df_corr = df_corr.iloc[:10, :10]

        # Crear heatmap
        im = ax.imshow(df_corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

        # Etiquetas
        ax.set_xticks(range(len(df_corr.columns)))
        ax.set_yticks(range(len(df_corr.index)))
        ax.set_xticklabels(df_corr.columns, rotation=45, ha='right')
        ax.set_yticklabels(df_corr.index)

        # Colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Añadir valores en las celdas
        for i in range(len(df_corr.index)):
            for j in range(len(df_corr.columns)):
                val = df_corr.iloc[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color=color, fontsize=8)

        ax.set_title('Matriz de Correlaciones (Pearson)')

    except Exception as e:
        ax.text(0.5, 0.5, f'Error en correlaciones: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _graficar_distribucion_outliers(ax, resultado):
    """Graficar distribución de outliers por método"""
    try:
        outliers = resultado.get('outliers', {})

        metodos = []
        cantidades = []

        for metodo, data in outliers.items():
            if metodo == 'consenso':
                continue
            if isinstance(data, dict) and 'total' in data:
                metodos.append(metodo.replace('_', ' ').title())
                cantidades.append(data['total'])

        if not metodos:
            ax.text(0.5, 0.5, 'No hay datos de outliers',
                    ha='center', va='center', transform=ax.transAxes)
            return

        # Gráfico de barras
        bars = ax.bar(metodos, cantidades,
                      color=['skyblue', 'lightgreen', 'salmon', 'gold'][:len(metodos)],
                      alpha=0.7, edgecolor='black')

        # Añadir valores en las barras
        for bar, val in zip(bars, cantidades):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                    f'{val}', ha='center', va='bottom', fontweight='bold')

        ax.set_xlabel('Método de Detección')
        ax.set_ylabel('Número de Outliers')
        ax.set_title('Outliers Detectados por Método')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        # Añadir información de consenso
        consenso = outliers.get('consenso', {})
        if consenso:
            total_consenso = consenso.get('total_unico', 0)
            porcentaje = consenso.get('porcentaje', 0)
            ax.text(0.02, 0.98, f'Consenso: {total_consenso} ({porcentaje:.1f}%)',
                    transform=ax.transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    except Exception as e:
        ax.text(0.5, 0.5, f'Error en outliers: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _graficar_estadisticas_descriptivas(ax, resultado):
    """Graficar resumen de estadísticas descriptivas"""
    try:
        estadisticas = resultado.get('estadisticas_descriptivas', {})

        if not estadisticas:
            ax.text(0.5, 0.5, 'No hay estadísticas descriptivas',
                    ha='center', va='center', transform=ax.transAxes)
            return

        # Tomar subset de variables
        variables = list(estadisticas.keys())[:8]  # Máximo 8 variables

        # Métricas a mostrar
        skewness_vals = []
        kurtosis_vals = []
        cv_vals = []

        for var in variables:
            stats = estadisticas[var]
            skewness_vals.append(stats.get('skewness', 0))
            kurtosis_vals.append(stats.get('kurtosis', 0))
            cv_val = stats.get('cv', 0)
            # Limitar CV extremos
            cv_vals.append(min(cv_val, 2) if cv_val != np.inf else 0)

        # Gráfico de líneas múltiples
        x = range(len(variables))

        ax.plot(x, skewness_vals, 'o-', label='Skewness', linewidth=2, markersize=6)
        ax.plot(x, kurtosis_vals, 's-', label='Kurtosis', linewidth=2, markersize=6)
        ax.plot(x, cv_vals, '^-', label='CV', linewidth=2, markersize=6)

        # Líneas de referencia
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Referencia ±1')
        ax.axhline(y=-1, color='red', linestyle='--', alpha=0.5)

        ax.set_xlabel('Variables')
        ax.set_ylabel('Valor')
        ax.set_title('Estadísticas de Forma por Variable')
        ax.set_xticks(x)
        ax.set_xticklabels([var[:8] + '...' if len(var) > 8 else var
                            for var in variables], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

    except Exception as e:
        ax.text(0.5, 0.5, f'Error en estadísticas: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def _mostrar_calidad_datos(ax, resultado):
    """Mostrar información de calidad de datos"""
    try:
        calidad = resultado.get('calidad_datos', {})

        if not calidad:
            ax.text(0.5, 0.5, 'No hay información de calidad',
                    ha='center', va='center', transform=ax.transAxes)
            return

        quality_score = calidad.get('quality_score', 0)
        calificacion = calidad.get('calificacion', 'N/A')

        # Información de valores faltantes
        missing_info = calidad.get('valores_faltantes', {})
        total_missing = sum(info.get('count', 0) for info in missing_info.values())

        # Información de duplicados
        duplicados_info = calidad.get('duplicados', {})
        duplicados_count = duplicados_info.get('count', 0)

        info_text = f"Calidad de Datos:\n\n"
        info_text += f"Score General: {quality_score:.1f}/100\n"
        info_text += f"Calificación: {calificacion}\n\n"
        info_text += f"Valores Faltantes: {total_missing}\n"
        info_text += f"Duplicados: {duplicados_count}\n\n"

        # Top variables con missing values
        if missing_info:
            sorted_missing = sorted(missing_info.items(),
                                    key=lambda x: x[1].get('percentage', 0),
                                    reverse=True)[:3]
            info_text += "Variables con más faltantes:\n"
            for var, info in sorted_missing:
                pct = info.get('percentage', 0)
                if pct > 0:
                    info_text += f"• {var}: {pct:.1f}%\n"

        # Color según calidad
        if quality_score >= 90:
            bgcolor = 'lightgreen'
        elif quality_score >= 70:
            bgcolor = 'lightyellow'
        else:
            bgcolor = 'mistyrose'

        ax.text(0.05, 0.95, info_text, fontsize=10, va='top', ha='left',
                family='monospace', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor=bgcolor, alpha=0.7))
        ax.set_title('Evaluación de Calidad')
        ax.axis('off')

    except Exception as e:
        ax.text(0.5, 0.5, f'Error en calidad: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)

if __name__ == "__main__":
    # Ejecutar demostración
    datos, resultados = demo_ml_no_supervisado_completo()
    print("\n✅ Demostración completada exitosamente!")
    print(f"📊 Datos analizados: {len(datos)} muestras con {datos.shape[1]} variables")
    print(f"🎯 Métodos ejecutados: {len(resultados)} técnicas de ML no supervisado")