"""
ml_functions_no_supervisado.py - Compatible con PyInstaller
Implementación de Machine Learning No Supervisado usando solo numpy, pandas y matplotlib
Versión optimizada para empaquetado con PyInstaller
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import warnings
from datetime import datetime
import json

warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])

# ==================== FUNCIONES DE GENERACIÓN DE DATOS ====================

def generar_datos_agua_realistas(n_muestras=200, seed=42, incluir_outliers=True):
    """Generar datos sintéticos realistas de calidad del agua"""
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

    # Variables principales
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
    datos['WT'] = np.clip(temperatura_values, 10, 35)
    datos['CTD'] = np.clip(conductividad_values, 50, 1500)

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

    # Variables adicionales
    for var, params in [
        ('FC', (100, 1, 10000)),
        ('TC', (500, 10, 50000)),
        ('NO3', (5, 0.1, 50)),
        ('NO2', (0.5, 0.01, 3)),
        ('N_NH3', (1, 0.01, 10)),
        ('TP', (0.5, 0.01, 5)),
        ('TN', (3, 0.5, 20)),
        ('TKN', (2, 0.1, 15)),
        ('TSS', (10, 1, 200)),
        ('TS', (150, 50, 2000)),
        ('Q', (50, 1, 500)),
        ('ALC', (150, 50, 500)),
        ('H', (20, 5, 200)),
        ('ET', (25, 15, 40))
    ]:
        scale, min_val, max_val = params
        datos[var] = np.clip(
            np.random.exponential(scale, n_muestras),
            min_val, max_val
        )

    # Crear DataFrame
    df = pd.DataFrame(datos)

    # Calcular índices de calidad
    df['WQI_IDEAM_6V'] = calcular_indice_calidad_simple(df)
    df['WQI_IDEAM_7V'] = df['WQI_IDEAM_6V'] * np.random.uniform(0.9, 1.1, n_muestras)
    df['WQI_NSF_9V'] = df['WQI_IDEAM_6V'] * np.random.uniform(0.8, 1.2, n_muestras)

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

    # Agregar outliers realistas
    if incluir_outliers:
        n_outliers = max(1, n_muestras // 50)
        outlier_indices = np.random.choice(df.index, n_outliers, replace=False)

        for idx in outlier_indices:
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

# ==================== IMPLEMENTACIONES MANUALES DE ALGORITMOS ====================

def manual_standard_scaler(X):
    """Escalado estándar manual"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Evitar división por cero
    std = np.where(std == 0, 1, std)
    return (X - mean) / std, mean, std

def manual_minmax_scaler(X):
    """Escalado min-max manual"""
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    # Evitar división por cero
    range_vals = max_vals - min_vals
    range_vals = np.where(range_vals == 0, 1, range_vals)
    return (X - min_vals) / range_vals, min_vals, range_vals

def manual_robust_scaler(X):
    """Escalado robusto manual usando mediana y IQR"""
    median = np.median(X, axis=0)
    q25 = np.percentile(X, 25, axis=0)
    q75 = np.percentile(X, 75, axis=0)
    iqr = q75 - q25
    # Evitar división por cero
    iqr = np.where(iqr == 0, 1, iqr)
    return (X - median) / iqr, median, iqr

def aplicar_escalado(X, metodo='standard'):
    """Aplicar escalado manual a los datos"""
    if isinstance(X, pd.DataFrame):
        X_vals = X.values
        columns = X.columns
        index = X.index
    else:
        X_vals = X
        columns = None
        index = None

    if metodo == 'standard':
        X_scaled, param1, param2 = manual_standard_scaler(X_vals)
        scaler_info = {'type': 'standard', 'mean': param1, 'std': param2}
    elif metodo == 'minmax':
        X_scaled, param1, param2 = manual_minmax_scaler(X_vals)
        scaler_info = {'type': 'minmax', 'min': param1, 'range': param2}
    elif metodo == 'robust':
        X_scaled, param1, param2 = manual_robust_scaler(X_vals)
        scaler_info = {'type': 'robust', 'median': param1, 'iqr': param2}
    else:
        X_scaled = X_vals
        scaler_info = {'type': 'none'}

    if columns is not None:
        X_scaled = pd.DataFrame(X_scaled, columns=columns, index=index)

    return X_scaled, scaler_info

def manual_pca(X, n_components=None):
    """Implementación manual de PCA"""
    # Centrar datos
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # Matriz de covarianza
    cov_matrix = np.cov(X_centered.T)

    # Eigendescomposición
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Ordenar por eigenvalores (descendente)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Seleccionar componentes
    if n_components is None:
        n_components = min(X.shape[0], X.shape[1])

    n_components = min(n_components, len(eigenvalues))

    # Componentes principales
    components = eigenvectors[:, :n_components]

    # Transformar datos
    X_transformed = np.dot(X_centered, components)

    # Varianza explicada
    total_var = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues[:n_components] / total_var

    return {
        'components': components.T,
        'explained_variance': eigenvalues[:n_components],
        'explained_variance_ratio': explained_variance_ratio,
        'mean': mean,
        'X_transformed': X_transformed
    }

def manual_kmeans(X, k, max_iters=100, random_state=42):
    """Implementación manual de K-Means"""
    np.random.seed(random_state)

    n_samples, n_features = X.shape

    # Inicialización aleatoria de centroides
    centroids = X[np.random.choice(n_samples, k, replace=False)]

    for iteration in range(max_iters):
        # Asignar puntos a clusters
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        # Actualizar centroides
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Verificar convergencia
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    # Calcular inercia
    inertia = sum(np.sum((X[labels == i] - centroids[i])**2) for i in range(k))

    return {
        'labels': labels,
        'centroids': centroids,
        'inertia': inertia,
        'n_iter': iteration + 1
    }

def manual_silhouette_score(X, labels):
    """Implementación manual del Silhouette Score"""
    n_samples = len(X)
    unique_labels = np.unique(labels)

    if len(unique_labels) == 1:
        return 0.0

    silhouette_vals = []

    for i in range(n_samples):
        # Cluster del punto actual
        own_cluster = labels[i]

        # Distancia promedio intra-cluster (a)
        same_cluster_points = X[labels == own_cluster]
        if len(same_cluster_points) > 1:
            a = np.mean([np.linalg.norm(X[i] - point) for point in same_cluster_points if not np.array_equal(X[i], point)])
        else:
            a = 0

        # Distancia promedio al cluster más cercano (b)
        b_values = []
        for cluster in unique_labels:
            if cluster != own_cluster:
                other_cluster_points = X[labels == cluster]
                if len(other_cluster_points) > 0:
                    b_cluster = np.mean([np.linalg.norm(X[i] - point) for point in other_cluster_points])
                    b_values.append(b_cluster)

        b = min(b_values) if b_values else 0

        # Silhouette score para el punto
        if max(a, b) == 0:
            s = 0
        else:
            s = (b - a) / max(a, b)

        silhouette_vals.append(s)

    return np.mean(silhouette_vals)

def manual_dbscan(X, eps=0.5, min_samples=5):
    """Implementación simplificada de DBSCAN"""
    n_samples = len(X)
    labels = np.full(n_samples, -1)  # -1 indica ruido
    cluster_id = 0

    for i in range(n_samples):
        if labels[i] != -1:  # Ya procesado
            continue

        # Encontrar vecinos
        distances = np.linalg.norm(X - X[i], axis=1)
        neighbors = np.where(distances <= eps)[0]

        if len(neighbors) < min_samples:
            continue  # Punto de ruido

        # Crear nuevo cluster
        labels[i] = cluster_id

        # Expandir cluster
        seed_set = set(neighbors)
        seed_set.discard(i)

        while seed_set:
            current_point = seed_set.pop()

            if labels[current_point] == -1:  # Era ruido
                labels[current_point] = cluster_id
            elif labels[current_point] != -1:  # Ya asignado
                continue
            else:
                labels[current_point] = cluster_id

            # Encontrar vecinos del punto actual
            current_distances = np.linalg.norm(X - X[current_point], axis=1)
            current_neighbors = np.where(current_distances <= eps)[0]

            if len(current_neighbors) >= min_samples:
                seed_set.update(current_neighbors)

        cluster_id += 1

    return labels

def detectar_outliers_zscore(X, threshold=3.0):
    """Detectar outliers usando Z-score"""
    z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
    return np.any(z_scores > threshold, axis=1)

def detectar_outliers_iqr(X):
    """Detectar outliers usando método IQR"""
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return np.any((X < lower_bound) | (X > upper_bound), axis=1)

def detectar_outliers_isolation_forest_simple(X, contamination=0.1):
    """Implementación simplificada de Isolation Forest"""
    n_samples = len(X)
    n_outliers = int(contamination * n_samples)

    # Método simplificado: usar distancias a la mediana
    median = np.median(X, axis=0)
    distances = np.linalg.norm(X - median, axis=1)

    # Los puntos más alejados son outliers
    outlier_indices = np.argpartition(distances, -n_outliers)[-n_outliers:]
    outliers = np.zeros(n_samples, dtype=bool)
    outliers[outlier_indices] = True

    return outliers

# ==================== ANÁLISIS PRINCIPAL ====================

def analizar_clusters_manual(X, labels, variables):
    """Análisis detallado de clusters"""
    cluster_stats = {}
    unique_labels = np.unique(labels)

    for cluster_id in unique_labels:
        mask = labels == cluster_id
        cluster_data = X[mask]

        if len(cluster_data) == 0:
            continue

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

        centroide = cluster_data[variables].mean().to_dict()

        cluster_stats[f'cluster_{cluster_id}'] = {
            'tamaño': int(np.sum(mask)),
            'proporcion': float(np.sum(mask) / len(labels)),
            'centroide': centroide,
            'estadisticas': stats_por_variable
        }

    return cluster_stats

def kmeans_optimizado_completo(data, variables=None, k_range=None,
                             escalado='standard', random_state=42, verbose=True):
    """K-Means optimizado con evaluación de K óptimo"""
    if verbose:
        print("🔍 Iniciando K-Means optimizado...")

    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()

    # Excluir columnas no relevantes
    exclude_cols = ['Points', 'Sampling_date', 'WQI_IDEAM_6V', 'WQI_IDEAM_7V', 'WQI_NSF_9V',
                   'Classification_6V', 'Classification_7V', 'Classification_9V']
    variables = [col for col in variables if col not in exclude_cols]

    X = data[variables].dropna()

    if verbose:
        print(f"📊 Datos preparados: {X.shape[0]} muestras, {X.shape[1]} variables")

    # Escalado
    X_scaled, scaler_info = aplicar_escalado(X, escalado)

    if k_range is None:
        k_range = range(2, min(10, len(X) // 5))

    resultados_kmeans = {}
    inercias = []
    silhouette_scores = []

    # Evaluar diferentes valores de K
    for k in k_range:
        if verbose:
            print(f"   Evaluando K={k}...")

        try:
            # K-Means manual
            kmeans_result = manual_kmeans(X_scaled.values, k, random_state=random_state)
            labels = kmeans_result['labels']

            # Métricas
            silhouette = manual_silhouette_score(X_scaled.values, labels)

            # Análisis de clusters
            cluster_analysis = analizar_clusters_manual(X, labels, variables)

            resultado_k = {
                'k': k,
                'labels': labels.tolist(),
                'centroids': kmeans_result['centroids'].tolist(),
                'inertia': float(kmeans_result['inertia']),
                'silhouette_score': float(silhouette),
                'n_iter': int(kmeans_result['n_iter']),
                'cluster_analysis': cluster_analysis
            }

            resultados_kmeans[k] = resultado_k
            inercias.append(resultado_k['inertia'])
            silhouette_scores.append(resultado_k['silhouette_score'])

        except Exception as e:
            if verbose:
                print(f"Error con K={k}: {e}")
            continue

    # Determinar K óptimo
    if silhouette_scores:
        k_optimo = list(k_range)[np.argmax(silhouette_scores)]
    else:
        k_optimo = list(k_range)[0] if k_range else 3

    if verbose:
        print(f"✅ K-Means completado. K recomendado: {k_optimo}")

    return {
        'tipo': 'kmeans_optimizado',
        'variables_utilizadas': variables,
        'k_range': list(k_range),
        'resultados_por_k': resultados_kmeans,
        'inercias': inercias,
        'silhouette_scores': silhouette_scores,
        'recomendacion_k': k_optimo,
        'datos_originales': X,
        'scaler_info': scaler_info,
        'recomendaciones': [
            f"K óptimo recomendado: {k_optimo}",
            f"Silhouette Score: {resultados_kmeans.get(k_optimo, {}).get('silhouette_score', 0):.3f}"
        ]
    }

def dbscan_optimizado(data, variables=None, optimizar_parametros=True,
                      escalado='standard', verbose=True):
    """DBSCAN optimizado con búsqueda de parámetros"""
    if verbose:
        print("🔍 Iniciando DBSCAN optimizado...")

    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()

    # Excluir columnas no relevantes
    exclude_cols = ['Points', 'Sampling_date', 'WQI_IDEAM_6V', 'WQI_IDEAM_7V', 'WQI_NSF_9V',
                   'Classification_6V', 'Classification_7V', 'Classification_9V']
    variables = [col for col in variables if col not in exclude_cols]

    X = data[variables].dropna()

    if verbose:
        print(f"📊 Datos preparados: {X.shape[0]} muestras, {X.shape[1]} variables")

    # Escalado
    X_scaled, scaler_info = aplicar_escalado(X, escalado)

    mejores_resultados = []

    if optimizar_parametros:
        # Rango de parámetros
        eps_range = np.linspace(0.1, 2.0, 10)
        min_samples_range = range(2, min(10, len(X) // 10))

        for eps in eps_range:
            for min_samples in min_samples_range:
                try:
                    labels = manual_dbscan(X_scaled.values, eps=eps, min_samples=min_samples)

                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    noise_ratio = n_noise / len(labels)

                    if n_clusters < 1 or noise_ratio > 0.8:
                        continue

                    # Calcular silhouette solo para puntos no outliers
                    if n_clusters > 1:
                        non_noise_mask = labels != -1
                        if np.sum(non_noise_mask) > 1:
                            silhouette = manual_silhouette_score(
                                X_scaled.values[non_noise_mask],
                                labels[non_noise_mask]
                            )
                        else:
                            silhouette = 0.5
                    else:
                        silhouette = 0.5

                    resultado = {
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'noise_ratio': noise_ratio,
                        'silhouette_score': silhouette,
                        'labels': labels.tolist(),
                        'score_compuesto': silhouette * (1 - noise_ratio)
                    }

                    mejores_resultados.append(resultado)

                except Exception as e:
                    continue

    # Usar configuración por defecto si no hay resultados
    if not mejores_resultados:
        labels = manual_dbscan(X_scaled.values, eps=0.5, min_samples=3)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        resultado = {
            'eps': 0.5,
            'min_samples': 3,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': n_noise / len(labels),
            'silhouette_score': 0.5,
            'labels': labels.tolist(),
            'score_compuesto': 0.5
        }
        mejores_resultados.append(resultado)

    # Ordenar por score compuesto
    mejores_resultados.sort(key=lambda x: x['score_compuesto'], reverse=True)
    mejor_resultado = mejores_resultados[0]

    if verbose:
        print(f"✅ DBSCAN completado: {mejor_resultado['n_clusters']} clusters, {mejor_resultado['n_noise']} outliers")

    return {
        'tipo': 'dbscan_optimizado',
        'variables_utilizadas': variables,
        'mejor_configuracion': mejor_resultado,
        'todas_configuraciones': mejores_resultados[:10],
        'datos_originales': X,
        'scaler_info': scaler_info,
        'recomendaciones': [
            f"DBSCAN detectó {mejor_resultado['n_clusters']} clusters",
            f"Outliers: {mejor_resultado['n_noise']} ({mejor_resultado['noise_ratio']:.1%})",
            f"Silhouette Score: {mejor_resultado['silhouette_score']:.3f}"
        ]
    }

# ==================== FUNCIONES AUXILIARES PRINCIPALES ====================

def calcular_estadisticas_avanzadas(X, variables):
    """Calcular estadísticas descriptivas avanzadas"""
    estadisticas = {}

    for variable in variables:
        if variable not in X.columns:
            continue

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

        # Estadísticas de forma
        mean_val = serie.mean()
        std_val = serie.std()

        if std_val > 0:
            skew_val = ((serie - mean_val) ** 3).mean() / (std_val ** 3)
            kurt_val = ((serie - mean_val) ** 4).mean() / (std_val ** 4) - 3
            cv = std_val / mean_val if mean_val != 0 else np.inf
        else:
            skew_val = 0
            kurt_val = 0
            cv = 0

        stats_forma = {
            'skewness': float(skew_val),
            'kurtosis': float(kurt_val),
            'cv': float(cv)
        }

        # Test de normalidad simplificado
        normalidad = {
            'es_normal': abs(skew_val) < 1 and abs(kurt_val) < 1
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
    corr_pearson = X[variables].corr()

    # Correlaciones significativas
    correlaciones_fuertes = []
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            if i < len(variables) and j < len(variables):
                var1, var2 = variables[i], variables[j]
                if var1 in corr_pearson.columns and var2 in corr_pearson.columns:
                    corr_val = corr_pearson.loc[var1, var2]

                    if not np.isnan(corr_val) and abs(corr_val) > 0.6:
                        correlaciones_fuertes.append({
                            'variable_1': var1,
                            'variable_2': var2,
                            'correlacion': float(corr_val),
                            'tipo': 'Fuerte positiva' if corr_val > 0.6 else 'Fuerte negativa'
                        })

    # Análisis de multicolinealidad
    try:
        condicion = np.linalg.cond(corr_pearson.values)
        multicolinealidad = 'Alta' if condicion > 1000 else 'Media' if condicion > 100 else 'Baja'
    except:
        condicion = 0
        multicolinealidad = 'No determinada'

    return {
        'matriz_pearson': corr_pearson.to_dict(),
        'correlaciones_fuertes': correlaciones_fuertes,
        'numero_condicion': float(condicion),
        'multicolinealidad': multicolinealidad
    }

def detectar_outliers_multiples_metodos(X, variables):
    """Detección de outliers usando múltiples métodos"""
    outliers_por_metodo = {}

    X_vals = X[variables].values

    # 1. Z-Score
    outliers_zscore = detectar_outliers_zscore(X_vals)
    outliers_zscore_indices = X[outliers_zscore].index.tolist()

    # 2. IQR
    outliers_iqr = detectar_outliers_iqr(X_vals)
    outliers_iqr_indices = X[outliers_iqr].index.tolist()

    # 3. Isolation Forest simplificado
    outliers_isolation = detectar_outliers_isolation_forest_simple(X_vals)
    outliers_isolation_indices = X[outliers_isolation].index.tolist()

    # Consolidar resultados
    todos_outliers = set(outliers_zscore_indices + outliers_iqr_indices + outliers_isolation_indices)

    return {
        'zscore': {
            'indices': outliers_zscore_indices,
            'total': len(outliers_zscore_indices)
        },
        'iqr': {
            'indices': outliers_iqr_indices,
            'total': len(outliers_iqr_indices)
        },
        'isolation_forest': {
            'indices': outliers_isolation_indices,
            'total': len(outliers_isolation_indices)
        },
        'consenso': {
            'indices_unicos': list(todos_outliers),
            'total_unico': len(todos_outliers),
            'porcentaje': float(len(todos_outliers) / len(X) * 100)
        }
    }

def analizar_distribuciones_avanzado(X, variables):
    """Análisis de distribuciones"""
    distribuciones = {}

    for variable in variables:
        if variable not in X.columns:
            continue

        serie = X[variable]

        # Estadísticas de forma
        mean_val = serie.mean()
        std_val = serie.std()

        if std_val > 0:
            skew_val = ((serie - mean_val) ** 3).mean() / (std_val ** 3)
            kurt_val = ((serie - mean_val) ** 4).mean() / (std_val ** 4) - 3
        else:
            skew_val = 0
            kurt_val = 0

        # Clasificar distribución
        if abs(skew_val) < 0.5 and abs(kurt_val) < 0.5:
            distribucion_tipo = 'normal'
            score = 1.0 - (abs(skew_val) + abs(kurt_val)) / 2
        elif skew_val > 1.5:
            distribucion_tipo = 'exponential'
            score = min(1.0, skew_val / 2)
        elif kurt_val < -1:
            distribucion_tipo = 'uniform'
            score = min(1.0, abs(kurt_val) / 2)
        else:
            distribucion_tipo = 'normal'
            score = 0.5

        distribuciones[variable] = {
            'mejor_ajuste': {
                'distribucion': distribucion_tipo,
                'score': score,
                'parametros': [mean_val, std_val]
            },
            'es_aproximadamente_normal': distribucion_tipo == 'normal',
            'skewness': float(skew_val),
            'kurtosis': float(kurt_val)
        }

    return distribuciones

def clustering_exploratorio_rapido(X, variables, escalado='standard'):
    """Clustering exploratorio rápido"""
    # Escalado
    X_scaled, _ = aplicar_escalado(X[variables], escalado)

    resultados_rapidos = {}

    for k in [3, 4, 5]:
        try:
            kmeans_result = manual_kmeans(X_scaled.values, k)
            labels = kmeans_result['labels']

            silhouette = manual_silhouette_score(X_scaled.values, labels)

            resultados_rapidos[k] = {
                'silhouette_score': float(silhouette),
                'inercia': float(kmeans_result['inertia']),
                'labels': labels.tolist()
            }
        except:
            continue

    # Mejor k rápido
    if resultados_rapidos:
        mejor_k_rapido = max(resultados_rapidos.keys(),
                           key=lambda k: resultados_rapidos[k]['silhouette_score'])
    else:
        mejor_k_rapido = 3

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
    missing_ratio = total_missing / (len(data_original) * len(variables)) if variables else 0
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

def generar_recomendaciones_exploratorio(estadisticas, correlaciones, outliers, distribuciones, calidad):
    """Generar recomendaciones basadas en el análisis exploratorio"""
    recomendaciones = []

    # Calidad de datos
    if calidad['quality_score'] < 85:
        recomendaciones.append("Considere limpiar los datos antes del análisis")

    # Correlaciones
    if 'correlaciones_fuertes' in correlaciones:
        n_corr = len(correlaciones['correlaciones_fuertes'])
        if n_corr > 0:
            recomendaciones.append(f"Se detectaron {n_corr} correlaciones fuertes")

    if correlaciones.get('multicolinealidad') == 'Alta':
        recomendaciones.append("Alta multicolinealidad detectada")

    # Outliers
    if 'consenso' in outliers:
        porcentaje_outliers = outliers['consenso']['porcentaje']
        if porcentaje_outliers > 10:
            recomendaciones.append(f"Alto porcentaje de outliers ({porcentaje_outliers:.1f}%)")
        elif porcentaje_outliers > 5:
            recomendaciones.append("Outliers moderados detectados")

    # Distribuciones
    if distribuciones:
        variables_no_normales = sum(1 for var_info in distribuciones.values()
                                  if not var_info.get('es_aproximadamente_normal', False))

        if variables_no_normales > len(distribuciones) * 0.7:
            recomendaciones.append("Muchas variables no siguen distribución normal")

    # Técnicas ML
    n_variables = len(estadisticas)
    if n_variables > 10:
        recomendaciones.append("Alto número de variables. PCA recomendado")

    if outliers and outliers.get('consenso', {}).get('porcentaje', 0) > 5:
        recomendaciones.append("Considere DBSCAN para clustering con outliers")
    else:
        recomendaciones.append("K-Means puede ser apropiado para clustering")

    return recomendaciones

def analisis_exploratorio_completo(data, variables=None, escalado='standard',
                                 handle_outliers=True, verbose=True):
    """Análisis exploratorio exhaustivo de los datos"""
    if verbose:
        print("🔍 Iniciando análisis exploratorio completo...")

    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()

    # Excluir columnas no relevantes
    exclude_cols = ['Points', 'Sampling_date', 'WQI_IDEAM_6V', 'WQI_IDEAM_7V', 'WQI_NSF_9V',
                   'Classification_6V', 'Classification_7V', 'Classification_9V']
    variables = [col for col in variables if col not in exclude_cols]

    X = data[variables].dropna()

    if verbose:
        print(f"📊 Datos preparados: {X.shape[0]} muestras, {X.shape[1]} variables")

    # 1. Estadísticas descriptivas
    estadisticas = calcular_estadisticas_avanzadas(X, variables)

    # 2. Análisis de correlaciones
    correlaciones = analizar_correlaciones_avanzado(X, variables)

    # 3. Detección de outliers
    outliers = detectar_outliers_multiples_metodos(X, variables) if handle_outliers else {}

    # 4. Análisis de distribuciones
    distribuciones = analizar_distribuciones_avanzado(X, variables)

    # 5. PCA exploratorio básico
    X_scaled, _ = aplicar_escalado(X, escalado)
    pca_basico = manual_pca(X_scaled.values, n_components=min(5, len(variables)))

    # 6. Clustering exploratorio rápido
    clustering_exploratorio = clustering_exploratorio_rapido(X, variables, escalado)

    # 7. Análisis de calidad de datos
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
            'varianza_explicada': pca_basico['explained_variance_ratio'].tolist(),
            'varianza_acumulada': np.cumsum(pca_basico['explained_variance_ratio']).tolist()
        },
        'clustering_exploratorio': clustering_exploratorio,
        'calidad_datos': calidad_datos,
        'datos_originales': X,
        'recomendaciones': recomendaciones
    }

def analizar_pca_detallado(pca_result, variables, varianza_objetivo):
    """Análisis detallado del PCA"""
    varianza_explicada = pca_result['explained_variance_ratio']
    varianza_acumulada = np.cumsum(varianza_explicada)

    # Encontrar número de componentes para objetivo
    n_componentes_objetivo = np.argmax(varianza_acumulada >= varianza_objetivo) + 1

    # Análisis de cada componente
    componentes_info = []
    for i in range(len(varianza_explicada)):
        loadings = pca_result['components'][i]

        # Variables más importantes
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
        'eigenvalues': pca_result['explained_variance'].tolist(),
        'n_componentes_objetivo': n_componentes_objetivo,
        'varianza_objetivo': varianza_objetivo,
        'componentes_info': componentes_info,
        'matriz_componentes': pca_result['components'].tolist()
    }

def pca_completo_avanzado(data, variables=None, explicar_varianza_objetivo=0.95,
                         escalado='standard', verbose=True):
    """Análisis de Componentes Principales avanzado"""
    if verbose:
        print("🔍 Iniciando PCA avanzado...")

    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()

    # Excluir columnas no relevantes
    exclude_cols = ['Points', 'Sampling_date', 'WQI_IDEAM_6V', 'WQI_IDEAM_7V', 'WQI_NSF_9V',
                   'Classification_6V', 'Classification_7V', 'Classification_9V']
    variables = [col for col in variables if col not in exclude_cols]

    X = data[variables].dropna()

    if verbose:
        print(f"📊 Datos preparados: {X.shape[0]} muestras, {X.shape[1]} variables")

    # Escalado
    X_scaled, scaler_info = aplicar_escalado(X, escalado)

    # PCA manual
    pca_result = manual_pca(X_scaled.values)

    # Análisis detallado
    analisis = analizar_pca_detallado(pca_result, variables, explicar_varianza_objetivo)

    if verbose:
        print(f"✅ PCA completado: {analisis['n_componentes_objetivo']} componentes recomendados")

    return {
        'tipo': 'pca_completo_avanzado',
        'variables_utilizadas': variables,
        'n_muestras': len(X),
        'resultados_por_metodo': {
            'linear': {
                'modelo_info': pca_result,
                'transformacion': pca_result['X_transformed'].tolist(),
                'analisis': analisis,
                'componentes_recomendados': analisis['n_componentes_objetivo']
            }
        },
        'datos_originales': X,
        'scaler_info': scaler_info,
        'recomendaciones': [
            f"Componentes recomendados: {analisis['n_componentes_objetivo']}",
            f"Varianza explicada: {analisis['varianza_acumulada'][analisis['n_componentes_objetivo']-1]:.1%}",
            f"Primera componente explica: {analisis['varianza_explicada'][0]:.1%}"
        ]
    }

# ==================== CLUSTERING JERÁRQUICO MANUAL ====================

def manual_hierarchical_clustering(X, method='ward', metric='euclidean'):
    """Implementación manual de clustering jerárquico"""
    n_samples = len(X)

    # Calcular matriz de distancias inicial
    dist_matrix = np.full((n_samples, n_samples), np.inf)
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if metric == 'euclidean':
                dist = np.linalg.norm(X[i] - X[j])
            elif metric == 'manhattan':
                dist = np.sum(np.abs(X[i] - X[j]))
            elif metric == 'cosine':
                dot_product = np.dot(X[i], X[j])
                norm_i = np.linalg.norm(X[i])
                norm_j = np.linalg.norm(X[j])
                if norm_i > 0 and norm_j > 0:
                    dist = 1 - (dot_product / (norm_i * norm_j))
                else:
                    dist = 1
            else:
                dist = np.linalg.norm(X[i] - X[j])

            dist_matrix[i, j] = dist_matrix[j, i] = dist

    # Inicializar clusters - cada punto es un cluster
    clusters = {i: [i] for i in range(n_samples)}
    cluster_centroids = {i: X[i].copy() for i in range(n_samples)}
    cluster_sizes = {i: 1 for i in range(n_samples)}

    # Matriz de linkage para el dendrograma
    linkage_matrix = []
    current_cluster_id = n_samples

    while len(clusters) > 1:
        # Encontrar los dos clusters más cercanos
        min_dist = np.inf
        merge_pair = None

        active_clusters = list(clusters.keys())

        for i in range(len(active_clusters)):
            for j in range(i + 1, len(active_clusters)):
                ci, cj = active_clusters[i], active_clusters[j]

                if method == 'ward':
                    # Ward: minimizar incremento en varianza
                    ni, nj = cluster_sizes[ci], cluster_sizes[cj]
                    centroid_i = cluster_centroids[ci]
                    centroid_j = cluster_centroids[cj]
                    dist = np.sqrt(((ni * nj) / (ni + nj)) * np.sum((centroid_i - centroid_j) ** 2))

                elif method == 'complete':
                    # Complete linkage: máxima distancia entre puntos
                    max_dist = 0
                    for pi in clusters[ci]:
                        for pj in clusters[cj]:
                            max_dist = max(max_dist, dist_matrix[pi, pj])
                    dist = max_dist

                elif method == 'average':
                    # Average linkage: distancia promedio entre puntos
                    total_dist = 0
                    count = 0
                    for pi in clusters[ci]:
                        for pj in clusters[cj]:
                            total_dist += dist_matrix[pi, pj]
                            count += 1
                    dist = total_dist / count if count > 0 else 0

                elif method == 'single':
                    # Single linkage: mínima distancia entre puntos
                    min_dist_single = np.inf
                    for pi in clusters[ci]:
                        for pj in clusters[cj]:
                            min_dist_single = min(min_dist_single, dist_matrix[pi, pj])
                    dist = min_dist_single

                else:
                    # Default a average
                    total_dist = 0
                    count = 0
                    for pi in clusters[ci]:
                        for pj in clusters[cj]:
                            total_dist += dist_matrix[pi, pj]
                            count += 1
                    dist = total_dist / count if count > 0 else 0

                if dist < min_dist:
                    min_dist = dist
                    merge_pair = (ci, cj)

        if merge_pair is None:
            break

        # Unir los clusters seleccionados
        ci, cj = merge_pair
        new_cluster = clusters[ci] + clusters[cj]
        new_size = cluster_sizes[ci] + cluster_sizes[cj]

        # Calcular nuevo centroide (promedio ponderado por tamaño)
        if method == 'ward':
            new_centroid = (cluster_sizes[ci] * cluster_centroids[ci] +
                          cluster_sizes[cj] * cluster_centroids[cj]) / new_size
        else:
            # Para otros métodos, usar promedio simple de todos los puntos
            all_points = np.array([X[i] for i in new_cluster])
            new_centroid = np.mean(all_points, axis=0)

        # Guardar en linkage matrix: [cluster1_id, cluster2_id, distancia, tamaño_nuevo_cluster]
        linkage_matrix.append([ci, cj, min_dist, new_size])

        # Actualizar estructuras
        clusters[current_cluster_id] = new_cluster
        cluster_centroids[current_cluster_id] = new_centroid
        cluster_sizes[current_cluster_id] = new_size

        # Eliminar clusters viejos
        del clusters[ci]
        del clusters[cj]
        del cluster_centroids[ci]
        del cluster_centroids[cj]
        del cluster_sizes[ci]
        del cluster_sizes[cj]

        current_cluster_id += 1

    return np.array(linkage_matrix)


def plot_dendrogram_manual_mejorado(linkage_matrix, labels=None, ax=None,
                                    max_display=30, color_threshold=None):
    """Crear dendrograma manual mejorado"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    n_samples = len(linkage_matrix) + 1

    if labels is None:
        labels = [f"S{i}" for i in range(n_samples)]

    # Submuestreo si hay muchas muestras
    display_labels = labels
    n_display = n_samples

    if n_samples > max_display:
        step = max(1, n_samples // max_display)
        sample_indices = list(range(0, n_samples, step))[:max_display]
        display_labels = [labels[i] for i in sample_indices if i < len(labels)]
        n_display = len(display_labels)

        ax.text(0.5, 0.95,
                f'Mostrando {n_display} de {n_samples} muestras (cada {step})',
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                fontsize=9)

    # Estructura para rastrear posiciones
    node_positions = {i: i for i in range(n_display)}
    node_heights = {i: 0 for i in range(n_display)}

    # Determinar umbral de color
    if color_threshold is None and len(linkage_matrix) > 0:
        heights = linkage_matrix[:, 2]
        color_threshold = 0.7 * np.max(heights)

    # Colores
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    color_idx = 0
    cluster_colors = {}

    # Procesar fusiones
    for i, (idx1, idx2, height, size) in enumerate(linkage_matrix):
        idx1, idx2 = int(idx1), int(idx2)
        current_node = n_display + i

        # Obtener posiciones con validación
        x1 = node_positions.get(idx1, idx1 if idx1 < n_display else n_display / 2)
        x2 = node_positions.get(idx2, idx2 if idx2 < n_display else n_display / 2)

        y1 = node_heights.get(idx1, 0)
        y2 = node_heights.get(idx2, 0)

        x_new = (x1 + x2) / 2

        node_positions[current_node] = x_new
        node_heights[current_node] = height

        # Determinar color
        if color_threshold and height > color_threshold:
            color = colors[color_idx % len(colors)]
            color_idx += 1
        else:
            parent_color = cluster_colors.get(idx1) or cluster_colors.get(idx2)
            color = parent_color if parent_color is not None else 'blue'

        cluster_colors[current_node] = color

        # Dibujar líneas
        ax.plot([x1, x1], [y1, height], color=color, linewidth=1.5, alpha=0.8)
        ax.plot([x2, x2], [y2, height], color=color, linewidth=1.5, alpha=0.8)
        ax.plot([x1, x2], [height, height], color=color, linewidth=2, alpha=0.8)

    # Configurar ejes
    ax.set_xticks(range(n_display))
    ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Distancia de Fusión', fontsize=11)
    ax.set_title('Dendrograma de Clustering Jerárquico', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Líneas de corte sugeridas
    if len(linkage_matrix) > 0:
        heights = linkage_matrix[:, 2]
        max_height = np.max(heights)

        for n_clusters in [2, 3, 4, 5]:
            if n_clusters < len(linkage_matrix):
                cut_height = heights[-(n_clusters)]

                ax.axhline(y=cut_height, color='red', linestyle='--',
                           alpha=0.4, linewidth=1, zorder=0)

                ax.text(n_display * 0.98, cut_height, f' {n_clusters}',
                        fontsize=8, va='center', ha='right',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', alpha=0.8, edgecolor='red'))

        ax.set_xlim(-0.5, n_display - 0.5)
        ax.set_ylim(0, max_height * 1.1)

    return ax

def get_clusters_from_linkage(linkage_matrix, n_clusters):
    """Obtener asignación de clusters cortando el dendrograma"""
    if n_clusters <= 1:
        return [0] * (len(linkage_matrix) + 1)

    n_samples = len(linkage_matrix) + 1

    # Si pedimos más clusters que muestras, cada muestra es su propio cluster
    if n_clusters >= n_samples:
        return list(range(n_samples))

    # Encontrar la altura de corte para obtener n_clusters
    if n_clusters - 1 >= len(linkage_matrix):
        cut_height = 0
    else:
        cut_height = linkage_matrix[-(n_clusters-1), 2]

    # Simular el proceso de clustering hasta la altura de corte
    clusters = {i: [i] for i in range(n_samples)}
    current_cluster_id = n_samples

    for i, (c1, c2, height, size) in enumerate(linkage_matrix):
        if height > cut_height:
            break

        c1, c2 = int(c1), int(c2)

        # Unir clusters
        if c1 in clusters and c2 in clusters:
            new_cluster = clusters[c1] + clusters[c2]
            clusters[current_cluster_id] = new_cluster
            del clusters[c1]
            del clusters[c2]
            current_cluster_id += 1

    # Crear array de etiquetas
    labels = np.zeros(n_samples, dtype=int)
    cluster_id = 0

    for cluster_points in clusters.values():
        for point in cluster_points:
            labels[point] = cluster_id
        cluster_id += 1

    return labels.tolist()

def clustering_jerarquico_completo(data, variables=None, metodos=['ward'],
                                 metricas=['euclidean'], max_clusters=10,
                                 escalado='standard', verbose=True):
    """Clustering jerárquico completo con dendrograma manual"""
    if verbose:
        print("🔍 Iniciando clustering jerárquico...")

    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()

    # Excluir columnas no relevantes
    exclude_cols = ['Points', 'Sampling_date', 'WQI_IDEAM_6V', 'WQI_IDEAM_7V', 'WQI_NSF_9V',
                   'Classification_6V', 'Classification_7V', 'Classification_9V']
    variables = [col for col in variables if col not in exclude_cols]

    X = data[variables].dropna()

    if verbose:
        print(f"📊 Datos preparados: {X.shape[0]} muestras, {X.shape[1]} variables")

    # Escalado
    X_scaled, scaler_info = aplicar_escalado(X, escalado)

    # Realizar clustering jerárquico manual
    metodo = metodos[0] if metodos else 'ward'
    metrica = metricas[0] if metricas else 'euclidean'

    if verbose:
        print(f"   Ejecutando clustering jerárquico: {metodo}-{metrica}")

    linkage_matrix = manual_hierarchical_clustering(X_scaled.values, method=metodo, metric=metrica)

    # Evaluar diferentes números de clusters
    resultados_por_k = {}
    max_k = min(max_clusters, len(X) - 1)

    for k in range(2, max_k + 1):
        try:
            # Obtener labels cortando el dendrograma
            labels = get_clusters_from_linkage(linkage_matrix, k)

            # Calcular silhouette score
            if len(set(labels)) > 1:
                silhouette = manual_silhouette_score(X_scaled.values, np.array(labels))
            else:
                silhouette = 0.0

            # Análisis de clusters
            cluster_analysis = analizar_clusters_manual(X, labels, variables)

            resultados_por_k[k] = {
                'silhouette_score': float(silhouette),
                'labels': labels,
                'cluster_stats': cluster_analysis
            }

        except Exception as e:
            if verbose:
                print(f"Error evaluando k={k}: {e}")
            continue

    # Encontrar mejor configuración
    if resultados_por_k:
        mejor_k = max(resultados_por_k.keys(),
                     key=lambda k: resultados_por_k[k]['silhouette_score'])
        mejor_silhouette = resultados_por_k[mejor_k]['silhouette_score']
        mejor_labels = resultados_por_k[mejor_k]['labels']
    else:
        mejor_k = 3
        mejor_silhouette = 0.5
        mejor_labels = [0] * len(X)

    mejor_configuracion = {
        'metodo': metodo,
        'metrica': metrica,
        'n_clusters_sugeridos': mejor_k,
        'silhouette_score': mejor_silhouette,
        'labels': mejor_labels
    }

    if verbose:
        print(f"✅ Clustering jerárquico completado. Mejor K: {mejor_k}")

    return {
        'tipo': 'clustering_jerarquico_completo',
        'variables_utilizadas': variables,
        'metodo_escalado': escalado,
        'linkage_matrix': linkage_matrix.tolist(),  # Matriz de linkage para dendrograma
        'resultados_por_k': resultados_por_k,
        'mejor_configuracion': mejor_configuracion,
        'datos_originales': X,
        'scaler_info': scaler_info,
        'sample_labels': [f"S{i}" for i in range(len(X))],  # Etiquetas para dendrograma
        'recomendaciones': [
            f"Mejor configuración: {metodo}-{metrica} con {mejor_k} clusters",
            f"Silhouette Score: {mejor_silhouette:.3f}",
            f"Método de linkage: {metodo}"
        ]
    }


# ==================== PROCESAMIENTO PARALELO ====================

class ParallelHierarchicalClustering:
    """Clustering jerárquico con procesamiento paralelo y monitoreo"""

    def __init__(self, max_workers=None):
        if max_workers is None:
            total_cores = os.cpu_count() or 4
            self.max_workers = max(2, min(total_cores - 2, int(total_cores * 0.75)))
            print(f"Sistema detectado: {total_cores} cores disponibles")
            print(f"Usando {self.max_workers} hilos para procesamiento paralelo")
        else:
            self.max_workers = max_workers

        self._lock = Lock()
        self.progress_callback = None
        self.status_callback = None
        self.thread_monitor_callback = None  # NUEVO
        self.active_threads = set()  # NUEVO

    def set_callbacks(self, progress_callback=None, status_callback=None, thread_monitor_callback=None):
        """Establecer callbacks incluyendo monitor de hilos"""
        self.progress_callback = progress_callback
        self.status_callback = status_callback
        self.thread_monitor_callback = thread_monitor_callback  # NUEVO

    def _emit_thread_activity(self, thread_id, active):
        """Emitir actividad de hilo (thread-safe)"""
        if self.thread_monitor_callback:
            try:
                with self._lock:
                    if active:
                        self.active_threads.add(thread_id)
                    else:
                        self.active_threads.discard(thread_id)
                    self.thread_monitor_callback(list(self.active_threads))
            except:
                pass

    def calculate_distance_matrix_parallel(self, X, metric='euclidean'):
        """Calcular matriz de distancias con monitoreo de hilos"""
        n_samples = len(X)
        dist_matrix = np.zeros((n_samples, n_samples))

        self._emit_status(f"Calculando distancias ({self.max_workers} hilos)...")

        def compute_row_distances(i):
            """Calcular distancias para una fila"""
            # Marcar hilo como activo
            thread_id = i % self.max_workers
            self._emit_thread_activity(thread_id, True)

            row_distances = np.zeros(n_samples)

            for j in range(i + 1, n_samples):
                if metric == 'euclidean':
                    dist = np.linalg.norm(X[i] - X[j])
                elif metric == 'manhattan':
                    dist = np.sum(np.abs(X[i] - X[j]))
                elif metric == 'cosine':
                    dot_product = np.dot(X[i], X[j])
                    norm_i = np.linalg.norm(X[i])
                    norm_j = np.linalg.norm(X[j])
                    if norm_i > 0 and norm_j > 0:
                        dist = 1 - (dot_product / (norm_i * norm_j))
                    else:
                        dist = 1
                else:
                    dist = np.linalg.norm(X[i] - X[j])

                row_distances[j] = dist

            # Marcar hilo como inactivo
            self._emit_thread_activity(thread_id, False)

            return i, row_distances

        use_parallel = n_samples > 100

        if use_parallel:
            self._emit_status(f"Procesamiento paralelo: {n_samples} filas en {self.max_workers} hilos")

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(compute_row_distances, i): i
                           for i in range(n_samples)}

                completed = 0
                for future in as_completed(futures):
                    i, row_distances = future.result()

                    with self._lock:
                        dist_matrix[i, :] = row_distances
                        dist_matrix[:, i] = row_distances

                        completed += 1
                        progress = int((completed / n_samples) * 30)
                        self._emit_progress(progress)

                        if completed % max(1, n_samples // 10) == 0:
                            self._emit_status(f"Distancias: {completed}/{n_samples} filas completadas")
        else:
            self._emit_status("Procesamiento secuencial (dataset pequeño)")
            for i in range(n_samples):
                _, row_distances = compute_row_distances(i)
                dist_matrix[i, :] = row_distances
                dist_matrix[:, i] = row_distances

                progress = int(((i + 1) / n_samples) * 30)
                self._emit_progress(progress)

        # Limpiar hilos activos
        self.active_threads.clear()
        self._emit_thread_activity(0, False)

        self._emit_status("Matriz de distancias completada")
        return dist_matrix


    def hierarchical_clustering_optimized(self, X, method='ward', metric='euclidean'):
        """Clustering jerárquico optimizado"""
        n_samples = len(X)

        self._emit_status(f"Clustering jerárquico ({method}-{metric})...")
        self._emit_progress(0)

        # Calcular distancias en paralelo
        dist_matrix = self.calculate_distance_matrix_parallel(X, metric)
        dist_matrix = np.where(np.isinf(dist_matrix), 1e10, dist_matrix)

        self._emit_progress(30)
        self._emit_status("Construyendo jerarquía...")

        # Inicializar
        clusters = {i: [i] for i in range(n_samples)}
        cluster_centroids = {i: X[i].copy() for i in range(n_samples)}
        cluster_sizes = {i: 1 for i in range(n_samples)}

        linkage_matrix = []
        current_cluster_id = n_samples
        total_merges = n_samples - 1
        merges_done = 0

        # Proceso de fusión
        while len(clusters) > 1:
            min_dist = np.inf
            merge_pair = None
            active_clusters = list(clusters.keys())

            # Encontrar pares más cercanos
            for i in range(len(active_clusters)):
                for j in range(i + 1, len(active_clusters)):
                    ci, cj = active_clusters[i], active_clusters[j]

                    # Calcular distancia según método
                    if method == 'ward':
                        ni, nj = cluster_sizes[ci], cluster_sizes[cj]
                        centroid_i = cluster_centroids[ci]
                        centroid_j = cluster_centroids[cj]
                        dist = np.sqrt(((ni * nj) / (ni + nj)) *
                                       np.sum((centroid_i - centroid_j) ** 2))
                    elif method == 'complete':
                        max_dist = 0
                        for pi in clusters[ci]:
                            for pj in clusters[cj]:
                                max_dist = max(max_dist, dist_matrix[pi, pj])
                        dist = max_dist
                    elif method == 'average':
                        total_dist = sum(dist_matrix[pi, pj]
                                         for pi in clusters[ci]
                                         for pj in clusters[cj])
                        dist = total_dist / (len(clusters[ci]) * len(clusters[cj]))
                    elif method == 'single':
                        min_dist_single = min(dist_matrix[pi, pj]
                                              for pi in clusters[ci]
                                              for pj in clusters[cj])
                        dist = min_dist_single
                    else:
                        dist = 0

                    if dist < min_dist:
                        min_dist = dist
                        merge_pair = (ci, cj)

            if merge_pair is None:
                break

            # Fusionar
            ci, cj = merge_pair
            new_cluster = clusters[ci] + clusters[cj]
            new_size = cluster_sizes[ci] + cluster_sizes[cj]

            if method == 'ward':
                new_centroid = (cluster_sizes[ci] * cluster_centroids[ci] +
                                cluster_sizes[cj] * cluster_centroids[cj]) / new_size
            else:
                all_points = np.array([X[i] for i in new_cluster])
                new_centroid = np.mean(all_points, axis=0)

            linkage_matrix.append([ci, cj, min_dist, new_size])

            clusters[current_cluster_id] = new_cluster
            cluster_centroids[current_cluster_id] = new_centroid
            cluster_sizes[current_cluster_id] = new_size

            del clusters[ci], clusters[cj]
            del cluster_centroids[ci], cluster_centroids[cj]
            del cluster_sizes[ci], cluster_sizes[cj]

            current_cluster_id += 1
            merges_done += 1

            progress = 30 + int((merges_done / total_merges) * 60)
            self._emit_progress(progress)

            if merges_done % max(1, total_merges // 10) == 0:
                self._emit_status(f"Fusiones: {merges_done}/{total_merges}")

        self._emit_progress(100)
        self._emit_status("Clustering completado")

        return np.array(linkage_matrix)

# ==================== FUNCIONES DE VISUALIZACIÓN ====================

def _graficar_clusters_pca(ax, resultado):
    """Graficar clusters usando PCA para reducir dimensionalidad"""
    try:
        datos = resultado['datos_originales']

        # Obtener labels según el tipo
        if resultado['tipo'] == 'kmeans_optimizado':
            k_opt = resultado.get('recomendacion_k')
            if k_opt and k_opt in resultado['resultados_por_k']:
                labels = resultado['resultados_por_k'][k_opt]['labels']
            else:
                labels = [0] * len(datos)
        elif resultado['tipo'] == 'dbscan_optimizado':
            labels = resultado['mejor_configuracion']['labels']
        else:
            labels = [0] * len(datos)

        # Aplicar PCA para 2D
        X_scaled, _ = aplicar_escalado(datos, 'standard')
        pca_result = manual_pca(X_scaled.values, n_components=2)
        datos_2d = pca_result['X_transformed']

        # Graficar puntos coloreados por cluster
        unique_labels = set(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for i, (label, color) in enumerate(zip(unique_labels, colors)):
            mask = np.array(labels) == label

            if label == -1:  # Outliers en DBSCAN
                ax.scatter(datos_2d[mask, 0], datos_2d[mask, 1],
                          c='black', marker='x', s=100, alpha=0.8, label='Outliers')
            else:
                ax.scatter(datos_2d[mask, 0], datos_2d[mask, 1],
                          c=[color], label=f'Cluster {label}',
                          s=60, alpha=0.7, edgecolors='black', linewidth=0.5)

        var_exp = pca_result['explained_variance_ratio']
        ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)')
        ax.set_title('Clusters en Espacio PCA')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

    except Exception as e:
        ax.text(0.5, 0.5, f'Error en visualización: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)

def _crear_viz_kmeans(resultado, fig):
    """Visualización para K-Means"""
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Evaluación de K
    ax1 = fig.add_subplot(gs[0, 0])
    k_vals = list(resultado['resultados_por_k'].keys())
    silhouette_vals = [resultado['resultados_por_k'][k]['silhouette_score'] for k in k_vals]

    ax1.plot(k_vals, silhouette_vals, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Número de Clusters (K)')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Evaluación de K óptimo')
    ax1.grid(True, alpha=0.3)

    # Marcar K óptimo
    k_opt = resultado.get('recomendacion_k')
    if k_opt in resultado['resultados_por_k']:
        best_score = resultado['resultados_por_k'][k_opt]['silhouette_score']
        ax1.plot(k_opt, best_score, 'ro', markersize=12, label=f'K óptimo = {k_opt}')
        ax1.legend()

    # 2. Distribución por cluster
    ax2 = fig.add_subplot(gs[0, 1])
    if k_opt and k_opt in resultado['resultados_por_k']:
        labels = resultado['resultados_por_k'][k_opt]['labels']
        unique_labels = np.unique(labels)
        tamaños = [labels.count(label) for label in unique_labels]

        bars = ax2.bar(range(len(unique_labels)), tamaños, alpha=0.7)
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Número de Puntos')
        ax2.set_title('Distribución por Cluster')
        ax2.set_xticks(range(len(unique_labels)))
        ax2.set_xticklabels([f'C{label}' for label in unique_labels])

        # Añadir valores
        for bar, tamaño in zip(bars, tamaños):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{tamaño}', ha='center', va='bottom')

    # 3. Visualización 2D con PCA
    ax3 = fig.add_subplot(gs[1, :])
    _graficar_clusters_pca(ax3, resultado)

    plt.suptitle('Análisis K-Means', fontsize=16, fontweight='bold')
    return fig

def _crear_viz_dbscan(resultado, fig):
    """Visualización para DBSCAN"""
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    mejor_config = resultado['mejor_configuracion']

    # 1. Información de parámetros
    ax1 = fig.add_subplot(gs[0, 0])
    info_text = f"Parámetros DBSCAN:\n\n"
    info_text += f"Eps: {mejor_config['eps']:.3f}\n"
    info_text += f"Min Samples: {mejor_config['min_samples']}\n\n"
    info_text += f"Resultados:\n"
    info_text += f"Clusters: {mejor_config['n_clusters']}\n"
    info_text += f"Outliers: {mejor_config['n_noise']}\n"
    info_text += f"Silhouette: {mejor_config['silhouette_score']:.3f}"

    ax1.text(0.05, 0.95, info_text, fontsize=10, va='top', ha='left',
             transform=ax1.transAxes, family='monospace')
    ax1.set_title('Configuración DBSCAN')
    ax1.axis('off')

    # 2. Distribución clusters vs outliers
    ax2 = fig.add_subplot(gs[0, 1])
    labels = mejor_config['labels']
    unique_labels = [l for l in set(labels) if l != -1]
    n_outliers = labels.count(-1)

    cluster_counts = [labels.count(label) for label in unique_labels]
    cluster_names = [f'Cluster {label}' for label in unique_labels]

    if n_outliers > 0:
        cluster_counts.append(n_outliers)
        cluster_names.append('Outliers')

    colors = ['red' if name == 'Outliers' else 'skyblue' for name in cluster_names]
    bars = ax2.bar(cluster_names, cluster_counts, color=colors, alpha=0.7)
    ax2.set_ylabel('Número de Puntos')
    ax2.set_title('Distribución por Cluster')
    ax2.tick_params(axis='x', rotation=45)

    # 3. Visualización 2D con PCA
    ax3 = fig.add_subplot(gs[1, :])
    _graficar_clusters_pca(ax3, resultado)

    plt.suptitle('Análisis DBSCAN', fontsize=16, fontweight='bold')
    return fig

def crear_visualizacion_clustering_puntos_muestreo(resultado, figsize=(16, 12)):
    """Crear visualización para clustering"""
    tipo = resultado.get('tipo', '')
    fig = plt.figure(figsize=figsize)

    if tipo == 'kmeans_optimizado':
        return _crear_viz_kmeans(resultado, fig)
    elif tipo == 'dbscan_optimizado':
        return _crear_viz_dbscan(resultado, fig)
    else:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f'Visualización no implementada para {tipo}',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Visualización No Disponible')
        return fig

def generar_visualizaciones_ml_no_supervisado(resultado: Dict[str, Any],
                                            figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
    """Generar visualizaciones para ML No Supervisado"""
    tipo = resultado.get('tipo', '')

    try:
        if tipo in ['kmeans_optimizado', 'dbscan_optimizado']:
            return crear_visualizacion_clustering_puntos_muestreo(resultado, figsize)
        elif tipo == 'pca_completo_avanzado':
            return _crear_visualizacion_pca_puntos_muestreo(resultado, figsize)
        elif tipo == 'clustering_jerarquico_completo':
            return _crear_visualizacion_jerarquico_puntos_muestreo(resultado, figsize)
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

def _crear_visualizacion_generica(resultado, figsize):
    """Crear visualización genérica"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    tipo = resultado.get('tipo', 'desconocido')
    ax.text(0.5, 0.5, f'Visualización para: {tipo}\n\nResultados disponibles',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=14, bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax.set_title(f'Resultado: {tipo}')
    ax.axis('off')

    return fig

def _crear_visualizacion_pca_puntos_muestreo(resultado, figsize=(16, 12)):
    """Crear visualización para PCA"""
    fig = plt.figure(figsize=figsize)

    if 'linear' not in resultado.get('resultados_por_metodo', {}):
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No hay resultados de PCA para visualizar',
                ha='center', va='center', transform=ax.transAxes)
        return fig

    linear_result = resultado['resultados_por_metodo']['linear']
    analisis = linear_result['analisis']

    # Layout 2x2
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Varianza explicada
    ax1 = fig.add_subplot(gs[0, 0])
    var_exp = analisis['varianza_explicada']
    x = range(1, len(var_exp) + 1)
    bars = ax1.bar(x, [v * 100 for v in var_exp], alpha=0.7, color='steelblue')
    ax1.set_xlabel('Componente Principal')
    ax1.set_ylabel('Varianza Explicada (%)')
    ax1.set_title('Varianza por Componente')
    ax1.grid(True, alpha=0.3)

    # Añadir valores en las barras
    for i, (bar, val) in enumerate(zip(bars, var_exp)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9)

    # 2. Varianza acumulada
    ax2 = fig.add_subplot(gs[0, 1])
    var_acum = analisis['varianza_acumulada']
    ax2.plot(x, [v * 100 for v in var_acum], 'o-', linewidth=2, markersize=8, color='darkred')
    ax2.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='95%')
    ax2.axhline(y=85, color='orange', linestyle='--', alpha=0.7, label='85%')
    ax2.set_xlabel('Componente Principal')
    ax2.set_ylabel('Varianza Acumulada (%)')
    ax2.set_title('Varianza Acumulada')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    # 3. Contribuciones de variables (PC1 y PC2)
    ax3 = fig.add_subplot(gs[1, 0])
    componentes_info = analisis['componentes_info']
    if len(componentes_info) >= 2:
        pc1_info = componentes_info[0]
        pc2_info = componentes_info[1]

        # Tomar top 5 variables de cada componente
        top_vars_pc1 = pc1_info['top_variables'][:5]
        top_vars_pc2 = pc2_info['top_variables'][:5]

        # Combinar variables únicas
        all_vars = {}
        for var in top_vars_pc1:
            all_vars[var['variable']] = [var['loading'], 0]
        for var in top_vars_pc2:
            if var['variable'] in all_vars:
                all_vars[var['variable']][1] = var['loading']
            else:
                all_vars[var['variable']] = [0, var['loading']]

        if all_vars:
            variables = list(all_vars.keys())
            pc1_loadings = [all_vars[var][0] for var in variables]
            pc2_loadings = [all_vars[var][1] for var in variables]

            x_pos = range(len(variables))
            width = 0.35

            bars1 = ax3.bar([x - width/2 for x in x_pos], pc1_loadings, width,
                           label=f'PC1 ({var_exp[0]*100:.1f}%)', alpha=0.8, color='steelblue')
            bars2 = ax3.bar([x + width/2 for x in x_pos], pc2_loadings, width,
                           label=f'PC2 ({var_exp[1]*100:.1f}%)', alpha=0.8, color='coral')

            ax3.set_xlabel('Variables')
            ax3.set_ylabel('Loading')
            ax3.set_title('Loadings de Variables en PC1 y PC2')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(variables, rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='black', linewidth=0.8)

    # 4. Scree plot y información
    ax4 = fig.add_subplot(gs[1, 1])
    eigenvalues = analisis['eigenvalues']
    ax4.plot(x, eigenvalues, 'o-', linewidth=2, markersize=8, color='purple')
    ax4.set_xlabel('Componente Principal')
    ax4.set_ylabel('Eigenvalue')
    ax4.set_title('Scree Plot - Eigenvalues')
    ax4.grid(True, alpha=0.3)

    # Línea de Kaiser (eigenvalue = 1)
    if max(eigenvalues) > 1:
        ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Kaiser criterion (λ=1)')
        ax4.legend()

    # Información adicional como texto
    n_comp_rec = linear_result['componentes_recomendados']
    info_text = f"Componentes recomendados: {n_comp_rec}\n"
    info_text += f"Varianza objetivo: {analisis['varianza_objetivo']*100:.0f}%\n"
    info_text += f"Varianza explicada: {var_acum[n_comp_rec-1]*100:.1f}%\n\n"
    info_text += f"Variables originales: {len(resultado['variables_utilizadas'])}\n"
    info_text += f"Reducción dimensional: {len(resultado['variables_utilizadas'])} → {n_comp_rec}"

    ax4.text(0.02, 0.98, info_text, transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            verticalalignment='top', fontsize=9, family='monospace')

    plt.suptitle('Análisis de Componentes Principales (PCA)', fontsize=16, fontweight='bold')
    return fig


def _crear_visualizacion_jerarquico_puntos_muestreo(resultado, figsize=(16, 12)):
    """Crear visualización para clustering jerárquico con dendrograma manual"""
    fig = plt.figure(figsize=figsize)

    mejor_config = resultado.get('mejor_configuracion', {})
    datos = resultado.get('datos_originales')
    linkage_matrix = resultado.get('linkage_matrix')
    sample_labels = resultado.get('sample_labels', [])

    if datos is None or len(datos) == 0:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No hay datos para visualizar',
                ha='center', va='center', transform=ax.transAxes)
        return fig

    # Layout 2x2
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Información del método
    ax1 = fig.add_subplot(gs[0, 0])
    info_text = f"Clustering Jerárquico Manual\n\n"
    info_text += f"Método: {mejor_config.get('metodo', 'ward')}\n"
    info_text += f"Métrica: {mejor_config.get('metrica', 'euclidean')}\n"
    info_text += f"Clusters sugeridos: {mejor_config.get('n_clusters_sugeridos', 'N/A')}\n"
    info_text += f"Silhouette Score: {mejor_config.get('silhouette_score', 0):.3f}\n\n"
    info_text += f"Muestras analizadas: {len(datos)}\n"
    info_text += f"Variables: {len(resultado.get('variables_utilizadas', []))}\n\n"
    info_text += f"Implementación:\n• Numpy puro\n• Sin scipy\n• Compatible PyInstaller"

    ax1.text(0.05, 0.95, info_text, fontsize=10, va='top', ha='left',
             transform=ax1.transAxes, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax1.set_title('Información del Análisis')
    ax1.axis('off')

    # 2. Evaluación de diferentes números de clusters
    ax2 = fig.add_subplot(gs[0, 1])
    resultados_por_k = resultado.get('resultados_por_k', {})
    if resultados_por_k:
        k_vals = list(resultados_por_k.keys())
        silhouette_vals = [resultados_por_k[k]['silhouette_score'] for k in k_vals]

        ax2.plot(k_vals, silhouette_vals, 'o-', linewidth=2, markersize=8, color='darkgreen')
        ax2.set_xlabel('Número de Clusters (K)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Evaluación de K óptimo')
        ax2.grid(True, alpha=0.3)

        # Marcar K óptimo
        k_opt = mejor_config.get('n_clusters_sugeridos')
        if k_opt and k_opt in resultados_por_k:
            best_score = resultados_por_k[k_opt]['silhouette_score']
            ax2.plot(k_opt, best_score, 'ro', markersize=12, label=f'K óptimo = {k_opt}')
            ax2.legend()

            # Añadir valores en los puntos
            for k, score in zip(k_vals, silhouette_vals):
                ax2.annotate(f'{score:.3f}', (k, score), textcoords="offset points",
                             xytext=(0, 10), ha='center', fontsize=8)

    # 3. Dendrograma manual mejorado
    ax3 = fig.add_subplot(gs[1, :])
    if linkage_matrix is not None and len(linkage_matrix) > 0:
        try:
            linkage_np = np.array(linkage_matrix)

            # Determinar si mostrar todas las muestras o una selección
            max_display = 30  # Máximo de etiquetas a mostrar

            if len(sample_labels) <= max_display:
                display_labels = sample_labels
                info_text = f'Dendrograma completo ({len(sample_labels)} muestras)'
                info_color = 'lightgreen'
            else:
                # Mostrar solo una muestra representativa
                step = len(sample_labels) // max_display
                display_labels = [sample_labels[i] for i in range(0, len(sample_labels), step)][:max_display]
                info_text = f'Dendrograma muestreado ({len(display_labels)} de {len(sample_labels)} muestras)'
                info_color = 'orange'

            ax3.text(0.02, 0.98, info_text,
                     transform=ax3.transAxes, va='top', ha='left',
                     bbox=dict(boxstyle='round', facecolor=info_color, alpha=0.7),
                     fontsize=9)

            # Dibujar dendrograma MEJORADO
            plot_dendrogram_manual_mejorado(linkage_np, labels=display_labels, ax=ax3, max_display=max_display)

            # Añadir información sobre niveles de corte
            ax3.text(0.98, 0.98,
                     'Líneas rojas indican\nniveles de corte para\n2-5 clusters',
                     transform=ax3.transAxes, va='top', ha='right',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                     fontsize=9)

        except Exception as e:
            ax3.text(0.5, 0.5, f'Error creando dendrograma:\n{str(e)[:100]}',
                     ha='center', va='center', transform=ax3.transAxes,
                     bbox=dict(boxstyle='round', facecolor='mistyrose'))
            ax3.set_title('Error en Dendrograma')
            ax3.axis('off')
    else:
        ax3.text(0.5, 0.5, 'No hay matriz de linkage disponible\npara crear el dendrograma',
                 ha='center', va='center', transform=ax3.transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))
        ax3.set_title('Dendrograma no disponible')
        ax3.axis('off')

    plt.suptitle('Análisis de Clustering Jerárquico Optimizado', fontsize=16, fontweight='bold')
    return fig

def visualizar_clustering_jerarquico(datos, resultados_por_k, k_opt, mejor_config, fig, gs, ax2):
    """Visualización completa de clustering jerárquico

    Args:
        datos: DataFrame con los datos originales
        resultados_por_k: Diccionario con resultados por cada k
        k_opt: Valor óptimo de k
        mejor_config: Diccionario con la mejor configuración
        fig: Figura de matplotlib
        gs: GridSpec para layout
        ax2: Axis para gráfico de silhouette

    Returns:
        fig: Figura de matplotlib con visualizaciones
    """
    # Graficar K óptimo si está disponible
    if k_opt in resultados_por_k:
        best_score = resultados_por_k[k_opt]['silhouette_score']
        ax2.plot(k_opt, best_score, 'ro', markersize=12, label=f'K óptimo = {k_opt}')
        ax2.legend()

    # 3. Dendrograma simplificado (solo para muestras pequeñas)
    ax3 = fig.add_subplot(gs[1, 0])
    if len(datos) <= 20:  # Solo para datasets pequeños
        try:
            # Crear dendrograma simplificado usando distancias
            X_scaled, _ = aplicar_escalado(datos, 'standard')
            X_vals = X_scaled.values

            # Calcular matriz de distancias
            n_samples = len(X_vals)
            dist_matrix = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    dist = np.linalg.norm(X_vals[i] - X_vals[j])
                    dist_matrix[i, j] = dist_matrix[j, i] = dist

            # Crear dendrograma visual simplificado
            _plot_simple_dendrogram(ax3, dist_matrix, max_clusters=min(6, n_samples - 1))
            ax3.set_title(f'Dendrograma Simplificado\n(n={n_samples} muestras)')

        except Exception as e:
            ax3.text(0.5, 0.5, f'Error creando dendrograma:\n{str(e)[:50]}',
                     ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Dendrograma no disponible')
    else:
        ax3.text(0.5, 0.5,
                 f'Dendrograma no mostrado\n(muchas muestras: {len(datos)})\n\n'
                 f'Use K-Means o DBSCAN\npara datasets grandes',
                 ha='center', va='center', transform=ax3.transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))
        ax3.set_title('Dendrograma no disponible')
    ax3.axis('off')

    # 4. Visualización 2D con PCA de los clusters
    ax4 = fig.add_subplot(gs[1, 1])
    if mejor_config.get('labels'):
        try:
            # Crear un resultado temporal para usar la función de PCA
            temp_resultado = {
                'tipo': 'clustering_temp',
                'datos_originales': datos,
                'mejor_configuracion': {'labels': mejor_config['labels']}
            }
            _graficar_clusters_pca(ax4, temp_resultado)
        except Exception as e:
            ax4.text(0.5, 0.5, f'Error en visualización 2D:\n{str(e)[:50]}',
                     ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Visualización 2D no disponible')

    plt.suptitle('Análisis de Clustering Jerárquico', fontsize=16, fontweight='bold')
    return fig


def _plot_simple_dendrogram(ax, dist_matrix, max_clusters=5):
    """Crear un dendrograma visual simplificado"""
    n = len(dist_matrix)
    if n <= 1:
        return

    # Algoritmo de clustering jerárquico simplificado
    clusters = [[i] for i in range(n)]
    heights = []
    merges = []

    while len(clusters) > max_clusters:
        # Encontrar los dos clusters más cercanos
        min_dist = float('inf')
        merge_pair = (0, 1)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Distancia promedio entre clusters
                distances = []
                for pi in clusters[i]:
                    for pj in clusters[j]:
                        distances.append(dist_matrix[pi][pj])
                avg_dist = np.mean(distances)

                if avg_dist < min_dist:
                    min_dist = avg_dist
                    merge_pair = (i, j)

        # Unir clusters
        i, j = merge_pair
        new_cluster = clusters[i] + clusters[j]

        # Guardar información para el dendrograma
        heights.append(min_dist)
        merges.append((clusters[i], clusters[j], min_dist))

        # Actualizar lista de clusters
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [new_cluster]

    # Dibujar dendrograma simplificado
    if merges:
        y_positions = {}
        x_positions = {i: i for i in range(n)}

        for level, (cluster1, cluster2, height) in enumerate(merges):
            # Calcular posiciones
            x1 = np.mean([x_positions.get(p, p) for p in cluster1])
            x2 = np.mean([x_positions.get(p, p) for p in cluster2])

            # Dibujar líneas del dendrograma
            ax.plot([x1, x1], [0, height], 'b-', linewidth=1)
            ax.plot([x2, x2], [0, height], 'b-', linewidth=1)
            ax.plot([x1, x2], [height, height], 'b-', linewidth=1)

            # Actualizar posiciones para el siguiente nivel
            new_pos = (x1 + x2) / 2
            all_points = cluster1 + cluster2
            for p in all_points:
                x_positions[p] = new_pos

    ax.set_xlabel('Muestras')
    ax.set_ylabel('Distancia')
    ax.set_xticks(range(n))
    ax.set_xticklabels([f'S{i}' for i in range(n)], rotation=45)


def _crear_visualizacion_exploratorio_puntos_muestreo(resultado, figsize=(16, 12)):
    """Crear visualización para análisis exploratorio"""
    fig = plt.figure(figsize=figsize)

    # Layout 2x2
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Matriz de correlaciones (simplificada)
    ax1 = fig.add_subplot(gs[0, 0])
    correlaciones = resultado.get('correlaciones', {})
    if 'matriz_pearson' in correlaciones:
        df_corr = pd.DataFrame(correlaciones['matriz_pearson'])

        # Tomar subset si hay muchas variables
        if len(df_corr.columns) > 8:
            df_corr = df_corr.iloc[:8, :8]

        im = ax1.imshow(df_corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax1.set_xticks(range(len(df_corr.columns)))
        ax1.set_yticks(range(len(df_corr.index)))
        ax1.set_xticklabels(df_corr.columns, rotation=45, ha='right')
        ax1.set_yticklabels(df_corr.index)
        plt.colorbar(im, ax=ax1, shrink=0.8)
        ax1.set_title('Matriz de Correlaciones')

    # 2. Distribución de outliers
    ax2 = fig.add_subplot(gs[0, 1])
    outliers = resultado.get('outliers', {})
    if outliers:
        metodos = []
        cantidades = []

        for metodo, data in outliers.items():
            if metodo != 'consenso' and isinstance(data, dict) and 'total' in data:
                metodos.append(metodo.replace('_', ' ').title())
                cantidades.append(data['total'])

        if metodos:
            bars = ax2.bar(metodos, cantidades, alpha=0.7, edgecolor='black')
            ax2.set_ylabel('Número de Outliers')
            ax2.set_title('Outliers por Método')
            ax2.tick_params(axis='x', rotation=45)

            # Añadir valores
            for bar, val in zip(bars, cantidades):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                         f'{val}', ha='center', va='bottom')

    # 3. Calidad de datos
    ax3 = fig.add_subplot(gs[1, 0])
    calidad = resultado.get('calidad_datos', {})
    if calidad:
        quality_score = calidad.get('quality_score', 0)
        calificacion = calidad.get('calificacion', 'N/A')

        # Gráfico de gauge simplificado
        categories = ['Excelente', 'Buena', 'Regular', 'Deficiente']
        colors = ['green', 'yellow', 'orange', 'red']
        values = [25, 25, 25, 25]

        wedges, texts = ax3.pie(values, labels=categories, colors=colors,
                                startangle=90, counterclock=False)

        # Añadir aguja
        angle = 90 - (quality_score / 100) * 360
        ax3.annotate('', xy=(0.7 * np.cos(np.radians(angle)),
                             0.7 * np.sin(np.radians(angle))),
                     xytext=(0, 0), arrowprops=dict(arrowstyle='->', lw=3, color='black'))

        ax3.set_title(f'Calidad de Datos: {quality_score:.1f}/100\n({calificacion})')

    # 4. Resumen de recomendaciones
    ax4 = fig.add_subplot(gs[1, 1])
    recomendaciones = resultado.get('recomendaciones', [])
    if recomendaciones:
        rec_text = "Recomendaciones:\n\n"
        for i, rec in enumerate(recomendaciones[:5], 1):
            rec_text += f"{i}. {rec}\n"

        ax4.text(0.05, 0.95, rec_text, fontsize=10, va='top', ha='left',
                 transform=ax4.transAxes, wrap=True)
    ax4.set_title('Recomendaciones')
    ax4.axis('off')

    plt.suptitle('Análisis Exploratorio de Datos', fontsize=16, fontweight='bold')
    return fig


# ==================== FUNCIÓN DEMO ====================

def demo_ml_no_supervisado_completo():
    """Demostración completa del sistema ML no supervisado"""
    print("🚀 Generando datos de demostración...")
    datos = generar_datos_agua_realistas(n_muestras=300, incluir_outliers=True)

    print("📊 Datos generados exitosamente")
    print(f"   Shape: {datos.shape}")
    print(f"   Columnas: {list(datos.columns)}")

    # Variables para análisis
    variables_analisis = ['pH', 'WT', 'DO', 'TBD', 'CTD', 'BOD5', 'COD', 'FC', 'TC', 'NO3']

    # Ejemplo 1: Análisis exploratorio
    print("\n🔍 Ejemplo 1: Análisis Exploratorio Completo")
    exploratorio = analisis_exploratorio_completo(datos, variables_analisis)
    print(f"   Calidad de datos: {exploratorio['calidad_datos']['calificacion']}")
    print(f"   Outliers detectados: {exploratorio['outliers']['consenso']['porcentaje']:.1f}%")

    # Ejemplo 2: K-Means optimizado
    print("\n🔍 Ejemplo 2: K-Means Optimizado")
    kmeans_opt = kmeans_optimizado_completo(datos, variables_analisis)
    print(f"   K recomendado: {kmeans_opt['recomendacion_k']}")

    # Ejemplo 3: DBSCAN optimizado
    print("\n🔍 Ejemplo 3: DBSCAN Optimizado")
    dbscan_opt = dbscan_optimizado(datos, variables_analisis)
    mejor_dbscan = dbscan_opt['mejor_configuracion']
    print(f"   Clusters: {mejor_dbscan['n_clusters']}, Outliers: {mejor_dbscan['n_noise']}")

    # Ejemplo 4: PCA avanzado
    print("\n🔍 Ejemplo 4: PCA Avanzado")
    pca_avanzado = pca_completo_avanzado(datos, variables_analisis)
    pca_linear = pca_avanzado['resultados_por_metodo']['linear']
    print(f"   Componentes recomendados: {pca_linear['componentes_recomendados']}")

    print("\n📋 Análisis completado exitosamente!")

    return datos, {
        'exploratorio': exploratorio,
        'kmeans': kmeans_opt,
        'dbscan': dbscan_opt,
        'pca': pca_avanzado
    }


if __name__ == "__main__":
    # Ejecutar demostración
    datos, resultados = demo_ml_no_supervisado_completo()
    print(f"📊 Datos analizados: {len(datos)} muestras")
    print(f"🎯 Métodos ejecutados: {len(resultados)} técnicas")