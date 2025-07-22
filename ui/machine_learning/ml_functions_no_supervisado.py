"""
ml_functions_no_supervisado.py - Funciones de Machine Learning No Supervisado MEJORADAS
Implementación completa y optimizada para clustering, PCA y análisis exploratorio
Incluye: Clustering Jerárquico, K-Means, DBSCAN, PCA avanzado, análisis exploratorio
Versión mejorada con visualizaciones avanzadas y análisis estadísticos completos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, cut_tree
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from sklearn.manifold import TSNE, MDS
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('default')
sns.set_palette("husl")

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
        'Estacion_ID': estacion_ids,
        'Fecha': pd.date_range('2023-01-01', periods=n_muestras, freq='D'),
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
    datos['Temperatura'] = np.clip(temperatura_values, 10, 35)
    datos['Conductividad'] = np.clip(conductividad_values, 50, 1500)

    # Variables correlacionadas
    datos['Oxigeno_Disuelto'] = np.clip(
        10 - 0.2 * (datos['Temperatura'] - 20) + np.random.normal(0, 1, n_muestras),
        2, 15
    )

    datos['Turbiedad'] = np.clip(
        np.random.exponential(2.0, n_muestras) * (1 + 0.1 * np.abs(datos['pH'] - 7)),
        0.1, 50
    )

    datos['DBO5'] = np.clip(
        np.random.exponential(3, n_muestras) * (1 + datos['Turbiedad'] / 20),
        0.5, 25
    )

    datos['Coliformes_Totales'] = np.clip(
        np.random.exponential(100, n_muestras) * (1 + datos['DBO5'] / 10),
        1, 10000
    )

    datos['Nitratos'] = np.clip(
        np.random.exponential(5, n_muestras) + 0.01 * datos['Conductividad'],
        0.1, 50
    )

    datos['Fosfatos'] = np.clip(
        np.random.exponential(0.5, n_muestras) * (1 + datos['DBO5'] / 15),
        0.01, 5
    )

    # Crear DataFrame
    df = pd.DataFrame(datos)

    # Calcular índice de calidad compuesto
    df['Indice_Calidad'] = calcular_indice_calidad_simple(df)

    # Clasificar calidad
    df['Categoria_Calidad'] = pd.cut(
        df['Indice_Calidad'],
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
            param = np.random.choice(['Turbiedad', 'DBO5', 'Coliformes_Totales'])
            df.loc[idx, param] *= factor

    return df

def calcular_indice_calidad_simple(df):
    """Calcular un índice de calidad simplificado"""
    # Normalizar parámetros (0-100)
    ph_score = 100 * np.exp(-0.5 * ((df['pH'] - 7.0) / 1.5) ** 2)

    do_score = np.minimum(100, (df['Oxigeno_Disuelto'] / 10.0) * 100)

    turb_score = np.maximum(0, 100 - (df['Turbiedad'] / 10) * 100)

    dbo_score = np.maximum(0, 100 - (df['DBO5'] / 10) * 100)

    # Promedio ponderado
    indice = (0.25 * ph_score + 0.35 * do_score + 0.2 * turb_score + 0.2 * dbo_score)

    return np.clip(indice, 0, 100)

# ==================== CLUSTERING JERÁRQUICO AVANZADO ====================

def clustering_jerarquico_completo(data, variables=None, metodos=['ward', 'complete', 'average'],
                                 metricas=['euclidean', 'manhattan', 'cosine'],
                                 max_clusters=10, escalado='standard'):
    """
    Análisis exhaustivo de clustering jerárquico con múltiples métodos y métricas
    """
    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()

    # Preparar datos
    X = data[variables].dropna()

    # Escalado de datos
    if escalado == 'standard':
        scaler = StandardScaler()
    elif escalado == 'robust':
        scaler = RobustScaler()
    elif escalado == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = None

    if scaler:
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
    else:
        X_scaled = X.copy()

    resultados_completos = {}

    # Probar diferentes combinaciones de método y métrica
    for metodo in metodos:
        for metrica in metricas:
            # Verificar compatibilidad
            if metodo == 'ward' and metrica != 'euclidean':
                continue

            try:
                # Calcular matriz de distancias
                if metodo == 'ward':
                    # Ward usa datos directos
                    linkage_matrix = linkage(X_scaled.values, method=metodo, metric=metrica)
                else:
                    # Otros métodos usan matriz de distancias
                    if metrica == 'cosine':
                        from sklearn.metrics.pairwise import cosine_distances
                        dist_matrix = cosine_distances(X_scaled)
                        linkage_matrix = linkage(squareform(dist_matrix, checks=False), method=metodo)
                    else:
                        distances = pdist(X_scaled.values, metric=metrica)
                        linkage_matrix = linkage(distances, method=metodo)

                # Evaluar diferentes números de clusters
                evaluaciones_k = {}
                labels_por_k = {}

                for k in range(2, min(max_clusters + 1, len(X) // 2)):
                    labels = fcluster(linkage_matrix, k, criterion='maxclust') - 1

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
                        print(f"Error evaluando k={k} para {metodo}-{metrica}: {e}")
                        continue

                # Encontrar número óptimo de clusters
                if evaluaciones_k:
                    # Usar silhouette score como criterio principal
                    mejor_k = max(evaluaciones_k.keys(),
                                key=lambda k: evaluaciones_k[k]['silhouette_score'])

                    # Análisis de estabilidad
                    estabilidad = analizar_estabilidad_clusters(
                        X_scaled, linkage_matrix, range(2, min(8, len(evaluaciones_k) + 2))
                    )

                    resultados_completos[f"{metodo}_{metrica}"] = {
                        'metodo': metodo,
                        'metrica': metrica,
                        'linkage_matrix': linkage_matrix.tolist(),
                        'evaluaciones_por_k': evaluaciones_k,
                        'mejor_k': mejor_k,
                        'mejor_silhouette': evaluaciones_k[mejor_k]['silhouette_score'],
                        'estabilidad': estabilidad,
                        'dendrograma_data': generar_datos_dendrograma(linkage_matrix)
                    }

            except Exception as e:
                print(f"Error con {metodo}-{metrica}: {e}")
                continue

    # Determinar la mejor configuración general
    mejor_config = None
    if resultados_completos:
        mejor_config = max(resultados_completos.keys(),
                          key=lambda k: resultados_completos[k]['mejor_silhouette'])

    return {
        'tipo': 'clustering_jerarquico_completo',
        'variables_utilizadas': variables,
        'metodo_escalado': escalado,
        'resultados_por_configuracion': resultados_completos,
        'mejor_configuracion': mejor_config,
        'datos_escalados': X_scaled.values.tolist(),
        'indices_muestras': X.index.tolist(),
        'resumen_evaluacion': generar_resumen_clustering(resultados_completos)
    }

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

def analizar_estabilidad_clusters(X, linkage_matrix, k_range):
    """Analizar la estabilidad de los clusters mediante bootstrap"""
    estabilidad_scores = {}

    for k in k_range:
        labels_original = fcluster(linkage_matrix, k, criterion='maxclust') - 1

        # Bootstrap sampling
        n_bootstrap = 20
        ari_scores = []

        for _ in range(n_bootstrap):
            # Muestra bootstrap
            bootstrap_indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap = X.iloc[bootstrap_indices]

            try:
                # Clustering en muestra bootstrap
                linkage_bootstrap = linkage(X_bootstrap.values, method='ward')
                labels_bootstrap = fcluster(linkage_bootstrap, k, criterion='maxclust') - 1

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

def generar_datos_dendrograma(linkage_matrix, max_nodes=50):
    """Generar datos simplificados para visualización de dendrograma"""
    # Para datasets grandes, tomar una muestra representativa
    if len(linkage_matrix) > max_nodes:
        # Tomar los últimos pasos de fusión (más importantes)
        indices = np.linspace(0, len(linkage_matrix) - 1, max_nodes, dtype=int)
        linkage_sample = linkage_matrix[indices]
    else:
        linkage_sample = linkage_matrix

    return {
        'linkage_data': linkage_sample.tolist(),
        'n_original': len(linkage_matrix),
        'n_sample': len(linkage_sample)
    }

# ==================== K-MEANS OPTIMIZADO ====================

def kmeans_optimizado_completo(data, variables=None, k_range=None, metodos_init=['k-means++', 'random'],
                             n_inicializaciones=20, criterios_optimo=['silhouette', 'elbow', 'gap']):
    """
    K-Means optimizado con múltiples criterios para determinar K óptimo
    """
    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()

    X = data[variables].dropna()

    # Escalado estándar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if k_range is None:
        k_range = range(2, min(15, len(X) // 10))

    resultados_kmeans = {}
    inercias = []
    silhouette_scores = []

    # Evaluar diferentes valores de K
    for k in k_range:
        mejor_resultado_k = None
        mejor_silhouette_k = -1

        # Probar diferentes métodos de inicialización
        for metodo_init in metodos_init:
            try:
                kmeans = KMeans(
                    n_clusters=k,
                    init=metodo_init,
                    n_init=n_inicializaciones,
                    max_iter=500,
                    random_state=42
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
                    'metodo_init': metodo_init,
                    'labels': labels.tolist(),
                    'centroides': kmeans.cluster_centers_.tolist(),
                    'inercia': float(kmeans.inertia_),
                    'silhouette_score': float(silhouette),
                    'davies_bouldin_score': float(davies_bouldin),
                    'calinski_harabasz_score': float(calinski_harabasz),
                    'n_iteraciones': int(kmeans.n_iter_),
                    'cluster_analysis': cluster_analysis
                }

                # Guardar el mejor resultado para este K
                if silhouette > mejor_silhouette_k:
                    mejor_silhouette_k = silhouette
                    mejor_resultado_k = resultado_k

            except Exception as e:
                print(f"Error con K={k}, init={metodo_init}: {e}")
                continue

        if mejor_resultado_k:
            resultados_kmeans[k] = mejor_resultado_k
            inercias.append(mejor_resultado_k['inercia'])
            silhouette_scores.append(mejor_resultado_k['silhouette_score'])

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

    return {
        'tipo': 'kmeans_optimizado',
        'variables_utilizadas': variables,
        'k_range': list(k_range),
        'resultados_por_k': resultados_kmeans,
        'inercias': inercias,
        'silhouette_scores': silhouette_scores,
        'k_optimos': k_optimos,
        'recomendacion_k': determinar_k_final(k_optimos),
        'datos_escalados': X_scaled.tolist(),
        'scaler_params': {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist()
        }
    }

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

# ==================== DBSCAN OPTIMIZADO ====================

def dbscan_optimizado(data, variables=None, eps_range=None, min_samples_range=None,
                     metrica='euclidean', optimizar_parametros=True):
    """
    DBSCAN optimizado con búsqueda automática de parámetros óptimos
    """
    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()

    X = data[variables].dropna()

    # Escalado de datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determinar rangos automáticamente si no se proporcionan
    if eps_range is None:
        eps_range = determinar_rango_eps(X_scaled)

    if min_samples_range is None:
        min_samples_range = range(3, min(20, len(X) // 20))

    mejores_resultados = []

    # Búsqueda en grid de parámetros
    for eps in eps_range:
        for min_samples in min_samples_range:
            try:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metrica)
                labels = dbscan.fit_predict(X_scaled)

                # Evaluar calidad del clustering
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                noise_ratio = n_noise / len(labels)

                # Filtros de calidad
                if n_clusters < 2 or noise_ratio > 0.5:
                    continue

                # Calcular métricas (solo para puntos no-ruido)
                labels_sin_ruido = labels[labels != -1]
                X_sin_ruido = X_scaled[labels != -1]

                if len(np.unique(labels_sin_ruido)) > 1:
                    silhouette = silhouette_score(X_sin_ruido, labels_sin_ruido)
                    davies_bouldin = davies_bouldin_score(X_sin_ruido, labels_sin_ruido)

                    # Análisis de clusters
                    cluster_analysis = analizar_clusters_dbscan(X, labels, variables)

                    resultado = {
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'noise_ratio': noise_ratio,
                        'silhouette_score': silhouette,
                        'davies_bouldin_score': davies_bouldin,
                        'labels': labels.tolist(),
                        'cluster_analysis': cluster_analysis,
                        'score_compuesto': silhouette * (1 - noise_ratio)  # Penalizar mucho ruido
                    }

                    mejores_resultados.append(resultado)

            except Exception as e:
                continue

    if not mejores_resultados:
        return {
            'tipo': 'dbscan_optimizado',
            'error': 'No se encontraron configuraciones válidas',
            'variables_utilizadas': variables
        }

    # Ordenar por score compuesto
    mejores_resultados.sort(key=lambda x: x['score_compuesto'], reverse=True)
    mejor_resultado = mejores_resultados[0]

    # Análisis adicional del mejor resultado
    labels_final = np.array(mejor_resultado['labels'])

    # Análisis de densidad
    analisis_densidad = analizar_densidad_clusters(X_scaled, labels_final, mejor_resultado['eps'])

    # Análisis de outliers
    analisis_outliers = analizar_outliers_dbscan(X, labels_final, variables)

    return {
        'tipo': 'dbscan_optimizado',
        'variables_utilizadas': variables,
        'metrica': metrica,
        'mejor_configuracion': mejor_resultado,
        'todas_configuraciones': mejores_resultados[:10],  # Top 10
        'analisis_densidad': analisis_densidad,
        'analisis_outliers': analisis_outliers,
        'parametros_probados': {
            'eps_range': list(eps_range),
            'min_samples_range': list(min_samples_range)
        }
    }

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
            from scipy.spatial.distance import pdist

            distances = pdist(cluster_points)
            densidad = len(cluster_points) / (np.mean(distances) ** len(X_scaled[0]))

            densidad_analysis[f'cluster_{cluster_id}'] = {
                'densidad': float(densidad),
                'distancia_promedio_intra': float(np.mean(distances)),
                'compacidad': float(np.std(distances))
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

# ==================== PCA AVANZADO ====================

def pca_completo_avanzado(data, variables=None, n_components=None, metodos=['linear', 'kernel'],
                         explicar_varianza_objetivo=0.95):
    """
    Análisis de componentes principales avanzado con múltiples métodos
    """
    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()

    X = data[variables].dropna()

    # Escalado de datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if n_components is None:
        n_components = min(len(variables), len(X) - 1)

    resultados_pca = {}

    # PCA Lineal estándar
    if 'linear' in metodos:
        pca_linear = PCA(n_components=n_components)
        X_pca_linear = pca_linear.fit_transform(X_scaled)

        # Análisis detallado
        analisis_linear = analizar_pca_detallado(pca_linear, variables, explicar_varianza_objetivo)

        resultados_pca['linear'] = {
            'modelo': pca_linear,
            'transformacion': X_pca_linear.tolist(),
            'analisis': analisis_linear,
            'componentes_recomendados': analisis_linear['n_componentes_objetivo']
        }

    # Kernel PCA
    if 'kernel' in metodos and len(X) <= 1000:  # Limitado por costo computacional
        kernels = ['rbf', 'poly', 'sigmoid']

        for kernel in kernels:
            try:
                if kernel == 'poly':
                    kpca = KernelPCA(n_components=min(10, n_components), kernel=kernel, degree=3)
                else:
                    kpca = KernelPCA(n_components=min(10, n_components), kernel=kernel)

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
                print(f"Error con kernel PCA {kernel}: {e}")
                continue

    # Análisis de correlación entre componentes originales y transformados
    if 'linear' in resultados_pca:
        correlaciones = analizar_correlaciones_pca(X, X_pca_linear, variables)
        resultados_pca['linear']['correlaciones'] = correlaciones

    # Análisis de contribución de variables
    if 'linear' in resultados_pca:
        contribuciones = analizar_contribuciones_variables(
            resultados_pca['linear']['modelo'], variables
        )
        resultados_pca['linear']['contribuciones'] = contribuciones

    return {
        'tipo': 'pca_completo_avanzado',
        'variables_utilizadas': variables,
        'n_muestras': len(X),
        'metodos_aplicados': list(resultados_pca.keys()),
        'resultados_por_metodo': resultados_pca,
        'recomendaciones': generar_recomendaciones_pca(resultados_pca),
        'datos_originales_escalados': X_scaled.tolist()
    }

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
    from scipy.spatial.distance import pdist
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
    dist_original = pdist(X_sample)
    dist_transformed = pdist(X_trans_sample)

    # Correlación de Spearman entre distancias
    corr_distancias, _ = spearmanr(dist_original, dist_transformed)

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

# ==================== ANÁLISIS EXPLORATORIO COMPLETO ====================

def analisis_exploratorio_completo(data, variables=None, incluir_visualizaciones=True):
    """
    Análisis exploratorio exhaustivo de los datos
    """
    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()

    X = data[variables].dropna()

    # 1. Estadísticas descriptivas avanzadas
    estadisticas = calcular_estadisticas_avanzadas(X, variables)

    # 2. Análisis de correlaciones
    correlaciones = analizar_correlaciones_avanzado(X, variables)

    # 3. Detección de outliers múltiples métodos
    outliers = detectar_outliers_multiples_metodos(X, variables)

    # 4. Análisis de distribuciones
    distribuciones = analizar_distribuciones_avanzado(X, variables)

    # 5. Análisis de componentes principales básico
    pca_basico = PCA(n_components=min(5, len(variables)))
    pca_basico.fit(StandardScaler().fit_transform(X))

    # 6. Clustering exploratorio rápido
    clustering_exploratorio = clustering_exploratorio_rapido(X, variables)

    # 7. Análisis de calidad de datos
    calidad_datos = evaluar_calidad_datos(data, variables)

    # 8. Recomendaciones automáticas
    recomendaciones = generar_recomendaciones_exploratorio(
        estadisticas, correlaciones, outliers, distribuciones, calidad_datos
    )

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

def calcular_estadisticas_avanzadas(X, variables):
    """Calcular estadísticas descriptivas avanzadas"""
    stats_avanzadas = {}

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

        # Estadísticas de forma
        stats_forma = {
            'skewness': float(stats.skew(serie)),
            'kurtosis': float(stats.kurtosis(serie)),
            'cv': float(serie.std() / serie.mean()) if serie.mean() != 0 else np.inf
        }

        # Tests de normalidad
        try:
            shapiro_stat, shapiro_p = stats.shapiro(serie.sample(min(5000, len(serie))))
            normalidad = {
                'shapiro_p': float(shapiro_p),
                'es_normal': shapiro_p > 0.05
            }
        except:
            normalidad = {'es_normal': False}

        stats_avanzadas[variable] = {
            **stats_basicas,
            **stats_forma,
            'normalidad': normalidad
        }

    return stats_avanzadas

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
    from numpy.linalg import cond
    condicion = cond(corr_pearson.values)

    return {
        'matriz_pearson': corr_pearson.to_dict(),
        'matriz_spearman': corr_spearman.to_dict(),
        'correlaciones_fuertes': correlaciones_fuertes,
        'numero_condicion': float(condicion),
        'multicolinealidad': 'Alta' if condicion > 1000 else 'Media' if condicion > 100 else 'Baja'
    }

def detectar_outliers_multiples_metodos(X, variables):
    """Detección de outliers usando múltiples métodos"""
    outliers_por_metodo = {}

    # 1. Z-Score
    z_scores = np.abs(stats.zscore(X))
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

    # 4. Distancia de Mahalanobis
    try:
        cov_matrix = np.cov(X.T)
        inv_cov = np.linalg.pinv(cov_matrix)
        mean_vec = np.mean(X, axis=0)

        mahalanobis_dist = []
        for _, row in X.iterrows():
            diff = row.values - mean_vec
            dist = np.sqrt(diff.T @ inv_cov @ diff)
            mahalanobis_dist.append(dist)

        # Outliers usando chi-cuadrado crítico
        threshold = stats.chi2.ppf(0.95, df=len(variables))
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
    """Análisis avanzado de distribuciones"""
    distribuciones = {}

    for variable in variables:
        serie = X[variable]

        # Ajustar diferentes distribuciones
        distribuciones_testear = [
            ('normal', stats.norm),
            ('lognormal', stats.lognorm),
            ('exponential', stats.expon),
            ('gamma', stats.gamma),
            ('uniform', stats.uniform)
        ]

        mejores_ajustes = []

        for nombre, distribucion in distribuciones_testear:
            try:
                # Ajustar parámetros
                params = distribucion.fit(serie)

                # Test de Kolmogorov-Smirnov
                ks_stat, ks_p = stats.kstest(serie, lambda x: distribucion.cdf(x, *params))

                mejores_ajustes.append({
                    'distribucion': nombre,
                    'ks_statistic': float(ks_stat),
                    'ks_p_value': float(ks_p),
                    'parametros': [float(p) for p in params]
                })

            except:
                continue

        # Ordenar por p-value de KS test
        mejores_ajustes.sort(key=lambda x: x['ks_p_value'], reverse=True)

        distribuciones[variable] = {
            'mejor_ajuste': mejores_ajustes[0] if mejores_ajustes else None,
            'todos_ajustes': mejores_ajustes,
            'es_aproximadamente_normal': mejores_ajustes[0]['distribucion'] == 'normal' if mejores_ajustes else False
        }

    return distribuciones

def clustering_exploratorio_rapido(X, variables):
    """Clustering exploratorio rápido para identificar patrones"""
    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

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

# ==================== RESUMEN Y UTILIDADES ====================

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

# ==================== DEMO COMPLETO ====================

def demo_ml_no_supervisado_completo():
    """
    Demostración completa del sistema ML no supervisado mejorado
    """
    print("🚀 Generando datos de demostración...")
    datos = generar_datos_agua_realistas(n_muestras=300, incluir_outliers=True)

    print("📊 Datos generados exitosamente")
    print(f"   Shape: {datos.shape}")
    print(f"   Estaciones: {datos['Estacion_ID'].nunique()}")

    # Variables para análisis
    variables_analisis = ['pH', 'Temperatura', 'Oxigeno_Disuelto', 'Turbiedad',
                         'Conductividad', 'DBO5', 'Nitratos', 'Fosfatos']

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
                                               metricas=['euclidean', 'manhattan'])
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
    pca_avanzado = pca_completo_avanzado(datos, variables_analisis,
                                        metodos=['linear', 'kernel'])
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

if __name__ == "__main__":
    # Ejecutar demostración
    datos, resultados = demo_ml_no_supervisado_completo()
    print("\n✅ Demostración completada exitosamente!")
    print(f"📊 Datos analizados: {len(datos)} muestras con {datos.shape[1]} variables")
    print(f"🎯 Métodos ejecutados: {len(resultados)} técnicas de ML no supervisado")