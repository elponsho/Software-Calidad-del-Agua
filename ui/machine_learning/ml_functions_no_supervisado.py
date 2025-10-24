"""
ml_functions_no_supervisado.py
Machine Learning No Supervisado con optimizaciones algorítmicas
Compatible con PyInstaller - Sin threading
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List, Optional
import warnings
from datetime import datetime
import time
from collections import defaultdict

warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)


# ==================== OPTIMIZACIONES DE NUMPY ====================

def optimized_euclidean_distance(X, Y):
    """Distancia euclidiana optimizada usando broadcasting numpy"""
    return np.sqrt(np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=2))


def optimized_pairwise_distances(X, metric='euclidean'):
    """Matriz de distancias optimizada usando operaciones vectorizadas"""
    if metric == 'euclidean':
        sq_norms = np.sum(X ** 2, axis=1)
        distances = np.sqrt(sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * np.dot(X, X.T))
        return distances
    elif metric == 'manhattan':
        return np.sum(np.abs(X[:, np.newaxis, :] - X[np.newaxis, :, :]), axis=2)
    elif metric == 'cosine':
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        X_normalized = X / norms
        return 1 - np.dot(X_normalized, X_normalized.T)
    else:
        return optimized_pairwise_distances(X, 'euclidean')


# ==================== K-MEANS ====================

def kmeans_plusplus_init(X, k, random_state=42):
    """Inicialización K-Means++"""
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    centroids = [X[np.random.randint(n_samples)]]

    for _ in range(1, k):
        distances = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
        probs = distances / distances.sum()
        cumprobs = np.cumsum(probs)
        r = np.random.rand()

        for i, p in enumerate(cumprobs):
            if r < p:
                centroids.append(X[i])
                break

    return np.array(centroids)


def optimized_kmeans(X, k, max_iters=100, random_state=42, tol=1e-4):
    """K-Means optimizado"""
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    use_minibatch = n_samples > 10000
    batch_size = min(1000, n_samples) if use_minibatch else n_samples
    centroids = kmeans_plusplus_init(X, k, random_state)

    for iteration in range(max_iters):
        if use_minibatch:
            indices = np.random.choice(n_samples, batch_size, replace=False)
            X_batch = X[indices]
        else:
            X_batch = X

        distances = optimized_euclidean_distance(X_batch, centroids)
        labels_batch = np.argmin(distances, axis=1)
        new_centroids = np.array([
            X_batch[labels_batch == i].mean(axis=0) if np.any(labels_batch == i)
            else centroids[i] for i in range(k)
        ])

        shift = np.sum((new_centroids - centroids) ** 2)
        if shift < tol:
            break
        centroids = new_centroids

    distances = optimized_euclidean_distance(X, centroids)
    labels = np.argmin(distances, axis=1)
    inertia = sum(np.sum((X[labels == i] - centroids[i]) ** 2)
                  for i in range(k) if np.any(labels == i))

    return {
        'labels': labels,
        'centroids': centroids,
        'inertia': inertia,
        'n_iter': iteration + 1
    }


def optimized_silhouette_score(X, labels, sample_size=None):
    """Silhouette Score optimizado"""
    n_samples = len(X)
    unique_labels = np.unique(labels)

    if len(unique_labels) == 1:
        return 0.0

    if sample_size is not None and n_samples > sample_size:
        indices = np.random.choice(n_samples, sample_size, replace=False)
        X_sample = X[indices]
        labels_sample = labels[indices]
    else:
        X_sample = X
        labels_sample = labels

    n_sample = len(X_sample)
    distances = optimized_pairwise_distances(X_sample, 'euclidean')
    silhouette_vals = np.zeros(n_sample)

    for idx in range(n_sample):
        own_cluster = labels_sample[idx]
        same_cluster_mask = labels_sample == own_cluster
        n_same = np.sum(same_cluster_mask) - 1

        a = np.sum(distances[idx, same_cluster_mask]) / n_same if n_same > 0 else 0

        b_values = []
        for cluster in unique_labels:
            if cluster != own_cluster:
                other_mask = labels_sample == cluster
                if np.any(other_mask):
                    b_values.append(np.mean(distances[idx, other_mask]))

        b = min(b_values) if b_values else 0
        silhouette_vals[idx] = (b - a) / max(a, b) if max(a, b) > 0 else 0

    return float(np.mean(silhouette_vals))


# ==================== DBSCAN ====================

def optimized_dbscan(X, eps=0.5, min_samples=5):
    """DBSCAN optimizado"""
    n_samples = len(X)
    labels = np.full(n_samples, -1)
    cluster_id = 0

    if n_samples > 5000:
        return _dbscan_grid_accelerated(X, eps, min_samples)

    distances = optimized_pairwise_distances(X, 'euclidean')
    neighbors_list = [np.where(distances[i] <= eps)[0] for i in range(n_samples)]
    is_core = np.array([len(neighbors) >= min_samples for neighbors in neighbors_list])

    for i in range(n_samples):
        if labels[i] != -1 or not is_core[i]:
            continue

        labels[i] = cluster_id
        seed_set = set(neighbors_list[i])
        seed_set.discard(i)

        while seed_set:
            current = seed_set.pop()
            if labels[current] == -1:
                labels[current] = cluster_id
                if is_core[current]:
                    seed_set.update(neighbors_list[current])

        cluster_id += 1

    return labels


def _dbscan_grid_accelerated(X, eps, min_samples):
    """DBSCAN acelerado con grid"""
    n_samples = len(X)
    labels = np.full(n_samples, -1)
    grid_size = eps
    grid_indices = (X / grid_size).astype(int)

    grid_dict = defaultdict(list)
    for idx, grid_idx in enumerate(grid_indices):
        grid_dict[tuple(grid_idx)].append(idx)

    cluster_id = 0

    for i in range(n_samples):
        if labels[i] != -1:
            continue

        cell = tuple(grid_indices[i])
        neighbors = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell[0] + dx, cell[1] + dy)
                if neighbor_cell in grid_dict:
                    for idx in grid_dict[neighbor_cell]:
                        if np.linalg.norm(X[i] - X[idx]) <= eps:
                            neighbors.append(idx)

        if len(neighbors) < min_samples:
            continue

        labels[i] = cluster_id
        seed_set = set(neighbors)
        seed_set.discard(i)

        while seed_set:
            current = seed_set.pop()
            if labels[current] == -1:
                labels[current] = cluster_id

                cell_current = tuple(grid_indices[current])
                neighbors_current = []

                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        neighbor_cell = (cell_current[0] + dx, cell_current[1] + dy)
                        if neighbor_cell in grid_dict:
                            for idx in grid_dict[neighbor_cell]:
                                if np.linalg.norm(X[current] - X[idx]) <= eps:
                                    neighbors_current.append(idx)

                if len(neighbors_current) >= min_samples:
                    seed_set.update(neighbors_current)

        cluster_id += 1

    return labels


# ==================== PCA ====================

def optimized_pca(X, n_components=None):
    """PCA optimizado usando SVD"""
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    if n_components is None:
        n_components = min(X.shape[0], X.shape[1])

    try:
        from scipy.linalg import svd
        U, S, Vt = svd(X_centered, full_matrices=False)
    except ImportError:
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    components = Vt[:n_components]
    X_transformed = np.dot(X_centered, components.T)
    explained_variance = (S ** 2) / (X.shape[0] - 1)
    total_var = np.sum(explained_variance)
    explained_variance_ratio = explained_variance[:n_components] / total_var

    return {
        'components': components,
        'explained_variance': explained_variance[:n_components],
        'explained_variance_ratio': explained_variance_ratio,
        'mean': mean,
        'X_transformed': X_transformed
    }


# ==================== CLUSTERING JERÁRQUICO ====================

class OptimizedHierarchicalClustering:
    """Clustering jerárquico optimizado"""

    def __init__(self):
        self.progress_callback = None
        self.status_callback = None

    def set_callbacks(self, progress_callback=None, status_callback=None):
        self.progress_callback = progress_callback
        self.status_callback = status_callback

    def _emit_progress(self, value):
        if self.progress_callback:
            try:
                self.progress_callback(value)
            except:
                pass

    def _emit_status(self, message):
        if self.status_callback:
            try:
                self.status_callback(message)
            except:
                pass

    def hierarchical_clustering_optimized(self, X, method='ward', metric='euclidean'):
        """Clustering jerárquico optimizado"""
        try:
            from scipy.cluster.hierarchy import linkage
            from scipy.spatial.distance import pdist

            self._emit_status("Usando scipy (optimizado)...")
            self._emit_progress(10)
            start_time = time.time()

            if method == 'ward':
                linkage_matrix = linkage(X, method='ward')
            else:
                distances = pdist(X, metric=metric)
                linkage_matrix = linkage(distances, method=method)

            calc_time = time.time() - start_time
            self._emit_status(f"✓ Completado en {calc_time:.2f}s (scipy)")
            self._emit_progress(100)
            return linkage_matrix

        except ImportError:
            self._emit_status("Scipy no disponible, usando implementación manual...")
            return self._hierarchical_manual_optimized(X, method, metric)

    def _hierarchical_manual_optimized(self, X, method='ward', metric='euclidean'):
        """Implementación manual optimizada"""
        n_samples = len(X)
        self._emit_status(f"Clustering manual optimizado ({n_samples} muestras)...")
        self._emit_progress(0)
        start_time = time.time()

        if metric == 'euclidean':
            dist_matrix = optimized_pairwise_distances(X, 'euclidean')
        elif metric == 'manhattan':
            dist_matrix = optimized_pairwise_distances(X, 'manhattan')
        elif metric == 'cosine':
            dist_matrix = optimized_pairwise_distances(X, 'cosine')
        else:
            dist_matrix = optimized_pairwise_distances(X, 'euclidean')

        dist_matrix = np.where(np.isnan(dist_matrix) | np.isinf(dist_matrix), 1e10, dist_matrix)
        np.fill_diagonal(dist_matrix, np.inf)
        self._emit_progress(20)

        active_clusters = set(range(n_samples))
        cluster_sizes = {i: 1 for i in range(n_samples)}
        cluster_centroids = {i: X[i].copy() for i in range(n_samples)}
        linkage_matrix = []
        current_id = n_samples
        total_merges = n_samples - 1

        for merge_idx in range(total_merges):
            active_list = list(active_clusters)
            n_active = len(active_list)

            if n_active < 2:
                break

            active_indices = np.array(active_list)
            sub_matrix = dist_matrix[np.ix_(active_indices, active_indices)]
            min_idx = np.argmin(sub_matrix)
            i, j = np.unravel_index(min_idx, sub_matrix.shape)
            ci, cj = active_list[i], active_list[j]
            min_dist = sub_matrix[i, j]
            new_size = cluster_sizes[ci] + cluster_sizes[cj]

            if method == 'ward':
                new_centroid = (cluster_sizes[ci] * cluster_centroids[ci] +
                               cluster_sizes[cj] * cluster_centroids[cj]) / new_size
            else:
                new_centroid = (cluster_centroids[ci] + cluster_centroids[cj]) / 2

            linkage_matrix.append([ci, cj, min_dist, new_size])
            cluster_sizes[current_id] = new_size
            cluster_centroids[current_id] = new_centroid

            if method == 'ward':
                for other in active_clusters:
                    if other != ci and other != cj:
                        ni, nj, nk = cluster_sizes[ci], cluster_sizes[cj], cluster_sizes[other]
                        dik = dist_matrix[ci, other]
                        djk = dist_matrix[cj, other]
                        dij = min_dist
                        new_dist = np.sqrt(
                            ((ni + nk) * dik ** 2 + (nj + nk) * djk ** 2 - nk * dij ** 2) /
                            (ni + nj + nk)
                        )
                        dist_matrix[current_id, other] = new_dist
                        dist_matrix[other, current_id] = new_dist
            else:
                for other in active_clusters:
                    if other != ci and other != cj:
                        if method == 'complete':
                            new_dist = max(dist_matrix[ci, other], dist_matrix[cj, other])
                        elif method == 'average':
                            new_dist = (dist_matrix[ci, other] * cluster_sizes[ci] +
                                       dist_matrix[cj, other] * cluster_sizes[cj]) / new_size
                        elif method == 'single':
                            new_dist = min(dist_matrix[ci, other], dist_matrix[cj, other])
                        else:
                            new_dist = (dist_matrix[ci, other] + dist_matrix[cj, other]) / 2

                        dist_matrix[current_id, other] = new_dist
                        dist_matrix[other, current_id] = new_dist

            active_clusters.discard(ci)
            active_clusters.discard(cj)
            active_clusters.add(current_id)
            dist_matrix[ci, :] = np.inf
            dist_matrix[:, ci] = np.inf
            dist_matrix[cj, :] = np.inf
            dist_matrix[:, cj] = np.inf
            current_id += 1

            if (merge_idx + 1) % max(1, total_merges // 10) == 0:
                progress = 20 + int(((merge_idx + 1) / total_merges) * 75)
                self._emit_progress(progress)

        calc_time = time.time() - start_time
        self._emit_status(f"✓ Completado en {calc_time:.2f}s (manual)")
        self._emit_progress(100)
        return np.array(linkage_matrix)


# ==================== FUNCIONES AUXILIARES ====================

def manual_standard_scaler(X):
    """Escalado estándar vectorizado"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std == 0, 1, std)
    return (X - mean) / std, mean, std


def manual_minmax_scaler(X):
    """Escalado min-max vectorizado"""
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    range_vals = max_vals - min_vals
    range_vals = np.where(range_vals == 0, 1, range_vals)
    return (X - min_vals) / range_vals, min_vals, range_vals


def manual_robust_scaler(X):
    """Escalado robusto vectorizado"""
    median = np.median(X, axis=0)
    q25 = np.percentile(X, 25, axis=0)
    q75 = np.percentile(X, 75, axis=0)
    iqr = q75 - q25
    iqr = np.where(iqr == 0, 1, iqr)
    return (X - median) / iqr, median, iqr


def aplicar_escalado(X, metodo='standard'):
    """Aplicar escalado con conversión automática"""
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


def detectar_outliers_zscore(X, threshold=3.0):
    """Z-score vectorizado"""
    z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
    return np.any(z_scores > threshold, axis=1)


def detectar_outliers_iqr(X):
    """IQR vectorizado"""
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return np.any((X < lower_bound) | (X > upper_bound), axis=1)


def detectar_outliers_isolation_forest_simple(X, contamination=0.1):
    """Isolation Forest simplificado"""
    n_samples = len(X)
    centroid = np.mean(X, axis=0)
    distances = np.linalg.norm(X - centroid, axis=1)
    threshold = np.percentile(distances, 100 * (1 - contamination))
    return distances > threshold


def analizar_clusters_manual(X, labels, variables):
    """Análisis de clusters vectorizado"""
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
                serie = cluster_data[var].values
                stats_por_variable[var] = {
                    'media': float(np.mean(serie)),
                    'std': float(np.std(serie)),
                    'min': float(np.min(serie)),
                    'max': float(np.max(serie)),
                    'mediana': float(np.median(serie))
                }

        centroide = {var: float(cluster_data[var].mean()) for var in variables if var in cluster_data.columns}

        cluster_stats[f'cluster_{cluster_id}'] = {
            'tamaño': int(np.sum(mask)),
            'proporcion': float(np.sum(mask) / len(labels)),
            'centroide': centroide,
            'estadisticas': stats_por_variable
        }

    return cluster_stats


def get_clusters_from_linkage(linkage_matrix, n_clusters):
    """Obtener clusters del dendrograma"""
    try:
        from scipy.cluster.hierarchy import fcluster
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        return (labels - 1).tolist()
    except ImportError:
        if n_clusters <= 1:
            return [0] * (len(linkage_matrix) + 1)

        n_samples = len(linkage_matrix) + 1

        if n_clusters >= n_samples:
            return list(range(n_samples))

        if n_clusters - 1 >= len(linkage_matrix):
            cut_height = 0
        else:
            cut_height = linkage_matrix[-(n_clusters - 1), 2]

        clusters = {i: [i] for i in range(n_samples)}
        current_cluster_id = n_samples

        for c1, c2, height, size in linkage_matrix:
            if height > cut_height:
                break

            c1, c2 = int(c1), int(c2)

            if c1 in clusters and c2 in clusters:
                clusters[current_cluster_id] = clusters[c1] + clusters[c2]
                del clusters[c1]
                del clusters[c2]
                current_cluster_id += 1

        labels = np.zeros(n_samples, dtype=int)
        for cluster_id, (_, points) in enumerate(clusters.items()):
            for point in points:
                labels[point] = cluster_id

        return labels.tolist()


# ==================== GENERACIÓN DE DATOS ====================

def generar_datos_agua_realistas(n_muestras=200, seed=42, incluir_outliers=True):
    """Generar datos sintéticos realistas de calidad del agua"""
    np.random.seed(seed)

    n_estaciones = min(5, n_muestras // 20)
    estacion_ids = np.random.choice(range(1, n_estaciones + 1), n_muestras)

    estacion_params = {}
    for i in range(1, n_estaciones + 1):
        estacion_params[i] = {
            'ph_base': np.random.uniform(6.8, 7.8),
            'temp_base': np.random.uniform(18, 26),
            'conduct_base': np.random.uniform(200, 800),
            'quality_factor': np.random.uniform(0.7, 1.3)
        }

    datos = {
        'Points': range(1, n_muestras + 1),
        'Sampling_date': pd.date_range('2023-01-01', periods=n_muestras, freq='D'),
    }

    ph_values = np.array([np.random.normal(estacion_params[e]['ph_base'], 0.3) for e in estacion_ids])
    temperatura_values = np.array([np.random.normal(estacion_params[e]['temp_base'], 2) for e in estacion_ids])
    conductividad_values = np.array([np.random.normal(estacion_params[e]['conduct_base'], 100) for e in estacion_ids])

    datos['pH'] = np.clip(ph_values, 5.5, 9.0)
    datos['WT'] = np.clip(temperatura_values, 10, 35)
    datos['CTD'] = np.clip(conductividad_values, 50, 1500)
    datos['DO'] = np.clip(10 - 0.2 * (datos['WT'] - 20) + np.random.normal(0, 1, n_muestras), 2, 15)
    datos['TBD'] = np.clip(np.random.exponential(2.0, n_muestras) * (1 + 0.1 * np.abs(datos['pH'] - 7)), 0.1, 50)
    datos['BOD5'] = np.clip(np.random.exponential(3, n_muestras) * (1 + datos['TBD'] / 20), 0.5, 25)
    datos['COD'] = np.clip(datos['BOD5'] * np.random.uniform(1.5, 3.0, n_muestras), 1, 100)

    for var, params in [
        ('FC', (100, 1, 10000)), ('TC', (500, 10, 50000)), ('NO3', (5, 0.1, 50)),
        ('NO2', (0.5, 0.01, 3)), ('N_NH3', (1, 0.01, 10)), ('TP', (0.5, 0.01, 5)),
        ('TN', (3, 0.5, 20)), ('TKN', (2, 0.1, 15)), ('TSS', (10, 1, 200)),
        ('TS', (150, 50, 2000)), ('Q', (50, 1, 500)), ('ALC', (150, 50, 500)),
        ('H', (20, 5, 200)), ('ET', (25, 15, 40))
    ]:
        scale, min_val, max_val = params
        datos[var] = np.clip(np.random.exponential(scale, n_muestras), min_val, max_val)

    df = pd.DataFrame(datos)

    df['WQI_IDEAM_6V'] = calcular_indice_calidad_simple(df)
    df['WQI_IDEAM_7V'] = df['WQI_IDEAM_6V'] * np.random.uniform(0.9, 1.1, n_muestras)
    df['WQI_NSF_9V'] = df['WQI_IDEAM_6V'] * np.random.uniform(0.8, 1.2, n_muestras)

    df['Classification_6V'] = pd.cut(df['WQI_IDEAM_6V'], bins=[0, 40, 60, 80, 100],
                                     labels=['Deficiente', 'Regular', 'Buena', 'Excelente'])
    df['Classification_7V'] = pd.cut(df['WQI_IDEAM_7V'], bins=[0, 40, 60, 80, 100],
                                     labels=['Deficiente', 'Regular', 'Buena', 'Excelente'])
    df['Classification_9V'] = pd.cut(df['WQI_NSF_9V'], bins=[0, 40, 60, 80, 100],
                                     labels=['Deficiente', 'Regular', 'Buena', 'Excelente'])

    if incluir_outliers:
        n_outliers = max(1, n_muestras // 50)
        outlier_indices = np.random.choice(df.index, n_outliers, replace=False)
        for idx in outlier_indices:
            factor = np.random.uniform(2, 5)
            param = np.random.choice(['TBD', 'BOD5', 'FC'])
            df.loc[idx, param] *= factor

    return df


def calcular_indice_calidad_simple(df):
    """Calcular índice de calidad"""
    ph_score = 100 * np.exp(-0.5 * ((df['pH'] - 7.0) / 1.5) ** 2)
    do_score = np.minimum(100, (df['DO'] / 10.0) * 100)
    turb_score = np.maximum(0, 100 - (df['TBD'] / 10) * 100)
    dbo_score = np.maximum(0, 100 - (df['BOD5'] / 10) * 100)
    indice = (0.25 * ph_score + 0.35 * do_score + 0.2 * turb_score + 0.2 * dbo_score)
    return np.clip(indice, 0, 100)


# ==================== FUNCIONES PRINCIPALES ====================

def kmeans_optimizado_completo(data, variables=None, k_range=None, escalado='standard',
                               random_state=42, verbose=True, progress_callback=None,
                               status_callback=None):
    """K-Means con evaluación de K óptimo"""

    def emit_progress(value):
        if progress_callback:
            try:
                progress_callback(value)
            except:
                pass

    def emit_status(msg):
        if status_callback:
            try:
                status_callback(msg)
            except:
                pass
        if verbose:
            print(f"  {msg}")

    if verbose:
        print("🔍 Iniciando K-Means...")
    emit_status("Iniciando K-Means...")
    emit_progress(0)

    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()

    exclude_cols = ['Points', 'Sampling_date', 'WQI_IDEAM_6V', 'WQI_IDEAM_7V', 'WQI_NSF_9V',
                   'Classification_6V', 'Classification_7V', 'Classification_9V']
    variables = [col for col in variables if col not in exclude_cols]

    X = data[variables].dropna()

    if verbose:
        print(f"📊 Datos: {X.shape[0]} muestras, {X.shape[1]} variables")
    emit_status(f"Datos: {X.shape[0]} muestras, {X.shape[1]} variables")
    emit_progress(10)

    X_scaled, scaler_info = aplicar_escalado(X, escalado)
    emit_progress(20)

    if k_range is None:
        k_range = range(2, min(10, len(X) // 5))

    resultados_kmeans = {}
    inercias = []
    silhouette_scores = []

    total_k = len(list(k_range))
    for idx, k in enumerate(k_range):
        if verbose:
            print(f"   Evaluando K={k}...")
        emit_status(f"Evaluando K={k}...")

        try:
            kmeans_result = optimized_kmeans(X_scaled.values, k, random_state=random_state)
            labels = kmeans_result['labels']
            sample_size = min(1000, len(X))
            silhouette = optimized_silhouette_score(X_scaled.values, labels, sample_size=sample_size)
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

            progress = 20 + int(((idx + 1) / total_k) * 70)
            emit_progress(progress)

        except Exception as e:
            if verbose:
                print(f"Error con K={k}: {e}")
            emit_status(f"Error con K={k}: {e}")
            continue

    if silhouette_scores:
        k_optimo = list(k_range)[np.argmax(silhouette_scores)]
    else:
        k_optimo = list(k_range)[0] if k_range else 3

    if verbose:
        print(f"✅ K-Means completado. K recomendado: {k_optimo}")
    emit_status(f"✓ K óptimo: {k_optimo}")
    emit_progress(100)

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


def dbscan_optimizado(data, variables=None, optimizar_parametros=True, escalado='standard',
                      verbose=True, progress_callback=None, status_callback=None):
    """DBSCAN con búsqueda de parámetros"""

    def emit_progress(value):
        if progress_callback:
            try:
                progress_callback(value)
            except:
                pass

    def emit_status(msg):
        if status_callback:
            try:
                status_callback(msg)
            except:
                pass
        if verbose:
            print(f"  {msg}")

    if verbose:
        print("🔍 Iniciando DBSCAN...")
    emit_status("Iniciando DBSCAN...")
    emit_progress(0)

    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()

    exclude_cols = ['Points', 'Sampling_date', 'WQI_IDEAM_6V', 'WQI_IDEAM_7V', 'WQI_NSF_9V',
                    'Classification_6V', 'Classification_7V', 'Classification_9V']
    variables = [col for col in variables if col not in exclude_cols]

    X = data[variables].dropna()

    if verbose:
        print(f"📊 Datos: {X.shape[0]} muestras, {X.shape[1]} variables")
    emit_status(f"Datos: {X.shape[0]} muestras, {X.shape[1]} variables")
    emit_progress(10)

    X_scaled, scaler_info = aplicar_escalado(X, escalado)
    emit_progress(20)

    mejores_resultados = []

    if optimizar_parametros:
        eps_range = np.linspace(0.1, 2.0, 8)
        min_samples_range = range(2, min(8, len(X) // 10))
        total_configs = len(eps_range) * len(min_samples_range)
        config_count = 0

        for eps in eps_range:
            for min_samples in min_samples_range:
                config_count += 1

                try:
                    labels = optimized_dbscan(X_scaled.values, eps=eps, min_samples=min_samples)
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = np.sum(labels == -1)
                    noise_ratio = n_noise / len(labels)

                    if n_clusters < 1 or noise_ratio > 0.8:
                        continue

                    if n_clusters > 1:
                        non_noise_mask = labels != -1
                        if np.sum(non_noise_mask) > 1:
                            sample_size = min(500, np.sum(non_noise_mask))
                            silhouette = optimized_silhouette_score(
                                X_scaled.values[non_noise_mask],
                                labels[non_noise_mask],
                                sample_size=sample_size
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

                if config_count % max(1, total_configs // 10) == 0:
                    progress = 20 + int((config_count / total_configs) * 70)
                    emit_progress(progress)
                    emit_status(f"Probando configuraciones: {config_count}/{total_configs}")

    if not mejores_resultados:
        labels = optimized_dbscan(X_scaled.values, eps=0.5, min_samples=3)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)

        resultado = {
            'eps': 0.5,
            'min_samples': 3,
            'n_clusters': n_clusters,
            'n_noise': int(n_noise),
            'noise_ratio': n_noise / len(labels),
            'silhouette_score': 0.5,
            'labels': labels.tolist(),
            'score_compuesto': 0.5
        }
        mejores_resultados.append(resultado)

    mejores_resultados.sort(key=lambda x: x['score_compuesto'], reverse=True)
    mejor_resultado = mejores_resultados[0]

    if verbose:
        print(f"✅ DBSCAN: {mejor_resultado['n_clusters']} clusters, {mejor_resultado['n_noise']} outliers")
    emit_status(f"✓ {mejor_resultado['n_clusters']} clusters, {mejor_resultado['n_noise']} outliers")
    emit_progress(100)

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


def pca_completo_avanzado(data, variables=None, explicar_varianza_objetivo=0.95, escalado='standard',
                          verbose=True, progress_callback=None, status_callback=None):
    """PCA avanzado"""

    def emit_progress(value):
        if progress_callback:
            try:
                progress_callback(value)
            except:
                pass

    def emit_status(msg):
        if status_callback:
            try:
                status_callback(msg)
            except:
                pass
        if verbose:
            print(f"  {msg}")

    if verbose:
        print("🔍 Iniciando PCA...")
    emit_status("Iniciando PCA...")
    emit_progress(0)

    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()

    exclude_cols = ['Points', 'Sampling_date', 'WQI_IDEAM_6V', 'WQI_IDEAM_7V', 'WQI_NSF_9V',
                    'Classification_6V', 'Classification_7V', 'Classification_9V']
    variables = [col for col in variables if col not in exclude_cols]

    X = data[variables].dropna()

    if verbose:
        print(f"📊 Datos: {X.shape[0]} muestras, {X.shape[1]} variables")
    emit_status(f"Datos: {X.shape[0]} muestras, {X.shape[1]} variables")
    emit_progress(20)

    X_scaled, scaler_info = aplicar_escalado(X, escalado)
    emit_progress(40)

    emit_status("Calculando PCA (SVD)...")
    pca_result = optimized_pca(X_scaled.values)
    emit_progress(70)

    emit_status("Analizando componentes...")
    analisis = analizar_pca_detallado(pca_result, variables, explicar_varianza_objetivo)
    emit_progress(90)

    if verbose:
        print(f"✅ PCA: {analisis['n_componentes_objetivo']} componentes recomendados")
    emit_status(f"✓ {analisis['n_componentes_objetivo']} componentes recomendados")
    emit_progress(100)

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
            f"Varianza explicada: {analisis['varianza_acumulada'][analisis['n_componentes_objetivo'] - 1]:.1%}",
            f"Primera componente explica: {analisis['varianza_explicada'][0]:.1%}"
        ]
    }


def analizar_pca_detallado(pca_result, variables, varianza_objetivo):
    """Análisis detallado del PCA"""
    varianza_explicada = pca_result['explained_variance_ratio']
    varianza_acumulada = np.cumsum(varianza_explicada)
    n_componentes_objetivo = np.argmax(varianza_acumulada >= varianza_objetivo) + 1

    componentes_info = []
    for i in range(len(varianza_explicada)):
        loadings = pca_result['components'][i]
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
            'componente': f'PC{i + 1}',
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


def clustering_jerarquico_completo(data, variables=None, metodos=['ward'], metricas=['euclidean'],
                                   max_clusters=10, escalado='standard', verbose=True,
                                   progress_callback=None, status_callback=None):
    """Clustering jerárquico completo - EXTRAE ETIQUETAS DE POINTS"""

    def emit_progress(value):
        if progress_callback:
            try:
                progress_callback(value)
            except:
                pass

    def emit_status(msg):
        if status_callback:
            try:
                status_callback(msg)
            except:
                pass
        if verbose:
            print(f"  {msg}")

    if verbose:
        print("🔍 Iniciando clustering jerárquico...")
    emit_status("Iniciando clustering jerárquico...")
    emit_progress(0)

    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()

    exclude_cols = ['Points', 'Sampling_date', 'WQI_IDEAM_6V', 'WQI_IDEAM_7V', 'WQI_NSF_9V',
                    'Classification_6V', 'Classification_7V', 'Classification_9V']
    variables = [col for col in variables if col not in exclude_cols]

    # IMPORTANTE: Extraer datos con índice preservado
    X = data[variables].dropna()

    # Extraer etiquetas Points - CORRECCIÓN CLAVE
    if 'Points' in data.columns:
        # Alinear Points con el DataFrame limpio (después de dropna)
        points_labels = data.loc[X.index, 'Points'].astype(str).tolist()
    else:
        # Si no hay columna Points, usar el índice
        points_labels = X.index.astype(str).tolist()

    if verbose:
        print(f"📊 Datos: {X.shape[0]} muestras, {X.shape[1]} variables")
        print(f"🏷️ Etiquetas extraídas: {len(points_labels)} (primeras 5: {points_labels[:5]})")
    emit_status(f"Datos: {X.shape[0]} muestras, {X.shape[1]} variables")
    emit_progress(5)

    X_scaled, scaler_info = aplicar_escalado(X, escalado)

    metodo = metodos[0] if metodos else 'ward'
    metrica = metricas[0] if metricas else 'euclidean'

    if verbose:
        print(f"   Método: {metodo}, Métrica: {metrica}")
    emit_status(f"Ejecutando {metodo}-{metrica}...")

    clustering = OptimizedHierarchicalClustering()
    clustering.set_callbacks(emit_progress, emit_status)

    start_total = time.time()
    linkage_matrix = clustering.hierarchical_clustering_optimized(
        X_scaled.values, method=metodo, metric=metrica
    )
    calc_time = time.time() - start_total

    if verbose:
        print(f"   ⚡ Clustering completado en {calc_time:.2f}s")
    emit_status(f"✓ Linkage calculado en {calc_time:.2f}s")
    emit_progress(95)

    emit_status("Evaluando diferentes K...")

    resultados_por_k = {}
    max_k = min(max_clusters, len(X) - 1)

    for k in range(2, max_k + 1):
        try:
            labels = get_clusters_from_linkage(linkage_matrix, k)

            if len(set(labels)) > 1:
                sample_size = min(500, len(X))
                silhouette = optimized_silhouette_score(X_scaled.values, np.array(labels), sample_size=sample_size)
            else:
                silhouette = 0.0

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

    if resultados_por_k:
        mejor_k = max(resultados_por_k.keys(), key=lambda k: resultados_por_k[k]['silhouette_score'])
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

    total_time = time.time() - start_total

    if verbose:
        print(f"✅ Clustering jerárquico: Mejor K={mejor_k} en {total_time:.2f}s")
    emit_status(f"✓ Completado: K={mejor_k} en {total_time:.2f}s")
    emit_progress(100)

    return {
        'tipo': 'clustering_jerarquico_completo',
        'variables_utilizadas': variables,
        'metodo_escalado': escalado,
        'linkage_matrix': linkage_matrix.tolist(),
        'resultados_por_k': resultados_por_k,
        'mejor_configuracion': mejor_configuracion,
        'datos_originales': X,
        'scaler_info': scaler_info,
        'sample_labels': points_labels,  # ← ETIQUETAS EXTRAÍDAS DE POINTS
        'tiempo_calculo': total_time,
        'usando_sklearn': False,
        'recomendaciones': [
            f"Mejor: {metodo}-{metrica} con {mejor_k} clusters",
            f"Silhouette Score: {mejor_silhouette:.3f}",
            f"Tiempo: {total_time:.2f}s"
        ]
    }


# ==================== ANÁLISIS EXPLORATORIO ====================

def calcular_estadisticas_avanzadas(X, variables):
    """Calcular estadísticas descriptivas"""
    estadisticas = {}

    for variable in variables:
        if variable not in X.columns:
            continue

        serie = X[variable].values
        stats_basicas = {
            'count': len(serie),
            'mean': float(np.mean(serie)),
            'std': float(np.std(serie)),
            'min': float(np.min(serie)),
            'max': float(np.max(serie)),
            'range': float(np.max(serie) - np.min(serie)),
            'median': float(np.median(serie)),
            'q1': float(np.percentile(serie, 25)),
            'q3': float(np.percentile(serie, 75)),
            'iqr': float(np.percentile(serie, 75) - np.percentile(serie, 25))
        }

        mean_val = np.mean(serie)
        std_val = np.std(serie)

        if std_val > 0:
            skew_val = np.mean(((serie - mean_val) / std_val) ** 3)
            kurt_val = np.mean(((serie - mean_val) / std_val) ** 4) - 3
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
    corr_matrix = np.corrcoef(X[variables].T)
    corr_pearson = pd.DataFrame(corr_matrix, index=variables, columns=variables)

    correlaciones_fuertes = []
    n_vars = len(variables)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            corr_val = corr_pearson.iloc[i, j]

            if not np.isnan(corr_val) and abs(corr_val) > 0.6:
                correlaciones_fuertes.append({
                    'variable_1': variables[i],
                    'variable_2': variables[j],
                    'correlacion': float(corr_val),
                    'tipo': 'Fuerte positiva' if corr_val > 0.6 else 'Fuerte negativa'
                })

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
    X_vals = X[variables].values

    outliers_zscore = detectar_outliers_zscore(X_vals)
    outliers_iqr = detectar_outliers_iqr(X_vals)
    outliers_isolation = detectar_outliers_isolation_forest_simple(X_vals)

    outliers_zscore_indices = X[outliers_zscore].index.tolist()
    outliers_iqr_indices = X[outliers_iqr].index.tolist()
    outliers_isolation_indices = X[outliers_isolation].index.tolist()

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

        serie = X[variable].values
        mean_val = np.mean(serie)
        std_val = np.std(serie)

        if std_val > 0:
            skew_val = np.mean(((serie - mean_val) / std_val) ** 3)
            kurt_val = np.mean(((serie - mean_val) / std_val) ** 4) - 3
        else:
            skew_val = 0
            kurt_val = 0

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
                'parametros': [float(mean_val), float(std_val)]
            },
            'es_aproximadamente_normal': distribucion_tipo == 'normal',
            'skewness': float(skew_val),
            'kurtosis': float(kurt_val)
        }

    return distribuciones


def clustering_exploratorio_rapido(X, variables, escalado='standard'):
    """Clustering exploratorio rápido"""
    X_scaled, _ = aplicar_escalado(X[variables], escalado)

    resultados_rapidos = {}

    for k in [3, 4, 5]:
        try:
            kmeans_result = optimized_kmeans(X_scaled.values, k, max_iters=50)
            labels = kmeans_result['labels']
            sample_size = min(500, len(X))
            silhouette = optimized_silhouette_score(X_scaled.values, labels, sample_size=sample_size)

            resultados_rapidos[k] = {
                'silhouette_score': float(silhouette),
                'inercia': float(kmeans_result['inertia']),
                'labels': labels.tolist()
            }
        except:
            continue

    if resultados_rapidos:
        mejor_k_rapido = max(resultados_rapidos.keys(), key=lambda k: resultados_rapidos[k]['silhouette_score'])
    else:
        mejor_k_rapido = 3

    return {
        'resultados': resultados_rapidos,
        'mejor_k_rapido': mejor_k_rapido,
        'recomendacion': f"Se sugiere explorar clustering con {mejor_k_rapido} grupos"
    }


def evaluar_calidad_datos(data_original, variables):
    """Evaluar calidad general de los datos"""
    missing_info = {}
    for var in variables:
        if var in data_original.columns:
            missing_count = data_original[var].isnull().sum()
            missing_info[var] = {
                'count': int(missing_count),
                'percentage': float(missing_count / len(data_original) * 100)
            }

    duplicados = data_original.duplicated().sum()

    unique_info = {}
    for var in variables:
        if var in data_original.columns:
            unique_count = data_original[var].nunique()
            unique_info[var] = {
                'unique_count': int(unique_count),
                'unique_ratio': float(unique_count / len(data_original))
            }

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

    if calidad['quality_score'] < 85:
        recomendaciones.append("⚠️ Considere limpiar los datos antes del análisis")

    if 'correlaciones_fuertes' in correlaciones:
        n_corr = len(correlaciones['correlaciones_fuertes'])
        if n_corr > 0:
            recomendaciones.append(f"🔗 Se detectaron {n_corr} correlaciones fuertes entre variables")

    if correlaciones.get('multicolinealidad') == 'Alta':
        recomendaciones.append("⚠️ Alta multicolinealidad detectada - Considere PCA")

    if 'consenso' in outliers:
        porcentaje_outliers = outliers['consenso']['porcentaje']
        if porcentaje_outliers > 10:
            recomendaciones.append(f"⚠️ Alto porcentaje de outliers ({porcentaje_outliers:.1f}%)")
        elif porcentaje_outliers > 5:
            recomendaciones.append(f"⚠️ Outliers moderados detectados ({porcentaje_outliers:.1f}%)")

    if distribuciones:
        variables_no_normales = sum(1 for var_info in distribuciones.values()
                                    if not var_info.get('es_aproximadamente_normal', False))

        if variables_no_normales > len(distribuciones) * 0.7:
            recomendaciones.append("📊 Muchas variables no siguen distribución normal - Use escalado robusto")

    n_variables = len(estadisticas)
    if n_variables > 10:
        recomendaciones.append(f"📉 Alto número de variables ({n_variables}) - PCA recomendado")

    if outliers and outliers.get('consenso', {}).get('porcentaje', 0) > 5:
        recomendaciones.append("🎯 DBSCAN recomendado para clustering con outliers")
    else:
        recomendaciones.append("🎯 K-Means puede ser apropiado para clustering")

    return recomendaciones


def analisis_exploratorio_completo(data, variables=None, escalado='standard', handle_outliers=True,
                                   verbose=True, progress_callback=None, status_callback=None):
    """Análisis exploratorio exhaustivo"""

    def emit_progress(value):
        if progress_callback:
            try:
                progress_callback(value)
            except:
                pass

    def emit_status(msg):
        if status_callback:
            try:
                status_callback(msg)
            except:
                pass
        if verbose:
            print(f"  {msg}")

    if verbose:
        print("🔍 Iniciando análisis exploratorio...")
    emit_status("Iniciando análisis exploratorio...")
    emit_progress(0)

    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()

    exclude_cols = ['Points', 'Sampling_date', 'WQI_IDEAM_6V', 'WQI_IDEAM_7V', 'WQI_NSF_9V',
                    'Classification_6V', 'Classification_7V', 'Classification_9V']
    variables = [col for col in variables if col not in exclude_cols]

    X = data[variables].dropna()

    if verbose:
        print(f"📊 Datos: {X.shape[0]} muestras, {X.shape[1]} variables")
    emit_status(f"Datos: {X.shape[0]} muestras, {X.shape[1]} variables")
    emit_progress(10)

    emit_status("Calculando estadísticas...")
    estadisticas = calcular_estadisticas_avanzadas(X, variables)
    emit_progress(25)

    emit_status("Analizando correlaciones...")
    correlaciones = analizar_correlaciones_avanzado(X, variables)
    emit_progress(40)

    emit_status("Detectando outliers...")
    outliers = detectar_outliers_multiples_metodos(X, variables) if handle_outliers else {}
    emit_progress(55)

    emit_status("Analizando distribuciones...")
    distribuciones = analizar_distribuciones_avanzado(X, variables)
    emit_progress(70)

    emit_status("Ejecutando PCA exploratorio...")
    X_scaled, _ = aplicar_escalado(X, escalado)
    pca_basico = optimized_pca(X_scaled.values, n_components=min(5, len(variables)))
    emit_progress(80)

    emit_status("Clustering exploratorio...")
    clustering_exploratorio = clustering_exploratorio_rapido(X, variables, escalado)
    emit_progress(90)

    emit_status("Evaluando calidad de datos...")
    calidad_datos = evaluar_calidad_datos(data, variables)
    emit_progress(95)

    recomendaciones = generar_recomendaciones_exploratorio(
        estadisticas, correlaciones, outliers, distribuciones, calidad_datos
    )

    if verbose:
        print("✅ Análisis exploratorio completado")
    emit_status("✓ Análisis completado")
    emit_progress(100)

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


# ==================== VISUALIZACIONES ====================

def generar_visualizaciones_ml_no_supervisado(resultado: Dict[str, Any],
                                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """Generar visualizaciones para ML No Supervisado"""
    tipo = resultado.get('tipo', '')

    try:
        if tipo in ['kmeans_optimizado', 'dbscan_optimizado']:
            return crear_visualizacion_clustering(resultado, figsize)
        elif tipo == 'pca_completo_avanzado':
            return crear_visualizacion_pca(resultado, figsize)
        elif tipo == 'clustering_jerarquico_completo':
            return crear_visualizacion_jerarquico(resultado, figsize)
        elif tipo == 'analisis_exploratorio_completo':
            return crear_visualizacion_exploratorio(resultado, figsize)
        else:
            return crear_visualizacion_generica(resultado, figsize)

    except Exception as e:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f'Error generando visualización:\n{str(e)[:100]}',
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='mistyrose'))
        ax.set_title('Error en Visualización')
        ax.axis('off')
        return fig


def crear_visualizacion_clustering(resultado, figsize=(12, 8)):
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
        return fig


def _crear_viz_kmeans(resultado, fig):
    """Visualización para K-Means"""
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Gráfico de Silhouette Score
    ax1 = fig.add_subplot(gs[0, 0])
    k_vals = list(resultado['resultados_por_k'].keys())
    silhouette_vals = [resultado['resultados_por_k'][k]['silhouette_score'] for k in k_vals]

    ax1.plot(k_vals, silhouette_vals, 'bo-', linewidth=2.5, markersize=10)
    ax1.set_xlabel('Número de Clusters (K)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
    ax1.set_title('Evaluación de K óptimo', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')

    k_opt = resultado.get('recomendacion_k')
    if k_opt in resultado['resultados_por_k']:
        best_score = resultado['resultados_por_k'][k_opt]['silhouette_score']
        ax1.plot(k_opt, best_score, 'ro', markersize=15, label=f'K óptimo = {k_opt}', zorder=5)
        ax1.legend(loc='best', frameon=True, shadow=True)

    # 2. Distribución por cluster
    ax2 = fig.add_subplot(gs[0, 1])
    if k_opt and k_opt in resultado['resultados_por_k']:
        labels = resultado['resultados_por_k'][k_opt]['labels']
        unique_labels = np.unique(labels)
        tamaños = [labels.count(label) for label in unique_labels]

        colors_bars = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        bars = ax2.bar(range(len(unique_labels)), tamaños, color=colors_bars, alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Cluster', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Número de Puntos', fontsize=11, fontweight='bold')
        ax2.set_title('Distribución por Cluster', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(unique_labels)))
        ax2.set_xticklabels([f'C{label}' for label in unique_labels])
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

        for bar, tamaño in zip(bars, tamaños):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{tamaño}', ha='center', va='bottom', fontweight='bold')

    # 3. Visualización PCA
    ax3 = fig.add_subplot(gs[1, :])
    _graficar_clusters_pca(ax3, resultado)

    plt.suptitle('Análisis K-Means', fontsize=16, fontweight='bold')
    return fig


def _crear_viz_dbscan(resultado, fig):
    """Visualización para DBSCAN"""
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    mejor_config = resultado['mejor_configuracion']

    # 1. Información de configuración
    ax1 = fig.add_subplot(gs[0, 0])
    info_text = f"⚙️ PARÁMETROS DBSCAN:\n\n"
    info_text += f"• Eps: {mejor_config['eps']:.3f}\n"
    info_text += f"• Min Samples: {mejor_config['min_samples']}\n\n"
    info_text += f"📊 RESULTADOS:\n\n"
    info_text += f"• Clusters: {mejor_config['n_clusters']}\n"
    info_text += f"• Outliers: {mejor_config['n_noise']}\n"
    info_text += f"• % Outliers: {mejor_config['noise_ratio']*100:.1f}%\n"
    info_text += f"• Silhouette: {mejor_config['silhouette_score']:.3f}"

    ax1.text(0.05, 0.95, info_text, fontsize=10, va='top', ha='left',
             transform=ax1.transAxes, family='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8, edgecolor='black'))
    ax1.set_title('Configuración DBSCAN', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 2. Distribución
    ax2 = fig.add_subplot(gs[0, 1])
    labels = mejor_config['labels']
    unique_labels = [l for l in set(labels) if l != -1]
    n_outliers = labels.count(-1)

    cluster_counts = [labels.count(label) for label in unique_labels]
    cluster_names = [f'Cluster {label}' for label in unique_labels]

    if n_outliers > 0:
        cluster_counts.append(n_outliers)
        cluster_names.append('Outliers')

    colors = ['red' if name == 'Outliers' else plt.cm.viridis(i/len(cluster_names))
              for i, name in enumerate(cluster_names)]
    bars = ax2.bar(cluster_names, cluster_counts, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Número de Puntos', fontsize=11, fontweight='bold')
    ax2.set_title('Distribución por Cluster', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

    for bar, count in zip(bars, cluster_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom', fontweight='bold')

    # 3. Visualización PCA
    ax3 = fig.add_subplot(gs[1, :])
    _graficar_clusters_pca(ax3, resultado)

    plt.suptitle('Análisis DBSCAN', fontsize=16, fontweight='bold')
    return fig


def _graficar_clusters_pca(ax, resultado):
    """Graficar clusters usando PCA"""
    try:
        datos = resultado['datos_originales']

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

        # Usar PCA
        X_scaled, _ = aplicar_escalado(datos, 'standard')
        pca_result = optimized_pca(X_scaled.values, n_components=2)
        datos_2d = pca_result['X_transformed']

        unique_labels = set(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for i, (label, color) in enumerate(zip(unique_labels, colors)):
            mask = np.array(labels) == label

            if label == -1:
                ax.scatter(datos_2d[mask, 0], datos_2d[mask, 1],
                          c='black', marker='x', s=100, alpha=0.8, label='Outliers', linewidths=2)
            else:
                ax.scatter(datos_2d[mask, 0], datos_2d[mask, 1],
                          c=[color], label=f'Cluster {label}',
                          s=60, alpha=0.7, edgecolors='black', linewidth=0.5)

        var_exp = pca_result['explained_variance_ratio']
        ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)', fontsize=11, fontweight='bold')
        ax.set_title('Clusters en Espacio PCA', fontsize=12, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')

    except Exception as e:
        ax.text(0.5, 0.5, f'Error en visualización: {str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes)


def crear_visualizacion_pca(resultado, figsize=(12, 8)):
    """Crear visualización para PCA"""
    fig = plt.figure(figsize=figsize)

    if 'linear' not in resultado.get('resultados_por_metodo', {}):
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No hay resultados de PCA para visualizar',
                ha='center', va='center', transform=ax.transAxes)
        return fig

    linear_result = resultado['resultados_por_metodo']['linear']
    analisis = linear_result['analisis']

    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Varianza por componente
    ax1 = fig.add_subplot(gs[0, 0])
    var_exp = analisis['varianza_explicada']
    x = range(1, len(var_exp) + 1)
    bars = ax1.bar(x, [v * 100 for v in var_exp], alpha=0.8,
                   color=plt.cm.viridis(np.linspace(0, 1, len(var_exp))),
                   edgecolor='black')
    ax1.set_xlabel('Componente Principal', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Varianza Explicada (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Varianza por Componente', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

    # 2. Varianza acumulada
    ax2 = fig.add_subplot(gs[0, 1])
    var_acum = analisis['varianza_acumulada']
    ax2.plot(x, [v * 100 for v in var_acum], 'o-', linewidth=2.5, markersize=10, color='darkred')
    ax2.axhline(y=95, color='red', linestyle='--', linewidth=2, alpha=0.7, label='95% varianza')
    ax2.set_xlabel('Componente Principal', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Varianza Acumulada (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Varianza Acumulada', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 105)

    # 3. Scree plot
    ax3 = fig.add_subplot(gs[1, :])
    eigenvalues = analisis['eigenvalues']
    ax3.plot(x, eigenvalues, 'o-', linewidth=2.5, markersize=10, color='purple')
    ax3.set_xlabel('Componente Principal', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Eigenvalue', fontsize=11, fontweight='bold')
    ax3.set_title('Scree Plot - Eigenvalues', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')

    if max(eigenvalues) > 1:
        ax3.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Kaiser criterion (λ=1)')
        ax3.legend(loc='best', frameon=True, shadow=True)

    plt.suptitle('Análisis PCA', fontsize=16, fontweight='bold')
    return fig


def crear_visualizacion_jerarquico(resultado, figsize=(16, 9)):
    """Crear visualización para clustering jerárquico - USA sample_labels"""

    mejor_config = resultado.get('mejor_configuracion', {})
    linkage_matrix = np.array(resultado.get('linkage_matrix', []))
    datos_originales = resultado.get('datos_originales', pd.DataFrame())
    sample_labels = resultado.get('sample_labels', [])  # ← USAR ETIQUETAS EXTRAÍDAS
    n_samples = len(datos_originales)
    tiempo_calculo = resultado.get('tiempo_calculo', 0)

    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_facecolor('white')

    if len(linkage_matrix) > 0:
        try:
            from scipy.cluster.hierarchy import dendrogram, fcluster

            # Calcular altura de corte
            altura_corte = np.percentile(linkage_matrix[:, 2], 70)
            n_clusters_corte = len([d for d in linkage_matrix[:, 2] if d > altura_corte]) + 1

            # Decidir truncado
            if n_samples > 80:
                truncate_mode = 'lastp'
                p_value = 50
                usar_labels = False
            else:
                truncate_mode = None
                p_value = None
                usar_labels = True

            # Configurar dendrograma
            dendrogram_params = {
                'ax': ax,
                'color_threshold': altura_corte,
                'above_threshold_color': '#95A5A6',
                'leaf_font_size': 8,
                'leaf_rotation': 90,
                'orientation': 'top',
                'distance_sort': 'ascending'
            }

            # Usar etiquetas extraídas
            if usar_labels and sample_labels:
                dendrogram_params['labels'] = sample_labels  # ← ETIQUETAS DE POINTS
                dendrogram_params['show_leaf_counts'] = False
            else:
                dendrogram_params['truncate_mode'] = truncate_mode
                dendrogram_params['p'] = p_value
                dendrogram_params['show_leaf_counts'] = True

            dendro_data = dendrogram(linkage_matrix, **dendrogram_params)

            # Calcular métricas
            try:
                k_optimo = mejor_config.get('n_clusters_sugeridos', 3)
                cluster_labels = fcluster(linkage_matrix, k_optimo, criterion='maxclust')

                variables = resultado.get('variables_utilizadas', [])
                if variables and not datos_originales.empty:
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

                    X_metricas = datos_originales[variables]
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_metricas)

                    silhouette = silhouette_score(X_scaled, cluster_labels[:len(X_scaled)])
                    davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels[:len(X_scaled)])
                    calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels[:len(X_scaled)])
                else:
                    silhouette = mejor_config.get('silhouette_score', 0)
                    davies_bouldin = 0
                    calinski_harabasz = 0
            except:
                silhouette = mejor_config.get('silhouette_score', 0)
                davies_bouldin = 0
                calinski_harabasz = 0

            # Título con métricas
            metodo = mejor_config.get('metodo', 'ward').upper()
            metrica = mejor_config.get('metrica', 'euclidean').capitalize()

            variables_list = resultado.get('variables_utilizadas', [])
            if len(variables_list) > 5:
                vars_texto = ', '.join(variables_list[:5]) + f' (+{len(variables_list) - 5} more)'
            else:
                vars_texto = ', '.join(variables_list)

            titulo = (
                f'Hierarchical Clustering Dendrogram - {metodo.title()} Method\n'
                f'(Distance Metric = {metrica}, considering {vars_texto})\n'
                f'(Silhouette Coef.: {silhouette:.3f}, DB: {davies_bouldin:.3f}, CH: {calinski_harabasz:.0f})'
            )

            ax.set_title(titulo, fontsize=11, fontweight='bold', color='#2c3e50', pad=12)
            ax.set_xlabel('Monitoring Points', fontsize=12, fontweight='600', color='#34495e', labelpad=8)
            ax.set_ylabel('Distance', fontsize=12, fontweight='600', color='#34495e', labelpad=8)

            # Línea de corte
            ax.axhline(y=altura_corte, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.85,
                       zorder=10, label=f'Optimal cut: {altura_corte:.3f} → {n_clusters_corte} clusters')

            # Estilo
            ax.yaxis.grid(True, linestyle='-', linewidth=0.3, color='#CCCCCC', alpha=0.5, zorder=0)
            ax.set_axisbelow(True)

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.2)
                spine.set_color('#7F8C8D')

            ax.tick_params(axis='y', labelsize=10, colors='#2c3e50', width=1.2)
            ax.tick_params(axis='x', labelsize=7, colors='#2c3e50', width=1.2, length=3)

            if usar_labels:
                for tick in ax.get_xticklabels():
                    tick.set_rotation(90)
                    tick.set_ha('center')

            legend = ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=10,
                               framealpha=0.98, edgecolor='#3498DB', facecolor='white', borderpad=0.8)
            legend.get_frame().set_linewidth(2)

            # Cuadro de información
            if silhouette > 0.7:
                calidad_texto = "Excellent ⭐⭐⭐"
                box_color = '#D5F4E6'
                edge_color = '#27AE60'
            elif silhouette > 0.5:
                calidad_texto = "Good ⭐⭐"
                box_color = '#FFF3CD'
                edge_color = '#F39C12'
            elif silhouette > 0.3:
                calidad_texto = "Acceptable ⭐"
                box_color = '#FFF3CD'
                edge_color = '#E67E22'
            else:
                calidad_texto = "Needs improvement"
                box_color = '#F8D7DA'
                edge_color = '#E74C3C'

            modo_texto = f"Truncated ({p_value} nodes)" if truncate_mode else "Complete"

            info_text = (
                f'📊 Stations: {n_samples}\n'
                f'🎯 Suggested K: {k_optimo}\n'
                f'⭐ Silhouette: {silhouette:.3f}\n'
                f'📈 Quality: {calidad_texto}\n'
                f'🔍 Mode: {modo_texto}\n'
                f'⏱️ Time: {tiempo_calculo:.2f}s'
            )

            ax.text(0.015, 0.98, info_text, transform=ax.transAxes, fontsize=8.5,
                    va='top', ha='left', family='monospace',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor=box_color,
                              alpha=0.95, edgecolor=edge_color, linewidth=2))

            # Instrucciones
            instrucciones = (
                '🖱️ Navigation:\n'
                '• Zoom: Mouse wheel\n'
                '• Pan: Click + Drag\n'
                '• Reset: Home 🏠'
            )

            ax.text(0.985, 0.02, instrucciones, transform=ax.transAxes, fontsize=7.5,
                    va='bottom', ha='right', style='italic', color='#7F8C8D',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F9FA',
                              alpha=0.9, edgecolor='#95A5A6', linewidth=1.2))

            # Ajustes de espaciado
            y_min, y_max = ax.get_ylim()
            ax.set_ylim(y_min - 0.01 * y_max, y_max * 1.08)

            if usar_labels and sample_labels:
                max_label_len = max(len(str(label)) for label in sample_labels)
                bottom_margin = min(0.22, 0.10 + max_label_len * 0.006)
            else:
                bottom_margin = 0.12

            plt.tight_layout()
            fig.subplots_adjust(bottom=bottom_margin, top=0.90, left=0.08, right=0.98)

        except ImportError:
            ax.text(0.5, 0.5,
                    '⚠️ Scipy not installed\n\n'
                    'Run: pip install scipy scikit-learn\n'
                    'to generate the dendrogram',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=13, color='#E67E22', weight='bold',
                    bbox=dict(boxstyle='round,pad=1.5', facecolor='#FFF3CD',
                              alpha=0.95, edgecolor='#E67E22', linewidth=2.5))
            ax.axis('off')

    else:
        ax.text(0.5, 0.5,
                '❌ No linkage data available\n\n'
                'Hierarchical clustering did not\n'
                'execute correctly',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=13, color='#C0392B', weight='bold',
                bbox=dict(boxstyle='round,pad=1.5', facecolor='#F8D7DA',
                          alpha=0.95, edgecolor='#C0392B', linewidth=2.5))
        ax.axis('off')

    return fig


def crear_visualizacion_exploratorio(resultado, figsize=(12, 8)):
    """Crear visualización para análisis exploratorio"""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Matriz de correlaciones (simplificada)
    ax1 = fig.add_subplot(gs[0, 0])
    correlaciones = resultado.get('correlaciones', {})
    if 'matriz_pearson' in correlaciones:
        df_corr = pd.DataFrame(correlaciones['matriz_pearson'])

        if len(df_corr.columns) > 8:
            df_corr = df_corr.iloc[:8, :8]

        im = ax1.imshow(df_corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax1.set_xticks(range(len(df_corr.columns)))
        ax1.set_yticks(range(len(df_corr.index)))
        ax1.set_xticklabels(df_corr.columns, rotation=45, ha='right', fontsize=9)
        ax1.set_yticklabels(df_corr.index, fontsize=9)
        plt.colorbar(im, ax=ax1, shrink=0.8)
        ax1.set_title('Matriz de Correlaciones', fontsize=12, fontweight='bold')

    # 2. Outliers por método
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
            colors_bars = plt.cm.Set3(np.linspace(0, 1, len(metodos)))
            bars = ax2.bar(metodos, cantidades, color=colors_bars, alpha=0.8, edgecolor='black')
            ax2.set_ylabel('Número de Outliers', fontsize=11, fontweight='bold')
            ax2.set_title('Outliers por Método', fontsize=12, fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

    # 3. Calidad de datos
    ax3 = fig.add_subplot(gs[1, 0])
    calidad = resultado.get('calidad_datos', {})
    if calidad:
        quality_score = calidad.get('quality_score', 0)
        calificacion = calidad.get('calificacion', 'N/A')

        ax3.text(0.5, 0.5, f'Calidad de Datos\n\n{quality_score:.1f}/100\n\n({calificacion})',
                ha='center', va='center', transform=ax3.transAxes, fontsize=14,
                bbox=dict(boxstyle='round', facecolor='lightgreen' if quality_score > 80 else 'lightyellow', alpha=0.9))
        ax3.set_title('Score de Calidad', fontsize=12, fontweight='bold')
        ax3.axis('off')

    # 4. Recomendaciones
    ax4 = fig.add_subplot(gs[1, 1])
    recomendaciones = resultado.get('recomendaciones', [])
    if recomendaciones:
        rec_text = "💡 RECOMENDACIONES:\n\n"
        for i, rec in enumerate(recomendaciones[:4], 1):
            rec_text += f"{i}. {rec}\n\n"

        ax4.text(0.05, 0.95, rec_text, fontsize=9, va='top', ha='left',
                 transform=ax4.transAxes, wrap=True,
                 bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.9, edgecolor='black'))
    ax4.set_title('📋 Recomendaciones', fontsize=12, fontweight='bold')
    ax4.axis('off')

    plt.suptitle('Análisis Exploratorio', fontsize=16, fontweight='bold')
    return fig


def crear_visualizacion_generica(resultado, figsize):
    """Crear visualización genérica"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    tipo = resultado.get('tipo', 'desconocido')
    ax.text(0.5, 0.5, f'Visualización para: {tipo}\n\n✓ Resultados disponibles',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.9, edgecolor='black'))
    ax.set_title(f'Resultado: {tipo}', fontsize=14, fontweight='bold')
    ax.axis('off')

    return fig


# ==================== DENDROGRAMA COMPLETO ====================

def calcular_altura_corte_optima(linkage_matrix, metodo='percentil', percentil=70):
    """Calcular altura óptima para cortar el dendrograma"""
    distancias = linkage_matrix[:, 2]

    if metodo == 'percentil':
        return np.percentile(distancias, percentil)
    elif metodo == 'mediana':
        return np.median(distancias)
    elif metodo == 'gap':
        diffs = np.diff(sorted(distancias))
        max_gap_idx = np.argmax(diffs)
        return sorted(distancias)[max_gap_idx]
    else:
        return np.percentile(distancias, 70)


def generar_dendrograma_completo(linkage_matrix, n_samples,
                                 truncate_mode='level', p=10,
                                 mostrar_linea_corte=True,
                                 altura_corte=None):
    """Generar figura completa con dendrograma y opciones avanzadas"""
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, height_ratios=[2, 1])

    # 1. Dendrograma principal
    ax_dendro = fig.add_subplot(gs[0, :])

    if altura_corte is None and mostrar_linea_corte:
        altura_corte = calcular_altura_corte_optima(linkage_matrix, metodo='percentil', percentil=70)

    try:
        from scipy.cluster.hierarchy import dendrogram

        dendrogram_params = {
            'ax': ax_dendro,
            'color_threshold': altura_corte if mostrar_linea_corte else 0,
            'above_threshold_color': '#95A5A6',
            'leaf_font_size': 8,
            'leaf_rotation': 90,
            'orientation': 'top',
            'distance_sort': 'ascending'
        }

        if truncate_mode:
            dendrogram_params['truncate_mode'] = truncate_mode
            dendrogram_params['p'] = p

        dendro_data = dendrogram(linkage_matrix, **dendrogram_params)

        if mostrar_linea_corte and altura_corte:
            ax_dendro.axhline(y=altura_corte, color='red', linestyle='--',
                            linewidth=2, label=f'Corte: {altura_corte:.2f}', alpha=0.7)
            ax_dendro.legend(loc='best', frameon=True, shadow=True)

        ax_dendro.set_xlabel('Muestra (o tamaño del cluster)', fontsize=11, fontweight='bold')
        ax_dendro.set_ylabel('Distancia', fontsize=11, fontweight='bold')
        ax_dendro.set_title('Dendrograma de Clustering Jerárquico', fontsize=13, fontweight='bold')
        ax_dendro.grid(True, alpha=0.3, axis='y', linestyle='--')

    except ImportError:
        ax_dendro.text(0.5, 0.5, '⚠️ Scipy requerido para dendrograma completo',
                      ha='center', va='center', transform=ax_dendro.transAxes)
        ax_dendro.axis('off')

    # 2. Información de distancias
    ax_dist = fig.add_subplot(gs[1, 0])
    distancias = linkage_matrix[:, 2]

    ax_dist.hist(distancias, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax_dist.axvline(np.median(distancias), color='red', linestyle='--',
                    linewidth=2, label=f'Mediana: {np.median(distancias):.2f}')
    if altura_corte:
        ax_dist.axvline(altura_corte, color='green', linestyle='--',
                        linewidth=2, label=f'Corte: {altura_corte:.2f}')
    ax_dist.set_xlabel('Distancia de Fusión', fontsize=10, fontweight='bold')
    ax_dist.set_ylabel('Frecuencia', fontsize=10, fontweight='bold')
    ax_dist.set_title('Distribución de Distancias', fontsize=11, fontweight='bold')
    ax_dist.legend(loc='best', frameon=True, shadow=True)
    ax_dist.grid(True, alpha=0.3, axis='y', linestyle='--')

    # 3. Estadísticas
    ax_stats = fig.add_subplot(gs[1, 1])

    stats_text = f"📊 ESTADÍSTICAS DEL DENDROGRAMA\n"
    stats_text += f"{'=' * 35}\n\n"
    stats_text += f"• Muestras: {n_samples}\n"
    stats_text += f"• Fusiones: {len(linkage_matrix)}\n"
    stats_text += f"• Distancia mín: {np.min(distancias):.3f}\n"
    stats_text += f"• Distancia máx: {np.max(distancias):.3f}\n"
    stats_text += f"• Distancia media: {np.mean(distancias):.3f}\n"
    stats_text += f"• Distancia mediana: {np.median(distancias):.3f}\n\n"

    if altura_corte:
        n_clusters_estimado = len([d for d in distancias if d > altura_corte]) + 1
        stats_text += f"• Clusters (corte): ~{n_clusters_estimado}\n"

    ax_stats.text(0.05, 0.95, stats_text, fontsize=9, va='top', ha='left',
                  transform=ax_stats.transAxes, family='monospace',
                  bbox=dict(boxstyle='round,pad=1', facecolor='lightgreen',
                            alpha=0.8, edgecolor='black'))
    ax_stats.set_title('Información', fontsize=11, fontweight='bold')
    ax_stats.axis('off')

    plt.suptitle('Dendrograma de Clustering Jerárquico',
                 fontsize=16, fontweight='bold')

    return fig


# ==================== FUNCIÓN DEMO ====================

def demo_ml_no_supervisado():
    """Demostración completa del sistema ML no supervisado"""
    print("🚀 Generando datos de demostración...")
    datos = generar_datos_agua_realistas(n_muestras=300, incluir_outliers=True)

    print("📊 Datos generados exitosamente")
    print(f"   Shape: {datos.shape}")

    variables_analisis = ['pH', 'WT', 'DO', 'TBD', 'CTD', 'BOD5', 'COD', 'FC', 'TC', 'NO3']

    print("\n⚡ Ejemplo 1: K-Means")
    start = time.time()
    kmeans_result = kmeans_optimizado_completo(datos, variables_analisis)
    print(f"   ⏱️ Tiempo: {time.time() - start:.2f}s")
    print(f"   K recomendado: {kmeans_result['recomendacion_k']}")

    print("\n⚡ Ejemplo 2: DBSCAN")
    start = time.time()
    dbscan_result = dbscan_optimizado(datos, variables_analisis)
    print(f"   ⏱️ Tiempo: {time.time() - start:.2f}s")
    mejor_dbscan = dbscan_result['mejor_configuracion']
    print(f"   Clusters: {mejor_dbscan['n_clusters']}, Outliers: {mejor_dbscan['n_noise']}")

    print("\n⚡ Ejemplo 3: PCA")
    start = time.time()
    pca_result = pca_completo_avanzado(datos, variables_analisis)
    print(f"   ⏱️ Tiempo: {time.time() - start:.2f}s")
    pca_linear = pca_result['resultados_por_metodo']['linear']
    print(f"   Componentes recomendados: {pca_linear['componentes_recomendados']}")

    print("\n⚡ Ejemplo 4: Clustering Jerárquico")
    start = time.time()
    jerarquico = clustering_jerarquico_completo(datos, variables_analisis)
    print(f"   ⏱️ Tiempo: {time.time() - start:.2f}s")
    print(f"   K recomendado: {jerarquico['mejor_configuracion']['n_clusters_sugeridos']}")

    print("\n⚡ Ejemplo 5: Análisis Exploratorio")
    start = time.time()
    exploratorio = analisis_exploratorio_completo(datos, variables_analisis)
    print(f"   ⏱️ Tiempo: {time.time() - start:.2f}s")
    print(f"   Calidad de datos: {exploratorio['calidad_datos']['calificacion']}")

    print("\n✅ Análisis completado exitosamente")

    return datos, {
        'kmeans': kmeans_result,
        'dbscan': dbscan_result,
        'pca': pca_result,
        'jerarquico': jerarquico,
        'exploratorio': exploratorio
    }


# ==================== MAIN ====================

if __name__ == "__main__":
    datos, resultados = demo_ml_no_supervisado()
    print(f"\n📊 Datos analizados: {len(datos)} muestras")
    print(f"🎯 Métodos ejecutados: {len(resultados)} técnicas")
    print(f"\n✅ Compatible con PyInstaller - Sin threading")
    print(f"🚀 Listo para usar en producción")