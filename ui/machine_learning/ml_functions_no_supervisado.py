"""
ml_functions_no_supervisado.py - Funciones de Machine Learning No Supervisado
Contiene funciones para clustering y PCA
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


def generar_datos_agua_optimizado(n_muestras=100, seed=42):
    """Generar datos sintéticos de calidad del agua"""
    np.random.seed(seed)

    datos = {
        'pH': np.clip(np.random.normal(7.2, 0.8, n_muestras), 6.0, 8.5),
        'Oxígeno_Disuelto': np.clip(np.random.normal(8.5, 1.5, n_muestras), 4.0, 12.0),
        'Turbidez': np.clip(np.random.exponential(2.0, n_muestras), 0.1, 8.0),
        'Conductividad': np.clip(np.random.normal(500, 200, n_muestras), 100, 1200)
    }

    # Calcular scores de calidad vectorizados
    calidad_scores = np.zeros(n_muestras)

    # pH score
    ph_score = np.where(
        (datos['pH'] >= 6.5) & (datos['pH'] <= 8.5),
        25,
        np.maximum(0, 25 - np.abs(datos['pH'] - 7.0) * 8)
    )

    # Oxígeno score
    oxigeno_score = np.where(
        datos['Oxígeno_Disuelto'] >= 6,
        25,
        datos['Oxígeno_Disuelto'] * 4
    )

    # Turbidez score
    turbidez_score = np.where(
        datos['Turbidez'] < 4,
        25,
        np.maximum(0, 25 - (datos['Turbidez'] - 4) * 5)
    )

    # Conductividad score
    conductividad_score = np.where(
        (datos['Conductividad'] >= 200) & (datos['Conductividad'] <= 800),
        25,
        np.maximum(0, 25 - np.abs(datos['Conductividad'] - 500) * 0.03)
    )

    calidad_scores = ph_score + oxigeno_score + turbidez_score + conductividad_score

    # Categorías de calidad
    calidades = np.select(
        [calidad_scores >= 80, calidad_scores >= 60, calidad_scores >= 40],
        ["Excelente", "Buena", "Regular"],
        default="Necesita Tratamiento"
    )

    datos['Calidad_Score'] = calidad_scores
    datos['Calidad'] = calidades

    return pd.DataFrame(datos)


def clustering_proceso(n_muestras=150):
    """Análisis de clustering con K-Means y Jerárquico"""
    try:
        df = generar_datos_agua_optimizado(n_muestras)
        X = df[['pH', 'Oxígeno_Disuelto', 'Turbidez', 'Conductividad']].values

        # Estandarizar datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # K-Means clustering
        kmeans_results = {}
        for k in range(2, 7):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_kmeans = kmeans.fit_predict(X_scaled)
            silhouette_kmeans = silhouette_score(X_scaled, labels_kmeans)

            kmeans_results[k] = {
                'labels': labels_kmeans.tolist(),
                'silhouette_score': float(silhouette_kmeans),
                'inertia': float(kmeans.inertia_),
                'centroids': kmeans.cluster_centers_.tolist()
            }

        # Clustering jerárquico
        distances = pdist(X_scaled)
        linkage_matrix = linkage(distances, method='ward')

        hierarchical_results = {}
        for k in range(2, 7):
            labels_hier = fcluster(linkage_matrix, k, criterion='maxclust') - 1
            silhouette_hier = silhouette_score(X_scaled, labels_hier)

            hierarchical_results[k] = {
                'labels': labels_hier.tolist(),
                'silhouette_score': float(silhouette_hier)
            }

        # Mejor configuración
        best_k_kmeans = max(kmeans_results.keys(),
                            key=lambda k: kmeans_results[k]['silhouette_score'])
        best_k_hier = max(hierarchical_results.keys(),
                          key=lambda k: hierarchical_results[k]['silhouette_score'])

        # Análisis de clusters del mejor K-Means
        df['Cluster_KMeans'] = kmeans_results[best_k_kmeans]['labels']

        analisis_clusters = {}
        for cluster in range(best_k_kmeans):
            cluster_data = df[df['Cluster_KMeans'] == cluster]

            analisis_clusters[f"Cluster {cluster}"] = {
                'tamaño': int(len(cluster_data)),
                'ph_promedio': float(cluster_data['pH'].mean()),
                'oxigeno_promedio': float(cluster_data['Oxígeno_Disuelto'].mean()),
                'turbidez_promedio': float(cluster_data['Turbidez'].mean()),
                'conductividad_promedio': float(cluster_data['Conductividad'].mean()),
                'calidad_promedio': float(cluster_data['Calidad_Score'].mean())
            }

        return {
            'tipo': 'clustering',
            'kmeans_results': kmeans_results,
            'hierarchical_results': hierarchical_results,
            'best_k_kmeans': best_k_kmeans,
            'best_k_hierarchical': best_k_hier,
            'linkage_matrix': linkage_matrix.tolist(),
            'analisis_clusters': analisis_clusters,
            'datos': df.to_dict('records'),
            'scaled_data': X_scaled.tolist(),
            'feature_names': ['pH', 'Oxígeno_Disuelto', 'Turbidez', 'Conductividad']
        }
    except Exception as e:
        return {'error': str(e)}


def pca_proceso(n_muestras=150):
    """Análisis de Componentes Principales"""
    try:
        df = generar_datos_agua_optimizado(n_muestras)
        X = df[['pH', 'Oxígeno_Disuelto', 'Turbidez', 'Conductividad']].values

        # Estandarizar datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA completo
        pca_full = PCA()
        X_pca_full = pca_full.fit_transform(X_scaled)

        # PCA con 2 componentes para visualización
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X_scaled)

        # PCA con 3 componentes
        pca_3d = PCA(n_components=3)
        X_pca_3d = pca_3d.fit_transform(X_scaled)

        # Análisis de componentes
        componentes_principales = []
        feature_names = ['pH', 'Oxígeno_Disuelto', 'Turbidez', 'Conductividad']

        for i in range(min(4, len(pca_full.components_))):
            componente = {
                'componente': f'PC{i + 1}',
                'varianza_explicada': float(pca_full.explained_variance_ratio_[i]),
                'contribuciones': {}
            }

            for j, feature in enumerate(feature_names):
                componente['contribuciones'][feature] = float(pca_full.components_[i][j])

            componentes_principales.append(componente)

        # Varianza acumulada
        varianza_acumulada = np.cumsum(pca_full.explained_variance_ratio_)

        # Determinar número óptimo de componentes (85% de varianza)
        n_componentes_85 = np.argmax(varianza_acumulada >= 0.85) + 1

        # Reconstrucción con diferentes números de componentes
        reconstrucciones = {}
        for n_comp in [1, 2, 3, 4]:
            if n_comp <= len(pca_full.components_):
                pca_temp = PCA(n_components=n_comp)
                X_temp = pca_temp.fit_transform(X_scaled)
                X_reconstructed = pca_temp.inverse_transform(X_temp)

                # Error de reconstrucción
                mse_recon = mean_squared_error(X_scaled, X_reconstructed)
                var_retenida = sum(pca_temp.explained_variance_ratio_)

                reconstrucciones[n_comp] = {
                    'varianza_retenida': float(var_retenida),
                    'error_reconstruccion': float(mse_recon)
                }

        return {
            'tipo': 'pca',
            'componentes_principales': componentes_principales,
            'varianza_explicada': [float(v) for v in pca_full.explained_variance_ratio_],
            'varianza_acumulada': [float(v) for v in varianza_acumulada],
            'n_componentes_85': int(n_componentes_85),
            'datos_transformados_2d': X_pca_2d.tolist(),
            'datos_transformados_3d': X_pca_3d.tolist(),
            'reconstrucciones': reconstrucciones,
            'datos_originales': df.to_dict('records'),
            'feature_names': feature_names
        }
    except Exception as e:
        return {'error': str(e)}