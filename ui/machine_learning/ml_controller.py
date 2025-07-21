"""
ml_controller.py - Controlador principal de Machine Learning
Maneja toda la lógica de ML separada de la interfaz
"""
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score,
                           confusion_matrix, classification_report, silhouette_score)
from typing import Dict, List, Tuple, Any, Optional
import joblib
import json
from datetime import datetime


class MLController:
    """Controlador principal para operaciones de Machine Learning"""

    def __init__(self):
        self.data = None
        self.processed_data = None
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.feature_names = []
        self.target_name = None

    # ==================== Manejo de Datos ====================

    def load_data(self, file_path: str) -> Dict[str, Any]:
        """Cargar datos desde archivo CSV"""
        try:
            self.data = pd.read_csv(file_path)

            # Análisis básico
            info = {
                'shape': self.data.shape,
                'columns': list(self.data.columns),
                'dtypes': self.data.dtypes.to_dict(),
                'missing_values': self.data.isnull().sum().to_dict(),
                'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(self.data.select_dtypes(include=['object']).columns),
                'summary': self.data.describe().to_dict()
            }

            return {'success': True, 'info': info}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def preprocess_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocesar datos según configuración"""
        try:
            df = self.data.copy()

            # Manejo de valores faltantes
            if config.get('handle_missing'):
                strategy = config.get('missing_strategy', 'drop')
                if strategy == 'drop':
                    df = df.dropna()
                elif strategy == 'mean':
                    df.fillna(df.mean(numeric_only=True), inplace=True)
                elif strategy == 'median':
                    df.fillna(df.median(numeric_only=True), inplace=True)
                elif strategy == 'forward_fill':
                    df.fillna(method='ffill', inplace=True)

            # Codificación de variables categóricas
            if config.get('encode_categorical'):
                categorical_cols = df.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    if col != config.get('target_column'):
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))

            # Normalización/Estandarización
            if config.get('scale_features'):
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                target_col = config.get('target_column')

                if target_col and target_col in numeric_cols:
                    feature_cols = [col for col in numeric_cols if col != target_col]
                else:
                    feature_cols = numeric_cols

                scaler_type = config.get('scaler_type', 'standard')
                if scaler_type == 'standard':
                    scaler = StandardScaler()
                else:
                    scaler = MinMaxScaler()

                df[feature_cols] = scaler.fit_transform(df[feature_cols])
                self.scalers['features'] = scaler

            self.processed_data = df

            return {
                'success': True,
                'shape': df.shape,
                'columns': list(df.columns)
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def split_data(self, target_column: str, test_size: float = 0.2,
                   random_state: int = 42) -> Dict[str, Any]:
        """Dividir datos en entrenamiento y prueba"""
        try:
            if self.processed_data is None:
                return {'success': False, 'error': 'No hay datos procesados'}

            # Separar características y objetivo
            X = self.processed_data.drop(columns=[target_column])
            y = self.processed_data[target_column]

            # Guardar nombres de características
            self.feature_names = list(X.columns)
            self.target_name = target_column

            # División de datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            return {
                'success': True,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ==================== Modelos Supervisados ====================

    def train_regression_model(self, model_type: str, X_train, y_train,
                             X_test, y_test, params: Dict = None) -> Dict[str, Any]:
        """Entrenar modelo de regresión"""
        try:
            from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
            from sklearn.svm import SVR
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.neural_network import MLPRegressor

            # Seleccionar modelo
            models = {
                'linear': LinearRegression(),
                'ridge': Ridge(),
                'lasso': Lasso(),
                'elastic_net': ElasticNet(),
                'svr': SVR(),
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(random_state=42),
                'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)
            }

            model = models.get(model_type)
            if not model:
                return {'success': False, 'error': f'Modelo {model_type} no disponible'}

            # Aplicar parámetros personalizados si existen
            if params:
                model.set_params(**params)

            # Entrenar modelo
            model.fit(X_train, y_train)

            # Predicciones
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Métricas
            metrics = {
                'train_mse': mean_squared_error(y_train, y_pred_train),
                'test_mse': mean_squared_error(y_test, y_pred_test),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test)
            }

            # Validación cruzada
            cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                      scoring='neg_mean_squared_error')
            metrics['cv_rmse'] = np.sqrt(-cv_scores.mean())
            metrics['cv_rmse_std'] = np.sqrt(cv_scores.std())

            # Importancia de características (si está disponible)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_names,
                                            model.feature_importances_))
            elif hasattr(model, 'coef_'):
                feature_importance = dict(zip(self.feature_names,
                                            np.abs(model.coef_)))

            # Guardar modelo
            self.models[f'regression_{model_type}'] = model

            return {
                'success': True,
                'model_type': model_type,
                'metrics': metrics,
                'predictions': {
                    'train': y_pred_train,
                    'test': y_pred_test
                },
                'feature_importance': feature_importance
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def train_classification_model(self, model_type: str, X_train, y_train,
                                 X_test, y_test, params: Dict = None) -> Dict[str, Any]:
        """Entrenar modelo de clasificación"""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.naive_bayes import GaussianNB
            from sklearn.neural_network import MLPClassifier

            # Seleccionar modelo
            models = {
                'logistic': LogisticRegression(max_iter=1000),
                'svm': SVC(probability=True),
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(random_state=42),
                'knn': KNeighborsClassifier(),
                'naive_bayes': GaussianNB(),
                'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
            }

            model = models.get(model_type)
            if not model:
                return {'success': False, 'error': f'Modelo {model_type} no disponible'}

            # Aplicar parámetros personalizados
            if params:
                model.set_params(**params)

            # Entrenar modelo
            model.fit(X_train, y_train)

            # Predicciones
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_proba_test = None

            if hasattr(model, 'predict_proba'):
                y_proba_test = model.predict_proba(X_test)

            # Métricas
            metrics = {
                'train_accuracy': accuracy_score(y_train, y_pred_train),
                'test_accuracy': accuracy_score(y_test, y_pred_test),
                'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist(),
                'classification_report': classification_report(y_test, y_pred_test,
                                                             output_dict=True)
            }

            # Validación cruzada
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            metrics['cv_accuracy'] = cv_scores.mean()
            metrics['cv_accuracy_std'] = cv_scores.std()

            # Importancia de características
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_names,
                                            model.feature_importances_))

            # Guardar modelo
            self.models[f'classification_{model_type}'] = model

            return {
                'success': True,
                'model_type': model_type,
                'metrics': metrics,
                'predictions': {
                    'train': y_pred_train,
                    'test': y_pred_test,
                    'probabilities': y_proba_test
                },
                'feature_importance': feature_importance
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ==================== Modelos No Supervisados ====================

    def perform_clustering(self, algorithm: str, X, params: Dict = None) -> Dict[str, Any]:
        """Realizar análisis de clustering"""
        try:
            from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
            from sklearn.mixture import GaussianMixture

            # Seleccionar algoritmo
            algorithms = {
                'kmeans': KMeans(n_clusters=params.get('n_clusters', 3), random_state=42),
                'dbscan': DBSCAN(eps=params.get('eps', 0.5),
                                min_samples=params.get('min_samples', 5)),
                'hierarchical': AgglomerativeClustering(n_clusters=params.get('n_clusters', 3)),
                'gaussian_mixture': GaussianMixture(n_components=params.get('n_components', 3),
                                                  random_state=42)
            }

            clusterer = algorithms.get(algorithm)
            if not clusterer:
                return {'success': False, 'error': f'Algoritmo {algorithm} no disponible'}

            # Aplicar parámetros adicionales
            if params and algorithm != 'kmeans':  # KMeans ya tiene params aplicados
                clusterer.set_params(**{k: v for k, v in params.items()
                                      if k in clusterer.get_params()})

            # Ajustar modelo
            labels = clusterer.fit_predict(X)

            # Métricas
            metrics = {}
            if len(np.unique(labels)) > 1:  # Solo si hay más de un cluster
                metrics['silhouette_score'] = silhouette_score(X, labels)

            # Análisis de clusters
            cluster_analysis = self._analyze_clusters(X, labels)

            # Guardar modelo
            self.models[f'clustering_{algorithm}'] = clusterer

            return {
                'success': True,
                'algorithm': algorithm,
                'labels': labels,
                'n_clusters': len(np.unique(labels)),
                'metrics': metrics,
                'cluster_analysis': cluster_analysis
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def perform_pca(self, n_components: int = None, variance_threshold: float = 0.95) -> Dict[str, Any]:
        """Realizar Análisis de Componentes Principales"""
        try:
            from sklearn.decomposition import PCA

            # Si no se especifica n_components, usar threshold de varianza
            if n_components is None:
                pca_temp = PCA()
                pca_temp.fit(self.processed_data)
                cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
                n_components = np.argmax(cumsum >= variance_threshold) + 1

            # Aplicar PCA
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(self.processed_data)

            # Resultados
            results = {
                'success': True,
                'n_components': n_components,
                'explained_variance': pca.explained_variance_,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
                'components': pca.components_,
                'transformed_data': X_pca,
                'feature_importance': self._get_pca_feature_importance(pca)
            }

            # Guardar modelo
            self.models['pca'] = pca

            return results

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def detect_anomalies(self, method: str = 'isolation_forest',
                        contamination: float = 0.1) -> Dict[str, Any]:
        """Detectar anomalías en los datos"""
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.neighbors import LocalOutlierFactor
            from sklearn.svm import OneClassSVM

            # Seleccionar método
            methods = {
                'isolation_forest': IsolationForest(contamination=contamination,
                                                  random_state=42),
                'lof': LocalOutlierFactor(contamination=contamination, novelty=True),
                'one_class_svm': OneClassSVM(nu=contamination)
            }

            detector = methods.get(method)
            if not detector:
                return {'success': False, 'error': f'Método {method} no disponible'}

            # Entrenar detector
            X = self.processed_data
            detector.fit(X)

            # Predicciones (-1 para anomalías, 1 para normales)
            predictions = detector.predict(X)
            scores = None

            if hasattr(detector, 'score_samples'):
                scores = detector.score_samples(X)
            elif hasattr(detector, 'decision_function'):
                scores = detector.decision_function(X)

            # Análisis de anomalías
            n_anomalies = np.sum(predictions == -1)
            anomaly_indices = np.where(predictions == -1)[0]

            return {
                'success': True,
                'method': method,
                'predictions': predictions,
                'scores': scores,
                'n_anomalies': n_anomalies,
                'anomaly_rate': n_anomalies / len(X),
                'anomaly_indices': anomaly_indices.tolist()
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ==================== Optimización de Hiperparámetros ====================

    def optimize_hyperparameters(self, model_type: str, model_class: str,
                                X_train, y_train, param_grid: Dict,
                                cv: int = 5, scoring: str = None) -> Dict[str, Any]:
        """Optimizar hiperparámetros usando GridSearchCV"""
        try:
            # Importar modelos necesarios
            if model_class == 'regression':
                from sklearn.linear_model import Ridge, Lasso
                from sklearn.svm import SVR
                from sklearn.ensemble import RandomForestRegressor

                models = {
                    'ridge': Ridge(),
                    'lasso': Lasso(),
                    'svr': SVR(),
                    'random_forest': RandomForestRegressor()
                }
                default_scoring = 'neg_mean_squared_error'

            else:  # classification
                from sklearn.linear_model import LogisticRegression
                from sklearn.svm import SVC
                from sklearn.ensemble import RandomForestClassifier

                models = {
                    'logistic': LogisticRegression(),
                    'svm': SVC(),
                    'random_forest': RandomForestClassifier()
                }
                default_scoring = 'accuracy'

            model = models.get(model_type)
            if not model:
                return {'success': False, 'error': f'Modelo {model_type} no disponible'}

            # Configurar scoring
            if scoring is None:
                scoring = default_scoring

            # Grid Search
            grid_search = GridSearchCV(
                model, param_grid, cv=cv, scoring=scoring,
                n_jobs=-1, verbose=1
            )

            grid_search.fit(X_train, y_train)

            # Resultados
            results = {
                'success': True,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': pd.DataFrame(grid_search.cv_results_).to_dict(),
                'best_estimator': grid_search.best_estimator_
            }

            # Guardar mejor modelo
            self.models[f'{model_class}_{model_type}_optimized'] = grid_search.best_estimator_

            return results

        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ==================== Utilidades ====================

    def _analyze_clusters(self, X, labels) -> Dict[str, Any]:
        """Analizar características de los clusters"""
        df_clusters = pd.DataFrame(X, columns=self.feature_names)
        df_clusters['cluster'] = labels

        analysis = {}
        for cluster in np.unique(labels):
            cluster_data = df_clusters[df_clusters['cluster'] == cluster]
            analysis[f'cluster_{cluster}'] = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df_clusters) * 100,
                'centroid': cluster_data[self.feature_names].mean().to_dict(),
                'std': cluster_data[self.feature_names].std().to_dict()
            }

        return analysis

    def _get_pca_feature_importance(self, pca) -> Dict[str, float]:
        """Calcular importancia de características en PCA"""
        # Calcular la contribución de cada característica
        importance = np.abs(pca.components_).mean(axis=0)
        return dict(zip(self.feature_names, importance))

    def save_model(self, model_name: str, filepath: str) -> bool:
        """Guardar modelo entrenado"""
        try:
            if model_name not in self.models:
                return False

            joblib.dump(self.models[model_name], filepath)

            # Guardar metadata
            metadata = {
                'model_name': model_name,
                'feature_names': self.feature_names,
                'target_name': self.target_name,
                'saved_date': datetime.now().isoformat()
            }

            with open(filepath.replace('.pkl', '_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)

            return True

        except Exception:
            return False

    def load_model(self, filepath: str) -> Dict[str, Any]:
        """Cargar modelo guardado"""
        try:
            model = joblib.load(filepath)

            # Cargar metadata
            metadata_path = filepath.replace('.pkl', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            model_name = metadata.get('model_name', 'loaded_model')
            self.models[model_name] = model

            if 'feature_names' in metadata:
                self.feature_names = metadata['feature_names']
            if 'target_name' in metadata:
                self.target_name = metadata['target_name']

            return {'success': True, 'model_name': model_name, 'metadata': metadata}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def export_results(self, results_name: str, filepath: str,
                      format: str = 'json') -> bool:
        """Exportar resultados del análisis"""
        try:
            if results_name not in self.results:
                return False

            results = self.results[results_name]

            if format == 'json':
                with open(filepath, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            elif format == 'csv' and 'dataframe' in results:
                results['dataframe'].to_csv(filepath, index=False)

            return True

        except Exception:
            return False