"""
ml_functions_supervisado.py - Sistema de Machine Learning Supervisado
Versión limpia y completa compatible con PyInstaller
Implementación 100% NumPy para evitar dependencias problemáticas
"""

# ==================== IMPORTACIONES ====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime
import json
import os
import sys

# Configuración de warnings y matplotlib
warnings.filterwarnings('ignore')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


# ==================== FUNCIONES UTILITARIAS SIN SKLEARN ====================

def manual_train_test_split(X, y, test_size=0.2, random_state=42):
    """División train/test manual sin sklearn"""
    np.random.seed(random_state)
    n_samples = len(X)
    n_test = int(n_samples * test_size)

    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_indices].reset_index(drop=True)
        X_test = X.iloc[test_indices].reset_index(drop=True)
    else:
        X_train = X[train_indices]
        X_test = X[test_indices]

    if isinstance(y, pd.Series):
        y_train = y.iloc[train_indices].reset_index(drop=True)
        y_test = y.iloc[test_indices].reset_index(drop=True)
    else:
        y_train = y[train_indices]
        y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def manual_standard_scaler(X_train, X_test=None):
    """Escalado estándar manual sin sklearn"""
    if isinstance(X_train, pd.DataFrame):
        mean = X_train.mean()
        std = X_train.std() + 1e-8
        X_train_scaled = (X_train - mean) / std

        if X_test is not None:
            X_test_scaled = (X_test - mean) / std
            return X_train_scaled, X_test_scaled, {'mean': mean, 'std': std}
        return X_train_scaled, None, {'mean': mean, 'std': std}
    else:
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0) + 1e-8
        X_train_scaled = (X_train - mean) / std

        if X_test is not None:
            X_test_scaled = (X_test - mean) / std
            return X_train_scaled, X_test_scaled, {'mean': mean, 'std': std}
        return X_train_scaled, None, {'mean': mean, 'std': std}


def manual_accuracy_score(y_true, y_pred):
    """Calcular accuracy manualmente"""
    return np.mean(y_true == y_pred)


def manual_r2_score(y_true, y_pred):
    """Calcular R² manualmente"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


def manual_confusion_matrix(y_true, y_pred):
    """Generar matriz de confusión manualmente"""
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    n_labels = len(unique_labels)
    cm = np.zeros((n_labels, n_labels), dtype=int)

    label_to_idx = {label: i for i, label in enumerate(unique_labels)}

    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = label_to_idx[true_label]
        pred_idx = label_to_idx[pred_label]
        cm[true_idx, pred_idx] += 1

    return cm, unique_labels


def manual_classification_report(y_true, y_pred):
    """Generar reporte de clasificación completo manualmente"""
    cm, labels = manual_confusion_matrix(y_true, y_pred)
    report = {}

    # Métricas por clase
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        support = np.sum(cm[i, :])

        report[str(label)] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': support
        }

    # Métricas generales
    total_support = len(y_true)

    # Promedios macro y weighted
    macro_precision = np.mean([report[str(label)]['precision'] for label in labels])
    macro_recall = np.mean([report[str(label)]['recall'] for label in labels])
    macro_f1 = np.mean([report[str(label)]['f1-score'] for label in labels])

    weighted_precision = np.sum(
        [report[str(label)]['precision'] * report[str(label)]['support'] for label in labels]) / total_support
    weighted_recall = np.sum(
        [report[str(label)]['recall'] * report[str(label)]['support'] for label in labels]) / total_support
    weighted_f1 = np.sum(
        [report[str(label)]['f1-score'] * report[str(label)]['support'] for label in labels]) / total_support

    report['accuracy'] = manual_accuracy_score(y_true, y_pred)
    report['macro avg'] = {
        'precision': macro_precision,
        'recall': macro_recall,
        'f1-score': macro_f1,
        'support': total_support
    }
    report['weighted avg'] = {
        'precision': weighted_precision,
        'recall': weighted_recall,
        'f1-score': weighted_f1,
        'support': total_support
    }

    return report


# ==================== MODELOS DE MACHINE LEARNING EN NUMPY ====================

class NumpyLinearRegression:
    """Regresión lineal implementada en NumPy puro"""

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.is_fitted = False

    def fit(self, X, y):
        """Entrenar modelo usando mínimos cuadrados"""
        X = np.array(X)
        y = np.array(y)

        # Añadir columna de unos para el intercepto
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

        try:
            # Ecuación normal: (X^T X)^-1 X^T y
            XtX = np.dot(X_with_intercept.T, X_with_intercept)
            Xty = np.dot(X_with_intercept.T, y)
            params = np.linalg.solve(XtX, Xty)

            self.intercept_ = params[0]
            self.coef_ = params[1:] if len(params) > 1 else params[1]
            self.is_fitted = True

        except np.linalg.LinAlgError:
            # Fallback usando pseudoinversa
            params = np.linalg.pinv(X_with_intercept).dot(y)
            self.intercept_ = params[0]
            self.coef_ = params[1:] if len(params) > 1 else params[1]
            self.is_fitted = True

        return self

    def predict(self, X):
        """Hacer predicciones"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")

        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return np.dot(X, self.coef_) + self.intercept_


class NumpyDecisionTree:
    """Árbol de decisión implementado en NumPy puro"""

    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        self.is_fitted = False
        self.feature_importances_ = None

    def _gini_impurity(self, y):
        """Calcular impureza de Gini"""
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _variance(self, y):
        """Calcular varianza para regresión"""
        if len(y) == 0:
            return 0
        return np.var(y)

    def _best_split(self, X, y, is_regression=False):
        """Encontrar la mejor división"""
        best_gain = 0
        best_feature = None
        best_threshold = None
        n_features = X.shape[1]

        # Calcular impureza inicial
        initial_impurity = self._variance(y) if is_regression else self._gini_impurity(y)

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds[:-1]:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                # Calcular ganancia de información
                if is_regression:
                    left_impurity = self._variance(y[left_mask])
                    right_impurity = self._variance(y[right_mask])
                else:
                    left_impurity = self._gini_impurity(y[left_mask])
                    right_impurity = self._gini_impurity(y[right_mask])

                n_left, n_right = np.sum(left_mask), np.sum(right_mask)
                weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / len(y)
                gain = initial_impurity - weighted_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth=0):
        """Construir árbol recursivamente"""
        n_samples = len(y)
        is_regression = len(np.unique(y)) > 10 or np.issubdtype(y.dtype, np.floating)

        # Criterios de parada
        if (depth >= self.max_depth or
                n_samples < self.min_samples_split or
                len(np.unique(y)) == 1):

            if is_regression:
                return {'type': 'leaf', 'value': np.mean(y), 'samples': n_samples}
            else:
                values, counts = np.unique(y, return_counts=True)
                majority_class = values[np.argmax(counts)]
                return {'type': 'leaf', 'value': majority_class, 'samples': n_samples}

        # Encontrar mejor división
        feature, threshold, gain = self._best_split(X, y, is_regression)

        if feature is None or gain <= 0:
            if is_regression:
                return {'type': 'leaf', 'value': np.mean(y), 'samples': n_samples}
            else:
                values, counts = np.unique(y, return_counts=True)
                majority_class = values[np.argmax(counts)]
                return {'type': 'leaf', 'value': majority_class, 'samples': n_samples}

        # Dividir datos y construir subárboles
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'type': 'split',
            'feature': feature,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree,
            'samples': n_samples
        }

    def fit(self, X, y):
        """Entrenar árbol de decisión"""
        X = np.array(X)
        y = np.array(y)

        self.tree = self._build_tree(X, y)
        self.is_fitted = True

        # Calcular importancia de características
        self.feature_importances_ = np.zeros(X.shape[1])
        self._calculate_feature_importance(self.tree, X.shape[1])

        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)

        return self

    def _calculate_feature_importance(self, node, n_features):
        """Calcular importancia de características"""
        if node['type'] == 'split':
            feature = node['feature']
            self.feature_importances_[feature] += node['samples']
            self._calculate_feature_importance(node['left'], n_features)
            self._calculate_feature_importance(node['right'], n_features)

    def _predict_single(self, x, node):
        """Predecir una muestra individual"""
        if node['type'] == 'leaf':
            return node['value']

        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])

    def predict(self, X):
        """Hacer predicciones"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")

        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = []
        for x in X:
            pred = self._predict_single(x, self.tree)
            predictions.append(pred)

        return np.array(predictions)


class NumpyRandomForest:
    """Random Forest implementado en NumPy puro"""

    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2,
                 max_features='sqrt', random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None
        self.is_fitted = False

    def _bootstrap_sample(self, X, y, random_state):
        """Crear muestra bootstrap"""
        np.random.seed(random_state)
        n_samples = len(X)
        indices = np.random.choice(n_samples, n_samples, replace=True)

        if isinstance(X, pd.DataFrame):
            return X.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True)
        else:
            return X[indices], y[indices]

    def _get_feature_subset(self, n_features, random_state):
        """Seleccionar subconjunto de características"""
        np.random.seed(random_state)

        if self.max_features == 'sqrt':
            n_selected = max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            n_selected = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            n_selected = min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            n_selected = max(1, int(self.max_features * n_features))
        else:
            n_selected = n_features

        return np.random.choice(n_features, n_selected, replace=False)

    def fit(self, X, y):
        """Entrenar Random Forest"""
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y

        n_samples, n_features = X.shape
        self.trees = []
        total_importances = np.zeros(n_features)

        for i in range(self.n_estimators):
            # Bootstrap sampling
            X_boot, y_boot = self._bootstrap_sample(X, y, self.random_state + i)

            # Feature selection
            feature_indices = self._get_feature_subset(n_features, self.random_state + i)
            X_subset = X_boot[:, feature_indices]

            # Crear y entrenar árbol
            tree = NumpyDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_subset, y_boot)

            # Guardar árbol con sus características
            self.trees.append({
                'tree': tree,
                'feature_indices': feature_indices
            })

            # Acumular importancias
            tree_importances = np.zeros(n_features)
            tree_importances[feature_indices] = tree.feature_importances_
            total_importances += tree_importances

        # Normalizar importancias
        self.feature_importances_ = total_importances / self.n_estimators
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)

        self.is_fitted = True
        return self

    def predict(self, X):
        """Hacer predicciones usando votación mayoritaria"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")

        X = np.array(X) if not isinstance(X, np.ndarray) else X
        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = []

        for x in X:
            tree_predictions = []

            for tree_info in self.trees:
                tree = tree_info['tree']
                feature_indices = tree_info['feature_indices']
                x_subset = x[feature_indices]
                pred = tree.predict(x_subset.reshape(1, -1))[0]
                tree_predictions.append(pred)

            # Votación mayoritaria
            unique_preds, counts = np.unique(tree_predictions, return_counts=True)
            majority_pred = unique_preds[np.argmax(counts)]
            predictions.append(majority_pred)

        return np.array(predictions)


# 1. Primero modifica la clase NumpySVM para extraer los coeficientes

class NumpySVM:
    """SVM simplificado usando descenso por gradiente con extracción de importancia"""

    def __init__(self, C=1.0, max_iter=1000, learning_rate=0.01, kernel='linear'):
        self.C = C
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.kernel = kernel
        self.w = None
        self.b = None
        self.classes_ = None
        self.n_support_ = 0
        self.is_fitted = False
        self.feature_importance_ = None  # ← NUEVA PROPIEDAD

    def fit(self, X, y):
        """Entrenar SVM usando descenso por gradiente"""
        X = np.array(X)
        y = np.array(y)

        # Verificar clasificación binaria
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError("SVM solo soporta clasificación binaria")

        # Convertir etiquetas a -1 y 1
        y_svm = np.where(y == unique_labels[0], -1, 1)
        self.classes_ = unique_labels

        n_samples, n_features = X.shape
        self.w = np.random.normal(0, 0.01, n_features)
        self.b = 0

        # Entrenamiento
        support_count = 0
        for iteration in range(self.max_iter):
            learning_rate = self.learning_rate / (1 + 0.01 * iteration)

            for i in range(n_samples):
                margin = y_svm[i] * (np.dot(X[i], self.w) + self.b)

                if margin < 1:
                    support_count += 1
                    self.w -= learning_rate * (2 * (1 / self.max_iter) * self.w - self.C * y_svm[i] * X[i])
                    self.b -= learning_rate * (-self.C * y_svm[i])
                else:
                    self.w -= learning_rate * (2 * (1 / self.max_iter) * self.w)

        self.n_support_ = support_count // self.max_iter

        # NUEVA: Calcular importancia de características basada en coeficientes
        self.feature_importance_ = np.abs(self.w)  # Valor absoluto de los coeficientes
        # Normalizar para que sumen 1
        if np.sum(self.feature_importance_) > 0:
            self.feature_importance_ = self.feature_importance_ / np.sum(self.feature_importance_)

        self.is_fitted = True
        return self

    def predict(self, X):
        """Hacer predicciones"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")

        X = np.array(X)
        scores = np.dot(X, self.w) + self.b
        binary_pred = np.where(scores >= 0, 1, -1)

        return np.where(binary_pred == 1, self.classes_[1], self.classes_[0])

    def get_feature_importance(self, feature_names=None):
        """Obtener importancia de características con nombres"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")

        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(self.feature_importance_))]

        return dict(zip(feature_names, self.feature_importance_))


# 2. Modifica la función svm_modelo para incluir importancia

def svm_modelo(data: pd.DataFrame, target_column: str,
               feature_columns: Optional[List[str]] = None,
               C: float = 1.0, kernel: str = 'linear') -> Dict[str, Any]:
    """Support Vector Machine con importancia de características"""
    try:
        prep_data = preparar_datos_supervisado_optimizado(
            data, target_column, feature_columns
        )

        if not prep_data['is_classification']:
            raise ValueError("SVM solo soporta clasificación")

        # Verificar número de clases
        unique_classes = np.unique(prep_data['y_train'])
        n_classes = len(unique_classes)

        if n_classes < 2:
            raise ValueError("Se necesitan al menos 2 clases para clasificación")
        elif n_classes == 2:
            # Clasificación binaria normal
            model = NumpySVM(C=C, kernel=kernel)
            model.fit(prep_data['X_train_scaled'], prep_data['y_train'])
            y_pred_train = model.predict(prep_data['X_train_scaled'])
            y_pred_test = model.predict(prep_data['X_test_scaled'])

            # Obtener importancia de características
            importancia_features = model.get_feature_importance(prep_data['feature_columns'])
            n_support = model.n_support_

        else:
            # Clasificación multiclase usando One-vs-Rest
            print(f"Detectadas {n_classes} clases. Usando estrategia One-vs-Rest...")
            models = {}
            y_pred_train_proba = np.zeros((len(prep_data['y_train']), n_classes))
            y_pred_test_proba = np.zeros((len(prep_data['y_test']), n_classes))

            # Para almacenar importancia agregada
            total_importance = np.zeros(len(prep_data['feature_columns']))

            # Entrenar un clasificador binario para cada clase
            for i, class_label in enumerate(unique_classes):
                # Crear problema binario: clase actual vs todas las demás
                y_binary_train = (prep_data['y_train'] == class_label).astype(int)

                # Entrenar modelo binario
                binary_model = NumpySVM(C=C, kernel=kernel, max_iter=500)
                binary_model.fit(prep_data['X_train_scaled'], y_binary_train)

                # Predicciones
                train_scores = binary_model.predict(prep_data['X_train_scaled'])
                test_scores = binary_model.predict(prep_data['X_test_scaled'])

                y_pred_train_proba[:, i] = train_scores
                y_pred_test_proba[:, i] = test_scores

                # Acumular importancia
                total_importance += binary_model.feature_importance_

                models[class_label] = binary_model

            # Predicción final: clase con mayor "score"
            y_pred_train = unique_classes[np.argmax(y_pred_train_proba, axis=1)]
            y_pred_test = unique_classes[np.argmax(y_pred_test_proba, axis=1)]

            # Promedio de importancia de todos los modelos
            avg_importance = total_importance / n_classes
            # Normalizar
            if np.sum(avg_importance) > 0:
                avg_importance = avg_importance / np.sum(avg_importance)

            importancia_features = dict(zip(prep_data['feature_columns'], avg_importance))

            # Crear objeto modelo combinado
            model = {'type': 'multiclass_svm', 'models': models, 'classes': unique_classes}
            n_support = sum([getattr(m, 'n_support_', 0) for m in models.values()])

        # Calcular métricas
        train_acc = manual_accuracy_score(prep_data['y_train'], y_pred_train)
        test_acc = manual_accuracy_score(prep_data['y_test'], y_pred_test)

        train_report = manual_classification_report(prep_data['y_train'], y_pred_train)
        test_report = manual_classification_report(prep_data['y_test'], y_pred_test)

        train_cm, train_labels = manual_confusion_matrix(prep_data['y_train'], y_pred_train)
        test_cm, test_labels = manual_confusion_matrix(prep_data['y_test'], y_pred_test)

        return {
            'tipo': 'svm',
            'modelo': model,
            'es_clasificacion': True,
            'n_clases': n_classes,
            'target_column': target_column,
            'feature_columns': prep_data['feature_columns'],
            'importancia_features': importancia_features,  # ← NUEVA LÍNEA
            'metricas': {
                'train': {'accuracy': train_acc, 'classification_report': train_report},
                'test': {'accuracy': test_acc, 'classification_report': test_report}
            },
            'parametros': {
                'C': C,
                'kernel': kernel,
                'n_support_vectors': n_support,
                'estrategia': 'binario' if n_classes == 2 else 'one-vs-rest'
            },
            'datos_prediccion': {
                'y_train': prep_data['y_train'].tolist(),
                'y_test': prep_data['y_test'].tolist(),
                'y_pred_train': y_pred_train.tolist(),
                'y_pred_test': y_pred_test.tolist(),
                'train_cm': train_cm,
                'test_cm': test_cm,
                'cm_labels': test_labels
            }
        }

    except Exception as e:
        return {'error': str(e)}


# 3. Actualiza la visualización SVM para incluir importancia real

def _plot_svm_results(self):
    """Gráficos específicos para SVM con importancia de características real"""
    try:
        # Layout especializado para SVM (3x3)
        gs = self.figure.add_gridspec(3, 3, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)

        # 1. Importancia de características (NUEVA - REAL)
        ax1 = self.figure.add_subplot(gs[0, 0])
        importancia = self.current_results.get('importancia_features', {})

        if importancia:
            features = list(importancia.keys())[:8]  # Top 8
            values = [importancia[f] for f in features]

            # Colores específicos para SVM (azul-púrpura)
            colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(features)))

            y_pos = np.arange(len(features))
            bars = ax1.barh(y_pos, values, color=colors, alpha=0.8,
                            edgecolor='darkblue', linewidth=1)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(features, fontsize=9)
            ax1.set_xlabel('Peso en el Hiperplano (|w|)')
            ax1.set_title('🔍 Importancia en SVM\n(Coeficientes del Hiperplano)',
                          fontweight='bold', fontsize=10)
            ax1.grid(True, alpha=0.3, axis='x')

            # Añadir valores
            for bar, value in zip(bars, values):
                width = bar.get_width()
                ax1.text(width + 0.005, bar.get_y() + bar.get_height() / 2,
                         f'{value:.3f}', ha='left', va='center', fontsize=8)
        else:
            ax1.text(0.5, 0.5, 'Importancia no disponible\npara este modelo SVM',
                     ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('🔍 Importancia SVM', fontweight='bold')

        # 2. Panel de información SVM
        ax2 = self.figure.add_subplot(gs[0, 1:])
        ax2.axis('off')

        parametros = self.current_results.get('parametros', {})
        metricas = self.current_results.get('metricas', {})

        svm_info = [
            "🔍 SUPPORT VECTOR MACHINE - ANÁLISIS COMPLETO",
            "=" * 55,
            "",
            "CONFIGURACIÓN DEL MODELO:",
            f"  • Kernel: {parametros.get('kernel', 'linear').upper()}",
            f"  • Parámetro C: {parametros.get('C', 1.0)} (Regularización)",
            f"  • Vectores de Soporte: {parametros.get('n_support_vectors', 0)}",
            f"  • Estrategia: {parametros.get('estrategia', 'binario').title()}",
            "",
            "CARACTERÍSTICAS DEL HIPERPLANO:",
            f"  • Variables de entrada: {len(self.current_results.get('feature_columns', []))}",
            f"  • Variable objetivo: {self.current_results.get('target_column', 'N/A')}",
            f"  • Número de clases: {self.current_results.get('n_clases', 2)}",
        ]

        if metricas:
            train_acc = metricas.get('train', {}).get('accuracy', 0)
            test_acc = metricas.get('test', {}).get('accuracy', 0)
            generalization_gap = abs(train_acc - test_acc)

            svm_info.extend([
                "",
                "RENDIMIENTO:",
                f"  • Accuracy Entrenamiento: {train_acc:.4f}",
                f"  • Accuracy Prueba: {test_acc:.4f}",
                f"  • Brecha de generalización: {generalization_gap:.4f}",
            ])

            if generalization_gap < 0.05:
                svm_info.append("  • Estado: ✅ Excelente generalización")
            elif generalization_gap < 0.1:
                svm_info.append("  • Estado: 🟡 Buena generalización")
            else:
                svm_info.append("  • Estado: ⚠️ Revisar parámetros")

        # Explicación de la importancia
        if importancia:
            svm_info.extend([
                "",
                "INTERPRETACIÓN DE IMPORTANCIA:",
                "  • Basada en coeficientes del hiperplano (|w|)",
                "  • Mayor valor = mayor influencia en la decisión",
                "  • Representa la 'distancia' que contribuye cada variable",
            ])

        info_text = '\n'.join(svm_info)
        ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, fontsize=9,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='lightcyan', alpha=0.9,
                           edgecolor='darkblue', linewidth=2))

        # 3. Visualización conceptual del SVM
        ax3 = self.figure.add_subplot(gs[1, 0])
        self._draw_svm_concept_enhanced(ax3, parametros)

        # 4. Análisis de rendimiento
        ax4 = self.figure.add_subplot(gs[1, 1])
        if metricas:
            train_acc = metricas.get('train', {}).get('accuracy', 0)
            test_acc = metricas.get('test', {}).get('accuracy', 0)

            bars = ax4.bar(['Train', 'Test'], [train_acc, test_acc],
                           color=['#3498db', '#e74c3c'], alpha=0.8, width=0.6)

            # Líneas de referencia
            ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Aleatorio')
            ax4.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Bueno')

            ax4.set_ylabel('Accuracy')
            ax4.set_title('🎯 Rendimiento SVM', fontweight='bold')
            ax4.set_ylim(0, 1.1)
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.legend(fontsize=8)

            for bar, acc in zip(bars, [train_acc, test_acc]):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                         f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

        # 5. Matriz de confusión
        ax5 = self.figure.add_subplot(gs[1, 2])
        datos_pred = self.current_results.get('datos_prediccion', {})
        test_cm = datos_pred.get('test_cm')

        if test_cm is not None and len(test_cm) > 0:
            im = ax5.imshow(test_cm, interpolation='nearest', cmap='Blues', alpha=0.8)

            for i in range(len(test_cm)):
                for j in range(len(test_cm)):
                    text = ax5.text(j, i, f'{test_cm[i, j]}',
                                    ha='center', va='center', fontweight='bold', fontsize=12)

            ax5.set_xlabel('Predicho')
            ax5.set_ylabel('Real')
            ax5.set_title('🔢 Matriz de Confusión', fontweight='bold')

        # 6. Análisis comparativo de características
        ax6 = self.figure.add_subplot(gs[2, :])

        if importancia:
            # Comparar importancia con estadísticas descriptivas
            feature_names = list(importancia.keys())
            importance_values = list(importancia.values())

            # Crear gráfico de barras horizontal más grande
            y_pos = np.arange(len(feature_names))
            colors = plt.cm.plasma(np.array(importance_values) / max(importance_values))

            bars = ax6.barh(y_pos, importance_values, color=colors, alpha=0.8,
                            edgecolor='black', linewidth=0.5)

            ax6.set_yticks(y_pos)
            ax6.set_yticklabels(feature_names, fontsize=10)
            ax6.set_xlabel('Importancia Relativa en el Hiperplano SVM', fontsize=12)
            ax6.set_title('📊 Análisis Completo de Importancia de Características',
                          fontweight='bold', fontsize=12)
            ax6.grid(True, alpha=0.3, axis='x')

            # Añadir línea de promedio
            avg_importance = np.mean(importance_values)
            ax6.axvline(x=avg_importance, color='red', linestyle='--', linewidth=2,
                        label=f'Promedio: {avg_importance:.3f}')

            # Añadir valores y porcentajes
            for bar, value in zip(bars, importance_values):
                width = bar.get_width()
                percentage = (value / sum(importance_values)) * 100
                ax6.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{value:.3f} ({percentage:.1f}%)',
                         ha='left', va='center', fontsize=9, fontweight='bold')

            ax6.legend()

            # Información adicional
            max_feature = feature_names[np.argmax(importance_values)]
            min_feature = feature_names[np.argmin(importance_values)]

            info_box = f"Variable más influyente: {max_feature}\n"
            info_box += f"Variable menos influyente: {min_feature}\n"
            info_box += f"Ratio máx/mín: {max(importance_values) / min(importance_values):.2f}"

            ax6.text(0.02, 0.98, info_box, transform=ax6.transAxes,
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                     verticalalignment='top', fontsize=9)

        self.figure.suptitle('🔍 Análisis Integral de SVM con Importancia de Características',
                             fontsize=14, fontweight='bold')

    except Exception as e:
        ax = self.figure.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, f'Error en visualización de SVM:\n{str(e)}',
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')


# ==================== FUNCIONES PRINCIPALES DE PREPARACIÓN ====================

def preparar_datos_supervisado_optimizado(data: pd.DataFrame, target_column: str,
                                          feature_columns: Optional[List[str]] = None,
                                          test_size: float = 0.2, random_state: int = 42,
                                          scale_data: bool = True) -> Dict[str, Any]:
    """Preparar datos para ML supervisado"""
    try:
        df = data.copy()

        # Seleccionar características
        if feature_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in numeric_columns if col != target_column]

        if len(feature_columns) == 0:
            raise ValueError("No hay características numéricas disponibles")

        # Separar X e y
        X = df[feature_columns].copy()
        y = df[target_column].copy()

        # Limpiar valores faltantes
        X = X.fillna(X.mean())
        y = y.dropna()
        X = X.loc[y.index]

        # Determinar tipo de problema
        unique_values = y.nunique()
        is_classification = unique_values < 10 or y.dtype == 'object'

        # Codificar target categórico
        label_encoder = None
        if is_classification and y.dtype == 'object':
            unique_labels = y.unique()
            label_encoder = {label: i for i, label in enumerate(unique_labels)}
            y = y.map(label_encoder)

        # División de datos
        X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_size, random_state)

        # Escalado
        scaler = None
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        if scale_data:
            X_train_scaled, X_test_scaled, scaler = manual_standard_scaler(X_train, X_test)

        return {
            'X_train': X_train,
            'X_test': X_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': feature_columns,
            'target_column': target_column,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'is_classification': is_classification
        }

    except Exception as e:
        raise Exception(f"Error preparando datos: {str(e)}")


def verificar_datos(data: pd.DataFrame, target_column: str = None,
                    feature_columns: List[str] = None) -> Dict[str, Any]:
    """Verificar calidad de los datos"""
    issues = []
    warnings = []

    if data.empty:
        issues.append("El DataFrame está vacío")
        return {'valid': False, 'issues': issues, 'warnings': warnings}

    if target_column and target_column not in data.columns:
        issues.append(f"La columna objetivo '{target_column}' no existe")

    if feature_columns:
        missing_cols = [col for col in feature_columns if col not in data.columns]
        if missing_cols:
            issues.append(f"Columnas faltantes: {missing_cols}")

    # Verificar valores faltantes
    missing_info = data.isnull().sum()
    cols_with_missing = missing_info[missing_info > 0]
    if len(cols_with_missing) > 0:
        warnings.append(f"Columnas con valores faltantes: {dict(cols_with_missing)}")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'shape': data.shape
    }


# ==================== FUNCIONES DE MODELOS ESPECÍFICOS ====================

def regresion_lineal_simple(data, x_column, y_column):
    """Regresión lineal simple"""
    try:
        return regresion_lineal_multiple(data, y_column, [x_column])
    except Exception as e:
        return {'error': str(e)}


def regresion_lineal_multiple(data: pd.DataFrame, target_column: str,
                              feature_columns: Optional[List[str]] = None,
                              test_size: float = 0.2) -> Dict[str, Any]:
    """Regresión lineal múltiple"""
    try:
        prep_data = preparar_datos_supervisado_optimizado(
            data, target_column, feature_columns, test_size=test_size
        )

        model = NumpyLinearRegression()
        model.fit(prep_data['X_train_scaled'], prep_data['y_train'])

        # Predicciones
        y_pred_train = model.predict(prep_data['X_train_scaled'])
        y_pred_test = model.predict(prep_data['X_test_scaled'])

        # Métricas
        train_r2 = manual_r2_score(prep_data['y_train'], y_pred_train)
        test_r2 = manual_r2_score(prep_data['y_test'], y_pred_test)
        train_mse = np.mean((prep_data['y_train'] - y_pred_train) ** 2)
        test_mse = np.mean((prep_data['y_test'] - y_pred_test) ** 2)

        # Coeficientes
        coeficientes = {}
        if hasattr(model.coef_, '__len__'):
            for i, feature in enumerate(prep_data['feature_columns']):
                coeficientes[feature] = float(model.coef_[i])
        else:
            coeficientes[prep_data['feature_columns'][0]] = float(model.coef_)

        return {
            'tipo': 'regresion_lineal_multiple',
            'modelo': model,
            'target_column': target_column,
            'feature_columns': prep_data['feature_columns'],
            'coeficientes': coeficientes,
            'intercepto': float(model.intercept_),
            'metricas': {
                'train': {'r2': train_r2, 'mse': train_mse, 'rmse': np.sqrt(train_mse)},
                'test': {'r2': test_r2, 'mse': test_mse, 'rmse': np.sqrt(test_mse)}
            },
            'datos_prediccion': {
                'y_train': prep_data['y_train'].tolist(),
                'y_test': prep_data['y_test'].tolist(),
                'y_pred_train': y_pred_train.tolist(),
                'y_pred_test': y_pred_test.tolist()
            }
        }

    except Exception as e:
        return {'error': str(e)}


def arbol_decision(data: pd.DataFrame, target_column: str,
                   feature_columns: Optional[List[str]] = None,
                   max_depth: Optional[int] = 5,
                   min_samples_split: int = 2) -> Dict[str, Any]:
    """Árbol de decisión"""
    try:
        prep_data = preparar_datos_supervisado_optimizado(
            data, target_column, feature_columns, scale_data=False
        )

        model = NumpyDecisionTree(max_depth=max_depth, min_samples_split=min_samples_split)
        model.fit(prep_data['X_train'], prep_data['y_train'])

        # Predicciones
        y_pred_train = model.predict(prep_data['X_train'])
        y_pred_test = model.predict(prep_data['X_test'])

        # Métricas según tipo de problema
        if prep_data['is_classification']:
            train_acc = manual_accuracy_score(prep_data['y_train'], y_pred_train)
            test_acc = manual_accuracy_score(prep_data['y_test'], y_pred_test)

            train_report = manual_classification_report(prep_data['y_train'], y_pred_train)
            test_report = manual_classification_report(prep_data['y_test'], y_pred_test)

            train_cm, train_labels = manual_confusion_matrix(prep_data['y_train'], y_pred_train)
            test_cm, test_labels = manual_confusion_matrix(prep_data['y_test'], y_pred_test)

            metricas = {
                'train': {'accuracy': train_acc, 'classification_report': train_report},
                'test': {'accuracy': test_acc, 'classification_report': test_report}
            }

            extra_data = {
                'train_cm': train_cm,
                'test_cm': test_cm,
                'cm_labels': test_labels,
                'y_pred_train': y_pred_train,
                'y_pred_test': y_pred_test
            }
        else:
            train_r2 = manual_r2_score(prep_data['y_train'], y_pred_train)
            test_r2 = manual_r2_score(prep_data['y_test'], y_pred_test)

            metricas = {
                'train': {'r2': train_r2},
                'test': {'r2': test_r2}
            }

            extra_data = {
                'y_pred_train': y_pred_train,
                'y_pred_test': y_pred_test
            }

            # Importancia de características
        importancia_features = dict(zip(
            prep_data['feature_columns'],
            model.feature_importances_
        ))

        return {
            'tipo': 'arbol_decision',
            'modelo': model,
            'es_clasificacion': prep_data['is_classification'],
            'target_column': target_column,
            'feature_columns': prep_data['feature_columns'],
            'metricas': metricas,
            'importancia_features': importancia_features,
            'tree_info': {
                'max_depth': max_depth,
                'min_samples_split': min_samples_split
            },
            'datos_prediccion': {
                'y_train': prep_data['y_train'].tolist(),
                'y_test': prep_data['y_test'].tolist(),
                **extra_data
            }
        }

    except Exception as e:
        return {'error': str(e)}

def random_forest(data: pd.DataFrame, target_column: str,
                  feature_columns: Optional[List[str]] = None,
                  n_estimators: int = 10, max_depth: Optional[int] = 5,
                  test_size: float = 0.2) -> Dict[str, Any]:
    """Random Forest"""
    try:
        prep_data = preparar_datos_supervisado_optimizado(
            data, target_column, feature_columns, test_size=test_size, scale_data=False
        )

        model = NumpyRandomForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(prep_data['X_train'], prep_data['y_train'])

        # Predicciones
        y_pred_train = model.predict(prep_data['X_train'])
        y_pred_test = model.predict(prep_data['X_test'])

        # Métricas según tipo de problema
        if prep_data['is_classification']:
            train_acc = manual_accuracy_score(prep_data['y_train'], y_pred_train)
            test_acc = manual_accuracy_score(prep_data['y_test'], y_pred_test)

            train_report = manual_classification_report(prep_data['y_train'], y_pred_train)
            test_report = manual_classification_report(prep_data['y_test'], y_pred_test)

            train_cm, train_labels = manual_confusion_matrix(prep_data['y_train'], y_pred_train)
            test_cm, test_labels = manual_confusion_matrix(prep_data['y_test'], y_pred_test)

            metricas = {
                'train': {'accuracy': train_acc, 'classification_report': train_report},
                'test': {'accuracy': test_acc, 'classification_report': test_report}
            }

            extra_data = {
                'train_cm': train_cm,
                'test_cm': test_cm,
                'cm_labels': test_labels,
                'y_pred_train': y_pred_train,
                'y_pred_test': y_pred_test
            }
        else:
            train_r2 = manual_r2_score(prep_data['y_train'], y_pred_train)
            test_r2 = manual_r2_score(prep_data['y_test'], y_pred_test)
            train_mse = np.mean((prep_data['y_train'] - y_pred_train) ** 2)
            test_mse = np.mean((prep_data['y_test'] - y_pred_test) ** 2)

            metricas = {
                'train': {'r2': train_r2, 'mse': train_mse},
                'test': {'r2': test_r2, 'mse': test_mse}
            }

            extra_data = {
                'y_pred_train': y_pred_train,
                'y_pred_test': y_pred_test
            }

        # Importancia de características
        importancia_features = dict(zip(
            prep_data['feature_columns'],
            model.feature_importances_
        ))

        return {
            'tipo': 'random_forest',
            'modelo': model,
            'es_clasificacion': prep_data['is_classification'],
            'target_column': target_column,
            'feature_columns': prep_data['feature_columns'],
            'metricas': metricas,
            'importancia_features': importancia_features,
            'parametros': {
                'n_estimators': n_estimators,
                'max_depth': max_depth
            },
            'datos_prediccion': {
                'y_train': prep_data['y_train'].tolist(),
                'y_test': prep_data['y_test'].tolist(),
                **extra_data
            }
        }

    except Exception as e:
        return {'error': str(e)}


def svm_modelo(data: pd.DataFrame, target_column: str,
               feature_columns: Optional[List[str]] = None,
               C: float = 1.0, kernel: str = 'linear') -> Dict[str, Any]:
    """Support Vector Machine - Mejorado para múltiples clases"""
    try:
        prep_data = preparar_datos_supervisado_optimizado(
            data, target_column, feature_columns
        )

        if not prep_data['is_classification']:
            raise ValueError("SVM solo soporta clasificación")

        # Verificar número de clases
        unique_classes = np.unique(prep_data['y_train'])
        n_classes = len(unique_classes)

        if n_classes < 2:
            raise ValueError("Se necesitan al menos 2 clases para clasificación")
        elif n_classes == 2:
            # Clasificación binaria normal
            model = NumpySVM(C=C, kernel=kernel)
            model.fit(prep_data['X_train_scaled'], prep_data['y_train'])
            y_pred_train = model.predict(prep_data['X_train_scaled'])
            y_pred_test = model.predict(prep_data['X_test_scaled'])
        else:
            # Clasificación multiclase usando One-vs-Rest
            print(f"Detectadas {n_classes} clases. Usando estrategia One-vs-Rest...")
            models = {}
            y_pred_train_proba = np.zeros((len(prep_data['y_train']), n_classes))
            y_pred_test_proba = np.zeros((len(prep_data['y_test']), n_classes))

            # Entrenar un clasificador binario para cada clase
            for i, class_label in enumerate(unique_classes):
                # Crear problema binario: clase actual vs todas las demás
                y_binary_train = (prep_data['y_train'] == class_label).astype(int)
                y_binary_test = (prep_data['y_test'] == class_label).astype(int)

                # Entrenar modelo binario
                binary_model = NumpySVM(C=C, kernel=kernel, max_iter=500)
                binary_model.fit(prep_data['X_train_scaled'], y_binary_train)

                # Predicciones (convertir a probabilidades simples)
                train_scores = binary_model.predict(prep_data['X_train_scaled'])
                test_scores = binary_model.predict(prep_data['X_test_scaled'])

                y_pred_train_proba[:, i] = train_scores
                y_pred_test_proba[:, i] = test_scores

                models[class_label] = binary_model

            # Predicción final: clase con mayor "score"
            y_pred_train = unique_classes[np.argmax(y_pred_train_proba, axis=1)]
            y_pred_test = unique_classes[np.argmax(y_pred_test_proba, axis=1)]

            # Crear objeto modelo combinado
            model = {'type': 'multiclass_svm', 'models': models, 'classes': unique_classes}

        # Calcular métricas
        train_acc = manual_accuracy_score(prep_data['y_train'], y_pred_train)
        test_acc = manual_accuracy_score(prep_data['y_test'], y_pred_test)

        train_report = manual_classification_report(prep_data['y_train'], y_pred_train)
        test_report = manual_classification_report(prep_data['y_test'], y_pred_test)

        train_cm, train_labels = manual_confusion_matrix(prep_data['y_train'], y_pred_train)
        test_cm, test_labels = manual_confusion_matrix(prep_data['y_test'], y_pred_test)

        # Información de vectores de soporte
        if n_classes == 2:
            n_support = getattr(model, 'n_support_', 0)
        else:
            # Para multiclase, sumar vectores de soporte de todos los modelos
            n_support = sum([getattr(m, 'n_support_', 0) for m in models.values()])

        return {
            'tipo': 'svm',
            'modelo': model,
            'es_clasificacion': True,
            'n_clases': n_classes,
            'target_column': target_column,
            'feature_columns': prep_data['feature_columns'],
            'metricas': {
                'train': {'accuracy': train_acc, 'classification_report': train_report},
                'test': {'accuracy': test_acc, 'classification_report': test_report}
            },
            'parametros': {
                'C': C,
                'kernel': kernel,
                'n_support_vectors': n_support,
                'estrategia': 'binario' if n_classes == 2 else 'one-vs-rest'
            },
            'datos_prediccion': {
                'y_train': prep_data['y_train'].tolist(),
                'y_test': prep_data['y_test'].tolist(),
                'y_pred_train': y_pred_train.tolist(),
                'y_pred_test': y_pred_test.tolist(),
                'train_cm': train_cm,
                'test_cm': test_cm,
                'cm_labels': test_labels
            }
        }

    except Exception as e:
        return {'error': str(e)}

def comparar_modelos_supervisado(data: pd.DataFrame, target_column: str,
                                 feature_columns: Optional[List[str]] = None,
                                 modelos: List[str] = None) -> Dict[str, Any]:
    """Comparar múltiples modelos"""
    if modelos is None:
        modelos = ['linear', 'tree', 'random_forest', 'svm']

    try:
        resultados = {
            'tipo': 'comparar_modelos',
            'target_column': target_column,
            'feature_columns': feature_columns,
            'modelos': {},
            'ranking': []
        }

        # Entrenar cada modelo
        for modelo_nombre in modelos:
            try:
                if modelo_nombre == 'linear':
                    resultado = regresion_lineal_multiple(data, target_column, feature_columns)
                elif modelo_nombre == 'tree':
                    resultado = arbol_decision(data, target_column, feature_columns)
                elif modelo_nombre == 'random_forest':
                    resultado = random_forest(data, target_column, feature_columns)
                elif modelo_nombre == 'svm':
                    resultado = svm_modelo(data, target_column, feature_columns)
                else:
                    continue

                if 'error' not in resultado:
                    resultados['modelos'][modelo_nombre] = resultado

            except Exception as e:
                resultados['modelos'][modelo_nombre] = {'error': str(e)}

        # Crear ranking
        ranking = []
        for nombre, resultado in resultados['modelos'].items():
            if 'error' not in resultado and 'metricas' in resultado:
                metricas = resultado['metricas']['test']

                if 'accuracy' in metricas:
                    score = metricas['accuracy']
                    metric_name = 'accuracy'
                elif 'r2' in metricas:
                    score = metricas['r2']
                    metric_name = 'r2'
                else:
                    continue

                ranking.append({
                    'modelo': nombre,
                    'score': float(score),
                    'metrica': metric_name
                })

        ranking.sort(key=lambda x: x['score'], reverse=True)
        resultados['ranking'] = ranking

        return resultados

    except Exception as e:
        return {'error': str(e)}

# ==================== FUNCIONES DE VISUALIZACIÓN ====================
# ESTAS FUNCIONES DEBEN ESTAR AL MISMO NIVEL QUE comparar_modelos_supervisado

def generar_visualizaciones_ml(resultado: Dict[str, Any], figsize: Tuple[int, int] = (16, 12)) -> Figure:
    """Generar visualizaciones según el tipo de modelo"""
    try:
        fig = Figure(figsize=figsize)
        tipo = resultado.get('tipo', '')

        if tipo == 'random_forest':
            return _plot_random_forest_advanced(resultado, fig)
        elif tipo == 'svm':
            return _plot_svm_advanced(resultado, fig)
        elif tipo == 'arbol_decision':
            return _plot_arbol_decision(resultado, fig)
        elif tipo == 'regresion_lineal_multiple':
            return _plot_regresion_multiple(resultado, fig)
        elif tipo == 'comparar_modelos':
            return _plot_comparacion_modelos(resultado, fig)
        else:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f'Visualización para {tipo}\nno implementada',
                    ha='center', va='center', transform=ax.transAxes, fontsize=16)
            return fig

    except Exception as e:
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, f'Error generando visualización:\n{str(e)}',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        return fig

def _plot_random_forest_advanced(resultado: Dict, fig: Figure) -> Figure:
    """Visualización avanzada para Random Forest"""
    try:
        if not resultado.get('es_clasificacion', False):
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'Random Forest - Regresión\nVisualización básica',
                    ha='center', va='center', transform=ax.transAxes)
            return fig

        # Obtener datos
        datos_pred = resultado.get('datos_prediccion', {})
        test_report = resultado.get('metricas', {}).get('test', {}).get('classification_report', {})
        test_cm = datos_pred.get('test_cm')
        cm_labels = datos_pred.get('cm_labels', [])
        importancia = resultado.get('importancia_features', {})

        # Layout principal
        fig.suptitle('Evaluación de Random Forest', fontsize=16, fontweight='bold')
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.5], width_ratios=[1, 1, 1, 1])

        # 1. Exactitud principal
        ax_acc = fig.add_subplot(gs[0, 0])
        ax_acc.axis('off')
        exactitud = test_report.get('accuracy', 0) if isinstance(test_report, dict) else 0
        ax_acc.text(0.5, 0.8, 'Evaluación de Random Forest',
                    ha='center', va='center', fontsize=12, fontweight='bold')
        ax_acc.text(0.5, 0.4, f'Exactitud: {exactitud:.4f}',
                    ha='center', va='center', fontsize=10)

        # 2. Reporte de clasificación
        ax_report = fig.add_subplot(gs[0:2, 1:3])
        ax_report.axis('off')

        if isinstance(test_report, dict) and cm_labels is not None:
            report_text = "Reporte de Clasificación:\n"
            report_text += f"{'':>12} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}\n\n"

            for label in cm_labels:
                label_str = str(label)
                if label_str in test_report:
                    metrics = test_report[label_str]
                    name = f'Class_{label}'
                    report_text += f"{name:>12} {metrics['precision']:>9.2f} {metrics['recall']:>9.2f} {metrics['f1-score']:>9.2f} {int(metrics['support']):>9}\n"

            if 'accuracy' in test_report:
                total_support = sum(
                    [test_report[str(label)]['support'] for label in cm_labels if str(label) in test_report])
                report_text += f"\n{'accuracy':>12} {'':>9} {'':>9} {test_report['accuracy']:>9.2f} {total_support:>9}\n"

            if 'macro avg' in test_report:
                macro = test_report['macro avg']
                report_text += f"{'macro avg':>12} {macro['precision']:>9.2f} {macro['recall']:>9.2f} {macro['f1-score']:>9.2f} {int(macro['support']):>9}\n"

            if 'weighted avg' in test_report:
                weighted = test_report['weighted avg']
                report_text += f"{'weighted avg':>12} {weighted['precision']:>9.2f} {weighted['recall']:>9.2f} {weighted['f1-score']:>9.2f} {int(weighted['support']):>9}\n"

            ax_report.text(0.02, 0.98, report_text, transform=ax_report.transAxes,
                           fontsize=8, fontfamily='monospace', verticalalignment='top')

        # 3. Matriz de confusión
        if test_cm is not None and len(test_cm) > 0:
            ax_cm = fig.add_subplot(gs[0:2, 3])
            ax_cm.axis('off')
            cm_text = "Matriz de Confusión:\n" + str(test_cm.tolist())
            ax_cm.text(0.02, 0.98, cm_text, transform=ax_cm.transAxes,
                       fontsize=9, fontfamily='monospace', verticalalignment='top')

        # 4. Importancia de variables
        ax_imp = fig.add_subplot(gs[2, :2])
        if importancia:
            sorted_features = sorted(importancia.items(), key=lambda x: x[1], reverse=True)
            features = [item[0] for item in sorted_features[:10]]
            values = [item[1] for item in sorted_features[:10]]

            y_pos = np.arange(len(features))
            bars = ax_imp.barh(y_pos, values, color='blue', alpha=0.7)
            ax_imp.set_yticks(y_pos)
            ax_imp.set_yticklabels(features, fontsize=8)
            ax_imp.set_xlabel('Importancia', fontsize=10)
            ax_imp.set_title('Importancia de Variables', fontsize=12)
            ax_imp.grid(True, alpha=0.3)

        # 5. Comparación Train vs Test
        ax_comp = fig.add_subplot(gs[2, 2:])
        train_acc = resultado.get('metricas', {}).get('train', {}).get('accuracy', 0)
        test_acc = exactitud

        if train_acc > 0:
            bars = ax_comp.bar(['Train', 'Test'], [train_acc, test_acc],
                               color=['lightgreen', 'lightcoral'], alpha=0.7)
            ax_comp.set_ylabel('Exactitud', fontsize=10)
            ax_comp.set_title('Comparación Train vs Test', fontsize=12)
            ax_comp.set_ylim(0, 1.1)
            ax_comp.grid(True, alpha=0.3)

            for bar, acc in zip(bars, [train_acc, test_acc]):
                ax_comp.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

        fig.tight_layout()
        return fig

    except Exception as e:
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, f'Error en visualización de Random Forest:\n{str(e)}',
                ha='center', va='center', transform=ax.transAxes)
        return fig

def _plot_svm_advanced(resultado: Dict, fig: Figure) -> Figure:
    """Visualización avanzada para SVM"""
    try:
        # Similar estructura a Random Forest pero adaptada para SVM
        datos_pred = resultado.get('datos_prediccion', {})
        test_report = resultado.get('metricas', {}).get('test', {}).get('classification_report', {})
        test_cm = datos_pred.get('test_cm')
        cm_labels = datos_pred.get('cm_labels', [])
        parametros = resultado.get('parametros', {})

        fig.suptitle('Evaluación de SVM', fontsize=16, fontweight='bold')
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.5], width_ratios=[1, 1, 1, 1])

        # 1. Exactitud principal
        ax_acc = fig.add_subplot(gs[0, 0])
        ax_acc.axis('off')
        exactitud = test_report.get('accuracy', 0) if isinstance(test_report, dict) else 0
        ax_acc.text(0.5, 0.8, 'Evaluación de SVM',
                    ha='center', va='center', fontsize=12, fontweight='bold')
        ax_acc.text(0.5, 0.4, f'Exactitud: {exactitud:.4f}',
                    ha='center', va='center', fontsize=10)

        # 2. Reporte de clasificación (igual que Random Forest)
        ax_report = fig.add_subplot(gs[0:2, 1:3])
        ax_report.axis('off')

        if isinstance(test_report, dict) and cm_labels is not None:
            report_text = "Reporte de Clasificación:\n"
            report_text += f"{'':>12} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}\n\n"

            for label in cm_labels:
                label_str = str(label)
                if label_str in test_report:
                    metrics = test_report[label_str]
                    name = f'Class_{label}'
                    report_text += f"{name:>12} {metrics['precision']:>9.2f} {metrics['recall']:>9.2f} {metrics['f1-score']:>9.2f} {int(metrics['support']):>9}\n"

            if 'accuracy' in test_report:
                total_support = sum(
                    [test_report[str(label)]['support'] for label in cm_labels if str(label) in test_report])
                report_text += f"\n{'accuracy':>12} {'':>9} {'':>9} {test_report['accuracy']:>9.2f} {total_support:>9}\n"

            ax_report.text(0.02, 0.98, report_text, transform=ax_report.transAxes,
                           fontsize=8, fontfamily='monospace', verticalalignment='top')

        # 3. Matriz de confusión
        if test_cm is not None and len(test_cm) > 0:
            ax_cm = fig.add_subplot(gs[0:2, 3])
            ax_cm.axis('off')
            cm_text = "Matriz de Confusión:\n" + str(test_cm.tolist())
            ax_cm.text(0.02, 0.98, cm_text, transform=ax_cm.transAxes,
                       fontsize=9, fontfamily='monospace', verticalalignment='top')

        # 4. Información del modelo SVM
        ax_info = fig.add_subplot(gs[2, :2])
        ax_info.axis('off')

        info_text = f"Parámetros del Modelo SVM:\n\n"
        info_text += f"• Kernel: {parametros.get('kernel', 'linear')}\n"
        info_text += f"• C: {parametros.get('C', 1.0)}\n"
        info_text += f"• Vectores de soporte: {parametros.get('n_support_vectors', 0)}\n\n"
        info_text += f"Características:\n"
        info_text += f"• Clasificación binaria\n"
        info_text += f"• Maximización del margen\n"
        info_text += f"• Separación óptima"

        ax_info.text(0.02, 0.98, info_text, transform=ax_info.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # 5. Comparación Train vs Test
        ax_comp = fig.add_subplot(gs[2, 2:])
        train_acc = resultado.get('metricas', {}).get('train', {}).get('accuracy', 0)
        test_acc = exactitud

        if train_acc > 0:
            bars = ax_comp.bar(['Train', 'Test'], [train_acc, test_acc],
                               color=['lightblue', 'lightcoral'], alpha=0.7)
            ax_comp.set_ylabel('Exactitud', fontsize=10)
            ax_comp.set_title('Comparación Train vs Test', fontsize=12)
            ax_comp.set_ylim(0, 1.1)
            ax_comp.grid(True, alpha=0.3)

            for bar, acc in zip(bars, [train_acc, test_acc]):
                ax_comp.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

        fig.tight_layout()
        return fig

    except Exception as e:
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, f'Error en visualización de SVM:\n{str(e)}',
                ha='center', va='center', transform=ax.transAxes)
        return fig

def _plot_arbol_decision(resultado: Dict, fig: Figure) -> Figure:
    """Visualización para árbol de decisión"""
    try:
        # Layout 2x2
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        # 1. Importancia de características
        importancia = resultado.get('importancia_features', {})
        if importancia:
            features = list(importancia.keys())[:10]
            values = [importancia[f] for f in features]

            y_pos = np.arange(len(features))
            ax1.barh(y_pos, values, color='forestgreen', alpha=0.7)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(features)
            ax1.set_xlabel('Importancia')
            ax1.set_title('Importancia de Características')
            ax1.grid(True, alpha=0.3)

        # 2. Información del modelo
        ax2.axis('off')
        info_text = f"Tipo: Árbol de Decisión\n"
        info_text += f"Problema: {'Clasificación' if resultado.get('es_clasificacion') else 'Regresión'}\n"

        tree_info = resultado.get('tree_info', {})
        if tree_info:
            info_text += f"Max Depth: {tree_info.get('max_depth', 'N/A')}\n"
            info_text += f"Min Samples Split: {tree_info.get('min_samples_split', 'N/A')}\n"

        metricas = resultado.get('metricas', {})
        if metricas:
            test_metrics = metricas.get('test', {})
            for metric_name, value in test_metrics.items():
                if isinstance(value, (int, float)):
                    info_text += f"{metric_name.upper()} (Test): {value:.4f}\n"

        ax2.text(0.1, 0.9, info_text, transform=ax2.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # 3. Comparación Train vs Test
        if metricas:
            train_metrics = metricas.get('train', {})
            test_metrics = metricas.get('test', {})

            if train_metrics and test_metrics:
                metric_name = list(train_metrics.keys())[0]
                train_val = list(train_metrics.values())[0]
                test_val = list(test_metrics.values())[0]

                ax3.bar(['Train', 'Test'], [train_val, test_val],
                        color=['lightgreen', 'lightcoral'], alpha=0.7)
                ax3.set_ylabel(metric_name.upper())
                ax3.set_title('Comparación Train vs Test')
                ax3.grid(True, alpha=0.3)

                ax3.text(0, train_val + 0.01, f'{train_val:.3f}',
                         ha='center', va='bottom', fontweight='bold')
                ax3.text(1, test_val + 0.01, f'{test_val:.3f}',
                         ha='center', va='bottom', fontweight='bold')

        # 4. Estructura del árbol
        ax4.axis('off')
        tree_text = """
        Estructura del Árbol:

           Raíz
          /    \\
       Rama1  Rama2
       /  \\    /  \\
      H1  H2  H3  H4

        H = Hoja (predicción)
        """
        ax4.text(0.1, 0.9, tree_text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        fig.suptitle(f'Árbol de Decisión - {resultado["target_column"]}', fontsize=14)
        fig.tight_layout()
        return fig

    except Exception as e:
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, f'Error en visualización de árbol:\n{str(e)}',
                ha='center', va='center', transform=ax.transAxes)
        return fig

def _plot_regresion_multiple(resultado: Dict, fig: Figure) -> Figure:
    """Visualización para regresión múltiple"""
    try:
        # Datos de predicciones
        datos_pred = resultado.get('datos_prediccion', {})
        y_test = np.array(datos_pred.get('y_test', []))
        y_pred = np.array(datos_pred.get('y_pred_test', []))

        if len(y_test) == 0 or len(y_pred) == 0:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No hay datos de predicción disponibles',
                    ha='center', va='center', transform=ax.transAxes)
            return fig

        # Layout 2x2
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        # 1. Predicciones vs Reales
        ax1.scatter(y_test, y_pred, alpha=0.6, color='blue')
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax1.set_xlabel('Valores Reales')
        ax1.set_ylabel('Predicciones')
        r2 = resultado.get('metricas', {}).get('test', {}).get('r2', 0)
        ax1.set_title(f'R² = {r2:.3f}')
        ax1.grid(True, alpha=0.3)

        # 2. Residuos
        residuos = y_test - y_pred
        ax2.scatter(y_pred, residuos, alpha=0.6, color='red')
        ax2.axhline(y=0, color='black', linestyle='--')
        ax2.set_xlabel('Predicciones')
        ax2.set_ylabel('Residuos')
        ax2.set_title('Análisis de Residuos')
        ax2.grid(True, alpha=0.3)

        # 3. Distribución de residuos
        ax3.hist(residuos, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax3.set_xlabel('Residuos')
        ax3.set_ylabel('Frecuencia')
        ax3.set_title('Distribución de Residuos')
        ax3.grid(True, alpha=0.3)

        # 4. Coeficientes
        coefs = resultado.get('coeficientes', {})
        if coefs:
            features = list(coefs.keys())[:10]
            values = [coefs[f] for f in features]
            colors = ['red' if v < 0 else 'blue' for v in values]

            y_pos = np.arange(len(features))
            ax4.barh(y_pos, values, color=colors, alpha=0.7)
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(features)
            ax4.set_xlabel('Coeficiente')
            ax4.set_title('Importancia de Variables')
            ax4.grid(True, alpha=0.3)

        fig.suptitle(f'Regresión Lineal Múltiple - {resultado["target_column"]}', fontsize=14)
        fig.tight_layout()
        return fig

    except Exception as e:
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, f'Error en visualización de regresión:\n{str(e)}',
                ha='center', va='center', transform=ax.transAxes)
        return fig

def _plot_comparacion_modelos(resultado: Dict, fig: Figure) -> Figure:
    """Visualización para comparación de modelos"""
    try:
        ranking = resultado.get('ranking', [])
        modelos_data = resultado.get('modelos', {})

        if not ranking:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No hay datos de comparación disponibles',
                    ha='center', va='center', transform=ax.transAxes)
            return fig

        # Gráfico de barras con ranking
        ax = fig.add_subplot(1, 1, 1)

        models = [item['modelo'] for item in ranking]
        scores = [item['score'] for item in ranking]

        colors = ['gold', 'silver', '#CD7F32'] + ['lightblue'] * (len(models) - 3)
        colors = colors[:len(models)]

        bars = ax.bar(models, scores, color=colors, alpha=0.8)
        ax.set_ylabel('Score')
        ax.set_title('Comparación de Modelos (Mayor es mejor)')
        ax.grid(True, alpha=0.3)

        # Añadir valores en las barras
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

        # Rotar etiquetas si hay muchos modelos
        if len(models) > 3:
            ax.tick_params(axis='x', rotation=45)

        fig.tight_layout()
        return fig

    except Exception as e:
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, f'Error en gráfico de comparación:\n{str(e)}',
                ha='center', va='center', transform=ax.transAxes)
        return fig

# ==================== FUNCIONES DE UTILIDAD Y EXPORTACIÓN ====================

def exportar_modelo(modelo, filepath: str) -> bool:
    """Exportar modelo a archivo"""
    try:
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(modelo, f)
        return True
    except Exception as e:
        try:
            # Fallback a JSON para modelos simples
            if hasattr(modelo, '__dict__'):
                model_dict = {}
                for key, value in modelo.__dict__.items():
                    if isinstance(value, (int, float, str, list, dict)):
                        model_dict[key] = value
                    elif isinstance(value, np.ndarray):
                        model_dict[key] = value.tolist()
                    else:
                        model_dict[key] = str(value)

                with open(filepath.replace('.pkl', '.json'), 'w') as f:
                    json.dump(model_dict, f, indent=2)
                return True
        except Exception as e2:
            print(f"Error exportando modelo: {e2}")
            return False

def limpiar_memoria():
    """Limpiar memoria"""
    import gc
    gc.collect()

# ==================== FUNCIONES DE COMPATIBILIDAD ====================

def visualizar_arbol_decision(resultado, **kwargs):
    """Visualización específica de árbol - compatibilidad"""
    return generar_visualizaciones_ml(resultado)

def crear_dashboard_arbol(resultado, **kwargs):
    """Dashboard de árbol - compatibilidad"""
    return generar_visualizaciones_ml(resultado)

def visualizar_importancia_features(resultado, **kwargs):
    """Visualización de importancia - compatibilidad"""
    return generar_visualizaciones_ml(resultado)

def crear_grafico_comparacion_modelos(resultados_comparacion: Dict[str, Any],
                                      figsize: Tuple[int, int] = (16, 10)) -> Figure:
    """Crear gráfico de comparación de modelos - función específica"""
    return _plot_comparacion_modelos(resultados_comparacion, Figure(figsize=figsize))

# ==================== CONFIGURACIÓN DE LOGGING ====================

def configurar_logging(nivel=logging.INFO):
    """Configurar sistema de logging"""
    logging.basicConfig(
        level=nivel,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

# ==================== FUNCIONES DE TESTING ====================

def test_modelos_numpy():
    """Función de testing para verificar que los modelos funcionan"""
    try:
        # Generar datos de prueba
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        y_reg = X.sum(axis=1) + np.random.randn(n_samples) * 0.1
        y_clf = (y_reg > 0).astype(int)

        # Test regresión lineal
        model_reg = NumpyLinearRegression()
        model_reg.fit(X, y_reg)
        pred_reg = model_reg.predict(X[:10])

        # Test árbol de decisión
        model_tree = NumpyDecisionTree(max_depth=3)
        model_tree.fit(X, y_clf)
        pred_tree = model_tree.predict(X[:10])

        # Test Random Forest
        model_rf = NumpyRandomForest(n_estimators=5, max_depth=3)
        model_rf.fit(X, y_clf)
        pred_rf = model_rf.predict(X[:10])

        # Test SVM
        model_svm = NumpySVM(max_iter=100)
        model_svm.fit(X, y_clf)
        pred_svm = model_svm.predict(X[:10])

        print("✅ Todos los modelos NumPy funcionan correctamente")
        return True

    except Exception as e:
        print(f"❌ Error en testing: {e}")
        return False

# ==================== PUNTO DE ENTRADA PARA TESTING ====================

if __name__ == "__main__":
    print("🧪 Ejecutando tests de ml_functions_supervisado...")

    # Configurar logging
    configurar_logging()

    # Ejecutar tests
    test_result = test_modelos_numpy()

    if test_result:
        print("✅ Sistema ML Supervisado listo para usar")
    else:
        print("❌ Hay problemas en el sistema ML")

    print("\n📋 Funciones disponibles:")
    print("- regresion_lineal_simple/multiple")
    print("- arbol_decision")
    print("- random_forest")
    print("- svm_modelo")
    print("- comparar_modelos_supervisado")
    print("- generar_visualizaciones_ml")
    print("- preparar_datos_supervisado_optimizado")
    print("- verificar_datos")
    print("- exportar_modelo")

    print("\n🎯 Compatible con PyInstaller - Sin dependencias sklearn")
    print("🔧 Implementación 100% NumPy")
    print("📊 Reportes de clasificación completos")
    print("📈 Visualizaciones avanzadas incluidas")