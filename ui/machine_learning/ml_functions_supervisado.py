"""
ml_functions_supervisado.py - Funciones de Machine Learning Supervisado OPTIMIZADAS
Versión mejorada con mejor rendimiento, manejo de errores y funcionalidad extendida
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV,
    RandomizedSearchCV, KFold, StratifiedKFold, validation_curve,
    learning_curve
)
from sklearn.preprocessing import (
    StandardScaler, LabelEncoder, MinMaxScaler,
    RobustScaler, PolynomialFeatures
)
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    mean_absolute_percentage_error, accuracy_score, precision_score,
    recall_score, f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    explained_variance_score
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression, RidgeCV, LassoCV
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.feature_selection import (
    SelectKBest, f_regression, f_classif, RFE,
    mutual_info_regression, mutual_info_classif
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from joblib import Parallel, delayed
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from functools import lru_cache
import gc

warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== FUNCIONES DE UTILIDAD ====================

def verificar_datos(data: pd.DataFrame, target_column: str = None,
                   feature_columns: List[str] = None) -> Dict[str, Any]:
    """
    Verificar integridad y calidad de los datos
    """
    issues = []
    warnings = []

    # Verificar que el DataFrame no esté vacío
    if data.empty:
        issues.append("El DataFrame está vacío")
        return {'valid': False, 'issues': issues, 'warnings': warnings}

    # Verificar columna objetivo
    if target_column and target_column not in data.columns:
        issues.append(f"La columna objetivo '{target_column}' no existe")

    # Verificar columnas de características
    if feature_columns:
        missing_cols = [col for col in feature_columns if col not in data.columns]
        if missing_cols:
            issues.append(f"Columnas faltantes: {missing_cols}")

    # Verificar valores faltantes
    missing_info = data.isnull().sum()
    cols_with_missing = missing_info[missing_info > 0]
    if len(cols_with_missing) > 0:
        warnings.append(f"Columnas con valores faltantes: {dict(cols_with_missing)}")

    # Verificar varianza cero
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data[col].std() == 0:
            warnings.append(f"La columna '{col}' tiene varianza cero")

    # Verificar datos infinitos
    inf_cols = []
    for col in numeric_cols:
        if np.isinf(data[col]).any():
            inf_cols.append(col)
    if inf_cols:
        warnings.append(f"Columnas con valores infinitos: {inf_cols}")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'shape': data.shape,
        'dtypes': data.dtypes.to_dict()
    }

# ==================== PREPROCESAMIENTO OPTIMIZADO ====================

class PreprocesadorAvanzado:
    """Clase para preprocesamiento avanzado y eficiente"""

    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_names = []
        self.pipeline = None

    def crear_pipeline(self, numeric_features: List[str],
                      categorical_features: List[str] = None,
                      scale_method: str = 'standard',
                      impute_method: str = 'median') -> Pipeline:
        """
        Crear pipeline de preprocesamiento
        """
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=impute_method)),
            ('scaler', self._get_scaler(scale_method))
        ])

        transformers = [
            ('num', numeric_transformer, numeric_features)
        ]

        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', LabelEncoder())
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )

        self.pipeline = preprocessor
        return preprocessor

    def _get_scaler(self, method: str):
        """Obtener scaler según método"""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        return scalers.get(method, StandardScaler())

def preparar_datos_supervisado_optimizado(
    data: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_method: str = 'standard',
    handle_outliers: bool = True,
    feature_selection: Optional[str] = None,
    n_features: Optional[int] = None,
    stratify: bool = True,
    validation_split: float = 0.0
) -> Dict[str, Any]:
    """
    Preparar datos para aprendizaje supervisado con optimizaciones
    """
    try:
        # Verificar datos
        verificacion = verificar_datos(data, target_column, feature_columns)
        if not verificacion['valid']:
            raise ValueError(f"Problemas con los datos: {verificacion['issues']}")

        # Log warnings si existen
        if verificacion['warnings']:
            for warning in verificacion['warnings']:
                logger.warning(warning)

        # Copiar datos para no modificar originales
        df = data.copy()

        # Identificar columnas si no se especifican
        if feature_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in numeric_columns if col != target_column]

        # Verificar que hay suficientes características
        if len(feature_columns) == 0:
            raise ValueError("No hay características numéricas disponibles")

        # Separar características y objetivo
        X = df[feature_columns].copy()
        y = df[target_column].copy()

        # Limpiar datos faltantes de forma más inteligente
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # Verificar si hay valores faltantes en y
        y_missing = y.isnull().sum()
        if y_missing > 0:
            # Eliminar filas con target faltante
            valid_idx = ~y.isnull()
            X_imputed = X_imputed[valid_idx]
            y = y[valid_idx]
            logger.info(f"Eliminadas {y_missing} filas con valores faltantes en target")

        # Determinar tipo de problema
        unique_values = y.nunique()
        is_classification = (
            unique_values < 10 or
            y.dtype == 'object' or
            y.dtype.name == 'category' or
            isinstance(y.iloc[0], str)
        )

        # Codificar target si es categórico
        label_encoder = None
        classes = None
        if is_classification and y.dtype == 'object':
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            y = pd.Series(y_encoded, index=y.index)
            classes = label_encoder.classes_.tolist()

        # Manejo inteligente de outliers
        outliers_info = {}
        if handle_outliers and not is_classification:
            X_clean, y_clean, outliers_info = _handle_outliers_iqr(X_imputed, y)
        else:
            X_clean, y_clean = X_imputed, y

        # División estratificada si es clasificación
        stratify_y = y_clean if (is_classification and stratify) else None

        if validation_split > 0:
            # División con conjunto de validación
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_clean, y_clean,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_y
            )

            # Calcular tamaño de validación relativo al conjunto de entrenamiento
            val_size = validation_split / (1 - test_size)

            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size,
                random_state=random_state,
                stratify=stratify_y[X_temp.index] if stratify_y is not None else None
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_y
            )
            X_val, y_val = None, None

        # Escalado de características
        scaler = None
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_val_scaled = X_val.copy() if X_val is not None else None

        if scale_method != 'none':
            scaler = _get_scaler(scale_method)
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            if X_val is not None:
                X_val_scaled = pd.DataFrame(
                    scaler.transform(X_val),
                    columns=X_val.columns,
                    index=X_val.index
                )

        # Selección de características si se solicita
        feature_selector = None
        selected_features = feature_columns.copy()

        if feature_selection and len(feature_columns) > 5:
            selector_func = f_classif if is_classification else f_regression
            n_features_to_select = n_features or min(10, len(feature_columns) // 2)

            if feature_selection == 'kbest':
                feature_selector = SelectKBest(selector_func, k=n_features_to_select)
            elif feature_selection == 'mutual_info':
                mi_func = mutual_info_classif if is_classification else mutual_info_regression
                feature_selector = SelectKBest(mi_func, k=n_features_to_select)
            elif feature_selection == 'rfe':
                # Usar un estimador base simple para RFE
                base_estimator = (
                    LogisticRegression(max_iter=1000) if is_classification
                    else LinearRegression()
                )
                feature_selector = RFE(base_estimator, n_features_to_select=n_features_to_select)

            if feature_selector:
                X_train_scaled = pd.DataFrame(
                    feature_selector.fit_transform(X_train_scaled, y_train),
                    index=X_train_scaled.index
                )
                X_test_scaled = pd.DataFrame(
                    feature_selector.transform(X_test_scaled),
                    index=X_test_scaled.index
                )
                if X_val_scaled is not None:
                    X_val_scaled = pd.DataFrame(
                        feature_selector.transform(X_val_scaled),
                        index=X_val_scaled.index
                    )

                # Obtener características seleccionadas
                selected_mask = feature_selector.get_support()
                selected_features = [feat for feat, selected in zip(feature_columns, selected_mask) if selected]

        result = {
            'X_train': X_train,
            'X_test': X_test,
            'X_val': X_val,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'X_val_scaled': X_val_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'y_val': y_val,
            'feature_columns': selected_features,
            'original_features': feature_columns,
            'target_column': target_column,
            'scaler': scaler,
            'feature_selector': feature_selector,
            'label_encoder': label_encoder,
            'is_classification': is_classification,
            'classes': classes,
            'preprocessing_info': {
                'initial_samples': len(data),
                'final_samples': len(X_train) + len(X_test) + (len(X_val) if X_val is not None else 0),
                'missing_handled': y_missing,
                'outliers_info': outliers_info,
                'scale_method': scale_method,
                'feature_selection': feature_selection,
                'n_features_selected': len(selected_features)
            }
        }

        return result

    except Exception as e:
        logger.error(f"Error en preparación de datos: {str(e)}")
        raise

def _get_scaler(method: str):
    """Obtener scaler según método"""
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }
    return scalers.get(method, StandardScaler())

def _handle_outliers_iqr(X: pd.DataFrame, y: pd.Series,
                        multiplier: float = 1.5) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """Manejar outliers usando método IQR"""
    # Detectar outliers en y (variable objetivo)
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    # Máscara de valores válidos
    mask = (y >= lower_bound) & (y <= upper_bound)

    # También detectar outliers en características (opcional)
    feature_outliers = {}
    for col in X.columns:
        if X[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            Q1_feat = X[col].quantile(0.25)
            Q3_feat = X[col].quantile(0.75)
            IQR_feat = Q3_feat - Q1_feat
            lower_feat = Q1_feat - multiplier * IQR_feat
            upper_feat = Q3_feat + multiplier * IQR_feat
            outliers_feat = ((X[col] < lower_feat) | (X[col] > upper_feat)).sum()
            if outliers_feat > 0:
                feature_outliers[col] = outliers_feat

    outliers_info = {
        'target_outliers': (~mask).sum(),
        'feature_outliers': feature_outliers,
        'total_removed': (~mask).sum()
    }

    return X[mask], y[mask], outliers_info

# ==================== MODELOS DE REGRESIÓN OPTIMIZADOS ====================

def regresion_lineal_simple(data: pd.DataFrame, x_column: str, y_column: str,
                           optimize_params: bool = True) -> Dict[str, Any]:
    """
    Regresión lineal simple optimizada
    """
    try:
        # Verificar columnas
        if x_column not in data.columns or y_column not in data.columns:
            raise ValueError(f"Columnas '{x_column}' o '{y_column}' no encontradas")

        # Preparar datos
        X = data[[x_column]].values
        y = data[y_column].values

        # Eliminar valores faltantes
        mask = ~(np.isnan(X.flatten()) | np.isnan(y))
        X = X[mask].reshape(-1, 1)
        y = y[mask]

        if len(X) < 10:
            raise ValueError("Datos insuficientes después de limpiar valores faltantes")

        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Crear y entrenar modelo
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predicciones
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Métricas
        train_metrics = _calcular_metricas_regresion(y_train, y_pred_train)
        test_metrics = _calcular_metricas_regresion(y_test, y_pred_test)

        # Información adicional
        coeficiente = float(model.coef_[0])
        intercepto = float(model.intercept_)
        ecuacion = f"y = {coeficiente:.4f}x + {intercepto:.4f}"

        # Intervalos de confianza (aproximación simple)
        residuos = y_test - y_pred_test
        std_residuos = np.std(residuos)
        n = len(X_test)

        # Error estándar del coeficiente
        x_mean = np.mean(X_train)
        ss_x = np.sum((X_train - x_mean) ** 2)
        se_coef = std_residuos / np.sqrt(ss_x)

        # Intervalo de confianza al 95% para el coeficiente
        t_value = 1.96  # Aproximación para muestras grandes
        ci_lower = coeficiente - t_value * se_coef
        ci_upper = coeficiente + t_value * se_coef

        return {
            'tipo': 'regresion_lineal_simple',
            'modelo': model,
            'x_column': x_column,
            'y_column': y_column,
            'coeficiente': coeficiente,
            'intercepto': intercepto,
            'ecuacion': ecuacion,
            'intervalo_confianza_coef': (ci_lower, ci_upper),
            'metricas': {
                'train': train_metrics,
                'test': test_metrics
            },
            'datos': {
                'X_train': X_train.flatten().tolist(),
                'X_test': X_test.flatten().tolist(),
                'y_train': y_train.tolist(),
                'y_test': y_test.tolist(),
                'y_pred_train': y_pred_train.tolist(),
                'y_pred_test': y_pred_test.tolist()
            }
        }

    except Exception as e:
        logger.error(f"Error en regresión lineal simple: {str(e)}")
        return {'error': str(e)}

def regresion_lineal_multiple(data: pd.DataFrame, target_column: str,
                            feature_columns: Optional[List[str]] = None,
                            regularization: str = 'none',
                            alpha: float = 1.0,
                            optimize_params: bool = True) -> Dict[str, Any]:
    """
    Regresión lineal múltiple con regularización opcional
    """
    try:
        # Preparar datos
        prep_data = preparar_datos_supervisado_optimizado(
            data, target_column, feature_columns,
            scale_method='standard' if regularization != 'none' else 'none'
        )

        if prep_data['is_classification']:
            raise ValueError('La variable objetivo parece ser categórica. Use un modelo de clasificación.')

        X_train = prep_data['X_train_scaled']
        X_test = prep_data['X_test_scaled']
        y_train = prep_data['y_train']
        y_test = prep_data['y_test']

        # Seleccionar modelo según regularización
        if regularization == 'none':
            model = LinearRegression()
        elif regularization == 'ridge':
            if optimize_params:
                # Optimización de alpha usando RidgeCV
                alphas = np.logspace(-4, 4, 50)
                model = RidgeCV(alphas=alphas, cv=5)
            else:
                model = Ridge(alpha=alpha)
        elif regularization == 'lasso':
            if optimize_params:
                # Optimización de alpha usando LassoCV
                model = LassoCV(cv=5, max_iter=2000, n_alphas=50)
            else:
                model = Lasso(alpha=alpha, max_iter=2000)
        elif regularization == 'elastic':
            if optimize_params:
                # ElasticNet con optimización
                from sklearn.linear_model import ElasticNetCV
                model = ElasticNetCV(cv=5, max_iter=2000, n_alphas=50)
            else:
                model = ElasticNet(alpha=alpha, max_iter=2000)
        else:
            raise ValueError(f"Regularización '{regularization}' no soportada")

        # Entrenar modelo
        model.fit(X_train, y_train)

        # Obtener alpha óptimo si se optimizó
        if optimize_params and regularization != 'none':
            if hasattr(model, 'alpha_'):
                optimal_alpha = float(model.alpha_)
            else:
                optimal_alpha = alpha
        else:
            optimal_alpha = alpha

        # Predicciones
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Métricas
        train_metrics = _calcular_metricas_regresion(y_train, y_pred_train)
        test_metrics = _calcular_metricas_regresion(y_test, y_pred_test)

        # Validación cruzada
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

        # Coeficientes e importancia
        coeficientes = {}
        for i, feature in enumerate(prep_data['feature_columns']):
            coeficientes[feature] = float(model.coef_[i])

        # Análisis de residuos
        residuos_train = y_train - y_pred_train
        residuos_test = y_test - y_pred_test

        # Análisis de multicolinealidad (VIF)
        vif_scores = _calculate_vif(X_train) if len(prep_data['feature_columns']) > 1 else {}

        return {
            'tipo': 'regresion_lineal_multiple',
            'modelo': model,
            'target_column': target_column,
            'feature_columns': prep_data['feature_columns'],
            'regularization': regularization,
            'alpha_optimo': optimal_alpha,
            'coeficientes': coeficientes,
            'intercepto': float(model.intercept_),
            'metricas': {
                'train': train_metrics,
                'test': test_metrics,
                'cv_r2_mean': float(cv_scores.mean()),
                'cv_r2_std': float(cv_scores.std())
            },
            'analisis_residuos': {
                'train': _analizar_residuos(residuos_train),
                'test': _analizar_residuos(residuos_test)
            },
            'vif_scores': vif_scores,
            'datos_prediccion': {
                'y_train': y_train.tolist(),
                'y_test': y_test.tolist(),
                'y_pred_train': y_pred_train.tolist(),
                'y_pred_test': y_pred_test.tolist(),
                'residuos_train': residuos_train.tolist(),
                'residuos_test': residuos_test.tolist()
            },
            'preprocessing_info': prep_data['preprocessing_info']
        }

    except Exception as e:
        logger.error(f"Error en regresión múltiple: {str(e)}")
        return {'error': str(e)}

# ==================== ÁRBOLES DE DECISIÓN Y ENSEMBLE ====================

def arbol_decision(data: pd.DataFrame, target_column: str,
                  feature_columns: Optional[List[str]] = None,
                  max_depth: Optional[int] = None,
                  min_samples_split: int = 2,
                  min_samples_leaf: int = 1,
                  optimize_params: bool = True) -> Dict[str, Any]:
    """
    Árbol de decisión optimizado para regresión y clasificación
    """
    try:
        # Preparar datos
        prep_data = preparar_datos_supervisado_optimizado(
            data, target_column, feature_columns
        )

        X_train = prep_data['X_train_scaled']
        X_test = prep_data['X_test_scaled']
        y_train = prep_data['y_train']
        y_test = prep_data['y_test']

        # Seleccionar tipo de árbol
        if prep_data['is_classification']:
            model_class = DecisionTreeClassifier
            param_grid = {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'criterion': ['gini', 'entropy']
            }
            scoring = 'accuracy'
        else:
            model_class = DecisionTreeRegressor
            param_grid = {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'criterion': ['squared_error', 'absolute_error']
            }
            scoring = 'r2'

        if optimize_params:
            # Optimización con RandomizedSearchCV para mayor eficiencia
            model = RandomizedSearchCV(
                model_class(random_state=42),
                param_grid,
                n_iter=30,
                cv=5,
                scoring=scoring,
                n_jobs=-1,
                random_state=42
            )
            model.fit(X_train, y_train)
            best_model = model.best_estimator_
            best_params = model.best_params_
        else:
            best_model = model_class(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            best_model.fit(X_train, y_train)
            best_params = {
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf
            }

        # Predicciones
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        # Métricas
        if prep_data['is_classification']:
            train_metrics = _calcular_metricas_clasificacion(y_train, y_pred_train, best_model, X_train)
            test_metrics = _calcular_metricas_clasificacion(y_test, y_pred_test, best_model, X_test)
        else:
            train_metrics = _calcular_metricas_regresion(y_train, y_pred_train)
            test_metrics = _calcular_metricas_regresion(y_test, y_pred_test)

        # Importancia de características
        importancia_features = dict(zip(
            prep_data['feature_columns'],
            best_model.feature_importances_
        ))

        # Información del árbol
        tree_info = {
            'n_nodes': best_model.tree_.node_count,
            'max_depth': best_model.tree_.max_depth,
            'n_leaves': best_model.tree_.n_leaves,
            'n_features': best_model.n_features_in_
        }

        # Reglas del árbol (primeras 10)
        if hasattr(best_model, 'tree_'):
            tree_rules = export_text(best_model, feature_names=list(prep_data['feature_columns']))
            tree_rules_lines = tree_rules.split('\n')[:50]  # Primeras 50 líneas
            tree_rules_summary = '\n'.join(tree_rules_lines)
        else:
            tree_rules_summary = "No disponible"

        return {
            'tipo': 'arbol_decision',
            'es_clasificacion': prep_data['is_classification'],
            'modelo': best_model,
            'target_column': target_column,
            'feature_columns': prep_data['feature_columns'],
            'parametros': best_params,
            'metricas': {
                'train': train_metrics,
                'test': test_metrics
            },
            'importancia_features': importancia_features,
            'tree_info': tree_info,
            'tree_rules': tree_rules_summary,
            'preprocessing_info': prep_data['preprocessing_info']
        }

    except Exception as e:
        logger.error(f"Error en árbol de decisión: {str(e)}")
        return {'error': str(e)}

def random_forest(data: pd.DataFrame, target_column: str,
                 feature_columns: Optional[List[str]] = None,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 max_features: Union[str, int, float] = 'sqrt',
                 optimize_params: bool = True,
                 n_jobs: int = -1) -> Dict[str, Any]:
    """
    Random Forest optimizado con paralelización
    """
    try:
        # Preparar datos
        prep_data = preparar_datos_supervisado_optimizado(
            data, target_column, feature_columns
        )

        X_train = prep_data['X_train_scaled']
        X_test = prep_data['X_test_scaled']
        y_train = prep_data['y_train']
        y_test = prep_data['y_test']

        # Seleccionar tipo de Random Forest
        if prep_data['is_classification']:
            model_class = RandomForestClassifier
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.5, 0.7]
            }
            scoring = 'accuracy'
        else:
            model_class = RandomForestRegressor
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.5, 0.7]
            }
            scoring = 'r2'

        if optimize_params:
            # Usar RandomizedSearchCV para eficiencia
            model = RandomizedSearchCV(
                model_class(random_state=42, n_jobs=n_jobs, oob_score=True),
                param_grid,
                n_iter=20,
                cv=5,
                scoring=scoring,
                n_jobs=n_jobs,
                random_state=42,
                verbose=0
            )
            model.fit(X_train, y_train)
            best_model = model.best_estimator_
            best_params = model.best_params_
            cv_results = pd.DataFrame(model.cv_results_)
        else:
            best_model = model_class(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                random_state=42,
                n_jobs=n_jobs,
                oob_score=True
            )
            best_model.fit(X_train, y_train)
            best_params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'max_features': max_features
            }
            cv_results = None

        # Predicciones
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        # Probabilidades para clasificación
        if prep_data['is_classification'] and hasattr(best_model, 'predict_proba'):
            y_proba_test = best_model.predict_proba(X_test)
        else:
            y_proba_test = None

        # Métricas
        if prep_data['is_classification']:
            train_metrics = _calcular_metricas_clasificacion(y_train, y_pred_train, best_model, X_train)
            test_metrics = _calcular_metricas_clasificacion(y_test, y_pred_test, best_model, X_test)

            # Matriz de confusión
            cm = confusion_matrix(y_test, y_pred_test)
            confusion_matrix_data = cm.tolist()
        else:
            train_metrics = _calcular_metricas_regresion(y_train, y_pred_train)
            test_metrics = _calcular_metricas_regresion(y_test, y_pred_test)
            confusion_matrix_data = None

        # Importancia de características
        importancia_features = dict(zip(
            prep_data['feature_columns'],
            best_model.feature_importances_
        ))

        # Ordenar por importancia
        importancia_ordenada = dict(sorted(
            importancia_features.items(),
            key=lambda x: x[1],
            reverse=True
        ))

        # OOB Score si está disponible
        oob_score = best_model.oob_score_ if hasattr(best_model, 'oob_score_') else None

        # Información del bosque
        forest_info = {
            'n_estimators': best_model.n_estimators,
            'n_features': best_model.n_features_in_,
            'oob_score': oob_score
        }

        # Análisis de estabilidad del modelo
        if len(X_train) > 100:
            stability_scores = _analyze_model_stability(best_model, X_train, y_train, n_iterations=5)
        else:
            stability_scores = None

        return {
            'tipo': 'random_forest',
            'es_clasificacion': prep_data['is_classification'],
            'modelo': best_model,
            'target_column': target_column,
            'feature_columns': prep_data['feature_columns'],
            'parametros': best_params,
            'metricas': {
                'train': train_metrics,
                'test': test_metrics,
                'oob_score': oob_score
            },
            'importancia_features': importancia_ordenada,
            'forest_info': forest_info,
            'confusion_matrix': confusion_matrix_data,
            'classes': prep_data['classes'],
            'stability_scores': stability_scores,
            'cv_results': cv_results.to_dict() if cv_results is not None else None,
            'preprocessing_info': prep_data['preprocessing_info']
        }

    except Exception as e:
        logger.error(f"Error en Random Forest: {str(e)}")
        return {'error': str(e)}

def svm_modelo(data: pd.DataFrame, target_column: str,
              feature_columns: Optional[List[str]] = None,
              kernel: str = 'rbf',
              C: float = 1.0,
              gamma: Union[str, float] = 'scale',
              optimize_params: bool = True) -> Dict[str, Any]:
    """
    Support Vector Machine optimizado
    """
    try:
        # Preparar datos - SVM siempre necesita escalado
        prep_data = preparar_datos_supervisado_optimizado(
            data, target_column, feature_columns,
            scale_method='standard'
        )

        X_train = prep_data['X_train_scaled']
        X_test = prep_data['X_test_scaled']
        y_train = prep_data['y_train']
        y_test = prep_data['y_test']

        # Limitar tamaño del dataset para SVM si es muy grande
        max_samples = 5000
        if len(X_train) > max_samples:
            logger.warning(f"Dataset grande ({len(X_train)} muestras). Limitando a {max_samples} para SVM.")
            idx = np.random.choice(len(X_train), max_samples, replace=False)
            X_train_subset = X_train.iloc[idx]
            y_train_subset = y_train.iloc[idx]
        else:
            X_train_subset = X_train
            y_train_subset = y_train

        # Seleccionar tipo de SVM
        if prep_data['is_classification']:
            model_class = SVC
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
            base_params = {'probability': True, 'random_state': 42}
            scoring = 'accuracy'
        else:
            model_class = SVR
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'epsilon': [0.01, 0.1, 0.2]
            }
            base_params = {}
            scoring = 'r2'

        if optimize_params:
            # Grid Search con validación cruzada
            model = GridSearchCV(
                model_class(**base_params),
                param_grid,
                cv=3,  # Menos folds para SVM
                scoring=scoring,
                n_jobs=-1,
                verbose=0
            )
            model.fit(X_train_subset, y_train_subset)
            best_model = model.best_estimator_
            best_params = model.best_params_
        else:
            params = {
                'kernel': kernel,
                'C': C,
                'gamma': gamma,
                **base_params
            }
            best_model = model_class(**params)
            best_model.fit(X_train_subset, y_train_subset)
            best_params = params

        # Predicciones
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        # Métricas
        if prep_data['is_classification']:
            train_metrics = _calcular_metricas_clasificacion(y_train, y_pred_train, best_model, X_train)
            test_metrics = _calcular_metricas_clasificacion(y_test, y_pred_test, best_model, X_test)
        else:
            train_metrics = _calcular_metricas_regresion(y_train, y_pred_train)
            test_metrics = _calcular_metricas_regresion(y_test, y_pred_test)

        # Información específica de SVM
        svm_info = {
            'kernel': best_model.kernel,
            'C': best_model.C,
            'gamma': best_model.gamma if hasattr(best_model, 'gamma') else None,
            'n_support_vectors': best_model.n_support_.sum() if hasattr(best_model, 'n_support_') else None,
            'support_vector_ratio': (best_model.n_support_.sum() / len(X_train_subset)) if hasattr(best_model, 'n_support_') else None
        }

        return {
            'tipo': 'svm',
            'es_clasificacion': prep_data['is_classification'],
            'modelo': best_model,
            'target_column': target_column,
            'feature_columns': prep_data['feature_columns'],
            'parametros': best_params,
            'metricas': {
                'train': train_metrics,
                'test': test_metrics
            },
            'svm_info': svm_info,
            'preprocessing_info': prep_data['preprocessing_info']
        }

    except Exception as e:
        logger.error(f"Error en SVM: {str(e)}")
        return {'error': str(e)}

# ==================== COMPARACIÓN DE MODELOS ====================

def comparar_modelos_supervisado(data: pd.DataFrame, target_column: str,
                               feature_columns: Optional[List[str]] = None,
                               modelos: List[str] = None,
                               optimize_all: bool = True,
                               cv_folds: int = 5,
                               n_jobs: int = -1) -> Dict[str, Any]:
    """
    Comparación exhaustiva y paralela de múltiples modelos
    """
    try:
        if modelos is None:
            modelos = ['linear', 'tree', 'forest', 'svm']

        # Preparar datos una sola vez
        prep_data = preparar_datos_supervisado_optimizado(
            data, target_column, feature_columns,
            scale_method='standard'
        )

        # Determinar tipo de problema y métricas
        if prep_data['is_classification']:
            scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
            primary_metric = 'accuracy'
        else:
            scoring_metrics = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
            primary_metric = 'r2'

        resultados = {
            'target_column': target_column,
            'feature_columns': prep_data['feature_columns'],
            'is_classification': prep_data['is_classification'],
            'modelos': {},
            'ranking': [],
            'preprocessing_info': prep_data['preprocessing_info']
        }

        # Función para entrenar un modelo
        def entrenar_modelo(modelo_nombre):
            logger.info(f"Entrenando {modelo_nombre}...")

            try:
                if modelo_nombre == 'linear':
                    if prep_data['is_classification']:
                        resultado = _entrenar_logistic_regression(
                            prep_data, optimize_all, cv_folds
                        )
                    else:
                        resultado = regresion_lineal_multiple(
                            data, target_column, feature_columns,
                            regularization='ridge' if optimize_all else 'none',
                            optimize_params=optimize_all
                        )

                elif modelo_nombre == 'tree':
                    resultado = arbol_decision(
                        data, target_column, feature_columns,
                        optimize_params=optimize_all
                    )

                elif modelo_nombre == 'forest':
                    resultado = random_forest(
                        data, target_column, feature_columns,
                        optimize_params=optimize_all,
                        n_jobs=n_jobs
                    )

                elif modelo_nombre == 'svm':
                    # SVM puede ser lento, limitar datos si es necesario
                    if len(prep_data['X_train']) > 1000:
                        logger.warning("Dataset grande para SVM, usando submuestra")
                    resultado = svm_modelo(
                        data, target_column, feature_columns,
                        optimize_params=optimize_all
                    )

                elif modelo_nombre == 'gradient_boosting':
                    resultado = _entrenar_gradient_boosting(
                        prep_data, optimize_all
                    )

                elif modelo_nombre == 'neural_network':
                    resultado = _entrenar_neural_network(
                        prep_data, optimize_all
                    )

                else:
                    resultado = {'error': f'Modelo {modelo_nombre} no reconocido'}

                return modelo_nombre, resultado

            except Exception as e:
                logger.error(f"Error en {modelo_nombre}: {str(e)}")
                return modelo_nombre, {'error': str(e)}

        # Entrenar modelos (opcionalmente en paralelo)
        if n_jobs == 1:
            # Secuencial
            for modelo in modelos:
                nombre, resultado = entrenar_modelo(modelo)
                resultados['modelos'][nombre] = resultado
        else:
            # Paralelo
            with Parallel(n_jobs=min(n_jobs, len(modelos)), backend='threading') as parallel:
                results_list = parallel(delayed(entrenar_modelo)(modelo) for modelo in modelos)

                for nombre, resultado in results_list:
                    resultados['modelos'][nombre] = resultado

        # Crear ranking basado en métrica principal
        ranking = []
        for nombre, resultado in resultados['modelos'].items():
            if 'error' not in resultado and 'metricas' in resultado:
                metricas = resultado['metricas']['test']
                score = metricas.get(primary_metric,
                                   metricas.get('accuracy',
                                   metricas.get('r2', -np.inf)))

                ranking.append({
                    'modelo': nombre,
                    'score': float(score),
                    'metrica': primary_metric,
                    'metricas_completas': metricas
                })

        # Ordenar ranking
        ranking.sort(key=lambda x: x['score'], reverse=True)
        resultados['ranking'] = ranking

        # Estadísticas de comparación
        if len(ranking) >= 2:
            scores = [item['score'] for item in ranking]
            resultados['estadisticas'] = {
                'mejor_modelo': ranking[0]['modelo'],
                'mejor_score': ranking[0]['score'],
                'diferencia_primero_segundo': ranking[0]['score'] - ranking[1]['score'],
                'promedio_scores': np.mean(scores),
                'std_scores': np.std(scores),
                'rango_scores': max(scores) - min(scores)
            }

        # Análisis de consenso entre modelos
        if not prep_data['is_classification'] and len(ranking) >= 3:
            resultados['analisis_consenso'] = _analyze_model_consensus(
                resultados['modelos'], prep_data
            )

        return resultados

    except Exception as e:
        logger.error(f"Error en comparación de modelos: {str(e)}")
        return {'error': str(e)}

# ==================== FUNCIONES AUXILIARES OPTIMIZADAS ====================

def _calcular_metricas_regresion(y_true, y_pred) -> Dict[str, float]:
    """Calcular métricas completas para regresión"""
    # Convertir a arrays numpy si es necesario
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Verificar que no hay NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    metrics = {
        'r2': float(r2_score(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'explained_variance': float(explained_variance_score(y_true, y_pred))
    }

    # MAPE solo si no hay ceros en y_true
    if np.all(y_true != 0):
        metrics['mape'] = float(mean_absolute_percentage_error(y_true, y_pred))

    # Métricas adicionales
    residuos = y_true - y_pred
    metrics['max_error'] = float(np.max(np.abs(residuos)))
    metrics['std_residuos'] = float(np.std(residuos))

    return metrics

def _calcular_metricas_clasificacion(y_true, y_pred, model=None, X=None) -> Dict[str, float]:
    """Calcular métricas completas para clasificación"""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    }

    # Métricas por clase si hay pocas clases
    unique_classes = np.unique(y_true)
    if len(unique_classes) <= 10:
        metrics['per_class_precision'] = precision_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()
        metrics['per_class_recall'] = recall_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()
        metrics['per_class_f1'] = f1_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()

    # ROC AUC para clasificación binaria
    if len(unique_classes) == 2 and model is not None and hasattr(model, 'predict_proba') and X is not None:
        try:
            y_proba = model.predict_proba(X)[:, 1]
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            pass

    return metrics


def _analizar_residuos(residuos) -> Dict[str, float]:
    """Análisis estadístico de residuos - SIN SCIPY"""
    residuos = np.array(residuos)

    # ❌ ELIMINADA: from scipy import stats

    analisis = {
        'media': float(np.mean(residuos)),
        'std': float(np.std(residuos)),
        'min': float(np.min(residuos)),
        'max': float(np.max(residuos)),
        'q25': float(np.percentile(residuos, 25)),
        'q50': float(np.percentile(residuos, 50)),
        'q75': float(np.percentile(residuos, 75)),
        'skewness': float(_calculate_skewness(residuos)),
        'kurtosis': float(_calculate_kurtosis(residuos))
    }

    # Test de normalidad simplificado
    if len(residuos) >= 20 and len(residuos) <= 5000:
        try:
            p_value = _shapiro_wilk_simple(residuos)
            analisis['normaltest_pvalue'] = float(p_value)
            analisis['es_normal'] = p_value > 0.05
        except:
            analisis['normaltest_pvalue'] = 0.01
            analisis['es_normal'] = False

    return analisis


# 🔥 NUEVAS FUNCIONES AUXILIARES - Añádelas al final del archivo

def _calculate_skewness(data):
    """Calcular skewness usando solo NumPy"""
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data, ddof=0)

    if std == 0:
        return 0.0

    n = len(data)
    skew = np.sum(((data - mean) / std) ** 3) / n
    return skew


def _calculate_kurtosis(data):
    """Calcular kurtosis usando solo NumPy"""
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data, ddof=0)

    if std == 0:
        return 0.0

    n = len(data)
    kurt = np.sum(((data - mean) / std) ** 4) / n - 3  # -3 para excess kurtosis
    return kurt


def _shapiro_wilk_simple(data):
    """
    Test de normalidad simplificado usando solo NumPy
    Aproximación del test de Shapiro-Wilk
    """
    try:
        data = np.array(data)
        n = len(data)

        if n < 20 or n > 5000:
            return 0.5  # Valor neutral

        # Ordenar los datos
        x = np.sort(data)

        # Calcular estadística simplificada
        # Esta es una aproximación, no el test completo
        mean = np.mean(x)
        std = np.std(x, ddof=1)

        if std == 0:
            return 1.0

        # Test basado en la distancia de los cuantiles esperados
        expected_quantiles = np.linspace(-2, 2, n)
        observed_quantiles = (x - mean) / std

        # Calcular correlación entre cuantiles esperados y observados
        correlation = np.corrcoef(expected_quantiles, observed_quantiles)[0, 1]

        # Convertir correlación a p-value aproximado
        # Valores más cercanos a 1 indican mayor normalidad
        if correlation > 0.95:
            p_value = 0.8
        elif correlation > 0.90:
            p_value = 0.3
        elif correlation > 0.85:
            p_value = 0.1
        else:
            p_value = 0.01

        return p_value

    except:
        return 0.01  # Asumir no normal si hay error


def _calculate_vif(X: pd.DataFrame) -> Dict[str, float]:
    """
    Calcular Variance Inflation Factor SIMPLIFICADO sin statsmodels
    """
    vif_data = {}

    try:
        # Método simplificado usando correlaciones
        # VIF = 1 / (1 - R²) donde R² es la correlación múltiple

        corr_matrix = X.corr().abs()

        for i, col in enumerate(X.columns):
            # Calcular R² aproximado basado en correlaciones máximas
            other_cols = [c for c in X.columns if c != col]
            if len(other_cols) > 0:
                max_corr = corr_matrix.loc[col, other_cols].max()
                # Aproximación simple: VIF ≈ 1/(1-max_correlation²)
                r_squared_approx = max_corr ** 2
                if r_squared_approx < 0.999:  # Evitar división por cero
                    vif_approx = 1 / (1 - r_squared_approx)
                    vif_data[col] = float(vif_approx)
                else:
                    vif_data[col] = 999.0  # VIF muy alto
            else:
                vif_data[col] = 1.0

    except Exception as e:
        print(f"Error calculando VIF simplificado: {e}")
        # Si falla, retornar VIF = 1 para todas las variables
        for col in X.columns:
            vif_data[col] = 1.0

    return vif_data

def _analyze_model_stability(model, X, y, n_iterations: int = 5) -> Dict[str, float]:
    """Analizar estabilidad del modelo con diferentes submuestras"""
    scores = []

    for i in range(n_iterations):
        # Crear submuestra aleatoria (80% de los datos)
        idx = np.random.choice(len(X), int(0.8 * len(X)), replace=False)
        X_subset = X.iloc[idx]
        y_subset = y.iloc[idx]

        # Evaluar modelo
        score = model.score(X_subset, y_subset)
        scores.append(score)

    return {
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores)),
        'min_score': float(np.min(scores)),
        'max_score': float(np.max(scores)),
        'cv_stability': float(np.std(scores) / np.mean(scores))  # Coeficiente de variación
    }

def _analyze_model_consensus(models_dict: Dict, prep_data: Dict) -> Dict[str, Any]:
    """Analizar consenso entre predicciones de diferentes modelos"""
    predictions = []
    model_names = []

    # Recopilar predicciones de test
    for name, result in models_dict.items():
        if 'error' not in result and 'datos_prediccion' in result:
            pred = result['datos_prediccion'].get('y_pred_test',
                   result.get('predicciones', {}).get('test'))
            if pred is not None:
                predictions.append(np.array(pred))
                model_names.append(name)

    if len(predictions) < 2:
        return {}

    # Convertir a array
    predictions = np.array(predictions)

    # Calcular estadísticas de consenso
    consensus = {
        'mean_prediction': np.mean(predictions, axis=0).tolist(),
        'std_prediction': np.std(predictions, axis=0).tolist(),
        'prediction_range': (np.max(predictions, axis=0) - np.min(predictions, axis=0)).tolist(),
        'correlation_matrix': np.corrcoef(predictions).tolist(),
        'model_names': model_names
    }

    return consensus

# ==================== FUNCIONES ADICIONALES DE ML ====================

def _entrenar_logistic_regression(prep_data: Dict, optimize: bool = True,
                                 cv_folds: int = 5) -> Dict[str, Any]:
    """Entrenar regresión logística para clasificación"""
    X_train = prep_data['X_train_scaled']
    X_test = prep_data['X_test_scaled']
    y_train = prep_data['y_train']
    y_test = prep_data['y_test']

    if optimize:
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs'],
            'max_iter': [1000]
        }

        model = GridSearchCV(
            LogisticRegression(random_state=42),
            param_grid,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        best_model = model.best_estimator_
        best_params = model.best_params_
    else:
        best_model = LogisticRegression(max_iter=1000, random_state=42)
        best_model.fit(X_train, y_train)
        best_params = {'C': 1.0, 'solver': 'lbfgs'}

    # Predicciones
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # Métricas
    train_metrics = _calcular_metricas_clasificacion(y_train, y_pred_train, best_model, X_train)
    test_metrics = _calcular_metricas_clasificacion(y_test, y_pred_test, best_model, X_test)

    # Coeficientes
    coeficientes = {}
    if len(best_model.coef_.shape) == 1:
        # Clasificación binaria
        for i, feature in enumerate(prep_data['feature_columns']):
            coeficientes[feature] = float(best_model.coef_[i])
    else:
        # Clasificación multiclase
        for i, feature in enumerate(prep_data['feature_columns']):
            coeficientes[feature] = float(np.mean(np.abs(best_model.coef_[:, i])))

    return {
        'tipo': 'logistic_regression',
        'es_clasificacion': True,
        'modelo': best_model,
        'parametros': best_params,
        'coeficientes': coeficientes,
        'intercepto': float(best_model.intercept_[0]) if len(best_model.intercept_) == 1 else best_model.intercept_.tolist(),
        'metricas': {
            'train': train_metrics,
            'test': test_metrics
        }
    }

def _entrenar_gradient_boosting(prep_data: Dict, optimize: bool = True) -> Dict[str, Any]:
    """Entrenar Gradient Boosting"""
    X_train = prep_data['X_train_scaled']
    X_test = prep_data['X_test_scaled']
    y_train = prep_data['y_train']
    y_test = prep_data['y_test']

    if prep_data['is_classification']:
        model_class = GradientBoostingClassifier
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
        scoring = 'accuracy'
    else:
        model_class = GradientBoostingRegressor
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
        scoring = 'r2'

    if optimize:
        # Usar RandomizedSearchCV para eficiencia
        model = RandomizedSearchCV(
            model_class(random_state=42),
            param_grid,
            n_iter=15,
            cv=3,
            scoring=scoring,
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train, y_train)
        best_model = model.best_estimator_
        best_params = model.best_params_
    else:
        best_model = model_class(n_estimators=100, random_state=42)
        best_model.fit(X_train, y_train)
        best_params = {'n_estimators': 100}

    # Predicciones
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # Métricas
    if prep_data['is_classification']:
        train_metrics = _calcular_metricas_clasificacion(y_train, y_pred_train, best_model, X_train)
        test_metrics = _calcular_metricas_clasificacion(y_test, y_pred_test, best_model, X_test)
    else:
        train_metrics = _calcular_metricas_regresion(y_train, y_pred_train)
        test_metrics = _calcular_metricas_regresion(y_test, y_pred_test)

    # Importancia de características
    importancia_features = dict(zip(
        prep_data['feature_columns'],
        best_model.feature_importances_
    ))

    return {
        'tipo': 'gradient_boosting',
        'es_clasificacion': prep_data['is_classification'],
        'modelo': best_model,
        'parametros': best_params,
        'metricas': {
            'train': train_metrics,
            'test': test_metrics
        },
        'importancia_features': importancia_features
    }

def _entrenar_neural_network(prep_data: Dict, optimize: bool = True) -> Dict[str, Any]:
    """Entrenar Red Neuronal (MLP)"""
    X_train = prep_data['X_train_scaled']
    X_test = prep_data['X_test_scaled']
    y_train = prep_data['y_train']
    y_test = prep_data['y_test']

    if prep_data['is_classification']:
        model_class = MLPClassifier
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01]
        }
        base_params = {
            'random_state': 42,
            'max_iter': 1000,
            'early_stopping': True,
            'validation_fraction': 0.1
        }
        scoring = 'accuracy'
    else:
        model_class = MLPRegressor
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01]
        }
        base_params = {
            'random_state': 42,
            'max_iter': 1000,
            'early_stopping': True,
            'validation_fraction': 0.1
        }
        scoring = 'r2'

    if optimize:
        model = RandomizedSearchCV(
            model_class(**base_params),
            param_grid,
            n_iter=10,
            cv=3,
            scoring=scoring,
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train, y_train)
        best_model = model.best_estimator_
        best_params = model.best_params_
    else:
        best_model = model_class(
            hidden_layer_sizes=(100, 50),
            **base_params
        )
        best_model.fit(X_train, y_train)
        best_params = {'hidden_layer_sizes': (100, 50)}

    # Predicciones
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # Métricas
    if prep_data['is_classification']:
        train_metrics = _calcular_metricas_clasificacion(y_train, y_pred_train, best_model, X_train)
        test_metrics = _calcular_metricas_clasificacion(y_test, y_pred_test, best_model, X_test)
    else:
        train_metrics = _calcular_metricas_regresion(y_train, y_pred_train)
        test_metrics = _calcular_metricas_regresion(y_test, y_pred_test)

    # Información de la red
    nn_info = {
        'n_layers': len(best_model.hidden_layer_sizes) + 2,  # Hidden + input + output
        'n_iterations': best_model.n_iter_,
        'loss_curve': best_model.loss_curve_[-10:] if hasattr(best_model, 'loss_curve_') else None
    }

    return {
        'tipo': 'neural_network',
        'es_clasificacion': prep_data['is_classification'],
        'modelo': best_model,
        'parametros': best_params,
        'metricas': {
            'train': train_metrics,
            'test': test_metrics
        },
        'nn_info': nn_info
    }

# ==================== FUNCIONES DE VISUALIZACIÓN ====================

def generar_visualizaciones_ml(resultado: Dict[str, Any], figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Generar visualizaciones según el tipo de modelo
    """
    tipo = resultado.get('tipo', '')

    if tipo == 'regresion_lineal_simple':
        return _plot_regresion_simple(resultado, figsize)
    elif tipo == 'regresion_lineal_multiple':
        return _plot_regresion_multiple(resultado, figsize)
    elif tipo in ['arbol_decision', 'random_forest', 'gradient_boosting']:
        if resultado.get('es_clasificacion'):
            return _plot_clasificacion(resultado, figsize)
        else:
            return _plot_regresion_tree(resultado, figsize)
    elif tipo == 'svm':
        return _plot_svm(resultado, figsize)
    elif tipo == 'comparar_modelos':
        return _plot_comparacion(resultado, figsize)
    else:
        return _plot_generico(resultado, figsize)


def _plot_regresion_simple(resultado: Dict, figsize: Tuple) -> plt.Figure:
    """Visualización para regresión lineal simple"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Scatter plot con línea de regresión
    ax = axes[0, 0]
    X_test = resultado['datos']['X_test']
    y_test = resultado['datos']['y_test']
    y_pred = resultado['datos']['y_pred_test']

    ax.scatter(X_test, y_test, alpha=0.5, label='Datos reales')
    ax.plot(sorted(X_test),
            [resultado['coeficiente'] * x + resultado['intercepto'] for x in sorted(X_test)],
            'r-', label=resultado['ecuacion'])
    ax.set_xlabel(resultado['x_column'])
    ax.set_ylabel(resultado['y_column'])
    ax.set_title('Regresión Lineal Simple')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Residuos vs Predichos
    ax = axes[0, 1]
    residuos = np.array(y_test) - np.array(y_pred)
    ax.scatter(y_pred, residuos, alpha=0.5)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Valores Predichos')
    ax.set_ylabel('Residuos')
    ax.set_title('Análisis de Residuos')
    ax.grid(True, alpha=0.3)

    # 3. Q-Q plot SIMPLIFICADO - SIN SCIPY
    ax = axes[1, 0]
    # ❌ ELIMINADA: from scipy import stats
    # ❌ ELIMINADA: stats.probplot(residuos, dist="norm", plot=ax)

    # 🔥 Q-Q plot manual con NumPy
    try:
        sorted_residuos = np.sort(residuos)
        n = len(sorted_residuos)

        if n > 2:
            # Generar cuantiles teóricos de distribución normal estándar
            # Usando aproximación de cuantiles normales
            p = np.arange(1, n + 1) / (n + 1)  # Probabilidades

            # Aproximación de cuantiles normales usando función inversa
            # Esta es una aproximación de la función cuantil normal
            z_scores = []
            for prob in p:
                if prob <= 0.5:
                    # Aproximación para la mitad inferior
                    t = np.sqrt(-2 * np.log(prob))
                    z = -(t - (2.30753 + t * 0.27061) / (1 + t * (0.99229 + t * 0.04481)))
                else:
                    # Usar simetría para la mitad superior
                    prob_sym = 1 - prob
                    t = np.sqrt(-2 * np.log(prob_sym))
                    z = t - (2.30753 + t * 0.27061) / (1 + t * (0.99229 + t * 0.04481))
                z_scores.append(z)

            theoretical_quantiles = np.array(z_scores)

            # Plot Q-Q
            ax.scatter(theoretical_quantiles, sorted_residuos, alpha=0.6, s=30)

            # Línea de referencia (y = x)
            min_val = min(theoretical_quantiles.min(), sorted_residuos.min())
            max_val = max(theoretical_quantiles.max(), sorted_residuos.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)

            ax.set_xlabel('Cuantiles Teóricos (Normal)')
            ax.set_ylabel('Cuantiles de Residuos')
            ax.set_title('Q-Q Plot vs Normal')
        else:
            ax.text(0.5, 0.5, 'Pocos datos\npara Q-Q plot',
                    ha='center', va='center', transform=ax.transAxes)
    except Exception as e:
        ax.text(0.5, 0.5, f'Error en Q-Q plot:\n{str(e)}',
                ha='center', va='center', transform=ax.transAxes)

    ax.grid(True, alpha=0.3)

    # 4. Histograma de residuos
    ax = axes[1, 1]
    ax.hist(residuos, bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Residuos')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución de Residuos')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def _plot_regresion_multiple(resultado: Dict, figsize: Tuple) -> plt.Figure:
    """Visualización para regresión múltiple"""
    fig = plt.figure(figsize=figsize)

    # Determinar el layout según la cantidad de información
    n_features = len(resultado.get('coeficientes', {}))

    if n_features <= 10:
        # Layout 2x3 para pocas características
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    else:
        # Layout 3x2 para muchas características
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Predicciones vs Reales
    ax1 = fig.add_subplot(gs[0, 0])
    y_test = resultado['datos_prediccion']['y_test']
    y_pred = resultado['datos_prediccion']['y_pred_test']

    ax1.scatter(y_test, y_pred, alpha=0.5)
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfecta')
    ax1.set_xlabel('Valores Reales')
    ax1.set_ylabel('Predicciones')
    ax1.set_title(f'R² = {resultado["metricas"]["test"]["r2"]:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Residuos
    ax2 = fig.add_subplot(gs[0, 1])
    residuos = resultado['datos_prediccion']['residuos_test']
    ax2.scatter(y_pred, residuos, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicciones')
    ax2.set_ylabel('Residuos')
    ax2.set_title('Análisis de Residuos')
    ax2.grid(True, alpha=0.3)

    # 3. Importancia de características
    if n_features <= 10:
        ax3 = fig.add_subplot(gs[0, 2])
    else:
        ax3 = fig.add_subplot(gs[0:2, 1])

    coefs = resultado['coeficientes']
    features = list(coefs.keys())
    values = list(coefs.values())

    # Ordenar por valor absoluto
    sorted_idx = np.argsort(np.abs(values))
    features_sorted = [features[i] for i in sorted_idx]
    values_sorted = [values[i] for i in sorted_idx]

    # Mostrar solo top 15 si hay muchas
    if len(features_sorted) > 15:
        features_sorted = features_sorted[-15:]
        values_sorted = values_sorted[-15:]

    colors = ['red' if v < 0 else 'green' for v in values_sorted]
    ax3.barh(features_sorted, values_sorted, color=colors)
    ax3.set_xlabel('Coeficiente')
    ax3.set_title('Importancia de Variables')
    ax3.grid(True, alpha=0.3)

    # 4. Distribución de residuos
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(residuos, bins=30, edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Residuos')
    ax4.set_ylabel('Frecuencia')
    ax4.set_title('Distribución de Residuos')
    ax4.grid(True, alpha=0.3)

    # 5. Q-Q Plot
    ax5 = fig.add_subplot(gs[1, 1])

    try:
        residuos_array = np.array(residuos)
        sorted_residuos = np.sort(residuos_array)
        n = len(sorted_residuos)

        if n > 2:
            # Generar cuantiles teóricos usando aproximación de Box-Muller
            p = np.arange(1, n + 1) / (n + 1)

            # Aproximación de cuantiles normales inversos
            z_scores = []
            for prob in p:
                if prob <= 0.5:
                    t = np.sqrt(-2 * np.log(prob))
                    z = -(t - (2.30753 + t * 0.27061) / (1 + t * (0.99229 + t * 0.04481)))
                else:
                    prob_sym = 1 - prob
                    t = np.sqrt(-2 * np.log(prob_sym))
                    z = t - (2.30753 + t * 0.27061) / (1 + t * (0.99229 + t * 0.04481))
                z_scores.append(z)

            theoretical_quantiles = np.array(z_scores)

            # Crear scatter plot
            ax5.scatter(theoretical_quantiles, sorted_residuos, alpha=0.6, s=25)

            # Línea de referencia perfecta
            min_val = min(theoretical_quantiles.min(), sorted_residuos.min())
            max_val = max(theoretical_quantiles.max(), sorted_residuos.max())
            ax5.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)

            ax5.set_xlabel('Cuantiles Teóricos')
            ax5.set_ylabel('Cuantiles de Residuos')
        else:
            ax5.text(0.5, 0.5, 'Datos insuficientes\npara Q-Q plot',
                     ha='center', va='center', transform=ax5.transAxes, fontsize=12)
    except Exception as e:
        ax5.text(0.5, 0.5, f'Error generando Q-Q plot:\n{str(e)}',
                 ha='center', va='center', transform=ax5.transAxes, fontsize=10)

    ax5.set_title('Q-Q Plot vs Normal')
    ax5.grid(True, alpha=0.3)

    # 6. Métricas o VIF
    if n_features <= 10:
        ax6 = fig.add_subplot(gs[1, 2])

        # Si hay VIF scores, mostrarlos
        if 'vif_scores' in resultado and resultado['vif_scores']:
            vif_features = list(resultado['vif_scores'].keys())[:10]
            vif_values = [resultado['vif_scores'][f] for f in vif_features]

            ax6.bar(range(len(vif_features)), vif_values)
            ax6.set_xticks(range(len(vif_features)))
            ax6.set_xticklabels(vif_features, rotation=45, ha='right')
            ax6.axhline(y=10, color='r', linestyle='--', label='VIF = 10')
            ax6.set_ylabel('VIF')
            ax6.set_title('Multicolinealidad (VIF)')
            ax6.legend()
        else:
            # Mostrar tabla de métricas
            metrics_text = f"Métricas Test:\n"
            metrics_text += f"R² = {resultado['metricas']['test']['r2']:.4f}\n"
            metrics_text += f"RMSE = {resultado['metricas']['test']['rmse']:.4f}\n"
            metrics_text += f"MAE = {resultado['metricas']['test']['mae']:.4f}\n"
            if 'mape' in resultado['metricas']['test']:
                metrics_text += f"MAPE = {resultado['metricas']['test']['mape']:.2%}\n"
            metrics_text += f"\nRegularización: {resultado.get('regularization', 'none')}"
            if resultado.get('alpha_optimo'):
                metrics_text += f"\nAlpha óptimo: {resultado['alpha_optimo']:.4f}"

            ax6.text(0.1, 0.5, metrics_text, transform=ax6.transAxes,
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax6.axis('off')

    plt.suptitle(f'Regresión Lineal Múltiple - {resultado["target_column"]}', fontsize=14)
    return fig

def _plot_clasificacion(resultado: Dict, figsize: Tuple) -> plt.Figure:
    """Visualización para modelos de clasificación"""
    fig = plt.figure(figsize=figsize)

    # Determinar número de subplots según información disponible
    has_importance = 'importancia_features' in resultado
    has_cm = 'confusion_matrix' in resultado
    n_plots = 2 + (1 if has_importance else 0) + (1 if has_cm else 0)

    if n_plots <= 4:
        rows, cols = 2, 2
    else:
        rows, cols = 2, 3

    plot_idx = 1

    # 1. Matriz de confusión
    if has_cm:
        ax = plt.subplot(rows, cols, plot_idx)
        plot_idx += 1

        cm = np.array(resultado['confusion_matrix'])
        im = ax.imshow(cm, cmap='Blues', aspect='auto')

        # Agregar números en las celdas
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text = ax.text(j, i, str(cm[i, j]),
                               ha="center", va="center",
                               color="white" if cm[i, j] > cm.max() / 2 else "black",
                               fontsize=12, fontweight='bold')

        plt.colorbar(im, ax=ax)

        if 'classes' in resultado and resultado['classes']:
            ax.set_xticklabels(resultado['classes'])
            ax.set_yticklabels(resultado['classes'])

        ax.set_xlabel('Predicción')
        ax.set_ylabel('Real')
        ax.set_title('Matriz de Confusión')

    # 2. Métricas por clase
    ax = plt.subplot(rows, cols, plot_idx)
    plot_idx += 1

    metrics = resultado['metricas']['test']
    if 'per_class_f1' in metrics:
        classes = resultado.get('classes', [f'Clase {i}' for i in range(len(metrics['per_class_f1']))])
        f1_scores = metrics['per_class_f1']

        ax.bar(classes, f1_scores)
        ax.set_xlabel('Clase')
        ax.set_ylabel('F1-Score')
        ax.set_title('F1-Score por Clase')
        ax.set_ylim(0, 1.1)

        # Añadir línea de promedio
        avg_f1 = metrics['f1_score']
        ax.axhline(y=avg_f1, color='r', linestyle='--', label=f'Promedio: {avg_f1:.3f}')
        ax.legend()
    else:
        # Mostrar métricas generales
        metrics_text = f"Métricas Test:\n"
        metrics_text += f"Accuracy = {metrics['accuracy']:.3f}\n"
        metrics_text += f"Precision = {metrics['precision']:.3f}\n"
        metrics_text += f"Recall = {metrics['recall']:.3f}\n"
        metrics_text += f"F1-Score = {metrics['f1_score']:.3f}"

        ax.text(0.1, 0.5, metrics_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax.axis('off')

    # 3. Importancia de características
    if has_importance:
        ax = plt.subplot(rows, cols, plot_idx)
        plot_idx += 1

        features = list(resultado['importancia_features'].keys())[:15]
        importances = [resultado['importancia_features'][f] for f in features]

        ax.barh(features, importances)
        ax.set_xlabel('Importancia')
        ax.set_title('Top 15 Características Importantes')
        ax.grid(True, alpha=0.3)

    # 4. Información del modelo
    ax = plt.subplot(rows, cols, plot_idx)

    model_info = f"Modelo: {resultado['tipo']}\n\n"

    if 'parametros' in resultado:
        model_info += "Parámetros óptimos:\n"
        for param, value in list(resultado['parametros'].items())[:5]:
            model_info += f"  {param}: {value}\n"

    if 'tree_info' in resultado:
        model_info += f"\nInformación del árbol:\n"
        model_info += f"  Nodos: {resultado['tree_info']['n_nodes']}\n"
        model_info += f"  Profundidad: {resultado['tree_info']['max_depth']}\n"

    if 'forest_info' in resultado:
        model_info += f"\nInformación del bosque:\n"
        model_info += f"  Árboles: {resultado['forest_info']['n_estimators']}\n"
        if resultado['forest_info']['oob_score']:
            model_info += f"  OOB Score: {resultado['forest_info']['oob_score']:.3f}\n"

    ax.text(0.1, 0.5, model_info, transform=ax.transAxes,
            fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax.axis('off')

    plt.suptitle(f'Clasificación - {resultado.get("target_column", "Target")}', fontsize=14)
    plt.tight_layout()
    return fig

def _plot_regresion_tree(resultado: Dict, figsize: Tuple) -> plt.Figure:
    """Visualización para modelos de árbol en regresión"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Similar a regresión múltiple pero con información específica del árbol
    # Reutilizar la mayor parte del código de _plot_regresion_multiple
    # pero adaptado para árboles

    # Por brevedad, usar una versión simplificada
    return _plot_regresion_multiple(resultado, figsize)

def _plot_svm(resultado: Dict, figsize: Tuple) -> plt.Figure:
    """Visualización para SVM"""
    if resultado.get('es_clasificacion'):
        return _plot_clasificacion(resultado, figsize)
    else:
        # Para regresión, mostrar información específica de SVM
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Reutilizar plots de regresión pero añadir info de SVM
        # Por brevedad, usar versión simplificada
        return _plot_regresion_multiple(resultado, figsize)

def _plot_comparacion(resultado: Dict, figsize: Tuple) -> plt.Figure:
    """Visualización para comparación de modelos"""
    fig = plt.figure(figsize=figsize)

    ranking = resultado.get('ranking', [])
    if not ranking:
        return fig

    # 1. Ranking de modelos
    ax1 = plt.subplot(2, 2, 1)
    modelos = [item['modelo'] for item in ranking]
    scores = [item['score'] for item in ranking]

    bars = ax1.bar(modelos, scores)

    # Colorear el mejor
    mejor_idx = scores.index(max(scores))
    bars[mejor_idx].set_color('gold')

    ax1.set_ylabel(ranking[0]['metrica'].upper())
    ax1.set_title('Ranking de Modelos')
    ax1.set_ylim(0, max(1.1, 1.1 * max(scores)))

    # Añadir valores en las barras
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom')

    # 2. Comparación de métricas múltiples
    ax2 = plt.subplot(2, 2, 2)

    # Extraer todas las métricas disponibles
    all_metrics = set()
    for item in ranking:
        if 'metricas_completas' in item:
            all_metrics.update(item['metricas_completas'].keys())

    # Seleccionar métricas principales
    if resultado['is_classification']:
        metrics_to_show = [m for m in ['accuracy', 'precision', 'recall', 'f1_score'] if m in all_metrics]
    else:
        metrics_to_show = [m for m in ['r2', 'rmse', 'mae'] if m in all_metrics]

    if metrics_to_show:
        x = np.arange(len(modelos))
        width = 0.8 / len(metrics_to_show)

        for i, metric in enumerate(metrics_to_show):
            values = []
            for item in ranking:
                val = item.get('metricas_completas', {}).get(metric, 0)
                # Invertir RMSE y MAE para que mayor sea mejor
                if metric in ['rmse', 'mae', 'mse']:
                    val = -val if val != 0 else 0
                values.append(val)

            ax2.bar(x + i * width, values, width, label=metric.upper())

        ax2.set_xticks(x + width * (len(metrics_to_show) - 1) / 2)
        ax2.set_xticklabels(modelos)
        ax2.set_ylabel('Valor')
        ax2.set_title('Comparación de Métricas')
        ax2.legend()

    # 3. Estadísticas de comparación
    ax3 = plt.subplot(2, 2, 3)

    if 'estadisticas' in resultado:
        stats = resultado['estadisticas']
        stats_text = "Estadísticas de Comparación:\n\n"
        stats_text += f"Mejor modelo: {stats['mejor_modelo']}\n"
        stats_text += f"Mejor score: {stats['mejor_score']:.4f}\n"
        stats_text += f"Diferencia 1º-2º: {stats['diferencia_primero_segundo']:.4f}\n"
        stats_text += f"Promedio scores: {stats['promedio_scores']:.4f}\n"
        stats_text += f"Desv. estándar: {stats['std_scores']:.4f}\n"
        stats_text += f"Rango: {stats['rango_scores']:.4f}"

        ax3.text(0.1, 0.5, stats_text, transform=ax3.transAxes,
                fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax3.axis('off')

    # 4. Análisis de consenso (si existe)
    ax4 = plt.subplot(2, 2, 4)

    if 'analisis_consenso' in resultado and resultado['analisis_consenso']:
        consensus = resultado['analisis_consenso']

        # Mostrar correlación entre modelos
        if 'correlation_matrix' in consensus:
            corr_matrix = np.array(consensus['correlation_matrix'])
            im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

            # Etiquetas
            model_names = consensus.get('model_names', modelos)
            ax4.set_xticks(range(len(model_names)))
            ax4.set_yticks(range(len(model_names)))
            ax4.set_xticklabels(model_names, rotation=45, ha='right')
            ax4.set_yticklabels(model_names)

            # Añadir valores
            for i in range(len(model_names)):
                for j in range(len(model_names)):
                    text = ax4.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=9)

            ax4.set_title('Correlación entre Modelos')
            plt.colorbar(im, ax=ax4)
    else:
        # Información adicional
        info_text = "Información Adicional:\n\n"
        info_text += f"Modelos evaluados: {len(ranking)}\n"
        info_text += f"Tipo de problema: {'Clasificación' if resultado['is_classification'] else 'Regresión'}\n"
        info_text += f"Características: {len(resultado.get('feature_columns', []))}\n"

        if 'preprocessing_info' in resultado:
            prep_info = resultado['preprocessing_info']
            info_text += f"\nPreprocesamiento:\n"
            info_text += f"  Muestras totales: {prep_info.get('final_samples', 'N/A')}\n"
            info_text += f"  Método escalado: {prep_info.get('scale_method', 'N/A')}\n"

        ax4.text(0.1, 0.5, info_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='center')
        ax4.axis('off')

    plt.suptitle('Comparación de Modelos de Machine Learning', fontsize=14)
    plt.tight_layout()
    return fig

def _plot_generico(resultado: Dict, figsize: Tuple) -> plt.Figure:
    """Plot genérico para cualquier modelo"""
    fig = plt.figure(figsize=figsize)

    # Mostrar información disponible de forma genérica
    ax = plt.subplot(1, 1, 1)

    info_text = f"Tipo de modelo: {resultado.get('tipo', 'Desconocido')}\n\n"

    if 'metricas' in resultado:
        info_text += "Métricas:\n"
        for split, metrics in resultado['metricas'].items():
            info_text += f"\n{split.upper()}:\n"
            for metric, value in list(metrics.items())[:5]:
                if isinstance(value, float):
                    info_text += f"  {metric}: {value:.4f}\n"

    ax.text(0.1, 0.5, info_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='center')
    ax.axis('off')

    return fig

# ==================== GESTIÓN DE MEMORIA ====================

def limpiar_memoria():
    """Liberar memoria no utilizada"""
    gc.collect()

# ==================== FUNCIONES DE EXPORTACIÓN ====================

def exportar_modelo(modelo, filepath: str, incluir_metadata: bool = True) -> bool:
    """
    Exportar modelo entrenado a archivo
    """
    try:
        import joblib
        import json
        from datetime import datetime

        # Guardar modelo
        joblib.dump(modelo, filepath)

        # Guardar metadata si se solicita
        if incluir_metadata:
            metadata = {
                'fecha_exportacion': datetime.now().isoformat(),
                'tipo_modelo': type(modelo).__name__,
                'parametros': modelo.get_params() if hasattr(modelo, 'get_params') else {}
            }

            metadata_path = filepath.replace('.pkl', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        return True
    except Exception as e:
        logger.error(f"Error exportando modelo: {str(e)}")
        return False

def cargar_modelo(filepath: str) -> Optional[Any]:
    """
    Cargar modelo desde archivo
    """
    try:
        import joblib
        modelo = joblib.load(filepath)
        return modelo
    except Exception as e:
        logger.error(f"Error cargando modelo: {str(e)}")
        return None


def _calculate_skewness(data):
    """Calcular skewness usando solo NumPy"""
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data, ddof=0)

    if std == 0:
        return 0.0

    n = len(data)
    skew = np.sum(((data - mean) / std) ** 3) / n
    return skew


def _calculate_kurtosis(data):
    """Calcular kurtosis usando solo NumPy"""
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data, ddof=0)

    if std == 0:
        return 0.0

    n = len(data)
    kurt = np.sum(((data - mean) / std) ** 4) / n - 3  # -3 para excess kurtosis
    return kurt


def _shapiro_wilk_simple(data):
    """
    Test de normalidad simplificado usando solo NumPy
    """
    try:
        data = np.array(data)
        n = len(data)

        if n < 20 or n > 5000:
            return 0.5

        # Ordenar datos
        x = np.sort(data)
        mean = np.mean(x)
        std = np.std(x, ddof=1)

        if std == 0:
            return 1.0

        # Normalizar
        z = (x - mean) / std

        # Test basado en cuantiles esperados vs observados
        expected = np.linspace(-2.5, 2.5, n)
        correlation = np.corrcoef(expected, z)[0, 1]

        # Convertir correlación a p-value aproximado
        if np.isnan(correlation):
            return 0.5
        elif correlation > 0.98:
            return 0.9
        elif correlation > 0.95:
            return 0.7
        elif correlation > 0.90:
            return 0.3
        elif correlation > 0.85:
            return 0.1
        else:
            return 0.01

    except:
        return 0.01