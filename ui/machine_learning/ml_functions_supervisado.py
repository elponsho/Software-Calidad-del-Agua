"""
ml_functions_supervisado.py - Funciones de Machine Learning Supervisado
Contiene todas las funciones para análisis supervisado
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')


def generar_datos_agua_optimizado(n_muestras=200, seed=42):
    """Generar datos sintéticos de calidad del agua"""
    np.random.seed(seed)
    print(f"🔬 Generando {n_muestras} muestras de datos de calidad del agua...")

    # Generar datos correlacionados más realistas
    datos = {}

    # pH - distribución normal centrada en 7
    datos['pH'] = np.clip(np.random.normal(7.2, 0.8, n_muestras), 6.0, 8.5)

    # Oxígeno disuelto - correlacionado con pH
    base_oxigeno = 8.5 + (datos['pH'] - 7.0) * 0.5
    datos['Oxígeno_Disuelto'] = np.clip(
        base_oxigeno + np.random.normal(0, 1.2, n_muestras),
        4.0, 12.0
    )

    # Turbidez - distribución exponencial (más valores bajos)
    datos['Turbidez'] = np.clip(np.random.exponential(2.0, n_muestras), 0.1, 10.0)

    # Conductividad - correlacionada inversamente con calidad
    base_conductividad = 500 + datos['Turbidez'] * 30
    datos['Conductividad'] = np.clip(
        base_conductividad + np.random.normal(0, 150, n_muestras),
        100, 1200
    )

    # Calcular scores de calidad de manera vectorizada
    print("📊 Calculando scores de calidad...")

    # pH score (25 puntos máximo)
    ph_score = np.where(
        (datos['pH'] >= 6.5) & (datos['pH'] <= 8.5),
        25,
        np.maximum(0, 25 - np.abs(datos['pH'] - 7.0) * 8)
    )

    # Oxígeno score (25 puntos máximo)
    oxigeno_score = np.where(
        datos['Oxígeno_Disuelto'] >= 6,
        25,
        np.maximum(0, datos['Oxígeno_Disuelto'] * 4.17)
    )

    # Turbidez score (25 puntos máximo)
    turbidez_score = np.where(
        datos['Turbidez'] <= 4,
        25,
        np.maximum(0, 25 - (datos['Turbidez'] - 4) * 3.57)
    )

    # Conductividad score (25 puntos máximo)
    conductividad_score = np.where(
        (datos['Conductividad'] >= 200) & (datos['Conductividad'] <= 800),
        25,
        np.maximum(0, 25 - np.abs(datos['Conductividad'] - 500) * 0.042)
    )

    # Score total (0-100)
    calidad_scores = ph_score + oxigeno_score + turbidez_score + conductividad_score

    # Categorías de calidad más específicas
    calidades = np.select(
        [
            calidad_scores >= 85,
            calidad_scores >= 70,
            calidad_scores >= 50,
            calidad_scores >= 30
        ],
        ["Excelente", "Buena", "Regular", "Mala"],
        default="Crítica"
    )

    datos['Calidad_Score'] = calidad_scores
    datos['Calidad'] = calidades

    df = pd.DataFrame(datos)
    print(f"✅ Dataset generado: {len(df)} muestras")
    print(f"📈 Score promedio: {df['Calidad_Score'].mean():.2f}")
    print(f"🏷️ Distribución de calidades: {df['Calidad'].value_counts().to_dict()}")

    return df


def regresion_multiple_proceso(n_muestras=200):
    """Regresión múltiple para predecir score de calidad"""
    try:
        print("🚀 Iniciando análisis de Regresión Múltiple...")

        # Generar datos
        df = generar_datos_agua_optimizado(n_muestras)
        X = df[['pH', 'Oxígeno_Disuelto', 'Turbidez', 'Conductividad']]
        y = df['Calidad_Score']

        print("📊 Dividiendo datos en entrenamiento y prueba...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Crear y entrenar modelo
        print("🔧 Entrenando modelo de regresión múltiple...")
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)

        # Predicciones
        print("🎯 Realizando predicciones...")
        y_pred = modelo.predict(X_test)

        # Métricas
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        print(f"📈 R² Score: {r2:.4f}")
        print(f"📈 MSE: {mse:.4f}")
        print(f"📈 MAE: {mae:.4f}")

        # Coeficientes con interpretación
        coeficientes = []
        for i, col in enumerate(X.columns):
            coef = modelo.coef_[i]
            coeficientes.append({
                'Parámetro': col,
                'Coeficiente': float(coef),
                'Impacto': float(abs(coef)),
                'Interpretación': 'Positivo' if coef > 0 else 'Negativo'
            })

        coeficientes_df = pd.DataFrame(coeficientes).sort_values('Impacto', ascending=False)

        # Ejemplos de predicción con análisis detallado
        print("🔍 Generando ejemplos de predicción...")
        ejemplos = []
        for i in range(min(8, len(X_test))):
            muestra = X_test.iloc[i]
            pred = modelo.predict([muestra])[0]
            real = y_test.iloc[i]
            error = abs(pred - real)
            error_porcentual = (error / real) * 100 if real > 0 else 0

            # Clasificar la predicción
            pred_categoria = np.select(
                [pred >= 85, pred >= 70, pred >= 50, pred >= 30],
                ["Excelente", "Buena", "Regular", "Mala"],
                default="Crítica"
            )

            real_categoria = np.select(
                [real >= 85, real >= 70, real >= 50, real >= 30],
                ["Excelente", "Buena", "Regular", "Mala"],
                default="Crítica"
            )

            ejemplos.append({
                'Muestra': i + 1,
                'pH': float(muestra['pH']),
                'Oxígeno': float(muestra['Oxígeno_Disuelto']),
                'Turbidez': float(muestra['Turbidez']),
                'Conductividad': float(muestra['Conductividad']),
                'Predicho': float(pred),
                'Real': float(real),
                'Error': float(error),
                'Error_%': float(error_porcentual),
                'Pred_Categoria': pred_categoria,
                'Real_Categoria': real_categoria,
                'Acierto_Categoria': pred_categoria == real_categoria
            })

        # Estadísticas adicionales
        residuos = y_test - y_pred

        resultado = {
            'tipo': 'regresion_multiple',
            'algoritmo': 'Regresión Múltiple',
            'n_muestras': int(n_muestras),
            'n_entrenamiento': int(len(X_train)),
            'n_prueba': int(len(X_test)),
            'r2_score': float(r2),
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'coeficientes': coeficientes_df.to_dict('records'),
            'ejemplos': ejemplos,
            'intercepto': float(modelo.intercept_),
            'estadisticas_residuos': {
                'media': float(np.mean(residuos)),
                'std': float(np.std(residuos)),
                'min': float(np.min(residuos)),
                'max': float(np.max(residuos))
            },
            'parametros_importancia': coeficientes_df['Parámetro'].tolist()
        }

        print("✅ Regresión múltiple completada exitosamente")
        return resultado

    except Exception as e:
        print(f"❌ Error en regresión múltiple: {str(e)}")
        return {'error': f"Error en regresión múltiple: {str(e)}"}


def svm_proceso(n_muestras=200):
    """Máquinas de Vectores de Soporte para clasificación"""
    try:
        print("🚀 Iniciando análisis SVM...")

        # Generar datos
        df = generar_datos_agua_optimizado(n_muestras)
        X = df[['pH', 'Oxígeno_Disuelto', 'Turbidez', 'Conductividad']]
        y = df['Calidad']

        print("📊 Dividiendo datos en entrenamiento y prueba...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Estandarizar datos (crucial para SVM)
        print("⚖️ Estandarizando características...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Crear y entrenar SVM
        print("🔧 Entrenando modelo SVM con kernel RBF...")
        modelo = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        modelo.fit(X_train_scaled, y_train)

        # Predicciones
        print("🎯 Realizando predicciones...")
        y_pred = modelo.predict(X_test_scaled)
        y_proba = modelo.predict_proba(X_test_scaled)

        # Métricas detalladas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f"🎯 Precisión: {accuracy:.4f}")
        print(f"📊 Precision: {precision:.4f}")
        print(f"📊 Recall: {recall:.4f}")
        print(f"📊 F1-Score: {f1:.4f}")

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        clases = modelo.classes_

        # Reporte de clasificación detallado
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Ejemplos con probabilidades detalladas
        print("🔍 Generando ejemplos con probabilidades...")
        ejemplos = []
        for i in range(min(8, len(X_test))):
            muestra = X_test.iloc[i]
            muestra_scaled = scaler.transform([muestra])
            pred = modelo.predict(muestra_scaled)[0]
            probas = modelo.predict_proba(muestra_scaled)[0]
            confianza = max(probas) * 100

            # Crear diccionario de probabilidades por clase
            prob_dict = {}
            for j, clase in enumerate(clases):
                prob_dict[f'Prob_{clase}'] = float(probas[j] * 100)

            ejemplo = {
                'Muestra': i + 1,
                'pH': float(muestra['pH']),
                'Oxígeno': float(muestra['Oxígeno_Disuelto']),
                'Turbidez': float(muestra['Turbidez']),
                'Conductividad': float(muestra['Conductividad']),
                'Predicción': pred,
                'Real': y_test.iloc[i],
                'Correcto': pred == y_test.iloc[i],
                'Confianza_%': float(confianza)
            }
            ejemplo.update(prob_dict)
            ejemplos.append(ejemplo)

        # Análisis por clase
        analisis_clases = {}
        for clase in clases:
            if clase in report:
                analisis_clases[clase] = {
                    'precision': float(report[clase]['precision']),
                    'recall': float(report[clase]['recall']),
                    'f1_score': float(report[clase]['f1-score']),
                    'support': int(report[clase]['support'])
                }

        resultado = {
            'tipo': 'svm',
            'algoritmo': 'Support Vector Machine',
            'kernel': 'RBF',
            'n_muestras': int(n_muestras),
            'n_entrenamiento': int(len(X_train)),
            'n_prueba': int(len(X_test)),
            'accuracy': float(accuracy * 100),
            'precision': float(precision * 100),
            'recall': float(recall * 100),
            'f1_score': float(f1 * 100),
            'matriz_confusion': cm.tolist(),
            'clases': clases.tolist(),
            'reporte': report,
            'analisis_clases': analisis_clases,
            'ejemplos': ejemplos,
            'n_vectores_soporte': int(modelo.n_support_.sum()),
            'parametros_modelo': {
                'C': modelo.C,
                'gamma': modelo.gamma,
                'kernel': modelo.kernel
            }
        }

        print("✅ SVM completado exitosamente")
        return resultado

    except Exception as e:
        print(f"❌ Error en SVM: {str(e)}")
        return {'error': f"Error en SVM: {str(e)}"}


def random_forest_proceso(n_muestras=200):
    """Random Forest para clasificación y regresión"""
    try:
        print("🚀 Iniciando análisis Random Forest...")

        # Generar datos
        df = generar_datos_agua_optimizado(n_muestras)
        X = df[['pH', 'Oxígeno_Disuelto', 'Turbidez', 'Conductividad']]

        # Parte 1: Clasificación
        print("🌳 Entrenando Random Forest para clasificación...")
        y_class = df['Calidad']
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X, y_class, test_size=0.3, random_state=42, stratify=y_class
        )

        rf_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        rf_classifier.fit(X_train_c, y_train_c)

        # Métricas de clasificación
        y_pred_c = rf_classifier.predict(X_test_c)
        accuracy_class = accuracy_score(y_test_c, y_pred_c)
        precision_class = precision_score(y_test_c, y_pred_c, average='weighted', zero_division=0)

        # Importancia de características para clasificación
        importancias_class = []
        for i, col in enumerate(X.columns):
            importancias_class.append({
                'Parámetro': col,
                'Importancia': float(rf_classifier.feature_importances_[i]),
                'Porcentaje': float(rf_classifier.feature_importances_[i] * 100)
            })
        importancias_class = sorted(importancias_class, key=lambda x: x['Importancia'], reverse=True)

        # Parte 2: Regresión
        print("🌳 Entrenando Random Forest para regresión...")
        y_reg = df['Calidad_Score']
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X, y_reg, test_size=0.3, random_state=42
        )

        rf_regressor = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        rf_regressor.fit(X_train_r, y_train_r)

        # Métricas de regresión
        y_pred_r = rf_regressor.predict(X_test_r)
        r2_reg = r2_score(y_test_r, y_pred_r)
        mse_reg = mean_squared_error(y_test_r, y_pred_r)
        mae_reg = mean_absolute_error(y_test_r, y_pred_r)

        # Out-of-bag score
        oob_score = rf_regressor.oob_score_ if hasattr(rf_regressor, 'oob_score_') else None

        print(f"🎯 Precisión Clasificación: {accuracy_class:.4f}")
        print(f"📊 R² Regresión: {r2_reg:.4f}")

        # Ejemplos combinados
        print("🔍 Generando ejemplos combinados...")
        ejemplos = []
        n_ejemplos = min(6, len(X_test_c), len(X_test_r))

        for i in range(n_ejemplos):
            muestra_c = X_test_c.iloc[i] if i < len(X_test_c) else X_test_c.iloc[0]
            muestra_r = X_test_r.iloc[i] if i < len(X_test_r) else X_test_r.iloc[0]

            pred_class = rf_classifier.predict([muestra_c])[0]
            pred_score = rf_regressor.predict([muestra_r])[0]

            # Probabilidades para clasificación
            proba_class = rf_classifier.predict_proba([muestra_c])[0]
            max_proba = max(proba_class) * 100

            ejemplos.append({
                'Muestra': i + 1,
                'pH': float(muestra_c['pH']),
                'Oxígeno': float(muestra_c['Oxígeno_Disuelto']),
                'Turbidez': float(muestra_c['Turbidez']),
                'Conductividad': float(muestra_c['Conductividad']),
                'Clase_Predicha': pred_class,
                'Score_Predicho': float(pred_score),
                'Confianza_%': float(max_proba),
                'Clase_Real': y_test_c.iloc[i] if i < len(y_test_c) else 'N/A',
                'Score_Real': float(y_test_r.iloc[i]) if i < len(y_test_r) else 0.0
            })

        # Análisis de árboles
        tree_depths = [tree.tree_.max_depth for tree in rf_classifier.estimators_[:10]]

        resultado = {
            'tipo': 'random_forest',
            'algoritmo': 'Random Forest',
            'n_estimadores': 100,
            'n_muestras': int(n_muestras),

            # Métricas de clasificación
            'accuracy_clasificacion': float(accuracy_class * 100),
            'precision_clasificacion': float(precision_class * 100),

            # Métricas de regresión
            'r2_regresion': float(r2_reg),
            'mse_regresion': float(mse_reg),
            'mae_regresion': float(mae_reg),

            # Importancia de características
            'importancias': importancias_class,

            # Ejemplos
            'ejemplos': ejemplos,

            # Información del modelo
            'oob_score': float(oob_score) if oob_score else None,
            'profundidad_promedio_arboles': float(np.mean(tree_depths)),
            'clases_clasificacion': rf_classifier.classes_.tolist(),

            # Parámetros del modelo
            'parametros': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }
        }

        print("✅ Random Forest completado exitosamente")
        return resultado

    except Exception as e:
        print(f"❌ Error en Random Forest: {str(e)}")
        return {'error': f"Error en Random Forest: {str(e)}"}


def regresion_lineal_proceso(n_muestras=200):
    """Regresión lineal simple, múltiple y regularizada"""
    try:
        print("🚀 Iniciando análisis de Regresión Lineal Completa...")

        # Generar datos
        df = generar_datos_agua_optimizado(n_muestras)
        X = df[['pH', 'Oxígeno_Disuelto', 'Turbidez', 'Conductividad']]
        y = df['Calidad_Score']

        print("📊 Dividiendo datos en entrenamiento y prueba...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # 1. Regresión lineal simple para cada variable
        print("🔍 Analizando regresiones simples...")
        resultados_simples = {}
        for col in X.columns:
            X_simple_train = X_train[[col]]
            X_simple_test = X_test[[col]]

            modelo_simple = LinearRegression()
            modelo_simple.fit(X_simple_train, y_train)

            y_pred_simple = modelo_simple.predict(X_simple_test)
            r2_simple = r2_score(y_test, y_pred_simple)
            mse_simple = mean_squared_error(y_test, y_pred_simple)

            resultados_simples[col] = {
                'r2': float(r2_simple),
                'mse': float(mse_simple),
                'coeficiente': float(modelo_simple.coef_[0]),
                'intercepto': float(modelo_simple.intercept_),
                'clasificacion': 'Fuerte' if r2_simple > 0.7 else 'Moderada' if r2_simple > 0.3 else 'Débil'
            }

        # 2. Regresión múltiple
        print("🔧 Entrenando regresión múltiple...")
        modelo_multiple = LinearRegression()
        modelo_multiple.fit(X_train, y_train)

        y_pred_multiple = modelo_multiple.predict(X_test)
        r2_multiple = r2_score(y_test, y_pred_multiple)
        mse_multiple = mean_squared_error(y_test, y_pred_multiple)
        mae_multiple = mean_absolute_error(y_test, y_pred_multiple)

        # 3. Regresión Ridge (L2)
        print("🔧 Entrenando regresión Ridge...")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        y_pred_ridge = ridge.predict(X_test)
        r2_ridge = r2_score(y_test, y_pred_ridge)
        mse_ridge = mean_squared_error(y_test, y_pred_ridge)

        # 4. Regresión Lasso (L1)
        print("🔧 Entrenando regresión Lasso...")
        lasso = Lasso(alpha=1.0, max_iter=2000)
        lasso.fit(X_train, y_train)
        y_pred_lasso = lasso.predict(X_test)
        r2_lasso = r2_score(y_test, y_pred_lasso)
        mse_lasso = mean_squared_error(y_test, y_pred_lasso)

        # Comparación de métodos
        metodos_comparacion = {
            'Multiple': {'r2': r2_multiple, 'mse': mse_multiple},
            'Ridge': {'r2': r2_ridge, 'mse': mse_ridge},
            'Lasso': {'r2': r2_lasso, 'mse': mse_lasso}
        }

        # Determinar mejor método
        mejor_metodo = max(metodos_comparacion.keys(),
                          key=lambda k: metodos_comparacion[k]['r2'])

        print(f"🏆 Mejor método: {mejor_metodo} (R² = {metodos_comparacion[mejor_metodo]['r2']:.4f})")

        # Coeficientes comparados
        coeficientes_comparacion = []
        for i, col in enumerate(X.columns):
            coef_data = {
                'Parámetro': col,
                'Multiple': float(modelo_multiple.coef_[i]),
                'Ridge': float(ridge.coef_[i]),
                'Lasso': float(lasso.coef_[i])
            }
            coeficientes_comparacion.append(coef_data)

        # Ejemplos de predicción
        print("🔍 Generando ejemplos de predicción...")
        ejemplos = []
        for i in range(min(6, len(X_test))):
            muestra = X_test.iloc[i]
            real = y_test.iloc[i]

            pred_multiple = modelo_multiple.predict([muestra])[0]
            pred_ridge = ridge.predict([muestra])[0]
            pred_lasso = lasso.predict([muestra])[0]

            ejemplos.append({
                'Muestra': i + 1,
                'pH': float(muestra['pH']),
                'Oxígeno': float(muestra['Oxígeno_Disuelto']),
                'Turbidez': float(muestra['Turbidez']),
                'Conductividad': float(muestra['Conductividad']),
                'Real': float(real),
                'Pred_Multiple': float(pred_multiple),
                'Pred_Ridge': float(pred_ridge),
                'Pred_Lasso': float(pred_lasso),
                'Error_Multiple': float(abs(pred_multiple - real)),
                'Error_Ridge': float(abs(pred_ridge - real)),
                'Error_Lasso': float(abs(pred_lasso - real))
            })

        resultado = {
            'tipo': 'regresion_lineal',
            'algoritmo': 'Regresión Lineal Completa',
            'n_muestras': int(n_muestras),
            'n_entrenamiento': int(len(X_train)),
            'n_prueba': int(len(X_test)),

            # Regresiones simples
            'regresiones_simples': resultados_simples,
            'mejor_variable_simple': max(resultados_simples.keys(),
                                       key=lambda k: resultados_simples[k]['r2']),

            # Regresión múltiple
            'r2_multiple': float(r2_multiple),
            'mse_multiple': float(mse_multiple),
            'mae_multiple': float(mae_multiple),

            # Regresión Ridge
            'r2_ridge': float(r2_ridge),
            'mse_ridge': float(mse_ridge),

            # Regresión Lasso
            'r2_lasso': float(r2_lasso),
            'mse_lasso': float(mse_lasso),

            # Comparación
            'mejor_metodo': mejor_metodo,
            'metodos_comparacion': metodos_comparacion,
            'coeficientes_comparacion': coeficientes_comparacion,

            # Coeficientes individuales
            'coeficientes_multiple': [float(c) for c in modelo_multiple.coef_],
            'intercepto_multiple': float(modelo_multiple.intercept_),
            'coeficientes_ridge': [float(c) for c in ridge.coef_],
            'intercepto_ridge': float(ridge.intercept_),
            'coeficientes_lasso': [float(c) for c in lasso.coef_],
            'intercepto_lasso': float(lasso.intercept_),

            # Ejemplos
            'ejemplos': ejemplos,

            # Parámetros de regularización
            'alpha_ridge': ridge.alpha,
            'alpha_lasso': lasso.alpha,

            # Análisis de regularización
            'regularizacion_necesaria': r2_ridge > r2_multiple or r2_lasso > r2_multiple,
            'recomendacion': 'Ridge' if r2_ridge >= max(r2_multiple, r2_lasso) else 'Lasso' if r2_lasso >= r2_multiple else 'Multiple'
        }

        print("✅ Regresión lineal completa exitosamente")
        return resultado

    except Exception as e:
        print(f"❌ Error en regresión lineal: {str(e)}")
        return {'error': f"Error en regresión lineal: {str(e)}"}


# Función de prueba para verificar que todo funciona
def test_all_functions():
    """Función para probar todas las funciones ML"""
    print("🧪 Iniciando pruebas de todas las funciones ML...")

    functions_to_test = [
        ("Regresión Múltiple", regresion_multiple_proceso),
        ("SVM", svm_proceso),
        ("Random Forest", random_forest_proceso),
        ("Regresión Lineal", regresion_lineal_proceso)
    ]

    results = {}

    for name, func in functions_to_test:
        print(f"\n🔬 Probando {name}...")
        try:
            result = func(100)  # Usar menos muestras para prueba rápida
            if 'error' in result:
                print(f"❌ {name} falló: {result['error']}")
                results[name] = False
            else:
                print(f"✅ {name} funcionó correctamente")
                results[name] = True
        except Exception as e:
            print(f"❌ {name} falló con excepción: {str(e)}")
            results[name] = False

    print(f"\n📊 Resumen de pruebas:")
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {name}")

    return results


if __name__ == "__main__":
    print("🚀 Ejecutando pruebas de funciones ML...")
    test_results = test_all_functions()

    if all(test_results.values()):
        print("\n🎉 ¡Todas las funciones ML funcionan correctamente!")
    else:
        print("\n⚠️ Algunas funciones presentaron errores.")

    print("\n📋 Las funciones están listas para ser utilizadas por supervisado_window.py")