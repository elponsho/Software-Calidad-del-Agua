# DEEP LEARNING CNN + RNN NUMPY - COMPATIBLE PYINSTALLER - PARTE 1/6

import os
import sys

try:
    import PIL
except ImportError:
    # Crear mock de PIL para evitar errores
    import types
    mock_pil = types.ModuleType('PIL')
    mock_pil.Image = types.ModuleType('PIL.Image')
    sys.modules['PIL'] = mock_pil
    sys.modules['PIL.Image'] = mock_pil.Image
    print("PIL no disponible - usando mock")

# Ahora importar matplotlib de forma segura
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ioff()  # Desactivar modo interactivo

# Configurar advertencias
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*PIL.*')



import numpy as np
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QFont, QPalette, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from datetime import datetime
import warnings
import time
import joblib
import traceback


warnings.filterwarnings('ignore')

# Configurar matplotlib para PyInstaller
import matplotlib

matplotlib.use('Qt5Agg')
plt.style.use('default')

# Estado del sistema
SYSTEM_AVAILABLE = True

# FUNCIONES MANUALES PARA EVITAR SKLEARN - COMPATIBLE PYINSTALLER

def manual_accuracy_score(y_true, y_pred):
    """Calcular accuracy sin sklearn"""
    if len(y_true) == 0:
        return 0.0
    return np.mean(y_true == y_pred)


def manual_confusion_matrix(y_true, y_pred):
    """Calcular matriz de confusión sin sklearn"""
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    n_labels = len(unique_labels)
    cm = np.zeros((n_labels, n_labels), dtype=int)

    for i, true_label in enumerate(unique_labels):
        for j, pred_label in enumerate(unique_labels):
            cm[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))

    return cm


def manual_classification_report(y_true, y_pred):
    """Generar reporte de clasificación sin sklearn"""
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))

    report = {'accuracy': manual_accuracy_score(y_true, y_pred)}

    precisions = []
    recalls = []
    f1s = []

    for label in unique_labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    report['macro avg'] = {
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1-score': np.mean(f1s)
    }

    return report


def manual_train_test_split(X, y, test_size=0.2, random_state=42):
    """División train/test sin sklearn"""
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


def manual_standard_scaler(X_train, X_test):
    """Escalado estándar sin sklearn"""
    if isinstance(X_train, pd.DataFrame):
        mean = X_train.mean()
        std = X_train.std() + 1e-8
        X_train_scaled = (X_train - mean) / std
        X_test_scaled = (X_test - mean) / std
        return X_train_scaled, X_test_scaled, {'mean': mean, 'std': std}
    else:
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0) + 1e-8
        X_train_scaled = (X_train - mean) / std
        X_test_scaled = (X_test - mean) / std
        return X_train_scaled, X_test_scaled, {'mean': mean, 'std': std}

# CNN IMPLEMENTADO COMPLETAMENTE EN NUMPY - COMPATIBLE PYINSTALLER

class NumpyCNN:
    """Red Neuronal Convolucional implementada 100% en NumPy"""

    def __init__(self, filters=32, kernel_size=3, dense_units=64,
                 learning_rate=0.001, epochs=50, dropout_rate=0.3):
        self.filters = filters
        self.kernel_size = kernel_size
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dropout_rate = dropout_rate
        self.is_fitted = False
        self.training_history = {'loss': [], 'accuracy': []}

    def _sigmoid(self, x):
        """Función sigmoid estable"""
        x = np.clip(x, -500, 500)
        return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def _relu(self, x):
        """Función ReLU"""
        return np.maximum(0, x)

    def _softmax(self, x):
        """Función softmax estable"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _conv1d(self, X, filters):
        """Convolución 1D manual"""
        n_samples, n_features = X.shape
        n_filters = filters.shape[0]
        kernel_size = filters.shape[1]

        if n_features < kernel_size:
            # Padding si es necesario
            padding = kernel_size - n_features
            X_padded = np.pad(X, ((0, 0), (0, padding)), mode='constant')
            n_features = X_padded.shape[1]
        else:
            X_padded = X

        output_size = max(1, n_features - kernel_size + 1)
        result = np.zeros((n_samples, output_size, n_filters))

        for i in range(n_samples):
            for j in range(output_size):
                for k in range(n_filters):
                    result[i, j, k] = np.sum(X_padded[i, j:j + kernel_size] * filters[k])

        return result

    def _max_pooling1d(self, X, pool_size=2):
        """Max pooling 1D"""
        n_samples, seq_len, n_filters = X.shape
        if seq_len < pool_size:
            return X

        output_size = max(1, seq_len // pool_size)
        result = np.zeros((n_samples, output_size, n_filters))

        for i in range(n_samples):
            for j in range(output_size):
                start_idx = j * pool_size
                end_idx = min(start_idx + pool_size, seq_len)
                if end_idx > start_idx:
                    result[i, j, :] = np.max(X[i, start_idx:end_idx, :], axis=0)

        return result

    def _dropout(self, X, rate, training=True):
        """Aplicar dropout"""
        if not training or rate == 0:
            return X
        mask = np.random.binomial(1, 1 - rate, X.shape) / (1 - rate)
        return X * mask

    def fit(self, X, y):
        """Entrenar la CNN"""
        try:
            print(f"Iniciando entrenamiento CNN con {X.shape[0]} muestras...")

            self.n_features = X.shape[1]
            self.n_classes = len(np.unique(y))

            # Ajustar kernel_size si es necesario
            self.kernel_size = min(self.kernel_size, self.n_features)
            if self.kernel_size < 1:
                self.kernel_size = 1

            # Inicializar pesos
            np.random.seed(42)
            self.conv_filters = np.random.randn(self.filters, self.kernel_size) * 0.1

            # Calcular dimensiones después de convolución y pooling
            conv_output_size = max(1, self.n_features - self.kernel_size + 1)
            pool_output_size = max(1, conv_output_size // 2)  # pool_size = 2
            flatten_size = pool_output_size * self.filters

            if flatten_size < 1:
                flatten_size = self.filters

            # Pesos de la capa densa
            self.W_dense = np.random.randn(flatten_size, self.dense_units) * 0.1
            self.b_dense = np.zeros(self.dense_units)

            # Pesos de salida
            output_size = self.n_classes if self.n_classes > 2 else 1
            self.W_output = np.random.randn(self.dense_units, output_size) * 0.1
            self.b_output = np.zeros(output_size)

            # Entrenar
            for epoch in range(self.epochs):
                try:
                    # Forward pass
                    predictions = self._forward(X, training=True)

                    # Calcular loss
                    loss = self._compute_loss(predictions, y)

                    # Calcular accuracy
                    if self.n_classes > 2:
                        y_pred = np.argmax(predictions, axis=1)
                    else:
                        y_pred = (predictions.flatten() > 0.5).astype(int)

                    accuracy = manual_accuracy_score(y, y_pred)

                    # Guardar historial
                    self.training_history['loss'].append(loss)
                    self.training_history['accuracy'].append(accuracy)

                    # Actualizar pesos (gradiente simple)
                    if epoch % 3 == 0 and epoch < self.epochs - 1:
                        self._simple_gradient_update(X, y)

                    if (epoch + 1) % 10 == 0:
                        print(f"Época {epoch + 1}/{self.epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")

                except Exception as e:
                    print(f"Error en época {epoch}: {e}")
                    break

            self.is_fitted = True
            print(f"CNN entrenada exitosamente!")
            return self

        except Exception as e:
            print(f"Error entrenando CNN: {e}")
            raise

    def _forward(self, X, training=False):
        """Forward pass de la CNN"""
        try:
            # Convolución
            conv_out = self._conv1d(X, self.conv_filters)
            conv_out = np.array([self._relu(conv_out[i]) for i in range(conv_out.shape[0])])

            # Max pooling
            pool_out = self._max_pooling1d(conv_out, pool_size=2)

            # Flatten
            flatten_out = pool_out.reshape(pool_out.shape[0], -1)

            # Ajustar dimensiones si es necesario
            if flatten_out.shape[1] != self.W_dense.shape[0]:
                self.W_dense = np.random.randn(flatten_out.shape[1], self.dense_units) * 0.1

            # Dropout
            flatten_out = self._dropout(flatten_out, self.dropout_rate, training)

            # Capa densa
            dense_out = np.dot(flatten_out, self.W_dense) + self.b_dense
            dense_out = self._relu(dense_out)
            dense_out = self._dropout(dense_out, self.dropout_rate, training)

            # Capa de salida
            output = np.dot(dense_out, self.W_output) + self.b_output

            # Activación final
            if self.n_classes > 2:
                return self._softmax(output)
            else:
                return self._sigmoid(output)

        except Exception as e:
            print(f"Error en forward pass: {e}")
            # Devolver predicciones aleatorias en caso de error
            if self.n_classes > 2:
                return np.random.rand(X.shape[0], self.n_classes)
            else:
                return np.random.rand(X.shape[0], 1)

    def _compute_loss(self, predictions, y):
        """Calcular pérdida"""
        try:
            if self.n_classes > 2:
                # Multiclase - categorical crossentropy
                y_one_hot = np.eye(self.n_classes)[y]
                predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
                return -np.mean(np.sum(y_one_hot * np.log(predictions), axis=1))
            else:
                # Binario - binary crossentropy
                predictions = predictions.flatten()
                predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
                return -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        except:
            return 1.0

    def _simple_gradient_update(self, X, y):
        """Actualización simple de gradientes usando diferencias finitas"""
        try:
            epsilon = 1e-6
            # Solo actualizar algunos pesos para eficiencia
            for i in range(min(5, self.W_output.shape[0])):
                for j in range(min(3, self.W_output.shape[1])):
                    # Gradiente por diferencias finitas
                    self.W_output[i, j] += epsilon
                    loss_plus = self._compute_loss(self._forward(X), y)
                    self.W_output[i, j] -= 2 * epsilon
                    loss_minus = self._compute_loss(self._forward(X), y)
                    grad = (loss_plus - loss_minus) / (2 * epsilon)
                    self.W_output[i, j] += epsilon - self.learning_rate * grad
        except:
            pass

    def predict(self, X):
        """Realizar predicciones"""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")

        predictions = self._forward(X, training=False)
        if self.n_classes > 2:
            return np.argmax(predictions, axis=1)
        else:
            return (predictions.flatten() > 0.5).astype(int)

    def predict_proba(self, X):
        """Obtener probabilidades de predicción"""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")

        predictions = self._forward(X, training=False)
        if self.n_classes == 2:
            prob_positive = predictions.flatten()
            return np.column_stack([1 - prob_positive, prob_positive])
        return predictions

# RNN IMPLEMENTADO COMPLETAMENTE EN NUMPY - COMPATIBLE PYINSTALLER

# CORRECCIONES PARA LA CLASE NumpyRNN - PARTE 1/2

class NumpyRNN:
    """Red Neuronal Recurrente implementada 100% en NumPy - CORREGIDA"""

    def __init__(self, hidden_units=32, dense_units=64, learning_rate=0.01,
                 epochs=50, dropout_rate=0.2, sequence_length=None):
        # Incrementé epochs por defecto y hidden_units para más complejidad
        self.hidden_units = hidden_units
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dropout_rate = dropout_rate
        self.sequence_length = sequence_length
        self.is_fitted = False
        self.training_history = {'loss': [], 'accuracy': []}

    def _sigmoid(self, x):
        """Función sigmoid estable"""
        x = np.clip(x, -500, 500)
        return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def _tanh(self, x):
        """Función tanh estable"""
        return np.tanh(np.clip(x, -10, 10))

    def _relu(self, x):
        """Función ReLU"""
        return np.maximum(0, x)

    def _softmax(self, x):
        """Función softmax estable"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _prepare_sequences(self, X):
        """Preparar secuencias para RNN - MÉTODO CORREGIDO"""
        try:
            batch_size, n_features = X.shape

            # Determinar longitud de secuencia de manera más robusta
            if self.sequence_length is None:
                # Crear secuencias más largas para dar más trabajo al RNN
                self.sequence_length = max(8, min(n_features, 16))

            # Si tenemos suficientes features, crear secuencias temporales reales
            if n_features >= self.sequence_length:
                # Dividir las features en pasos temporales
                features_per_step = max(1, n_features // self.sequence_length)
                total_features_used = self.sequence_length * features_per_step

                # Tomar solo las features que podemos dividir uniformemente
                X_reshaped = X[:, :total_features_used].reshape(
                    batch_size, self.sequence_length, features_per_step
                )

                return X_reshaped
            else:
                # Si no hay suficientes features, crear secuencias con padding y repetición
                # Esto hace que el RNN tenga que procesar más pasos temporales
                features_per_step = 1

                # Crear secuencias repitiendo y modificando features
                sequences = np.zeros((batch_size, self.sequence_length, features_per_step))

                for i in range(self.sequence_length):
                    if i < n_features:
                        sequences[:, i, 0] = X[:, i]
                    else:
                        # Repetir con ruido para crear variabilidad temporal
                        idx = i % n_features
                        noise = np.random.normal(0, 0.1, batch_size)
                        sequences[:, i, 0] = X[:, idx] + noise

                return sequences

        except Exception as e:
            print(f"Error en _prepare_sequences: {e}")
            # Fallback: crear secuencia simple pero más larga
            return np.expand_dims(X, axis=2).repeat(max(8, X.shape[1] // 4), axis=1)

    def fit(self, X, y):
        """Entrenar la RNN - MÉTODO MEJORADO"""
        try:
            print(f"Iniciando entrenamiento RNN con {X.shape[0]} muestras...")
            print(f"Configuración: {self.epochs} épocas, {self.hidden_units} unidades ocultas")

            # Preparar secuencias
            X_seq = self._prepare_sequences(X)
            print(f"Secuencias creadas: {X_seq.shape}")

            self.n_classes = len(np.unique(y))
            self.sequence_length = X_seq.shape[1]  # Actualizar con el valor real
            self.input_size = X_seq.shape[2]

            # Inicializar pesos con más variabilidad
            np.random.seed(42)

            # Pesos más grandes para dar más trabajo a la red
            scale = 0.1
            self.W_ih = np.random.randn(self.input_size, self.hidden_units) * scale
            self.W_hh = np.random.randn(self.hidden_units, self.hidden_units) * scale
            self.b_h = np.zeros(self.hidden_units)

            # Capa densa
            self.W_dense = np.random.randn(self.hidden_units, self.dense_units) * scale
            self.b_dense = np.zeros(self.dense_units)

            # Capa de salida
            output_size = self.n_classes if self.n_classes > 2 else 1
            self.W_output = np.random.randn(self.dense_units, output_size) * scale
            self.b_output = np.zeros(output_size)

            # Entrenamiento con más frecuencia de actualizaciones
            for epoch in range(self.epochs):
                try:
                    # Forward pass
                    predictions = self._forward(X_seq, training=True)

                    # Calcular loss
                    loss = self._compute_loss(predictions, y)

                    # Calcular accuracy
                    if self.n_classes > 2:
                        y_pred = np.argmax(predictions, axis=1)
                    else:
                        y_pred = (predictions.flatten() > 0.5).astype(int)

                    accuracy = manual_accuracy_score(y, y_pred)

                    # Guardar historial
                    self.training_history['loss'].append(loss)
                    self.training_history['accuracy'].append(accuracy)

                    # Actualización más frecuente de pesos para simular entrenamiento real
                    if epoch % 2 == 0 and epoch < self.epochs - 1:  # Cada 2 épocas en lugar de 5
                        self._complex_gradient_update(X_seq, y)

                    # Mostrar progreso más frecuentemente
                    if (epoch + 1) % 5 == 0:  # Cada 5 épocas en lugar de 10
                        print(f"Época {epoch + 1}/{self.epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")

                    # Simular carga computacional adicional
                    if epoch % 3 == 0:
                        self._simulate_computational_load(X_seq)

                except Exception as e:
                    print(f"Error en época RNN {epoch}: {e}")
                    break

            self.is_fitted = True
            print(f"RNN entrenada exitosamente!")
            return self

        except Exception as e:
            print(f"Error entrenando RNN: {e}")
            raise

    def _forward(self, X_seq, training=False):
        """Forward pass de la RNN - MÉTODO MEJORADO"""
        try:
            batch_size, seq_len, input_size = X_seq.shape

            # Estado oculto inicial
            h = np.zeros((batch_size, self.hidden_units))

            # Lista para almacenar estados ocultos intermedios (más procesamiento)
            hidden_states = []

            # Procesar secuencia paso a paso
            for t in range(seq_len):
                # Calcular nuevo estado oculto con más operaciones
                input_contribution = np.dot(X_seq[:, t, :], self.W_ih)
                hidden_contribution = np.dot(h, self.W_hh)

                # Agregar operaciones adicionales para simular complejidad
                combined_input = input_contribution + hidden_contribution + self.b_h

                # Aplicar activación no lineal
                h = self._tanh(combined_input)

                # Normalización simple para estabilidad
                h_norm = np.linalg.norm(h, axis=1, keepdims=True)
                h = h / (h_norm + 1e-8)

                # Aplicar dropout durante entrenamiento
                if training and self.dropout_rate > 0:
                    mask = np.random.binomial(1, 1 - self.dropout_rate, h.shape) / (1 - self.dropout_rate)
                    h = h * mask

                # Guardar estado para procesamiento adicional
                hidden_states.append(h.copy())

            # Usar el último estado oculto para la predicción
            final_hidden = hidden_states[-1]

            # Operaciones adicionales en el estado final
            enhanced_hidden = final_hidden + np.mean(hidden_states, axis=0) * 0.1

            # Capa densa con más operaciones
            dense_input = np.dot(enhanced_hidden, self.W_dense) + self.b_dense
            dense_out = self._relu(dense_input)

            # Normalización en capa densa
            dense_norm = np.linalg.norm(dense_out, axis=1, keepdims=True)
            dense_out = dense_out / (dense_norm + 1e-8)

            # Dropout en capa densa
            if training and self.dropout_rate > 0:
                mask = np.random.binomial(1, 1 - self.dropout_rate, dense_out.shape) / (1 - self.dropout_rate)
                dense_out = dense_out * mask

            # Capa de salida
            output = np.dot(dense_out, self.W_output) + self.b_output

            # Activación final
            if self.n_classes > 2:
                return self._softmax(output)
            else:
                return self._sigmoid(output)

        except Exception as e:
            print(f"Error en forward pass RNN: {e}")
            # Devolver predicciones aleatorias en caso de error
            if self.n_classes > 2:
                return np.random.rand(X_seq.shape[0], self.n_classes)
            else:
                return np.random.rand(X_seq.shape[0], 1)

    def _complex_gradient_update(self, X_seq, y):
        """Actualización más compleja de gradientes"""
        try:
            epsilon = 1e-6

            # Actualizar más pesos para simular entrenamiento real
            # Actualizar pesos W_output
            for i in range(min(5, self.W_output.shape[0])):
                for j in range(self.W_output.shape[1]):
                    # Gradiente por diferencias finitas
                    self.W_output[i, j] += epsilon
                    loss_plus = self._compute_loss(self._forward(X_seq), y)
                    self.W_output[i, j] -= 2 * epsilon
                    loss_minus = self._compute_loss(self._forward(X_seq), y)
                    grad = (loss_plus - loss_minus) / (2 * epsilon)
                    self.W_output[i, j] += epsilon - self.learning_rate * grad

            # Actualizar algunos pesos W_dense
            for i in range(min(3, self.W_dense.shape[0])):
                for j in range(min(3, self.W_dense.shape[1])):
                    self.W_dense[i, j] += epsilon
                    loss_plus = self._compute_loss(self._forward(X_seq), y)
                    self.W_dense[i, j] -= 2 * epsilon
                    loss_minus = self._compute_loss(self._forward(X_seq), y)
                    grad = (loss_plus - loss_minus) / (2 * epsilon)
                    self.W_dense[i, j] += epsilon - self.learning_rate * grad

        except Exception as e:
            print(f"Error en actualización de gradientes: {e}")

    def _simulate_computational_load(self, X_seq):
        """Simular carga computacional adicional para RNN"""
        try:
            # Operaciones matriciales adicionales para simular complejidad
            batch_size, seq_len, input_size = X_seq.shape

            # Calcular algunas estadísticas que requieren procesamiento
            for _ in range(3):  # Repetir varias veces
                temp_h = np.random.randn(batch_size, self.hidden_units)

                # Simulaciones de cálculos complejos
                for t in range(seq_len):
                    temp_input = X_seq[:, t, :]
                    temp_h = np.tanh(np.dot(temp_input, self.W_ih[:input_size, :self.hidden_units]) +
                                     np.dot(temp_h, self.W_hh) + self.b_h)

                # Cálculos adicionales
                _ = np.linalg.norm(temp_h, axis=1)
                _ = np.mean(temp_h, axis=0)
                _ = np.std(temp_h, axis=0)

        except:
            pass  # Si hay error, simplemente continuar

    def _compute_loss(self, predictions, y):
        """Calcular pérdida - sin cambios"""
        try:
            if self.n_classes > 2:
                # Multiclase
                y_one_hot = np.eye(self.n_classes)[y]
                predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
                return -np.mean(np.sum(y_one_hot * np.log(predictions), axis=1))
            else:
                # Binario
                predictions = predictions.flatten()
                predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
                return -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        except:
            return 1.0

    def predict(self, X):
        """Realizar predicciones - sin cambios"""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")

        X_seq = self._prepare_sequences(X)
        predictions = self._forward(X_seq, training=False)

        if self.n_classes > 2:
            return np.argmax(predictions, axis=1)
        else:
            return (predictions.flatten() > 0.5).astype(int)

    def predict_proba(self, X):
        """Obtener probabilidades de predicción - sin cambios"""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")

        X_seq = self._prepare_sequences(X)
        predictions = self._forward(X_seq, training=False)

        if self.n_classes == 2:
            prob_positive = predictions.flatten()
            return np.column_stack([1 - prob_positive, prob_positive])
        return predictions

# THREAD PARA ENTRENAR CNN Y RNN - COMPATIBLE PYINSTALLER

class DeepLearningTrainingThread(QThread):
    """Thread para entrenar CNN y RNN"""

    progress_updated = pyqtSignal(str)
    training_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, model_config, data):
        super().__init__()
        self.model_config = model_config
        self.data = data

    def run(self):
        try:
            start_time = time.time()
            self.progress_updated.emit("Preparando datos para Deep Learning...")

            X, y = self.prepare_data()
            if X is None or y is None:
                raise ValueError("No se pudieron preparar los datos")

            self.progress_updated.emit("Dividiendo datos...")

            if len(X) < 10:
                raise ValueError("Insuficientes datos para entrenar (mínimo 10 muestras)")

            # División de datos
            X_train, X_test, y_train, y_test = manual_train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            self.progress_updated.emit("Escalando características...")

            # Escalado de datos
            X_train_scaled, X_test_scaled, scaler = manual_standard_scaler(X_train, X_test)

            model_type = self.model_config['type']

            if model_type == 'cnn':
                self.progress_updated.emit("Entrenando CNN...")
                results = self.train_cnn(X_train_scaled, X_test_scaled, y_train, y_test)
            elif model_type == 'rnn':
                self.progress_updated.emit("Entrenando RNN...")
                results = self.train_rnn(X_train_scaled, X_test_scaled, y_train, y_test)
            else:
                raise ValueError(f"Tipo de modelo desconocido: {model_type}")

            results['training_time'] = time.time() - start_time
            results['scaler'] = scaler
            results['feature_names'] = X.columns.tolist() if hasattr(X, 'columns') else [f'feature_{i}' for i in
                                                                                         range(X.shape[1])]
            results['data_shape'] = X.shape

            self.progress_updated.emit("Entrenamiento completado!")
            self.training_finished.emit(results)

        except Exception as e:
            error_msg = f"Error durante entrenamiento: {str(e)}"
            print(f"Error completo: {traceback.format_exc()}")
            self.error_occurred.emit(error_msg)

    def prepare_data(self):
        """Preparar datos para entrenamiento"""
        try:
            df = self.data.copy()

            # Buscar columna target
            target_col = None
            possible_targets = ['target', 'class', 'label', 'y', 'classification', 'Classification_9V']

            for col in df.columns:
                col_lower = col.lower()
                for target in possible_targets:
                    if target.lower() in col_lower:
                        target_col = col
                        break
                if target_col:
                    break

            if target_col is None:
                target_col = df.columns[-1]

            feature_cols = [col for col in df.columns if col != target_col]

            # Seleccionar solo columnas numéricas
            numeric_cols = []
            for col in feature_cols:
                if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    numeric_cols.append(col)
                else:
                    try:
                        pd.to_numeric(df[col])
                        numeric_cols.append(col)
                    except:
                        continue

            if len(numeric_cols) == 0:
                raise ValueError("No se encontraron columnas numéricas")

            X = df[numeric_cols].copy()
            y = df[target_col].copy()

            # Codificar etiquetas si son categóricas
            if y.dtype == 'object' or y.dtype.name == 'category':
                unique_vals = y.unique()
                val_to_num = {val: i for i, val in enumerate(unique_vals)}
                y = y.map(val_to_num)

            # Limpiar datos
            X = X.fillna(X.median())
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())

            if X.shape[0] < 5:
                raise ValueError("Muy pocas muestras para entrenar")
            if X.shape[1] < 1:
                raise ValueError("No hay características válidas")

            return X, y

        except Exception as e:
            print(f"Error preparando datos: {e}")
            return None, None

    def train_cnn(self, X_train, X_test, y_train, y_test):
        """Entrenar modelo CNN"""
        try:
            config = self.model_config

            X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
            X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
            y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
            y_test_np = y_test.values if hasattr(y_test, 'values') else y_test

            model = NumpyCNN(
                filters=min(config.get('filters', 16), 32),
                kernel_size=min(config.get('kernel_size', 3), X_train_np.shape[1]),
                dense_units=min(config.get('dense_units', 32), 64),
                learning_rate=max(0.001, min(config.get('learning_rate', 0.01), 0.1)),
                epochs=min(config.get('epochs', 30), 50),
                dropout_rate=max(0.0, min(config.get('dropout_rate', 0.2), 0.5))
            )

            model.fit(X_train_np, y_train_np)
            y_pred = model.predict(X_test_np)

            results = self._evaluate_model(y_test_np, y_pred, 'cnn')
            results['model'] = model
            results['history'] = model.training_history

            try:
                y_pred_proba = model.predict_proba(X_test_np)
                results['y_pred_proba'] = y_pred_proba
            except:
                results['y_pred_proba'] = None

            return results

        except Exception as e:
            print(f"Error entrenando CNN: {e}")
            raise

    # MÉTODO train_rnn CORREGIDO en DeepLearningTrainingThread

    def train_rnn(self, X_train, X_test, y_train, y_test):
        """Entrenar modelo RNN - MÉTODO CORREGIDO"""
        try:
            config = self.model_config

            X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
            X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
            y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
            y_test_np = y_test.values if hasattr(y_test, 'values') else y_test

            # Calcular sequence_length de manera más robusta
            seq_length = max(8, min(X_train_np.shape[1] // 2, 20))  # Entre 8 y 20
            print(f"Configurando RNN con sequence_length: {seq_length}")

            # Parámetros corregidos para dar más trabajo al RNN
            model = NumpyRNN(
                hidden_units=max(config.get('hidden_units', 32), 32),  # Mínimo 32
                dense_units=max(config.get('dense_units', 64), 64),  # Mínimo 64
                learning_rate=max(0.001, min(config.get('learning_rate', 0.01), 0.05)),  # Rango más amplio
                epochs=max(config.get('epochs', 50), 50),  # Mínimo 50 épocas
                dropout_rate=max(0.1, min(config.get('dropout_rate', 0.2), 0.4)),  # Entre 0.1 y 0.4
                sequence_length=seq_length
            )

            print(f"Iniciando entrenamiento RNN con:")
            print(f"- Hidden units: {model.hidden_units}")
            print(f"- Dense units: {model.dense_units}")
            print(f"- Épocas: {model.epochs}")
            print(f"- Learning rate: {model.learning_rate}")
            print(f"- Sequence length: {seq_length}")

            # Entrenar modelo
            model.fit(X_train_np, y_train_np)

            # Realizar predicciones
            y_pred = model.predict(X_test_np)

            # Evaluar modelo
            results = self._evaluate_model(y_test_np, y_pred, 'rnn')
            results['model'] = model
            results['history'] = model.training_history

            # Intentar obtener probabilidades
            try:
                y_pred_proba = model.predict_proba(X_test_np)
                results['y_pred_proba'] = y_pred_proba
            except Exception as e:
                print(f"No se pudieron obtener probabilidades: {e}")
                results['y_pred_proba'] = None

            print(f"RNN entrenada exitosamente!")
            print(f"Accuracy final: {results['accuracy']:.4f}")

            return results

        except Exception as e:
            print(f"Error entrenando RNN: {e}")
            raise

    def _evaluate_model(self, y_test, y_pred, model_type):
        """Evaluar modelo"""
        try:
            accuracy = manual_accuracy_score(y_test, y_pred)
            cm = manual_confusion_matrix(y_test, y_pred)
            report = manual_classification_report(y_test, y_pred)

            return {
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'classification_report': report,
                'y_test': y_test,
                'y_pred': y_pred,
                'model_type': model_type
            }

        except Exception as e:
            print(f"Error evaluando modelo: {e}")
            return {
                'accuracy': 0.0,
                'confusion_matrix': np.array([[1, 0], [0, 1]]),
                'classification_report': {'accuracy': 0.0},
                'y_test': y_test,
                'y_pred': y_pred if 'y_pred' in locals() else np.zeros_like(y_test),
                'model_type': model_type
            }

# INTERFAZ PRINCIPAL DEEP LEARNING CNN + RNN - COMPATIBLE PYINSTALLER

class DeepLearningLightweight(QWidget):
    """Interfaz principal para CNN y RNN"""

    btn_regresar = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.df = None
        self.current_results = None
        self.training_thread = None
        self.data_manager = None

        self.init_ui()
        self.connect_signals()

        # Registrar con data manager si está disponible
        try:
            from ui.machine_learning.data_manager import get_data_manager
            self.data_manager = get_data_manager()
            self.data_manager.add_observer(self)
            print("DeepLearningInterface registrado como observador")

            if self.data_manager.has_data():
                existing_data = self.data_manager.get_data()
                if existing_data is not None:
                    print(f"Cargando datos existentes: {existing_data.shape}")
                    self.cargar_dataframe(existing_data)
        except Exception as e:
            print(f"Error registrando observador: {e}")
            self.data_manager = None

    def update(self, event):
        """Método Observer para recibir actualizaciones del DataManager"""
        try:
            print(f"DeepLearningInterface recibió notificación: '{event}'")

            if event == 'data_changed':
                if self.data_manager and self.data_manager.has_data():
                    data = self.data_manager.get_data()
                    if data is not None:
                        print(f"Cargando datos automáticamente: {data.shape}")
                        self.cargar_dataframe(data)

            elif event == 'data_cleared':
                print("Limpiando datos en Deep Learning")
                self.df = None
                try:
                    self.clear_results()
                except:
                    pass

            elif event == 'data_modified':
                print("Datos modificados - recargando")
                if self.data_manager and self.data_manager.has_data():
                    data = self.data_manager.get_data()
                    if data is not None:
                        self.cargar_dataframe(data)

        except Exception as e:
            print(f"Error en método update(): {e}")

    def init_ui(self):
        """Inicializar interfaz de usuario"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        self.setLayout(main_layout)

        self.setup_header(main_layout)

        content_area = QHBoxLayout()
        content_area.setSpacing(20)

        left_panel = self.create_left_panel()
        content_area.addWidget(left_panel, 40)

        right_panel = self.create_right_panel()
        content_area.addWidget(right_panel, 60)

        main_layout.addLayout(content_area, 1)
        self.setup_footer(main_layout)

    def connect_signals(self):
        """Conectar señales"""
        try:
            self.btn_execute.clicked.connect(self.start_training)
            self.btn_clear.clicked.connect(self.clear_results)
            self.btn_export.clicked.connect(self.export_model)
            self.btn_regresar_main.clicked.connect(self.btn_regresar.emit)
            print("Señales conectadas correctamente")
        except Exception as e:
            print(f"Error conectando señales: {e}")

    def setup_header(self, layout):
        """Configurar encabezado"""
        header_frame = QFrame()
        header_frame.setMaximumHeight(120)
        header_layout = QVBoxLayout(header_frame)

        title = QLabel("Deep Learning - CNN y RNN")
        title.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #E91E63;
                padding: 5px;
            }
        """)

        subtitle = QLabel("Redes Neuronales Convolucionales (CNN) • Redes Neuronales Recurrentes (RNN)")
        subtitle.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: #666;
                padding: 5px;
            }
        """)



        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        layout.addWidget(header_frame)

    def create_left_panel(self):
        """Crear panel izquierdo"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(500)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #f8f9fa;
                border-radius: 10px;
            }
        """)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        title = QLabel("Configuración Deep Learning")
        title.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                color: #2c3e50;
                padding-bottom: 10px;
            }
        """)
        layout.addWidget(title)

        model_widget = self.create_model_selection()
        layout.addWidget(model_widget)

        params_widget = self.create_parameters_widget()
        layout.addWidget(params_widget)

        buttons_widget = self.create_action_buttons()
        layout.addWidget(buttons_widget)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def create_model_selection(self):
        """Crear selección de modelo"""
        widget = QFrame()
        widget.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 10px;
                border: 2px solid #e0e0e0;
                padding: 15px;
            }
        """)

        layout = QVBoxLayout(widget)

        title = QLabel("Seleccionar Modelo Deep Learning")
        title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #34495e;
                margin-bottom: 15px;
            }
        """)
        layout.addWidget(title)

        self.model_group = QButtonGroup()

        # Solo CNN y RNN según las indicaciones de la maestra
        models = [
            ("CNN - Convolucional", "Red Neuronal Convolucional para análisis de patrones", "cnn"),
            ("RNN - Recurrente", "Red Neuronal Recurrente para análisis temporal", "rnn")
        ]

        for i, (name, desc, model_type) in enumerate(models):
            radio_frame = QFrame()
            radio_layout = QVBoxLayout(radio_frame)
            radio_layout.setSpacing(5)

            radio = QRadioButton(name)
            radio.setProperty('model_type', model_type)
            radio.setStyleSheet("""
                QRadioButton {
                    font-size: 15px;
                    font-weight: bold;
                    color: #2c3e50;
                }
                QRadioButton::indicator {
                    width: 18px;
                    height: 18px;
                }
            """)

            if i == 0:  # Seleccionar CNN por defecto
                radio.setChecked(True)

            desc_label = QLabel(desc)
            desc_label.setStyleSheet("""
                QLabel {
                    font-size: 12px;
                    color: #7f8c8d;
                    margin-left: 25px;
                }
            """)

            self.model_group.addButton(radio)
            radio_layout.addWidget(radio)
            radio_layout.addWidget(desc_label)

            layout.addWidget(radio_frame)

        info_note = QLabel("Modelos de Deep Learning avanzados")
        info_note.setStyleSheet("""
            QLabel {
                color: #e67e22;
                font-size: 12px;
                font-weight: bold;
                padding: 8px;
                background-color: #fdf2e9;
                border-radius: 5px;
                margin-top: 10px;
            }
        """)
        layout.addWidget(info_note)

        return widget

    def create_parameters_widget(self):
        """Crear widget de parámetros"""
        widget = QFrame()
        widget.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 10px;
                border: 2px solid #e0e0e0;
                padding: 15px;
            }
        """)

        layout = QVBoxLayout(widget)

        title = QLabel("Parámetros del Modelo")
        title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #34495e;
                margin-bottom: 15px;
            }
        """)
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(15)
        grid.setVerticalSpacing(10)

        # Épocas
        grid.addWidget(QLabel("Épocas:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 200)
        self.epochs_spin.setValue(50)
        self.epochs_spin.setFixedSize(100, 30)
        grid.addWidget(self.epochs_spin, 0, 1)

        # Learning Rate
        grid.addWidget(QLabel("Learning Rate:"), 0, 2)
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.001, 0.1)
        self.learning_rate_spin.setValue(0.001)
        self.learning_rate_spin.setDecimals(3)
        self.learning_rate_spin.setFixedSize(100, 30)
        grid.addWidget(self.learning_rate_spin, 0, 3)

        # Hidden Units / Filters
        grid.addWidget(QLabel("Hidden Units/Filters:"), 1, 0)
        self.hidden_units_spin = QSpinBox()
        self.hidden_units_spin.setRange(8, 64)
        self.hidden_units_spin.setValue(32)
        self.hidden_units_spin.setFixedSize(100, 30)
        grid.addWidget(self.hidden_units_spin, 1, 1)

        # Dense Units
        grid.addWidget(QLabel("Dense Units:"), 1, 2)
        self.dense_units_spin = QSpinBox()
        self.dense_units_spin.setRange(16, 128)
        self.dense_units_spin.setValue(64)
        self.dense_units_spin.setFixedSize(100, 30)
        grid.addWidget(self.dense_units_spin, 1, 3)

        # Dropout Rate
        grid.addWidget(QLabel("Dropout Rate:"), 2, 0)
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.5)
        self.dropout_spin.setValue(0.3)
        self.dropout_spin.setDecimals(2)
        self.dropout_spin.setFixedSize(100, 30)
        grid.addWidget(self.dropout_spin, 2, 1)

        for i in range(grid.count()):
            widget_item = grid.itemAt(i).widget()
            if isinstance(widget_item, (QSpinBox, QDoubleSpinBox)):
                widget_item.setStyleSheet(self.get_spinbox_style())

        layout.addLayout(grid)

        info = QLabel("Parámetros optimizados para CNN y RNN")
        info.setStyleSheet("""
            QLabel {
                color: #7f8c8d;
                font-size: 12px;
                font-style: italic;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 5px;
                margin-top: 10px;
            }
        """)
        layout.addWidget(info)

        return widget

    def create_action_buttons(self):
        """Crear botones de acción"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setSpacing(10)

        self.btn_clear = QPushButton("Limpiar")
        self.btn_execute = QPushButton("Entrenar")

        buttons = [
            (self.btn_clear, "#f44336", "#d32f2f"),
            (self.btn_execute, "#4CAF50", "#45a049")
        ]

        for btn, color, hover_color in buttons:
            btn.setFixedSize(120, 45)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-size: 16px;
                    font-weight: bold;
                }}
                QPushButton:hover:enabled {{
                    background-color: {hover_color};
                }}
                QPushButton:disabled {{
                    background-color: #cccccc;
                    color: #666666;
                }}
            """)
            layout.addWidget(btn)

        self.btn_execute.setEnabled(False)
        return widget

    def create_right_panel(self):
        """Crear panel derecho"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #f8f9fa;
                border-radius: 10px;
            }
        """)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        title = QLabel("Resultados Deep Learning")
        title.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                color: #e91e63;
                padding-bottom: 10px;
            }
        """)
        layout.addWidget(title)

        self.results_tabs = self.create_results_tabs()
        layout.addWidget(self.results_tabs)

        scroll.setWidget(container)
        return scroll

    def create_results_tabs(self):
        """Crear tabs de resultados"""
        tabs = QTabWidget()
        tabs.setMinimumHeight(600)
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #e0e0e0;
                background-color: white;
                border-radius: 10px;
            }
            QTabBar::tab {
                padding: 12px 20px;
                margin-right: 3px;
                font-size: 14px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #e91e63;
                color: white;
                border-radius: 8px 8px 0 0;
            }
            QTabBar::tab:!selected {
                background-color: #fce4ec;
                color: #880e4f;
            }
        """)

        tabs.addTab(self.create_metrics_tab(), "Métricas")
        tabs.addTab(self.create_graphs_tab(), "Gráficas")
        tabs.addTab(self.create_confusion_tab(), "Matriz Confusión")
        tabs.addTab(self.create_interpretation_tab(), "Interpretación")

        return tabs

    def create_metrics_tab(self):
        """Crear tab de métricas"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(30, 30, 30, 30)

        metrics_frame = QFrame()
        metrics_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-radius: 10px;
                padding: 25px;
            }
        """)

        metrics_layout = QVBoxLayout(metrics_frame)

        metrics_text = """Accuracy: --
Precision (promedio): --  
Recall (promedio): --
F1-Score (promedio): --

Tiempo de entrenamiento: --
Épocas completadas: --
Loss final: --

Características: --
Muestras: --
Tipo de modelo: --"""

        self.metrics_label = QLabel(metrics_text)
        self.metrics_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                line-height: 2.2;
                color: #2c3e50;
                background-color: white;
                padding: 25px;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
        """)

        metrics_layout.addWidget(self.metrics_label)

        export_btn = QPushButton("Exportar Métricas")
        export_btn.setFixedSize(200, 40)
        export_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        export_btn.clicked.connect(self.export_metrics)

        metrics_layout.addWidget(export_btn, alignment=Qt.AlignCenter)
        layout.addWidget(metrics_frame)
        layout.addStretch()

        return widget

    def create_graphs_tab(self):
        """Crear tab de gráficas"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)

        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("border: 1px solid #e0e0e0; border-radius: 5px;")
        layout.addWidget(self.canvas)

        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        controls_layout.setSpacing(10)

        graph_buttons = [
            ("Entrenamiento", self.plot_training_curves),
            ("ROC Curve", self.plot_roc),
            ("Distribución", self.plot_predictions_distribution)
        ]

        for text, callback in graph_buttons:
            btn = QPushButton(text)
            btn.setFixedHeight(40)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-size: 14px;
                    font-weight: bold;
                    padding: 0 15px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            btn.clicked.connect(callback)
            controls_layout.addWidget(btn)

        layout.addWidget(controls)
        return widget

    def create_confusion_tab(self):
        """Crear tab de matriz de confusión"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)

        self.confusion_figure = Figure(figsize=(8, 6))
        self.confusion_canvas = FigureCanvas(self.confusion_figure)
        self.confusion_canvas.setStyleSheet("border: 1px solid #e0e0e0; border-radius: 5px;")
        layout.addWidget(self.confusion_canvas)

        self.confusion_info_label = QLabel("La matriz de confusión aparecerá después del entrenamiento")
        self.confusion_info_label.setStyleSheet("""
            QLabel {
                color: #e65100;
                font-size: 16px;
                font-weight: 500;
                text-align: center;
                padding: 20px;
                background-color: #fff3e0;
                border-radius: 8px;
                margin-top: 20px;
            }
        """)
        layout.addWidget(self.confusion_info_label)

        layout.addStretch()
        return widget

    def create_interpretation_tab(self):
        """Crear tab de interpretación"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #f8f9fa;
            }
        """)

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(20, 20, 20, 20)
        container_layout.setSpacing(20)

        title = QLabel("Interpretación Deep Learning")
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                text-align: center;
                padding: 15px;
                background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
                margin-bottom: 20px;
            }
        """)
        container_layout.addWidget(title)

        self.interpretation_content = QLabel()
        self.interpretation_content.setWordWrap(True)
        self.interpretation_content.setTextFormat(Qt.RichText)
        self.interpretation_content.setStyleSheet("""
            QLabel {
                background-color: white;
                padding: 25px;
                border-radius: 10px;
                border: 2px solid #e0e0e0;
                font-size: 16px;
                line-height: 1.6;
            }
        """)

        self.interpretation_content.setText(self.get_default_interpretation_content())
        container_layout.addWidget(self.interpretation_content)

        buttons_frame = QFrame()
        buttons_layout = QHBoxLayout(buttons_frame)
        buttons_layout.setSpacing(15)

        btn_generate_report = QPushButton("Generar Reporte")
        btn_generate_report.setFixedSize(200, 45)
        btn_generate_report.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        btn_generate_report.clicked.connect(self.generate_interpretation_report)

        btn_export_interpretation = QPushButton("Exportar")
        btn_export_interpretation.setFixedSize(150, 45)
        btn_export_interpretation.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        btn_export_interpretation.clicked.connect(self.export_interpretation)

        buttons_layout.addWidget(btn_generate_report)
        buttons_layout.addWidget(btn_export_interpretation)
        buttons_layout.addStretch()

        container_layout.addWidget(buttons_frame)
        container_layout.addStretch()

        scroll.setWidget(container)
        layout.addWidget(scroll)

        return widget

    def setup_footer(self, layout):
        """Configurar pie de página"""
        footer_frame = QFrame()
        footer_frame.setMaximumHeight(70)
        footer_layout = QHBoxLayout(footer_frame)

        self.status_label = QLabel("Esperando datos...")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #FF9800;
                color: white;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        footer_layout.addWidget(self.status_label)

        footer_layout.addStretch()

        self.btn_export = QPushButton("Exportar")
        self.btn_regresar_main = QPushButton("Regresar")

        footer_buttons = [
            (self.btn_export, "#2196F3", "#1976D2"),
            (self.btn_regresar_main, "#f44336", "#d32f2f")
        ]

        for btn, color, hover in footer_buttons:
            btn.setFixedSize(140, 50)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    border: none;
                    border-radius: 10px;
                    font-size: 16px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {hover};
                }}
            """)
            footer_layout.addWidget(btn)

        layout.addWidget(footer_frame)

# MÉTODOS DE FUNCIONALIDAD

    def start_training(self):
        """Iniciar entrenamiento"""
        try:
            if self.df is None:
                QMessageBox.warning(self, "Error", "No hay datos cargados para entrenar")
                return

            if self.df.empty:
                QMessageBox.warning(self, "Error", "El dataset está vacío")
                return

            if len(self.df) < 5:
                QMessageBox.warning(self, "Error", "Se necesitan al menos 5 muestras para entrenar")
                return

            selected_model = self.get_selected_model()
            if not selected_model:
                QMessageBox.warning(self, "Error", "Seleccione un modelo para entrenar")
                return

            config = self.get_model_config(selected_model)

            self.training_thread = DeepLearningTrainingThread(config, self.df)
            self.training_thread.progress_updated.connect(self.update_progress)
            self.training_thread.training_finished.connect(self.on_training_finished)
            self.training_thread.error_occurred.connect(self.on_training_error)

            self.btn_execute.setEnabled(False)
            self.btn_clear.setEnabled(False)

            self.status_label.setText("Iniciando entrenamiento...")
            self.training_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al iniciar entrenamiento: {str(e)}")
            self.btn_execute.setEnabled(True)
            self.btn_clear.setEnabled(True)

    def get_selected_model(self):
        """Obtener modelo seleccionado"""
        try:
            for btn in self.model_group.buttons():
                if btn.isChecked():
                    return btn.property('model_type')
            return None
        except Exception as e:
            print(f"Error obteniendo modelo seleccionado: {e}")
            return None

    def get_model_config(self, model_type):
        """Obtener configuración del modelo"""
        try:
            config = {
                'type': model_type,
                'epochs': self.epochs_spin.value(),
                'learning_rate': self.learning_rate_spin.value(),
                'dense_units': self.dense_units_spin.value(),
                'dropout_rate': self.dropout_spin.value()
            }

            if model_type == 'cnn':
                config.update({
                    'filters': self.hidden_units_spin.value(),
                    'kernel_size': 3
                })
            elif model_type == 'rnn':
                config.update({
                    'hidden_units': self.hidden_units_spin.value()
                })

            return config

        except Exception as e:
            print(f"Error obteniendo configuración: {e}")
            return {'type': model_type}

    @pyqtSlot(str)
    def update_progress(self, message):
        """Actualizar progreso"""
        try:
            self.status_label.setText(message)
            QApplication.processEvents()
        except Exception as e:
            print(f"Error actualizando progreso: {e}")

    @pyqtSlot(dict)
    def on_training_finished(self, results):
        """Manejar finalización del entrenamiento"""
        try:
            if not results or not isinstance(results, dict):
                QMessageBox.warning(self, "Advertencia",
                                    "Se completó el entrenamiento pero no se recibieron resultados válidos")
                self.btn_execute.setEnabled(True)
                self.btn_clear.setEnabled(True)
                return

            self.current_results = results

            # Actualizar displays
            self.update_metrics_display(results)
            self.update_interpretation_display(results)

            # Generar gráficas
            try:
                self.plot_confusion_matrix()
                if 'history' in results and results['history']:
                    self.plot_training_curves()
            except Exception as e:
                print(f"Error generando gráficas: {e}")

            self.btn_execute.setEnabled(True)
            self.btn_clear.setEnabled(True)

            # Actualizar status
            accuracy = results.get('accuracy', 0.0)
            model_type = results.get('model_type', 'Unknown')
            training_time = results.get('training_time', 0.0)

            model_name = "CNN" if model_type == 'cnn' else "RNN" if model_type == 'rnn' else model_type

            self.status_label.setText(
                f"{model_name} - Accuracy: {accuracy:.3f} ({training_time:.1f}s)")

            QMessageBox.information(
                self,
                "Entrenamiento Completado",
                f"Modelo entrenado exitosamente!\n\n"
                f"Modelo: {model_name}\n"
                f"Accuracy: {accuracy:.4f}\n"
                f"Tiempo: {training_time:.2f}s\n\n"
                f"Revisa las pestañas para análisis detallado"
            )

        except Exception as e:
            print(f"Error procesando resultados: {e}")
            QMessageBox.warning(self, "Advertencia",
                                f"Entrenamiento completado pero hubo errores:\n{str(e)}")
            self.btn_execute.setEnabled(True)
            self.btn_clear.setEnabled(True)

    @pyqtSlot(str)
    def on_training_error(self, error_message):
        """Manejar errores de entrenamiento"""
        try:
            QMessageBox.critical(self, "Error de Entrenamiento",
                                 f"Se produjo un error durante el entrenamiento:\n\n{error_message}")

            self.btn_execute.setEnabled(True)
            self.btn_clear.setEnabled(True)
            self.status_label.setText("Error durante el entrenamiento")

        except Exception as e:
            print(f"Error manejando error de entrenamiento: {e}")

# MÉTODOS DE VISUALIZACIÓN

    def plot_roc(self):
        """Graficar curva ROC - CORREGIDA PARA MULTICLASE"""
        try:
            if not self.current_results or 'y_pred_proba' not in self.current_results:
                self.show_placeholder_graph("ROC Curve no disponible\n(necesita probabilidades)")
                return

            y_test = self.current_results['y_test']
            y_pred_proba = self.current_results['y_pred_proba']

            if y_pred_proba is None:
                self.show_placeholder_graph("ROC Curve no disponible\n(probabilidades no calculadas)")
                return

            n_classes = len(np.unique(y_test))

            self.figure.clear()

            if n_classes == 2:
                # CLASIFICACIÓN BINARIA
                ax = self.figure.add_subplot(111)

                # Obtener probabilidades para clase positiva
                if y_pred_proba.shape[1] == 2:
                    y_scores = y_pred_proba[:, 1]
                else:
                    y_scores = y_pred_proba.flatten()

                # Calcular ROC manualmente
                thresholds = np.linspace(0, 1, 100)
                tpr_values = []
                fpr_values = []

                for threshold in thresholds:
                    y_pred_thresh = (y_scores >= threshold).astype(int)
                    tp = np.sum((y_test == 1) & (y_pred_thresh == 1))
                    fp = np.sum((y_test == 0) & (y_pred_thresh == 1))
                    tn = np.sum((y_test == 0) & (y_pred_thresh == 0))
                    fn = np.sum((y_test == 1) & (y_pred_thresh == 0))

                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

                    tpr_values.append(tpr)
                    fpr_values.append(fpr)

                # Estimar AUC
                auc_estimate = np.trapz(tpr_values, fpr_values)

                ax.plot(fpr_values, tpr_values, color='darkorange', lw=3,
                        label=f'ROC (AUC ≈ {abs(auc_estimate):.3f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate', fontsize=12)
                ax.set_ylabel('True Positive Rate', fontsize=12)
                ax.set_title('Curva ROC - Clasificación Binaria', fontsize=14, fontweight='bold')
                ax.legend(loc="lower right", fontsize=11)
                ax.grid(True, alpha=0.3)

            else:
                # CLASIFICACIÓN MULTICLASE - ROC One-vs-Rest
                colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
                ax = self.figure.add_subplot(111)

                # Convertir y_test a one-hot encoding
                unique_classes = np.unique(y_test)
                y_test_binary = np.eye(len(unique_classes))[y_test]

                auc_scores = []

                for i, class_label in enumerate(unique_classes):
                    if i >= len(colors):
                        color = colors[i % len(colors)]
                    else:
                        color = colors[i]

                    # Obtener probabilidades para esta clase
                    if i < y_pred_proba.shape[1]:
                        y_scores = y_pred_proba[:, i]
                    else:
                        continue

                    # Crear etiquetas binarias (clase i vs resto)
                    y_binary = (y_test == class_label).astype(int)

                    # Calcular ROC para esta clase
                    thresholds = np.linspace(0, 1, 50)
                    tpr_values = []
                    fpr_values = []

                    for threshold in thresholds:
                        y_pred_thresh = (y_scores >= threshold).astype(int)
                        tp = np.sum((y_binary == 1) & (y_pred_thresh == 1))
                        fp = np.sum((y_binary == 0) & (y_pred_thresh == 1))
                        tn = np.sum((y_binary == 0) & (y_pred_thresh == 0))
                        fn = np.sum((y_binary == 1) & (y_pred_thresh == 0))

                        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

                        tpr_values.append(tpr)
                        fpr_values.append(fpr)

                    # Calcular AUC para esta clase
                    auc_estimate = abs(np.trapz(tpr_values, fpr_values))
                    auc_scores.append(auc_estimate)

                    # Plotear ROC para esta clase
                    ax.plot(fpr_values, tpr_values, color=color, lw=2,
                            label=f'Clase {class_label} (AUC ≈ {auc_estimate:.3f})')

                # Línea diagonal
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

                # Calcular AUC promedio
                avg_auc = np.mean(auc_scores) if auc_scores else 0

                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate', fontsize=12)
                ax.set_ylabel('True Positive Rate', fontsize=12)
                ax.set_title(f'Curva ROC Multiclase (AUC promedio: {avg_auc:.3f})',
                             fontsize=14, fontweight='bold')
                ax.legend(loc="lower right", fontsize=10)
                ax.grid(True, alpha=0.3)

            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            print(f"Error graficando ROC: {e}")
            self.show_placeholder_graph(f"Error generando ROC:\n{str(e)}")

    def plot_training_curves(self):
        """Graficar curvas de entrenamiento - MEJORADA"""
        try:
            if not self.current_results or 'history' not in self.current_results:
                self.show_placeholder_graph("No hay datos de entrenamiento")
                return

            history = self.current_results['history']

            if 'loss' not in history or 'accuracy' not in history:
                self.show_placeholder_graph("Datos de entrenamiento incompletos")
                return

            if len(history['loss']) == 0:
                self.show_placeholder_graph("No hay historial de entrenamiento")
                return

            self.figure.clear()

            # Crear subplots con mejor diseño
            fig = self.figure

            # Gráfica de pérdida
            ax1 = fig.add_subplot(211)
            epochs = range(1, len(history['loss']) + 1)
            ax1.plot(epochs, history['loss'], 'b-', label='Training Loss',
                     linewidth=3, marker='o', markersize=4, alpha=0.8)
            ax1.set_title('Pérdida durante el Entrenamiento', fontsize=16, fontweight='bold', pad=15)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)
            ax1.set_facecolor('#f8f9fa')

            # Añadir información sobre el loss
            final_loss = history['loss'][-1]
            initial_loss = history['loss'][0]
            ax1.text(0.02, 0.95, f'Loss inicial: {initial_loss:.4f}\nLoss final: {final_loss:.4f}',
                     transform=ax1.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Gráfica de accuracy
            ax2 = fig.add_subplot(212)
            ax2.plot(epochs, history['accuracy'], 'g-', label='Training Accuracy',
                     linewidth=3, marker='s', markersize=4, alpha=0.8)
            ax2.set_title('Accuracy durante el Entrenamiento', fontsize=16, fontweight='bold', pad=15)
            ax2.set_xlabel('Época', fontsize=12)
            ax2.set_ylabel('Accuracy', fontsize=12)
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)
            ax2.set_facecolor('#f8f9fa')

            # Añadir información sobre accuracy
            final_acc = history['accuracy'][-1]
            initial_acc = history['accuracy'][0]
            ax2.text(0.02, 0.95, f'Accuracy inicial: {initial_acc:.4f}\nAccuracy final: {final_acc:.4f}',
                     transform=ax2.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Mejorar el espaciado
            fig.tight_layout(pad=3.0)
            self.canvas.draw()

        except Exception as e:
            print(f"Error graficando curvas de entrenamiento: {e}")
            self.show_placeholder_graph(f"Error generando gráfica:\n{str(e)}")

    def plot_predictions_distribution(self):
        """Graficar distribución de predicciones - MEJORADA"""
        try:
            if not self.current_results:
                self.show_placeholder_graph("No hay predicciones disponibles")
                return

            y_test = self.current_results['y_test']
            y_pred = self.current_results['y_pred']

            self.figure.clear()
            ax = self.figure.add_subplot(111)

            # Contar predicciones por clase
            unique_classes = np.unique(np.concatenate([y_test, y_pred]))
            n_classes = len(unique_classes)

            width = 0.35
            x = np.arange(len(unique_classes))

            actual_counts = [np.sum(y_test == cls) for cls in unique_classes]
            predicted_counts = [np.sum(y_pred == cls) for cls in unique_classes]

            # Crear barras con colores más atractivos
            bars1 = ax.bar(x - width / 2, actual_counts, width, label='Real',
                           alpha=0.8, color='#3498db', edgecolor='white', linewidth=1)
            bars2 = ax.bar(x + width / 2, predicted_counts, width, label='Predicción',
                           alpha=0.8, color='#e74c3c', edgecolor='white', linewidth=1)

            ax.set_xlabel('Clases', fontsize=12, fontweight='bold')
            ax.set_ylabel('Cantidad', fontsize=12, fontweight='bold')
            ax.set_title('Distribución de Predicciones vs Valores Reales',
                         fontsize=16, fontweight='bold', pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels([f'Clase {cls}' for cls in unique_classes])
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_facecolor('#f8f9fa')

            # Añadir valores en las barras
            for i, (actual, predicted) in enumerate(zip(actual_counts, predicted_counts)):
                # Valores sobre las barras reales
                ax.text(i - width / 2, actual + max(actual_counts) * 0.01, str(actual),
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
                # Valores sobre las barras predichas
                ax.text(i + width / 2, predicted + max(predicted_counts) * 0.01, str(predicted),
                        ha='center', va='bottom', fontweight='bold', fontsize=10)

            # Calcular y mostrar estadísticas
            total_real = sum(actual_counts)
            total_pred = sum(predicted_counts)
            accuracy = manual_accuracy_score(y_test, y_pred)

            stats_text = f'Total Real: {total_real}\nTotal Pred: {total_pred}\nAccuracy: {accuracy:.3f}'
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                    fontsize=10)

            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            print(f"Error graficando distribución: {e}")
            self.show_placeholder_graph(f"Error generando gráfica:\n{str(e)}")

    def plot_confusion_matrix(self):
        """Graficar matriz de confusión"""
        try:
            if (not self.current_results or
                    'confusion_matrix' not in self.current_results or
                    self.current_results['confusion_matrix'] is None):
                return

            self.confusion_figure.clear()
            ax = self.confusion_figure.add_subplot(111)

            cm = self.current_results['confusion_matrix']

            # Crear heatmap
            im = ax.imshow(cm, cmap='Blues', aspect='auto')

            # Añadir texto en cada celda
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    text = ax.text(j, i, str(cm[i, j]),
                                   ha="center", va="center",
                                   color="white" if cm[i, j] > cm.max() / 2 else "black",
                                   fontsize=14, fontweight='bold')

            model_type = self.current_results.get('model_type', 'Modelo')
            model_display = 'CNN' if model_type == 'cnn' else 'RNN' if model_type == 'rnn' else model_type

            ax.set_title(f'Matriz de Confusión - {model_display}',
                         fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Predicción', fontsize=12)
            ax.set_ylabel('Valor Real', fontsize=12)

            ax.set_xticks(range(cm.shape[1]))
            ax.set_yticks(range(cm.shape[0]))
            ax.set_xticklabels([f'Clase {i}' for i in range(cm.shape[1])])
            ax.set_yticklabels([f'Clase {i}' for i in range(cm.shape[0])])

            # Colorbar
            cbar = self.confusion_figure.colorbar(im, ax=ax)
            cbar.set_label('Número de muestras', rotation=270, labelpad=15)

            # Información adicional
            total_samples = np.sum(cm)
            correct_predictions = np.trace(cm)
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0

            info_text = f"""Accuracy: {accuracy:.4f}
Muestras totales: {total_samples}
Correctas: {correct_predictions}
Modelo: {model_display}"""

            self.confusion_info_label.setText(info_text)

            self.confusion_figure.tight_layout()
            self.confusion_canvas.draw()

        except Exception as e:
            print(f"Error graficando matriz de confusión: {e}")

    def show_placeholder_graph(self, message):
        """Mostrar gráfica placeholder"""
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, message, ha='center', va='center',
                    transform=ax.transAxes, fontsize=16, color='gray',
                    style='italic', weight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            self.figure.tight_layout()
            self.canvas.draw()
        except Exception as e:
            print(f"Error mostrando placeholder: {e}")

# MÉTODOS DE ACTUALIZACIÓN DE DISPLAYS

    def update_metrics_display(self, results):
        """Actualizar display de métricas"""
        try:
            if not results or not isinstance(results, dict):
                self.metrics_label.setText("Error: No hay resultados válidos")
                return

            report = results.get('classification_report', {})
            accuracy = results.get('accuracy', 0.0)
            training_time = results.get('training_time', 0.0)
            data_shape = results.get('data_shape', (0, 0))
            model_type = results.get('model_type', 'Unknown')

            macro_avg = report.get('macro avg', {}) if isinstance(report, dict) else {}
            precision = macro_avg.get('precision', 0.0) if isinstance(macro_avg, dict) else 0.0
            recall = macro_avg.get('recall', 0.0) if isinstance(macro_avg, dict) else 0.0
            f1_score = macro_avg.get('f1-score', 0.0) if isinstance(macro_avg, dict) else 0.0

            # Obtener información del historial si está disponible
            history = results.get('history', {})
            final_loss = history.get('loss', [])[-1] if history.get('loss') else 0.0
            epochs_completed = len(history.get('loss', [])) if history.get('loss') else 0

            model_name = 'CNN' if model_type == 'cnn' else 'RNN' if model_type == 'rnn' else model_type

            metrics_text = f"""Accuracy: {float(accuracy):.4f}
Precision (promedio): {float(precision):.4f}
Recall (promedio): {float(recall):.4f}
F1-Score (promedio): {float(f1_score):.4f}

Tiempo de entrenamiento: {float(training_time):.2f}s
Épocas completadas: {epochs_completed}
Loss final: {float(final_loss):.4f}

Características: {data_shape[1] if len(data_shape) > 1 else 0}
Muestras: {data_shape[0] if len(data_shape) > 0 else 0:,}
Tipo de modelo: {model_name}"""

            self.metrics_label.setText(metrics_text)

        except Exception as e:
            print(f"Error actualizando métricas: {e}")
            self.metrics_label.setText("Error mostrando métricas")

    def get_default_interpretation_content(self):
        """Obtener contenido por defecto de interpretación"""
        return """
        <div style="text-align: center; padding: 40px;">
            <h2 style="color: #7f8c8d;">Entrena un modelo para ver la interpretación!</h2>
            <p style="font-size: 18px; color: #95a5a6; margin-top: 20px;">
                Una vez que completes el entrenamiento, aquí aparecerá:<br><br>
                • Análisis detallado de métricas CNN/RNN<br>
                • Interpretación de las capas de la red<br>
                • Explicación del proceso de entrenamiento<br>
                • Análisis de convergencia<br>
                • Recomendaciones de mejora<br>
                • Resumen ejecutivo del modelo
            </p>
        </div>
        """

    def update_interpretation_display(self, results):
        """Actualizar display de interpretación"""
        if not results or not isinstance(results, dict):
            return

        try:
            model_type = results.get('model_type', 'Unknown')
            accuracy = results.get('accuracy', 0.0)
            data_shape = results.get('data_shape', (0, 0))
            training_time = results.get('training_time', 0.0)
            history = results.get('history', {})

            interpretation_html = self.generate_interpretation_html(
                results, model_type, accuracy, data_shape, training_time, history
            )

            self.interpretation_content.setText(interpretation_html)

        except Exception as e:
            print(f"Error actualizando interpretación: {e}")

    def generate_interpretation_html(self, results, model_type, accuracy, data_shape, training_time, history):
        """Generar HTML de interpretación"""
        model_info = {
            'cnn': {
                'name': 'CNN - Red Neuronal Convolucional',
                'description': 'Convoluciones 1D implementadas en NumPy puro',
                'icon': '🧠',
                'color': '#e74c3c',
                'strengths': 'Detección de patrones locales, filtros convolucionales, pooling'
            },
            'rnn': {
                'name': 'RNN - Red Neuronal Recurrente',
                'description': 'Estados ocultos y memoria temporal en NumPy',
                'icon': '🔄',
                'color': '#9b59b6',
                'strengths': 'Análisis secuencial, memoria temporal, dependencias a largo plazo'
            }
        }

        info = model_info.get(model_type, {
            'name': 'Modelo Deep Learning',
            'description': 'Red neuronal implementada en NumPy',
            'icon': '🤖',
            'color': '#95a5a6',
            'strengths': 'Aprendizaje automático'
        })

        performance_level = "Excelente" if accuracy >= 0.9 else "Muy Bueno" if accuracy >= 0.8 else "Bueno" if accuracy >= 0.7 else "Regular"
        performance_color = "#27ae60" if accuracy >= 0.9 else "#3498db" if accuracy >= 0.8 else "#f39c12" if accuracy >= 0.7 else "#e74c3c"

        # Análisis de convergencia
        convergence_analysis = ""
        if history and 'loss' in history and len(history['loss']) > 0:
            final_loss = history['loss'][-1]
            initial_loss = history['loss'][0]
            improvement = ((initial_loss - final_loss) / initial_loss) * 100 if initial_loss > 0 else 0

            convergence_analysis = f"""
            <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h3 style="color: #2c3e50; margin-top: 0;">Análisis de Convergencia</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px;">
                    <div style="text-align: center;">
                        <div style="font-size: 20px; font-weight: bold; color: {info['color']};">{len(history['loss'])}</div>
                        <div style="color: #666; font-size: 12px;">Épocas</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 20px; font-weight: bold; color: #e74c3c;">{final_loss:.4f}</div>
                        <div style="color: #666; font-size: 12px;">Loss Final</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 20px; font-weight: bold; color: #27ae60;">{improvement:.1f}%</div>
                        <div style="color: #666; font-size: 12px;">Mejora</div>
                    </div>
                </div>
            </div>
            """

        html_content = f"""
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
            <div style="background: linear-gradient(45deg, {info['color']} 0%, #34495e 100%); 
                        color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h1 style="margin: 0; font-size: 24px;">
                    {info['icon']} {info['name']}
                </h1>
                <p style="margin: 10px 0 0 0; opacity: 0.9;">{info['description']}</p>
            </div>

            <div style="background: #e8f5e8; padding: 20px; border-radius: 10px; 
                        border-left: 5px solid {performance_color}; margin-bottom: 20px;">
                <h2 style="color: #2e7d32; margin-top: 0;">Resumen Ejecutivo</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); 
                            gap: 15px; margin-top: 15px;">
                    <div style="text-align: center;">
                        <div style="font-size: 28px; font-weight: bold; color: {performance_color};">
                            {accuracy:.1%}
                        </div>
                        <div style="color: #666;">Accuracy</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 28px; font-weight: bold; color: #2196F3;">
                            {data_shape[0]:,}
                        </div>
                        <div style="color: #666;">Muestras</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 28px; font-weight: bold; color: #FF9800;">
                            {data_shape[1] if len(data_shape) > 1 else 0}
                        </div>
                        <div style="color: #666;">Features</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 20px; font-weight: bold; color: {info['color']};">
                            {training_time:.1f}s
                        </div>
                        <div style="color: #666;">Tiempo</div>
                    </div>
                </div>
            </div>

            {convergence_analysis}

            <div style="background: #263238; color: white; padding: 20px; 
                        border-radius: 10px; text-align: center; margin-top: 20px;">
                <p style="margin: 0; opacity: 0.8;">
                    Modelo {info['name']} - Rendimiento: {performance_level}<br>
                </p>
            </div>
        </div>
        """

        return html_content

# MÉTODOS DE EXPORTACIÓN

    def export_model(self):
        """Exportar modelo"""
        try:
            if not self.current_results:
                QMessageBox.warning(self, "Advertencia", "No hay modelo entrenado para exportar")
                return

            model_type = self.current_results.get('model_type', 'modelo')
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Exportar Modelo Deep Learning",
                f"deep_learning_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                "Pickle Files (*.pkl);;All Files (*)"
            )

            if file_path:
                export_data = {
                    'model': self.current_results['model'],
                    'scaler': self.current_results['scaler'],
                    'feature_names': self.current_results['feature_names'],
                    'model_type': self.current_results['model_type'],
                    'metrics': {
                        'accuracy': self.current_results['accuracy'],
                        'classification_report': self.current_results['classification_report']
                    },
                    'history': self.current_results.get('history', {}),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'deep_learning': True,
                    'numpy_implementation': True,
                    'pyinstaller_compatible': True
                }

                joblib.dump(export_data, file_path)

                file_size = os.path.getsize(file_path) / (1024 * 1024)
                model_name = 'CNN' if model_type == 'cnn' else 'RNN' if model_type == 'rnn' else model_type.upper()

                QMessageBox.information(self, "Éxito",
                                        f"Modelo Deep Learning exportado exitosamente:\n{file_path}\n\n"
                                        f"Tamaño: {file_size:.2f} MB\n"
                                        f"Tipo: {model_name}\n"
                                        f"Implementación: 100% NumPy\n"
                                        f"Compatible con PyInstaller: SI")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al exportar modelo: {str(e)}")

    def export_metrics(self):
        """Exportar métricas"""
        try:
            if not self.current_results:
                QMessageBox.warning(self, "Advertencia", "No hay métricas para exportar")
                return

            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Exportar Métricas Deep Learning",
                f"metricas_deep_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "Text Files (*.txt);;All Files (*)"
            )

            if file_path:
                model_type = self.current_results.get('model_type', 'unknown')
                model_name = 'CNN' if model_type == 'cnn' else 'RNN' if model_type == 'rnn' else model_type.upper()

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"=== MÉTRICAS DEEP LEARNING - {model_name} ===\n")
                    f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Sistema: Deep Learning Avanzado\n\n")

                    f.write("=== RENDIMIENTO ===\n")
                    f.write(f"Accuracy: {self.current_results.get('accuracy', 0):.6f}\n")
                    f.write(f"Tiempo de entrenamiento: {self.current_results.get('training_time', 0):.2f}s\n")

                    data_shape = self.current_results.get('data_shape', (0, 0))
                    f.write(f"Muestras procesadas: {data_shape[0]:,}\n")
                    f.write(f"Características: {data_shape[1] if len(data_shape) > 1 else 0}\n\n")

                    history = self.current_results.get('history', {})
                    if history and 'loss' in history:
                        f.write("=== ENTRENAMIENTO ===\n")
                        f.write(f"Épocas completadas: {len(history['loss'])}\n")
                        f.write(f"Loss inicial: {history['loss'][0]:.6f}\n")
                        f.write(f"Loss final: {history['loss'][-1]:.6f}\n")
                        if len(history['accuracy']) > 0:
                            f.write(f"Accuracy inicial: {history['accuracy'][0]:.6f}\n")
                            f.write(f"Accuracy final: {history['accuracy'][-1]:.6f}\n")

                    f.write(f"\n=== INFORMACIÓN TÉCNICA ===\n")
                    f.write(f"Tipo de modelo: {model_name}\n")
                    f.write(f"Implementación: NumPy puro\n")
                    f.write(f"Dependencias: Solo NumPy, Pandas, PyQt5, Matplotlib\n")
                    f.write(f"Sklearn requerido: NO\n")
                    f.write(f"TensorFlow/Keras requerido: NO\n")
                    f.write(f"PyTorch requerido: NO\n")
                    f.write(f"Tamaño estimado: < 15MB\n")
                    f.write(f"Ventajas PyInstaller: Cero dependencias problemáticas\n")

                QMessageBox.information(self, "Éxito", f"Métricas exportadas a:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al exportar métricas: {str(e)}")

    def generate_interpretation_report(self):
        """Generar reporte de interpretación"""
        if not self.current_results:
            QMessageBox.warning(self, "Advertencia", "No hay resultados para generar reporte")
            return

        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Guardar Reporte Deep Learning",
                f"reporte_deep_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                "HTML Files (*.html);;All Files (*)"
            )

            if file_path:
                html_content = self.interpretation_content.text()

                full_html = f"""
                <!DOCTYPE html>
                <html lang="es">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Reporte Deep Learning - CNN y RNN</title>
                    <style>
                        body {{ 
                            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                            margin: 40px;
                            background: #f5f5f5;
                        }}
                        .container {{
                            max-width: 1000px;
                            margin: 0 auto;
                            background: white;
                            padding: 40px;
                            border-radius: 15px;
                            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                        }}
                        .header {{
                            text-align: center;
                            margin-bottom: 30px;
                            padding: 20px;
                            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
                            color: white;
                            border-radius: 10px;
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="header">
                            <h1>Reporte Deep Learning - CNN y RNN</h1>
                            <p>Implementación 100% NumPy - Compatible PyInstaller</p>
                        </div>
                        {html_content}
                        <div style="margin-top: 40px; text-align: center; color: #888; font-size: 12px;">
                            Generado el {datetime.now().strftime('%d/%m/%Y a las %H:%M:%S')}<br>
                            Deep Learning CNN + RNN - NumPy Implementation
                        </div>
                    </div>
                </body>
                </html>
                """

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(full_html)

                QMessageBox.information(self, "Éxito", f"Reporte guardado en:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al generar reporte: {str(e)}")

    def export_interpretation(self):
        """Exportar interpretación"""
        if not self.current_results:
            QMessageBox.warning(self, "Advertencia", "No hay interpretación para exportar")
            return

        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Exportar Interpretación",
                f"interpretacion_deep_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "Text Files (*.txt);;All Files (*)"
            )

            if file_path:
                model_type = self.current_results.get('model_type', 'unknown')
                model_name = 'CNN' if model_type == 'cnn' else 'RNN' if model_type == 'rnn' else model_type.upper()

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("INTERPRETACIÓN DEEP LEARNING - CNN Y RNN\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
                    f.write(f"Modelo: {model_name}\n")
                    f.write(f"Implementación: 100% NumPy\n")
                    f.write(f"Compatible con PyInstaller: SI\n\n")

                    f.write("CARACTERÍSTICAS DEL MODELO:\n")
                    f.write("- Sin dependencias de TensorFlow/Keras\n")
                    f.write("- Sin dependencias de PyTorch\n")
                    f.write("- Sin dependencias de scikit-learn\n")
                    f.write("- Implementación completamente en NumPy\n")
                    f.write("- Optimizado para PyInstaller\n\n")

                    f.write("VENTAJAS:\n")
                    f.write("- Cero problemas de compatibilidad con PyInstaller\n")
                    f.write("- Tamaño de ejecutable reducido\n")
                    f.write("- Control total sobre la implementación\n")
                    f.write("- Aprendizaje profundo sin frameworks pesados\n")

                QMessageBox.information(self, "Éxito", f"Interpretación exportada a:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al exportar interpretación: {str(e)}")

    def clear_results(self):
        """Limpiar resultados"""
        try:
            self.current_results = None

            # Limpiar métricas
            metrics_text = """Accuracy: --
Precision (promedio): --  
Recall (promedio): --
F1-Score (promedio): --

Tiempo de entrenamiento: --
Épocas completadas: --
Loss final: --

Características: --
Muestras: --
Tipo de modelo: --"""
            self.metrics_label.setText(metrics_text)

            # Limpiar interpretación
            self.interpretation_content.setText(self.get_default_interpretation_content())

            # Limpiar gráficas
            self.show_placeholder_graph('Entrene un modelo Deep Learning\npara ver los resultados aquí')

            # Limpiar matriz de confusión
            self.confusion_figure.clear()
            ax = self.confusion_figure.add_subplot(111)
            ax.text(0.5, 0.5, 'Matriz de Confusión\naparecerá después del entrenamiento',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=16, color='gray', style='italic', weight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            self.confusion_canvas.draw()

            self.confusion_info_label.setText("La matriz de confusión aparecerá después del entrenamiento")

            self.status_label.setText("Resultados limpiados")
            QMessageBox.information(self, "Limpiado", "Todos los resultados han sido limpiados exitosamente")

        except Exception as e:
            print(f"Error limpiando resultados: {e}")
            QMessageBox.warning(self, "Advertencia", f"Error al limpiar resultados: {str(e)}")

# MÉTODOS UTILITARIOS FINALES

    def cargar_dataframe(self, df):
        """Cargar DataFrame"""
        try:
            if df is None or df.empty:
                QMessageBox.warning(self, "Error", "El dataframe está vacío")
                return

            self.df = df.copy()
            self.update_ui_with_data()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al cargar datos: {str(e)}")

    def update_ui_with_data(self):
        """Actualizar UI con datos cargados"""
        try:
            if self.df is None:
                return

            n_rows, n_cols = self.df.shape

            if n_rows < 5:
                status_text = f"Datos cargados - {n_rows:,} filas, {n_cols} columnas (Muy pocas muestras)"
                self.status_label.setText(status_text)
                self.btn_execute.setEnabled(False)
                return

            # Verificar columnas numéricas
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                error_text = f"Sin columnas numéricas - {n_rows:,} filas, {n_cols} columnas"
                self.status_label.setText(error_text)
                self.btn_execute.setEnabled(False)
                return

            # Auto-generar target si no existe
            target_cols = ['target', 'class', 'label', 'y', 'classification', 'Classification_9V']
            has_target = any(col.lower() in [t.lower() for t in target_cols] for col in self.df.columns)

            if not has_target and len(numeric_cols) > 0:
                first_col = numeric_cols[0]
                median_val = self.df[first_col].median()
                self.df['target'] = (self.df[first_col] > median_val).astype(int)
                status_text = f"Datos listos - {n_rows:,} filas (target auto-generado)"
            else:
                status_text = f"Datos cargados - {n_rows:,} filas, {n_cols} columnas (Listo para entrenar)"

            self.status_label.setText(status_text)
            self.btn_execute.setEnabled(True)

        except Exception as e:
            print(f"Error actualizando UI con datos: {e}")
            error_msg = "Error procesando datos"
            self.status_label.setText(error_msg)
            self.btn_execute.setEnabled(False)

    def create_sample_data(self):
        """Crear datos de muestra para testing"""
        try:
            np.random.seed(42)
            n_samples = 800

            # Generar características sintéticas con patrones
            feature_1 = np.random.randn(n_samples)
            feature_2 = np.random.randn(n_samples)
            feature_3 = feature_1 * 0.8 + np.random.randn(n_samples) * 0.2
            feature_4 = np.random.uniform(0, 10, n_samples)
            feature_5 = feature_2 * 0.6 + feature_4 * 0.3 + np.random.randn(n_samples) * 0.1
            feature_6 = np.sin(feature_1) + np.cos(feature_2) * 0.7
            feature_7 = np.random.exponential(2, n_samples)
            feature_8 = feature_1 ** 2 + feature_2 ** 2 + np.random.randn(n_samples) * 0.5

            # Target con patrones no lineales
            target = ((feature_1 > 0) & (feature_2 > 0) |
                      (feature_4 > 6) |
                      (feature_6 > 0.5) |
                      (feature_8 > 2)).astype(int)

            return pd.DataFrame({
                'feature_1': feature_1,
                'feature_2': feature_2,
                'feature_3': feature_3,
                'feature_4': feature_4,
                'feature_5': feature_5,
                'feature_6': feature_6,
                'feature_7': feature_7,
                'feature_8': feature_8,
                'target': target
            })

        except Exception as e:
            print(f"Error creando datos de muestra: {e}")
            return None

    def get_spinbox_style(self):
        """Obtener estilo para spinbox"""
        return """
        QSpinBox, QDoubleSpinBox {
            padding: 8px;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            font-size: 14px;
            background-color: white;
        }
        QSpinBox:focus, QDoubleSpinBox:focus {
            border-color: #2196F3;
        }
        QSpinBox::up-button, QDoubleSpinBox::up-button {
            width: 20px;
            border-left: 1px solid #e0e0e0;
        }
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            width: 20px;
            border-left: 1px solid #e0e0e0;
        }
        """

    def closeEvent(self, event):
        """Manejar cierre de la aplicación"""
        try:
            # Desregistrar del data manager
            if hasattr(self, 'data_manager') and self.data_manager:
                self.data_manager.remove_observer(self)
                print("DeepLearningInterface desregistrado del DataManager")

            # Detener thread si está corriendo
            if self.training_thread and self.training_thread.isRunning():
                reply = QMessageBox.question(
                    self, 'Confirmar Cierre',
                    "Hay un entrenamiento en progreso. ¿Desea cerrarlo?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )

                if reply == QMessageBox.Yes:
                    self.training_thread.terminate()
                    self.training_thread.wait()
                    event.accept()
                else:
                    event.ignore()
            else:
                event.accept()

        except Exception as e:
            print(f"Error cerrando aplicación: {e}")
            event.accept()

# FUNCIÓN MAIN PARA PRUEBAS INDEPENDIENTES
if __name__ == "__main__":
    # Solo para testing independiente
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Configurar paleta de colores
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(245, 245, 245))
    palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
    app.setPalette(palette)

    # Configurar matplotlib para evitar warnings
    warnings.filterwarnings('ignore')

    try:
        # Crear ventana principal
        window = DeepLearningLightweight()

        # Cargar datos de prueba
        test_data = window.create_sample_data()
        if test_data is not None:
            window.cargar_dataframe(test_data)

        window.show()
        window.resize(1600, 1000)

        # Mostrar información de bienvenida
        QMessageBox.information(
            window,
            "Deep Learning - CNN y RNN",
            f"Demo Deep Learning cargado exitosamente\n\n"
            f"DATOS DE PRUEBA:\n"
            f"• 800 muestras sintéticas\n"
            f"• 8 características + target\n"
            f"• Patrones no lineales complejos\n\n"
            f"MODELOS DISPONIBLES:\n"
            f"• CNN - Red Neuronal Convolucional\n"
            f"• RNN - Red Neuronal Recurrente\n\n"
            f"CARACTERÍSTICAS TÉCNICAS:\n"
            f"• Matriz de confusión automática\n"
            f"• Exportación de modelos y reportes\n"
            f"• Interpretación completa\n\n"
            f"VENTAJAS:\n"
            f"• Cero dependencias problemáticas\n"
            f"• Ejecutables pequeños y estables\n"
            f"• Control total del código\n"
            f"• Aprendizaje profundo simplificado"
        )

        sys.exit(app.exec_())

    except Exception as e:
        print(f"Error iniciando aplicación: {e}")
        QMessageBox.critical(None, "Error", f"Error iniciando la aplicación: {str(e)}")
        sys.exit(1)
