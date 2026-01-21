# Planificación del Proyecto de IA: Ejercicios LSTM y CNN con Keras

## 1. Claridad en los Objetivos

**Objetivo principal del negocio:**
Desarrollar y evaluar modelos de deep learning (LSTM para series temporales y CNN para imágenes) usando Keras, con el fin de comprender y aplicar técnicas modernas de IA en problemas reales de predicción y clasificación.

**Criterio técnico de éxito:**
- Para LSTM: Obtener un modelo capaz de predecir secuencias temporales con baja pérdida en el conjunto de prueba.
- Para CNN: Lograr una precisión competitiva (>65%) en la clasificación de imágenes del dataset CIFAR-10.

## 2. Desglose de Tareas

### Ejercicio 1: LSTM para Series Temporales
1. Importar librerías y cargar datos sintéticos de series temporales.
2. Preprocesar datos: normalización y división en train/test.
3. Reestructurar datos al formato [muestras, pasos de tiempo, características].
4. Definir y compilar el modelo LSTM (Keras Sequential).
5. Entrenar el modelo y guardar el historial.
6. Evaluar el modelo en test y realizar predicciones.
7. Visualizar la pérdida y comparar predicciones vs. valores reales.

### Ejercicio 2: CNN para CIFAR-10
1. Importar librerías y cargar el dataset CIFAR-10.
2. Preprocesar imágenes y etiquetas (normalización y one-hot encoding).
3. Definir la arquitectura CNN (capas Conv2D, MaxPooling, Dense, Softmax).
4. Compilar el modelo con categorical_crossentropy y Adam.
5. Entrenar el modelo y validar.
6. Evaluar precisión en test.
7. Visualizar pérdida y precisión, y mostrar matriz de confusión.

## 3. Librerías Importantes
- **Numpy**: Manipulación de datos numéricos.
- **Pandas**: Carga y manejo de datasets.
- **Matplotlib/Seaborn**: Visualización de resultados.
- **TensorFlow/Keras**: Construcción y entrenamiento de modelos LSTM y CNN.
- **sklearn**: Preprocesamiento y métricas (especialmente para la matriz de confusión).

## 4. Modelos y Métricas
- **Modelos:**
  - LSTM (Long Short-Term Memory) para series temporales.
  - CNN (Convolutional Neural Network) para imágenes.
- **Métricas:**
  - LSTM: MSE (Error cuadrático medio), MAE (Error absoluto medio).
  - CNN: Precisión (accuracy), matriz de confusión.

## 5. Gráficas Interesantes
- Curvas de pérdida (train/val) para ambos modelos.
- Curva de precisión (train/val) para CNN.
- Gráfico de predicciones vs. valores reales para LSTM.
- Matriz de confusión para CNN.

---

# Plantilla Jupyter (Código)

```python
# Imports generales
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers

# --- Ejercicio 1: LSTM para Series Temporales ---
def load_synthetic_series():
    # TODO: Generar o cargar datos sintéticos
    pass

def preprocess_series(data):
    # TODO: Normalizar y dividir en train/test
    pass

def build_lstm_model(input_shape):
    # TODO: Definir arquitectura LSTM
    pass

def plot_lstm_results(history, y_test, y_pred):
    # TODO: Visualizar pérdida y predicciones
    pass

# --- Ejercicio 2: CNN para CIFAR-10 ---
def load_cifar10():
    # TODO: Cargar y preprocesar CIFAR-10
    pass

def build_cnn_model(input_shape, num_classes):
    # TODO: Definir arquitectura CNN
    pass

def plot_cnn_results(history, y_test, y_pred):
    # TODO: Visualizar pérdida, precisión y matriz de confusión
    pass

# --- Main execution (ejemplo de uso) ---
if __name__ == "__main__":
    # TODO: Llamar funciones y ejecutar flujos de trabajo
    pass
```
