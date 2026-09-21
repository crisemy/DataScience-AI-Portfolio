# Explicación Paso a Paso de Neuronal Script

Este documento explica cada paso del notebook neuronalScript.ipynb, que implementa una red neuronal de tipo Multi-Layer Perceptron (MLP) para predecir la supervivencia en el Titanic.

---

## Paso 0: Importaciones

El notebook comienza importando las librerías necesarias:

- `numpy as np`: Para operaciones numéricas y manipulación de arrays
- `tensorflow as tf`: El framework de deep learning
- `from tensorflow import keras`: API de alto nivel para construir redes neuronales
- `from tensorflow.keras import capas`: Capas de Keras para la arquitectura del modelo
- `matplotlib.pyplot as plt`: Para visualizar los resultados del entrenamiento
- `pandas as pd`: Para manipulación de datos y carga de CSV
- `from sklearn.model_selection import train_test_split`: Para dividir los conjuntos de datos en entrenamiento y prueba

Estas librerías proporcionan todas las herramientas necesarias para el procesamiento de datos, construcción del modelo, entrenamiento y evaluación.

---

## Paso 1: Cargar el dataset desde el archivo CSV y preprocesar

Este primer paso importante carga el dataset del Titanic y lo preprocesa para que coincida con el formato de entrada esperado por la red neuronal.

### 1.1 Cargar el dataset CSV

```python
df = pd.read_csv('/content/sample_data/titanic.csv')
```

- Carga el dataset del Titanic desde un archivo CSV
- El dataset contiene 890 pasajeros con 16 características que incluyen el estado de supervivencia, clase del pasajero, sexo, edad, etc.

### 1.2 Preprocesar la característica 'Sex' (de categórico a numérico)

```python
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
```

- Convierte la columna 'sex' de strings ('male', 'female') a valores numéricos (0, 1)
- Masculino → 0, Femenino → 1
- Esto es necesario porque las redes neuronales requieren entrada numérica

### 1.3 Rellenar valores nulos en 'Edad' con la mediana

```python
df['age'] = df['age'].fillna(df['age'].median())
```

- Calcula la mediana de edad de los valores existentes
- Reemplaza cualquier valor nulo/faltante de edad con esta mediana
- Una estrategia de imputación común que es robusta a valores atípicos

### 1.4 Preprocesar la característica 'Embarcado' (one-hot encoding)

```python
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0]) # Rellenar nulos antes de one-hot
df = pd.get_dummies(df, columns=['embarked'], prefix='Embarked'
```

- Rellena los valores faltantes del puerto de embarque con la moda (el puerto más frecuente: S, C o Q)
- Convierte la columna única 'embarcado' en tres columnas binarias usando one-hot encoding:
  - `Embarked_C` (embarcado en Cherbourg)
  - `Embarked_Q` (embarcado en Queenstown)
  - `Embarked_S` (embarcado en Southampton)
- Si un valor original de 'Embarcado' no existe, su columna one-hot será todos ceros
- Esta técnica transforma la variable categórica a un formato que la red neuronal puede entender

### 1.5 Seleccionar las 9 características

```python
X = df[['sex', 'pclass', 'age', 'sibsp', 'parch', 'fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
```

- Selecciona exactamente 9 características del dataset preprocesado:
  1. `sex` (0=hombre, 1-mujer)
  2. `pclass` (clase del pasajero: 1, 2 o 3)
  3. `age` (en años, con valores faltantes llenados)
  4. `sibsp` (número de hermanos/cónyuges a bordo)
  5. `parch` (número de padres/hijos a bordo)
  6. `fare` (precio del boleto)
  7. `Embarked_C` (one-hot: embarcado en Cherbourg)
  8. `Embarked_Q` (one-hot: embarcado en Queenstown)
  9. `Embarked_S` (one-hot: embarcado en Southampton)

### 1.6 Definir la variable objetivo

```python
y = df['survived']
```

- Establece la variable objetivo como la columna 'survived' (0 = no, 1 = sí)

### 1.7 Dividir el dataset en conjuntos de entrenamiento y prueba

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

- Divide los datos en 80% conjunto de entrenamiento y 20% conjunto de prueba
- `random_state=42` asegura resultados reproducibles
- `stratify=y` mantiene la misma proporción de supervivencia en ambas divisiones
- Esto permite evaluar la capacidad del modelo para generalizar a datos desconocidos

### 1.8 Convertir a arrays de NumPy para Keras

```python
titanic_train = X_train.to_numpy().astype(np.float32)
titanic_labels_train = y_train.to_numpy()
titanic_test = X_test.to_numpy().astype(np.float32)
titanic_labels_test = y_test.to_numpy()
```

- Convierte los DataFrames/Series de pandas a arrays de NumPy
- Asegura `dtype float32`, que es la precisión por defecto de TensorFlow/Keras
- Crea cuatro arrays: características de entrenamiento, etiquetas de entrenamiento, características de prueba, etiquetas de prueba

---

## Paso 2: Definir la arquitectura de la red neuronal

Este paso construye el modelo MLP usando la API Sequential de Keras.

```python
model = keras.Sequential([
    layers.Input(shape=(9,)),              # 9 características de entrada (actualizado de 7)
    layers.Dense(32, activation='relu'),   # Capa oculta 1
    layers.Dropout(0.2),                   # Regularización para evitar overfitting
    layers.Dense(16, activation='relu'),   # Capa oculta 2
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')  # Salida: probabilidad de sobrevivir
]
```

**Detalles de la arquitectura:**

- **Capa de entrada**: `shape=(9,)` acepta las 9 características preprocesadas. El comentario nota "actualizado de 7" porque originalmente había 7 categorías de características, pero después del one-hot encoding de `embarcado` (3 columnas), el total se convirtió en 9.

- **Capa oculta 1**: 32 neuronas con activación ReLU. ReLU (Rectified Linear Unit) introduce no-linealidad y ayuda al modelo a aprender patrones complejos.

- **Dropout 1**: Tasa de dropout del 20%. Aleatoriamente elimina el 20% de las neuronas durante el entrenamiento para prevenir overfitting.

- **Capa oculta 2**: 16 neuronas con activación ReLU. Menor que la primera capa oculta, creando un "cuello de botella" que obliga al modelo a aprender representaciones comprimidas.

- **Dropout 2**: Otra tasa de dropout del 20% para regularización.

- **Capa de salida**: 1 neurona con activación sigmoid. Sigmoid comprime la salida a una probabilidad entre 0 y 1,representando la probabilidad de supervivencia.

---

## Paso 3: Compilar el modelo

Este paso configura el modelo para el entrenamiento.

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',   # Adecuado para clasificación binaria
    metrics=['accuracy']
)
model.summary()
```

**Detalles de compilación:**

- **Optimizer: 'adam'** (Adaptive Moment Estimation)
  - Un algoritmo avanzado de descenso de gradiente
  - Adaptación de las tasas de aprendizaje para cada parámetro, convergencia más rápida
  - La tasa de aprendizaje por defecto funciona bien para la mayoría de los problemas

- **Loss: 'binary_crossentropy'**
  - Mide la diferencia entre la probabilidad predicha y el resultado binario real (0 o 1)
  - Adecuado para problemas de clasificación binaria (superviviente/no sobreviviente)
  - El modelo intenta minimizar esta pérdida durante el entrenamiento

- **Metrics: ['accuracy']**
  - Rastrea la precisión durante el entrenamiento y validación
  - `model.summary()` imprime la arquitectura del modelo, conteo de parámetros y formas de salida

---

## Paso 4: Entrenar el modelo

Este paso entrena la red neuronal sobre los datos de entrenamiento.

```python
history = model.fit(
    titanic_train, titanic_labels_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,        # 20% para validación
    verbose=1
)
```

**Detalles del entrenamiento:**

- **Input**: `titanic_train` (características) y `titanic_labels_train` (objetivos)
- **Epochs: 30**
  - Un epoch = un pase completo a través de todo el dataset de entrenamiento
  - 30 epochs fueron elegidos como equilibrio entre aprendizaje y overfitting
- **Batch size: 32**
  - Número de muestras procesadas antes de que el modelo se actualice
  - Lotes más pequeños proporcionan estimaciones de gradiente más ruidosas pero ayudan la generalización
- **Validation split: 0.2**
  - El 20% de los datos de entrenamiento se reserva para validación durante el entrenamiento
  - El rendimiento del modelo en este conjunto de validación se monitorea
  - Ayuda a detectar overfitting (cuando las métricas de entrenamiento mejoran pero las de validación se empeoran)
- **Verbose: 1**
  - Barra de progreso que muestra la pérdida y métricas por epoch

El objeto `history` devuelto contiene las métricas de entrenamiento y validación por epoch, las cuales luego se usarán para visualización.

---

## Paso 5: Evaluar el modelo

Este paso evalúa el modelo entrenado sobre el conjunto de prueba sin ver durante el entrenamiento.

```python
test_loss, test_accuracy = model.evaluate(titanic_test, titanic_labels_test, verbose=0)
print(f"\nPrecisión en el conjunto de prueba: {test_accuracy:.4f}")
```

**Detalles de evaluación:**

- `model.evaluate()` computa la performance del modelo sobre el conjunto de prueba (datos que NO fueron vistos durante el entrenamiento)
- Devuelve dos valores:
  - `test_loss`: La pérdida de binary_crossentropy sobre el conjunto de prueba
  - `test_accuracy`: La precisión de clasificación sobre el conjunto de prueba
- `verbose=0` suprime la salida de progreso
- La precisión impresa muestra qué tan bien el modelo generaliza a datos completamente nuevos de pasajeros
- Esta es la verdadera medida del poder predictivo del modelo, no solo su capacidad de memorizar los datos de entrenamiento

---

## Paso 6: Visualizar el entrenamiento

Este paso crea gráficos que muestran cómo evolucionó el rendimiento del modelo durante el entrenamiento.

```python
plt.figure(figsize=(12, 4))

# Izquierda: Gráfico de Precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

# Derecha: Gráfico de Pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida (Loss)')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

**Detalles de visualización:**

- **Tamaño de figura**: 12 pulgadas de ancho por 4 pulgadas de alto
- **Subplot izquierdo (1, 2, 1)**: Muestra la precisión por epoch
  - `history.history['accuracy']`: Precisión de entrenamiento por epoch
  - `history.history['val_accuracy']`: Precisión de validación por epoch
  - Título: "Precisión"
  - Eje X: "Épocas"
  - Eje Y: "Precisión"
  - Leyenda que distingue entrenamiento vs validación

- **Subplot derecho (1, 2, 2)**: Muestra la pérdida (error) por epoch
  - `history.history['loss']`: Pérdida de entrenamiento por epoch
  - `history.history['val_loss']`: Pérdida de validación por epoch
  - Título: "Pérdida (Loss)"
  - Eje X: "Épocas"
  - Eje Y: "Loss"
  - Leyenda que distingue entrenamiento vs validación

- **`plt.tight_layout()`**: Ajusta los parámetros de los subplots para dar un espacio apropiado entre los gráficos

 Estos gráficos ayudan a diagnosticar:

- **Convergencia**: Si el modelo está aprendiendo (pérdida decreciente, precisión creciente)
- **Overfitting**: Si la precisión/pérdida de entrenamiento sigue mejorando pero las métricas de validación se estancan o empeoran
- **Underfitting**: Si ambas métricas de entrenamiento y validation son pobres y no mejoran

---

## Paso 7: Hacer predicciones

Este paso final usa el modelo entrenado para hacer predicciones sobre el conjunto de prueba y muestra los resultados.

```python
predicciones = model.predict(titanic_test[:30])

survived_preds = []
not_survived_preds = []

for i, pred in enumerate(predicciones):
    passenger_info = f"  Pasajero {i+1}: {pred[0]:.4f}"
    if pred[0] > 0.5:
        survived_preds.append(f"{passenger_info} -> Sobrevivió")
    else:
        not_survived_preds.append(f"{passenger_info} -> No sobrevivió")

print("\n--- Predicciones de Sobrevivientes ---")
if survived_preds:
    for p in survived_preds:
        print(p)
else:
    print("  Ningún pasajero predicho como sobreviviente en esta muestra.")

print("\n--- Predicciones de No Sobrevivientes ---")
if not_survived_preds:
    for p in not_survived_preds:
        print(p)
else:
    print("  Ningún pasajero predicho como no sobreviviente en esta muestra.")
```

**Detalles de predicción:**

- `model.predict(titanic_test[:30])`: Realiza predicciones para los primeros 30 pasajeros del conjunto de prueba
- Devuelve un array de 30 probabilidades (valores entre 0 y 1)

**Lógica del umbral:**

- Se usa un umbral de 0.5: si la probabilidad predicha > 0.5, clasificar como "sobreviviente"; de otro modo "no sobreviviente"
- Este es un umbral por defecto común para clasificación binaria con salida sigmoid

**Organización de salida:**

- `survived_preds`: Lista de pasajeros predichos como sobrevivientes, con su probabilidad
- `not_survived_preds`: Lista de pasajeros predichos como no sobrevivientes, con su probabilidad

**Salida impresa:**

- Dos secciones que muestran:
  - "Predicciones de Sobrevivientes": Pasajeros con probabilidad >0.5 de sobrevivir
  - "Predicciones de No Sobrevivientes": Pasajeros con probabilidad ≤0.5 de sobrevivir
- Si no hay pasajeros en ninguna de las categorías ( poco probable con 30 predicciones), se muestra un mensaje al respecto

Las predicciones muestran las clasificaciones reales del modelo con probabilidades, permitiendo el análisis de qué características/rasgos de los pasajeros el modelo consideró importantes para la predicción de supervivencia.