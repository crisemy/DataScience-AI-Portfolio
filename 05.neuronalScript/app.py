import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Script Neuronal - Predicción del Titanic", layout="wide")

st.title("🧠 Script Neuronal - Predicción de Supervivencia en el Titanic")
st.markdown("Modelo de Aprendizaje Profundo usando arquitectura MLP para predecir el supervivencia en el Titanic")

@st.cache_data
def load_data():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "sample_data", "titanic.csv")
    df = pd.read_csv(csv_path)
    return df

@st.cache_resource
def train_model():
    df = load_data()

    # Preprocesamiento
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['age'] = df['age'].fillna(df['age'].median())
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    df = pd.get_dummies(df, columns=['embarked'], prefix='Embarked')

    X = df[['sex', 'pclass', 'age', 'sibsp', 'parch', 'fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']].astype(np.float32)
    y = df['survived'].astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    titanic_train = X_train.to_numpy()
    titanic_labels_train = y_train.to_numpy()
    titanic_test = X_test.to_numpy()
    titanic_labels_test = y_test.to_numpy()

    # Build model
    model = keras.Sequential([
        layers.Input(shape=(9,)),              # 9 características de entrada (actualizado de 7)
        layers.Dense(32, activation='relu'),   # Capa oculta 1
        layers.Dropout(0.2),                   # Regularización para evitar overfitting
        layers.Dense(16, activation='relu'),   # Capa oculta 2
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')  # Salida: probabilidad de sobrevivir
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(
        titanic_train, titanic_labels_train,
        epochs=30, batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    test_loss, test_accuracy = model.evaluate(titanic_test, titanic_labels_test, verbose=0)

    return model, history, test_accuracy, titanic_test, titanic_labels_test

df = load_data()

st.sidebar.header("Resumen del Dataset")
st.sidebar.write(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
st.sidebar.write("Columnas:", list(df.columns))

if st.sidebar.button("Entrenar Modelo"):
    with st.spinner("Entrenando red neuronal..."):
        model, history, test_accuracy, X_test, y_test = train_model()

    st.success(f"¡Entrenamiento completo! Precisión en el test: **{test_accuracy:.4f}**")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Evolución de la Precisión")
        st.markdown(
            "*Entrenamiento (azul)*: Rendimiento del modelo en la división de entrenamiento del 80%, "
            "mostrando qué tan bien aprende los patrones de los pasajeros a lo largo de 30 épocas.\n"
            "*Validación (naranja)*: Capacidad del modelo para generalizar a datos nuevos; "
            "si esta curva alcanza un máximo y luego disminuye mientras el entrenamiento continúa mejorando, "
            "indica overfitting (sobreajuste) a los detalles específicos del entrenamiento."
        )
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            y=history.history['accuracy'],
            mode='lines+markers',
            name='Entrenamiento',
            line=dict(color='blue')
        ))
        fig1.add_trace(go.Scatter(
            y=history.history['val_accuracy'],
            mode='lines+markers',
            name='Validación',
            line=dict(color='orange')
        ))
        fig1.update_layout(
            title='Precisión',
            xaxis_title='Épocas',
            yaxis_title='Precisión',
            hovermode='x unified',
            template='streamlit',
            width=600,
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Evolución de la Pérdida")
        st.markdown(
            "*Entrenamiento (azul)*: Error decreciente de binary crossentropy mientras el modelo "
            "minimiza la diferencia entre las probabilidades predichas de supervivencia y los resultados reales.\n"
            "*Validación (naranja)*: Métrica crítica de generalización; si aumenta mientras la pérdida de entrenamiento continúa disminuyendo, "
            "el modelo está memorizando ruido en lugar de aprender patrones significativos para nuevos datos de pasajeros."
        )
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            y=history.history['loss'],
            mode='lines+markers',
            name='Entrenamiento',
            line=dict(color='blue')
        ))
        fig2.add_trace(go.Scatter(
            y=history.history['val_loss'],
            mode='lines+markers',
            name='Validación',
            line=dict(color='orange')
        ))
        fig2.update_layout(
            title='Pérdida',
            xaxis_title='Épocas',
            yaxis_title='Loss',
            hovermode='x unified',
            template='streamlit',
            width=600,
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Predicciones de Muestra")
    predictions = model.predict(X_test[:30]).flatten()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Sobrevivientes Predichos** (>0.5)")
        for i, pred in enumerate(predictions[:15]):
            if pred > 0.5:
                st.write(f"Pasajero {i+1}: {pred:.4f} -> Sobrevivió")
    
    with col2:
        st.markdown("**No Sobrevivientes Predichos** (<=0.5)")
        for i, pred in enumerate(predictions[15:30]):
            if pred <= 0.5:
                st.write(f"Pasajero {i+16}: {pred:.4f} -> No sobrevivió")

else:
    st.info("Haz clic en 'Entrenar Modelo' en la barra lateral para comenzar el análisis")