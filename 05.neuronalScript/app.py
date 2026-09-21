import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Neuronal Script - Titanic Survival", layout="wide")

st.title("🧠 Neuronal Script - Titanic Survival Prediction")
st.markdown("Deep Learning model using MLP architecture to predict Titanic survival")

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

    # Preprocessing
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
        layers.Input(shape=(9,)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
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

st.sidebar.header("Dataset Overview")
st.sidebar.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
st.sidebar.write("Columns:", list(df.columns))

if st.sidebar.button("Train Model"):
    with st.spinner("Training neural network..."):
        model, history, test_accuracy, X_test, y_test = train_model()

    st.success(f"Training complete! Test accuracy: **{test_accuracy:.4f}**")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Accuracy Evolution")
        st.markdown(
            "*Training (blue)*: Model's performance on the 80% training split, "
            "showing how well it learns passenger patterns across 30 epochs.\n"
            "*Validation (orange)*: Model's ability to generalize to unseen data; "
            "if this curve peaks then declines while training continues improving, "
            "it indicates overfitting to training specifics."
        )
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            y=history.history['accuracy'],
            mode='lines+markers',
            name='Training',
            line=dict(color='blue')
        ))
        fig1.add_trace(go.Scatter(
            y=history.history['val_accuracy'],
            mode='lines+markers',
            name='Validation',
            line=dict(color='orange')
        ))
        fig1.update_layout(
            title='Accuracy',
            xaxis_title='Epochs',
            yaxis_title='Accuracy',
            hovermode='x unified',
            template='streamlit',
            width=600,
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Loss Evolution")
        st.markdown(
            "*Training (blue)*: Decreasing binary crossentropy error as the model "
            "minimizes the difference between predicted survival probabilities and actual outcomes.\n"
            "*Validation (orange)*: Critical generalization metric; if it rises while "
            "training loss continues falling, the model is memorizing noise rather than "
            "learning meaningful patterns for new passenger data."
        )
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            y=history.history['loss'],
            mode='lines+markers',
            name='Training',
            line=dict(color='blue')
        ))
        fig2.add_trace(go.Scatter(
            y=history.history['val_loss'],
            mode='lines+markers',
            name='Validation',
            line=dict(color='orange')
        ))
        fig2.update_layout(
            title='Loss',
            xaxis_title='Epochs',
            yaxis_title='Loss',
            hovermode='x unified',
            template='streamlit',
            width=600,
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Sample Predictions")
    predictions = model.predict(X_test[:30]).flatten()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Predicted Survivors** (>0.5)")
        for i, pred in enumerate(predictions[:15]):
            if pred > 0.5:
                st.write(f"Passenger {i+1}: {pred:.4f} -> Sobrevivió")
    
    with col2:
        st.markdown("**Predicted Non-Survivors** (<=0.5)")
        for i, pred in enumerate(predictions[15:30]):
            if pred <= 0.5:
                st.write(f"Passenger {i+16}: {pred:.4f} -> No sobrevivió")

else:
    st.info("Click 'Train Model' in the sidebar to start the analysis")