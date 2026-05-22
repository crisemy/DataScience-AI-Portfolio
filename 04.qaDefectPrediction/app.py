import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import base64
from pathlib import Path
from sklearn.metrics import (
    roc_curve, confusion_matrix, precision_recall_curve,
    roc_auc_score, average_precision_score
)

APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / 'model'

st.set_page_config(page_title="QA-Cortex | Defect Predictor", layout="wide", page_icon="")

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

hero_img_base64 = ""
hero_path = APP_DIR / ".." / "assets" / "hero.png"
if hero_path.exists():
    with open(hero_path, 'rb') as f:
        hero_img_base64 = base64.b64encode(f.read()).decode()

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Outfit', sans-serif; }}
    .main {{ background-color: #0f172a; color: #f8fafc; }}
    .header-banner {{
        background-image: linear-gradient(rgba(15,23,42,0.7), rgba(15,23,42,0.7)), url("data:image/png;base64,{hero_img_base64}");
        background-size: cover; background-position: center; height: 180px;
        border-radius: 20px; display: flex; flex-direction: column;
        justify-content: center; align-items: center; margin-bottom: 30px;
        border: 1px solid rgba(56,189,248,0.3); box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }}
    .header-banner h1 {{ color: #38bdf8 !important; font-size: 3rem !important; margin: 0 !important; text-shadow: 2px 2px 10px rgba(0,0,0,0.8); }}
    .header-banner p {{ color: #94a3b8 !important; font-size: 1.2rem; margin-top: 5px; }}
    .stMetric {{ background: rgba(30,41,59,0.7); padding: 20px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(10px); transition: transform 0.3s ease; }}
    .stMetric:hover {{ transform: translateY(-5px); border-color: #38bdf8; }}
    .stButton>button {{ background: linear-gradient(90deg,#38bdf8 0%,#818cf8 100%); color: white; border: none; padding: 10px 25px; border-radius: 10px; font-weight: 600; width: 100%; transition: all 0.3s ease; }}
    .stButton>button:hover {{ opacity: 0.9; box-shadow: 0 4px 15px rgba(56,189,248,0.4); }}
    .sidebar .sidebar-content {{ background: #0f172a; }}
    div[data-testid="stExpander"] {{ background: rgba(255,255,255,0.03); border-radius: 10px; border: 1px solid rgba(255,255,255,0.05); }}
    </style>
    <div class="header-banner">
        <h1>QA-CORTEX</h1>
        <p>Inteligencia Artificial para el Control de Calidad de Software</p>
    </div>
""", unsafe_allow_html=True)

MODEL_NAMES = {
    'logistic_regression': 'Logistic Regression',
    'random_forest': 'Random Forest',
    'svm_rbf': 'SVM (RBF)',
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM'
}

@st.cache_resource
def load_main_model():
    try:
        model = joblib.load(MODEL_DIR / 'defect_prediction_model.pkl')
        scaler = joblib.load(MODEL_DIR / 'scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

@st.cache_resource
def load_all_models():
    models = {}
    for key, name in MODEL_NAMES.items():
        path = MODEL_DIR / f'model_{key}.pkl'
        if path.exists():
            models[name] = joblib.load(path)
    return models

@st.cache_resource
def load_eval_data():
    try:
        metrics = joblib.load(MODEL_DIR / 'metrics.pkl')
        importance = joblib.load(MODEL_DIR / 'feature_importance.pkl')
        test_data = joblib.load(MODEL_DIR / 'test_data.pkl')
        return metrics, importance, test_data
    except FileNotFoundError:
        return None, None, None

@st.cache_resource
def load_comparison():
    path = MODEL_DIR / 'comparison_results.pkl'
    if path.exists():
        return pd.DataFrame(joblib.load(path))
    return None

main_model, scaler = load_main_model()
all_models = load_all_models()
eval_metrics, feature_imp, test_data = load_eval_data()
comparison_df = load_comparison()

PAGES = ["Inicio", "Predicción Individual", "Predicción por Lotes (CSV)", "Comparación de Modelos", "Análisis del Modelo"]
with st.sidebar:
    st.header("Navegación")
    page = st.selectbox("Ir a", PAGES)
    st.markdown("---")
    model_name = eval_metrics.get('model_name', 'Desconocido') if eval_metrics else 'No disponible'
    st.info(f"**QA-Cortex v2.0**\n\nModelo activo: {model_name}\nUnidad 1 y 2\nMaestría en Ciencia de Datos e IA")

if page == "Inicio":
    st.write("""
    ### Análisis Predictivo de Defectos
    Esta plataforma utiliza modelos avanzados de **Machine Learning** entrenados con métricas estáticas de código
    (NASA MDP Dataset) para identificar módulos de software con alta probabilidad de fallos.

    #### ¿Cómo funciona?
    1. **Recolección:** Se extraen métricas McCabe y Halstead del código fuente.
    2. **Normalización:** Los datos se procesan para asegurar la consistencia.
    3. **Inferencia:** El modelo predice la probabilidad de defectos.
    4. **Acción:** Los equipos de QA priorizan las pruebas en los módulos críticos.
    """)

    st.markdown("### Métricas del Sistema")
    if eval_metrics:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{eval_metrics['accuracy']:.1%}")
        col2.metric("ROC AUC", f"{eval_metrics['roc_auc']:.3f}")
        col3.metric("Precision", f"{eval_metrics['precision']:.3f}")
        col4.metric("Recall", f"{eval_metrics['recall']:.3f}")
        col1.metric("F1-Score", f"{eval_metrics['f1']:.3f}")
        col2.metric("PR-AUC", f"{eval_metrics['pr_auc']:.3f}")
        col3.metric("MCC", f"{eval_metrics['mcc']:.3f}")
        col4.metric("Log Loss", f"{eval_metrics['log_loss']:.3f}")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Estado", "Sin datos", "Ejecutar notebook primero")
        col2.metric("Modelo", "No cargado")
        col3.metric("Dataset", "SoftwareDefectDataset.csv")

elif page == "Predicción Individual":
    st.header("Análisis de Módulo Individual")

    model_list = list(all_models.keys()) if all_models else [eval_metrics.get('model_name', 'Modelo único')] if eval_metrics else ['Modelo único']
    selected_model_name = st.selectbox("Seleccionar modelo (TASK-26)", model_list)

    pred_model = all_models.get(selected_model_name) if all_models else main_model

    threshold = st.slider("Umbral de decisión (TASK-25)", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                          help="Ajusta la sensibilidad vs especificidad. Menor umbral = detecta más defectos (mayor recall, menor precisión).")

    with st.container():
        st.write("Ingrese las métricas técnicas del módulo para obtener un diagnóstico basado en IA.")
        col1, col2 = st.columns(2)
        with col1:
            loc = st.number_input("Lines of Code (LOC)", min_value=0.0, value=15.0)
            cyclo = st.number_input("Cyclomatic Complexity (CYCLO)", min_value=0.0, value=5.0)
            length = st.number_input("Halstead Length (LENGTH)", min_value=0.0, value=45.0)
            volume = st.number_input("Halstead Volume (VOLUME)", min_value=0.0, value=250.0)
            difficulty = st.number_input("Halstead Difficulty (DIFFICULTY)", min_value=0.0, value=12.0)
        with col2:
            fan_in = st.number_input("Internal Fan-In (INT_FAN_IN)", min_value=0.0, value=2.0)
            fan_out = st.number_input("Internal Fan-Out (INT_FAN_OUT)", min_value=0.0, value=3.0)
            num_ops = st.number_input("Number of Operators (NUM_OPERATORS)", min_value=0.0, value=15.0)
            num_opnds = st.number_input("Number of Operands (NUM_OPERANDS)", min_value=0.0, value=10.0)
            branches = st.number_input("Branch Count (BRANCH_COUNT)", min_value=0.0, value=8.0)

    if st.button("Ejecutar Predicción"):
        if pred_model and scaler:
            input_data = pd.DataFrame([[
                loc, cyclo, length, volume, difficulty,
                fan_in, fan_out, num_ops, num_opnds, branches
            ]], columns=[
                'LOC', 'CYCLO', 'LENGTH', 'VOLUME', 'DIFFICULTY',
                'INT_FAN_IN', 'INT_FAN_OUT', 'NUM_OPERATORS', 'NUM_OPERANDS', 'BRANCH_COUNT'
            ])
            try:
                input_scaled = scaler.transform(input_data)
                probability = pred_model.predict_proba(input_scaled)[0][1]
                prediction = 1 if probability >= threshold else 0

                st.markdown("---")
                st.write(f"**Modelo:** {selected_model_name}")
                st.write(f"**Umbral:** {threshold:.2f}")
                if prediction == 1:
                    st.error(f"### MÓDULO DEFECTUOSO DETECTADO")
                    st.write(f"Probabilidad: **{probability:.1%}** (umbral {threshold:.2f})")
                    st.progress(probability)
                else:
                    st.success(f"### MÓDULO LIMPIO")
                    st.write(f"Probabilidad de fallo: **{probability:.1%}** (umbral {threshold:.2f})")
                    st.progress(probability)
            except Exception as e:
                st.error(f"Error en la transformación de datos: {e}")
        else:
            st.warning("Modelo no cargado correctamente.")

elif page == "Predicción por Lotes (CSV)":
    st.header("Procesamiento por Lotes")
    st.write("Suba un archivo CSV para analizar múltiples módulos simultáneamente.")
    uploaded_file = st.file_uploader("Sube tu archivo CSV con métricas de código", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        with st.expander("Vista previa de datos cargados"):
            st.dataframe(df.head(), use_container_width=True)
        if main_model and scaler:
            feature_cols = [
                'LOC', 'CYCLO', 'LENGTH', 'VOLUME', 'DIFFICULTY',
                'INT_FAN_IN', 'INT_FAN_OUT', 'NUM_OPERATORS', 'NUM_OPERANDS', 'BRANCH_COUNT'
            ]
            if all(col in df.columns for col in feature_cols):
                X = df[feature_cols]
                X_scaled = scaler.transform(X)
                predictions = main_model.predict(X_scaled)
                probabilities = main_model.predict_proba(X_scaled)[:, 1]
                df['Resultado'] = ["Defectuoso" if p == 1 else "Limpio" for p in predictions]
                df['Probabilidad'] = [f"{prob:.1%}" for prob in probabilities]
                st.subheader("Resultados del Análisis")
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Descargar Reporte Completo", csv, "reporte_qa_defectos.csv", "text/csv")
            else:
                missing = [c for col in feature_cols if col not in df.columns]
                st.error(f"El archivo CSV debe contener las columnas: {feature_cols}. Faltan: {missing}")

elif page == "Comparación de Modelos":
    st.header("Comparación de Modelos (TASK-27)")

    if comparison_df is None or comparison_df.empty:
        st.warning("Datos de comparación no disponibles. Ejecute el notebook primero.")
        st.stop()

    st.subheader("Tabla de Métricas por Modelo")
    styled = comparison_df.style.background_gradient(cmap='viridis', subset=['PR-AUC', 'ROC-AUC', 'F1-Score', 'Recall'])
    st.dataframe(styled, use_container_width=True)

    if not all_models or not test_data:
        st.warning("Modelos individuales o datos de test no disponibles.")
        st.stop()

    X_test, y_test, feat_names = test_data
    st.subheader("Curvas ROC - Todos los Modelos")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    for name, m in all_models.items():
        y_proba = m.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Aleatorio')
    ax.set_xlabel('FPR', color='white')
    ax.set_ylabel('TPR', color='white')
    ax.set_title('Comparación de Curvas ROC', color='white')
    ax.tick_params(colors='white')
    ax.legend()
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)

    st.subheader("Curvas Precision-Recall - Todos los Modelos")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    for name, m in all_models.items():
        y_proba = m.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        prauc = average_precision_score(y_test, y_proba)
        ax.plot(rec, prec, lw=2, label=f'{name} (PR-AUC={prauc:.3f})')
    ax.set_xlabel('Recall', color='white')
    ax.set_ylabel('Precision', color='white')
    ax.set_title('Comparación de Curvas PR', color='white')
    ax.tick_params(colors='white')
    ax.legend()
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)

elif page == "Análisis del Modelo":
    st.header("Inteligencia del Modelo")

    if not all([eval_metrics, feature_imp, test_data]):
        st.warning("Datos de evaluación no disponibles. Ejecute el notebook primero.")
        st.stop()

    model_name = eval_metrics['model_name']
    st.info(f"**Modelo activo:** {model_name} | Entrenado con SMOTE + class_weight='balanced'")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Importancia de Características")
        imp_df = pd.DataFrame(list(feature_imp.items()), columns=['Feature', 'Importance'])
        imp_df = imp_df.sort_values('Importance', ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        sns.barplot(data=imp_df, y='Feature', x='Importance', palette='viridis', ax=ax)
        ax.set_xlabel('Importancia Relativa', color='white')
        ax.set_ylabel('', color='white')
        ax.tick_params(colors='white')
        st.pyplot(fig)

    with col2:
        st.subheader("Métricas del Modelo")
        metrics_df = pd.DataFrame([
            ('Accuracy', f"{eval_metrics['accuracy']:.4f}"),
            ('Precision', f"{eval_metrics['precision']:.4f}"),
            ('Recall', f"{eval_metrics['recall']:.4f}"),
            ('F1-Score', f"{eval_metrics['f1']:.4f}"),
            ('ROC-AUC', f"{eval_metrics['roc_auc']:.4f}"),
            ('PR-AUC', f"{eval_metrics['pr_auc']:.4f}"),
            ('MCC', f"{eval_metrics['mcc']:.4f}"),
            ('Log Loss', f"{eval_metrics['log_loss']:.4f}"),
        ], columns=['Métrica', 'Valor'])
        st.table(metrics_df)

    st.markdown("---")

    X_test, y_test, _ = test_data
    y_proba = main_model.predict_proba(X_test)[:, 1]
    y_pred = main_model.predict(X_test)

    tab1, tab2, tab3 = st.tabs(["Curva ROC", "Matriz de Confusión", "Curva Precision-Recall"])

    with tab1:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        ax.plot(fpr, tpr, color='#38bdf8', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Aleatorio')
        ax.set_xlabel('Tasa de Falsos Positivos (FPR)', color='white')
        ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)', color='white')
        ax.set_title('Curva ROC', color='white')
        ax.tick_params(colors='white')
        ax.legend()
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)

    with tab2:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Limpio', 'Defectuoso'],
                    yticklabels=['Limpio', 'Defectuoso'])
        ax.set_xlabel('Predicho', color='white')
        ax.set_ylabel('Real', color='white')
        ax.set_title('Matriz de Confusión', color='white')
        ax.tick_params(colors='white')
        st.pyplot(fig)
        tn, fp, fn, tp = cm.ravel()
        st.write(f"VN: {tn} | FP: {fp} | FN: {fn} | VP: {tp}")

    with tab3:
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        ax.plot(rec, prec, color='#818cf8', lw=2, label=f'PR (AUC = {pr_auc:.3f})')
        ax.set_xlabel('Recall', color='white')
        ax.set_ylabel('Precision', color='white')
        ax.set_title('Curva Precision-Recall', color='white')
        ax.tick_params(colors='white')
        ax.legend()
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Explicación de Características")
    st.write("""
    - **LOC**: Representa el tamaño del módulo. A mayor tamaño, mayor probabilidad de error.
    - **CYCLO**: Mide la complejidad lógica. Valores altos indican código difícil de testear.
    - **Halstead Metrics**: Analizan la riqueza del vocabulario y dificultad del algoritmo.
    """)
