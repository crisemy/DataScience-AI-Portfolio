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

st.set_page_config(page_title="QA-Cortex | Defect Predictor", layout="wide", page_icon="")

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def get_img_with_href(local_img_path):
    img_format = local_img_path.split('.')[-1]
    binary_data = get_base64_of_bin_file(local_img_path)
    return f'data:image/{img_format};base64,{binary_data}'

hero_img_base64 = ""
hero_path = APP_DIR / ".." / "assets" / "hero.png"
if hero_path.exists():
    hero_img_base64 = get_img_with_href(str(hero_path))

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Outfit', sans-serif; }}
    .main {{ background-color: #0f172a; color: #f8fafc; }}
    .header-banner {{
        background-image: linear-gradient(rgba(15,23,42,0.7), rgba(15,23,42,0.7)), url("{hero_img_base64}");
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

@st.cache_resource
def load_model():
    try:
        model = joblib.load(APP_DIR / 'model' / 'defect_prediction_model.pkl')
        scaler = joblib.load(APP_DIR / 'model' / 'scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error(f"No se encontraron los archivos del modelo en {APP_DIR / 'model/'}")
        return None, None

@st.cache_resource
def load_eval_data():
    try:
        metrics = joblib.load(APP_DIR / 'model' / 'metrics.pkl')
        importance = joblib.load(APP_DIR / 'model' / 'feature_importance.pkl')
        test_data = joblib.load(APP_DIR / 'model' / 'test_data.pkl')
        return metrics, importance, test_data
    except FileNotFoundError:
        return None, None, None

model, scaler = load_model()
eval_metrics, feature_imp, test_data = load_eval_data()

with st.sidebar:
    st.header("Navegación")
    page = st.selectbox("Ir a", ["Inicio", "Predicción Individual", "Predicción por Lotes (CSV)", "Análisis del Modelo"])
    st.markdown("---")
    model_name = eval_metrics.get('model_name', 'Desconocido') if eval_metrics else 'No disponible'
    st.info(f"**QA-Cortex v2.0**\n\nModelo: {model_name}\nUnidad 1 y 2\nMaestría en Ciencia de Datos e IA")

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
        if model and scaler:
            input_data = pd.DataFrame([[
                loc, cyclo, length, volume, difficulty,
                fan_in, fan_out, num_ops, num_opnds, branches
            ]], columns=[
                'LOC', 'CYCLO', 'LENGTH', 'VOLUME', 'DIFFICULTY',
                'INT_FAN_IN', 'INT_FAN_OUT', 'NUM_OPERATORS', 'NUM_OPERANDS', 'BRANCH_COUNT'
            ])
            try:
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1]
                st.markdown("---")
                if prediction == 1:
                    st.error(f"### MODULO DEFECTUOSO DETECTADO")
                    st.write(f"Existe un **{probability:.1%}** de probabilidad de que este modulo contenga errores criticos.")
                    st.progress(probability)
                else:
                    st.success(f"### MODULO LIMPIO")
                    st.write(f"El analisis indica que el modulo es estable (Probabilidad de fallo: **{probability:.1%}**).")
                    st.progress(probability)
            except Exception as e:
                st.error(f"Error en la transformacion de datos: {e}")
        else:
            st.warning("Modelo no cargado correctamente.")

elif page == "Predicción por Lotes (CSV)":
    st.header("Procesamiento por Lotes")
    st.write("Suba un archivo CSV para analizar multiples modulos simultaneamente.")
    uploaded_file = st.file_uploader("Sube tu archivo CSV con metricas de codigo", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        with st.expander("Vista previa de datos cargados"):
            st.dataframe(df.head(), use_container_width=True)
        if model and scaler:
            feature_cols = [
                'LOC', 'CYCLO', 'LENGTH', 'VOLUME', 'DIFFICULTY',
                'INT_FAN_IN', 'INT_FAN_OUT', 'NUM_OPERATORS', 'NUM_OPERANDS', 'BRANCH_COUNT'
            ]
            if all(col in df.columns for col in feature_cols):
                X = df[feature_cols]
                X_scaled = scaler.transform(X)
                predictions = model.predict(X_scaled)
                probabilities = model.predict_proba(X_scaled)[:, 1]
                df['Resultado'] = ["Defectuoso" if p == 1 else "Limpio" for p in predictions]
                df['Probabilidad'] = [f"{prob:.1%}" for prob in probabilities]
                st.subheader("Resultados del Analisis")
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Descargar Reporte Completo", csv, "reporte_qa_defectos.csv", "text/csv")
            else:
                missing = [c for col in feature_cols if col not in df.columns]
                st.error(f"El archivo CSV debe contener las columnas: {feature_cols}. Faltan: {missing}")

elif page == "Análisis del Modelo":
    st.header("Inteligencia del Modelo")

    if not all([eval_metrics, feature_imp, test_data]):
        st.warning("Datos de evaluacion no disponibles. Ejecute el notebook primero para generar los archivos necesarios.")
        st.stop()

    model_name = eval_metrics['model_name']

    st.info(f"**Modelo activo:** {model_name} | Entrenado con SMOTE + class_weight='balanced'")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Importancia de Caracteristicas")
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
        st.subheader("Metricas del Modelo")
        metrics_df = pd.DataFrame([
            ('Accuracy', f"{eval_metrics['accuracy']:.4f}"),
            ('Precision', f"{eval_metrics['precision']:.4f}"),
            ('Recall', f"{eval_metrics['recall']:.4f}"),
            ('F1-Score', f"{eval_metrics['f1']:.4f}"),
            ('ROC-AUC', f"{eval_metrics['roc_auc']:.4f}"),
            ('PR-AUC', f"{eval_metrics['pr_auc']:.4f}"),
            ('MCC', f"{eval_metrics['mcc']:.4f}"),
            ('Log Loss', f"{eval_metrics['log_loss']:.4f}"),
        ], columns=['Metrica', 'Valor'])
        st.table(metrics_df)

    st.markdown("---")

    X_test, y_test, feature_names = test_data
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    tab1, tab2, tab3 = st.tabs(["Curva ROC", "Matriz de Confusion", "Curva Precision-Recall"])

    with tab1:
        st.subheader("Curva ROC")
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
        st.subheader("Matriz de Confusion")
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Limpio', 'Defectuoso'],
                    yticklabels=['Limpio', 'Defectuoso'])
        ax.set_xlabel('Predicho', color='white')
        ax.set_ylabel('Real', color='white')
        ax.set_title('Matriz de Confusion', color='white')
        ax.tick_params(colors='white')
        st.pyplot(fig)

        tn, fp, fn, tp = cm.ravel()
        st.write(f"VN: {tn} | FP: {fp} | FN: {fn} | VP: {tp}")

    with tab3:
        st.subheader("Curva Precision-Recall")
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
    st.subheader("Explicacion de Caracteristicas")
    st.write("""
    - **LOC**: Representa el tamano del modulo. A mayor tamano, mayor probabilidad de error.
    - **CYCLO**: Mide la complejidad logica. Valores altos indican codigo dificil de testear.
    - **Halstead Metrics**: Analizan la riqueza del vocabulario y dificultad del algoritmo.
    """)
