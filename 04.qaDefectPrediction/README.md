# QA Defect Prediction - Modelo de Machine Learning

Proyecto desarrollado para la **Unidad 1** del curso de **Machine Learning y Deep Learning** (Maestría en Ciencia de Datos e IA).

**Objetivo**: Desarrollar un modelo de clasificación para predecir si un módulo de software es defectuoso o no, utilizando métricas estáticas de código.

## Descripción

Este proyecto implementa un modelo de **Clasificación Binaria** usando **Logistic Regression** (scikit-learn) para predecir defectos en módulos de software.  
Es una aplicación práctica orientada a **QA Architect** y **IA Engineer**, útil para Risk-Based Testing y optimización de esfuerzos de testing.

### Características principales
- Clasificación binaria (defectuoso / limpio)
- Análisis exploratorio de datos
- Entrenamiento y evaluación del modelo
- Dashboard interactivo con **Streamlit**
- Feature Importance

--- 

## Tecnologías utilizadas

- Python 3.10+
- pandas, numpy, scikit-learn
- matplotlib + seaborn
- Streamlit (Dashboard)
- joblib (persistencia de modelo)

---

## Estructura del proyecto

```bash
qa-defect-prediction/
├── data/
│   └── software_defect_data.csv
├── notebooks/
│   └── 01_defect_prediction.ipynb
├── src/
│   └── defect_model.py
├── app.py
├── requirements.txt
├── model/
│   ├── defect_prediction_model.pkl
│   └── scaler.pkl
├── README.md
└── .venv/
```

### Cómo ejecutar
1. Clonar / Descargar el proyecto
2. Crear entorno virtual
```bash
python -m venv .venv
source .venv/bin/activate        # Linux / Mac
.venv\Scripts\activate         # Windows
```
3. Instalar dependencias
```bash
pip install -r requirements.txt
```
4. Ejecutar el Dashboard
```bash
streamlit run app.py
```
---

## Dataset

- Se utiliza el dataset Software Defect Prediction disponible en Kaggle.
- Contiene métricas de código (LOC, complejidad ciclomática, Halstead, etc.) y la variable objetivo defects.

---

## Insights del modelo

- Las métricas de complejidad del código son las más predictivas.
- Permite priorizar módulos de alto riesgo en campañas de testing.
- Baseline interpretable y listo para producción.

Autor: Cris N.- QA Architect
Materia: Machine Learning y Deep Learning
Fecha: Mayo 2026

---
