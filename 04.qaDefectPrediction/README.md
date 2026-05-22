# QA Defect Prediction - Modelo de Machine Learning

Proyecto desarrollado para la **Unidad 1 y 2** del curso de **Machine Learning y Deep Learning** (Maestría en Ciencia de Datos e IA).

**Objetivo**: Desarrollar un modelo de clasificación para predecir si un módulo de software es defectuoso o no, utilizando métricas estáticas de código.

## Descripción

Este proyecto implementa **5 modelos de clasificación binaria** para predecir defectos en módulos de software, con técnicas avanzadas de manejo de desbalanceo, validación rigurosa y explicabilidad.  
Es una aplicación práctica orientada a **QA Architect** y **IA Engineer**, útil para Risk-Based Testing y optimización de esfuerzos de testing.

### Características principales

- **5 modelos**: Logistic Regression, Random Forest, SVM (RBF), XGBoost, LightGBM
- **Manejo de desbalanceo**: SMOTE + class_weight='balanced'
- **Evaluación rigurosa**: Cross-Validation estratificada, GridSearchCV, curvas de aprendizaje
- **Métricas avanzadas**: PR-AUC, MCC, LogLoss, optimización de threshold (Youden's J)
- **Explicabilidad**: SHAP values (summary plot e individuales)
- **Dashboard interactivo** con Streamlit:
  - Predicción individual con selector de modelo y ajuste de umbral
  - Procesamiento por lotes (CSV)
  - Comparación de modelos (tabla + curvas ROC/PR)
  - Análisis del modelo (feature importance, curva ROC, matriz de confusión, PR curve)

---

## Tecnologías utilizadas

- Python 3.10+
- pandas, numpy, scikit-learn
- matplotlib + seaborn
- XGBoost, LightGBM
- imbalanced-learn (SMOTE)
- SHAP (explicabilidad)
- Streamlit (Dashboard)
- joblib (persistencia de modelo)

---

## Estructura del proyecto

```bash
qa-defect-prediction/
├── data/
│   └── SoftwareDefectDataset.csv
├── notebooks/
│   └── 01_defect_prediction.ipynb
├── app.py
├── requirements.txt
├── model/
│   ├── defect_prediction_model.pkl
│   ├── scaler.pkl
│   ├── test_data.pkl
│   ├── metrics.pkl
│   ├── feature_importance.pkl
│   ├── comparison_results.pkl
│   ├── model_logistic_regression.pkl
│   ├── model_random_forest.pkl
│   ├── model_svm_rbf.pkl
│   ├── model_xgboost.pkl
│   └── model_lightgbm.pkl
├── plan_implementacion.md
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

1. Instalar dependencias

```bash
pip install -r requirements.txt
```

1. Ejecutar el notebook (genera los modelos)

```bash
jupyter notebook notebooks/01_defect_prediction.ipynb
```

1. Ejecutar el Dashboard

```bash
cd 04.qaDefectPrediction
streamlit run app.py
```

---

## Dataset

- Se utiliza el dataset Software Defect Prediction disponible en Kaggle.
- Contiene métricas de código (LOC, complejidad ciclomática, Halstead, etc.) y la variable objetivo DEFECT_LABEL.
- Distribución: ~67% módulos limpios, ~33% defectuosos (desbalanceado).

---

## Modelos Implementados

| Modelo | Técnica de Desbalanceo | Hiperparámetros (GridSearch) |
| ------- | -------------------- | ----------------------------- |
| Logistic Regression | class_weight='balanced' + SMOTE | C, penalty (L1/L2) |
| Random Forest | class_weight='balanced' + SMOTE | n_estimators, max_depth, min_samples_split |
| SVM (RBF) | class_weight='balanced' + SMOTE | C, gamma |
| XGBoost | scale_pos_weight + SMOTE | n_estimators, learning_rate, max_depth |
| LightGBM | class_weight='balanced' + SMOTE | (por defecto) |

## Métricas de Evaluación

- **Accuracy**: Exactitud general
- **Precision**: Proporción de positivos correctos
- **Recall**: Proporción de defectos detectados
- **F1-Score**: Media armónica de Precision y Recall
- **ROC-AUC**: Área bajo la curva ROC
- **PR-AUC**: Área bajo la curva Precision-Recall (mejor para desbalanceo)
- **MCC**: Matthews Correlation Coefficient
- **Log Loss**: Pérdida logarítmica (calibración de probabilidades)

---

## Insights del modelo

- Las métricas de complejidad del código son las más predictivas (LOC, BRANCH_COUNT, NUM_OPERANDS).
- SMOTE combinado con class_weight mejora significativamente el recall de la clase defectuosa.
- El threshold óptimo (Youden's J) permite ajustar sensibilidad vs especificidad según necesidades de QA.
- SHAP revela que métricas de tamaño y complejidad aumentan el riesgo, mientras que métricas de estructura pueden reducirlo.

Autor: Cris N.- QA Architect
Materia: Machine Learning y Deep Learning
Fecha: Mayo 2026

---
