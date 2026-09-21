# Data Science & Artificial Intelligence Portfolio

This repository contains my academic and applied work developed during my Master's Degree in Data Science and Artificial Intelligence, along with complementary projects created to strengthen my professional profile as a Data Scientist.

The purpose of this portfolio is to demonstrate my technical skills, analytical thinking, and end-to-end approach to data-driven problem solving, especially in scenarios where academic knowledge is translated into practical, real-world applications.

---

## Objectives of This Repository

- Showcase hands-on projects in Data Science, Machine Learning, and AI
- Demonstrate solid foundations in statistics, data analysis, and modeling
- Apply engineering best practices to data workflows
- Build a professional portfolio to support my transition into a Data Science role

---

## Topics Covered

- Data Analysis & Visualization
- Exploratory Data Analysis (EDA)
- Statistical Foundations
- Machine Learning (Regression, Classification, Clustering)
- Deep Learning (foundations and experiments)
- Feature Engineering & Model Evaluation
- Introductory MLOps concepts (pipelines, experiments, reproducibility)

---

## Repository Structure

```Here it is the list of the most prominent projects.
01-stressBot/           → StressBot: A Fuzzy Expert System for Stress Assessment
02-chessOpenings/       → ChessOpenings: Pattern Discovery and Clustering in Chess Openings
03-vaccinationCoverage/ → Case Study: DTP3 Vaccination Coverage (2020–2024) – Global Public Health
04-qaDefectPrediction/  → QA Defect Prediction using Machine Learning
05-neuronalScript/      → Neuronal Script: MLP Neural Network for Titanic Survival Prediction
```

---

## 05-neuronalScript: MLP Neural Network for Titanic Survival Prediction

A deep learning project that implements a Multi-Layer Perceptron (MLP) to predict passenger survival on the Titanic disaster. The model uses 9 preprocessed features (sex, passenger class, age, siblings/spouses, parents/children, fare, and one-hot encoded embarkation port) to classify whether a passenger survived or not.

### Project Overview

This notebook applies MLP fundamentals covered in the documentation:
- **Architecture**: Input layer (9 features) → Dense(32, relu) → Dropout(0.2) → Dense(16, relu) → Dropout(0.2) → Dense(1, sigmoid)
- **Preprocessing**: Sex mapping, median age filling, Embarked one-hot encoding with 3 columns
- **Training**: Adam optimizer, binary crossentropy loss, 30 epochs with 20% validation split
- **Regularization**: Dropout layers to prevent overfitting
- **Evaluation**: Test accuracy reporting and prediction visualization

### Dashboard

An interactive Streamlit dashboard visualizes the training process:

1. **Dataset Overview**: Shows the Titanic CSV structure and column names
2. **Train Model**: Builds and trains the neural network (takes ~30 epochs)
3. **Accuracy Evolution**: Interactive Plotly chart showing training vs validation accuracy across 30 epochs
   - *Training (blue)*: Model's performance on the 80% training split
   - *Validation (orange)*: Model's ability to generalize to unseen data
4. **Loss Evolution**: Interactive Plotly chart showing training vs validation loss
   - *Training (blue)*: Decreasing binary crossentropy error
   - *Validation (orange)*: Generalization metric; rising validation loss while training drops indicates overfitting
5. **Sample Predictions**: Displays sample predictions from the test set with survival probability thresholds

### Running the Dashboard

```bash
# Navigate to the project directory
cd 05.neuronalScript

# Install dependencies
pip install -r requirements.txt

# Start the Streamlit server
streamlit run app.py
```

The dashboard will be available at `http://localhost:8504` (or the port Streamlit assigns). Click **"Train Model"** in the sidebar to start the analysis and view the interactive training diagrams.

---

## Other Projects

- **01-stressBot/** → StressBot: A Fuzzy Expert System for Stress Assessment
- **02-chessOpenings/** → ChessOpenings: Pattern Discovery and Clustering in Chess Openings
- **03-vaccinationCoverage/** → Case Study: DTP3 Vaccination Coverage (2020–2024) – Global Public Health
- **04-qaDefectPrediction/** → QA Defect Prediction using Machine Learning