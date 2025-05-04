# 🎯 Student Performance Prediction using Machine Learning

Welcome! This project is a **full-cycle ML solution** that predicts students' math scores using demographic and academic features. It demonstrates my ability to build **real-world machine learning systems** — from data preprocessing to model evaluation — and is designed with clarity, performance, and interpretability in mind.

---

## 🚀 Project Goals

- Predict math scores based on features like gender, parental education, and reading/writing scores.
- Apply, compare, and evaluate a wide range of regression models.
- Showcase skills across **EDA**, **feature engineering**, **ML modeling**, and **performance tuning**.

---

## 🧠 What I Did

✅ **Exploratory Data Analysis**  
✔️ Identified patterns in performance by gender, parental education, and test prep  
✔️ Visualized distributions using histograms and pairplots

✅ **Feature Engineering**  
✔️ Created `total score` and `average` score columns  
✔️ Classified high/low performers for possible classification use case (next version)

✅ **Data Preprocessing**  
✔️ One-hot encoded categorical features  
✔️ Standard scaled numeric features using `ColumnTransformer`

✅ **Model Building & Evaluation**  
✔️ Trained and evaluated 9 regression models:  
   - Linear Regression, Ridge, Lasso  
   - Random Forest, Decision Tree, KNN  
   - XGBoost, CatBoost, AdaBoost

✔️ Evaluated using:
   - MAE, RMSE, R² Score  
   - Compared training vs testing scores to analyze overfitting

✅ **Best Model**:  
✔️ **Ridge Regression** with **88.06% R² score** on unseen test data

✅ **Visualizations**  
✔️ Actual vs Predicted score scatter plots  
✔️ Distribution plots with gender-based analysis

---

## 🔧 Tech Stack & Tools

| Area | Tools & Libraries |
|------|-------------------|
| Language | Python 3.8+ |
| IDE | VS Code, Jupyter |
| EDA & Visualization | pandas, numpy, seaborn, matplotlib |
| Modeling | scikit-learn, XGBoost, CatBoost |
| Preprocessing | OneHotEncoder, StandardScaler, ColumnTransformer |
| Evaluation | MAE, RMSE, R² from `sklearn.metrics` |
| Version Control | Git, GitHub |

---

## 📈 Key Results

| Model                  | R² Score (Test) |
|------------------------|-----------------|
| ✅ Ridge Regression     | **0.8806**       |
| Linear Regression      | 0.8792          |
| Random Forest Regressor| 0.8520          |
| CatBoost Regressor     | 0.8516          |
| AdaBoost Regressor     | 0.8471          |


