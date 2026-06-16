# Telco Customer Churn Prediction

## Project Overview

This project focuses on building a machine learning system to predict customer churn using the Telco Customer Churn dataset.

The objective is to identify customers who are likely to leave a telecom service, enabling proactive retention strategies and reducing revenue loss.

This project demonstrates an end-to-end machine learning pipeline including data analysis, preprocessing, model training, evaluation, and interpretation.

---

## Problem Statement

Telecom companies experience significant revenue loss due to customer churn.

The goal of this project is to predict whether a customer will churn based on historical account information, service usage, and billing data.

Type of problem: Binary classification  
Target variable: Churn (Yes / No)

---

## Dataset Description

- Dataset: Telco Customer Churn
- Number of rows: 7043
- Number of columns: 21

The dataset contains:
- Customer demographics
- Subscription services
- Account information
- Billing and payment details

---

## Machine Learning Pipeline

The project follows a structured machine learning pipeline:

Data Collection  
→ Exploratory Data Analysis (EDA)  
→ Data Cleaning  
→ Feature Engineering  
→ Encoding categorical variables  
→ Train/Test Split  
→ Feature Scaling  
→ Model Training  
→ Model Evaluation  
→ Model Persistence  

---

## Exploratory Data Analysis

Key observations:

- The dataset is imbalanced:
  - No churn ≈ 73%
  - Churn ≈ 27%

- Important patterns identified:
  - Short tenure customers are more likely to churn
  - Higher monthly charges increase churn probability
  - Month-to-month contracts show higher churn rates
  - Fiber optic internet users show higher churn

EDA was performed using:
- Distribution analysis
- Count plots
- Feature vs target comparisons

---

## Data Cleaning and Preprocessing

### Data Cleaning Steps

- Converted TotalCharges from object to numeric
- Handled missing values after conversion
- Removed invalid or missing records
- Removed customerID column

### Feature Encoding

- Applied One-Hot Encoding using pd.get_dummies(drop_first=True)
- Converted categorical variables into numerical format

### Target Encoding

- Churn:
  - Yes → 1
  - No → 0

---

## Feature Scaling

StandardScaler was used to normalize numerical features.

Scaling is required for:
- Logistic Regression
- K-Nearest Neighbors

Scaling is not required for:
- Decision Tree
- Random Forest
- Gradient Boosting

---

## Train-Test Split

Data was split using stratified sampling to maintain class distribution:

train_test_split(..., stratify=y)

This ensures that both training and testing sets preserve the original churn ratio.

---

## Models Trained

The following machine learning models were implemented:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Gradient Boosting

---

## Model Evaluation

Models were evaluated using multiple metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

### Key Insight

Accuracy alone is not sufficient due to class imbalance.  
ROC-AUC and F1-score provide a more reliable evaluation.

---

## Model Performance Summary

Logistic Regression achieved the best overall balance between performance and interpretability.

Decision Tree and ensemble models performed well but showed varying trade-offs in recall and precision.

KNN showed lower performance due to sensitivity to feature scaling and distance-based learning limitations.

---

## ROC Curve and AUC

The ROC curve evaluates model performance across classification thresholds.

Key definitions:

- True Positive Rate (Recall): correctly predicted churn cases
- False Positive Rate: incorrectly predicted churn cases

AUC (Area Under Curve):
- 0.5 → random model
- 1.0 → perfect model

---

## Feature Importance (Logistic Regression)

Key factors influencing churn:

- Fiber optic internet increases churn probability
- High monthly charges increase churn risk
- Short tenure strongly correlates with churn
- Month-to-month contracts are high risk

These insights are useful for business decision-making.

---

## Model Persistence

Trained model and preprocessing objects were saved using Joblib:

- churn_model.pkl
- scaler.pkl

These files allow:
- Reuse without retraining
- Consistent preprocessing during inference
- Deployment readiness

---

## Key Challenges

- Handling missing values hidden after type conversion
- Fixing scaler vs model loading confusion
- Managing categorical encoding complexity
- Ensuring consistent preprocessing pipeline
- Debugging model evaluation inconsistencies

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib
- Joblib

---

## Key Learnings

- End-to-end machine learning pipeline design
- Importance of proper data cleaning
- Handling imbalanced datasets
- Model comparison and evaluation strategies
- Interpretation of model outputs for business insights
- Importance of reproducibility in ML workflows

---

## Future Improvements

- Hyperparameter tuning using GridSearchCV
- Cross-validation for more robust evaluation
- Feature selection optimization
- Deployment using Streamlit or Flask
- API-based model serving using FastAPI

---

## How to Run the Project

Clone repository:

git clone https://github.com/wassilawess/Customer-Churn-Predictor.git

Install dependencies:

pip install -r requirements.txt

Run notebooks:

jupyter notebook

---

## Conclusion

This project demonstrates a complete machine learning workflow from raw data to a trained and interpretable predictive model.

The system is capable of identifying key drivers of customer churn and provides a strong foundation for deployment and production-level enhancement.