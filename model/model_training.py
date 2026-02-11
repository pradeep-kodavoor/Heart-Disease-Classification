"""
ML Assignment 2 - Model Training Script
========================================
Dataset: Heart Disease Dataset (UCI / Kaggle)
URL: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
      or https://archive.ics.uci.edu/dataset/45/heart+disease

Binary Classification: Predict presence of heart disease (target: 0 or 1)
Features: 13 clinical attributes
Instances: 1025

How to use:
-----------
1. Download the dataset CSV from Kaggle (heart.csv)
2. Place it in the project root or model/ folder
3. Run this script: python model/model_training.py
4. Trained models will be saved as .pkl files in the model/ folder
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. LOAD DATASET
# ============================================================
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

# Update this path to where your heart.csv file is located
DATA_PATH = "heart.csv"

# Try multiple paths
for path in [DATA_PATH, "model/heart.csv", "../heart.csv"]:
    if os.path.exists(path):
        DATA_PATH = path
        break

df = pd.read_csv(DATA_PATH)

print(f"Dataset Shape: {df.shape}")
print(f"Number of Features: {df.shape[1] - 1}")
print(f"Number of Instances: {df.shape[0]}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nDataset Info:")
print(df.info())

print(f"\nClass Distribution:")
print(df['target'].value_counts())
print(f"\nMissing Values:\n{df.isnull().sum()}")

# ============================================================
# 2. DATA PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Data Preprocessing")
print("=" * 60)

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Feature names for later use
feature_names = X.columns.tolist()

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler and test data for the Streamlit app
MODEL_DIR = "model"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

# Save test data as CSV for Streamlit app upload demo
test_df = pd.DataFrame(X_test, columns=feature_names)
test_df['target'] = y_test.values
test_df.to_csv("test_data.csv", index=False)
print("Test data saved to test_data.csv (use this for Streamlit app demo)")

# ============================================================
# 3. MODEL TRAINING & EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Model Training & Evaluation")
print("=" * 60)

# Define all 6 models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "XGBoost (Ensemble)": XGBClassifier(
        n_estimators=100, random_state=42, use_label_encoder=False,
        eval_metric='logloss'
    ),
}

# Store results
results = []

for name, model in models.items():
    print(f"\n--- Training: {name} ---")
    
    # Use scaled data for models that need it
    if name in ["Logistic Regression", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate all 6 metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall_val = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "AUC": round(auc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall_val, 4),
        "F1 Score": round(f1, 4),
        "MCC": round(mcc, 4),
    })
    
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall_val:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  MCC:       {mcc:.4f}")
    
    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:\n{cm}")
    
    # Print classification report
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the trained model
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    model_path = os.path.join(MODEL_DIR, f"{safe_name}.pkl")
    joblib.dump(model, model_path)
    print(f"  Model saved to: {model_path}")

# ============================================================
# 4. COMPARISON TABLE
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Model Comparison Table")
print("=" * 60)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Save results for Streamlit app
results_df.to_csv(os.path.join(MODEL_DIR, "model_results.csv"), index=False)

# ============================================================
# 5. VISUALIZATION - Confusion Matrices
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Generating Visualizations")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, (name, model) in enumerate(models.items()):
    if name in ["Logistic Regression", "KNN"]:
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    axes[idx].set_title(f'{name}')
    axes[idx].set_ylabel('Actual')
    axes[idx].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrices.png"), dpi=150)
plt.show()
print("Confusion matrices saved to model/confusion_matrices.png")

# Bar chart comparison
fig, ax = plt.subplots(figsize=(14, 6))
metrics_to_plot = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
x = np.arange(len(results_df['Model']))
width = 0.12

for i, metric in enumerate(metrics_to_plot):
    ax.bar(x + i * width, results_df[metric], width, label=metric)

ax.set_xlabel('Models')
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x + width * 2.5)
ax.set_xticklabels(results_df['Model'], rotation=15, ha='right')
ax.legend()
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "model_comparison.png"), dpi=150)
plt.show()
print("Model comparison chart saved to model/model_comparison.png")

print("\n" + "=" * 60)
print("TRAINING COMPLETE! All models saved in the 'model/' folder.")
print("=" * 60)
