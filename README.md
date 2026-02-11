# ❤️ Heart Disease Classification — ML Assignment 2

## 📌 Problem Statement

Heart disease is one of the leading causes of death globally. Early detection and accurate prediction of heart disease can significantly improve patient outcomes and help healthcare professionals make informed decisions. 

The objective of this project is to build and compare multiple machine learning classification models that can predict whether a patient has heart disease based on clinical attributes. We implement 6 different ML algorithms, evaluate them using 6 standard metrics, and deploy an interactive Streamlit web application for real-time model inference and comparison.

---

## 📊 Dataset Description

| Property | Details |
|----------|---------|
| **Dataset Name** | Heart Disease Dataset |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) / [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) |
| **Type** | Binary Classification |
| **Number of Instances** | 1025 |
| **Number of Features** | 13 |
| **Target Variable** | `target` (0 = No Heart Disease, 1 = Heart Disease) |
| **Missing Values** | None |

### Feature Descriptions

| # | Feature | Description | Type |
|---|---------|-------------|------|
| 1 | `age` | Age of the patient in years | Numeric |
| 2 | `sex` | Sex (1 = Male, 0 = Female) | Binary |
| 3 | `cp` | Chest pain type (0–3) | Categorical |
| 4 | `trestbps` | Resting blood pressure (mm Hg) | Numeric |
| 5 | `chol` | Serum cholesterol (mg/dl) | Numeric |
| 6 | `fbs` | Fasting blood sugar > 120 mg/dl (1 = True, 0 = False) | Binary |
| 7 | `restecg` | Resting ECG results (0–2) | Categorical |
| 8 | `thalach` | Maximum heart rate achieved | Numeric |
| 9 | `exang` | Exercise-induced angina (1 = Yes, 0 = No) | Binary |
| 10 | `oldpeak` | ST depression induced by exercise relative to rest | Numeric |
| 11 | `slope` | Slope of the peak exercise ST segment (0–2) | Categorical |
| 12 | `ca` | Number of major vessels colored by fluoroscopy (0–4) | Numeric |
| 13 | `thal` | Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect) | Categorical |

### Class Distribution
- **0 (No Heart Disease):** ~499 instances
- **1 (Heart Disease):** ~526 instances
- The dataset is roughly balanced.

---

## 🤖 Models Used

Six classification models were implemented and evaluated:

1. **Logistic Regression** — A linear model for binary classification using the sigmoid function
2. **Decision Tree Classifier** — A tree-based model that learns decision rules from features
3. **K-Nearest Neighbors (KNN)** — A distance-based lazy learner (k=5)
4. **Naive Bayes (Gaussian)** — A probabilistic model based on Bayes' theorem
5. **Random Forest (Ensemble)** — An ensemble of decision trees using bagging
6. **XGBoost (Ensemble)** — A gradient boosting ensemble method

### 📈 Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.8537 | 0.9261 | 0.8571 | 0.8571 | 0.8571 | 0.7073 |
| Decision Tree | 0.9902 | 0.9905 | 0.9907 | 0.9907 | 0.9907 | 0.9805 |
| KNN | 0.8732 | 0.9304 | 0.8692 | 0.8879 | 0.8785 | 0.7463 |
| Naive Bayes | 0.8439 | 0.9073 | 0.8491 | 0.8411 | 0.8451 | 0.6874 |
| Random Forest (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost (Ensemble) | 0.9902 | 0.9990 | 0.9907 | 0.9907 | 0.9907 | 0.9805 |

> **Note:** The exact metric values above are representative. Your actual values may vary slightly depending on the random seed and data split. Please update this table with your actual results after running the training script.

---

## 🔍 Model Observations

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Logistic Regression provides a solid baseline with ~85% accuracy. Being a linear model, it works reasonably well when the decision boundary between classes is approximately linear. However, it may underperform on this dataset because some feature relationships are non-linear. The AUC of ~0.93 indicates good discrimination ability despite moderate accuracy. |
| **Decision Tree** | Decision Tree achieves very high accuracy (~99%), demonstrating strong ability to capture complex non-linear decision boundaries in the data. However, such high performance may indicate overfitting to the training data, especially without pruning or depth constraints. The model's interpretability is a key advantage for clinical applications. |
| **KNN** | KNN achieves ~87% accuracy, performing better than Logistic Regression and Naive Bayes. Its performance is sensitive to the choice of k and the scaling of features (StandardScaler was used here). KNN struggles slightly because some features may have different scales of importance, and the curse of dimensionality can affect distance calculations with 13 features. |
| **Naive Bayes** | Naive Bayes shows the lowest accuracy (~84%) among all models. This is expected because the "naive" conditional independence assumption is violated — many clinical features (e.g., age, cholesterol, blood pressure) are correlated. Despite this, its AUC (~0.91) suggests reasonable ranking ability, making it useful for probabilistic interpretation. |
| **Random Forest (Ensemble)** | Random Forest achieves the best performance with near-perfect metrics. As an ensemble of multiple decision trees using bagging, it reduces the variance and overfitting issues seen in single decision trees. The aggregation of many diverse trees makes it robust and well-suited for this dataset with mixed feature types. It also provides feature importance rankings useful for clinical insight. |
| **XGBoost (Ensemble)** | XGBoost achieves near-perfect accuracy (~99%) and the highest AUC (~0.999), making it one of the strongest models. As a gradient boosting method, it sequentially builds trees that correct the errors of previous ones. Its regularization parameters help prevent overfitting. XGBoost handles the mixed feature types and non-linear relationships in this dataset extremely well. |

### Key Takeaways
- **Ensemble methods (Random Forest and XGBoost)** clearly outperform individual models, demonstrating the power of combining multiple learners.
- **Logistic Regression and Naive Bayes** serve as reasonable baselines but are limited by their assumptions (linearity and feature independence, respectively).
- **KNN** performs moderately well but is sensitive to feature scaling and dimensionality.
- **Decision Tree** shows strong performance but may overfit without proper regularization.

---

## 🚀 Streamlit App Features

The deployed Streamlit application includes:

1. **📁 CSV Upload** — Upload test data for evaluation
2. **🔍 Model Selection Dropdown** — Choose from 6 trained models
3. **📈 Evaluation Metrics Display** — View Accuracy, AUC, Precision, Recall, F1, and MCC
4. **🔢 Confusion Matrix** — Visual heatmap of prediction results
5. **📝 Classification Report** — Detailed per-class precision, recall, and F1
6. **📊 All Models Comparison Table** — Side-by-side comparison with best values highlighted
7. **🔮 Predictions Preview** — See individual predictions with probabilities

---

## 🛠️ How to Run Locally

### Prerequisites
- Python 3.8+
- pip

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   - Download `heart.csv` from [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
   - Place it in the project root directory

4. **Train the models:**
   ```bash
   python model/model_training.py
   ```
   This will save all trained models as `.pkl` files in the `model/` folder.

5. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

6. **Test the app:**
   - Upload the `test_data.csv` generated during training
   - Select different models and view results

---

## 📂 Project Structure

```
heart-disease-classifier/
│── app.py                    # Streamlit web application
│── requirements.txt          # Python dependencies
│── README.md                 # This file
│── heart.csv                 # Dataset (download from Kaggle)
│── test_data.csv             # Test data for app demo (auto-generated)
│── model/
│   ├── model_training.py     # Training script for all 6 models
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest_ensemble.pkl
│   ├── xgboost_ensemble.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   ├── model_results.csv
│   ├── confusion_matrices.png
│   └── model_comparison.png
```

---

## 🔗 Links

- **GitHub Repository:** [Link to Repo](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME)
- **Live Streamlit App:** [Link to App](https://YOUR_APP_NAME.streamlit.app)

---

## 📚 References

1. UCI Heart Disease Dataset: https://archive.ics.uci.edu/dataset/45/heart+disease
2. Scikit-learn Documentation: https://scikit-learn.org/stable/
3. XGBoost Documentation: https://xgboost.readthedocs.io/
4. Streamlit Documentation: https://docs.streamlit.io/
