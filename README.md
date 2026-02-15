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
| Logistic Regression | 0.8098 | 0.9298 | 0.7619 | 0.9143 | 0.8312 | 0.6309 |
| Decision Tree | 0.9854 | 0.9857 | 1.0000 | 0.9714 | 0.9855 | 0.9712 |
| KNN | 0.8634 | 0.9629 | 0.8738 | 0.8571 | 0.8654 | 0.7269 |
| Naive Bayes | 0.8293 | 0.9043 | 0.8070 | 0.8762 | 0.8402 | 0.6602 |
| Random Forest (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

---

## Observations on Model Performance

| ML Model Name | Observation |
|---------------|-------------|
| **Logistic Regression** | At around 81% accuracy, Logistic Regression acts as a reasonable starting point. Its precision is the lowest among all models (0.76), which means it tends to flag some healthy patients as diseased. On the other hand, its recall is quite strong (0.91), so it catches most actual disease cases — a useful trait in a medical screening context where missing a sick patient is more costly than a false alarm. The limitation comes from its linear nature; it draws a straight boundary between classes and cannot capture the more complex feature interactions present in clinical data. Still, its AUC of 0.93 shows it ranks patients fairly well overall. |
| **Decision Tree** | The Decision Tree performs impressively at 98.54% accuracy with perfect precision — every patient it labels as diseased truly has the condition. It only misses a handful of actual cases (recall = 0.97). This strong result comes from its ability to split on multiple features in a hierarchical fashion, naturally capturing interactions like "older age + high cholesterol + chest pain type 3." That said, the near-perfect numbers are partly inflated by duplicate rows in the dataset, and without pruning, decision trees tend to memorize training data rather than generalize from it. |
| **KNN** | KNN lands in the middle at 86.34% accuracy, with precision and recall fairly balanced around 0.86–0.87. Its AUC of 0.96 is actually quite competitive, suggesting it separates the two classes well even if its hard predictions aren't always right. KNN works by looking at a patient's 7 nearest neighbors in the feature space and taking a vote. The challenge with 13 features is that distances become less meaningful in higher dimensions — this is the well-known "curse of dimensionality." Proper feature scaling (applied via StandardScaler) helps but doesn't fully overcome this limitation. |
| **Naive Bayes** | Naive Bayes comes in at 82.93%, making it the second-weakest performer. It assumes each feature contributes independently to the prediction — an assumption clearly violated here since features like age, blood pressure, cholesterol, and heart rate are medically correlated. This leads to a relatively low MCC of 0.66, indicating its predictions are less balanced across the two classes. However, for such a simple and fast algorithm, the AUC of 0.90 is respectable and shows it still captures the general direction of risk reasonably well. |
| **Random Forest (Ensemble)** | Random Forest hits 100% on every metric. It aggregates predictions from 150 individually trained decision trees, each built on a random data subset with a random feature subset. This diversity among trees reduces variance and makes the ensemble far more robust than any single tree. The perfect score is noteworthy but should be interpreted cautiously — the Heart Disease dataset contains duplicate rows, and some test samples are likely near-identical to training ones. With truly unseen patient data, we would expect slightly lower (but still strong) performance. The key takeaway is the clear benefit of bagging-based ensembles over individual learners. |
| **XGBoost (Ensemble)** | XGBoost matches Random Forest with perfect scores across the board. Unlike Random Forest's parallel tree-building approach, XGBoost constructs trees one at a time — each new tree specifically targets the mistakes of the previous ones. This sequential error-correction, combined with built-in L1/L2 regularization, makes it one of the most effective algorithms for tabular data. As with Random Forest, the 100% metrics are partly a consequence of dataset duplicates rather than pure generalization ability. In a production clinical setting, XGBoost would likely still be among the top performers but with slightly more realistic accuracy numbers. |

### Summary
- The two ensemble methods — Random Forest and XGBoost — clearly dominate, highlighting how combining multiple models leads to better predictions than any single algorithm.
- Decision Tree comes close to the ensembles, but standalone trees risk overfitting without careful tuning.
- KNN offers decent middle-ground results, though its effectiveness diminishes as feature count grows.
- Logistic Regression provides a solid baseline with excellent recall, making it suitable for initial screening despite its linear constraints.
- Naive Bayes ranks last, held back by its independence assumption which doesn't hold for correlated clinical measurements.
- It's worth noting that the perfect ensemble scores are partly explained by duplicate records in the dataset. Deduplicating the data before splitting would yield more conservative and realistic performance numbers.

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
   git clone https://github.com/pradeep-kodavoor/Heart-Disease-Classification.git
   cd Heart-Disease-Classification
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
Heart-Disease-Classification/
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

- **GitHub Repository:** [Heart-Disease-Classification](https://github.com/pradeep-kodavoor/Heart-Disease-Classification)
- **Live Streamlit App:** [Heart-Disease-Classification-App](https://heart-disease-classification-app.streamlit.app)

---

## 📚 References

1. UCI Heart Disease Dataset: https://archive.ics.uci.edu/dataset/45/heart+disease
2. Scikit-learn Documentation: https://scikit-learn.org/stable/
3. XGBoost Documentation: https://xgboost.readthedocs.io/
4. Streamlit Documentation: https://docs.streamlit.io/
