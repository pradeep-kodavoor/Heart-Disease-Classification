"""
ML Assignment 2 - Streamlit Web Application
=============================================
Heart Disease Classification - Interactive ML Dashboard

Features:
- CSV upload for test data
- Model selection dropdown
- Evaluation metrics display
- Confusion matrix visualization
- Classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

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
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Heart Disease ML Classifier",
    page_icon="❤️",
    layout="wide"
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
MODEL_DIR = "model"

# Model file mapping
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
    "XGBoost (Ensemble)": "xgboost_ensemble.pkl",
}

# Models that require scaled input
SCALED_MODELS = ["Logistic Regression", "KNN"]


@st.cache_resource
def load_model(model_name):
    """Load a saved model from disk."""
    model_file = MODEL_FILES[model_name]
    model_path = os.path.join(MODEL_DIR, model_file)
    return joblib.load(model_path)


@st.cache_resource
def load_scaler():
    """Load the saved StandardScaler."""
    return joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))


@st.cache_resource
def load_feature_names():
    """Load saved feature names."""
    return joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))


def compute_metrics(y_true, y_pred, y_prob):
    """Compute all 6 evaluation metrics."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC Score": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['No Disease (0)', 'Disease (1)'],
        yticklabels=['No Disease (0)', 'Disease (1)']
    )
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    return fig


# ============================================================
# MAIN APP
# ============================================================
def main():
    # --- Header ---
    st.title("❤️ Heart Disease Classification Dashboard")
    st.markdown("""
    This application demonstrates **6 Machine Learning classification models** 
    trained on the **Heart Disease Dataset** from UCI/Kaggle.  
    Upload your test data (CSV), select a model, and view the evaluation results.
    """)

    st.divider()

    # --- Sidebar ---
    st.sidebar.header("⚙️ Configuration")

    # Model Selection Dropdown
    selected_model = st.sidebar.selectbox(
        "🔍 Select ML Model",
        list(MODEL_FILES.keys()),
        help="Choose one of the 6 trained classification models"
    )

    st.sidebar.divider()
    st.sidebar.markdown("### 📊 Dataset Info")
    st.sidebar.markdown("""
    **Dataset:** Heart Disease (UCI)  
    **Features:** 13 clinical attributes  
    **Target:** Binary (0 = No Disease, 1 = Disease)  
    **Train/Test Split:** 80/20  
    """)

    st.sidebar.divider()
    st.sidebar.markdown("### 📋 Feature Descriptions")
    st.sidebar.markdown("""
    | Feature | Description |
    |---------|-------------|
    | age | Age in years |
    | sex | Sex (1=male, 0=female) |
    | cp | Chest pain type (0-3) |
    | trestbps | Resting blood pressure |
    | chol | Serum cholesterol (mg/dl) |
    | fbs | Fasting blood sugar > 120 |
    | restecg | Resting ECG results |
    | thalach | Max heart rate achieved |
    | exang | Exercise induced angina |
    | oldpeak | ST depression |
    | slope | Slope of peak ST segment |
    | ca | Number of major vessels |
    | thal | Thalassemia type |
    """)

    # --- File Upload ---
    st.header("📁 Upload Test Data")
    st.markdown("""
    Upload a CSV file containing test data. The CSV should have the same 
    features as the training data plus a `target` column for evaluation.  
    *You can use the `test_data.csv` generated during training.*
    """)

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload test data with features and target column"
    )

    if uploaded_file is not None:
        # Read uploaded data
        try:
            data = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Shape: {data.shape}")

            # Show preview
            with st.expander("🔎 Preview Uploaded Data", expanded=False):
                st.dataframe(data.head(10), use_container_width=True)

            # Check if target column exists
            if 'target' not in data.columns:
                st.error("❌ The uploaded CSV must contain a 'target' column for evaluation.")
                return

            # Load required assets
            feature_names = load_feature_names()
            scaler = load_scaler()

            # Separate features and target
            X_test = data[feature_names]
            y_test = data['target']

            st.info(f"**Model Selected:** {selected_model}")

            # Load the selected model
            model = load_model(selected_model)

            # Prepare input (scale if needed)
            if selected_model in SCALED_MODELS:
                X_input = scaler.transform(X_test)
            else:
                X_input = X_test

            # Make predictions
            y_pred = model.predict(X_input)
            y_prob = model.predict_proba(X_input)[:, 1]

            st.divider()

            # --- Evaluation Metrics ---
            st.header(f"📈 Evaluation Metrics — {selected_model}")

            metrics = compute_metrics(y_test, y_pred, y_prob)

            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)

            col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            col2.metric("AUC Score", f"{metrics['AUC Score']:.4f}")
            col3.metric("Precision", f"{metrics['Precision']:.4f}")
            col4.metric("Recall", f"{metrics['Recall']:.4f}")
            col5.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
            col6.metric("MCC", f"{metrics['MCC']:.4f}")

            st.divider()

            # --- Confusion Matrix ---
            st.header(f"🔢 Confusion Matrix — {selected_model}")
            col_cm, col_cr = st.columns(2)

            with col_cm:
                fig = plot_confusion_matrix(y_test, y_pred, selected_model)
                st.pyplot(fig)

            with col_cr:
                st.subheader("📝 Classification Report")
                report = classification_report(
                    y_test, y_pred,
                    target_names=['No Disease (0)', 'Disease (1)'],
                    output_dict=True
                )
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

            st.divider()

            # --- All Models Comparison ---
            st.header("📊 All Models Comparison")

            # Load pre-computed results if available
            results_path = os.path.join(MODEL_DIR, "model_results.csv")
            if os.path.exists(results_path):
                all_results = pd.read_csv(results_path)
                st.dataframe(
                    all_results.style.highlight_max(
                        subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC'],
                        color='lightgreen'
                    ),
                    use_container_width=True
                )
            else:
                # Compute on-the-fly for all models
                all_results = []
                for name in MODEL_FILES.keys():
                    m = load_model(name)
                    if name in SCALED_MODELS:
                        x_in = scaler.transform(X_test)
                    else:
                        x_in = X_test
                    yp = m.predict(x_in)
                    ypr = m.predict_proba(x_in)[:, 1]
                    mets = compute_metrics(y_test, yp, ypr)
                    mets["Model"] = name
                    all_results.append(mets)

                all_results_df = pd.DataFrame(all_results)
                cols = ["Model"] + [c for c in all_results_df.columns if c != "Model"]
                all_results_df = all_results_df[cols]
                st.dataframe(
                    all_results_df.style.highlight_max(
                        subset=['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC'],
                        color='lightgreen'
                    ),
                    use_container_width=True
                )

            # --- Predictions Preview ---
            st.divider()
            st.header("🔮 Predictions Preview")
            pred_df = data.copy()
            pred_df['Predicted'] = y_pred
            pred_df['Probability'] = y_prob.round(4)
            pred_df['Correct'] = (pred_df['target'] == pred_df['Predicted']).map(
                {True: '✅', False: '❌'}
            )
            st.dataframe(pred_df.head(20), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)

    else:
        # Show instructions when no file is uploaded
        st.info("👆 Please upload a test CSV file to get started.")

        st.markdown("### 📌 Expected CSV Format")
        st.markdown("""
        Your CSV should contain these columns:
        
        `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target`
        
        The `target` column (0 or 1) is required for computing evaluation metrics.
        """)

        # Show sample data
        sample_data = {
            'age': [63, 37, 41, 56, 57],
            'sex': [1, 1, 0, 1, 0],
            'cp': [3, 2, 1, 1, 0],
            'trestbps': [145, 130, 130, 120, 120],
            'chol': [233, 250, 204, 236, 354],
            'fbs': [1, 0, 0, 0, 0],
            'restecg': [0, 1, 0, 1, 1],
            'thalach': [150, 187, 172, 178, 163],
            'exang': [0, 0, 0, 0, 1],
            'oldpeak': [2.3, 3.5, 1.4, 0.8, 0.6],
            'slope': [0, 0, 2, 2, 2],
            'ca': [0, 0, 0, 0, 0],
            'thal': [1, 2, 2, 2, 2],
            'target': [1, 1, 1, 1, 1],
        }
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)


if __name__ == "__main__":
    main()
