from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title='Breast Cancer Classification - ML Assignment 2', layout='wide')
ROOT = Path(__file__).resolve().parent

MODEL_FILES = {
    'Logistic Regression': 'model/logistic_regression.joblib',
    'Decision Tree': 'model/decision_tree.joblib',
    'kNN': 'model/knn.joblib',
    'Naive Bayes': 'model/naive_bayes.joblib',
    'Random Forest': 'model/random_forest.joblib',
}

st.title('Machine Learning Assignment 2 - Classification Model Explorer')
st.caption('Dataset: Breast Cancer Wisconsin (Diagnostic). Target: 0 = malignant, 1 = benign.')

st.sidebar.header('Controls')
model_name = st.sidebar.selectbox('Select model', list(MODEL_FILES.keys()))
uploaded = st.sidebar.file_uploader('Upload test CSV', type=['csv'])

st.info('Upload the supplied test_data.csv. It contains 30 feature columns plus the target column.')

if uploaded is not None:
    df = pd.read_csv(uploaded)
    if 'target' not in df.columns:
        st.error("The uploaded CSV must contain a 'target' column for evaluation.")
        st.stop()

    X = df.drop(columns=['target'])
    y = df['target']
    model = joblib.load(ROOT / MODEL_FILES[model_name])

    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y, pred),
        'AUC': roc_auc_score(y, prob),
        'Precision': precision_score(y, pred, zero_division=0),
        'Recall': recall_score(y, pred, zero_division=0),
        'F1 Score': f1_score(y, pred, zero_division=0),
        'MCC Score': matthews_corrcoef(y, pred),
    }

    st.subheader(f'Evaluation Metrics - {model_name}')
    cols = st.columns(3)
    for i, (label, value) in enumerate(metrics.items()):
        cols[i % 3].metric(label, f'{value:.4f}')

    left, right = st.columns(2)
    with left:
        st.subheader('Confusion Matrix')
        cm = confusion_matrix(y, pred)
        cm_df = pd.DataFrame(cm, index=['Actual Malignant (0)', 'Actual Benign (1)'],
                             columns=['Pred Malignant (0)', 'Pred Benign (1)'])
        st.dataframe(cm_df, use_container_width=True)

    with right:
        st.subheader('Classification Report')
        report = classification_report(y, pred, target_names=['Malignant', 'Benign'], output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

    st.subheader('Sample Predictions')
    preview = df.copy()
    preview['predicted_target'] = pred
    preview['probability_benign'] = prob
    st.dataframe(preview.head(20), use_container_width=True)
else:
    st.warning('Upload test_data.csv from the repository to display model results.')
