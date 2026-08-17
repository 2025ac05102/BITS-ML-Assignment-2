# Machine Learning Assignment 2 - Classification Model Explorer

## a. Problem Statement
The objective is to compare multiple supervised classification algorithms on one public dataset and expose the trained models through an interactive Streamlit application. The application allows a user to upload held-out test data, select a model, and inspect evaluation metrics and a confusion matrix/classification report.

## b. Dataset Description
**Dataset:** Breast Cancer Wisconsin (Diagnostic), originally from the UCI Machine Learning Repository and available through scikit-learn.

- Problem type: Binary classification
- Instances: 569
- Input features: 30 numeric features
- Target classes: malignant (0) and benign (1)
- Train/test split: 80/20, stratified, `random_state=42`
- Held-out test rows: 114

The dataset satisfies the assignment minimum of 500 instances and 12 features.

## c. GitHub Repository Link
**GitHub Repository:** `PASTE_YOUR_GITHUB_REPOSITORY_LINK_HERE`

## d. Models Used and Evaluation Metrics
Five models are implemented because the assignment's enumerated model list and comparison table name five models, although one sentence refers to "6 ML models". The implementation follows the explicitly listed models.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| kNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) | 0.9474 | 0.9937 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |

### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall result on the held-out test set. It achieved the highest accuracy, F1 and MCC, while also producing the highest AUC. Standardization helps the linear classifier handle features measured on different scales. |
| Decision Tree | Lowest overall performance. The tree captures non-linear decision rules but is more sensitive to the particular train/test split and can overfit compared with the other methods. |
| kNN | Strong performance after standardization. Recall is particularly high, showing that most benign samples are correctly identified, but it is slightly weaker than Logistic Regression overall. |
| Naive Bayes | Good AUC and recall despite its conditional-independence assumption. Its lower accuracy/MCC indicates that the independence assumption is not fully appropriate for the correlated diagnostic measurements. |
| Random Forest (Ensemble) | Strong and balanced performance with excellent AUC. Ensembling reduces the instability of a single decision tree, though on this split it does not exceed Logistic Regression. |
| **Overall Winner** | **Logistic Regression**, because it has the highest Accuracy (0.9825), AUC (0.9954), F1 (0.9861) and MCC (0.9623) on the held-out test data. |

## Streamlit Application
**Live App:** `PASTE_YOUR_STREAMLIT_APP_LINK_HERE`

The app provides:
1. CSV test-data upload
2. Model-selection dropdown
3. Accuracy, AUC, Precision, Recall, F1 and MCC
4. Confusion matrix and classification report
5. Sample predictions

## Repository Structure
```text
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- model_metrics.csv
|-- feature_names.json
|-- model/
    |-- train_models.py
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- naive_bayes.joblib
    |-- random_forest.joblib
```

## How to Run Locally
```bash
pip install -r requirements.txt
python model/train_models.py
streamlit run app.py
```

Then upload `test_data.csv` in the sidebar and choose a model.

## Reproducibility
All models use the same dataset and held-out test split. The split is stratified and uses `random_state=42`. Logistic Regression and kNN use `StandardScaler` inside a pipeline to avoid data leakage.

## BITS Virtual Lab Evidence
Add the required screenshot showing successful assignment execution on BITS Virtual Lab to the final submission PDF.
