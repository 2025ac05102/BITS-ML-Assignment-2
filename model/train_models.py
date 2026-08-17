from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'model'
MODEL_DIR.mkdir(exist_ok=True)

# Public dataset: Breast Cancer Wisconsin (Diagnostic), originally from UCI.
data = load_breast_cancer(as_frame=True)
X = data.data.copy()
y = data.target.copy()  # 0 = malignant, 1 = benign

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Save exactly the held-out test data used for evaluation.
test_df = X_test.copy()
test_df['target'] = y_test.values
test_df.to_csv(ROOT / 'test_data.csv', index=False)

models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=2000, random_state=42))
    ]),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'kNN': Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier(n_neighbors=5))
    ]),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight='balanced'
    ),
}

file_names = {
    'Logistic Regression': 'logistic_regression.joblib',
    'Decision Tree': 'decision_tree.joblib',
    'kNN': 'knn.joblib',
    'Naive Bayes': 'naive_bayes.joblib',
    'Random Forest': 'random_forest.joblib',
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        'ML Model Name': name,
        'Accuracy': accuracy_score(y_test, pred),
        'AUC': roc_auc_score(y_test, prob),
        'Precision': precision_score(y_test, pred, zero_division=0),
        'Recall': recall_score(y_test, pred, zero_division=0),
        'F1': f1_score(y_test, pred, zero_division=0),
        'MCC': matthews_corrcoef(y_test, pred),
    }
    results.append(metrics)
    joblib.dump(model, MODEL_DIR / file_names[name])

results_df = pd.DataFrame(results)
results_df.to_csv(ROOT / 'model_metrics.csv', index=False)
with open(ROOT / 'feature_names.json', 'w') as f:
    json.dump(list(X.columns), f, indent=2)

print(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
