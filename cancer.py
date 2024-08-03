# training_model.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import shap

# Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Exploratory Data Analysis (EDA)
print("First 5 rows of data:")
print(X.head())

print("\nClass distribution:")
print(y.value_counts())

# Visualizing the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(X.corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Advanced Feature Engineering
# Adding polynomial features or other transformations could go here.
# For this example, we are using the raw features directly.

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Comparison and Hyperparameter Tuning
models = {
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'SVC': SVC(probability=True)
}

param_grid = {
    'RandomForest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
    'GradientBoosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
    'SVC': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
}

best_models = {}
best_scores = {}

for model_name in models.keys():
    print(f"\nTuning {model_name}...")
    clf = GridSearchCV(models[model_name], param_grid[model_name], cv=5, scoring='accuracy', n_jobs=-1)
    clf.fit(X_train, y_train)
    best_models[model_name] = clf.best_estimator_
    best_scores[model_name] = clf.best_score_
    print(f"Best parameters for {model_name}: {clf.best_params_}")
    print(f"Best cross-validation score: {clf.best_score_}")

# Select the best model based on cross-validation performance
best_model_name = max(best_scores, key=best_scores.get)
best_model = best_models[best_model_name]

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
print(f"\nBest model: {best_model_name}")
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Model Explainability using SHAP
explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_test)
model_filename = 'best_model.pkl'
joblib.dump(best_model, model_filename)


print(f"\nModel saved as '{model_filename}'")
