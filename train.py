
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Create folders
os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Loading Breast Cancer Dataset...")

# Load dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Dataset Shape:", X.shape)
print("Features:", len(data.feature_names))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Define models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

results = {}
best_model = None
best_accuracy = 0
best_model_name = ""

print("\nTraining Models...")

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy

    print(f"{name} Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=data.target_names
        )
    )

    # Save individual model
    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, f"model/{filename}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name
        best_predictions = predictions

# Save best model and metadata
joblib.dump(best_model, "model/best_model.joblib")
joblib.dump(data.feature_names, "model/feature_names.joblib")
joblib.dump(data.target_names, "model/target_names.joblib")

# Save accuracy comparison
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values())
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.savefig("results/accuracy_comparison.png")
plt.close()

# Confusion matrix for best model
cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=data.target_names,
    yticklabels=data.target_names
)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("results/confusion_matrix.png")
plt.close()

# Save classification report
report = classification_report(
    y_test,
    best_predictions,
    target_names=data.target_names
)

with open("results/classification_report.txt", "w") as f:
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Accuracy: {best_accuracy * 100:.2f}%\n\n")
    f.write(report)

print("\n------------------------------")
print("TRAINING COMPLETED SUCCESSFULLY")
print("------------------------------")
print("Best Model:", best_model_name)
print(f"Best Accuracy: {best_accuracy * 100:.2f}%")
print("Models saved in model/")
print("Results saved in results/")
