
import joblib
import numpy as np
import pandas as pd

model = joblib.load("model/best_model.joblib")
features = joblib.load("model/feature_names.joblib")
targets = joblib.load("model/target_names.joblib")

print("\n===================================")
print("     BREAST CANCER PREDICTION")
print("===================================")
print("Enter the 30 feature values.")
print("Use sample values only for testing.\n")

# Example values from the first dataset record
sample = [
    17.99, 10.38, 122.8, 1001.0, 0.1184,
    0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399,
    0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622,
    0.6656, 0.7119, 0.2654, 0.4601, 0.1189
]

try:
    values = []

    for i, feature in enumerate(features):
        value = input(
            f"{feature} [{sample[i]}]: "
        ).strip()

        values.append(float(value) if value else sample[i])

    patient = pd.DataFrame([values], columns=features)

    prediction = model.predict(patient)[0]
    probabilities = model.predict_proba(patient)[0]

    print("\n========== RESULT ==========")
    print("Predicted class:", targets[prediction].upper())

    print("\nClass probabilities:")

    for name, probability in zip(targets, probabilities):
        print(f"{name}: {probability * 100:.2f}%")

    print("============================")
    print("Educational demonstration only.")
    print("Not a medical diagnosis.")

except ValueError:
    print("\nInvalid input. Please enter numerical values.")
