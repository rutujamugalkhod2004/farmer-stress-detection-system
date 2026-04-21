import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix

# ---------------------------
# LOAD DATA
# ---------------------------
data = pd.read_csv("farmer_stress_dataset.csv")

X = data.drop("stress_label", axis=1)
y = data["stress_label"]

# ---------------------------
# TRAIN TEST SPLIT
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------------------
# SCALING
# ---------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------
# TRAIN MODEL
# ---------------------------
model = RandomForestClassifier(n_estimators=150, max_depth=10)
model.fit(X_train, y_train)

# ---------------------------
# ACCURACY
# ---------------------------
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# ---------------------------
# SAVE MODEL
# ---------------------------
joblib.dump(model, "stress_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model saved successfully!")

# ---------------------------
# ANALYSIS FUNCTION (FIXED)
# ---------------------------
def analyze_stress(input_data, model, feature_names):
    importances = model.feature_importances_

    feature_impact = list(zip(feature_names, importances, input_data))
    feature_impact.sort(key=lambda x: x[1], reverse=True)

    print("\nAnalysis:")
    for feature, importance, value in feature_impact[:4]:
        print(f"- {feature} is influencing stress (value: {value})")

# ---------------------------
# PREDICTION FUNCTION (FIXED WARNING)
# ---------------------------
def predict_stress(input_data):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    if prediction == 0:
        return "Low Stress"
    elif prediction == 1:
        return "Medium Stress"
    else:
        return "High Stress"

# ---------------------------
# USER INPUT (WITH RANGES)
# ---------------------------
print("\nEnter Farmer Details (within valid ranges):\n")

heart_rate = float(input("Heart Rate (60–120): "))
hrv = float(input("HRV (20–100): "))
eda = float(input("EDA (0.1–1.0): "))
skin_temp = float(input("Skin Temperature (30–37): "))
pitch = float(input("Pitch (80–300): "))
energy = float(input("Energy (0.1–1.0): "))
work_hours = float(input("Work Hours (4–14): "))
financial_stress = int(input("Financial Stress (0 or 1): "))
weather_stress = int(input("Weather Stress (0 or 1): "))

# ---------------------------
# PREDICTION
# ---------------------------
input_data = [
    heart_rate, hrv, eda, skin_temp,
    pitch, energy, work_hours,
    financial_stress, weather_stress
]

result = predict_stress(input_data)
print("\nPredicted Stress Level:", result)

# ---------------------------
# ANALYSIS CALL
# ---------------------------
feature_names = X.columns
analyze_stress(input_data, model, feature_names)

# ===========================
# GRAPH SECTION
# ===========================

# ---------------------------
# ROC CURVE
# ---------------------------
probs = model.predict_proba(X_test)
probs_class1 = probs[:, 1]

fpr, tpr, _ = roc_curve(y_test, probs_class1, pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label="Random Forest (AUC = %.2f)" % roc_auc)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("roc_curve.png", dpi=300)
plt.close()

# ---------------------------
# FEATURE IMPORTANCE
# ---------------------------
plt.figure()
importances = model.feature_importances_

plt.barh(X.columns, importances)
plt.xlabel("Importance")
plt.title("Feature Importance")

plt.savefig("feature_importance.png", dpi=400)
plt.close()

# ---------------------------
# CONFUSION MATRIX
# ---------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")

plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

print("\nGraphs saved successfully!")