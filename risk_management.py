import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)
import io

st.set_page_config(page_title="Drug Overdose Prediction App", layout="centered")

st.title("💊 Drug Overdose Prediction App")

# 1. Data & Model Setup (Same as before)
data = {
    "age": [25, 40, 33, 50, 22, 60, 29, 45, 38, 55],
    "weight": [70, 85, 60, 95, 65, 80, 72, 78, 68, 90],
    "dosage": [15, 20, 12, 25, 10, 30, 18, 22, 14, 28],
    "duration": [5, 10, 3, 7, 4, 15, 6, 8, 5, 12],
    "symptoms": [
        "headache",
        "fainting",
        "dizziness",
        "chest pain",
        "nausea",
        "shortness of breath",
        "headache",
        "confusion",
        "fainting",
        "nausea",
    ],
    "overdose": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
}

df = pd.DataFrame(data)
df_encoded = pd.get_dummies(df, columns=["symptoms"], drop_first=True)
X = df_encoded.drop("overdose", axis=1)
y = df_encoded["overdose"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

# --- User Inputs ---
st.header("👤 Patient Data Input")
with st.form("patient_form", clear_on_submit=False):
    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=30,
        help="Enter patient age in years",
    )
    weight = st.number_input(
        "Weight (kg)",
        min_value=1,
        max_value=300,
        value=70,
        help="Enter patient weight in kilograms",
    )
    dosage = st.number_input(
        "Dosage (mg)",
        min_value=1,
        max_value=500,
        value=20,
        help="Current drug dosage in milligrams",
    )
    duration = st.number_input(
        "Duration (days)",
        min_value=1,
        max_value=365,
        value=5,
        help="Duration of drug usage in days",
    )
    symptom_options = df["symptoms"].unique().tolist()
    symptom = st.selectbox("Primary Symptom", symptom_options)

    submitted = st.form_submit_button("Predict Overdose Risk")

if not submitted:
    st.info("Fill out patient details and click 'Predict Overdose Risk'.")
    st.stop()

# Prepare input data for prediction
input_dict = {
    "age": [age],
    "weight": [weight],
    "dosage": [dosage],
    "duration": [duration],
}

# Set symptom dummies
for col in X.columns:
    if col.startswith("symptoms_"):
        input_dict[col] = [0]

symptom_col = f"symptoms_{symptom}"
if symptom_col in input_dict:
    input_dict[symptom_col] = [1]

input_df = pd.DataFrame(input_dict)
input_scaled = scaler.transform(input_df)

# Prediction & Probability
prediction = model.predict(input_scaled)[0]
prediction_proba = model.predict_proba(input_scaled)[0][1]

# --- 3. Prediction Probability Visualization ---
st.header("📊 Prediction Result & Risk Level")

# Risk categorization
def risk_category(prob):
    if prob < 0.3:
        return "Low Risk", "#4CAF50"  # green
    elif prob < 0.7:
        return "Medium Risk", "#FFC107"  # amber
    else:
        return "High Risk", "#F44336"  # red

risk_level, risk_color = risk_category(prediction_proba)

# Show risk level with color and emoji
st.markdown(
    f"<h3 style='color:{risk_color};'>"
    f"{'✔️' if prediction==0 else '⚠️'} {risk_level} Detected "
    f"(Probability: {prediction_proba:.2f})"
    f"</h3>",
    unsafe_allow_html=True,
)

# Progress bar for risk probability
st.progress(min(max(prediction_proba, 0), 1))

# --- 6. Patient Data Summary with ability to edit ---
st.header("📝 Patient Data Summary")
st.dataframe(input_df.T.rename(columns={0: "Value"}))

# --- 2. Feature Importances + Explanation ---
st.header("🔍 Model Insights & Feature Importance")

importances = pd.DataFrame(
    model.feature_importances_, index=X.columns, columns=["Importance"]
).sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(8, 4))
importances.plot(kind="barh", legend=False, ax=ax, color="#1976D2")
ax.set_xlabel("Importance")
ax.set_ylabel("Features")
ax.set_title("Feature Importances")
st.pyplot(fig)

# Simple feature explanation dictionary
feature_explanations = {
    "age": "Age of patient (years).",
    "weight": "Body weight (kg).",
    "dosage": "Dosage of drug (mg).",
    "duration": "Duration of drug use (days).",
}
for col in X.columns:
    if col.startswith("symptoms_"):
        feature_explanations[col] = f"Symptom: {col.replace('symptoms_', '').capitalize()}"

st.markdown("*Feature Explanation:*")
for feat, desc in feature_explanations.items():
    st.write(f"- *{feat}*: {desc}")

# --- 4. Model Performance Metrics ---
st.header("📈 Model Performance on Training Data")

y_train_pred = model.predict(X_scaled)
accuracy = accuracy_score(y, y_train_pred)
precision = precision_score(y, y_train_pred)
recall = recall_score(y, y_train_pred)
f1 = f1_score(y, y_train_pred)
auc = roc_auc_score(y, model.predict_proba(X_scaled)[:, 1])

# Metrics summary
st.write(
    f"""
- Accuracy: *{accuracy:.2f}*  
- Precision: *{precision:.2f}*  
- Recall: *{recall:.2f}*  
- F1 Score: *{f1:.2f}*  
- AUC Score: *{auc:.2f}*
"""
)

# Confusion matrix heatmap
cm = confusion_matrix(y, y_train_pred)
fig2, ax2 = plt.subplots()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Overdose", "Overdose"],
    yticklabels=["No Overdose", "Overdose"],
    ax=ax2,
)
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
ax2.set_title("Confusion Matrix")
st.pyplot(fig2)

# ROC Curve
fpr, tpr, _ = roc_curve(y, model.predict_proba(X_scaled)[:, 1])
fig3, ax3 = plt.subplots()
ax3.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color="#1976D2")
ax3.plot([0, 1], [0, 1], "k--")
ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.set_title("ROC Curve")
ax3.legend(loc="lower right")
st.pyplot(fig3)

# --- 5. Personalized Recommendations ---
st.header("📋 Personalized Recommendations")

if prediction_proba < 0.3:
    st.success(
        """
✔️ *Low Risk:* Keep up good habits!  
- Balanced diet with vegetables and protein  
- Moderate exercise (30+ min/day)  
- Follow medication schedule carefully
"""
    )
elif prediction_proba < 0.7:
    st.warning(
        """
⚠️ *Medium Risk:* Consult a healthcare provider within the next week.  
- Monitor symptoms closely  
- Avoid alcohol and sedatives  
- Eat nutrient-rich foods and stay hydrated  
- Moderate exercise, avoid strenuous activity  
"""
    )
else:
    st.error(
        """
⛔ *High Risk:* Immediate medical attention required!  
- Avoid all non-prescribed medications  
- Keep emergency contacts ready  
- Follow healthcare professional instructions strictly  
- Consider emergency services if symptoms worsen
"""
    )

# --- Helpful Resources with fixed clickable links and emergency button ---
st.markdown(
    """
*Helpful Resources:*  
- [Substance Abuse and Mental Health Services Administration (SAMHSA)](https://www.samhsa.gov)  
- [National Poison Control Center](https://www.poison.org)  
"""
)

if st.button("Call Emergency Services (911)"):
    st.write("Please call 911 immediately in case of an emergency.")

# --- 8. Data Privacy & Security Notice ---
st.markdown(
    """
---
🔒 *Data Privacy & Disclaimer*  
This app collects no data and predictions are for informational purposes only.  
It is not a substitute for professional medical advice, diagnosis, or treatment.  
Always seek the advice of your physician or other qualified health provider with any questions you may have.
"""
)

# --- 9. Download Prediction Report ---
st.header("📥 Download Prediction Report")

def create_report():
    report = io.StringIO()
    report.write("Drug Overdose Prediction Report\n")
    report.write("------------------------------\n")
    report.write(f"Age: {age}\n")
    report.write(f"Weight: {weight} kg\n")
    report.write(f"Dosage: {dosage} mg\n")
    report.write(f"Duration: {duration} days\n")
    report.write(f"Symptom: {symptom}\n\n")
    report.write(f"Prediction: {'Overdose Risk' if prediction==1 else 'No Overdose Risk'}\n")
    report.write(f"Probability: {prediction_proba:.2f}\n\n")
    report.write("Personalized Recommendations:\n")
    if prediction_proba < 0.3:
        report.write("Low Risk - Keep up good habits.\n")
    elif prediction_proba < 0.7:
        report.write("Medium Risk - Consult healthcare provider soon.\n")
    else:
        report.write("High Risk - Seek immediate medical attention.\n")
    return report.getvalue()

report_content = create_report()
st.download_button(
    label="Download Report as TXT",
    data=report_content,
    file_name="overdose_prediction_report.txt",
    mime="text/plain",
)