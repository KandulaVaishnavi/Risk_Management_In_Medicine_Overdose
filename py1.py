# Glucose Level Prediction Project
# --------------------------------
# Steps: Load data → Clean → EDA → Feature Engineering → Modeling → Evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix

# ------------------------------
# 1. Load the dataset
# ------------------------------
df = pd.read_csv("framingham.csv")

print("Initial shape:", df.shape)
print(df.head())

# ------------------------------
# 2. Data Cleaning
# ------------------------------
# Convert numeric columns
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Handle missing values (impute glucose with median, drop rows if too many NAs)
glucose_median = df["glucose"].median(skipna=True)
df["glucose"].fillna(glucose_median, inplace=True)

# Drop rows with all NaNs in predictors
df.dropna(subset=["age", "BMI", "sysBP", "diaBP"], inplace=True)

print("After cleaning shape:", df.shape)

# ------------------------------
# 3. Exploratory Data Analysis
# ------------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["glucose"], bins=20, kde=True)
plt.title("Distribution of Glucose Levels")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x="BMI", y="glucose", data=df)
plt.title("Glucose vs BMI")
plt.show()

# ------------------------------
# 4. Feature Engineering
# ------------------------------
# Binary target: high_glucose if glucose ≥ 125
df["high_glucose"] = (df["glucose"] >= 125).astype(int)

predictors = ["age", "BMI", "sysBP", "diaBP", "totChol", "heartRate", "cigsPerDay"]
X = df[predictors]
y_reg = df["glucose"]
y_clf = df["high_glucose"]

# ------------------------------
# 5. Regression (Predict glucose values)
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.3, random_state=42)

reg = RandomForestRegressor(random_state=42)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print("\n--- Regression Results ---")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))

# Feature importance
feat_imp = pd.DataFrame({"Feature": predictors, "Importance": reg.feature_importances_}).sort_values(by="Importance", ascending=False)
print("\nFeature Importances (Regression):\n", feat_imp)

plt.figure(figsize=(7,4))
sns.barplot(x="Importance", y="Feature", data=feat_imp)
plt.title("Feature Importances - Regression")
plt.show()

# ------------------------------
# 6. Classification (if both classes exist)
# ------------------------------
if df["high_glucose"].nunique() > 1:
    X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.3, random_state=42)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n--- Classification Results ---")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    feat_imp_clf = pd.DataFrame({"Feature": predictors, "Importance": clf.feature_importances_}).sort_values(by="Importance", ascending=False)
    plt.figure(figsize=(7,4))
    sns.barplot(x="Importance", y="Feature", data=feat_imp_clf)
    plt.title("Feature Importances - Classification")
    plt.show()
else:
    print("\nClassification skipped: only one class present in this dataset sample.")

# ------------------------------
# 7. Insights
# ------------------------------
print("\nConclusion:")
print(f"- Median glucose imputed: {glucose_median}")
print("- Regression shows which features explain glucose variability.")
print("- Classification requires both normal and high-glucose cases (≥125). Use full dataset for better results.")
