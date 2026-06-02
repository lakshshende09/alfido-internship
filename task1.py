import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

df = pd.read_csv("diabetes.csv", names=columns)

print("Dataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())
# Features (Input)
X = df.drop("Outcome", axis=1)

# Target (Output)
y = df["Outcome"]

print("X Shape:", X.shape)
print("Y Shape:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)
# Create Model
model = LogisticRegression(max_iter=1000)

# Train Model
model.fit(X_train, y_train)

# Make Predictions
predictions = model.predict(X_test)

print("First 10 Predictions:")
print(predictions[:10])
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
roc_auc = roc_auc_score(y_test, predictions)

print("ROC-AUC Score:", roc_auc)
rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

rf_predictions = rf.predict(X_test)

print("\n----- Random Forest Results -----")

print("Accuracy:",
      accuracy_score(y_test, rf_predictions))

print("Precision:",
      precision_score(y_test, rf_predictions))

print("Recall:",
      recall_score(y_test, rf_predictions))

print("F1 Score:",
      f1_score(y_test, rf_predictions))

print("ROC-AUC:",
      roc_auc_score(y_test, rf_predictions))
scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

print("\nCross Validation Scores:")
print(scores)

print("Average CV Score:",
      scores.mean())