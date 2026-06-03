# Task 3 - Diabetes Prediction API

This task deploys the diabetes prediction model using Flask.

## Files
- task3.py
- diabetes_model.pkl
- Dockerfile

## Endpoint
POST /predict

## Example Input
{
  "Pregnancies": 6,
  "Glucose": 148,
  "BloodPressure": 72,
  "SkinThickness": 35,
  "Insulin": 0,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.627,
  "Age": 50
}

## Example Output
{
  "prediction": 1
}
