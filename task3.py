from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("diabetes_model.pkl")

@app.route("/")
def home():
    return "Diabetes Prediction API is Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    features = [[
        data["Pregnancies"],
        data["Glucose"],
        data["BloodPressure"],
        data["SkinThickness"],
        data["Insulin"],
        data["BMI"],
        data["DiabetesPedigreeFunction"],
        data["Age"]
    ]]

    prediction = int(model.predict(features)[0])

    return jsonify({
        "prediction": prediction
    })

if __name__ == "__main__":
    app.run(debug=True)