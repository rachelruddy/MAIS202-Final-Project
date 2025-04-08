'''from flask import Flask, render_template, jsonify
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from heartData import X_test, y_test
from logRegHeartDisease import logRegHeartDisease
from modelTestingHeartPred import best_params, y_pred
from sklearn.metrics import accuracy_score, confusion_matrix

app = Flask(__name__)

def train_model():
    global accuracy
    model = logRegHeartDisease(learning_rate=best_params[0], max_iters=best_params[1], epsilon=best_params[2])
    model.fit(X_test, y_test)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,5))
    plt.pie([accuracy, 1 - accuracy], labels=['Correct', 'Incorrect'], colors=['#f19ef7', '#fcffa8'], autopct='%1.1f%%')
    plt.title('Model Accuracy')
    plt.savefig('static/accuracy_pie.png')
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='BuPu', xticklabels=['Low Risk', 'Mid Risk', 'High Risk'], yticklabels=['Low Risk', 'Mid Risk', 'High Risk'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('static/confusion_matrix.png')
    plt.close()


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def retrain():
    train_model()
    return jsonify({"accuracy": accuracy})

if __name__ == "__main__":
    train_model()
    app.run(debug=True)'''

from flask import Flask, request, render_template_string
import numpy as np
from heartData import scaler, X_train, y_train
from logRegHeartDisease import logRegHeartDisease
from modelTestingHeartPred import best_params

# Retrain model using best parameters from modelTestingHeartPred.py
BEST_LR = best_params[0]
BEST_ITERS = best_params[1]
BEST_EPSILON = best_params[2]

model = logRegHeartDisease(learning_rate=BEST_LR, max_iters=BEST_ITERS, epsilon=BEST_EPSILON)
model.fit(X_train, y_train)

app = Flask(__name__)

# HTML template for input form
form_template = """
<!doctype html>
<html lang="en">
<head>
    <title>Heart Disease Predictor</title>
    <style>
        body {
            background-color: #cce7ff;
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .form-container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        input[type=number], input[type=submit] {
            margin: 8px 0;
            padding: 8px;
            width: 100%;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        input[type=submit] {
            background-color: #007bff;
            color: white;
            border: none;
            cursor: pointer;
        }
        input[type=submit]:hover {
            background-color: #0056b3;
        }
        h2, h3 {
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="form-container">
        <h2>Enter Patient Info:</h2>
        <form method=post action="/predict">
          Age: <input type=number name=Age required><br>
          Gender (0=Female, 1=Male): <input type=number name=Gender min=0 max=1 required><br>
          Blood Pressure: <input type=number name=BloodPressure required><br>
          Cholesterol: <input type=number name=Cholesterol required><br>
          Heart Rate: <input type=number name=HeartRate required><br>
          Quantum Pattern Feature: <input type=number step=any name=QuantumPatternFeature required><br>
          <input type=submit value=Predict>
        </form>
        {% if prediction is not none %}
          <h3>Prediction: {{ prediction }}</h3>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(form_template, prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse input features
        features = [
            float(request.form["Age"]),
            float(request.form["Gender"]),
            float(request.form["BloodPressure"]),
            float(request.form["Cholesterol"]),
            float(request.form["HeartRate"]),
            float(request.form["QuantumPatternFeature"])
        ]

        input_array = np.array(features).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        pred = model.predict(input_scaled)[0]
        result = "Heart Disease" if pred == 1 else "No Heart Disease"

        return render_template_string(form_template, prediction=result)
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
