from flask import Flask, render_template, jsonify
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from data import X_test, y_test
from logisticRegression import logisticRegression
from model_testing import best_params
from sklearn.metrics import accuracy_score, confusion_matrix

app = Flask(__name__)

def train_model():
    global accuracy
    model = logisticRegression(learning_rate=best_params[0], max_iters=best_params[1], epsilon=best_params[2])
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
    app.run(debug=True)

'''@app.route("/data", methods=["POST"])
def data():
    print(type(request.data))
    my_dict = {'data': 1+1, 'my_other_data': 3}
    return my_dict
# Load the ML model
ml_model = model.load_model()  # Ensure model.py has a function to load the model

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_features = data.get("features")  # Expecting a list of features

    if not input_features:
        return jsonify({"error": "No input features provided"}), 400

    try:
        prediction = ml_model.predict([input_features])  # Ensure model supports this input format
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)'''