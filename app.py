from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model
with open("knn_model.pkl", "rb") as f:
    model, iris = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html', feature_names=iris.feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form.get(f)) for f in iris.feature_names]
    prediction = model.predict([features])[0]
    pred_class = iris.target_names[prediction]
    return render_template('index.html', feature_names=iris.feature_names, prediction=pred_class)

if __name__ == '__main__':
    app.run(debug=True)
