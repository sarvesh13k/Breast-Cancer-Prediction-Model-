# app.py

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import cancer
app = Flask(__name__)

# Load the trained model
model_path = cancer.model_filename
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file {model_path} does not exist. Please ensure the model is trained and saved.")

model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = data['features']
        prediction = model.predict([features])
        response = {'prediction': int(prediction[0])}
    except Exception as e:
        response = {'error': str(e)}
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
