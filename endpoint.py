# Serve model as a flask application

import warnings
warnings.filterwarnings("ignore")
from flask import Flask, request, jsonify
import json
import os
import flask
import pandas as pd
import numpy as np
import joblib
import sys
import catboost
import pickle

app = flask.Flask(__name__)
app.config["DEBUG"] = True
app = Flask('app')


model = None


def load_model():
    global model
    # model variable refers to the global variable
    with open('./model_scores_XGBoost.p', 'rb') as f:
        model = pickle.load(f)

@app.route('/test', methods=['GET'])
def test():
    return 'XGBoost Model Application!!'


@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        data = request.get_json()  # Get data posted as a json
        data = np.array(data)[np.newaxis, :]  # converts shape from (4,) to (1, 4)
        prediction = model.predict(data)  # runs globally loaded model on the data
    return str(prediction[0])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
