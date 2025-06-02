import joblib
import numpy as np
import json
from azureml.core.model import Model

def init():
    global model, scaler
    model_path = Model.get_model_path('linear_regression_model')
    scaler_path = Model.get_model_path('scaler_model')

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

def run(raw_data):
    try:
        data = np.array(json.loads(raw_data)['data'])
        data_scaled = scaler.transform(data)
        predictions = model.predict(data_scaled)
        return predictions.tolist()
    except Exception as e:
        return str(e)
