import joblib
import numpy as np
import json
from azureml.core.model import Model

def init():
    global model, scaler
    model = joblib.load(Model.get_model_path('linear_regression_model'))
    scaler = joblib.load(Model.get_model_path('scaler_model'))

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    data_scaled = scaler.transform(data)
    return model.predict(data_scaled).tolist()