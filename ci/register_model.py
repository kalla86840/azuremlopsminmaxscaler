
from azureml.core import Workspace, Model
import json
import os

def load_config(path='config.json'):
    with open(path) as f:
        return json.load(f)

def register_model():
    config = load_config()
    ws = Workspace.get(
        name=config['workspace_name'],
        subscription_id=config['subscription_id'],
        resource_group=config['resource_group']
    )

    Model.register(workspace=ws, model_path='model.joblib', model_name='linear_regression_model')
    Model.register(workspace=ws, model_path='scaler.joblib', model_name='scaler_model')

if __name__ == '__main__':
    register_model()
