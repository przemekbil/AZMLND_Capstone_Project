import json
import numpy as np
import pandas as pd
import os
import pickle
import joblib

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.joblib')
    model = joblib.load(model_path)

def run(raw_data):

    data = json.loads(raw_data)['data']
    data_frame = pd.DataFrame.from_dict(data)
    # make prediction
    
    y_hat = model.predict(data_frame)
    # you can return any data type as long as it is JSON-serializable
    return y_hat.tolist()