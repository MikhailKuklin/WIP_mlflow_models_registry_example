from pyexpat import features
import requests
import numpy as np
import json

features = np.load("features.npy")

json_str = json.dumps({"features": features.tolist()})

url = "http://localhost:6000/predict"
response = requests.post(url, json=json_str)
print(response.json())
