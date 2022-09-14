from pyexpat import features
import requests
import numpy as np
import json
import predict

# features = np.load("features.npy")

# features = {"F": features}

# features_json = json.dumps({"features": features.tolist()})

# features_json = json.dumps({features.tolist()})

url = "http://localhost:6000/predict"

# response = requests.post(url, json=features)
response = requests.post(url, json=features)
print(response.json())

# pred = predict.predict(features)
# print(pred)
