from pyexpat import features
import requests
import numpy as np
import json
import predict

features = np.load("features.npy")

# features = {"PULocationID": 10, "DOLocationID": 50, "trip_distance": 40}

# features_json = json.dumps({"features": features.tolist()})

# features_json = json.dumps({features.tolist()})

# url = "http://localhost:6000/predict"

# response = requests.post(url, json=features_json)
# print(response.json())

pred = predict.predict(features)
print(pred)
