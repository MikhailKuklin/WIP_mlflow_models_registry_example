import requests
import numpy as np

features = np.load("features.npy")

url = 'http://localhost:6000/predict'
response = requests.post(url, json=features)
print(response.json())
