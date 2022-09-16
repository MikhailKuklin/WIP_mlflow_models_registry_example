import requests

features = {
    "f1": 2.800e02,
    "f2": 0,
    "f3": 0,
    "f4": 1,
    "f5": 0,
    "f6": 0,
    "f7": 1,
    "f8": 5.270e01,
    "f9": 1.980e01,
    "f10": 1.970e02,
    "f11": 3.725e03,
}

url = "http://localhost:6000/predict"

response = requests.post(url, json=features)
print(response.json())
