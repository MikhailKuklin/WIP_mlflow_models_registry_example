import os
import pickle
import requests

from pymongo import MongoClient

from flask import Flask, request, jsomify

MODEL_FILE = os.getenv("MODEL_FILE", "model_log_reg.bin")
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")
EVIDENTLY_SERVICE_ADDRESS = os.genenv("EVIDENTLY_SERVICE_", "https:///127.0.0.1.:5000")

# open trained model
with open(MODEL_FILE, "rb") as f_in:
    model = pickle.load(f_in)

app = Flask("penguins-prediction")

mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database("prediction_service")
collection = db.get_collection("data")

# transform features to array formar as required by code
def prepare_features(features):
    X = np.array(
        [
            [
                features["f1"],
                features["f2"],
                features["f3"],
                features["f4"],
                features["f5"],
                features["f6"],
                features["f7"],
                features["f8"],
                features["f9"],
                features["f10"],
                features["f11"],
            ]
        ]
    )
    return X


def predict(X):
    preds = model.predict(X)
    return preds


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    features = request.get_json()
    to_predict = prepare_features(features)
    pred = predict(to_predict)
    transformed_pred = transform_predict(pred)
    result = {"penguin-sex": transformed_pred}

    save_to_db(features, prediction)
    send_to_evidently_service(features, prediction)

    return jsonify(result)


def save_to_db(features, prediction):
    rec = features.copy()
    rec["prediction"] = prediction
    collection.insert_one(rec)


def send_to_evidently_service(features, prediction):
    rec = features.copy()
    rec["prediction"] = prediction
    request.post(f"{EVIDENTLY_SERVICE_ADDRESS }/ iterate / penguins", json=[features])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=6000)
