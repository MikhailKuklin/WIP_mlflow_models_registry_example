import pickle
import numpy as np

from flask import Flask, request, jsonify

with open("model_log_reg.bin", "rb") as f_in:
    model = pickle.load(f_in)


def predict():
    features = np.load("features.npy")
    preds = model.predict(features)
    # return float(preds[0])
    return preds


app = Flask("penguins-prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    features = request.get_json()

    pred = predict(features)

    result = {"penguin_sex": pred}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=6000)
