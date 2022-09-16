import pickle
import numpy as np

from flask import Flask, request, jsonify

with open("model_log_reg.bin", "rb") as f_in:
    model = pickle.load(f_in)


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


def transform_predict(preds):
    if preds[0] == 1.0:
        return "male"
    elif preds[0] == 0.0:
        return "female"


app = Flask("penguins-prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    features = request.get_json()

    to_predict = prepare_features(features)

    pred = predict(to_predict)

    transformed_pred = transform_predict(pred)

    result = {"penguin-sex": transformed_pred}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=6000)
