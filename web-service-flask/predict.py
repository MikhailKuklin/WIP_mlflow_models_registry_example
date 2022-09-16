import pickle
import numpy as np

from flask import Flask, request, jsonify

with open("model_log_reg.bin", "rb") as f_in:
    model = pickle.load(f_in)

def prepare_features(features):
    X = np.array([[features['f1'], features['f2'], features['f3'],
    features['f4'], features['f5'], features['f6'], features['f7'], features['f8'],
    features['f9'], features['f10'], features['f11']])
    return X


def predict(X):
    preds = model.predict(X)
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
