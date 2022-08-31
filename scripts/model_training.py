# load necessary modules

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import category_encoders as ce
import pickle
import mlflow

from pandas import DataFrame
from sklearn import metrics, datasets, preprocessing
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    plot_confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("penguins_log_reg_pipe")


def pull_data(path_to_dataset: Path):
    # load data
    df = pd.read_csv(path_to_dataset)

    # data preprocessing
    df = df.drop_duplicates()
    df.dropna(axis=0, inplace=True)

    return df


def preprocess_data(df: DataFrame):
    # one hot encoding - nominal data
    encoder = ce.OneHotEncoder(
        cols=["species", "island"],
        handle_unknown="return_nan",
        return_df=True,
        use_cat_names=True,
    )
    df_enc = encoder.fit_transform(df)

    # add a new column with labels
    df_enc.loc[df_enc.sex == "male", "label"] = int(1)
    df_enc.loc[df_enc.sex == "female", "label"] = int(0)
    df_enc["label"].astype("float")

    # check number of males/females to inspect if dataset is imbalanced
    df_enc["label"].value_counts()

    # get labels
    labels = df_enc[["label"]]
    y = labels.to_numpy().reshape(
        -1,
    )

    # drop useless columns
    df_enc.drop(columns=["rowid", "sex"], axis=1, inplace=True)

    return df_enc


def get_train_val_datasets(df: DataFrame):
    X = df_enc.iloc[:, :11].values

    # split dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
    )

    return X_train, X_val, y_train, y_val


def train_log_reg(X_train, y_train, regularization):
    np.random.seed(0)
    scaler = StandardScaler()
    log_reg = LogisticRegression(
        C=regularization,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class="ovr",
        n_jobs=None,
        penalty="l2",
        random_state=None,
        solver="lbfgs",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )
    pipe_def_model = Pipeline([("scaler", scaler), ("log_reg", log_reg)])
    pipe_def_model.fit(X_train, y_train)

    return pipe_def_model


def predict(model, X):
    y_pred = model.predict(X)
    return y_pred


def predict_prob(model, X):
    y_pred = model.predict_proba(X)
    return y_pred


def get_metrics_train(y_true, y_pred, y_pred_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1_s = f1_score(y_true, y_pred)

    return {
        "train-accuracy": acc,
        "train-precision": prec,
        "train-recall": recall,
        "train-f1-score": f1_s,
    }


def get_metrics_val(y_true, y_pred, y_pred_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1_s = f1_score(y_true, y_pred)

    return {
        "val-accuracy": acc,
        "val-precision": prec,
        "val-recall": recall,
        "val-f1-score": f1_s,
    }


def get_confusion_matrix(clf, X, y, name):
    plot_confusion_matrix(clf, X, y)
    plt.savefig(name)


def get_roc(clf, X, y, name):
    metrics.plot_roc_curve(clf, X, y)
    plt.savefig(name)


if __name__ == "__main__":
    df = pull_data("../data/penguins.csv")
    df_enc = preprocess_data(df)
    X_train, X_val, y_train, y_val = get_train_val_datasets(df_enc)
    train_log_reg(X_train, y_train, 0.0001)
