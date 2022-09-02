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

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner


@task
def pull_data(path_to_dataset: Path):
    # load data
    df = pd.read_csv(path_to_dataset)

    # data preprocessing
    df = df.drop_duplicates()
    df.dropna(axis=0, inplace=True)

    return df


@task
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

    # get labels
    labels = df_enc[["label"]]
    y = labels.to_numpy().reshape(
        -1,
    )

    # drop useless columns
    df_enc.drop(columns=["rowid", "sex"], axis=1, inplace=True)

    return df_enc


@task
def get_train_val_datasets(df: DataFrame):
    X = df.iloc[:, :11].values
    labels = df[["label"]]
    y = labels.to_numpy().reshape(
        -1,
    )

    # split dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
    )

    return X_train, X_val, y_train, y_val


def get_confusion_matrix(clf, X, y, name):
    plot_confusion_matrix(clf, X, y)
    plt.savefig(name)


def get_roc(clf, X, y, name):
    metrics.plot_roc_curve(clf, X, y)
    plt.savefig(name)


@task
def train_log_reg(X_train, X_val, y_train, y_val, regularization):
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
        random_state=42,
        solver="lbfgs",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )

    pipe_def_model = Pipeline([("scaler", scaler), ("log_reg", log_reg)])

    model = pipe_def_model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    # metrics

    def get_metrics(y_true, y_pred, acc_name, prec_name, recall_name, f1_name):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1_s = f1_score(y_true, y_pred)

        return {
            acc_name: acc,
            prec_name: prec,
            recall_name: recall,
            f1_name: f1_s,
        }

    metrics_train = get_metrics(
        y_train,
        y_pred_train,
        "train_accuracy",
        "train_precision",
        "train_recall",
        "train_f1_score",
    )

    metrics_val = get_metrics(
        y_val,
        y_pred_val,
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1_score",
    )

    # confusion_matric_artifact = get_confusion_matrix(
    #     model, X_train, y_train, "confusion_matrix_train.png"
    # )
    # roc_artifact = get_roc(model, X_train, y_train, "roc_train.png")

    # confusion_matric_artifact = get_confusion_matrix(
    #     model, X_val, y_val, "confusion_matrix_val.png"
    # )
    # roc_artifact = get_roc(model, X_val, y_val, "roc_val.png")

    with mlflow.start_run():

        for metric in metrics_train:
            mlflow.log_metric(metric, metrics_train[metric])

        for metric in metrics_val:
            mlflow.log_metric(metric, metrics_val[metric])

        # mlflow.log_artifact("confusion_matrix_train.png", "confusion_matrix_train")
        # mlflow.log_artifact("confusion_matrix_val.png", "confusion_matrix_val")
        # mlflow.log_artifact("roc_train.png", "roc_train")
        # mlflow.log_artifact("roc_val.png", "roc_val")
        mlflow.log_param("Regularization", 0.0001)

        mlflow.set_tag("Data", "../data/penguins.csv")

        mlflow.sklearn.log_model(model, "log_reg")

        mlflow.sklearn.log_model(model, artifact_path="models_mlflow")


@flow
def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("test")
    df = pull_data("../data/penguins.csv")
    df_enc = preprocess_data(df)
    X_train, X_val, y_train, y_val = get_train_val_datasets(df_enc)
    train_log_reg(X_train, X_val, y_train, y_val, 0.0001)


main()
