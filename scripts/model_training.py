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


if __name__ == "__main__":
    df = pull_data("../data/penguins.csv")
    df_enc = preprocess_data(df)
