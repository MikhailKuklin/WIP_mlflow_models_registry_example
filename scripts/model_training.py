# load necessary modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import category_encoders as ce
import pickle
import mlflow

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


def load_data(path_to_dataset):
    # load data
    df = pd.read_csv(path_to_dataset)

    # data preprocessing
    df = df.drop_duplicates()
    df.dropna(axis=0, inplace=True)

    return df


if __name__ == "__main__":
    df = load_data("../data/penguins.csv")
