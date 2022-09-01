import pytest
import pandas as pd

from scripts.model_training import pull_data


def test_pull_data():
    df = pull_data("../data/penguins.csv")
    assert df.shape[1] > 0
