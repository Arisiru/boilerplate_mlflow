"""
It is a boilerplate for dataset preporcessing
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare(file):
    y_field = "quality"

    # Read data from file
    data = pd.read_csv(file)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_ys = train[y_field]
    test_ys = test[y_field]
    train_xs = train.drop([y_field], axis=1)
    test_xs = test.drop([y_field], axis=1)

    return train_xs, train_ys, test_xs, test_ys
