"""
It is a boilerplate for model fit and evaluation
"""

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet

import numpy as np

import mlflow.sklearn


def eval_metrics(model, xs, ys):
    predicted_ys = model.predict(xs)

    rmse = np.sqrt(mean_squared_error(ys, predicted_ys))
    mae = mean_absolute_error(ys, predicted_ys)
    r_2 = r2_score(ys, predicted_ys)

    return rmse, mae, r_2


def fit(params, xs, ys):
    model = ElasticNet(
        alpha=params["alpha"], l1_ratio=params["l1_ratio"], random_state=params["seed"]
    )
    model.fit(xs, ys)

    return model, mlflow.sklearn
