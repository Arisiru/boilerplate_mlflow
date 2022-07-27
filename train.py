"""
It is a boilerplate for training step.
Use it as an example of training a models with sklearn and mlflow.
It has implemnted tracking and model registry calls.
Please look for `# >>>>>>>>> MLFLOW` comments they indicate code related to mlFlow
"""

import os
import warnings
import sys
import logging
from urllib.parse import urlparse

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

EXPERIMENT_TAG = os.environ.get("EXPERIMENT_TAG")
REMOTE_SERVER_URI = os.environ.get("MLFLOW_TRACKING_URI")
PREDICT_FIELD = "quality"


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r_2 = r2_score(actual, pred)
    return rmse, mae, r_2


def data_prep(file):
    # Read data from file
    try:
        data = pd.read_csv(file)
    except IOError as exeption:
        logger.exception("Unable to open training & test CSV, Error: %s", exeption)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_ys = train[PREDICT_FIELD]
    test_ys = test[PREDICT_FIELD]
    train_xs = train.drop([PREDICT_FIELD], axis=1)
    test_xs = test.drop([PREDICT_FIELD], axis=1)

    return train_xs, train_ys, test_xs, test_ys


def main():
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Get params from arguments
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    data_file = str(sys.argv[3]) if len(sys.argv) > 3 else ""

    train_xs, train_ys, test_xs, test_ys = data_prep(data_file)

    # >>>>>>>>> MLFLOW
    # set mlFlow destination from MLFLOW_TRACKING_URI in the env
    # mlflow.set_tracking_uri(REMOTE_SERVER_URI)
    # print(f"TRACKING URL: {mlflow.tracking.get_tracking_uri()}")

    # set an experiment tag
    mlflow.set_experiment(EXPERIMENT_TAG)
    # <<<<<<<<< MLFLOW

    # >>>>>>>>> MLFLOW
    # scope for mlFlow
    with mlflow.start_run():

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(train_xs, train_ys)

        predicted_qualities = model.predict(test_xs)

        (rmse, mae, r_2) = eval_metrics(test_ys, predicted_qualities)

        print(f"Elasticnet model (alpha={alpha}, l1_ratio={l1_ratio}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r_2}")

        # >>>>>>>>> MLFLOW
        # log prameters and metrics to the mlFlow
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("data_input", data_file)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r_2)
        mlflow.log_metric("mae", mae)
        # <<<<<<<<< MLFLOW

        # >>>>>>>>> MLFLOW
        # check if mlFlow track localy in file or into a server
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # <<<<<<<<< MLFLOW

        # >>>>>>>>> MLFLOW
        # Dependse on mlFlow store deside to use Model registry.
        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                model, "model", registered_model_name=EXPERIMENT_TAG
            )
        else:
            mlflow.sklearn.log_model(model, "model")

    # <<<<<<<<< MLFLOW


if __name__ == "__main__":
    main()
