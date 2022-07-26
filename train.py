# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

experiment_tag = os.environ.get('EXPERIMENT_TAG')
remote_server_uri = os.environ.get('MLFLOW_TRACKING_URI')

predict_field = "quality"

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def data_prep(csv_filename):
    # Read data from file
    try:
        data = pd.read_csv(csv_filename)
    except Exception as e:
        logger.exception(
            "Unable to open training & test CSV, Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_y = train[predict_field]
    test_y = test[predict_field]
    train_x = train.drop([predict_field], axis=1)
    test_x = test.drop([predict_field], axis=1)

    return {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y
    }


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Get params from arguments
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    csv_filename = str(sys.argv[3]) if len(sys.argv) > 3 else ''

    data = data_prep(csv_filename)
    
    #mlflow.set_tracking_uri(remote_server_uri)  # or set the MLFLOW_TRACKING_URI in the env
    #print("TRACKING URL: ", mlflow.tracking.get_tracking_uri())
    mlflow.set_experiment(experiment_tag)


    with mlflow.start_run():

        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(data["train_x"], data["train_y"])

        predicted_qualities = lr.predict(data["test_x"])

        (rmse, mae, r2) = eval_metrics(data["test_y"], predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("data_input", csv_filename)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lr, "model", registered_model_name=experiment_tag)
        else:
            mlflow.sklearn.log_model(lr, "model")
