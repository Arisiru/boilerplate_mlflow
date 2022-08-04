"""
It is a boilerplate for training step.
Use it as an example of training a models with sklearn and mlflow.
It has implemnted tracking and model registry calls.
"""

import sys
import warnings
import logging
import yaml
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn

from src.preprocessing import dataset
from src.training import modelfactory

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def main():

    # Get params from arguments
    training_config_file = str(sys.argv[1]) if len(sys.argv) > 1 else None
    if not training_config_file:
        training_config_file = "training_config.yaml"
        logger.warning("Using default path for training configuration")

    training_config = {}
    with open(training_config_file, "r", encoding="utf-8") as f_r:
        training_config = yaml.safe_load(f_r)

    # set mlFlow destination from MLFLOW_TRACKING_URI in the env
    # mlflow.set_tracking_uri(REMOTE_SERVER_URI)
    # print(f"TRACKING URL: {mlflow.tracking.get_tracking_uri()}")

    # set an experiment tag
    experiment_id = training_config["run"]["experiment"]["id"]
    mlflow.set_experiment(experiment_id)

    # scope for mlFlow run
    with mlflow.start_run():
        # log params as dictionary
        mlflow.log_params(training_config["hyperparameters"])

        train_xs, train_ys, test_xs, test_ys = dataset.prepare(
            training_config["data_file"]
        )

        model, model_flavor = modelfactory.fit(
            training_config["hyperparameters"], train_xs, train_ys
        )

        rmse, mae, r_2 = modelfactory.eval_metrics(model, test_xs, test_ys)

        # rmse: root of mean square error 0 is perfect
        #  mae: mead absolute error 0 is perfect
        #  r_2: regression function 1 is perfect -inf is worst

        print(
            f"Elasticnet model \
            (alpha={training_config['hyperparameters']['alpha']}, \
            l1_ratio={training_config['hyperparameters']['l1_ratio']}):"
        )
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r_2}")

        # log metrics to the mlFlow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r_2)
        mlflow.log_metric("mae", mae)

        # check if mlFlow track localy in file or into a server
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Dependse on mlFlow store deside to use Model registry.
        # Model registry does not work with file store
        if tracking_url_type_store == "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            model_flavor.log_model(model, "model", registered_model_name=experiment_id)
        else:
            model_flavor.log_model(model, "model")


if __name__ == "__main__":
    main()
