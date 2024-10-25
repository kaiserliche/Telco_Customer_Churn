from dataLoader import TelcoDataLoader
import yaml
from xgboost import XGBClassifier
from argparse import ArgumentParser
from sklearn.metrics import classification_report, accuracy_score
import logging
import os
import json

os.makedirs("logs", exist_ok=True)

# Set up the first logger for general application logs
general_logger = logging.getLogger("general_logger")
general_logger.setLevel(logging.INFO)

general_handler = logging.FileHandler("logs/general.log")
general_handler.setLevel(logging.INFO)
general_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
general_handler.setFormatter(general_formatter)

general_logger.addHandler(general_handler)

# Set up the second logger for evaluation metrics logs
metrics_logger = logging.getLogger("metrics_logger")
metrics_logger.setLevel(logging.INFO)

metrics_handler = logging.FileHandler("logs/metrics.log")
metrics_handler.setLevel(logging.INFO)
metrics_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
metrics_handler.setFormatter(metrics_formatter)

metrics_logger.addHandler(metrics_handler)


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        config_dict = yaml.safe_load(stream)

    general_logger.info(f"Loading Telco Dataloader")
    dataloader = TelcoDataLoader(config_dict["data_path"])

    general_logger.info(f"Cleaning the data")
    dataloader.clean_data()

    general_logger.info(f"Starting Feature Reduction")
    dataloader.feature_reduction(config_dict["model"]["features"])

    general_logger.info(f"Splitting the data for Training and Evaluation")
    data_dict = dataloader.get_splits(
        test_size=config_dict["model"]["training"]["test_size"], seed=config_dict["seed"]
    )

    general_logger.info(f"Instantiating XGBClassifier model")
    model = XGBClassifier(**config_dict["model"]["xgb_params"])

    general_logger.info(f"Starting the the training")
    model.fit(data_dict["X_train"], data_dict["y_train"])
    general_logger.info(f"Model Trained Successfully !!")

    ### starting evaluation ###
    general_logger.info("Starting Metric Evaluation")

    y_pred = model.predict(data_dict["X_test"])
    accuracy = accuracy_score(data_dict["y_test"], y_pred)
    report = classification_report(data_dict["y_test"], y_pred , output_dict=True)
    metrics_logger.info(f"Accuracy : {accuracy}")

    metrics_logger.info("Classification Report:\n%s", json.dumps(report, indent=4))

    general_logger.info("Evaluation Completed.")

    ### Saving the model ###
    model.save_model(f"model.json")


if __name__=="__main__":
    main()