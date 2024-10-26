from airflow import DAG
from airflow.decorators import task
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
import pandas as pd
import numpy as np
import os
from typing import List


# Configurations
class Config:
    RAW_PATH = "../data/raw_data/Telco_customer_churn.xlsx"
    SAVE_DIR = "../data/processed_data"
    SELECTED_FEATURES = [
        "Contract_Month-to-month",
        "Internet Service_Fiber optic",
        "Payment Method_Electronic check",
        "Internet Service_No",
        "Contract_Two year",
        "Tenure Months",
    ]
    TEST_SIZE = 0.3
    SEED = 42


os.makedirs(Config.SAVE_DIR, exist_ok=True)
# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 10, 25),
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# Defining the DAG
with DAG(
    dag_id="dataProcessingPipeline",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
) as dag:

    @task()
    def data_cleaning(raw_data_path: str) -> dict:
        df = pd.read_excel(raw_data_path)
        irrelevant_columns = [
            "Latitude",
            "Longitude",
            "CustomerID",
            "Count",
            "Country",
            "State",
            "City",
            "Zip Code",
            "Lat Long",
            "Churn Reason",
            "Churn Label",
            "CLTV",
            "Churn Score",
        ]

        df = df.drop(irrelevant_columns, axis=1, errors="ignore")
        no_internet_columns = [
            "Online Security",
            "Online Backup",
            "Device Protection",
            "Tech Support",
            "Streaming TV",
            "Streaming Movies",
        ]  # all service columns

        for feature in no_internet_columns:
            df[feature] = df[feature].map(
                {"No": "No", "Yes": "Yes", "No internet service": "No"}
            )

        df["Multiple Lines"] = df["Multiple Lines"].map(
            {"No": "No", "Yes": "Yes", "No phone service": "No"}
        )
        df["Gender"] = df["Gender"].map({"Male": "Yes", "Female": "No"})
        df_enc = pd.get_dummies(
            df, columns=["Contract", "Internet Service", "Payment Method"]
        )
        for col in list(df_enc.columns):
            if col in [
                "Tenure Months",
                "Monthly Charges",
                "Total Charges",
                "Churn Value",
            ]:
                continue
            if df_enc[col].dtype == "bool":
                df_enc[col] = df_enc[col].map({True: 1, False: 0})
            else:
                df_enc[col] = df_enc[col].map({"Yes": 1, "No": 0})
        df_enc["Total Charges"] = pd.to_numeric(
            df_enc["Total Charges"], errors="coerce"
        )
        null_mask = df_enc.isnull().any(axis=1)
        null_rows = df_enc[null_mask].index
        df_enc.drop(index=null_rows, inplace=True, errors="raise")
        df_enc = df_enc.reset_index(drop=True)

        return df_enc.to_dict()

    @task()
    def feature_reduction(cleaned_data, features_to_select: List):
        df = pd.DataFrame.from_dict(cleaned_data)
        ### numeric features scaling ###
        df_label = df["Churn Value"]
        df = df.drop(["Churn Value"], axis=1)
        numeric_features = ["Tenure Months", "Monthly Charges", "Total Charges"]
        df_numeric = df[numeric_features]
        df = df.drop(numeric_features, axis=1)
        scaler = StandardScaler()
        df_numeric = pd.DataFrame(
            scaler.fit_transform(df_numeric), columns=numeric_features
        )
        df = pd.concat([df, df_numeric], axis=1)

        ### selection ###
        df = df[features_to_select]
        return {"features": df.to_dict(), "labels": df_label.to_dict()}

    @task()
    def data_splitting(transformed_data: dict) -> None:
        # Convert dictionaries back to DataFrame/Series
        df = pd.DataFrame.from_dict(transformed_data["features"])
        df_label = pd.Series(transformed_data["labels"])

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            df, df_label, test_size=Config.TEST_SIZE, random_state=Config.SEED
        )

        # Display the shape of the resulting arrays
        logger.info(f"Shape of X_train: {X_train.shape}")
        logger.info(f"Shape of X_test: {X_test.shape}")
        logger.info(f"Shape of y_train: {y_train.shape}")
        logger.info(f"Shape of y_test: {y_test.shape}")

        # Saving the split data for the model training pipeline
        np.save(f"{Config.SAVE_DIR}/X_train.npy", X_train)
        np.save(f"{Config.SAVE_DIR}/y_train.npy", y_train)
        np.save(f"{Config.SAVE_DIR}/X_test.npy", X_test)
        np.save(f"{Config.SAVE_DIR}/y_test.npy", y_test)

        logger.info(f"Processed Data Saved Successfully at {Config.SAVE_DIR}")

    cleaned_data = data_cleaning(Config.RAW_PATH)
    transformed_data = feature_reduction(cleaned_data, Config.SELECTED_FEATURES)
    data_splitting(transformed_data)
