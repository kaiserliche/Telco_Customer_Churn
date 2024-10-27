from airflow import DAG
from airflow.decorators import task
from datetime import datetime, timedelta
from airflow.providers.http.hooks.http import HttpHook
import os
import logging
import zipfile


# Configurations
class Config:
    DOWNLOAD_DIR = "../data/raw_data/"  # Updated to absolute path
    ZIP_FILE_PATH = os.path.join(DOWNLOAD_DIR, "dataset.zip")
    # DB_CONN = "postgres_default"
    API_CONN_ID = "Telco_Kaggle"  # Ensure 'Telco' connection is defined in Airflow
    HOST = "https://www.kaggle.com"
    END_POINT = "/api/v1/datasets/download/yeanzc/telco-customer-churn-ibm-dataset"

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
    dag_id="dataExtractionPipeline",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
) as dag:

    @task()
    def download_raw_data():
        """Task to download the dataset using Airflow's HttpHook"""
        if not os.path.exists(Config.DOWNLOAD_DIR):
            os.makedirs(Config.DOWNLOAD_DIR)

        # Use HttpHook with connection ID
        http_hook = HttpHook(http_conn_id=Config.API_CONN_ID, method="GET")

        # Make the request via the HTTP Hook
        response = http_hook.run(Config.END_POINT)

        if response.status_code == 200:
            with open(Config.ZIP_FILE_PATH, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            logger.info(f"Dataset downloaded successfully to {Config.ZIP_FILE_PATH}")
        else:
            raise Exception(
                f"Failed to download dataset. Status code: {response.status_code}"
            )

    @task()
    def unzip_dataset():
        """Task to unzip the downloaded dataset"""
        with zipfile.ZipFile(Config.ZIP_FILE_PATH, "r") as zip_ref:
            zip_ref.extractall(Config.DOWNLOAD_DIR)
        logger.info(f"Dataset extracted to {Config.DOWNLOAD_DIR}")

    @task()
    def cleanup_zip_file():
        """Task to clean up the zip file"""
        if os.path.exists(Config.ZIP_FILE_PATH):
            os.remove(Config.ZIP_FILE_PATH)
            logger.info(f"Cleanup: Deleted zip file {Config.ZIP_FILE_PATH}")
        else:
            logger.warning(f"Cleanup: No zip file found at {Config.ZIP_FILE_PATH}")

    # Defining the task sequence
    download_task = download_raw_data()
    unzip_task = unzip_dataset()
    cleanup_task = cleanup_zip_file()

    download_task >> unzip_task >> cleanup_task
