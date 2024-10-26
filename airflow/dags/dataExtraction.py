from airflow import DAG
from airflow.decorators import task
from datetime import datetime, timedelta
from airflow.providers.http.hooks.http import HttpHook
import os
import logging
import zipfile


# Configurations
class Config:
    DOWNLOAD_DIR = "../data"  # Updated to absolute path
    ZIP_FILE_PATH = os.path.join(DOWNLOAD_DIR, "dataset.zip")
    # DB_CONN = "postgres_default"
    API_CONN_ID = "Telco"  # Ensure 'Telco' connection is defined in Airflow
    HOST = "https://storage.googleapis.com"
    END_POINT = "/kaggle-data-sets/1020460/1720381/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241023%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241023T092408Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=c8b93fa3ee3ca7322f7cf3bf34ee060ccad95dfc640b33b80ba9f5653923be784089e7a68ada612a0d23e2be2166b2a55a866722aa1789c14126dfeb111bce14822004c6d7c723fe85fdd30f75b5b1a64ac07cc342d2f8bfc9f1421b487d0e11c3e03bfbe3c5255e3babbd8fc8cbeecba2ed88f734a65a04dcc695d35b46206d04a76f31022b474af56d2960831134882b392c0aa68ccdf21807e8b7001e4d757c94854d4aa875ea4337bc8c54028e49aeb0cda8727731018ef94a27fd7acce8df78791a63b5750db9e9a6c14a2f5f17c8753743e35d8776deb970a3125180b4903f7648ccf8120798127163faf64dbacff17fce2328aa063aa16eccdd311f85"


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
