from airflow import DAG
from airflow.decorators import task
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import json
import numpy as np
import logging
import base64

class Config:
    MODELS_DIR = "../models/"
    DATA_DIR = "../data/processed_data"
    POSTGRES_CONN_ID = "postgres_default"
    AUGMENT = False

class ModelPipeline:
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger("pipeline_logger")
        logging.basicConfig(level=logging.INFO)
    
    def create_pipeline(self, dag: DAG):
        """Create all pipeline tasks and set up their dependencies."""
        
        @task
        def load_data():
            """Load and preprocess the data."""
            self.logger.info("Loading data...")
            X_train = np.load(f"{self.config.DATA_DIR}/X_train.npy")
            y_train = np.load(f"{self.config.DATA_DIR}/y_train.npy")
            X_test = np.load(f"{self.config.DATA_DIR}/X_test.npy")
            y_test = np.load(f"{self.config.DATA_DIR}/y_test.npy")
            
            return {
                "X_train": X_train.tolist(),
                "y_train": y_train.tolist(),
                "X_test": X_test.tolist(),
                "y_test": y_test.tolist()
            }
            
        @task
        def fit_model(data: dict):
            """Train the model and return predictions."""
            self.logger.info("Training model...")
            
            X_train = np.array(data["X_train"])
            y_train = np.array(data["y_train"])
            
            model = XGBClassifier(
                learning_rate=0.01,
                max_depth=3,
                n_estimators=500,
                scale_pos_weight=3
            )
            
            model.fit(X_train, y_train)
            
            # Save model to a temporary file and read bytes
            temp_path = "/tmp/temp_model.json"
            model.save_model(temp_path)
            with open(temp_path, 'rb') as f:
                model_bytes = f.read()
            os.remove(temp_path)
            
            # Convert bytes to base64 string as model is not serializable in airflow env
            model_base64 = base64.b64encode(model_bytes).decode('utf-8')
            
            return {
                "model_base64": model_base64,
                "train_data": data
            }
            
        @task
        def evaluate_model(model_data: dict):
            """Evaluate the model performance."""
            self.logger.info("Evaluating model...")
            
            # Decode base64 string back to bytes
            model_bytes = base64.b64decode(model_data["model_base64"])
            
            # Load model from bytes
            temp_path = "/tmp/temp_model.json"
            with open(temp_path, 'wb') as f:
                f.write(model_bytes)
            
            model = XGBClassifier()
            model.load_model(temp_path)
            os.remove(temp_path)
            
            # Get test data
            data = model_data["train_data"]
            X_test = np.array(data["X_test"])
            y_test = np.array(data["y_test"])
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = float(accuracy_score(y_test, y_pred))
            report = classification_report(y_test, y_pred, output_dict=True)
            
            self.logger.info(f"Model Accuracy: {accuracy:.4f}")
            self.logger.info("Classification Report:\n%s", json.dumps(report, indent=4))
            
            return {
                "model_base64": model_data["model_base64"],
                "metrics": {
                    "accuracy": accuracy,
                    "report": report
                }
            }
            
        @task
        def save_model(model_data: dict):
            """Save the model with versioning."""
            self.logger.info("Saving model...")
            
            # Decode base64 string back to bytes
            model_bytes = base64.b64decode(model_data["model_base64"])
            
            os.makedirs(self.config.MODELS_DIR, exist_ok = True)
            # Get existing versions
            existing_versions = [
                int(folder) for folder in os.listdir(self.config.MODELS_DIR)
                if folder.isdigit() and os.path.isdir(os.path.join(self.config.MODELS_DIR, folder))
            ]
            
            # Set next version
            next_version = max(existing_versions, default=0) + 1
            model_dir = f"{self.config.MODELS_DIR}/{next_version}"
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            model_path = f"{model_dir}/xgb_model.json"
            with open(model_path, 'wb') as f:
                f.write(model_bytes)
            
            self.logger.info(f"Model saved at {model_path}")
            return {
                "model_path": model_path,
                "metrics": model_data["metrics"]
            }
        
        # Set up task dependencies
        data = load_data()
        model_data = fit_model(data)
        eval_data = evaluate_model(model_data)
        results = save_model(eval_data)
        
        return results

# Create DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 10, 25),
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="modelTrainingPipeline",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
) as dag:
    # Initialize pipeline and create tasks
    pipeline = ModelPipeline()
    results = pipeline.create_pipeline(dag)