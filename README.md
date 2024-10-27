# Telco Customer Churn

## Project Structure

```
├── ariflow                 # Local Airflow Dags
│   ├──dags
│        └── <dags.py>        
├── pipelines               # Docker container dags
│       └── <dags.py>
├── analysis/
│   ├── eda.ipynb         # Exploratory Data Analysis
│   └── modelling.ipynb   # Model Development
├── configs/
│   └── local.yaml        # Configuration files
├── data/
│   └── Telco_customer_churn.xlsx
├── dataLoader.py         # Data processing utilities
├── modelTraining.py
├── connection.txt        # Model training script
├── README.md
└── environment_setup.md  
```

## Analysis Notebooks

### 1. Exploratory Data Analysis ([EDA](analysis/eda.ipynb))
- Comprehensive data exploration
- Univariate and bivariate analysis
- Feature relationship visualization
- Insights into churn factors

### 2. Modeling ([Modelling](analysis/modelling.ipynb))
- Data preprocessing and cleaning
- Feature engineering and selection
- Comparison of 5 binary classification models
- K-fold cross-validation
- Hyperparameter optimization
- Final model selection (XGBoost)

## Python Scripts

### Data Processing (`dataLoader.py`)
- Custom data loader for "Telco Customer Churn: IBM" dataset
- Extensible structure for similar datasets
- Data preprocessing and transformation utilities

### Model Training (`modelTraining.py`)
- Implementation of XGBoost classifier (compatible with any sklearn model)
- Configurable training pipeline
- Integrated with data loader
- Model evaluation on test set
- Logging system for experiments and metrics

### Logging System
Two separate loggers implemented:
1. General Information Logger: Training progress and system information
2. Metrics Logger: Model performance metrics for experiment comparison

## Usage

### Training the Model

Basic usage:
```bash
python modelTraining.py --config configs/local.yaml
```

Custom configuration:
```bash
python modelTraining.py --config path/to/your/config/file
```

The trained model is saved as 'model.json' in the root directory.

## Configuration

Model parameters and training settings can be controlled through YAML configuration files in the `configs/` directory.

Example configuration structure:
```yaml
data_path: data/Telco_customer_churn.xlsx
model:
  training:
    test_size : 0.3
  
  ## selected features after experimentation ##
  features:
    - Contract_Month-to-month
    - Internet Service_Fiber optic
    - Payment Method_Electronic check
    - Internet Service_No
    - Contract_Two year
    - Tenure Months

  ## hyperparameter tuned after experimentation ##
  xgb_params:
    learning_rate: 0.01
    max_depth: 3
    n_estimators: 200
    scale_pos_weight: 3
    enable_categorical": true
  
  save_path : "model.json"
seed : 42
```

## Future Improvements

1. Complete automation of the training pipeline
2. Generalization for different models and datasets
3. Enhanced configuration options

## Notes
- The project provides a foundation for automated ML pipelines
- Current implementation focuses on XGBoost but can be extended
- Configuration-driven approach allows for easy experimentation

## Data and Model Pipelines

### Data Extraction
DAG ID : dataExtractionPipeline

- Data Download from Dataset API using HTTP connection
- Data Unzipping and checking for corruptions
- Deleting the zipped file

### Data Processing
DAG ID : dataProcessingPipeline

- Cleaning and Transformation of the dataset namely columns dropping, categorical data mapping (label encoding), feature transformation and data type handling   
- Feature Reduction or Feature selection based on the study given in the config.
- Data Splitting into training and test dataset and saving it to a local storage

### Model Training
DAG ID: modelTrainingPipeline

- Loading the data onto the RAM and returning it in form of Dict[List]for serialization.
- Model Fitting to the data loaded and saving it to a temporary path in `base64` due to serielization issue.
- Model evaluation on the Test set and logging the complete classification report that includes F1 score, Recall, Precision and accuracy for both the classes.
- Saving the model in a versioned fashion. It checks the existing subfolders and creates a newer version subfolder. 
- This DAG has been written inside a class to better encapsulate the relationships among tasks. 

## Usage

Pipelines have been implemented for local running as well as running on docker containers.

### 1. Running Locally using Airflow

You should have `apache-airflow` install in your **environment**. To install it go to : [Airflow Installation](https://airflow.apache.org/docs/apache-airflow/stable/installation/index.html).

1. Run the following command in a terminal from root:
```bash 
            cp -r airflow/dags/*.py ~/airflow/dags
```
2. Start the airflow webserver in a new terminal:
```bash
            airflow webserver -p <PORT>
```
If the default PORT : `8080` is available, you can runn the command without port flag.

3. Start the airflow scheduler in another new terminal:

```bash
            airflow scheduler
```

Note that you would need to create a user and initialize the airflow db if you are new to Airflow. You can see [environment_setup.md](environment_setup.md) for details or visit airflow website.

Next thing, you can go to the [localhost:\<PORT>](localhost:8080) where you will see the UI. You will see multiple DAGS there and Trigger the DAGS manually in order of :
```
dataExtractionPipeline >> dataProcessingPipeline >> modelTrainingPipeline

```
For logs you can access airflow db. You can see the trained models and data folder inside `~/airflow/.` directory.

### 2. Running on Docker Container using Astronomer

Astronomer is very popular tool to deploy pipelines to a very large scale. It's reliability, simplicity, good documentation and attractive features made me choose it for this use case. 

For getting started with it, you can go to [Getting Started with Airflow](https://www.astronomer.io/docs/learn/get-started-with-airflow).\

Make sure you have `docker-compose` installed and running.

Follow the steps:

```bash
cd pipelines/
astro dev start
```

It will automatically open Airflow UI in a web browser. There you will need to log in with your created user id and password.

Next you will need to make a HTTP connection:

The details are provided in the [connections.txt](connections.txt). 

After this you can trigger the DAGS in following order.  

```
dataExtractionPipeline >> dataProcessingPipeline >> modelTrainingPipeline

```

You can see the airflow logs in docker container at `/usr/local/airflow/.`. Also you can see the Event logs in the UI for every Task. 

## Future Improvements

1. Post all the logs in a persistent memory like a Postgres DB
2. Fetch the config files from an API
3. Post the Model like objects in an object storage buckets e.g. S3
4. Develop Inference Pipeline
5. Include Unittests
6. Add Documentation