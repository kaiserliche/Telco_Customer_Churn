# Environment Setup

## Option 1: Using Conda (Recommended)

### Method A: Create from environment.yml

```yaml
name: telco-churn
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - apache-airflow>=2.7.0
  - matplotlib>=3.7.0
  - numpy>=1.24.0
  - pandas>=2.0.0
  - seaborn>=0.12.0
  - scikit-learn>=1.3.0
  - xgboost>=2.0.0
  - pyyaml>=6.0.0
  - pip
```

Create and activate the environment:
```bash
conda env create -f environment.yml
conda activate telco-churn
```

### Method B: Manual Creation

```bash
# Create new environment
conda create -n telco-churn python=3.12

# Activate environment
conda activate telco-churn

# Install packages
conda install -c conda-forge apache-airflow
conda install matplotlib numpy pandas seaborn scikit-learn xgboost pyyaml openpyxl
```

## Option 2: Using Python venv

1. Create and activate virtual environment:
```bash
# Create environment
python -m venv venv

# On Unix/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

2. Install packages using requirements.txt:

Create `requirements.txt`:
```txt
apache-airflow>=2.7.0
matplotlib>=3.7.0
numpy>=1.24.0
pandas>=2.0.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=2.0.0
pyyaml>=6.0.0
```

Install requirements:
```bash
pip install -r requirements.txt
```

## Notes

1. **Apache Airflow Setup**: After installation, initialize the Airflow database:
```bash
# Set the Airflow home directory
export AIRFLOW_HOME=~/airflow

# Initialize the database
airflow db init

# Create an admin user (follow the prompts)
airflow users create \
    --username admin \
    --firstname YourName \
    --lastname YourLastName \
    --role Admin \
    --email your@email.com
```

2. **Package Versions**: 
- The versions specified are recommended minimums
- Use `conda list` or `pip freeze` to check installed versions
- Update packages if needed: `conda update <package-name>` or `pip install --upgrade <package-name>`

3. **Additional Development Tools** (Optional):
```bash
conda install -c conda-forge jupyter notebook jupyterlab  # For notebook development
conda install -c conda-forge black flake8 pylint  # For code formatting and linting
```

4. **Environment Management**:
```bash
# Export environment
conda env export > environment.yml  # For conda
pip freeze > requirements.txt       # For venv

# Remove environment
conda env remove -n telco-churn    # For conda
deactivate && rm -rf venv          # For venv
```

5. **Troubleshooting**:
- If you encounter conflicts in conda, try installing packages one at a time
- For Airflow, ensure you're using a compatible Python version (3.8-3.12)
- XGBoost may require additional setup for GPU support
- There maybe some issue on openpyxl installation. Do it manually if occurs.