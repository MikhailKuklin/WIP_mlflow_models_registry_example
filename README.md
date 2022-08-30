# Example of MLflow models and register functionality

Repo contains notebook (/notebooks) which demonstrate functionality of MLFlow models and register:

Step 1: create conda environment
conda create -n mlflow-reg-env python=3.9

conda activate mlflow-reg-env
Step 2: install requirements
pip install -r requirements
Step 3: launch MLFlow UI
Option 1: without using tracking server

cd /notebooks #mlflow should be always launched from the folder with notebooks/scripts

mlflow ui
Go to http://127.0.0.1:5000 which will open UI for tracked experiments.

Option 2: with local tracking server

cd /notebooks #mlflow should be always launched from the folder with notebooks/scripts

mlflow server --backend-store-uri sqlite:///backend.db --default-artifact-root ./artifacts_local
Step 4: run notebook
