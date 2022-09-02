# WIP: Example of MLflow models and register functionality using Prefect orchestation tool

![GitHub Actions](https://github.com/MikhailKuklin/mlflow_models_registry_with_prefect/actions/workflows/main.yml/badge.svg)


## Step 1: create conda environment

```sh
conda create -n mlflow_reg-env python=3.9

conda activate mlflow_reg-env
```

## Step 2: install requirements

```sh
pip install -r requirements.txt

#NOTE that Mac M1 requires (on 02-09-2022) installation of 
greenlet from the source code. Read more here: https://github.com/neovim/pynvim/issues/502


```

## Step 3: launch MLFlow UI

*Option 1*: without using tracking server

```sh
cd /notebooks #mlflow should be always launched from the folder with notebooks/scripts

mlflow ui
```

Go to http://127.0.0.1:5000 which will open UI for tracked experiments.

*Option 2*: with local tracking server

```sh
cd /notebooks #mlflow should be always launched from the folder with notebooks/scripts

mlflow server --backend-store-uri sqlite:///backend.db --default-artifact-root ./artifacts_local
```

## Step 4: run notebook

