# Predicting Patient's Diagnosis of Heart Disease

**Problem Statement**: 
For a given historical medicine data, I will demonstrate, how health parameters influence on your heart diseases. I will use the [Cardiovascular Diseases Dataset](https://www.kaggle.com/datasets/waalbannyantudre/south-african-heart-disease-dataset). The dataset is about coronary heart disease (CHD) obtained from the Coronary Risk Factor Study conducted in South Africa in 1983. It contains features such as: blood pressure, cumulative tobacco, cholesterol, overweight, family history of heart disease, Type-A behavior, excessive fat accumulation, alcohol consumption, age and finally heart disease. The objective is to predict the heart disease based on health parameters using the other features. In order to achieve this in a real-world scenario, I use ZenML framework to build a production-ready pipeline to predict the patient's sickness based on their health parameters. 

## What Tools Are Used in the Project?
* ZenML: Manages pipelines and interacts with tools like MLflow for deployment, tracking and more
* Apache Spark: Speeds up data ingestion and cleaning operations.
* MongoDB: Facilitates fast data operations and storage.
* Docker: manages MongoDB and Apache Spark containers without need to install them locally
* Python Libraries: libraries like: `numpy`, `pandas`, `scikit-learn`, `optuna`, `matplotlib` are used for data transformations, operations, applying machine learning algorithms, optimizing model hyperparameters and data visualization

## What Elements Does the Project Have?
* Data ingesting for loading and preparing data for further processing
* Data cleaning to preprocess the data and make it suitable for prediction
* Data transformations to reduce the number of dimensions in the data
* Model training for prediction and optimizing its hyperparameters
* Model evaluation to assess model performance and visualize feature importance
* Model deployment to continuously predicting and deploying the model, while testing it simultaneously
* Inference process to make prediction using a trained model on a new dataset

## Requirements:
* Python version 3.10.12
* Docker Desktop, (installation guide [here](https://docs.docker.com/desktop/install/windows-install/))
* Java version 11.0.24

## Prepare steps:
First of all you need to copy this repository and install necessary packages:

```bash
git clone https://github.com/WiktorPodwin/ProjectML.git
cd ProjectML
pip install -r requirements.txt
```

Now it is time for creating Docker image, which contains MongoDB and Apache Spark containers. Make sure Docker Desktop is running, then execute:
``` bash
docker-compose up -d
```

ZenML offers a React-based dashboard, where you can monitor your stacks, stack components and pipeline DAGs. To access this you need to tun these commands:
``` bash
pip install zenml['server']
zenml up
```

We need to install MLflow, which will need to be integrated with ZenML, to achieve this run:
``` bash
zenml integration install mlflow -y
```

To correctly working project we need to configure ZenML stack, let's create experiment-tracker, model-deployer and stack:
``` bash
zenml experiment-tracker register mlflow_experiment_tracker --flavor=mlflow
zenml model-deployer register mlflow_model_deployer --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow_model_deployer -e mlflow_experiment_tracker --set
```

## The Solution
In  a real-world workflow we face a more difficult scenario than just training the model once and predicting patient heart diseases. This requires an end-to-end pipeline for continuously predicting and deploying the machine learning model. This pipeline can be deployed not only in a local environment, but also on the cloud to scale up, to meet our needs, and ensure that every run tracks parameters and data. It handles everything from raw data input to feature extraction, model training and prediction outputs.

### Training Pipeline
This is a basic pipeline, that includes several steps, with the most important components logged into MLflow:
- `ingest_data`: This step ingests the data and creates a DataFrame
- `clean_data`: This step cleans the data, divides it into training and testing datasets and standardizes values
- `transform_data`: This step extracts features using `scikit-learn` algorithms
- `model_train`: This step trains the model and finds its hyperparameters
- `model_evaluate`: This step evaluates the model's performance

### Deployment Pipeline
Here is a deployment pipeline `deployment_pipeline.py` that builds on the training pipeline and implements a continuous deployment workflow. This pipeline handles data ingestion and processing, trains a model, and then redeploys the prediction server, if the model meets evaluation criteria. The criterion is a configurable threshold on the training accuracy. The first five steps of the pipeline are identical to those in the training pipeline, but here are added the following steps:

- `deployment_trigger`: This step verifies if the newly trained model satisfies the deployment criteria.
- `model_deployer`: If the criteria are met, this step deploys the model as a service using MLflow.
In the deployment pipeline, ZenML's MLflow tracking integration is utilized to log transformation methods, hyperparameter values, the trained model and model evaluation metrics as MLflow experiment tracking artifacts into the local MLflow backend. Additionally, this pipeline starts a local MLflow deployment server to serve the latest model if it meets the accuracy threshold.

The MLflow deployment server operates locally as a daemon process, continuing to run in the background even after the execution is complete. Whenever a new pipeline run produces a model that meets the accuracy threshold, the pipeline automatically updates the active MLflow deployment server to serve the new model, replacing the previous one.

### Streamlit App
This end-to-end application uses deployment pipeline to retrieve the allready deployed model for new data prediction. This fragment of code is responsible for this:
``` python
service = prediction_service_loader(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            running=False
        )
service.predict(data)
```

## How to launch it?
* Training Pipeline:
```bash 
python run_pipeline.py
```
* Deployment Pipeline:
```bash
python run_deployment.py
```
* Streamlit App
``` bash
streamlit run streamlit_app.py
```