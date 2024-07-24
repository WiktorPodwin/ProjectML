FROM python:3.11-slim

WORKDIR /usr/src/app

COPY . .

RUN mkdir -p /tmp/alphafold

RUN pip install --no-cache-dir -r requirements.txt

RUN zenml init 

RUN zenml experiment-tracker register mlflow_tracker_customer --flavor=mlflow && \
zenml model-deployer register mlflow_customer --flavor=mlflow && \
zenml stack register mlflow_stack_customer \
-a default \
-o default \
-d mlflow_customer \
-e mlflow_tracker_customer \
--set

EXPOSE 8237

CMD ["python", "run_deployment.py", "--config", "deploy_and_predict"]
