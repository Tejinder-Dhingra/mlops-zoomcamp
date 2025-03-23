conda activate exp-tracking-env
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts