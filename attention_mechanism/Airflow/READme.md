# Implement Airflow along with MLFlow for the ML model

**Prerequisites**:

1. Python (3.8+) & pip
2. Docker & Docker Compose (allocate maybe 4GB RAM, 2 CPUs if
   possible)
3. ML model code
4. MLFlow tracking server : decide where MLFlow trackign
                            server will run from these options:
                            - **Locally in Docker**: Run it as
                                   another container alongside Airflow (good for local dev.)
                            - **Remote Server**: Run it on s
                                   small cloud VM.
                            - **Managed Service**: Use a
                                   managed MlFlow service (e.g: Databricks)

**Setup Airflow Environment (Local Docker Compose)**:

1. **Prepare Airflow Directories**:

        - Make a directory for Airflow
        - cd to Airflow and make sub-directories for dags,
          logs, plugins and config.
        - Set Airflow User ID to avoid permission issues with
          Docker volume mounts. **(echo -e "AIRFLOW_UID=$(id -u)" > .env)**

2. Download **docker-compose.yaml** for a specific Airflow
   version (2.9.2)

3. Initialize Airflow Database - (docker compose up airflow-init) runs a temporary container to set up the metadata DB.

4. Start Airflow services: Run airflow webserver, scheduler, triggerer, etc. (command: docker compose up -d # -d runs services in the background).

- check status with **docker compose ps**

5. Access Airflow UI: <http://localhost:8080>. Login with username **airflow** and password **airflow**

# Setup MLFlow Tracking Server (Local Docker)

1. Add MLFlow to **docker-compose.yaml** and configure Airflow services to know about the MLFlow server.

2. Restart Airflow with MLFlow
        - docker compose down # stop current services
        - docker compose up -d # start all services including
                                 # MLFlow

3. Access MLFlow UI: Open **http:localhost:5000**

# Create Airflow DAG with MLFlow integration

1. Create DAG File: Inside the **~/airflow/dags** directory, create a python file, e.g: stock_pred_dag.py

2. Define DAG structure. Use operators like **PythonOperator**

# Run and Monitor

1. Place DAG in ~/airflow/dags directory for hte scheduler to
   detect.

2. Use Airflow UI:
    - Go to **<http://localhost:8080>**
    - Find the **stock_pred_mlflow_dag** DAG
    - It will initially be paused; toggle to unswitch.
    - Click play button to trigger DAG to run it manually.
    - Watch the tasks execute in the grid/graph view. Click on
      tasks to see logs.
3. Use MLFlow UI:
        - Go to **<http://localhost:5001>**
        - Look for the "Stock Predictor Attention Model"
          experiment.
        - Check for new runs, parameters, metrics, tags and
          artifacts logged by **_train_model** task.
        - If the **_register_model** task runs successfully,
          the model will appear under "Models" tab
