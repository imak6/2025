from __future__ import annotations

import pendulum
import logging
import pandas as pd
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

# Import functions defined in stock_prediction_model.py:
# import sys
# sys.path.append('/Users/ajithraghavan/Dropbox/My Mac (Ajiths-MBP.hitronhub.home)/Desktop/2025/attention_mechanism/Airflow/stock_prediction_model.py')
from ..plugins.stock_prediction_model import (
    setup_directories_task,
    fetch_data_task,
    preprocess_data_task,
    scale_data_task,
    create_sequences_task,
    split_data_task,
    train_model_task,
    evaluate_model_task,
    register_model_task,
    get_last_sequence_task,
    predict_future_task,
    visualize_results_task
)

# Configuration
TICKER = 'AAPL'
START_DATE = '2022-01-01'
# Using current date (approx) as end_date to get most recent data for prediction
# Note: yfinance `end` is exclusive for daily data
# Set end_date a bit into the future to ensure we get today's data if available
TODAY = pd.Timestamp.now().strftime('%Y-%m-%d')
END_DATE = (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d') # Ensures we get up to 'today'
SEQUENCE_LENGTH = 60
PREDICTION_DAYS = 4
TRAIN_SPLIT_RATIO = 0.8
EPOCHS = 50 # Reduced epochs for faster example, increase as needed (original was 200)
BATCH_SIZE = 256 # Adjusted batch size

# MLFlow configuration
MLFLOW_EXPERIMENT_NAME = "Stock_Pred_LSTMAttention"
MLFLOW_MODEL_NAME = "LSTM_Attention"
MLFLOW_RUN_NAME = f"stock_pred_run_{TODAY}"
ACCURACY_THRESHOLD = 0.80 # example evaluation threshold

# Assume a base data directory accessible by Airflow workers
# In docker compose, map the volume to this path inside containers
BASE_DATA_DIR = '/opt/airflow/data/stock_predictor'


log = logging.getLogger(__name__)

# --- DAG Definition ---
with DAG(
    dag_id='stock_predictor_mlflow_v1', # Changed name slightly
    start_date=pendulum.datetime(2024, 4, 21, tz="UTC"), # Use a fixed past date
    schedule=None, # Manual trigger
    catchup=False,
    tags=['ml', 'stocks', 'mlflow', 'attention'],
) as dag:

    start = EmptyOperator(task_id='start')

    # Task to create directories for this run
    setup_task = PythonOperator(
        task_id='setup_directories_task',
        python_callable=setup_directories_task
        )

    # Task definitions using the refactored functions
    fetch_task = PythonOperator(
        task_id='fetch_data_task',
        python_callable=fetch_data_task,
        op_kwargs={'ticker': TICKER, 'start_date': START_DATE, 'end_date': END_DATE}
        )

    preprocess_task = PythonOperator(
        task_id='preprocess_data_task',
        python_callable=preprocess_data_task
        )

    scale_task = PythonOperator(
        task_id='scale_data_task',
        python_callable=scale_data_task
        )

    sequences_task = PythonOperator(
        task_id='create_sequences_task',
        python_callable=create_sequences_task,
        op_kwargs={'sequence_length': SEQUENCE_LENGTH}
        )

    split_task = PythonOperator(
        task_id='split_data_task',
        python_callable=split_data_task,
        op_kwargs={'train_split_ratio': TRAIN_SPLIT_RATIO}
        )

    train_task = PythonOperator(
        task_id='train_model_task',
        python_callable=train_model_task,
        op_kwargs={'sequence_length': SEQUENCE_LENGTH, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE}
        )

    evaluate_task = PythonOperator(
        task_id='evaluate_model_task',
        python_callable=evaluate_model_task,
        op_kwargs={'accuracy_threshold': ACCURACY_THRESHOLD} # Pass your threshold
        )

    register_task = PythonOperator(
        task_id='register_model_task',
        python_callable=register_model_task,
        op_kwargs={'mlflow_model_name': MLFLOW_MODEL_NAME}
        # Consider adding trigger_rule=TriggerRule.ALL_SUCCESS if evaluate_task can be skipped but registration should still check
        )

    get_sequence_for_pred_task = PythonOperator(
        task_id='get_last_sequence_task',
        python_callable=get_last_sequence_task,
        op_kwargs={'sequence_length': SEQUENCE_LENGTH}
        )

    predict_task = PythonOperator(
        task_id='predict_future_task',
        python_callable=predict_future_task,
        op_kwargs={'prediction_days': PREDICTION_DAYS}
        )

    visualize_task = PythonOperator(
        task_id='visualize_results_task',
        python_callable=visualize_results_task,
        op_kwargs={'sequence_length': SEQUENCE_LENGTH} # Used for plotting historical actuals
        )

    end = EmptyOperator(task_id='end')

    # Define task dependencies
    setup_task >> fetch_task >> preprocess_task >> scale_task >> sequences_task >> split_task >> train_task

    # Evaluation and Registration Path
    train_task >> evaluate_task >> register_task

    # Prediction Path (depends on scaled data, last sequence, and trained model)
    [scale_task, train_task] >> get_sequence_for_pred_task >> predict_task

    # Visualization Path (depends on processed data for history, predictions, and MLflow run_id from training)
    [preprocess_task, predict_task, train_task] >> visualize_task

    # Define end dependencies
    [register_task, visualize_task] >> end
