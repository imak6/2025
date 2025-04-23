import os
import joblib
import tensorflow as tf
import keras
import yfinance as yf
import mplfinance as mpf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging # For better information display
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import (Dense, Flatten, LSTM, Input,
                                     AdditiveAttention, Multiply,
                                     BatchNormalization, Dropout)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error

# MLFlow
import mlflow
import mlflow.keras # MlFlow keras integration
import mlflow.tensorflow # mlFlow tensorflow integration
import json

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

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
log.info(f"Tensorflow version: {tf.__version__}")
log.info(f"Running DAG for date: {TODAY}")


# Utility functions

# def flatten_multiindex_columns(df):
#     """Flattens MultiIndex columns if present."""
#     if isinstance(df.columns, pd.MultiIndex):
#         logging.debug("Flattening MultiIndex columns...")
#         # Use the first level of the MultiIndex
#         df.columns = df.columns.get_level_values(0)
#         # Optional: Handle potential duplicate column names if needed here
#     return df

def _create_sequences_from_data(scaled_data, sequence_length):
    """Helper to create sequences.""""
    X,y = [], []
    if scaled_data is None or len(scaled_data) <= sequence_length:
        raise ValueError(f"Not enough data ({len(scaled_data) if scaled_data is not None else 0} points) to create sequences of length {sequence_length}.")
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y)

def _build_keras_model(sequence_length):
    """Builds the Keras Functional API model."""
    log.info("Building LSTM Attention model...")
    input_layer = Input(shape=(sequence_length, 1))
    lstm_layer1 = LSTM(50, return_sequences=True)(input_layer)
    lstm_layer2 = LSTM(50, return_sequences=True)(lstm_layer1)

    attention_layer = AdditiveAttention(name='attention_weight')([lstm_layer2, lstm_layer2])
    attention_result = attention_layer([lstm_layer2, lstm_layer2])
    multiply_layer = Multiply()([lstm_layer2, attention_result])
    context_vector = Flatten()(multiply_layer)

    dense_out = Dropout(0.2)(context_vector)
    dense_out = BatchNormalization()(dense_out)
    output_layer = Dense(1)(dense_out)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    log.info("Model built successfully.")
    model.summary(print_fn=log.info)
    return model

# AIRFLOW Task Functions

def setup_directories_task(**context):
    """Creates necessary directories for the DAG run."""
    run_data_dir = os.path.join(BASE_DATA_DIR, context['dag_run'].run_id,TODAY)
    os.makedirs(run_data_dir, exist_ok=True)
    log.info(f"Created data directory for run: {run_data_dir}")
    # Push the directory path for other tasks to use
    context['ti'].xcom_push(key='run_data_dir', value=run_data_dir)
    return run_data_dir

def fetch_data_task(ticker: str, start_date: str, end_date: str, **context):
    """Downloads stock data from yfinance and saves it to a file."""
    run_data_dir = context['ti'].xcom_pull(task_ids='setup_directories_task', key='run_data_dir')
    raw_data_path = os.path.join(run_data_dir, f"{ticker}_raw_data.csv")
    log.info(f"Fetching data for {ticker} from {start_date} to {end_date}")

    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            log.info("Flattening MultiIndex columns...")
            data.columns = data.columns.get_level_values(0)

        log.info(f"Data downloaded successfully. shape: {data.shape}")
        if data.empty:
            raise ValueError("No data downloaded. Check ticker and date range.")
        if 'Close' not in data.columns:
            raise ValueError("Expected 'Close' column in the data.")

        data.to_csv(raw_data_path, index=True)
        log.info(f"Data saved to {raw_data_path}")
        return raw_data_path
    except Exception as e:
        log.error(f"Error fetching data: {e}")
        raise

def preprocess_data_task(raw_data_path: str, **context):
    """Reads raw data, handles NaNs, ensures numeric, saves processed data."""
    run_data_dir = context['ti'].xcom_pull(task_ids='setup_directories_task', key='run_data_dir')
    raw_data_path = context['ti'].xcom_pull(task_ids='fetch_data_task')
    processed_data_path = os.path.join(run_data_dir, f"{TICKER}_processed_data.csv")

    log.info(f"Preprocessing data from {raw_data_path}")
    try:
        data = pd.read_csv(raw_data_path, index_col='Date', parse_dates=True)

        initial_missing = data.isna().sum().sum()
        if initial_missing > 0:
            log.info(f"Initial NaN count: {initial_missing}. Interpolating...")
            data = data.fillna(method='ffill').fillna(method='bfill')
            final_missing = data.isna().sum().sum()
            if final_missing > 0:
                log.warning(f"Final NaN count: {final_missing}. Some data may still be missing.")
                data.dropna(inplace=True)
            else:
                log.info("All NaNs successfully handled.")
        else:
            log.info("No NaNs found in the data.")
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        data.dropna(subset=numeric_columns, inplace=True)
        data.to_csv(processed_data_path)
        log.info(f"Processed data saved to {processed_data_path}")
        return processed_data_path
        # return data
    except Exception as e:
        log.error(f"Error preprocessing data: {e}")
        raise

def scale_data_task(**context):
    """Reads processed data, scales 'Close' price, saves scaler and scaled data."""
    run_data_dir = context['ti'].xcom_pull(task_ids='setup_directories_task', key='run_data_dir')
    processed_data_path = context['ti'].xcom_pull(task_ids='preprocess_data_task')
    scaler_path = os.path.join(run_data_dir, f"{TICKER}_scaler.joblib")
    scaled_data_path = os.path.join(run_data_dir, f"{TICKER}_scaled_data.npy")

    log.info(f"Scaling data from {processed_data_path}")
    try:
        data = pd.read_csv(processed_data_path, index_col='Date', parse_dates=True)
        if 'Close' not in data.columns:
            raise ValueError("Expected 'Close' column in the data.")

        scaler = MinMaxScaler(feature_range=(0, 1))
        closing_prices = data['Close'].values.reshape(-1, 1)
        scaled_data = scaler.fit_transform(closing_prices)

        joblib.dump(scaler, scaler_path)
        np.save(scaled_data_path, scaled_data)

        log.info(f"Scaler saved to {scaler_path}")
        log.info(f"Scaled data saved to {scaled_data_path}")
        # return scaled_data_path

        # Push paths via XCom for use in other tasks
        ti = context['ti']
        ti.xcom_push(key='scaler_path', value=scaler_path)
        ti.xcom_push(key='scaled_data_path', value=scaled_data_path)
        return scaler_path, scaled_data_path
        # return scaler, scaled_data
    except Exception as e:
        log.error(f"Error scaling data: {e}")
        raise

def create_sequences_task(sequence_length: int, **context):
    run_data_dir = context['ti'].xcom_pull(task_ids='setup_directories_task', key='run_data_dir')
    scaled_data_path = context['ti'].xcom_pull(task_ids='scale_data_task', key='scaled_data_path')
    X_path = os.path.join(run_data_dir, f"X_sequences_{TICKER}.npy")
    y_path = os.path.join(run_data_dir, f"y_sequences_{TICKER}.npy")

    log.info(f"Creating sequences from: {scaled_data_path}")
    scaled_data = np.load(scaled_data_path)
    X, y = _create_sequences_from_data(scaled_data, sequence_length)

    np.save(X_path, X)
    np.save(y_path, y)

    log.info(f"X sequences saved to: {X_path} (Shape: {X.shape})")
    log.info(f"y sequences saved to: {y_path} (Shape: {y.shape})")

    # Push paths via XComs
    ti = context['ti']
    ti.xcom_push(key='X_path', value=X_path)
    ti.xcom_push(key='y_path', value=y_path)

def split_data_task(train_split_ratio: float, **context):
    """Splits sequence data into train and test sets."""
    run_data_dir = context['ti'].xcom_pull(task_ids='setup_directories_task', key='run_data_dir')
    X_path = context['ti'].xcom_pull(task_ids='create_sequences_task', key='X_path')
    y_path = context['ti'].xcom_pull(task_ids='create_sequences_task', key='y_path')

    X_train_path = os.path.join(run_data_dir, f"X_train_{TICKER}.npy")
    y_train_path = os.path.join(run_data_dir, f"y_train_{TICKER}.npy")
    X_test_path = os.path.join(run_data_dir, f"X_test_{TICKER}.npy")
    y_test_path = os.path.join(run_data_dir, f"y_test_{TICKER}.npy")

    log.info(f"Splitting data from {X_path} and {y_path}")
    X = np.load(X_path)
    y = np.load(y_path)

    train_size = int(len(X) * train_split_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    np.save(X_train_path, X_train)
    np.save(y_train_path, y_train)
    np.save(X_test_path, X_test)
    np.save(y_test_path, y_test)

    log.info(f"Train/Test sets saved. X_train: {X_train.shape}, X_test: {X_test.shape}")

    # Push paths via XComs
    ti = context['ti']
    ti.xcom_push(key='X_train_path', value=X_train_path)
    ti.xcom_push(key='y_train_path', value=y_train_path)
    ti.xcom_push(key='X_test_path', value=X_test_path)
    ti.xcom_push(key='y_test_path', value=y_test_path)

def train_model_task(sequence_length: int, epochs: int, batch_size: int, **context):
    """Builds, trains the model and logs results to MLflow."""
    X_train_path = context['ti'].xcom_pull(task_ids='split_data_task', key='X_train_path')
    y_train_path = context['ti'].xcom_pull(task_ids='split_data_task', key='y_train_path')

    log.info("Starting model training task...")
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)

    # Ensure X_train is 3D for LSTM
    if X_train.ndim == 2:
         X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    # --- MLflow Integration ---
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME) # Ensure experiment exists
    with mlflow.start_run(run_name=f"airflow_train_{context['dag_run'].run_id}") as run:
        run_id = run.info.run_id
        log.info(f"MLflow Run ID: {run_id}")

        # Log parameters
        params = {
            'ticker': TICKER, 'start_date': START_DATE, 'end_date': END_DATE, # Log data params
            'sequence_length': sequence_length, 'train_split_ratio': TRAIN_SPLIT_RATIO, # Log structure params
            'epochs': epochs, 'batch_size': batch_size, # Log training params
            'model_type': 'LSTM_Attention'
        }
        mlflow.log_params(params)
        log.info(f"Logged Params: {params}")

        # Build and Train
        model = _build_keras_model(sequence_length)
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min', restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2, # Using fixed validation split here
            callbacks=[early_stopping, mlflow.keras.MlflowCallback(run=run)], # Use MLflow callback for auto-logging metrics/epochs
            verbose=2 # Reduce verbosity for logs
        )
        log.info("Model training finished.")

        # Log final validation loss (optional, as callback logs per epoch)
        final_val_loss = history.history['val_loss'][-1]
        mlflow.log_metric("final_val_loss", final_val_loss)
        log.info(f"Final Validation Loss: {final_val_loss:.4f}")

        # Log model artifact using MLflow's Keras integration
        # This saves the model in MLflow's format under the 'model' artifact path
        model_artifact_path = "model" # Standard path MLflow uses
        mlflow.keras.log_model(model, artifact_path=model_artifact_path, registered_model_name=None) # Don't register here yet
        log.info(f"Model logged to MLflow artifact path: {model_artifact_path}")

        # Log a tag
        mlflow.set_tag("triggered_by", "airflow_dag")
        mlflow.set_tag("dag_run_id", context['dag_run'].run_id)

    log.info("Training task and MLflow run finished.")

    # Push run_id and model path for downstream tasks
    ti = context['ti']
    ti.xcom_push(key='mlflow_run_id', value=run_id)
    ti.xcom_push(key='model_artifact_path', value=model_artifact_path) # The relative path logged

def evaluate_model_task(accuracy_threshold: float, **context):
    """Loads model from MLflow run, evaluates on test set, logs test metrics."""
    X_test_path = context['ti'].xcom_pull(task_ids='split_data_task', key='X_test_path')
    y_test_path = context['ti'].xcom_pull(task_ids='split_data_task', key='y_test_path')
    scaler_path = context['ti'].xcom_pull(task_ids='scale_data_task', key='scaler_path')
    run_id = context['ti'].xcom_pull(task_ids='train_model_task', key='mlflow_run_id')
    model_artifact_path = context['ti'].xcom_pull(task_ids='train_model_task', key='model_artifact_path')

    if not all([X_test_path, y_test_path, scaler_path, run_id, model_artifact_path]):
        raise ValueError("Missing required inputs from upstream tasks via XCom.")

    log.info(f"Evaluating model from MLflow run: {run_id}, artifact path: {model_artifact_path}")
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    scaler = joblib.load(scaler_path)

    # Ensure X_test is 3D
    if X_test.ndim == 2:
         X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Load model from MLflow run
    model_uri = f"runs:/{run_id}/{model_artifact_path}"
    loaded_model = mlflow.keras.load_model(model_uri)

    # Perform evaluation
    test_loss = loaded_model.evaluate(X_test, y_test, verbose=0)
    log.info(f"Test loss (MSE): {test_loss:.4f}")

    y_pred_scaled = loaded_model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    log.info(f"Test MAE: {mae:.4f}")
    log.info(f"Test RMSE: {rmse:.4f}")

    # --- Log test metrics back to the SAME MLflow run ---
    # Note: We don't use start_run() here, just log to the existing run
    try:
        with mlflow.start_run(run_id=run_id, nested=True): # Use nested=True if logging inside existing run context is needed
            mlflow.log_metric("test_loss_mse", test_loss)
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_rmse", rmse)
            # Add accuracy if relevant and calculated
            # accuracy = calculate_your_accuracy(y_test_actual, y_pred) # Implement this
            # mlflow.log_metric("test_accuracy", accuracy)
            log.info(f"Logged test metrics to MLflow run {run_id}")
            # evaluation_passed = accuracy >= accuracy_threshold
            evaluation_passed = True # Placeholder - use your real metric check
    except Exception as e:
        log.error(f"Failed to log evaluation metrics to MLflow run {run_id}: {e}")
        evaluation_passed = False # Fail evaluation if logging fails

    log.info(f"Evaluation passed threshold: {evaluation_passed}")
    context['ti'].xcom_push(key='evaluation_passed', value=evaluation_passed)
    return evaluation_passed

def register_model_task(mlflow_model_name: str, **context):
    """Conditionally registers the model in MLflow Model Registry."""
    evaluation_passed = context['ti'].xcom_pull(task_ids='evaluate_model_task', key='evaluation_passed')
    run_id = context['ti'].xcom_pull(task_ids='train_model_task', key='mlflow_run_id')
    model_artifact_path = context['ti'].xcom_pull(task_ids='train_model_task', key='model_artifact_path')

    if not run_id or not model_artifact_path:
         raise ValueError("Missing model info from training task via XCom.")

    if evaluation_passed:
        model_uri = f"runs:/{run_id}/{model_artifact_path}"
        log.info(f"Registering model {model_uri} as {mlflow_model_name}")
        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=mlflow_model_name
            )
            log.info(f"Model registered: {registered_model.name} version {registered_model.version}")
            # Optionally push version number
            context['ti'].xcom_push(key='registered_model_version', value=registered_model.version)
        except Exception as e:
            log.error(f"Failed to register model: {e}")
            raise # Fail the task if registration fails when expected
    else:
        log.info("Evaluation failed. Skipping model registration.")

def get_last_sequence_task(sequence_length: int, **context):
    """Gets the last sequence from scaled data for prediction."""
    run_data_dir = context['ti'].xcom_pull(task_ids='setup_directories_task', key='run_data_dir')
    scaled_data_path = context['ti'].xcom_pull(task_ids='scale_data_task', key='scaled_data_path')
    last_sequence_path = os.path.join(run_data_dir, f"last_sequence_{TICKER}.npy")

    log.info(f"Getting last sequence from: {scaled_data_path}")
    scaled_data = np.load(scaled_data_path)

    if scaled_data is None or len(scaled_data) < sequence_length:
        raise ValueError(f"Not enough scaled data ({len(scaled_data)}) for sequence length {sequence_length}.")

    last_sequence = scaled_data[-sequence_length:]
    np.save(last_sequence_path, last_sequence)
    log.info(f"Last sequence saved to: {last_sequence_path}")
    return last_sequence_path


def predict_future_task(prediction_days: int, **context):
    """Loads model and predicts future prices."""
    run_data_dir = context['ti'].xcom_pull(task_ids='setup_directories_task', key='run_data_dir')
    scaler_path = context['ti'].xcom_pull(task_ids='scale_data_task', key='scaler_path')
    last_sequence_path = context['ti'].xcom_pull(task_ids='get_last_sequence_task')
    run_id = context['ti'].xcom_pull(task_ids='train_model_task', key='mlflow_run_id')
    model_artifact_path = context['ti'].xcom_pull(task_ids='train_model_task', key='model_artifact_path')

    if not all([scaler_path, last_sequence_path, run_id, model_artifact_path]):
        raise ValueError("Missing required inputs from upstream tasks via XCom for prediction.")

    predictions_path = os.path.join(run_data_dir, f"predictions_{TICKER}.csv")

    log.info(f"Loading model from run {run_id} for prediction...")
    model_uri = f"runs:/{run_id}/{model_artifact_path}"
    loaded_model = mlflow.keras.load_model(model_uri)
    scaler = joblib.load(scaler_path)
    last_sequence_scaled = np.load(last_sequence_path)
    sequence_length = last_sequence_scaled.shape[0] # Infer sequence length

    log.info(f"Predicting next {prediction_days} days...")
    predicted_prices_scaled = []
    current_batch = last_sequence_scaled.reshape((1, sequence_length, 1))

    for _ in range(prediction_days):
        next_prediction_scaled = loaded_model.predict(current_batch, verbose=0)
        predicted_prices_scaled.append(next_prediction_scaled[0, 0])
        next_prediction_reshaped = next_prediction_scaled.reshape((1, 1, 1))
        current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)

    predicted_prices = scaler.inverse_transform(np.array(predicted_prices_scaled).reshape(-1, 1))
    log.info(f"Predicted prices: {predicted_prices.flatten()}")

    # Create predictions DataFrame
    processed_data_path = context['ti'].xcom_pull(task_ids='preprocess_data_task')
    original_data = pd.read_csv(processed_data_path, index_col='Date', parse_dates=True)
    last_known_date = original_data.index[-1]
    prediction_dates = pd.date_range(
        start=last_known_date + pd.offsets.BDay(1),
        periods=prediction_days,
        freq='B' # Use 'B' for Business Day frequency
    )
    predictions_df = pd.DataFrame(
        {'Predicted Price': predicted_prices.flatten()},
        index=prediction_dates
    )
    predictions_df.index.name = 'Date'

    predictions_df.to_csv(predictions_path)
    log.info(f"Predictions saved to: {predictions_path}")

    # --- Log predictions as artifact to the SAME MLflow run ---
    try:
        with mlflow.start_run(run_id=run_id, nested=True):
             mlflow.log_artifact(predictions_path, artifact_path="predictions")
             log.info(f"Logged predictions artifact to MLflow run {run_id}")
    except Exception as e:
        log.error(f"Failed to log predictions artifact to MLflow run {run_id}: {e}")
        # Decide if this should fail the task

    return predictions_path


def visualize_results_task(sequence_length: int, **context):
    """Generates plots and logs them to MLflow."""
    processed_data_path = context['ti'].xcom_pull(task_ids='preprocess_data_task')
    predictions_path = context['ti'].xcom_pull(task_ids='predict_future_task')
    run_id = context['ti'].xcom_pull(task_ids='train_model_task', key='mlflow_run_id')

    if not all([processed_data_path, predictions_path, run_id]):
         raise ValueError("Missing required inputs from upstream tasks via XCom for visualization.")

    log.info(f"Generating visualizations for run {run_id}...")
    history_data = pd.read_csv(processed_data_path, index_col='Date', parse_dates=True)
    predictions_df = pd.read_csv(predictions_path, index_col='Date', parse_dates=True)

    # --- Use the visualizer class logic ---
    # You might instantiate the visualizer here or make plotting functions static/standalone
    # For simplicity, let's put plotting logic here directly using matplotlib/mplfinance

    # 1. Line Comparison Plot
    fig_line, ax_line = plt.subplots(figsize=(14, 7))
    actual_plot_data = history_data['Close'].iloc[-sequence_length:] # Plot history length = sequence length
    ax_line.plot(actual_plot_data.index, actual_plot_data.values,
                linestyle='-', marker='.', color='blue', label=f'Actual Data (Last {sequence_length} days)')
    ax_line.plot(predictions_df.index, predictions_df['Predicted Price'],
                linestyle='-', marker='o', color='red', label=f'Predicted Data ({len(predictions_df)} days)')
    ax_line.set_title(f"{TICKER} Stock Price: Actual vs. Predicted")
    ax_line.set_xlabel('Date')
    ax_line.set_ylabel('Price ($)')
    ax_line.legend()
    ax_line.grid(True)
    plt.tight_layout()

    # 2. Candlestick Plot (Optional - depends if needed/mplfinance installed)
    # Ensure history_data has OHLC columns if doing this
    fig_candle = None
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if all(col in history_data.columns for col in required_cols):
        try:
            fig_candle, ax_candle = mpf.plot(
                history_data.iloc[-120:], # Plot last 120 days history
                type='candle', style='yahoo', volume=True,
                title=f'{TICKER} History & {len(predictions_df)}-Day Prediction',
                ylabel='Price ($)', figratio=(15, 7), returnfig=True
            )
            # Add prediction overlay to candlestick
            ax_candle[0].plot(predictions_df.index, predictions_df['Predicted Price'],
                        linestyle='dashed', marker='o', markersize=5, color='red',
                        label=f'Predicted {len(predictions_df)} days')
            ax_candle[0].legend()
        except Exception as e:
            log.error(f"Could not generate mplfinance candlestick plot: {e}")
            fig_candle = None # Ensure fig_candle is None if plotting fails
    else:
        log.warning("OHLCV columns not available in processed data for candlestick plot.")


    # --- Log plots to the SAME MLflow run ---
    try:
        with mlflow.start_run(run_id=run_id, nested=True):
            if fig_line:
                mlflow.log_figure(fig_line, f"plots/{TICKER}_LineComp_{TODAY}.png")
                log.info("Logged line comparison plot to MLflow.")
            if fig_candle:
                mlflow.log_figure(fig_candle, f"plots/{TICKER}_Candlestick_{TODAY}.png")
                log.info("Logged candlestick plot to MLflow.")
    except Exception as e:
        log.error(f"Failed to log plots to MLflow run {run_id}: {e}")
        # Decide if this should fail the task
    finally:
        # Close plots to free memory
        if fig_line: plt.close(fig_line)
        # mplfinance might handle closing, but explicitly close if needed
        # if fig_candle: plt.close(fig_candle)
