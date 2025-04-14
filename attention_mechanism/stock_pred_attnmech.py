import os
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

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"Tensorflow version: {tf.__version__}")
logging.info(f"Current Date: {TODAY}")

# Utility function
def flatten_multiindex_columns(df):
    """Flattens MultiIndex columns if present."""
    if isinstance(df.columns, pd.MultiIndex):
        logging.debug("Flattening MultiIndex columns...")
        # Use the first level of the MultiIndex
        df.columns = df.columns.get_level_values(0)
        # Optional: Handle potential duplicate column names if needed here
    return df

# Data Handling Class
class StockDataHandler:
    """Handles fetching, preprocessing, and sequencing stock data."""
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.scaled_data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_data(self):
        """Downloads stock data from Yahoo Finance."""
        logging.info(f"Downloading {self.ticker} data from {self.start_date} to {self.end_date}")
        try:
            self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
            self.data = flatten_multiindex_columns(self.data)
            logging.info(f"Data downloaded successfully. Shape: {self.data.shape}")
            if self.data.empty:
                logging.error("Downloaded data is empty.")
                raise ValueError("Downloaded data is empty.")
            if 'Close' not in self.data.columns:
                logging.error("'Close' column not found in downloaded data.")
                raise ValueError("'Close' column not found in downloaded data.")
            logging.info(f"Data columns: {self.data.columns}")
            logging.info(f"Data index: {self.data.index}")
            logging.info(f"Data shape: {self.data.shape}")
            logging.info(f"Data head: {self.data.head()}")
        except Exception as e:
            logging.error(f"Error downloading data: {e}")
            raise

    def preprocess_data(self):
        """Preprocesses the data by scaling, sequencing and handling missing values."""
        if self.data is None:
            logging.error("Data not loaded. Cannot preprocess. Call load_data() first.")
            # raise ValueError("Data not loaded. Cannot preprocess. Call load_data() first.")
            return
        logging.info("Preprocessing data...")
        initial_missing = self.data.isnull().sum().sum()
        if initial_missing > 0:
            logging.warning(f"Initial missing values detected: {initial_missing}")
            # "ffill" and "bfill" the missing values
            self.data = self.data.fillna(method='ffill').fillna(method='bfill')
            final_missing = self.data.isnull().sum().sum()
            if final_missing > 0:
                logging.error(f"Final missing values detected: {final_missing}")
                # raise ValueError(f"Final missing values detected: {final_missing}")
                self.data.dropna(inplace=True)
            else:
                logging.info("All missing values handled.")
        else:
            logging.info("No missing values detected.")

        # Ensure data types are numeric for relevant columns
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        self.data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

        # try:
        #     closing_prices = self.data['Close'].values.reshape(-1, 1)
        #     self.scaled_data = self.scaler.fit_transform(closing_prices)
        #     logging.info(f"Data scaled. Shape: {self.scaled_data.shape}")
        # except Exception as e:
        #     logging.error(f"Error preprocessing data: {e}")
        #     raise

    def scale_data(self):
        """Scales the data using MinMaxScaler."""
        if self.data is None or 'Close' not in self.data.columns:
            logging.error("Data not loaded or 'Close' column missing. Cannot scale. Call load_data() first.")
            # raise ValueError("Data not loaded. Cannot scale. Call load_data() first.")
            return
        logging.info("Scaling data...")
        closing_prices = self.data['Close'].values.reshape(-1, 1)
        self.scaled_data = self.scaler.fit_transform(closing_prices)
        logging.info(f"Data scaled. Shape: {self.scaled_data.shape}")
        logging.debug(f"Scaled data sample: {self.scaled_data[:5].flatten()}")

    def create_sequences(self, sequence_length):
        """Creates sequences of data for LSTM input."""
        if self.scaled_data is None:
            logging.error("Data not scaled. Cannot create sequences. Call scale_data() first.")
            # raise ValueError("Data not scaled. Cannot create sequences. Call scale_data() first.")
            return None, None
        logging.info(f"Creating sequences with sequence length: {sequence_length}")
        X, y = [], []
        for i in range(sequence_length, len(self.scaled_data)):
            X.append(self.scaled_data[i-sequence_length:i, 0])
            y.append(self.scaled_data[i, 0])
        X, y = np.array(X), np.array(y)

        if len(X) == 0:
            # logging.error(f"Not enough data ({len(self.scaled_data)} points) to create sequences of length {sequence_length}.")
            raise ValueError(f"Not enough data ({len(self.scaled_data)} points) to create sequences of length {sequence_length}.")
            # return None, None
        logging.info(f"Sequences created. X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def get_last_sequence(self, sequence_length):
        """Gets the last sequence for prediction."""
        # if self.scaled_data is None or len(self.scaled_data):
        #     logging.error("Data not scaled. Cannot get last sequence. Call scale_data() first.")
        #     # raise ValueError("Data not scaled. Cannot get last sequence. Call scale_data() first.")
        #     return None
        # logging.info(f"Getting last sequence with sequence length: {sequence_length}")
        # last_sequence = self.scaled_data[-sequence_length:]
        # if len(last_sequence) < sequence_length:
        #     logging.error(f"Not enough data ({len(self.scaled_data)} points) to get last sequence of length {sequence_length}.")
        #     # raise ValueError(f"Not enough data ({len(self.scaled_data)} points) to get last sequence of length {sequence_length}.")
        #     return None
        # logging.info(f"Last sequence retrieved. Shape: {last_sequence.shape}")
        # return last_sequence.reshape(1, sequence_length, 1)
        if self.scaled_data is None or len(self.scaled_data) < sequence_length:
            logging.error("Not enough scaled data available for the last sequence.")
            return None
        return self.scaled_data[-sequence_length:]

    def get_original_data(self):
        """Returns the original data."""
        if self.data is None:
            logging.error("Data not loaded. Cannot get original data.")
            # raise ValueError("Data not loaded. Cannot get original data.")
            return None
        return self.data

    def get_scaler(self):
        """Returns the scaler object."""
        return self.scaler

# Model Class
class LSTMAttentionModel:
    """Builds, trains, and uses the LSTM with Attention model."""
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
        self.model = None
        self.history = None

    def build_model(self):
        """Builds the Keras Functional API model."""
        logging.info("Building LSTM with Attention model...")
        input_layer = Input(shape=(self.sequence_length, 1))
        lstm_out1 = LSTM(units=50, return_sequences=True)(input_layer)
        lstm_out2 = LSTM(units=50, return_sequences=True)(lstm_out1)

        attention_layer = AdditiveAttention(name='attention_weight')
        attention_result = attention_layer([lstm_out2, lstm_out2])
        multiply_layer = Multiply()([lstm_out2, attention_result])
        context_vector = Flatten()(multiply_layer)

        dense_out = Dropout(0.2)(context_vector)
        dense_out = BatchNormalization()(dense_out)
        output_layer = Dense(1)(dense_out)

        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        logging.info("Model built successfully.")
        self.model.summary(print_fn=logging.info) # Log model summary

    def train(self, X_train, y_train, epochs, batch_size, validation_split=0.2):
        """Trains the model."""
        if self.model is None:
            logging.error("Model not built. Cannot train.")
            return
        logging.info(f"Training model for {epochs} epochs with batch size {batch_size}...")

        # Reshape X_train if it's not already 3D
        if X_train.ndim == 2:
             X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

        early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min', restore_best_weights=True)

        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        logging.info("Model training finished.")

    def evaluate(self, X_test, y_test, scaler):
        """Evaluates the model on test data."""
        if self.model is None:
            logging.error("Model not trained. Cannot evaluate.")
            return None, None, None

        logging.info("Evaluating model...")
        # Reshape X_test if it's not already 3D
        if X_test.ndim == 2:
             X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        test_loss = self.model.evaluate(X_test, y_test, verbose=0)
        logging.info(f"Test loss (MSE): {test_loss:.4f}")

        y_pred_scaled = self.model.predict(X_test, verbose=0)
        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        mae = mean_absolute_error(y_test_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
        logging.info(f"Test MAE: {mae:.4f}")
        logging.info(f"Test RMSE: {rmse:.4f}")
        return test_loss, mae, rmse

    def predict_future(self, last_sequence_scaled, scaler, prediction_days):
        """Predicts future stock prices iteratively."""
        if self.model is None:
            logging.error("Model not trained. Cannot predict.")
            return None

        if last_sequence_scaled is None or len(last_sequence_scaled) != self.sequence_length:
             logging.error(f"Invalid last sequence provided. Expected length {self.sequence_length}.")
             return None

        logging.info(f"Predicting next {prediction_days} days...")
        predicted_prices_scaled = []
        current_batch = last_sequence_scaled.reshape((1, self.sequence_length, 1))

        for _ in range(prediction_days):
            next_prediction_scaled = self.model.predict(current_batch, verbose=0)
            predicted_prices_scaled.append(next_prediction_scaled[0, 0])
            # Reshape prediction for appending
            next_prediction_reshaped = next_prediction_scaled.reshape((1, 1, 1))
            # Update batch: remove first element, append new prediction
            current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)

        # Inverse transform all predictions at once
        predicted_prices = scaler.inverse_transform(np.array(predicted_prices_scaled).reshape(-1, 1))
        logging.info(f"Predicted prices: {predicted_prices.flatten()}")
        return predicted_prices

# --- Visualization Class ---
class StockVisualizer:
    """Handles plotting of stock data and predictions."""
    def __init__(self, ticker):
        self.ticker = ticker

    def plot_candlestick_with_predictions(self, history_data, predictions_df, window_size=120, save_path=None):
        """Plots candlestick chart with future predictions using mplfinance.
        Args:
            history_data (pd.DataFrame): DataFrame with historical OHLCV data.
            predictions_df (pd.DataFrame): DataFrame with 'Predicted Price' and dates.
            window_size (int): Number of historical days to plot.
            save_path (str, optional): Full path to save the plot image (e.g., 'plot.png'). Defaults to None.
        """
        if history_data is None or predictions_df is None:
            logging.error("Cannot plot candlestick: Missing history or prediction data.")
            return

        logging.info("Plotting candlestick chart with predictions...")
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in history_data.columns for col in required_cols):
            logging.error(f"History data missing required columns for candlestick plot: {required_cols}")
            return

        plot_data = history_data.copy()
        # Ensure required columns are numeric (already done in handler, but double-check)
        for col in required_cols:
             plot_data[col] = pd.to_numeric(plot_data[col], errors='coerce')
        plot_data.dropna(subset=required_cols, inplace=True)

        if plot_data.empty:
            logging.error("No valid data left for candlestick plot after cleaning.")
            return

        # Prepare savefig arguments for mplfinance if save_path is provided
        savefig_args = None
        if save_path:
            # Ensure the directory exists (creates if not present)
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
                logging.info(f"Created directory for plot save: {save_dir}")

            savefig_args = dict(fname=save_path, dpi=300, pad_inches=0.25)
            logging.info(f"Plot will be saved to: {save_path}")

        try:
            fig, ax = mpf.plot(
                plot_data.iloc[-window_size:], # Plot last 'window_size' days
                type='candle',
                style='yahoo',
                volume=True,
                title=f'{self.ticker} Stock Price History & {len(predictions_df)}-Day Prediction',
                ylabel='Price ($)',
                figratio=(15, 7),
                returnfig=True, # Return fig and axes objects
                savefig=savefig_args
            )

            # Plot predicted data points on the main price axes (ax[0])
            if 'Predicted Price' in predictions_df.columns:
                 ax[0].plot(predictions_df.index, predictions_df['Predicted Price'],
                            linestyle='dashed', marker='o', markersize=5, color='red',
                            label=f'Predicted {len(predictions_df)} days')
                 ax[0].legend() # Add legend to the price plot
            else:
                logging.warning(" 'Predicted Price' column not found in predictions_df for plotting.")


            # Adjust layout and display
            fig.tight_layout()
            mpf.show() # Use mpf.show() to display the plot
            logging.info("Candlestick plot displayed.")
            if save_path:
                logging.info(f"Candlestick plot saved to: {save_path}")

        except Exception as e:
            logging.error(f"Error during mplfinance plotting/saving: {e}")


    def plot_line_comparison(self, actual_data, predictions_df, history_points, save_path=None):
        """Plots actual closing prices vs. predicted prices using Matplotlib.
        Args:
            actual_data (pd.DataFrame): DataFrame with historical data including 'Close'.
            predictions_df (pd.DataFrame): DataFrame with 'Predicted Price' and dates.
            history_points (int): Number of historical actual closing prices to plot.
            save_path (str, optional): Full path to save the plot image (e.g., 'plot.png'). Defaults to None.
        """
        if actual_data is None or predictions_df is None:
            logging.error("Cannot plot line comparison: Missing actual or prediction data.")
            return
        if 'Close' not in actual_data.columns:
            logging.error("Actual data must contain a 'Close' column.")
            return
        if 'Predicted Price' not in predictions_df.columns:
            logging.error("Predictions DataFrame must contain a 'Predicted Price' column.")
            return

        logging.info("Plotting line comparison chart...")
        plt.figure(figsize=(14, 7))

        # Plot historical actual data (last 'history_points')
        actual_plot_data = actual_data['Close'].iloc[-history_points:]
        plt.plot(actual_plot_data.index, actual_plot_data.values,
                 linestyle='-', marker='.', color='blue', label=f'Actual Data (Last {history_points} days)')

        # Plot predicted data
        plt.plot(predictions_df.index, predictions_df['Predicted Price'],
                 linestyle='-', marker='o', color='red', label=f'Predicted Data ({len(predictions_df)} days)')

        plt.title(f"{self.ticker} Stock Price: Actual vs. Predicted")
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot if save_path is provided
        if save_path:
            try:
                # Ensure the directory exists (creates if not present)
                save_dir = os.path.dirname(save_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    logging.info(f"Created directory for plot save: {save_dir}")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Line comparison plot saved to: {save_path}")
            except Exception as e:
                logging.error(f"Error saving line comparison plot to {save_path}: {e}")
        plt.show()
        logging.info("Line comparison plot displayed.")


# --- Main Execution Logic ---
if __name__ == "__main__":
    # 1. Data Handling
    data_handler = StockDataHandler(TICKER, START_DATE, END_DATE)
    try:
        data_handler.load_data()
        data_handler.preprocess_data()
        data_handler.scale_data()
        X, y = data_handler.create_sequences(SEQUENCE_LENGTH)
    except Exception as e:
        logging.error(f"Failed during data handling phase: {e}")
        exit() # Exit if data preparation fails

    # 2. Data Splitting
    if X is None or y is None:
         logging.error("Sequence creation failed. Exiting.")
         exit()

    train_size = int(len(X) * TRAIN_SPLIT_RATIO)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    logging.info(f"Data split: X_train shape={X_train.shape}, X_test shape={X_test.shape}")

    # 3. Model Training and Evaluation
    lstm_model = LSTMAttentionModel(SEQUENCE_LENGTH)
    lstm_model.build_model()
    lstm_model.train(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    scaler = data_handler.get_scaler() # Get scaler for evaluation and prediction
    lstm_model.evaluate(X_test, y_test, scaler)

    # 4. Prediction
    last_sequence = data_handler.get_last_sequence(SEQUENCE_LENGTH)
    predicted_prices = lstm_model.predict_future(last_sequence, scaler, PREDICTION_DAYS)

    # 5. Visualization
    if predicted_prices is not None:
        original_data = data_handler.get_original_data()
        last_known_date = original_data.index[-1]

        # Create prediction dates (ensure they are business days if needed)
        prediction_dates = pd.date_range(
            start=last_known_date + pd.offsets.BDay(1), # Start from next business day
            periods=PREDICTION_DAYS,
            freq=pd.offsets.BDay() # Ensure dates are business days
        )

        # Create predictions DataFrame
        predictions_df = pd.DataFrame(
            {'Predicted Price': predicted_prices.flatten()},
            index=prediction_dates
        )
        logging.info(f"\nPredictions:\n{predictions_df}")

        visualizer = StockVisualizer(TICKER)

        # Define save paths for plots
        # Create a sub-directory for plots if it doesn't exist
        plots_dir = os.path.join(os.getcwd(), 'stock_plots')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
            logging.info(f"Created directory for plot save: {plots_dir}")

        candlestick_save_path = os.path.join(plots_dir, f"{TICKER}_CStick_{TODAY}.png")
        line_comparison_save_path = os.path.join(plots_dir, f"{TICKER}_LineComp_{TODAY}.png")
        # Plot 1: Candlestick with Predictions
        visualizer.plot_candlestick_with_predictions(original_data,
                                                     predictions_df,
                                                     window_size=120,
                                                     save_path=candlestick_save_path)

        # Plot 2: Line comparison
        visualizer.plot_line_comparison(original_data,
                                        predictions_df,
                                        history_points=SEQUENCE_LENGTH,
                                        save_path=line_comparison_save_path)

    logging.info("Script finished.")