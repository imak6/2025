import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from synth_data_loader import load_dataset, preprocess_data

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, LeakyReLU

import matplotlib.pyplot as plt

def ae_fraud_detection():
    # load the data using the load_dataset function
    data = load_dataset()
    # preprocess the data using preprocess_data function
    preprocessed_data = preprocess_data(data)

    # Split the data intot training and test sets
    X_train, X_test = train_test_split(preprocessed_data, test_size=0.2, random_state=42)

    # Scale the data using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Use the preprocessed data as input for autoencoder model
    input_layer = Input(shape=(X_train.shape[1],))

    # Encoder layers
    encoded_layer = Dense(128, activation='relu')(input_layer)
    encoded_layer = Dense(64, activation='relu')(encoded_layer)
    encoded_layer = Dense(32, activation='relu')(encoded_layer)
    # encoded_layer = BatchNormalization()(encoded_layer)
    # encoded_layer = LeakyReLU(alpha=0.3)(encoded_layer)

    # Latent space (compressed representation of input layer)
    latent_layer = Dense(16, activation='relu')(encoded_layer)

    # Decoded layers
    decoded_layer = Dense(32, activation='relu')(latent_layer)
    decoded_layer = Dense(64, activation='relu')(decoded_layer)
    decoded_layer = Dense(128, activation='relu')(decoded_layer)

    # Output layer
    # output_layer = Dense(preprocessed_data.shape[1], activation='sigmoid')(decoded_layer)
    output_layer = Dense(preprocessed_data.shape[1], activation='linear')(decoded_layer)

    # Build the autoencoder model
    autoencoder = Model(input_layer, output_layer)

    # Compile the model
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Print the model summary
    autoencoder.summary()

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

    # Train the model
    history=autoencoder.fit(X_train, X_train, epochs=2, batch_size=256, validation_data=(X_test, X_test), callbacks=[early_stopping, model_checkpoint])

    # Plot the training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_validation_loss.png')  # Save the plot as a PNG file
    plt.show()
    plt.close()  # Close the plot to free up memory

    # Evaluate the model on test set
    test_loss = autoencoder.evaluate(X_test, X_test)
    print(f"Test loss:{test_loss}")

    # Get the model's predictions (reconstructed data)
    reconstructed_data = autoencoder.predict(X_test)

    # Plot the original data and reconstructed data (for the first 5 samples)
    plt.figure(figsize=(10, 6))
    for i in range(5):
        plt.subplot(2, 5, i+1)
        plt.plot(X_test[i], label='Original Data')
        plt.subplot(2, 5, i+6)
        plt.plot(reconstructed_data[i], label='Reconstructed Data')
        plt.legend()
        plt.show()

    plt.tight_layout()
    plt.savefig('original_vs_reconstructed.png')  # Save the comparison plot
    plt.close()

    # Calculate the reconstruction error
    reconstruction_error = np.mean(np.square(X_test - reconstructed_data), axis=1)
    print(f"Reconstruction Error: {reconstruction_error}")
    print(f"Reconstruction Error shape: {reconstruction_error.shape}")
    # Set a threshold for anomaly detection
    threshold = np.percentile(reconstruction_error, 95)

    print(f"Threshold: {threshold}")
    # Identify anomalies based on the threshold
    anomalies = reconstruction_error > threshold
    print(f"Anomalies: {anomalies}")

    print(f"Anomalies shape: {anomalies.shape}")
    print(f"Anomalies sum: {anomalies.sum()}")

    # Create a DataFrame with the original data, reconstruction error, and anomaly flag
    '''X_test data is a 2D array, but while creating a DF with original data,
    reconstruction error and anomaly as columns, all the columns should be of same shape which X_test is not,
    whereas the other two are (1D array shape).
    So, convert X_test into 1D array
    '''
    # Flatten X_test to 1D array before creating the DataFrame
    # X_test = X_test.flatten()
    # X_test = X_test.reshape(-1, 1)

    '''Other method instead of flattening array is to create a DF with
    separate columns for each feature in X_test
    '''
    # Flatten X_test into individual columns for each feature
    columns = [f"feature_{i+1}" for i in range(X_test.shape[1])]
    results_df = pd.DataFrame(X_test, columns=columns)

    # Add the reconstruction error and anomaly flag to the DataFrame
    results_df['Reconstruction Error'] = reconstruction_error
    results_df['Anomaly'] = anomalies

    # Save results to CSV
    results_df.to_csv('ae_fraud_results.csv', index=False)
    print(results_df.head())
    print(results_df.shape)
    # Save the model
    autoencoder.save('ae_fraud_model.keras')
    return autoencoder

if __name__ == "__main__":
    ae_fraud_detection()