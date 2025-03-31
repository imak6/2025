import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam
from data_loader import load_and_preprocess
import pandas as pd
import cv2

def object_detection_cnn(train_path, valid_path, test_path, target_size=(224, 224)):
    # load the data
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_and_preprocess(train_path, valid_path, test_path, target_size)

    # Reshape the loaded data (if necessary)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 3)
    X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], X_valid.shape[2], 3)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 3)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"X_valid shape: {X_valid.shape}")

    # Normalize the data (especially in CNNs, normalizing the inputs is often crucial for better convergence and performance.)
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_valid = X_valid / 255.0
    print(f"X_train shape after normalizing: {X_train.shape}")
    print(f"X_test shape after normalizing: {X_test.shape}")
    print(f"X_valid shape after normalizing: {X_valid.shape}")


    # Equalize the images
    tr_eq = cv2.equalizeHist(X_train)
    te_eq = cv2.equalizeHist(X_test)
    va_eq = cv2.equalizeHist(X_valid)
    # def visualize_batch(X_train, y_train, start_index=0, num_images=5):
    # Display 'num_images' from X_train
    num_images = 5
    start_index = 0
    plt.figure(figsize=(15, 5))

    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        image = X_train[start_index + i]

        # Rescale for visualization
        image_rescaled = image * 255.0
        image_rescaled = image_rescaled.astype('uint8')

        plt.imshow(image_rescaled)
        plt.axis('off')
        plt.title(f"Label: {y_train[start_index + i]}")

    plt.show()

# Example usage: Visualizing the first 5 images in X_train
# visualize_batch(X_train, y_train, start_index=0, num_images=5)


    # Build the model
    model=Sequential()
    model.add(Input(shape=(target_size[0], target_size[1], 3)))
    # Add convolution layer
    model.add(Conv2D(32, (3,3), activation='relu'))
    # Add pooling layer
    model.add(MaxPool2D(2,2))
    # Add another convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # Add another pooling layer
    model.add(MaxPool2D(2,2))
    # Add fully connected layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    # Add output layer
    model.add(Dense(2, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train/fit the model
    model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_valid, y_valid))
    model.summary()
    # Evaluate the model
    model.evaluate(X_test, y_test)

    # Make predictions
    predictions = model.predict(X_test)
    # print(f"Predictions made by the trained model: {predictions}")
    predictions_df = pd.DataFrame(predictions, columns=['cat', 'dog'])
    predictions_df.to_csv('predictions.csv')
    return model

if __name__ == "__main__":
    img_path = '/Users/ajithraghavan/.cache/kagglehub/datasets/rajarshi2712/dogs-and-cats-classifier/versions/1/'
    train_path = img_path + 'train/**/*.jpg'
    test_path = img_path + 'test/**/*.jpg'
    valid_path = img_path + 'valid/**/*.jpg'
    object_detection_cnn(train_path, test_path, valid_path, target_size=(224, 224))