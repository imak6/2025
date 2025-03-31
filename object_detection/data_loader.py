'''Module to load and show images from directories and sub-directories using OpenCV
'''
import cv2
import glob
import os
import numpy as np


def load_and_preprocess(train_path, test_path, valid_path, target_size=(224, 224)):
    def load_data_from_directory(directory_path):
        images=[]
        labels=[]

        # Loop through images in the directory
        for img in glob.glob(directory_path):
            if os.path.isfile(img):
                img_read = cv2.imread(img)
                print(f"Reading image: {img}")
                print(f"Image shape: {img_read.shape}")
                print(f"Image dtype: {img_read.dtype}")
                print(f"Image size: {img_read.size}")
                print(f"Image max value: {img_read.max()}") # Check the maximum pixel value
                print(f"Image min value: {img_read.min()}")  # Check the minimum pixel value
                # print(f"Image mean value: {img_read.mean()}")  # Check the mean pixel value
                # print(f"Image std value: {img_read.std()}")  # Check the standard deviation of pixel values
                # print(f"Image variance: {img_read.var()}")  # Check the variance of pixel values

                # Check if the image was loaded successfully
                if img_read is None:
                    print(f"Failed to load image: {img}")
                    continue # skip this image

                # Resize the image to the target size
                img_resized = cv2.resize(img_read, target_size)
                img_resized = img_resized.astype('float32') / 255.0

                # Convert the numpy array to RGB format if necessary
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                # cv2.imshow('img', img_rgb)
                images.append(img_rgb)

                label = 0 if 'dog' in img else 1
                labels.append(label)
                cv2.waitKey(1) # wait
                cv2.destroyAllWindows()
        return np.array(images), np.array(labels)

    # Load data from each directory
    X_train, y_train = load_data_from_directory(train_path)
    X_test, y_test = load_data_from_directory(test_path)
    X_valid, y_valid = load_data_from_directory(valid_path)
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

if __name__ == "__main__":
    train_path = '/Users/ajithraghavan/.cache/kagglehub/datasets/rajarshi2712/dogs-and-cats-classifier/versions/1/train/**/*.jpg'
    test_path = '/Users/ajithraghavan/.cache/kagglehub/datasets/rajarshi2712/dogs-and-cats-classifier/versions/1/test/**/*.jpg'
    valid_path = '/Users/ajithraghavan/.cache/kagglehub/datasets/rajarshi2712/dogs-and-cats-classifier/versions/1/valid/**/*.jpg'
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_and_preprocess(train_path, test_path, valid_path)
    # print(X_train.shape, y_train.shape)
    # print(X_valid.shape, y_valid.shape)
    # print(X_test.shape, y_test.shape)
    # print(X_train[0])
    # print(y_train[0])
    # print(X_valid[0])
    # print(y_valid[0])
    # print(X_test[0])