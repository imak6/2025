'''Data Pipeline using Tensorflow/Keras
to load, preprocess/transform and batch the image data automatically
'''
import tensorflow as tf
import cv2
import os
import glob

def load_image(img_path):
    # read and decode the image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3) # Modify the format (e.g: PNG) if needed
    img = tf.cast(img, tf.float32) / 255.0 # Normalize the image to [0,1]
    return img

def  preprocess_image(img):
    # Resize the image to a fixed size
    img = tf.image.resize(img, (224, 224)) # Modify the size as needed
    return img

def load_labels(label_path):
    # Assuming labels are in a text format ro as a JSON file
    # In object detection, labels could be bounding boxes or class labels
    # Example: Read bounding box coordinates from a .txt file
    with open(label_path, 'r') as f:
        labels = f.read().splitlines()
    return labels # Can be customized based on annotations format

def process_imagedata(img_path, label_path):
    img = load_image(img_path)
    img = preprocess_image(img)
    labels = load_labels(label_path)
    return img, labels

def create_pipeline(img_dir, label_dir, batch_size=32):
    # List of image paths and corresponding label paths
    img_paths = glob.glob(os.path.join(img_dir, '*.jpg'))
    label_paths = [os.path.join(label_dir, os.path.basename(img_path).replace('.jpg', '.txt')) for img_path in img_paths]

    # create a Tensorflow dataset from image and label pairs
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, label_paths))

    # map the process function to load images and labels
    dataset = dataset.map(lambda img_path, label_path: process_imagedata(img_path, label_path), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle, batch, and prefetch the dataset for optimal performance
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

if __name__ == "__main__":
    img_dir = '/Users/ajithraghavan/.cache/kagglehub/datasets/rajarshi2712/dogs-and-cats-classifier/versions/1/train/'
    label_dir = '/Users/ajithraghavan/.cache/kagglehub/datasets/rajarshi2712/dogs-and-cats-classifier/versions/1/train/'

    pipeline = create_pipeline(img_dir, label_dir)

    # Iterate over the data pipeline to visualize the images
    for images, labels in pipeline.take(1):
        print(images.shape, labels)
        # for image in images:
        #     # Convert the image tensor to a numpy array for visualization
        #     image_np = image.numpy()
        #     # Display the image using OpenCV
        #     cv2.imshow('Image', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        #     cv2.waitKey(0) # Wait for a key press to proceed to the next image or set the wait time to avoid key press
        #     cv2.destroyAllWindows()