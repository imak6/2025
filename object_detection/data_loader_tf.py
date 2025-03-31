import tensorflow as tf
import cv2


def load_data_tf(train_path, test_path, valid_path):
    # Load the dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        batch_size=32,
        image_size=(224, 224),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False
    )

    valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
        valid_path,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        batch_size=32,
        image_size=(224, 224),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False
    )

    test_ds= tf.keras.preprocessing.image_dataset_from_directory(
        test_path,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        batch_size=32,
        image_size=(224, 224),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False
    )

    # Print the shapes of the datasets
    print(train_ds)
    # print(valid_ds)
    # print(test_ds)

    # Print the class names
    print("training class labels:",train_ds.class_names)
    print("validation class labels:", valid_ds.class_names)
    print("testing class labels:", test_ds.class_names)

    for images, labels in train_ds.take(1):
        print(images.shape, labels)
    # for image in images:
    #     # Convert the image tensor to a numpy array for visualization
    #     image_np = image.numpy()
    #     # Display the image using OpenCV
    #     cv2.imshow('Image', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    #     cv2.waitKey(0) # Wait for a key press to proceed to the next image or set the wait time to avoid key press
    #     cv2.destroyAllWindows()
    return train_ds, valid_ds, test_ds
