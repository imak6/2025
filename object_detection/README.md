# **Object detection project**

- **Data**:
Use dogs and cats classifier dataset downloaded from kaggle using kagglehub.

- **Data Preprocessing**:
When dealing with image data, preprocessing is important for the model to perform better.

Image preprocessing - process of manipulating raw image data into a usable and meaningful format, by eliminating unwanted distortions and enhance specific qualities essential for CV applications.

- **Techniques used in image preprocessing**:
**Resizing**: resizing all the images in a dataset to a uniform size is improtant for the algorithms to function properly. Use **OpenCV's** **resize()**.

**Grayscaling**: Converting colour/raw images data to grayscale simplifies the image data and reduces computational needs. **cvtColor()** can be used to convert RGB to grayscale.

**Noise Reduction**: Smoothing, blurring, and filtering techniques can be applied to remove unwanted noise from images. **GaussianBlur()** and **medianBlur()** methods are commonly used for this.

**Normalization**: adjusts the intensity values of pixels to a desired range, often between "0" and "1". **Normalize()** from scikit-image can be used for this.

**Binarization**: converts grayscale images to black and white by thresholding. The **threshold()** method is used to binarize images in openCV.

**Contrast enhancement**: contrast of the images can be adjusted using histogram equalization. The **equalizeHist()** method enhances the contrast of images.

-**Data Annotation**

**Bounding boxes**:
Most common type of annotation in computer vision.
These are rectangular boxes used to "define the location of target object".
Can be determined by the "x-axis" and "y-axis" coordinates in the upper-left corner and the coordinates in the bottom-left corner of the rectangle (box).
Generally used in object detection and localization tasks.

Usually represented by either two coordinates (x1,y1) and (x2,y2) or by one coordinate (x1,y1) and width (w), height (h) of the bounding box.

**Polygonal Segmentation**:
Since objects are not always rectangular in shape, polygons are used to annotate raw image data to define the shape and location of the object in a precise way.

**Semantic Segmentation**:
This is a "pixel wise annotation", where every pixel in the image is assigned a class. These classes could be pedestrain, car, bus, cat, dog, sidewalk, etc., and each pixel carry a semantic meaning.

Primarily used in cases where environmental context is very important. E.g: This method is used in "self-driving" cars and robotics so the models can understand the environment they are operating in easily.

**3D cuboids**:
Similar to bounding boxes with additional "depth" information about the object. As the name suggests, we get a 3D representation of an object which allows systems to distinguish features like "volume" and "position" in a 3D space.

E.g: Used in self-driving cars, where it can use depth information to measure the distance of ibjects from the car.

**Key-Point and Landmark**:
Used to detect small objects and shape variations by creating dots across the images.

Useful in detecting facial features such as "expressions, emotions, human body parts and poses".

**Lines and Splines**:
commonly used in autonomous vehicles for lane detection and recognition.

**Image Annotation Formats**:
There is no standard format for image annotation. But below are few commonly used annotation formats:

**COCO**:
It has 5 annotation types: for "object detection", "keypoint detection", "stuff segmentation", "panoptic segmentation", and "image captioning".

Annotations are stored using "JSON".

**Pascal VOC**:
stores annotation in "XML" file.
**YOLO**:
a ".txt" file with same name is created for each image file in the same directory.
Each ".txt" file contains annotations for the corresponding image file i.e., object class, coordinates, height, and width.
For each object, a new line is created. An image may contain various objects.

**Image Annotation Tools**:
MakeSense.AI
LabelImg
VGG image annotator
LabelMe
Scalable
RectLabel
