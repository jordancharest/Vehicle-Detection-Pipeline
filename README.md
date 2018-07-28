# Vehicle Detection
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Build a pipeline to accept an image, run a sliding window search on the image and run windows through the classifier to detect vehicles
* Build a heatmap of the detections and threshold to filter out false detections and get a better estimate of bounding boxes with multiple detections
* Run the pipeline on a video stream and output a resulting video with bounding boxes drawn around all cars

[//]: # (Image References)
[car]: ./images/car.png
[notcar]: ./images/notcar.png
[HOG]: ./images/HOG.png
[test1]: ./images/test1.png
[test2]: ./images/test2.png
[video1]: ./test_videos_output/result.mp4


---
### Histogram of Oriented Gradients (HOG)

#### 1. How I extracted HOG features from the training images

The HOG feature extraction is done by the two functions defined right after the import statements in `tracking_pipeline.ipynb`: `extract_features` and `get_hog_features`, which use the skimage method `hog' 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `cars` and `notcars` classes, respectively:

![alt text][car]
![alt_text][notcar]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][HOG]

#### 2. How did I settle on my final choice of HOG parameters?

Guess and check. My number one goal was to maximize the classifier accuracy. I iterated through a bunch of different combinations of HOG parameters and colorspace and finally settled on YUV colorspace, with 11 orientations, 16 pixels per cell, and 2 cells per block. This combination seemed to result in the highest classifier accuracy.

#### 3. How I trained a classifier using your selected HOG features.

Even at accuracies of ~95%, a lot of false positives show up in the video stream. It's not until 98%+ accuracy that my pipeline starts to perform well. Thinking I was being clever, I originally used an SVM with an RBF kernel (despite suggestions to use a linear kernel) and searched for optimal parameters with grid search, but found that I couldn't get an accuracy much higher than around 95%, even after choosing optimal HOG parameters, and it took a few minutes to train. As mentioned before, 95% accuracy wasn't good enough to get a good resultant video. I eventually tried a LinearSVM and it made an enormous difference. The training time dropped to a few seconds (!) and the accuracy rose to 98-99%.

Training is done in the second code cell in step 2. Implementing and training an SVM is trivial thanks to sci-kit learn.

### Sliding Window Search

#### 1. How I implemented a sliding window search
I used the `find_cars` method to take an image, a start and stop row, and a scale, and search the image via sliding windows. Of course, cars will be at a different scale in different parts of the image, so in my tracking pipeline, I actually call `find_cars` several times with different start and stop rows and different scales for the sliding windows.


#### 2. Examples of test images to demonstrate how the pipeline is working.
The optimizations i made were extracting HOG features from the entire image at once, and only searching a small number of windows in the image where it was likely that cars would be. A couple output images are posted below. There are lots of false positives, even with a 98%+ classifier accuracy. I used the heatmap method to get rid of these, discussed below.



![alt text][test1]
![alt text][test2]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./test_videos_output)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

As I said previously, I used the `find_cars` method several times in the detection pipeline. All the sliding windows overlapped, so in most cases, a car in the image would result in several overlapping bounding boxes. After every search at different scales, I used the `add_heat` method to increment the pixels that were within a bounding box. After all searches were complete, I thresholded the heatmap with `heat_threshold` so that only locations with multiple detections were kept. I then used `scipy.ndimage.measurements.label()' to uniquely label each remaining area and drew a bounding box around each one.

---

### Discussion

#### 1. Where will the pipeline likely fail?  What could I do to make it more robust?
I found the feature engineering necessary for using an SVM to be extremely tedious, and not work as well as I had hoped. Further, it was fairly computationally expensive, and ran on my modern laptop at only about 1.5 frames/second. As with many machine learning methods, it would fail to recognize cars that are not generally described by the training set. If a tractor trailer or Dunkin Donuts marketing car drives by, my pipeline would completely fail to recognize it. The solution for my particulr pipeline is more training data.

If I were to do the project again, I would use deep learning and take a real-time object detection approach. I used an implementation of YOLOv3, pre-trained on the CoCo dataset, and it performed better on the project video than my pipeline, while also running faster. Which is of course a little disappointing. Lastly, the great part about deep learning is no feature engineering is necessary...simply feed the entire image into the classifier and it will draw bounding boxes. Of course, building a network is significantly more involved than a couple API calls to set up and train an SVM. Personally, I would go for the deep learning method if I was implementing this project for a real self-driving car, but the method you pick all depends on your goals and what you enjoy doing.
