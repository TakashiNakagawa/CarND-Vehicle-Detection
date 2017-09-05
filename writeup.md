## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[img1]: ./output_images/car_notcar_image.png
[img2]: ./output_images/car_hog.png
[img3]: ./output_images/notcar_hog.png
[img4]: ./output_images/test6.jpg
[img5]: ./output_images/colorspace_hsv.png
[img6]: ./output_images/colorspace_hls.png
[img7]: ./output_images/colorspace_luv.png
[img8]: ./output_images/colorspace_rgb.png
[img9]: ./output_images/colorspace_ycc.png
[img10]: ./output_images/colorspace_yuv.png
[img11]: ./output_images/scale11_5_2_result.png
[img12]: ./output_images/process_result.png
[img13]: ./output_images/gurdlane.png
## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the 'vehicle' and 'non-vehicle' images by doing this.

```
cars = glob.glob("../vehicles/*/*.png")  
notcars = glob.glob("../non-vehicles/*/*.png")
```
The code for this setp is contained int the in the lines # 28,29 of the file called 'learning_image.py'.

Here is sample images of car and notcar.
![alt text][img1]

I checked 6 type of color space.
test6.jpg contained black and white cars. So if black and white were clearly separated from other colors, it was prefereable color space.
It was hard to say which color space was good from blow result, but YCrCb and YUV seemd good because balck and white color were grouped. So I chose YCrCb color space for this project.  

test6 image
![alt text][img4]
RGB color space  
![alt text][img8]
HSV color space
![alt text][img5]
LUV color space
![alt text][img7]
HLS color space
![alt text][img6]
YCrCb color space
![alt text][img9]
YUV color space
![alt text][img10]


HOG features were extracted according order.

1. converted to RGB to YCrCb color space
2. scale between 0 - 1
3. each channel

The code for this setp is contained int the in the lines # 28,29 of the file called 'hog_sample.py'.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][img2]  

![alt text][img3]


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried below combinations.  
(a) orient = 9, px_per_cell = 8, cell_per_block = 2
(b) orient = 11, px_per_cell = 8, cell_per_block = 2 --> 0.9865
(a) is same as the lesson.
(a) and (b) seemed not so different.  I chose (b).
Execution time was so big that I could't try much parameters.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used spatial feature, hist feature and hog feature(3 channel).
I used YCrCb color space.
I trained above features by support vector machine.
I decided to use rbf kernel. Firstly I used linear svm. Though test accuracy was high(about 99.5%), result was not good. Then I tried rbf kernel, test accuracy was down to 98.6%, result seemed better. (But it was too slow...)



### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

コードの場所を書く
I used three scales of 1, 1.5, 2. Other parameters were same as the lesson. So overlap was 75%.
Below image is the sliding result. Blue is scale 1, green is scale 1.5, red is 2 (thow which is not shown).
![alt text][img11]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

top left image is the final result.
top right image is the sliding window result.
bottom left image is heatmap of average 10 frames.
bottom right image is after applying threshold which is then used for final result.

To stable the result, I used average frames and adjusted threshold value.

![alt text][img12]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./final_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections which was made by three scales in each frame of the video. From the positive detections I created a heatmap.
After I averaged 10 frames of heatmap, then thresholded that map to identify vehicle positions.  
I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- My pipline likly deteced gurdlane as car.
![alt text][img13]

- SVM kernel of rbf is heavily slow.
