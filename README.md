**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/camera_undistort.png "Undistorted"
[image2]: ./output_images/video_image_undistort_result.png "Road Transformed"
[image3]: ./output_images/current_masking_pipeline.png "Binary Pipeline"
[image4]: ./output_images/perspective_transform_result.jpg "Warp Example"
[image5a]: ./output_images/lane_detection_mask_warped_no_prior_information.png "Fit Visual windowing no prior"
[image5b]: ./output_images/lane_detection_mask_warped_with_prior_information.png "Fit Visual windowing w prior"
[image6a]: ./output_images/lane_line_detection_no_prior_information.jpg "Output no prior"
[image6b]: ./output_images/lane_line_detection_with_prior_information.jpg "Output w prior"
[video1]: ./result/solution_project_video.mp4 "Project Video Output"
[video2]: ./result/solution_challenge_video.mp4 "Challenge Video Output"
[video3]: ./result/solution_even_harder_challenge_video.mp4 "Even Harder Challenge Video Output"


### Description
## Camera Calibration

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Video Processing Pipeline 
Process video camera data, one frame at a time, to extract lane lines from the image. Following are the main steps involved in finding the lane lines - 

#### 1. Camera distortion correction

Undistort each frame using the camera distortion coefficients calculated from the previous step which was calculated once for the camera and saved for us to use in the pipeline. Here is the image before and after the distortion correction.

![alt text][image2]

#### 2. Detecting lane line pixels

I used a combination of region, color and gradient thresholds to generate a binary image. This is implemented as function `laneLineMasking()` in the `TrafficLaneLines.py` in lines 128 through 220. Following image describes the process

![alt text][image3]

#### 3. Perspective transform

The goal of this step is to warp the image in order to get a birds eye view of the lane lines, which will allow us to fit a curve to the lane lines. Perspective transform is calculated using `cv2.getPerspectiveTransform()` which appears in lines 54 through 64 in the file `TrafficLaneLines.py`. It needs source and destination points to calculate the transform. Code in lines 15 to 48 can be uncommented in order to load the straight line test images and we can select points by clicking on the image. Later,I came across some recommended source and destination points provided by Udacity the course providers and I swithched to using them as they frankly worked better. Here is the what the source and destination points used in the code look like - 

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. Here is the output of the perspective transform for a image from the project video. 

![alt text][image4]

#### 4. Finding the lines - windowing and fitting

I have used a sliding window algorithm using convolution method to detect hot lane pixels. This is done once fo reach image at this step by calling the function `find_window_centroids()` in `TrafficLaneLines.py` on lines 264 to 408. Here is the visualization of the windowing when no prior lane information is assumed to be available except that the base of the left lane will be in the left 1/3rd of the image frame width and that the right lane line base position will be in the right 1/3 of the image width. After the left and right lane pixels have been identified, I have used numpy `polyfit()` function to fit a second order polynomial to the individual lane lines. Here is the result of both the steps with convolution based search for windowing.


![alt text][image5a]

However, if a good averaged lane line has already been detected only the pixels in a window around the lane line estimate are searched to find the lane line in the next image before fitting the new detected pixels. When a good average lane line over 3 frames is available for any given lane line, we use this better to get good results in a quick way. Here is what the windows look like when we have a pervious good average available.

![alt text][image5b]

#### 5. Calculation of radius of curvature and position of vehicle in the lane

Finally at the end, if there is average lane line information available, then the lane points are corrected for scaling in x and y and then refit to a second order polynomial. Then the radius of curvature is calculated calculated for the point of interest using the formaula for radius of curvature of a polynomial of second order. This is calculated in the lines 602 and 752 for left lane line and right lane line respectively. Also, if both good average lane line information is available, I calculate the position of the vehicle in the lane by measuring the bias in the lane line positions from the postion of the camera. The camera is assumed to be mounted in the center of the vehicle, resulting in the expectation that the lane lines will be equidistant from the center of the image if vehicle is in exact center of the lane.

#### 6. Unwarp the detected lane lines mask

Following is the output when the detected lane lines in the perspective transformed view gets annoted back on the original image frame.

![alt text][image6a]

![alt text][image6b]

Notice that the windows are present in this output but not the final videos submitted below. For the images from the pipeline that get printed and saved, I find it helpful to have the search windows printed which may help understand performance of the lane line detection algorithms overall.
### Pipeline (video)

#### 1. The outoput for the project video

[![Project video output](https://img.youtube.com/vi/TEVjFoscgBM/mqdefault.jpg)](https://youtu.be/TEVjFoscgBM "Project video output")


---

### Discussion

#### 1. Current shortcomings and potential future improvements

The next section covers all the next steps that I can think of for improving the performance of the pipeline on more challenging aspects of the problem. Here is the output for the challenge video provided.

[![Project video output](https://img.youtube.com/vi/lt7cC0eb5xQ/mqdefault.jpg)](https://youtu.be/lt7cC0eb5xQ "Challenge video output")

1. Not using sobel direction masking. Color transform based pixel masking works poorly in dynamic lighting conditions or when lane markings are worn out. Once a good average fit is available, I would like to start calculating the expected gradient magnitude for cells in each window for the lane. If that is implemented, we can implement, histogram equalizaiton algorithm for better dynamic performance and apply highly specific gradient direction thresholds to identify lane pixels more reliably in dynamic lighting conditions. This will help improve performance on the even harder challenge video significantly.

2. Current averaging method can be improved. A simple low pass filter with slower response for higher order coeffecients of the polynomial fit may be another easy light way to get better filtering response. Also, average x position of the lane line can be calculated by averaging the x coefficient of the lane lines for each y value.

3. I would also like to understand how to understand the quality of fit imformation in order to give high confidence or low confidence to each new line measured.

4. We can also use other information like making sure the lane lines curves are somewhat parallel to detect problems with averaged lane line belief and re search lane lines using convolution instead of just the window around last fit.


