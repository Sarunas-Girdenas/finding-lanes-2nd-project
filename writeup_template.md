## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

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

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "P2.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

I've wrapped _Camera_ as a class for easier use:

```
class Camera(object):
    """
    Camera that we use to take pictures/.
    """
    
    def __init__(self, num_x_corners=9, num_y_corners=6, location='camera_cal'):
        """
        Camera constructor
        """
        
        self.calibrated = False
        self.num_x_corners = num_x_corners
        self.num_y_corners = num_y_corners
        self.location = location
        
        # object points
        self.objp = np.zeros((num_x_corners * num_y_corners, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:num_x_corners, 0:num_y_corners].T.reshape(-1, 2)
        
        return None
    
    def calibrate(self):
        """
        Calibrates camera given pictures
        """
        
        # 3D points in real-world space
        object_points = []
        # 2D image points
        image_points = []
        
        # load pictures for calibration
        for img in glob.glob(f"{self.location}/*"):
            # find corners
            image = cv2.imread(img)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            ret, corners = cv2.findChessboardCorners(gray, (self.num_x_corners, self.num_y_corners), None)
            
            if ret:
                object_points.append(self.objp)
                image_points.append(corners)
        
        # calibrate camera!
        # gray.shape[::-1] - picture size
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(
                object_points, image_points, gray.shape[::-1],
                None, None
            )
        
        self.calibrated = True
        
        return None
    
    def undistort_picture(self, picture):
        """
        Fixes picture given camera is calibrated
        """
        
        if not self.calibrated:
            raise ValueError("Run camera.calibrate() first!")
        
        undistorted = cv2.undistort(picture, self.mtx, self.dist, None, self.mtx)
        
        return undistorted
```

It uses pictures of chessboards provided in camera_cal directory to calibrate the camera (compute distortion coeffiecients which are use to _undistort_ images.


### Pipeline (single images)

#### 1. Example of corrected image (from the _Camera_ class)

Sample image after _undistortion_:
![calibrated_chessboard](https://github.com/Sarunas-Girdenas/finding-lanes-2nd-project/blob/master/chessboard.png)

#### 2. Color Transforms and Gradients

I've used in total 4 different transforms (these functions are taken from Exercises):
1. `abs_sobel_thresh()` - compute direction gradient 
2. `mag_threshold()` - compute gradient magnitude (how big is the change of colours between nearby pixels)
3. `dir_threshold()` - compute gradient direction
4. `hls_select()` - converts picture to HLS color space and choose S channel (as the most informative one)
5. `apply_thresholds()` - combines all the above functions

This is wrapped in the class called `CombineTresholds()` in the notebook.

Sample output (on the _undistorted_ picture):

![combined_threshold](https://github.com/Sarunas-Girdenas/finding-lanes-2nd-project/blob/master/colors.png)

#### 3. Perspective Transform

I've wrapped perspective transformer into `PerspectiveTransform()` class in the code. As you can see, there are 2 options in the class. First, we use some rule-of-thumb to find the mapping between source and destination image. For the source image, it ust divides the width by 2 and takes 62% of the height. For the source, we take 100 (arbitrary number). However, this did not work well at all on the video output. Hence I've hardcoded the following coordinates:

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 280, 700      | 250, 720      | 
| 595, 460      | 250, 0        |
| 725, 460      | 1065, 0       |
| 1125, 700     | 1065, 720     |

After applying `PerspectiveTransform.transform_perspective()` on a _undistorted_ and _thresholded_ picture, we get this:

![perspective](https://github.com/Sarunas-Girdenas/finding-lanes-2nd-project/blob/master/perspective.png)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

As suggested in the course, I've used the histogram approach. The idea is to compute histogram (sum) of pixels along the height of the picture. The location with the largest sum suggests the position of the lane. This is performed in the `FindLanes()` class. Function `FindLanes.find_lanes()` takes binary warped image, calculates histogram and fits the 2nd degree polynomial on the newly found lane points.

![perspective](https://github.com/Sarunas-Girdenas/finding-lanes-2nd-project/blob/master/lanes0.png)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In the same `FindLanes()` class I've added the function `measure_lane_curvature()`. I've hardcoded meters per pixel (as suggested in the exercises). Function `distance_from_camera()` calculates vehicle position with respect to center (calculating how far the lanes are from the horizontal midpoint of the picture assuming that camera is in the middle of the car).

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Function `FindLanes.draw_lanes_on_image()` draws back the newly inferred lanes:

Here's a [link to my video result](https://github.com/Sarunas-Girdenas/finding-lanes-2nd-project/blob/master/fin_lanes.png)

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://github.com/Sarunas-Girdenas/finding-lanes-2nd-project/blob/master/challenge_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Firstly, I have not tested what happens if the other car drives in front. Such case would break the pipeline because it would interfere with our algorithm that assumes that there are no obstructions in the way. That is, we use various thresholds (colours and shapes) to extract lanes, but having car in front most certainly would pose a challenge to this approach as we would be detecting car shape.

Secondly, I have not considered the case where one lane dissapears (due to ongoing roadworks for instance). In such case perhaps we should drive using just the one (visible) lane, but that would be dangerous.

Thirdly, same as in the first project, in this case lane estimates still appear a bit "jumpy" and uneven, especially when car wobbles a bit on the road. I would need to implement some kind of smoothing to hopefully reduce that. 
