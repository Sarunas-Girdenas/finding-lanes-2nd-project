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

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

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

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
