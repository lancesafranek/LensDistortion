# generate 'perfect' data
#   goal:   create before/after images with known radial distortion parameters
#
#            intended to be used as training data for CNN


# 1. Model radial distortion
#       OpenCV has 'undistort' method but seemingly no 'distort' method
#       see: https://stackoverflow.com/questions/21615298/opencv-distort-back

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sliderplot import SliderPlot

# read image# REVISIT, picking width/heigth of training data is really important
f = "standard-grid.jpg"
pth = os.path.join(os.path.dirname(__file__), '../data', f)
img = cv2.imread(pth, 0)
(w,h) = img.shape


# generate start/end points of grid, [-w/2, w/2] x [-h/2, h/2]
gridN = 10 # number of lines in grid (both vertical and horizontal)
gridRes = 15 # number of points along line
gridStartPoints = []
gridEndPoints = []

for i in range(gridN + 1):
    for j in range(gridRes):
        # add horizontal start/end point
        startX = w * j / (gridRes - 1)
        endX = w * (j + 1) / (gridRes - 1)
        startY = h * i/(gridN-1)
        endY = h * i/(gridN-1)
        gridStartPoints.append((startX,startY))
        gridEndPoints.append((endX,endY))

        # add vertical start/end point
        startX = w * i/(gridN-1)
        endX = w * i/(gridN-1)
        startY = h * j / (gridRes - 1)
        endY = h * (j+1) / (gridRes - 1)
        gridStartPoints.append((startX,startY))
        gridEndPoints.append((endX,endY))

# make blank image to plot lines on
gridImg = np.zeros((h,w,3), np.uint8)

# draw lines on gridImg
for i in range(len(gridStartPoints)):
    (x1,y1) = gridStartPoints[i]
    (x2,y2) = gridEndPoints[i]
    cv2.line(gridImg,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)

# method to apply distortion to a list of points ( a point is a 2-tuple )
def distort_points(points, cameraMatrix, distortionMatrix):
    # build camera matrix
    # there's a nice summary here of the units: https://answers.opencv.org/question/189506/understanding-the-result-of-camera-calibration-and-its-units/
    #   also a lot of information about this stuff on the docs: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    distortedPoints = []
    for (px,py) in points:
        # unpack parameters
        fx = cameraMatrix[0,0]
        fy = cameraMatrix[1,1]
        cx = cameraMatrix[0,2]
        cy = cameraMatrix[1,2]
        k1 = distortionMatrix[0,0]
        k2 = distortionMatrix[1,0]
        k3 = 0 # REVISIT!!!
        p1 = distortionMatrix[2,0]
        p2 = distortionMatrix[3,0]

        # transform to relative coordinates
        x = (px - cx) / fx
        y = (py - cy) / fy

        # apply distortion
        r2 = x*x + y*y;
        xCorrected = x * (1. + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2);
        yCorrected = y * (1. + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2);
        
        xCorrected = xCorrected + (2. * p1 * x * y + p2 * (r2 + 2. * x * x));
        yCorrected = yCorrected + (p1 * (r2 + 2. * y * y) + 2. * p2 * x * y);
        
        xCorrected = xCorrected * fx + cx;
        yCorrected = yCorrected * fy + cy;

        # cache distorted point
        distortedPoints.append((xCorrected, yCorrected))
    return distortedPoints

def calculate_warped_grid(vals):
    # build camera matrix
    # there's a nice summary here of the units: https://answers.opencv.org/question/189506/understanding-the-result-of-camera-calibration-and-its-units/
    #   also a lot of information about this stuff on the docs: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    cameraMatrix = np.eye(3, dtype=np.float32)
    cameraMatrix[0,2] = w/2.0
    cameraMatrix[1,2] = h/2.0
    cameraMatrix[0,0] = vals[0] # focal length x [pixels]
    cameraMatrix[1,1] = vals[1] # focal length y [pixels]

    # build distortion coefficients
    distortionParameters = np.zeros((4,1), np.float64)
    distortionParameters[0,0] = vals[2] # radial distortion parameter, > 0 => barrel distortion [dimensionless?]
    distortionParameters[1,0] = vals[3] # radial distortion parameter [dimensionless?]

    # map grid under distortion
    distortedGridStartPoints = distort_points(gridStartPoints, cameraMatrix, distortionParameters)
    distortedGridEndPoints = distort_points(gridEndPoints, cameraMatrix, distortionParameters)

    # draw warped points on new image
    warpedGridImg = np.zeros((h,w,3), np.uint8)

    for i in range(len(gridStartPoints)):
        (x1,y1) = distortedGridStartPoints[i]
        (x2,y2) = distortedGridEndPoints[i]
        cv2.line(warpedGridImg,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)

    # return warped grid image
    return warpedGridImg


initialValues = [12,12,-0.0005,0.0]
# create new slider plot
sldplt = SliderPlot(gridImg, calculate_warped_grid(initialValues))
# add sliders
sldplt.add_slider(0, 30, initialValues[0], 'fx')
sldplt.add_slider(0, 30, initialValues[1], 'fy')
sldplt.add_slider(-1e-3, 1e-3, initialValues[2], 'k1')
sldplt.add_slider(-1e-4, 1e-4, initialValues[3], 'k2')

# set callback, number of sliders has to be equal to the number of entries in list of argument of callback (e.g. len(vals) == num_sliders)
sldplt.set_update_callback(calculate_warped_grid)
# display plot
sldplt.show()