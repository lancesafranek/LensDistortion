# 1. Model radial distortion
#       OpenCV has 'undistort' method but seemingly no 'distort' method
#       see: https://stackoverflow.com/questions/21615298/opencv-distort-back

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sliderplot import SliderPlot
from utilities import *

# read image# REVISIT, picking width/heigth of training data is really important
f = "standard-grid.jpg"
pth = os.path.join(os.path.dirname(__file__), '../data', f)
img = cv2.imread(pth, 0)
(w, h) = img.shape

# make blank image to plot lines on
gridImg = np.zeros((h, w, 3), np.uint8)

# get points on grid
(gridStartPoints, gridEndPoints) = generate_grid_points(6, 10, w, h)

# draw lines on gridImg
for i in range(len(gridStartPoints)):
    (x1, y1) = gridStartPoints[i]
    (x2, y2) = gridEndPoints[i]
    cv2.line(gridImg, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

def calculate_warped_grid(vals):
    # get camera properties
    (cameraMatrix, distortionParameters) = build_camera_matrices(vals, w, h)

    # map grid under distortion
    distortedGridStartPoints = distort_points(
        gridStartPoints, cameraMatrix, distortionParameters)
    distortedGridEndPoints = distort_points(
        gridEndPoints, cameraMatrix, distortionParameters)

    # draw warped points on new image
    warpedGridImg = np.zeros((h, w, 3), np.uint8)
    for i in range(len(gridStartPoints)):
        (x1, y1) = distortedGridStartPoints[i]
        (x2, y2) = distortedGridEndPoints[i]
        # cv2 line method needs coordinates to be integers
        cv2.line(warpedGridImg, (int(x1), int(y1)),
                 (int(x2), int(y2)), (0, 0, 255), 2)

    # return warped grid image
    return warpedGridImg


def undistort_grid_callback(vals):
    (cameraMatrix, distortionParameters) = build_camera_matrices(vals, w, h)
    warpedGridImg = calculate_warped_grid(vals)  # get warped grid
    # call opencvs undistort method on warped grid with camera properties
    unwarpedGrid = cv2.undistort(
        warpedGridImg, cameraMatrix, distortionParameters)

    return unwarpedGrid


initialValues = [12, 12, -0.0001, 0.0]
# create new slider plot
sldplt = SliderPlot(gridImg, calculate_warped_grid(
    initialValues), 'Warped Grid', 'Unwarped Grid (cv2.undistort(...))')
# add sliders
sldplt.add_slider(0, 30, initialValues[0], 'fx')
sldplt.add_slider(0, 30, initialValues[1], 'fy')
sldplt.add_slider(-1e-3, 1e-3, initialValues[2], 'k1')
sldplt.add_slider(-1e-4, 1e-4, initialValues[3], 'k2')
# set callback, number of sliders has to be equal to the number of entries in list of argument of callback (e.g. len(vals) == num_sliders)
sldplt.set_update_callbacks(calculate_warped_grid, undistort_grid_callback)
# display plot
sldplt.show()
