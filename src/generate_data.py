# generate'perfect' data
# outputs to '../training_data'
# generates images, 1.jpg, 2.jpg, etc.
# generates list of camera properties used
#   '../training_data/camera_properties.csv'
#   row i is the properties of <i>.jpg

from utilities import *
import numpy as np
import cv2
import os
import random
import csv

# path to file that will contain camera properties of generated images
params_filename = "camera_properties.csv"
train_dir = os.path.join(os.path.dirname(__file__), '../training_data')
# make training_data directory if necessary
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)
params_path = os.path.join(train_dir, params_filename)

# Number of images to generate
N = 100

# size of each image
w = 512
h = 256

# size of image after cropping
w_crop = 256
h_crop = 128

# pixels to crop before writing image to file (from each edge)
w_margin = int((w-w_crop)/2.0)
h_margin = int((h-h_crop)/2.0)

# make blank image to plot lines on
gridImg = np.zeros((h, w, 3), np.uint8)
# generate grid points
lines_num = 30
lines_res = 10
(gridStartPoints, gridEndPoints) = generate_grid_points(lines_num, lines_res, w, h)

# list to hold camera properties
params = []

for idx in range(N):
    # generate 'random' camera properties
    # for now, assuming 4 properties REVISIT - need reasonable bounds
    fx = 12  # random.randint(10,50)
    fy = fx
    k1 = random.uniform(-0.0005, 0)
    k2 = 0  # random.uniform(-0.0001, 0)

    # transform parameters into form that opencv wants
    (cameraMatrix, distortionParameters) = build_camera_matrices(
        [fx, fy, k1, k2], w, h)

    # distort grid points using camera properties
    distortedGridStartPoints = distort_points(
        gridStartPoints, cameraMatrix, distortionParameters)
    distortedGridEndPoints = distort_points(
        gridEndPoints, cameraMatrix, distortionParameters)

    # make a blank image (grayscale)
    distortedGridImg = np.zeros((h, w), np.uint8)
    # draw lines
    for i in range(len(gridStartPoints)):
        (x1, y1) = distortedGridStartPoints[i]
        (x2, y2) = distortedGridEndPoints[i]
        # cv2 line method needs coordinates to be integers
        cv2.line(distortedGridImg, (int(x1), int(y1)),
                 (int(x2), int(y2)), (255, 255, 255), 1)

    # crop image (edges can get warped pretty bad)
    im_crop = distortedGridImg[h_margin:h-h_margin, w_margin:w-w_margin]

    # write image to file
    f = str(idx) + ".jpg"
    pth = os.path.join(os.path.dirname(__file__), '../training_data', f)
    cv2.imwrite(pth, im_crop)

    # cache camera properties
    # REVISIT, cropping image likely means fx/fy need to be scaled appropriately
    params.append([idx, fx, fy, k1, k2])

# write parameters to file
with open(params_path, 'w') as f:
    writer = csv.writer(f)
    # write header
    writer.writerow(['image_idx', 'fx', 'fy', 'k1', 'k2'])
    writer.writerows(params)
