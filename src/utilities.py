# common methods
import cv2
import numpy as np


def build_camera_matrices(vals, width, height):
    # build camera matrix
    # Apply lens distortion to a list of points
    #
    # input:
    #   vals   - list of camera properties (assumes camera principal point is center of image)
    #           fx, fy, k1, k2, p1, p2 | focal point, radial distortion, tangential distortion
    #   width  - width of image, used for camera principal point
    #   height - height of image, used for camera principal point
    #
    # output:
    #   cameraMatrix     - 3x3 numpy matrix containing focal length and principal point of camera
    #                            see: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    #   distortionMatrix - 4x1 numpy vector containing distortion parameters
    #                            see: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

    # Check args
    if list != type(vals):
        raise ValueError('"vals" is not list')
    if 4 > len(vals):
        raise ValueError('Too few items in list (min 4): {0}'.format(len(vals)))
    elif 6 < len(vals):
        raise ValueError('Too many items in list (max 6): {0}'.format(len(vals)))
    #.end

    cameraMatrix = np.eye(3, dtype=np.float32)
    cameraMatrix[0, 2] = width/2.0  # principal point (center of image)
    cameraMatrix[1, 2] = height/2.0  # principal point (center of image)
    cameraMatrix[0, 0] = vals[0]  # focal length x [pixels]
    cameraMatrix[1, 1] = vals[1]  # focal length y [pixels]

    # build distortion coefficients
    distortionParameters = np.zeros((4, 1), np.float64)
    # radial distortion parameter, > 0 => barrel distortion [dimensionless?]
    distortionParameters[0, 0] = vals[2]
    # radial distortion parameter [dimensionless?]
    distortionParameters[1, 0] = vals[3]
    if (len(vals) > 4):
        distortionParameters[2, 0] = vals[4]  # tangential distortion
    if (len(vals) > 5):
        distortionParameters[3, 0] = vals[5]  # tangential distortion

    # return camera properties
    return (cameraMatrix, distortionParameters)


# method to apply distortion to a list of points ( a point is a 2-tuple )
def distort_points(points, cameraMatrix, distortionMatrix):
    # Apply lens distortion to a list of points
    #
    # input:
    #   points           - a list of 2-tuples (points) to be distorted
    #   cameraMatrix     - 3x3 numpy matrix containing focal length and principal point of camera
    #                            see: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    #   distortionMatrix - 4x1 numpy vector containing distortion parameters
    #                            see: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

    # unpack parameters
    fx = cameraMatrix[0, 0]  # focal length [pixels]
    fy = cameraMatrix[1, 1]  # focal length [pixels]
    cx = cameraMatrix[0, 2]  # principal point (usually center of image)
    cy = cameraMatrix[1, 2]  # principal point (usually center of image)
    k1 = distortionMatrix[0, 0]  # radial distortion parameter
    k2 = distortionMatrix[1, 0]  # radial distortion parameter
    p1 = distortionMatrix[2, 0]  # tangential distortion parameter
    p2 = distortionMatrix[3, 0]  # tangential distortion parameter
    k3 = 0
    if (len(distortionMatrix) > 4):
        k3 = distortionMatrix[4]

    distortedPoints = []
    # loop through points and apply distortion
    for (px, py) in points:
        # transform to relative coordinates
        x = (px - cx) / fx
        y = (py - cy) / fy

        # apply distortion
        r2 = x*x + y*y
        xCorrected = x * (1. + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
        yCorrected = y * (1. + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)

        xCorrected = xCorrected + (2. * p1 * x * y + p2 * (r2 + 2. * x * x))
        yCorrected = yCorrected + (p1 * (r2 + 2. * y * y) + 2. * p2 * x * y)

        xCorrected = xCorrected * fx + cx
        yCorrected = yCorrected * fy + cy

        # cache distorted point
        distortedPoints.append((xCorrected, yCorrected))
    return distortedPoints


def generate_grid_points(lines_num, line_res, w, h):
    # generate start/end points of grid, [0,w] x [0,h]
    #
    # input:
    #   lines_num - number of lines in image
    #   line_res  - number of points along each line
    #   w         - width of image
    #   h         - height of image
    #
    # output:
    #   (startPoints, endPoints) - returns 2-tuple, each entry is a list of points
    #                              lines of grid are startPoints[i]->EndPoints[i]

    gridStartPoints = []
    gridEndPoints = []

    for i in range(lines_num):
        for j in range(line_res-1):
            # add horizontal start/end point
            startX = w * j / (line_res - 1)
            endX = w * (j + 1) / (line_res - 1)
            startY = h * i/(lines_num-1)
            endY = h * i/(lines_num-1)
            gridStartPoints.append((startX, startY))
            gridEndPoints.append((endX, endY))

            # add vertical start/end point
            startX = w * i/(lines_num-1)
            endX = w * i/(lines_num-1)
            startY = h * j / (line_res - 1)
            endY = h * (j+1) / (line_res - 1)
            gridStartPoints.append((startX, startY))
            gridEndPoints.append((endX, endY))

    return((gridStartPoints, gridEndPoints))
