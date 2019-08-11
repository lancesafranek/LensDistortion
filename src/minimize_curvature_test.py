import numpy as np
import matplotlib.pyplot as plt
from utilities import *
from scipy.optimize import minimize
from math import sqrt
import cv2

# size of each image
w = 512
h = 512

# number of points on each line
N = 16

def eval_line(start_point,end_point,s):
    # s should in [0,1]
    px = s*start_point[0] + (1-s) * end_point[0]
    py = s*start_point[1] + (1-s) * end_point[1]
    return (px,py)

def eval_total_curvature(pts):
    # assumes paramerization from [0,1]
    # approximates:
    #       /int ((d/ds)^2 l(s) )^2 ds
    crv = 0
    ds = 1.0/float(N-1)
    for i in range(1,N-1):
        dx2 = pts[i+1,0,0] - 2*pts[i,0,0] + pts[i-1,0,0]
        dy2 = pts[i+1,0,1] - 2*pts[i,0,1] + pts[i-1,0,1]
        crv += ds * (dx2/ds/ds + dy2/ds/ds)**2

    return crv

def eval_arclength(pts):
    # calculate arclength given points (assuming parameterization s in [0,1])
    ds = 1/(N-1)
    arclength = 0
    for i in range(N):
        dx = 1/ds * (pts[i,0,0] - pts[i-1,0,0])
        dy = 1/ds * (pts[i,0,1] - pts[i-1,0,1])
        arclength += ds * sqrt(dx**2 + dy**2)
    return arclength

def list_to_np(points):
    # convert list of 2-tuples to N x 1 x 2 array
    np_arr = np.zeros((N,1,2))
    for i in range(N):
        np_arr[i,0,:] = points[i]
    return np_arr

# camera properties
fx = 12
fy = 12
k1 = -0.0001
k2 = 0 

# transform parameters into form that opencv wants
(cameraMatrix, distortionParameters) = build_camera_matrices(
    [fx, fy, k1, k2], w, h)
cx = cameraMatrix[0, 2] # principal point (width/2)
cy = cameraMatrix[1, 2] # principal point (height/2)    

# build lines (list of lists of 2-tuples, list of lists of points)
lines = []
startPoints = [(0,0), (0,0), (h-1,w-1), (h-1,w-1)]
endPoints = [(0,w-1), (h-1,0), (0,w-1), (h-1,0)]
# startPoints = [(3*h/4, 0), (0, 3*w/4), (0, w/4), (h/4, 0)]
# endPoints = [(3*h/4, w), (h, 3*w/4), (h, w/4), (h/4, w) ]
numLines = len(startPoints)
distortedLines = []
distortedLinesRelative = []
distortedLinesArclength = []

# build each line and apply distortion with known camera properties
for j in range(numLines):
    # for each line, sample between start/end points
    startPoint = startPoints[j]
    endPoint = endPoints[j]
    points = []
    for i in range(N):
        s = float(i)/float(N-1)
        points.append(eval_line(startPoint,endPoint,s))
    # distort points on current lines
    distortedPoints = distort_points(points, cameraMatrix, distortionParameters)
    # convert to np array (N x 1 x 2)
    distortedPointsNp = list_to_np(distortedPoints)

    # convert points on line to np array
    pointsRelative = list_to_np(points)
    # convert to relative coordinates
    pointsRelative[:,:,0] = (pointsRelative[:,:,0] - cx ) / fx
    pointsRelative[:,:,1] = (pointsRelative[:,:,1] - cy ) / fy
    distortedPointsRelative = np.copy(distortedPointsNp)
    distortedPointsRelative[:,:,0] = (distortedPointsRelative[:,:,0] - cx ) / fx
    distortedPointsRelative[:,:,1] = (distortedPointsRelative[:,:,1] - cy ) / fy

    # cache points on current line
    lines.append(pointsRelative)
    # cache points on current distorted line
    distortedLines.append(distortedPointsNp)
    distortedLinesRelative.append(distortedPointsRelative)

    # evaluate arclength of current distorted line
    distortedLinesArclength.append(eval_arclength(distortedPointsRelative))

# objective function to minimize
def loss_function(vals):

    # initialize loss
    loss = 0.0
    # weight to attempt to preserve arclength REVISIT
    alpha = 0.1
    # weight to address translation invariance (anchor to endpoints)
    beta = 0.0

    # build camera properties
    (cameraMatrix, distortionParameters) = build_camera_matrices(
    vals, w, h)

    # loop through lines, compute and add to loss
    for i in range(numLines):
        distortedPoints = distortedLines[i]
        # undistort previously distorted points with new camera properties
        pts = cv2.undistortPoints(distortedPoints, cameraMatrix, distortionParameters)
        # NOTE pts is in relative coordinates,
        # to convert to pixel space do this:
        # pts[:,:,0] = pts[:,:,0]*fx + cx
        # pts[:,:,1] = pts[:,:,1]*fy + cy

        # calculate total curvature of pts (relative space)
        curvaturePenalty = eval_total_curvature(pts)

        # calculate penalty for shrinking/growing arc length
        originalArcLength = distortedLinesArclength[i]
        arclength = eval_arclength(pts)
        arcLengthPenalty = alpha * (arclength - originalArcLength)**2

        # penalty for translation from observed line to undistorted line
        anchorPenalty = beta * ( np.linalg.norm(distortedPointsRelative[0,:,:] - pts[0,:,:]) + beta * np.linalg.norm(distortedPointsRelative[-1,:,:] - pts[-1,:,:]))

        # increment loss from various penalties
        loss += curvaturePenalty + arcLengthPenalty + anchorPenalty

    return loss

# sanity check, calculate loss of loss_function w/ exact params
vals = [fx,fy,k1,k2]
r = loss_function(vals)
print("Loss of exact solution: ")
print(r)

# call solver and see if you can recover vals
opts = {'maxiter':10000}
x0 = [9, 9, 0, 0] # initial guess
res = minimize(loss_function, x0 = x0, method = 'Nelder-Mead', options=opts)
print("Loss of approximate solution: ")
print(res.fun)

# use approximate solution and undistort observed lines
#   this should^TM recover the original grid lines
vals_new = res.x
(cameraMatrix, distortionParameters) = build_camera_matrices(vals_new, w, h)
# undistort previously distorted points with new camera properties
approxLines = []
for distortedPoints in distortedLines:
    approxPoints = cv2.undistortPoints(distortedPoints, cameraMatrix, distortionParameters)
    approxLines.append(approxPoints)

plt.figure(1, figsize=(10,10))
plt.clf()
# iterate through lines and plot approx solution applied to observed lines
for i in range(numLines):
    originalPoints = lines[i]
    distortedPointsRelative = distortedLinesRelative[i]
    approxPoints = approxLines[i]

    # plot
    plt.plot(originalPoints[:,:,0],originalPoints[:,:,1],'b',linewidth=2)
    plt.plot(distortedPointsRelative[:,:,0],distortedPointsRelative[:,:,1],'y',linewidth=2)
    plt.plot(approxPoints[:,:,0],approxPoints[:,:,1],'k',linewidth=2)

plt.axis('off')
plt.legend(['Original','Distorted','Solved'])
plt.title("Approx Solution Loss: " + str(res.fun))
plt.show()