# testing ground for sliderplot class
import cv2
import numpy as np
from sliderplot import SliderPlot
import os

# read image
f = "distorted-grid.jpg"
pth = os.path.join(os.path.dirname(__file__), '../data', f)
img = cv2.imread(pth, 0)
(w, h) = img.shape

# callback for applying 'undistortion' to image
def undistort_callback(vals):
    distCoeff = np.zeros((4, 1), np.float64)
    cam = np.eye(3, dtype=np.float32)
    cam[0, 2] = h/2.0
    cam[1, 2] = w/2.0
    cam[0, 0] = vals[4]  # focal length x
    cam[1, 1] = vals[5]  # focal length y
    distCoeff[0, 0] = vals[0]
    distCoeff[1, 0] = vals[1]
    distCoeff[2, 0] = vals[2]
    distCoeff[3, 0] = vals[3]
    img_new = cv2.undistort(img, cam, distCoeff)
    return img_new


# create new slider plot
sldplt = SliderPlot(img, img)
# add sliders
sldplt.add_slider(-1e-1, 1e-1, 0, 'k1')
sldplt.add_slider(-1e-4, 1e-4, 1.0e-5, 'k2')
sldplt.add_slider(-1e-1, 1e-1, 0, 'p1')
sldplt.add_slider(-1e-1, 1e-1, 0, 'p2')
sldplt.add_slider(0, 100, 30, 'fx')
sldplt.add_slider(0, 100, 30, 'fy')
# set callback, number of sliders has to be equal to the number of entries in list of argument of callback (e.g. len(vals) == num_sliders)
sldplt.set_update_callback(undistort_callback)
# display plot
sldplt.show()
