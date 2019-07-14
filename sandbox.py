# this is intended as an opencv python sandbox
# docs: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

f = "barrel_distortion.png"
img = cv2.imread(f, 0)
edges = cv2.Canny(img,100,200)

#%% apply barrel distortion
# get size of image
(w,h) = img.shape
k1 = 0.0#1.0e-5
k2 = 1.0e-5
p1 = 0.0
p2 = 0.0

distCoeff= np.zeros((4,1), np.float64)
distCoeff[0,0] = k1
distCoeff[1,0] = k2
distCoeff[2,0] = p1
distCoeff[3,0] = p2

cam = np.eye(3, dtype=np.float32)
cam[0,2] = w/2.0
cam[1,2] = h/2.0
cam[0,0] = 12 # focal length x
cam[1,1] = 12 # focal length y

dst = cv2.undistort(img,cam,distCoeff)
fig, ax = plt.subplots()
plt.subplot(121)
plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122)
dst_plot = plt.imshow(dst,cmap = 'gray')
plt.title('(un)distorted image'), plt.xticks([]), plt.yticks([])

ax_h = 0.04;
ax_margin = 0.005;

ax = plt.axes([0.05, 0*ax_h, 0.8, ax_h - ax_margin])
k1slider = Slider(ax, 'k1', -1e-1, 1e-1, valinit=k1)
ax2 = plt.axes([0.05, 1*ax_h, 0.8,  ax_h - ax_margin])
k2slider = Slider(ax2, 'k2', -1e-4, 1e-4, valinit=k2)

ax3 = plt.axes([0.05, 2*ax_h, 0.8,  ax_h - ax_margin])
p1slider = Slider(ax3, 'p1', -1e-1, 1e-1, valinit=p1)
ax4 = plt.axes([0.05, 3*ax_h, 0.8,  ax_h - ax_margin])
p2slider = Slider(ax4, 'p2', -1e-1, 1e-1, valinit=p2)

ax5 = plt.axes([0.05, 4*ax_h, 0.8,  ax_h - ax_margin])
fx = Slider(ax5, 'fx', 0, 30, valinit=cam[0,0])
ax6 = plt.axes([0.05, 5*ax_h, 0.8, ax_h - ax_margin])
fy = Slider(ax6, 'fy', 0, 30, valinit=cam[1,1])

def update(val):
    k1 = k1slider.val
    k2 = k2slider.val
    p1 = p1slider.val
    p2 = p2slider.val
    cam[0,0] = fx.val # focal length x
    cam[1,1] = fy.val # focal length y
    distCoeff[0,0] = k1
    distCoeff[1,0] = k2
    distCoeff[2,0] = p1
    distCoeff[3,0] = p2
    img_new = cv2.undistort(dst,cam,distCoeff)

    dst_plot.set_data(img_new)
    fig.canvas.draw_idle()

p1slider.on_changed(update)
p2slider.on_changed(update)
k1slider.on_changed(update)
k2slider.on_changed(update)
fx.on_changed(update)
fy.on_changed(update)

plt.show()