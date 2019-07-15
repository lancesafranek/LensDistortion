# sandbox for hough transform
import cv2
import numpy as np
from sliderplot import slider_plot
import os

f = "standard-grid.jpg"
pth = os.path.join(os.path.dirname(__file__), '../data', f)
img = cv2.imread(pth, 0)
(w,h) = img.shape
edges = cv2.Canny(img,50,150,apertureSize = 3)

# callback for extracting lines and drawing on blank image
def hough_callback(vals):
    line_img = np.zeros((h,w,3), np.uint8)
    lines = cv2.HoughLines(edges,vals[1],np.pi/180.0,int(vals[0]))
    
    # catch case where lines is None
    if lines is None:
        return line_img
    
    # loop through lines and draw
    for line in lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(line_img,(x1,y1),(x2,y2),(0,0,255),2)
    return line_img

# create new slider plot
sldplt = slider_plot(img, edges)
# add sliders
sldplt.add_slider(0,1000, 200, 'thresh')
sldplt.add_slider(0, 5, 2, 'rho_')
# set callback, number of sliders has to be equal to the number of entries in list of argument of callback (e.g. len(vals) == num_sliders)
sldplt.set_update_callback(hough_callback)
# display plot
sldplt.show()
