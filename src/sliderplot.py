# supports plotting two images and adding arbitrary sliders
# intended for before/after applying some image transformation that has parameters
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


class SliderPlot():

    def __init__(self, img1, img2, title1="", title2=""):

        # member variables
        self.sliders = []  # list of sliders
        # flag for if plots can be shown yet or not
        self.canShow = False

        # slider margin + spacing params
        self.ax_h = 0.03  # vertical space
        self.ax_y_padding = 0.005  # vertical padding
        self.ax_x = 0.1  # starting x val
        self.ax_x_len = 0.8  # width
        self.ax_y_len = self.ax_h - self.ax_y_padding

        # callbacks for updating images when sliders are modified
        self.plot1_callback = None
        self.plot2_callback = None

        # figure/axes for plotting
        self.fig = None
        self.ax = None

        # constructor
        # make two plots here
        self.fig, self.ax = plt.subplots()

        # initialize first plot
        plt.subplot(121)
        self.plot1 = plt.imshow(img1, cmap='gray')
        plt.title(title1)

        # intitialize second plot
        plt.subplot(122)
        self.plot2 = plt.imshow(img2, cmap='gray')
        plt.title(title2)

    def set_update_callback(self, fn):
        # fn is a function that accepts a list of n parameters ( assumes n sliders)
        # fn returns an image
        self.plot2_callback = fn

    def set_update_callbacks(self, fn, gn):
        # fn/gn are functions that accept a list of n parameters (assumes n sliders)
        # fn/gn return images
        self.plot1_callback = fn
        self.plot2_callback = gn

    def update(self, slider_val):
        # internal callback for slider values changing

        # get new value of sliders
        vals = []
        for sld in self.sliders:
            vals.append(sld.val)

        # pass slider values to supplied callback(s) (assumes set_update_callback(s) has been called)
        if (self.plot1_callback is not None):
            img_new = self.plot1_callback(vals)
            self.plot1.set_data(img_new)

        if (self.plot2_callback is not None):
            img_new = self.plot2_callback(vals)
            self.plot2.set_data(img_new)

        if (self.canShow):
            self.show()

    def show(self):
        if(not self.canShow):
            self.canShow = True
            self.update(1)
            plt.show()

        self.fig.canvas.draw_idle()

    def add_slider(self, val_min, val_max, val_init, s=""):
        num_sliders = len(self.sliders)
        ax_new = plt.axes([self.ax_x, num_sliders*self.ax_h,
                           self.ax_x_len, self.ax_y_len])
        slider_new = Slider(ax_new, s, val_min, val_max, val_init)
        slider_new.on_changed(self.update)
        self.sliders.append(slider_new)
