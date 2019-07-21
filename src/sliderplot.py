# supports plotting two images and adding arbitrary sliders
# intended for before/after applying some image transformation that has parameters
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

class SliderPlot():
    # member variables
    sliders = [] # list of sliders
    # flag for if plots can be shown yet or not
    canShow = False

    # slider margin + spacing params
    ax_h = 0.03; # vertical space
    ax_y_padding = 0.005 # vertical padding
    ax_x = 0.1 # starting x val
    ax_x_len = 0.8 # width
    ax_y_len = ax_h - ax_y_padding

    def set_update_callback(self, fn):
        # fn is a function that accepts a list of n parameters ( assumes n sliders)
        # fn returns an image
        self.callback = fn

    def update(self, slider_val):
        # internal callback for slider values changing

        # get new value of sliders
        vals = []
        for sld in self.sliders:
            vals.append(sld.val)
        
        # pass slider values to supplied callback (assumes set_update_callback has been called)
        img_new = self.callback(vals)
        self.plot2.set_data(img_new)
        
        if (self.canShow):
            self.show()

    def show(self):
        if(not self.canShow):
            self.canShow = True
            self.update(1)
            plt.show()

        self.fig.canvas.draw_idle()

    def add_slider(self, val_min, val_max, val_init, s = ""):
        num_sliders = len(self.sliders)
        ax_new = plt.axes([self.ax_x, num_sliders*self.ax_h, self.ax_x_len, self.ax_y_len ])
        slider_new = Slider(ax_new, s, val_min, val_max, val_init)
        slider_new.on_changed(self.update)
        self.sliders.append(slider_new)
    
    def __init__(self, img1, img2, title1 = "", title2 = ""):
        # constructor
        # make two plots here
        self.fig, self.ax = plt.subplots()
        
        # initialize first plot
        plt.subplot(121)
        self.plot1 = plt.imshow(img1, cmap = 'gray');
        plt.title(title1)

        # intitialize second plot
        plt.subplot(122)
        self.plot2 = plt.imshow(img2, cmap = 'gray')
        plt.title(title2)