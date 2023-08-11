import os, sys
import numpy as np

from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QGridLayout

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

from scipy.io import loadmat


mpl.style.use("fast")

class Plotting(QGridLayout):
    def __init__(self, n, parent):
        super(Plotting, self).__init__()

        self.fig, self.axes = plt.subplots(nrows=2, ncols=(n+2)//2)
         
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, parent)

        self.gpsax2d = self.axes[0, 0]
        self.gpsax3d = self.axes[1, 0]

        self.addWidget(self.toolbar, 0, 1)
        self.addWidget(self.canvas, 1, 1)
        
        #self.fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9 , top=0.9, hspace=0.4)
        self.fig.tight_layout()
          
    def add_map(self, map_file):       
        gpsmap = loadmat(map_file)
        latlim = gpsmap['latlim']
        lonlim = gpsmap['lonlim']
        mapdata = gpsmap['ZA']    
        self.gpsax2d.imshow(mapdata, extent=[*latlim[0], *lonlim[0]])

 
