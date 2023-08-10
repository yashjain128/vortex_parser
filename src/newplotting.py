import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from scipy.io import loadmat

class Plotting:
    def __init__(n):
        

        self.fig, self.axes = plt.subplots() 
        
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.gpsax2d = self.figure.add_subplot()

    def add_map(file):




