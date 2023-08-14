import os, sys
import numpy as np

from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QGridLayout

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

from pandas import read_excel

from scipy.io import loadmat


mpl.style.use("fast")
class Graph():
    def __init__():
        pass
        
class Plotting(QGridLayout):
    def __init__(self, n, parent):
        super(Plotting, self).__init__()

        self.fig, self.axes = plt.subplots(nrows=2, ncols=(n+2)//2)
         
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, parent)

        self.gpsax2d = self.axes[0, 0]
        self.gpsax3d = self.axes[1, 0]
        self.graphs = self.axes[:, 1:].flatten() # First column is reserved for gps

        self.addWidget(self.toolbar, 0, 1)
        self.addWidget(self.canvas, 1, 1)
        
        #self.fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9 , top=0.9, hspace=0.4)
        self.fig.tight_layout()

    def start_excel(file_path):
        formatloc = read_excel(file_path, 'Format', skiprows=0, nrows=3, usecols="C:D", names=[0, 1])
      
        graphformat = read_excel(file_path, 'Format', skiprows=formatloc[0][0] - 1, nrows=formatloc[1][0], usecols="C:H", names=range(6))

        for index, row in graphformat.iterrows():
            plotting.DataGraph(index, *row)

        instrumentformat = read_excel(file_path, 'Format', skiprows=formatloc[0][1] - 1, nrows=formatloc[1][1], usecols="C:O", names=range(0, 13))

        for index, row in instrumentformat.iterrows():
            g, color, protocol, signed, *bytedata = row.tolist()
            plotting.graphs[g].addline(color, protocol, signed, bytedata)

        housekeepingformat = read_excel(file_path, 'Format', skiprows=formatloc[0][2] - 1, nrows=formatloc[1][2], usecols="C:J", names=range(0, 8))
        for index, row in housekeepingformat.iterrows():
            plotting.HouseKeepingData(*row)
    
    def add_map(self, map_file):       
        gpsmap = loadmat(map_file)
        latlim = gpsmap['latlim']
        lonlim = gpsmap['lonlim']
        mapdata = gpsmap['ZA']    
        self.gpsax2d.imshow(mapdata, extent=[*latlim[0], *lonlim[0]])

    def init_graph(self, n, title, xlabel, ylabel, xlim, ylim1, ylim2 ):
        ax = self.axes[n]

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel(xlabel=xlabel, fontsize=8)
        ax.set_ylabel(ylabel=ylabel, fontsize=8)
        ax.set_xlim(0, xlim)
        ax.set_ylim(ylim1 * 1.1, ylim2 * 1.1)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.ticklabel_format(axis='both', scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(6)
        ax.xaxis.get_offset_text().set_fontsize(6)
        lines = []
