import os, sys
import numpy as np

from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QGridLayout, QFrame

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

from pandas import read_excel

from scipy.io import loadmat


mpl.style.use("fast")
        
class Plotting(QFrame):
    def __init__(self, parent):
        super(Plotting, self).__init__()

        self.layout = QGridLayout()

        self.fig = plt.figure()
        self.gpsfig, self.pltfig = self.fig.subfigures(1, 2, squeeze=True)

        self.gpsax2d = self.gpsfig.add_subplot(2, 1, 1)
        self.gpsax3d = self.gpsfig.add_subplot(2, 1, 2)
        self.pltaxes = []

        # Convert matplotlib fig and toolbar into pyqt widgets
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, parent)
        self.toolbar.setVisible(False)
        self.layout.addWidget(self.toolbar, 0, 1)
        self.layout.addWidget(self.canvas, 1, 1)
        
        self.fig.tight_layout()
        self.setLayout(self.layout)

    def start_excel(self, file_path):
        formatloc = read_excel(file_path, 'Format', skiprows=0, nrows=3, usecols="C:D", names=[0, 1])
        self.pltfig.clf()
        self.pltaxes = self.pltfig.subplots(2, (formatloc[1][0]+1)//2).flatten()
        graphformat = read_excel(file_path, 'Format', skiprows=formatloc[0][0] - 1, nrows=formatloc[1][0], usecols="C:H", names=range(6))
        for ind, row in graphformat.iterrows():
            self.init_graph(ind, *row)
        self.fig.canvas.draw()
        

        ## instrumentformat = read_excel(file_path, 'Format', skiprows=formatloc[0][1] - 1, nrows=formatloc[1][1], usecols="C:O", names=range(0, 13))

        ## for index, row in instrumentformat.iterrows():
        ##     g, color, protocol, signed, *bytedata = row.tolist()
        ##     plotting.graphs[g].addline(color, protocol, signed, bytedata)

        ## housekeepingformat = read_excel(file_path, 'Format', skiprows=formatloc[0][2] - 1, nrows=formatloc[1][2], usecols="C:J", names=range(0, 8))
        ## for index, row in housekeepingformat.iterrows():
        ##     plotting.HouseKeepingData(*row)
    
    def add_map(self, map_file):       
        gpsmap = loadmat(map_file)
        latlim = gpsmap['latlim']
        lonlim = gpsmap['lonlim']
        mapdata = gpsmap['ZA']    
        self.gpsax2d.imshow(mapdata, extent=[*latlim[0], *lonlim[0]])

    def init_graph(self, n, title, xlabel, ylabel, xlim, ylim1, ylim2 ):
        ax = self.pltaxes[n]

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

plot = None