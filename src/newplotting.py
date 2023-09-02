import os, sys
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

import openpyxl

from PyQt5.QtWidgets import (QApplication, QCheckBox, QGridLayout, QGroupBox, QComboBox,
                             QMenu, QPushButton, QRadioButton, QWidget, QLabel, QLineEdit, QFileDialog)

from scipy.io import loadmat

class Plotting():
    def __init__(self, win):
        self.win = win
        self.fig = None
        self.gpsfig, self.pltfig = None, None
        self.hkNames = ["Temp1", "Temp2", "Temp3", "Int. Temp", "V Bat", "-12 V", "+12 V", "+5 V", "+3.3", "VBat Mon", "Dig. Temp"]
    def on_close(self, close_msg):
        self.win.pickInstrCombo.setEnabled(True)
        self.win.pickInstrButton.setEnabled(True)
        self.win.pickInstrCombo.setCurrentIndex(0)
        self.win.instr_file = None
    def start_excel(self, file_path):
        self.fig = plt.figure(figsize=(16, 8))
        self.fig.canvas.mpl_connect('close_event', self.on_close)

        self.gpsfig, self.pltfig = self.fig.subfigures(1, 2, width_ratios=[1, 3])
        self.gpsax2d = self.gpsfig.add_subplot(2, 1, 1)
        self.gpsax3d = self.gpsfig.add_subplot(2, 1, 2, projection='3d')

        self.gpsax2d.set_title("GPS position", fontsize=10, fontweight='bold')
        self.gpsax2d.set_xlabel(xlabel="Longitude", fontsize=8)
        self.gpsax2d.set_ylabel(ylabel="Latitude", fontsize=8)
        self.gpsax2d.tick_params(axis='both', which='major', labelsize=6)
        self.gpsax2d.ticklabel_format(axis='both', scilimits=(0, 0))
        self.gpsax2d.yaxis.get_offset_text().set_fontsize(6)
        self.gpsax2d.xaxis.get_offset_text().set_fontsize(6)
        self.pltaxes = []

        self.gpsax3d.yaxis.get_offset_text().set_fontsize(6)
        
        self.gpsax3d.xaxis.get_offset_text().set_fontsize(6)
        
        workbook = openpyxl.load_workbook(file_path, data_only=True)  
        sheet = workbook.active

        getval = lambda c: str(sheet[c].value)
    
        self.pltaxes = self.pltfig.subplots(2, (int(getval('D3'))-int(getval('C3'))+2)//2).flatten()
        for ind, row in enumerate(sheet[ 'C'+getval('C3') : 'H'+getval('D3')]):
            self.init_graph(ind, *[i.value for i in row])

        for row in sheet[ 'C'+getval('C4') : 'O'+getval('D4')]: 
            temp = [i.value for i in row]

        for row in sheet[ 'C'+getval('C5') : 'V'+getval('D5')]:
            self.init_housekeeping(*[i.value for i in row])

        self.gpsfig.subplots_adjust(left=0.2, bottom=0.08, right=0.9, top=0.95, hspace=0.25, wspace=0.25)
        self.pltfig.subplots_adjust(left=0.03, bottom=0.08, right=0.97, top=0.95, hspace=0.25, wspace=0.25)
        self.fig.canvas.draw()

        plt.show()
    
    def add_map(self, map_file): 
        gpsmap = loadmat(map_file)
        latlim = gpsmap['latlim'][0]
        lonlim = gpsmap['lonlim'][0]
        mapdata = gpsmap['ZA']

        self.gpsax2d.imshow(mapdata, extent=[*latlim, *lonlim])

        mapdata = mapdata/255

        x_1 = np.arange(latlim[0], latlim[1], (latlim[1]-latlim[0])/mapdata.shape[1])
        y_1 = np.arange(lonlim[0],lonlim[1], (lonlim[1]-lonlim[0])/mapdata.shape[0])
        x_1, y_1 = np.meshgrid(x_1, y_1)
        
        self.gpsax3d.plot_surface(
            x_1, y_1, np.array([[0]]), cstride=1, rstride=1, facecolors=mapdata, shade=False
        )
        self.gpsax3d.axes.set_zlim3d(bottom=0, top=150)
        self.fig.canvas.draw()

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
    
    def init_housekeeping(self, title, protocol, boardID, length, rate, numpoints, nbyte, nbitmask, nbitshift, *ttable):
        groupBox = QGroupBox(title)
        hkLayout = QGridLayout()
        hkValues = []
        for ind, do_hk in enumerate(ttable):
            hkLabel = QLabel(self.hkNames[ind])
            hkValue = QLineEdit()
            if not do_hk:
                hkLabel.setEnabled(False)
                hkValue.setEnabled(False)
            
            hkValue.setFixedWidth(50)
            hkValue.setReadOnly(True)
            
            hkLayout.addWidget(hkLabel, ind, 0)
            hkLayout.addWidget(hkValue, ind, 1)
            
            hkValues.append(hkValues)

        groupBox.setLayout(hkLayout)

        self.win.hkLayout.addWidget(groupBox)
    def start(self):
        print("Start not implemetntes")

