import os, sys
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

import openpyxl

from scipy.io import loadmat

class Plotting():
    def __init__(self, on_close):
        self.on_close = on_close
        self.fig = None
        self.gpsfig, self.pltfig = None, None
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
        
        wb = openpyxl.load_workbook(file_path, data_only=True)  
        sh = wb.active

        getval = lambda c: str(sh[c].value)
    
        self.pltaxes = self.pltfig.subplots(2, (int(getval('D3'))-int(getval('C3'))+2)//2).flatten()
        for ind, row in enumerate(sh[ 'C'+getval('C3') : 'H'+getval('D3')]):
            self.init_graph(ind, *[i.value for i in row])

        for row in sh[ 'C'+getval('C4') : 'O'+getval('D4')]: 
            temp = [i.value for i in row]

        for row in sh[ 'C'+getval('C5') : 'U'+getval('D5')]:
            temp = [i.value for i in row]

        self.gpsfig.subplots_adjust(left=0.2, bottom=0.08, right=0.9, top=0.95, hspace=0.25, wspace=0.25)
        self.pltfig.subplots_adjust(left=0.03, bottom=0.08, right=0.97, top=0.95, hspace=0.25, wspace=0.25)
        self.fig.canvas.draw()

        plt.show()

        ## instrumentformat = read_excel(file_path, 'Format', skiprows=formatloc[0][1] - 1, nrows=formatloc[1][1], usecols="C:O", names=range(0, 13))

        ## for index, row in instrumentformat.iterrows():
        ##     g, color, protocol, signed, *bytedata = row.tolist()
        ##     plotting.graphs[g].addline(color, protocol, signed, bytedata)

        ## housekeepingformat = read_excel(file_path, 'Format', skiprows=formatloc[0][2] - 1, nrows=formatloc[1][2], usecols="C:J", names=range(0, 8))
        ## for index, row in housekeepingformat.iterrows():
        ##     plotting.HouseKeepingData(*row)

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

