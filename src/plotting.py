import os, sys
import socket

import numpy as np

import matplotlib.pyplot as plt

import openpyxl

from PyQt5.QtWidgets import QGridLayout, QGroupBox, QWidgetItem, QSpacerItem, QLabel, QLineEdit
from scipy.io import loadmat

MINFRAME_LEN = 2 * 40
PACKET_LENGTH = MINFRAME_LEN + 44  
MAX_READ_LENGTH = PACKET_LENGTH * 5000  
SYNC = [64, 40, 107, 254]


e = np.arange(MINFRAME_LEN)
for i in range(0, MINFRAME_LEN, 4):
    e[i:i+4] = e[i:i+4][::-1]
sync_arr = np.array(SYNC)
target_sync = np.dot(sync_arr, sync_arr)
def find_SYNC(seq):
    candidates = np.where(np.correlate(seq, sync_arr, mode='valid') == target_sync)[0]
    check = candidates[:, np.newaxis] + np.arange(4)
    mask = np.all((np.take(seq, check) == sync_arr), axis=-1)
    return candidates[mask]   

class Channel():
    def __init__():
        pass

class Plotting():
    def __init__(self, win):
        self.win = win
        self.fig = None
        self.gpsfig, self.pltfig = None, None
        self.hkNames = ["Temp1", "Temp2", "Temp3", "Int. Temp", "V Bat", "-12 V", "+12 V", "+5 V", "+3.3", "VBat Mon", "Dig. Temp"]
        self.gpsNames = ["Longitude (deg)", "Latitude (deg)", "Altitude (km)", "vEast (m/s)", "vNorth (m/s)", "vUp (m/s)", "Horz. Speed (m/s)", "Num Sats"]


        gpsGroupBox = QGroupBox("GPS")
        gpsLayout = QGridLayout()
        gpsValues = []
        for ind, name in enumerate(self.gpsNames):
            
            gpsLabel = QLabel(name)
            gpsValue = QLineEdit()

            gpsValue.setFixedWidth(50)
            gpsValue.setReadOnly(True)
            
            gpsLayout.addWidget(gpsLabel, ind, 0)
            gpsLayout.addWidget(gpsValue, ind, 1)
            
        gpsGroupBox.setLayout(gpsLayout)

        self.win.gpsLayout.addWidget(gpsGroupBox)

    def on_close(self, close_msg):
        self.win.pickInstrCombo.setEnabled(True)
        self.win.pickInstrButton.setEnabled(True)
        self.win.pickInstrCombo.setCurrentIndex(0)
        self.win.instr_file = None

        self.clear_layout(self.win.hkLayout)
        self.win.gpsWidget.hide()
        self.win.hkWidget.hide()
        

    def clear_layout(self, layout):    
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)

            if isinstance(item, QWidgetItem):
                print("widget" + str(item))
                item.widget().close()

            elif isinstance(item, QSpacerItem):
                print("spacer " + str(item))
            else:
                print("layout " + str(item))
                self.clear_layout(item.layout())

            # remove the item from layout
            layout.removeItem(item)
                
    def close(self):    
        plt.close()

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
        
        self.gpsfig.subplots_adjust(left=0.2, bottom=0.08, right=0.9, top=0.95, hspace=0.25, wspace=0.25)
        

        workbook = openpyxl.load_workbook(file_path, data_only=True)  
        sheet = workbook.active

        getval = lambda c: str(sheet[c].value)

        # Graphs    
        self.pltaxes = self.pltfig.subplots(2, (int(getval('D3'))-int(getval('C3'))+2)//2).flatten('F')
        for title, xlabel, ylabel, xlim, ylim1, ylim2, ax in zip(*np.transpose(sheet[ 'C'+getval('C3'):'H'+getval('D3')]), self.pltaxes):
            ax.set_title(title.value, fontsize=10, fontweight='bold')
            ax.set_xlabel(xlabel=xlabel.value, fontsize=8)
            ax.set_ylabel(ylabel=ylabel.value, fontsize=8)
            ax.set_xlim(0, xlim.value)
            ax.set_ylim(ylim1.value * 1.1, ylim2.value * 1.1)
            ax.tick_params(axis='both', which='major', labelsize=6)
            ax.ticklabel_format(axis='both', scilimits=(0, 0))
            ax.yaxis.get_offset_text().set_fontsize(6)
            ax.xaxis.get_offset_text().set_fontsize(6)
        self.pltfig.subplots_adjust(left=0.03, bottom=0.08, right=0.97, top=0.95, hspace=0.25, wspace=0.25)
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        plt.pause(0.1)

        # Channels
        for graphn, color, protocol, signed, *b in sheet[ 'C'+getval('C4') : 'O'+getval('D4')]:
            datay = np.zeros(int(self.pltaxes[graphn.value].get_xlim()[1]))
            datax = np.arange(int(self.pltaxes[graphn.value].get_xlim()[1]))
            line, = self.pltaxes[graphn.value].plot(datax, datay, color=color.value, lw=1, linestyle='None', marker='.', markersize=0.1, animated=True)
            self.pltaxes[graphn.value].draw_artist(line)
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

        # Housekeeping
        for title, protocol, boardID, length, rate, numpoints, nbyte, nbitmask, nbitshift, *ttable in sheet[ 'C'+getval('C5') : 'V'+getval('D5')]:
            hkGroupBox = QGroupBox(title.value)
            hkLayout = QGridLayout()
            hkValues = []

            for ind, do_hk in enumerate(ttable):
                hkLabel = QLabel(self.hkNames[ind])
                hkValue = QLineEdit()
                if not do_hk.value:
                    hkLabel.setEnabled(False)
                    hkValue.setEnabled(False)
                
                hkValue.setFixedWidth(50)
                hkValue.setReadOnly(True)
                
                hkLayout.addWidget(hkLabel, ind, 0)
                hkLayout.addWidget(hkValue, ind, 1)
                
                hkValues.append(hkValues)

            hkGroupBox.setLayout(hkLayout)

            self.win.hkLayout.addWidget(hkGroupBox)
        self.win.hkWidget.show()
        self.win.gpsWidget.show()

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
   
    def start_housekeeping(self, title, protocol, boardID, length, rate, numpoints, nbyte, nbitmask, nbitshift, *ttable):
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
    
    def start_channel():
        pass
    def parse(self, mode):
        print("Starts") #dlt
        read_file = None
        write_file = None
        if mode == 0:
            read_file = open( self.win.read_file, "rb")
        elif mode == 1:
            udp_ip = self.win.hostInputLine.text()
            port = self.win.portInputLine.text()

            print(f"[Debug] Connected\nIP: {udp_ip}\n Port: {port}")    
            sock = socket.socket(socket.AF_INET, # Internet
                        socket.SOCK_DGRAM) # UDP
            sock.bind((udp_ip, port)) 


        while True:
            print("loops")
            if mode == 0:
                raw_data = np.fromfile(read_file, dtype=np.uint8, count=MAX_READ_LENGTH)
            elif mode == 1:
                raw_data, addr = sock.recvfrom(MAX_READ_LENGTH)

            if len(raw_data) == 0:
                break
            
            if self.win.do_write:
                raw_data.tofile(self.win.write_file)

            inds = find_SYNC(raw_data)       
            prev_ind = inds[-1]
            inds = inds[:-1][(np.diff(inds) == PACKET_LENGTH)]
            inds[:-1] = inds[:-1][(np.diff(raw_data[inds + 6]) != 0)]

            minframes = raw_data[inds[:, None] + e].astype(int)

            oddframe = minframes[np.where(minframes[:, 57] & 3 == 1)]
            evenframe = minframes[np.where(minframes[:, 57] & 3 == 2)]
            oddsfid = minframes[np.where(minframes[:, 5] % 2 == 1)]
            evensfid = minframes[np.where(minframes[:, 5] % 2 == 0)]
        
        self.win.readStart.setChecked(True)
        self.win.toggle_parse()
        print("Stopped")