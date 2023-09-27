"""
Module to handle plotting, housekeeping, and GPS

Written by Yash Jain
"""

import socket
import time

import numpy as np
import openpyxl
from vispy import scene, plot, app
import pyproj

from PyQt5.QtWidgets import QGridLayout, QGroupBox, QWidgetItem, QSpacerItem, QLabel, QLineEdit, QWidget
from PyQt5.QtCore import QObject, pyqtSignal, Qt
from scipy.io import loadmat

SYNC = [64, 40, 107, 254]
MINFRAME_LEN = 2 * 40
PACKET_LENGTH = MINFRAME_LEN + 44  
MAX_READ_LENGTH = PACKET_LENGTH * 5000  

RV_HEADER = [114, 86, 48, 50, 65]
RV_LENGTH = 48


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

rv_arr = np.array(RV_HEADER)
target_rv = np.dot(rv_arr, rv_arr)
def find_RV(seq):
    #print(seq, rv_arr)
    candidates = np.where(np.correlate(seq, rv_arr, mode='valid') == target_rv)[0]
    check = candidates[:, np.newaxis] + np.arange(5)
    mask = np.all((np.take(seq, check) == rv_arr), axis=-1)
    return candidates[mask] 

point_transformer = pyproj.Transformer.from_crs(
    {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
    {"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'},
)

hkNames = ["Temp1", "Temp2", "Temp3", "Int. Temp", "V Bat", "-12 V", "+12 V", "+5 V", "+3.3", "VBat Mon", "Dig. Temp"]
gpsNames = ["Longitude (deg)", "Latitude (deg)", "Altitude (km)", "vEast (m/s)", "vNorth (m/s)", "vUp (m/s)", "Horz. Speed (m/s)", "Num Sats"]

protocols = ['all', 'odd frame', 'even frame', 'odd sfid', 'even sfid']

class Graph():
    def __init__(self, ax, title, xlabel, ylabel, numpoints=1, ylim=(0, 1)):
        self.numpoints = numpoints
        self.ylim = ylim

        self.ax = ax
        self.ax.border_color = '#000000'
        self.ax.bgcolor = '#000000'

        grid = ax.add_grid()
        self.viewbox = grid.add_view(row=1, col=1, camera='panzoom')
        self.viewbox.camera.set_range(x=(0, self.numpoints), y=self.ylim)

        title = scene.Label(title, color='#ffffff', bold=True, font_size=12)
        title.height_max = 40
        grid.add_widget(title, row=0, col=1) 
        
        x_axis = scene.AxisWidget(orientation='bottom', axis_label=xlabel, axis_font_size=8, axis_label_margin=20, tick_label_margin=5)
        x_axis.stretch = (1, 0.2)
        grid.add_widget(x_axis, row=2, col=1)
        x_axis.link_view(self.viewbox)

        y_axis = scene.AxisWidget(orientation='left', axis_label=ylabel, axis_font_size=8, axis_label_margin=20, tick_label_margin=5)
        y_axis.stretch = (0.2, 1)
        grid.add_widget(y_axis, row=1, col=0)
        y_axis.link_view(self.viewbox)

        

class Channel():
    def __init__(self, ax: Graph, color, signed, b):
        self.ax = ax
        self.color = color
        self.signed = signed
        self.datax = np.arange(ax.numpoints)
        self.datay = np.zeros(ax.numpoints)
        self.line = scene.Markers(pos=np.transpose(np.array([self.datax, self.datay])), edge_width=0, size=1, face_color=self.color, parent=self.ax.viewbox.scene, antialias=False)
        self.byte_info = [b[i:i+3] for i in range(0,9,3) if b[i]!=-1] 
    def new_data(self, minframes):
        l = len(minframes)
        self.datay[:l] = np.zeros(l)
        for ind, mask, shift in self.byte_info:
            if shift < 0:
                self.datay[:l] += (minframes[:, ind] & mask) >> abs(shift)
            else:
                self.datay[:l] += (minframes[:, ind] & mask) << shift
        
        if self.signed:
            self.datay[:l] = self.datay[:l]+(self.datay[:l] >= self.ax.ylim[1])*(2*self.ax.ylim[0])
        self.datay = np.roll(self.datay, -l)

    def update(self):
        #print(self.data)
        self.line.set_data(pos=np.transpose(np.array([self.datax, self.datay])), edge_width=0, size=1, face_color=self.color)
        
        
class Housekeeping:
    def __init__(self, board_id, length, rate, numpoints, b_ind, b_mask, b_shift, hkvalues):
        self.board_id = board_id
        self.length = length
        self.indcol = np.arange(length)[:, None]
        self.rate = rate
        self.data = np.zeros((length, numpoints))
        self.b_ind, self.b_mask, self.b_shift = b_ind, b_mask, b_shift
        self.hkvalues = hkvalues
        self.hkrange = 10

    def new_data(self, minframes):
        databuffer = minframes[:, self.b_ind] & self.b_mask << abs(self.b_shift)
        inds = np.where(databuffer == self.board_id)[0]
        inds = inds[np.where(np.diff(inds) == self.length)[0]]
        self.data = np.roll(self.data, inds.size, axis=1)
        self.data[:, :inds.size] = databuffer[self.indcol + inds]

        self.hkrange = min(10, inds.size)

    def update(self):
        for edit, data_row in zip(self.hkvalues, self.data):
            if edit.isEnabled():
                edit.setText(str(np.average(data_row[:self.hkrange])))

class Plotting(QWidget):
    def __init__(self, win):
        QWidget.__init__(self)
        self.win = win
        self.fig = None
        self.gpsfig, self.pltfig = None, None
        self.widget_layout = QGridLayout()
        self.setLayout(self.widget_layout) 
        self.setWindowTitle("Figure")

        gpsGroupBox = QGroupBox("GPS")
        gpsLayout = QGridLayout()
        gpsValues = []
        
        for ind, name in enumerate(gpsNames):
            
            gpsLabel = QLabel(name)
            gpsValue = QLineEdit()

            gpsValue.setFixedWidth(50)
            gpsValue.setReadOnly(True)
            
            gpsLayout.addWidget(gpsLabel, ind, 0)
            gpsLayout.addWidget(gpsValue, ind, 1)
            
        gpsGroupBox.setLayout(gpsLayout)

        self.win.gpsLayout.addWidget(gpsGroupBox)

        self.channels = None
    def closeEvent(self, event):
        # Reset GUI after closing plotting window
        self.win.pickInstrCombo.setEnabled(True)
        self.win.pickInstrButton.setEnabled(True)
        self.win.plotHertzSpin.setEnabled(True)
        self.win.plotHertzLabel.setEnabled(True)
        self.win.pickInstrCombo.setCurrentIndex(0)
        self.win.instr_file = None

        self.clear_layout(self.win.hkLayout)
        self.clear_layout(self.widget_layout)
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

    
    def start_excel(self, file_path):
        plot_width = self.win.plotWidthSpin.value()
         
        start_time = time.perf_counter()
        self.fig = plot.Fig(show=False)  
        self.gpsax2d = Graph(self.fig[0, 0], "GPS position", "Longitude", "Latitude")
        self.gpsax3d = Graph(self.fig[1, 0], "GPS position", "Longitude", "Latitude")
        
        self.gps_pos_lat = np.zeros(25000, float)
        self.gps_pos_lon = np.zeros(25000, float)
        self.gps_pos_alt = np.zeros(25000, float)
        #self.gps_points = scene.Markers(face_color="#ff0000", edge_width=0, size=4, parent=self.gpsax2d.viewbox.scene, antialias=False, symbol='s')
        
        xl_sheet = openpyxl.load_workbook(file_path, data_only=True).active
        getval = lambda c: str(xl_sheet[c].value)

# Graphs
        self.pltaxes = []        
        graph_arr = [[i]+list(map(lambda x:x.value, row)) for i, row in enumerate(xl_sheet[ 'C'+getval('C3'):'H'+getval('D3')])]
        for i, title, xlabel, ylabel, numpoints, ylim1, ylim2 in graph_arr:
            ax = Graph(self.fig[i%2, i//2+1], 
                       title, xlabel, ylabel, numpoints*plot_width, (ylim1, ylim2))
            self.pltaxes.append(ax)
        self.show()
        self.pltaxes[0].viewbox.update()
        self.resize(600, 400)
# Channels
        self.channels = [[], [], [], [], []]
        channel_arr = [list(map(lambda x:x.value, row)) for row in xl_sheet[ 'C'+getval('C4') : 'O'+getval('D4')]];
        for graphn, color, protocol, signed, *b in channel_arr:
            ax = self.pltaxes[graphn]
            channel = Channel(ax, color, signed, b)
            self.channels[protocols.index(protocol)].append(channel)
        #self.fig.context.flush()        
        print(time.perf_counter()-start_time)
        #self.fig.show()
        
# Housekeeping
        self.housekeeping = [[], [], [], [], []]
        for title, protocol, boardID, length, rate, numpoints, b_ind, b_mask, b_shift, *ttable in xl_sheet[ 'C'+getval('C5') : 'V'+getval('D5')]:
            hkGroupBox = QGroupBox(title.value)
            hkLayout = QGridLayout()
            hkLayout.setAlignment(Qt.AlignTop)
            hkValues = []

            for ind, do_hk in enumerate(ttable):
                hkLabel = QLabel(hkNames[ind])
                hkValue = QLineEdit()
                if not do_hk.value:
                    hkLabel.setEnabled(False)
                    hkValue.setEnabled(False)
                
                hkValue.setFixedWidth(50)
                hkValue.setReadOnly(True)
                
                hkLayout.addWidget(hkLabel, ind, 0)
                hkLayout.addWidget(hkValue, ind, 1)
                
                hkValues.append(hkValue)

            hkGroupBox.setLayout(hkLayout)

            self.win.hkLayout.addWidget(hkGroupBox)
            
            self.housekeeping[protocols.index(protocol.value)].append(Housekeeping(boardID.value, length.value, rate.value, numpoints.value, b_ind.value, b_mask.value, b_shift.value, hkValues))
        self.win.hkWidget.show()
        self.win.gpsWidget.show()
        #app.Canvas.on_close = lambda: print("hi")

        self.widget_layout.addWidget(self.fig.native)
        #app.process_events()

    def add_map(self, map_file): 
        gpsmap = loadmat(map_file)
        latlim = gpsmap['latlim'][0]
        lonlim = gpsmap['lonlim'][0]
        mapdata = gpsmap['ZA']

        image = scene.visuals.Image(mapdata, parent=self.gpsax2d.viewbox.scene, method='subdivide')
    def parse(self):
        mode=self.win.read_mode
        
        plt_hertz=1/self.win.plotHertzSpin.value()
        timer = time.perf_counter()
        do_update = True

        read_file = None
        write_file = None
        if mode == 0:
            read_file = open( self.win.read_file, "rb")
        elif mode == 1:
            udp_ip = self.win.hostInputLine.text()
            port = int(self.win.portInputLine.text())

            print(f"[Debug] Connected\nIP: {udp_ip}\n Port: {port}")    
            sock = socket.socket(socket.AF_INET, # Internet
                        socket.SOCK_DGRAM) # UDP
            sock.bind(("", port)) 

        start_time = time.perf_counter()
        self.run = True
        while self.run:
            if mode == 0:
                raw_data = np.fromfile(read_file, dtype=np.uint8, count=MAX_READ_LENGTH)
            elif mode == 1:
                soc_data = sock.recv(1024)
                raw_data = np.frombuffer(soc_data, dtype=np.uint8)
                
            if len(raw_data) == 0:
                break
            
            if self.win.do_write:
                raw_data.tofile(self.win.write_file)
            

            inds = find_SYNC(raw_data)
        
            if len(inds)==0:
                print("No valid sync frames")
                continue

            prev_ind = inds[-1]
            inds = inds[:-1][(np.diff(inds) == PACKET_LENGTH)]
            inds[:-1] = inds[:-1][(np.diff(raw_data[inds + 6]) != 0)]

            all_minframes = raw_data[inds[:, None] + e].astype(int)
            protocol_minframes = [all_minframes,
                all_minframes[np.where(all_minframes[:, 57] & 3 == 1)],
                all_minframes[np.where(all_minframes[:, 57] & 3 == 2)],
                all_minframes[np.where(all_minframes[:, 5] % 2 == 1)],
                all_minframes[np.where(all_minframes[:, 5] % 2 == 0)]]
            
            cur_time = time.perf_counter()
            if (cur_time-timer > plt_hertz):
                #print("Update")
                timer = cur_time
                do_update = True
            else:
                pass
                #print("Skipped")          
            
            for chs, hks, minframes in zip(self.channels, self.housekeeping, protocol_minframes):
                for ch in chs:
                    ch.new_data(minframes)
                    if do_update:
                        ch.update()

                for hk in hks:
                    hk.new_data(minframes)
                    if do_update:
                        hk.update()
                
            app.process_events()

            do_update = False
            '''
            gps_raw_data = all_minframes[:, [6, 26, 46, 66]].flatten()
            gps_check = all_minframes[:, [7, 27, 47, 67]].flatten()
            gps_data = gps_raw_data[np.where(gps_check==128)]

            if len(gps_data)==0:
                continue
            gps_inds = find_RV(gps_data)

            if len(gps_data) - gps_inds[-1] < RV_LENGTH:
                gps_inds = gps_inds[:-1]

            gpsmatrix = gps_data[np.add.outer(gps_inds, np.arange(48))].astype(np.uint64)
            num_RV = np.shape(gpsmatrix)[0]
            # Note skipped check sum

            gps_pos_ecef = ((gpsmatrix[:, [12, 20, 28]] << 32) +
                            (gpsmatrix[:, [11, 19, 27]] << 24) +
                            (gpsmatrix[:, [10, 18, 26]] << 16) +
                            (gpsmatrix[:, [9, 17, 25]] << 8) +
                            (gpsmatrix[:, [16, 24, 32]]) -
                            ((gpsmatrix[:, [12, 20, 28]]>=128)*(2**40))).transpose()/10000

            self.gps_pos_lat[:num_RV], self.gps_pos_lon[:num_RV], self.gps_pos_alt[:num_RV] = point_transformer.transform(*gps_pos_ecef, radians=False)
            self.gps_pos_lon = np.roll(self.gps_pos_lon, -num_RV)
            self.gps_pos_lat = np.roll(self.gps_pos_lat, -num_RV)
            self.gps_points.set_data(self.gps_pos_lon, self.gps_pos_lat)
            ''' 
        print(f"Done : {time.perf_counter()-start_time}")
        #self.moveToThread(self.win.mainThread)
        self.win.setupGroupBox.setEnabled(True)
        self.win.readStart.setText("Start")
        self.win.readStart.setStyleSheet("background-color: #e34040")
        self.win.readStart.setChecked(False)
        self.win.timer.stop()
        
        #self.finished.emit()