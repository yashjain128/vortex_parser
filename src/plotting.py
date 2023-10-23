"""
Module to handle plotting, housekeeping, and GPS

Written by Yash Jain
"""
import socket
import time

import numpy as np
import openpyxl
from vispy import scene, plot, app
from vispy.visuals.transforms import STTransform
import pyproj

from PyQt5.QtWidgets import QGridLayout, QGroupBox, QWidgetItem, QSpacerItem, QLabel, QLineEdit, QWidget
from PyQt5.QtCore import Qt, QTimer
from scipy.io import loadmat

from scrollingplotwidget import ScrollingPlotWidget


SYNC = [64, 40, 107, 254]
MINFRAME_LEN = 2 * 40
PACKET_LENGTH = MINFRAME_LEN + 44  
MAX_READ_LENGTH = PACKET_LENGTH * 5000  

RV_HEADER = [114, 86, 48, 50, 65]
RV_LEN = 48

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

class Channel:
    def __init__(self, ax, color, signed, b):
        
        self.signed = signed
        self.datax = np.arange(ax.xlims[1])
        self.datay = np.zeros(ax.xlims[1])
        self.pltaxes = []        
        self.byte_info = [b[i:i+3] for i in range(0,9,3) if b[i]!=-1]

        self.color = color
        self.line = scene.Markers(pos=np.transpose(np.array([self.datax, self.datay])), edge_width=0, size=1, face_color=self.color, antialias=False)
        ax.add_line(self.line)

        self.xlims, self.ylims = ax.xlims, ax.ylims

    def new_data(self, minframes):
        l = len(minframes)
        self.datay[:l] = np.zeros(l)
        for ind, mask, shift in self.byte_info:
            if shift < 0:
                self.datay[:l] += (minframes[:, ind] & mask) >> abs(shift)
            else:
                self.datay[:l] += (minframes[:, ind] & mask) << shift
        
        if self.signed:
            self.datay[:l] = self.datay[:l]+(self.datay[:l] >= self.ylims[1])*(2*self.ylims[0])
        self.datay = np.roll(self.datay, -l)

        data = np.transpose(np.array([self.datax, self.datay]))
        self.line.set_data(pos=data, edge_width=0, size=1, face_color=self.color)

    def reset(self):
        self.datay = np.zeros(self.xlims[1])
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
        self.setWindowTitle("Figure 1")

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
            
            gpsValues.append(gpsValue)
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
                #print("widget" + str(item))
                item.widget().close()

            elif isinstance(item, QSpacerItem):
                #print("spacer " + str(item))
                pass
            else:
                #print("layout " + str(item))
                self.clear_layout(item.layout())

            # remove the item from layout
            layout.removeItem(item)

    def on_key_press(self, event):
        if (event.text=='\x12'): # When Ctrl+R is pressed reset the bounds of every axes
            for ax in self.pltaxes:
                ax.reset_bounds()
            self.gpsax2d.reset_bounds()
    
    def reset_channels(self):
        for chs in self.channels:
            for ch in chs:
                ch.reset()

    def start_excel(self, file_path, plot_width):

        # Create a plotting figure and add it to the GUI
        self.fig = plot.Fig(size=(1200, 800), show=False, keys=None)
        
        # Temporarily unfreeze the figure to add a key press event and set the default class to a custom scrolling plot widget
        self.fig.unfreeze()
        self.fig.on_key_press = self.on_key_press
        self.fig._grid._default_class = ScrollingPlotWidget
        self.fig.freeze()

        self.widget_layout.addWidget(self.fig.native)
        # Set fig default class to a custom scrolling plot widget
        
        
        self.gpsax2d = self.fig[0, 0].configure2d(title="GPS position", 
                                                xlabel="Longitude", 
                                                ylabel="Latitude")

        self.gpsax3d = self.fig[1, 0].configure3d(title="GPS position", 
                                                xlabel="Longitude", 
                                                ylabel="Latitude",
                                                zlabel="Altitude")
        
        self.gps_pos_lat = np.zeros(25000, float)
        self.gps_pos_lon = np.zeros(25000, float)
        self.gps_pos_alt = np.zeros(25000, float)
        
        self.gps_points = scene.Markers(pos=np.transpose(np.array([self.gps_pos_lat, self.gps_pos_lon])),face_color="#ff0000", edge_width=0, size=5, parent=self.gpsax2d.plot_view.scene, antialias=False, symbol='s')
        
        xl_sheet = openpyxl.load_workbook(file_path, data_only=True).active
        getval = lambda c: str(xl_sheet[c].value)
        self.show()

        # Graphs
        self.pltaxes = []        
        graph_arr = [[i]+list(map(lambda x:x.value, row)) for i, row in enumerate(xl_sheet[ 'C'+getval('C3'):'H'+getval('D3')])]
        for i, title, xlabel, ylabel, numpoints, ylim1, ylim2 in graph_arr:
            ax = self.fig[i%2, i//2+1].configure2d( title, xlabel, ylabel,(0, numpoints*plot_width), (ylim1, ylim2)) 
            self.pltaxes.append(ax)

        # Channels
        self.channels = [[], [], [], [], []]
        channel_arr = [list(map(lambda x:x.value, row)) for row in xl_sheet[ 'C'+getval('C4') : 'O'+getval('D4')]];
        for graphn, color, protocol, signed, *b in channel_arr:
            ax = self.pltaxes[graphn]
            channel = Channel(ax, color, signed, b)
            self.channels[protocols.index(protocol)].append(channel)
        for ax in self.pltaxes:
            ax.add_gridlines()

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

    def add_map(self, map_file): 
        # Get data from .mat file
        gpsmap = loadmat(map_file)
        latlim = gpsmap['latlim'][0]
        lonlim = gpsmap['lonlim'][0]
        mapdata = gpsmap['ZA']

        # plot and scale 2d map
        map2d = scene.visuals.Image(mapdata, method='subdivide', parent=self.gpsax2d.plot_view.scene)
        img_width, img_height = lonlim[1]-lonlim[0], latlim[1]-latlim[0]
        transform2d = STTransform(scale=(img_width/mapdata.shape[1], img_height/mapdata.shape[0]), translate=(lonlim[0], latlim[0]))
        map2d.transform = transform2d
        self.gpsax2d.xlims = lonlim
        self.gpsax2d.ylims = latlim
        self.gpsax2d.reset_bounds()
        # plot and scale 3d map
        transform3d = STTransform(scale=(1/mapdata.shape[1], 1/mapdata.shape[0]))
        map3d = scene.visuals.Image(mapdata, method='subdivide', parent=self.gpsax3d.plot_view.scene )
        map3d.transform = transform3d
        self.gpsax3d.xaxis.domain = lonlim
        self.gpsax3d.yaxis.domain = latlim

    def parse(self, read_mode, read_file, udp_ip, udp_port): 
        plt_hertz=self.win.plotHertzSpin.value()
        read_length = MAX_READ_LENGTH//plt_hertz
        timer = time.perf_counter()
        do_update = True

        read_file = None
        write_file = None
        if read_mode == 0:
            read_file = open(self.win.read_file, "rb")
        elif read_mode == 1:
            print(f"[Debug] Connected\nIP: {udp_ip}\n Port: {udp_port}")    
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
            sock.bind(("", udp_port)) 

        self.reset_channels()
        self.run = True
        start_time = time.perf_counter()
        while self.run:
            if read_mode == 0:
                raw_data = np.fromfile(read_file, dtype=np.uint8, count=read_length)
            elif read_mode == 1:
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
            #print(all_minframes)
            protocol_minframes = [all_minframes,
                all_minframes[np.where(all_minframes[:, 57] & 3 == 1)],
                all_minframes[np.where(all_minframes[:, 57] & 3 == 2)],
                all_minframes[np.where(all_minframes[:, 5] % 2 == 1)],
                all_minframes[np.where(all_minframes[:, 5] % 2 == 0)]]
            
            gps_raw_data = all_minframes[:, [6, 26, 46, 66]].flatten()
            gps_check = all_minframes[:, [7, 27, 47, 67]].flatten()
            gps_data = gps_raw_data[np.where(gps_check==128)]

            if len(gps_data)==0:
                continue
            gps_inds = find_RV(gps_data)

            if len(gps_data) - gps_inds[-1] < RV_LEN:
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

            # check if plt_hertz time has elapsed, set do_update to true 
            for chs, hks, minframes in zip(self.channels, self.housekeeping, protocol_minframes):
                for ch in chs:
                    ch.new_data(minframes)

                for hk in hks:
                    hk.new_data(minframes)

            self.gps_points.set_data(pos=np.transpose(np.array([self.gps_pos_lat, self.gps_pos_lon])) ,face_color="#ff0000", edge_width=0, size=3, symbol='s')
            app.process_events()

            pause_time = max((1/plt_hertz) - (time.perf_counter()-start_time), 0) 
            # Change pause time with threading?
            time.sleep(pause_time)
            start_time = time.perf_counter()
        print(f"Done : {time.perf_counter()-start_time}")
 