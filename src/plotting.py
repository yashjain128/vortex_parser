"""
Module to handle plotting, housekeeping, and GPS

Written by Yash Jain
"""
import socket
import time
import math

import numpy as np
from vispy import scene, plot, app
from vispy.visuals.transforms import STTransform

from pymap3d.ecef import ecef2geodetic, ecef2enuv

from scipy.io import loadmat

from scrollingplotwidget import ScrollingPlotWidget


SYNC = [64, 40, 107, 254]
MINFRAME_LEN = 2 * 40
PACKET_LENGTH = MINFRAME_LEN + 44  
MAX_READ_LENGTH = PACKET_LENGTH * 5000

RV_HEADER = [114, 86, 48, 50, 65]
RV_LEN = 48

# how many decimal places to round gps data
DEC_PLACES = 3

plot_width = 5

HK_NAMES = ["Temp1", "Temp2", "Temp3", "Int. Temp", "V Bat", "-12 V", "+12 V", "+5 V", "+3.3", "VBat Mon", "Dig. Temp"]
GPS_NAMES = ["Longitude (deg)", "Latitude (deg)", "Altitude (km)", "vEast (m/s)", "vNorth (m/s)", "vUp (m/s)", "Horz. Speed (m/s)", "Num Sats"]
GPS_NAMES_ID = ["lon", "lat", "alt", "veast", "vnorth", "vup", "shorz", "numsats"]

PROTOCOLS = ['all', 'odd frame', 'even frame', 'odd sfid', 'even sfid']

windows = []
figures = {}
plot_graphs = []
map_graphs = []

data_channels = {protocol:[] for protocol in PROTOCOLS} # Sort channels and hk by protocol
gps2d_points = []
gps3d_points = []

close_signal = None
# Allocate memory for gps data
gps_data = {gps_name:np.zeros(25000, float) for gps_name in GPS_NAMES_ID}

running = True
closing = False
# Swap endianness: [3, 2, 1, 0, 7, 6, 5, 4 ... 79, 78, 77, 76]
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

class Channel:
    def __init__(self, color, signed, numpoints, *raw_byte_info):
        self.signed = signed
        self.color = color



        bit_num = 0
        self.byte_info = []

        for i in range(len(raw_byte_info), 0, -2): 
            # index and mask are given
            # infer bit shift from first bit in mask and how many bits have passed
            
            ind = raw_byte_info[i-2]
            mask = raw_byte_info[i-1]
            shift = int(bit_num - math.log2(mask & -mask))

            self.byte_info.append([ind, mask, shift])            
            bit_num += mask.bit_count()

        self.xlims = [0, numpoints]
        # Infer y limits from number of bits
        if self.signed:
            self.ylims = [-2**(bit_num-1), 2**(bit_num-1)]
        else:
            self.ylims = [0, 2**bit_num]

        self.datay = np.zeros(numpoints)
        self.datax = np.arange(numpoints)

        self.line = scene.Markers(pos=np.transpose(np.array([self.datax, self.datay])), edge_width=0, size=1, face_color=self.color, antialias=False)

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
    def __init__(self, board_id, length, numpoints, b_ind, b_mask, values):
        self.board_id = board_id
        self.length = length
        self.indcol = np.arange(length)[:, None]
        self.data = np.zeros((length, numpoints))
        self.b_ind, self.b_mask = b_ind, b_mask
        self.values = values
        self.hkrange = 10

    def new_data(self, minframes):
        databuffer = minframes[:, self.b_ind] & self.b_mask
        inds = np.where(databuffer == self.board_id)[0]
        inds = inds[np.where(np.diff(inds) == self.length)[0]]
        self.data = np.roll(self.data, inds.size, axis=1)
        self.data[:, :inds.size] = databuffer[self.indcol + inds]

        self.hkrange = min(10, inds.size)

        for edit, data_row in zip(self.values, self.data):
            if self.hkrange==0:
                edit.setText("null")
            elif edit.isEnabled():
                edit.setText(str(np.average(data_row[:self.hkrange])))

    def reset(self):
        for value in self.values:
            value.setText("")

def get_fig(figure):
    # Add figure if it does not exist already
    if figure not in figures:
        figures[figure] = plot.Fig(title=figure, show=False, keys=None)
        fig = figures[figure]
        
        fig.unfreeze()
        fig.on_key_press = on_key_press
        fig._grid._default_class = ScrollingPlotWidget # Change the default class with cu
        fig.native.closeEvent = on_close # Quick fix of Issue 1201 with vispy: https://github.com/vispy/vispy/issues/1201
        fig.freeze()

        return fig 
    
    else:
        return figures[figure]
def on_key_press(key):
    if (key.text=='\x12'): # When Ctrl+R is pressed reset the bounds of every axes
        for graph in plot_graphs:
            graph.reset_bounds()

def on_close(event):
    global running, closing, windows, figures, plot_graphs, data_channels, housekeeping_arr, channels_arr
    if closing:
        return
    running = False
    closing = True

    for fig in figures.values():
        fig.close()

    close_signal()

    figures.clear()
    plot_graphs.clear()
    [obj_arr.clear() for obj_arr in data_channels.values()] # Sort channels and hk by protocol

    closing = False

def add_graph(figure, title, row, col, xlabel, ylabel, numpoints):
    row = int(row)
    col = int(col)
    numpoints = int(numpoints)

    fig = get_fig(figure)

    plot_graphs.append(
        fig[int(row), int(col)].configure2d(title, xlabel, ylabel, xlims=[0, numpoints*plot_width])
    )
    
def add_channel(color, protocol, signed, *byte_info):
    signed = signed=="True"
    byte_info = [int(i) for i in byte_info]
    # Take last added graph
    graph = plot_graphs[-1]
    
    channel = Channel(color, signed, graph.xlims[1], *byte_info)
    graph.add_line(channel.line)

    # Graph must fit the channel data
    graph.ylims[0] = min(graph.ylims[0], channel.ylims[0])
    graph.ylims[1] = max(graph.ylims[1], channel.ylims[1])
    
    data_channels[protocol].append(channel)

def add_map(figure, name, row, col, type):
    fig = get_fig(figure)
    if type=="2d":
        fig[int(row), int(col)].configure2d(title=name, xlabel="Longitude", ylabel="Latitude")

        gps2d_points.append(
            scene.Markers(pos=np.transpose(np.array([gps_data["lat"], gps_data["lon"]])),face_color="#ff0000", edge_width=0, size=5, parent=fig[int(row), int(col)].plot_view.scene, antialias=False, symbol='s')
        )
    elif type=="3d":
        # axis labels wont work yet
        fig[int(row), int(col)].configure3d(title=name, xlabel="Longitude", ylabel="Latitude", zlabel="Altitude")    

    map_graphs.append(fig[int(row), int(col)])

def add_housekeeping(protocol, board_id, length, numpoints, byte_ind, bitmask, hkvalues):
    board_id = int(board_id)
    length = int(length)
    numpoints = int(numpoints)
    byte_ind = int(byte_ind)
    bitmask = int(bitmask)
    housekeeping_ = Housekeeping(board_id, length, numpoints, byte_ind, bitmask, hkvalues)
    data_channels[protocol].append(housekeeping_)

def finish_creating():
    for graph in plot_graphs:
        graph.reset_bounds()
        graph.add_gridlines()

    for fig in figures.values():
        fig.show()

def reset_graphs():
    for obj_arr in data_channels.values():
        for obj in obj_arr:
            obj.reset()

def set_map(map_file):
    gpsmap = loadmat(map_file)
    latlim = gpsmap['latlim'][0]
    lonlim = gpsmap['lonlim'][0]
    mapdata = gpsmap['ZA']

    for map_graph in map_graphs:
        if map_graph.dimensions == 2:
            # plot and scale 2d map
            map2d = scene.visuals.Image(mapdata, method='subdivide', parent = map_graph.plot_view.scene)
            img_width, img_height = lonlim[1]-lonlim[0], latlim[1]-latlim[0]
            transform2d = STTransform(scale=(img_width/mapdata.shape[1], img_height/mapdata.shape[0]), translate=(lonlim[0], latlim[0]))
            map2d.transform = transform2d
            
            map_graph.xlims = lonlim
            map_graph.ylims = latlim
            map_graph.reset_bounds()
        
        elif map_graph.dimensions == 3:
            # plot and scale 3d map
            transform3d = STTransform(scale=(1/mapdata.shape[1], 1/mapdata.shape[0]))
            map3d = scene.visuals.Image(mapdata, method='subdivide', parent=map_graph.plot_view.scene )
            map3d.transform = transform3d
            map_graph.xaxis.domain = lonlim
            map_graph.yaxis.domain = latlim

def parse(read_mode, plot_hertz, read_file, udp_ip, udp_port):
        global running, gps_data
        read_length = MAX_READ_LENGTH//plot_hertz
        timer = time.perf_counter()
        do_update = True

        if read_mode == 0:
            read_file = open(read_file, "rb")

        elif read_mode == 1:
            print(f"[Debug] Connected\nIP: {udp_ip}\n Port: {udp_port}")    
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
            sock.bind(("", udp_port)) 
        
        reset_graphs()
        
        running = True
        start_time = time.perf_counter()

        while running:
            if read_mode == 0:
                raw_data = np.fromfile(read_file, dtype=np.uint8, count=read_length)

            elif read_mode == 1:
                soc_data = sock.recv(1024)
                raw_data = np.frombuffer(soc_data, dtype=np.uint8)
                
            if len(raw_data) == 0:
                break
            
            #if self.win.do_write:
            #    raw_data.tofile(self.win.write_file)

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
            

            # Gps bytes are at 6, 26, 46, 66 when the next byte == 128
            gps_raw_data = all_minframes[:, [6, 26, 46, 66]].flatten()
            gps_check = all_minframes[:, [7, 27, 47, 67]].flatten()
            gps_data = gps_raw_data[np.where(gps_check==128)]

            # Parse gps data when there are bytes available
            if len(gps_data) != 0:
                gps_inds = find_RV(gps_data)

                if len(gps_data) - gps_inds[-1] < RV_LEN:
                    gps_inds = gps_inds[:-1]

                gpsmatrix = gps_data[np.add.outer(gps_inds, np.arange(48))].astype(np.uint64)

                # Number of rv packets
                num_RV = np.shape(gpsmatrix)[0]
                # Note skipped check sum

                # Signed position data
                gps_pos_ecef = (((gpsmatrix[:, [12, 20, 28]] << 32) |
                                (gpsmatrix[:, [11, 19, 27]] << 24) |
                                (gpsmatrix[:, [10, 18, 26]] << 16) |
                                (gpsmatrix[:, [ 9, 17, 25]] <<  8) |
                                (gpsmatrix[:, [16, 24, 32]])) - 
                                ((gpsmatrix[:, [12, 20, 28]]>=128)*(1<<40))).transpose()/10000
                
                gps_vel_ecef = (((gpsmatrix[:, [37, 41, 45]] << 20) |
                                 (gpsmatrix[:, [36, 40, 44]] << 12) |
                                 (gpsmatrix[:, [35, 39, 43]] << 4) |
                                 (gpsmatrix[:, [34, 38, 42]] >> 4)) - 
                                 ((gpsmatrix[:, [37, 41, 45]]>=128)*(1<<28))).transpose()/10000

                # Replace old data with new data from the start of the array
                gps_data["lat"][:num_RV], gps_pos_lon[:num_RV],  gps_data["alt"][:num_RV] = ecef2geodetic(*gps_pos_ecef)

                gps_data["lat"] = np.roll(gps_data["lat"], -num_RV)
                gps_pos_lon = np.roll(gps_pos_lon, -num_RV)
                # gps_vel_east, gps_vel_north, gps_vel_up = ecef2enuv(*gps_vel_ecef, gps_pos_lat, gps_pos_lon)
                #self.gps_num_sat = gpsmatrix[:, 16] & 0b00011111 # 0001-1111 -> take 5 digits
                ##self.gps_num_sat = np.roll(self.gps_num_sat, -num_RV)
                ## Set the gps values to the values in the last rv packet
                #self.gpsValues[0].setText(f"{round(gps_pos_lat[-1], DEC_PLACES)}")
                #self.gpsValues[1].setText(f"{round(gps_pos_lon[-1], DEC_PLACES)}")
                #self.gpsValues[2].setText(f"{round(gps_pos_alt[-1], DEC_PLACES)}") 
                #self.gpsValues[3].setText(f"{round(gps_vel_east[-1], DEC_PLACES)}")
                #self.gpsValues[4].setText(f"{round(gps_vel_north[-1], DEC_PLACES)}")
                #self.gpsValues[5].setText(f"{round(gps_vel_up[-1], DEC_PLACES)}")
                
                #self.gpsValues[7].setText(f"{self.gps_num_sat[-1]}")

            # check if plt_hertz time has elapsed, set do_update to true 
            for d, minframes in zip(data_channels.values(), protocol_minframes):
                for i in d:
                    i.new_data(minframes)
            
            for gps_markers in gps2d_points:
                print(gps_data["lat"], gps_pos_lon)
                gps_markers.set_data(pos=np.transpose(np.array([gps_pos_lon, gps_data["lat"]])) ,face_color="#ff0000", edge_width=0, size=3, symbol='s')

            #for gps_markers in gps3d_oints:
            #    gps_markers.set_data(pos=np.transpose(np.array([gps_pos_lat, gps_pos_lon, gps_pos_alt])) ,face_color="#ff0000", edge_width=0, size=3, symbol='s')

            app.process_events()

            pause_time = max((1/plot_hertz) - (time.perf_counter()-start_time), 0) 
            # Change pause time with threading?
            time.sleep(pause_time)
            start_time = time.perf_counter()

'''
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
        self.gpsValues = []
        
        for ind, name in enumerate(gpsNames):
            
            gpsLabel = QLabel(name)
            gpsValue = QLineEdit()

            gpsValue.setFixedWidth(50)
            gpsValue.setReadOnly(True)
            
            gpsLayout.addWidget(gpsLabel, ind, 0)
            gpsLayout.addWidget(gpsValue, ind, 1)
            
            self.gpsValues.append(gpsValue)
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
    
    def on_close():
        print("HIT")

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

        self.fig.show()
        # self.widget_layout.addWidget(self.fig.native)
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

        self.gps_num_sat = np.zeros(25000, int)
        
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
        map2d = scene.visuals.image(mapdata, method='subdivide', parent=self.gpsax2d.plot_view.scene)
        img_width, img_height = lonlim[1]-lonlim[0], latlim[1]-latlim[0]
        transform2d = sttransform(scale=(img_width/mapdata.shape[1], img_height/mapdata.shape[0]), translate=(lonlim[0], latlim[0]))
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
            

            # Gps bytes are at 6, 26, 46, 66 when the next byte == 128
            gps_raw_data = all_minframes[:, [6, 26, 46, 66]].flatten()
            gps_check = all_minframes[:, [7, 27, 47, 67]].flatten()
            gps_data = gps_raw_data[np.where(gps_check==128)]

            # Parse gps data when there are bytes available
            if len(gps_data) != 0:
                gps_inds = find_RV(gps_data)

                if len(gps_data) - gps_inds[-1] < RV_LEN:
                    gps_inds = gps_inds[:-1]

                gpsmatrix = gps_data[np.add.outer(gps_inds, np.arange(48))].astype(np.uint64)

                # Number of rv packets
                num_RV = np.shape(gpsmatrix)[0]
                # Note skipped check sum

                # Signed position data
                gps_pos_ecef = (((gpsmatrix[:, [12, 20, 28]] << 32) |
                                (gpsmatrix[:, [11, 19, 27]] << 24) |
                                (gpsmatrix[:, [10, 18, 26]] << 16) |
                                (gpsmatrix[:, [ 9, 17, 25]] <<  8) |
                                (gpsmatrix[:, [16, 24, 32]])) - 
                                ((gpsmatrix[:, [12, 20, 28]]>=128)*(1<<40))).transpose()/10000
                
                gps_vel_ecef = (((gpsmatrix[:, [37, 41, 45]] << 20) |
                                 (gpsmatrix[:, [36, 40, 44]] << 12) |
                                 (gpsmatrix[:, [35, 39, 43]] << 4) |
                                 (gpsmatrix[:, [34, 38, 42]] >> 4)) - 
                                 ((gpsmatrix[:, [37, 41, 45]]>=128)*(1<<28))).transpose()/10000

                # Replace old data with new data from the start of the array
                gps_pos_lat, gps_pos_lon,  gps_pos_alt = ecef2geodetic(*gps_pos_ecef)
        
                gps_vel_east, gps_vel_north, gps_vel_up = ecef2enuv(*gps_vel_ecef, gps_pos_lat, gps_pos_lon)
                self.gps_num_sat = gpsmatrix[:, 16] & 0b00011111 # 0001-1111 -> take 5 digits
                #self.gps_num_sat = np.roll(self.gps_num_sat, -num_RV)
                # Set the gps values to the values in the last rv packet
                self.gpsValues[0].setText(f"{round(gps_pos_lat[-1], DEC_PLACES)}")
                self.gpsValues[1].setText(f"{round(gps_pos_lon[-1], DEC_PLACES)}")
                self.gpsValues[2].setText(f"{round(gps_pos_alt[-1], DEC_PLACES)}") 
                self.gpsValues[3].setText(f"{round(gps_vel_east[-1], DEC_PLACES)}")
                self.gpsValues[4].setText(f"{round(gps_vel_north[-1], DEC_PLACES)}")
                self.gpsValues[5].setText(f"{round(gps_vel_up[-1], DEC_PLACES)}")
                
                self.gpsValues[7].setText(f"{self.gps_num_sat[-1]}")

            # check if plt_hertz time has elapsed, set do_update to true 
            for chs, hks, minframes in zip(self.channels, self.housekeeping, protocol_minframes):
                for ch in chs:
                    ch.new_data(minframes)

                for hk in hks:
                    hk.new_data(minframes)

            self.gps_points.set_data(pos=np.transpose(np.array([gps_pos_lat, gps_pos_lon])) ,face_color="#ff0000", edge_width=0, size=3, symbol='s')
            app.process_events()

            pause_time = max((1/plt_hertz) - (time.perf_counter()-start_time), 0) 
            # Change pause time with threading?
            time.sleep(pause_time)
            start_time = time.perf_counter()
        print("Parsing Completed")
        #print(f"Done : {time.perf_counter()-start_time}")
''' 