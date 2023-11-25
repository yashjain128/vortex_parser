"""
Module to handle plotting, housekeeping, and GPS

Written for the Space and Atmospheric Instrumentation Laboratory
by Yash Jain
"""
import time, math
import socket

import numpy as np

from vispy import scene, plot, app
from vispy.visuals.transforms import STTransform, MatrixTransform

from pymap3d.ecef import ecef2geodetic, ecef2enuv

from scipy.io import loadmat

SYNC = [64, 40, 107, 254]
MINFRAME_LEN = 2 * 40
PACKET_LENGTH = MINFRAME_LEN + 44  
MAX_READ_LENGTH = PACKET_LENGTH * 5000

RV_HEADER = [114, 86, 48, 50, 65]
RV_LEN = 48

# how many decimal places to round gps data
DEC_PLACES = 3
AVG_NUMPOINTS = 10

plot_width = 5
do_write = False
write_file = None

HK_NAMES = ["Temp1", "Temp2", "Temp3", "Int. Temp", "V Bat", "-12 V", "+12 V", "+5 V", "+3.3", "VBat Mon", "Dig. Temp"]
GPS_NAMES = ["Longitude (deg)", "Latitude (deg)", "Altitude (km)", "vEast (m/s)", "vNorth (m/s)", "vUp (m/s)", "Horz. Speed (m/s)", "Num Sats"]
GPS_NAMES_ID = ["lon", "lat", "alt", "veast", "vnorth", "vup", "shorz", "numsats"]

# Housekeeping coefficients and constants for units
do_hkunits = True
HK_COEF = np.array([1, -76.9231, -76.9231, -76.9231, -76.9231, 16, 6.15, 7.329, 3, 2, 2], dtype=np.float64) [:, None]
HK_ADD = np.array([0, 202.54, 202.54, 202.54, 202.54, 0, -16.88, 0, 0, 0, 0], dtype=np.float64) [:, None]

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
gps_values = {}

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

def set_hkunits(hkunits):
    global do_hkunits
    do_hkunits = hkunits

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
    
    [gps_data_arr.fill(0) for gps_data_arr in gps_data.values()]
    
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
    row = int(row)
    col = int(col)
    fig = get_fig(figure)
    if type=="2d":
        fig[row, col].configure2d(title=name, xlabel="Longitude", ylabel="Latitude")

        gps2d_points.append(
            scene.Markers(pos=np.transpose(np.array([gps_data["lat"], gps_data["lon"]])), face_color="#ff0000", edge_width=0, size=5, parent=fig[int(row), int(col)].plot_view.scene, antialias=False, symbol='s')
        )
    elif type=="3d":
        # axis labels wont work yet
        fig[row, col].configure3d(title=name, xlabel="Longitude", ylabel="Latitude", zlabel="Altitude")    
        fig[row, col].ylims = [0, 100] 

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
    
    for _gps in gps_data.values():
        _gps.fill(0)
    

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
            ############## Read Data #################
            if read_mode == 0:
                raw_data = np.fromfile(read_file, dtype=np.uint8, count=read_length)

            elif read_mode == 1:
                soc_data = sock.recv(1024)
                raw_data = np.frombuffer(soc_data, dtype=np.uint8)
                
            if len(raw_data) == 0:
                break
            
            if do_write:
                raw_data.tofile(write_file)

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
            
            ############## GPS #################
            # Gps bytes are at 6, 26, 46, 66 when the next byte == 128
            gps_raw_data = all_minframes[:, [6, 26, 46, 66]].flatten()
            gps_check = all_minframes[:, [7, 27, 47, 67]].flatten()
            gps_data_d = gps_raw_data[np.where(gps_check==128)]

            # Parse gps data when there are bytes available
            if len(gps_data_d) != 0:
                gps_inds = find_RV(gps_data_d)

                if len(gps_data_d) - gps_inds[-1] < RV_LEN:
                    gps_inds = gps_inds[:-1]

                gpsmatrix = gps_data_d[np.add.outer(gps_inds, np.arange(48))].astype(np.uint64)

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
                gps_data["lat"][:num_RV], gps_data["lon"][:num_RV],  gps_data["alt"][:num_RV] = ecef2geodetic(*gps_pos_ecef) # Use ecef2geodetic to get position in lat, lon, alt
                gps_data["veast"][:num_RV], gps_data["vnorth"][:num_RV], gps_data["vup"][:num_RV] = ecef2enuv(*gps_vel_ecef, gps_data["lat"][:num_RV], gps_data["lon"][:num_RV]) # Use ecef2enuv to get velocity in east, north, up 
                gps_data["shorz"][:num_RV] = np.hypot(gps_data["veast"][:num_RV], gps_data["vnorth"][:num_RV]) # Get horizontal speed from the hypotonuse of east and north velocity
                gps_data["numsats"][:num_RV] = gpsmatrix[:, 16] & 0b00011111 # 0001-1111 -> take 5 digits
                
                # Shift data to the left by num_RV and set the last parsed value as text
                for val in GPS_NAMES_ID:
                    gps_data[val] = np.roll(gps_data[val], -num_RV)
                    gps_values[val].setText(f"{gps_data[val][-1] : .{DEC_PLACES}f}")


                for gps_markers in gps2d_points:
                    gps_markers.set_data(pos=np.transpose(np.array([gps_data["lon"], gps_data["lat"]])) ,face_color="#ff0000", edge_width=0, size=3, symbol='s')

            # Housekeeping

            # check if plt_hertz time has elapsed, set do_update to true 
            for d, minframes in zip(data_channels.values(), protocol_minframes):
                for i in d:
                    i.new_data(minframes)

            #for gps_markers in gps3d_oints:
            #    gps_markers.set_data(pos=np.transpose(np.array([gps_pos_lat, gps_pos_lon, gps_pos_alt])) ,face_color="#ff0000", edge_width=0, size=3, symbol='s')

            app.process_events()

            # do_hk counts here
            if do_hkunits:
                pass
                
            pause_time = max((1/plot_hertz) - (time.perf_counter()-start_time), 0) 
            # Change pause time with threading?
            time.sleep(pause_time)
            start_time = time.perf_counter()

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

        self.b_ind, self.b_mask = b_ind, b_mask 
        self.rate = self.b_mask.bit_count()/8

        if self.rate == 8/8: 
            self.board_id = board_id
        elif self.rate == 4/8:
            self.board_id = [board_id>>4, board_id&0xF]
        else:
            raise ValueError("Unsupported housekeeping rate")
        self.numpoints = int(numpoints*self.rate)
        self.length = 11//self.rate

        self.indcol = np.array(np.arange(11)//self.rate, dtype=np.uint8)[:, None]
        self.data = np.zeros((11, self.numpoints))
        self.values = values
        self.maxhkrange = AVG_NUMPOINTS

    def new_data(self, minframes):
        databuffer = minframes[:, self.b_ind] & self.b_mask
        if self.rate == 8/8:
            inds = np.where(databuffer == self.board_id)[0]
            inds = inds[np.where(np.diff(inds) == self.length)[0]]
            if inds.size != 0:
                self.data[:, :inds.size] = databuffer[self.indcol + inds]

        elif self.rate == 4/8:
            inds = np.where( (databuffer==self.board_id[0])[:-1] & (databuffer==self.board_id[1])[1:])[0]
            inds = inds[np.where(np.diff(inds) == self.length)[0]][:-1]
            self.data[:, :inds.size] = databuffer[self.indcol + inds]<<4 | databuffer[self.indcol+1 + inds]

        if do_hkunits:
            #print(do_hkunits)
            self.data[:, :inds.size] = HK_COEF * (self.data[:, :inds.size]*2.5/256 - 0.5*2.5/256) + HK_ADD
        
        self.data = np.roll(self.data, -inds.size, axis=1)
        hkrange = min(self.maxhkrange, inds.size)



        for edit, data_row in zip(self.values, self.data):
            if edit.isEnabled():
                if hkrange==0:
                    edit.setText("null")
                else:
                    edit.setText(f"{np.average(data_row[-hkrange:]): .{DEC_PLACES}f}")

    def reset(self):
        self.data = np.zeros((11, self.numpoints))
        for value in self.values:
            value.setText("")

class ScrollingPlotWidget(scene.Widget):
    """
    Widget for 2d and 3d plots built on top of scene.Widget.

    To use this with Fig changing  the default class is necessary
    """
    def __init__(self, *args, **kwargs):
        self.grid = None
        self.plot_grid = None
        self.plot_view = None
        self.camera = None
        self.title = None
        self.title_widget = None
        self.yaxis, self.xaxis, self.zaxis = None, None, None
        self.ylabel, self.xlabel, self.zlabel = None, None, None
        self.ylims, self.xlims, self.zlims = None, None, None
        self.dimensions = None
        self.data = None
        self.view_grid = None
        self._configured = False
        self.visuals = []
        self.section_y_x = None

        super(ScrollingPlotWidget, self).__init__(*args, **kwargs)
        self.grid = self.add_grid(spacing=0, margin=10)
        

    def configure2d(self, title, xlabel, ylabel, xlims=[0, 1], ylims=[-1, 1]):

        fg = "#000000"
        self.xlims = xlims[:]
        self.ylims = ylims[:]

        self.dimensions = 2

        # padding left
        padding_left = self.grid.add_widget(None, row=0, row_span=3, col=0)
        padding_left.width_min = 5
        padding_left.width_max = 10

        # padding right
        padding_right = self.grid.add_widget(None, row=0, row_span=3, col=3)
        padding_right.width_min = 15
        padding_right.width_max = 25

        # padding down
        padding_bottom = self.grid.add_widget(None, row=3, col=0, col_span=4)
        padding_bottom.height_min = 15
        padding_bottom.height_max = 15

        # row 0
        # title - column 4 to 5
        self.title = scene.Label(title, font_size=8, color="#000000")
        self.title_widget = self.grid.add_widget(self.title, row=0, col=2)
        self.title_widget.height_min = 35
        self.title_widget.height_max = 35

        # row 1
        # yaxis - column 1
        # view - column 2

        self.yaxis = scene.AxisWidget(orientation='left',
                                      text_color=fg,
                                      axis_color=fg,
                                      tick_color=fg,
                                      axis_width=2,
                                      minor_tick_length=4, 
                                      major_tick_length=6,
                                      tick_font_size=6,
                                      axis_font_size=6,
                                      axis_label=ylabel,
                                      axis_label_margin=14,
                                      tick_label_margin=2)
        yaxis_widget = self.grid.add_widget(self.yaxis, row=1, col=1)
        yaxis_widget.width_min = 20
        yaxis_widget.width_max = 50
        yaxis_widget.height_min = 50

        # xaxis - column 3
        self.xaxis = scene.AxisWidget(orientation='bottom', 
                                      text_color=fg,
                                      axis_color=fg,
                                      axis_label=xlabel,
                                      axis_font_size=6,
                                      axis_label_margin=26,
                                      tick_color=fg,
                                      tick_font_size=6,
                                      tick_label_margin=20,
                                      minor_tick_length=2,
                                      major_tick_length=5)
        xaxis_widget = self.grid.add_widget(self.xaxis, row=2, col=2)
        xaxis_widget.height_max = 30
        xaxis_widget.width_min = 50

        # This needs to be added to the grid last (to fix #1742)
        self.plot_view = self.grid.add_view(row=1, col=2, border_color='grey', bgcolor="#efefef") 
        self.plot_view.camera = 'panzoom'
        self.camera = self.plot_view.camera
        self.camera.set_range(x=self.xlims, y=self.ylims)

        self._configured = True
        self.xaxis.link_view(self.plot_view)
        self.yaxis.link_view(self.plot_view)

        return self

    def configure3d(self, title, xlabel, ylabel, zlabel, xlims=[0, 1], ylims=[0, 1], zlims=[0, 1]):

        self.dimensions = 3
        fg = "#000000"
        self.plot_view = self.grid.add_view(row=1, col_span=3, col=1, border_color='grey', bgcolor="#efefef")
        self.plot_view.camera = 'turntable'
        self.plot_view.camera.center = (0.5, 0.5, 0.5)
        # self.plot_view.camera.fov = 80
        #.center((0.5, 0.5, 0.5))   
        self.camera = self.plot_view.camera
        
        #return
        self.ylims = ylims
        self.xlims = xlims

        # padding left
        padding_left = self.grid.add_widget(None, row=1, row_span=2, col=0)
        padding_left.width_min = 20
        padding_left.width_max = 20

        # padding right
        padding_right = self.grid.add_widget(None, row=1, row_span=2, col=2)
        padding_right.width_min = 20
        padding_right.width_max = 30

        # padding down
        padding_bottom = self.grid.add_widget(None, row=2, col=0, col_span=3)
        padding_bottom.height_min = 20
        padding_bottom.height_max = 20

        # row 0 
        # title - column 4 to 5
        self.title = scene.Label(title, font_size=8, color=fg)
        self.title_widget = self.grid.add_widget(self.title, row=0, col=1, col_span=3)
        self.title_widget.height_min = 30
        self.title_widget.height_max = 30    

        self.xaxis = scene.Axis(pos=[[0, 0], [1, 0]], tick_direction=(0, -1), axis_width=0.1, tick_width=1, domain=xlims,
                                axis_color=fg, tick_color=fg, text_color=fg, font_size=20, parent=self.plot_view.scene)
        self.yaxis = scene.Axis(pos=[[0, 0], [0, 1]], tick_direction=(-1, 0), axis_width=1, tick_width=1, domain=ylims,
                                axis_color=fg, tick_color=fg, text_color=fg, font_size=20, parent=self.plot_view.scene)
        self.zaxis = scene.Axis(pos=[[0, 0], [-1, 0]], tick_direction=(0, -1), axis_width=1, tick_width=1, domain=zlims,
                                axis_color=fg, tick_color=fg, text_color=fg, font_size=20, parent=self.plot_view.scene)
        self.zaxis.transform = scene.transforms.MatrixTransform()  # its acutally an inverted xaxis
        self.zaxis.transform.rotate(90, (0, 1, 0))  # rotate cw around yaxis
        self.zaxis.transform.rotate(-45, (0, 0, 1))  # tick direction towards (-1,-1)
        self._configured = True

        return self
    def add_line(self, line):
        self.plot_view.add(line) 
        
        return line
    def add_gridlines(self):
        self.view_grid = scene.visuals.GridLines(color=(0, 0, 0, 0.5))
        self.view_grid.set_gl_state('translucent')
        self.plot_view.add(self.view_grid)
    
    def reset_bounds(self):
        self.camera.set_range(x=self.xlims, y=self.ylims)