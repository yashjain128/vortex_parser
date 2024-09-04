"""
Module to handle incoming data

Written for the Space and Atmospheric Instrumentation Laboratory at ERAU
by Yash Jain
"""
from os.path import dirname, abspath

import time                           
from datetime import datetime

import numpy as np                                # Vectorization with numpy arrays
from math import log2                             # Parsing byte data          
from openpyxl import load_workbook                # Reading excel format files
from pymap3d.ecef import ecef2geodetic, ecef2enuv # For coordinates
import socket                                     # Recieving data with socket

# SYNC frames to identify minor frames
# All minor frames end in SYNC
SYNC = [64, 40, 107, 254]
MINFRAME_LEN = 2 * 40
PACKET_LENGTH = MINFRAME_LEN + 44
bytes_ps = PACKET_LENGTH * 5000 # bytes_ps will be manually set in the excel format
# Types of frames
PROTOCOLS = ['all', 'odd frame', 'even frame', 'odd sfid', 'even sfid']

data_channels = {} # Sort channels and hk by protocol
all_data = {}

MAX_NUMPOINTS = 50000

# GPS label names
GPS_NAMES_ID = ["lon", "lat", "alt", "veast", "vnorth", "vup", "shorz", "numsats"]
GPS_NAMES = ["Longitude (deg)", "Latitude (deg)", "Altitude (km)", "vEast (m/s)", "vNorth (m/s)", "vUp (m/s)", "Horz. Speed (m/s)", "Num Sats"]
# Identify gps data in RV frames
RV_HEADER = [114, 86, 48, 50, 65]
RV_LEN = 48
gps_data = None

# Housekeeping coefficients and constants for converting from counts to units
hkunits = True # When true counts will be converted to units
HK_LENGTH = 11
HK_NAMES =          ["Temp1" , "Temp2" , "Temp3" , "Int. Temp", "V Bat", "-12 V", "+12 V", "+5 V", "+3.3 V", "VBat Mon"]
HK_COEF  = np.array([-76.9231, -76.9231, -76.9231, -76.9231   , 16     , 6.15   , 7.329  , 3     ,  2      , 2         ], dtype=np.float64)[:, None]
HK_ADD   = np.array([202.54  , 202.54  , 202.54  , 202.54     , 0      , -16.88 , 0      , 0     ,  0      , 0         ], dtype=np.float64)[:, None]
# Housekeeping display constants
DEC_PLACES = 3     # Decimals of precision for housekeeping 
AVG_NUMPOINTS = 10 # Number of housekeeping points to average
# ACC's dig temp is a hardcoded housekeeping value
acc_dig_temp = None
acc_dig_temp_data = np.zeros(25000, np.uint32)

# When read mode is 0 then a read_file is the read
# When read mode is 1 then the socket is used
read_mode = 0

# Socket variables
sock = None
sock_timeout = 5.0 # How long to wait for data before ending. 
sock_wait = 0.1
sock_rep = int(sock_timeout/sock_wait)

read_file = None

# Write file
write_mode = False
write_file = None
raw_data = None
last_ind_arr = np.empty(0)

# Plot rate settings
plot_hertz = 5
plot_width = 5

# Excel Sheet
xl_sheet = None
# The type of values in the excel sheet
GRAPH_ROW_TYPE =   [str, str, int,  int,  str, str, int]
CHANNEL_ROW_TYPE = [str, str, bool, list, list]
MAP_ROW_TYPE =     [str, str, int,  int,  str]
HK_ROW_TYPE =      [str, int, str,  int,  list, list, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool]

def getval(cell, t):
    '''
    Read a cell with
    '''
    if xl_sheet==None:
        return None
    val = xl_sheet[cell].value
    if t==int:
        return val
    elif t==str:
        return val
    elif t==bool:
        return val
    elif t==list:
        return [int(i) for i in str(val).split(';')]
    else:
        return val
        

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

def add_channel(graph_name, protocol, signed, byte_ind, bitmask):
    # Graph must fit the channel data
    data_channels[graph_name] = Channel(protocol, signed, byte_ind, bitmask)
    all_data[name] = None

def add_housekeeping(name, numpoints, protocol, board_id, byte_ind, bitmask, *hkvalues):
    data_channels[name] = Housekeeping(board_id,  protocol, numpoints, byte_ind, bitmask)
    all_data[name] = None



def crc_16(arr):
    """
    Checksum for gps data
    """
    crc = 0
    for i in arr:
        crc ^= (i<<8)
        for i in range(0,8):
            if (crc&0x8000)>0:
                crc <<= 1
                crc ^= 0x1021
            else:
                crc <<= 1
        crc &= (1<<16)-1
    return crc
    
def init(format_file, read_mode=1, udp_ip="127.0.0.1", udp_port="5000", read_file_name="", do_write=0, write_file_name="", do_hkunits=1, hertz=5, width=5) -> None:
    '''
    Setup for parsing by specifying how the data

    format_file               The path as a string to a valid excel format file
    read_mode  =1             Set to 1 for parsing udp data and 0 for read from read_file
    udp_ip     ="127.0.0.1"   The socket ip address [read_mode=1]
    udp_port   ="5000"        The socket port [read_mode=1]
    read_file  =""            The path as a string to a recording file [read_mode=0]
    write_mode =0             Set to 1 to write to write_file
    write_file =""            The name as a string for the file to write to. Will default to a name with todays date.
    hk_units   =1             Set to 1 to have housekeeping data in units, 0 for counts
    hertz      =5             parse() will read (1/plot_hertz) seconds of data
    width      =5             The amount of seconds of data to store 
    '''
    global sock, read_length, read_file, plot_width, plot_hertz, raw_data, gps_data, write_mode, write_file, hk_units, xl_sheet

    plot_hertz = hertz
    plot_width = width
    
    # Initialize a write file
    dir = dirname(dirname(abspath(__file__)))
    write_mode = do_write
    if write_mode:
        if (write_file_name==""):
            write_file = "Recording"+datetime.today().strftime('%Y-%m-%d')
        write_file = open(dir+"/recordings/"+write_file_name+".udp", "ab")

    # Load the excel format file    
    xl_sheet = load_workbook(format_file, data_only=True).active

    # Bytes/second
    bytes_ps = getval("D3", int)
    read_length = bytes_ps//plot_hertz
    read_length += 126 - (read_length%126)

    # Set up gps data dictionary
    #gps_data = {gps_name:np.zeros([getval("D4", int)*width], float) for gps_name in GPS_NAMES_ID}

    # Set hk units
    hk_units = do_hkunits

    '''
    # Plots
    graph_rows = getval("D7", list)
    for row_num in range(graph_rows[0], graph_rows[1]+1):
        row = [getval("B"+str(row_num))]
        i = 1
        while row[-1] != "None":
            row.append( getval(chr(ord("B")+i) + str(row_num)) )
            i += 1
        # Remove last cell "None"
        row = row[:-1]
        
        if len(row) == 0:
            continue
        elif row[0][0] == '#':
            add_channel(*row)
    '''
    '''
    graph_rows = getval("D6", list)
    for row_num in range(graph_rows[0],graph_rows[1]+1):
        name = getval("D"+str(row_num), str)
        numpoints = getval("I"+str(row_num), str)
        graph_numpoints[name] = numpoints
    '''

    # Channels
    channel_rows = getval("D7", list)
    for row_num in range(channel_rows[0],channel_rows[1]+1):
        row = [getval(chr(i)+str(row_num),t) for i, t in zip(range(ord('C'),ord('C')+len(CHANNEL_ROW_TYPE)), CHANNEL_ROW_TYPE)]
        add_channel(*row)

    # Housekeeping
    hk_rows = getval("D9", list)
    for row_num in range(hk_rows[0], hk_rows[1]+1):
        row = [getval(chr(i)+str(row_num),t) for i, t in zip(range(ord('C'), ord('C')+len(HK_ROW_TYPE)), HK_ROW_TYPE)]
        add_housekeeping(*row)
    '''
        if (row[0]=="ACC"):
            add_housekeeping(*row[1:7])
        else:
            add_housekeeping(*row[1:7])
    '''
            
    '''
    self.valuesWidget.show()
    plotting.finish_creating()
    ''' 

    raw_data = np.zeros(bytes_ps)
    if read_mode == 0:
        print("Opening recording")
        read_file = open(read_file_name, "rb")

    elif read_mode == 1:
        print("Connecting Socket...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
        sock.bind((udp_ip, udp_port))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,bytes_ps) # Set the socket max read buffer so data doesn't overflow.
        print(f"Socket connected\nIP: {udp_ip}\nPort: {udp_port}")    
        sock.setblocking(0)




def parse() -> dict:
    '''
    Will parse the required number of bytes according to specifications in init() and returns data as dictionary

    Here is what the dictionary will look like
    dict = {
            "PIP1" : [1, 2, 3 ...]  # Plotted Channels
            "PIP2" : [3, 5, 7 ...]
            "ACC1" : [2, 4, 6 ...]
            ...
            "GPS_Long" : [4, 5, 6 ...] # GPS values
            "GPS_Lat" : [4, 9, 16 ...]
            ...
            "HKPIP" : [34, 1, 2 ...] # Housekeeping values
            "HKmNLP" : [18, 1, 2 ...]
            ...
            }
        
    '''
    global running, raw_data, last_ind_arr
    running = True
    cur_length = 0
    if read_mode == 1:
        cur_length = min(read_length, read_file.tell())
        raw_data[:cur_length] = np.fromfile(read_file, dtype=np.uint8, count=read_length)
        if len(raw_data) == 0:
            print("Finished reading file")
            running = False
            return

    else:
        cur_length = 0
        running = True
        while (cur_length<read_length and running):
            try:
                raw_data[cur_length:cur_length+126] = np.frombuffer(sock.recv(126), np.uint8)
                cur_length+=126
            except BlockingIOError:
                # Only process events again if it has been a tenth of a second
                time.sleep(0.01)

            except WindowsError:
                print("Avoided socket error")
                cur_length= 0

        if (not running):
            return

        '''
    if (next_process_events==0):
        draw_start_time = time.perf_counter()
        app.process_events()
        draw_time += time.perf_counter()-draw_start_time
    next_process_events = 0 
        '''

    if write_mode:
        raw_data[:cur_length].tofile(write_file)
    # Add the remaining bytes from the p
    data_arr = np.concatenate([last_ind_arr, raw_data[:cur_length]])
    # Must process the gui so it does not freeze


    calc_start_time = time.perf_counter()
    
    inds = find_SYNC(data_arr)
    if len(inds)==0:
        print("No valid sync frames")
        return

    # Save last index for next cycle
    last_ind_arr = data_arr[inds[-1]:]

    # Check for all indexes if the length between them is correct
    inds = inds[:-1][(np.diff(inds) == PACKET_LENGTH)]

    #
    inds = inds[:-1][(np.diff(data_arr[inds + 6]) != 0)]

    all_minframes = data_arr[inds[:, None] + e].astype(int)

    # Frame types
    protocol_minframes = [all_minframes,
        all_minframes[np.where(all_minframes[:, 57] & 3 == 1)],
        all_minframes[np.where(all_minframes[:, 57] & 3 == 2)],
        all_minframes[np.where(all_minframes[:, 5] % 2 == 1)],
        all_minframes[np.where(all_minframes[:, 5] % 2 == 0)]]
    
    # Gps bytes are at 6, 26, 46, 66 when the next byte == 128
    gps_raw_data = all_minframes[:, [6, 26, 46, 66]].flatten()
    gps_check = all_minframes[:, [7, 27, 47, 67]].flatten()
    gps_data_d = gps_raw_data[np.where(gps_check==128)]
    
    # Gps indices
    gps_inds = np.array([])
    if len(gps_data_d)>0:
        gps_inds = find_RV(gps_data_d)
    
    # Check if last index is inside the gps stream
    if len(gps_inds)>0 and gps_inds[-1]+RV_LEN>len(gps_data_d):
        gps_inds = gps_inds[:-1]

    # Parse gps data when there are bytes available
    if len(gps_inds)>0:
        if len(gps_data_d) - gps_inds[-1] < RV_LEN:
            gps_inds = gps_inds[:-1]

        gpsmatrix = gps_data_d[np.add.outer(gps_inds, np.arange(48))].astype(np.uint32)

        # Number of rv packets
        num_RV = np.shape(gpsmatrix)[0]

        # All rv_packets whose checksum is equal to the last 2 bytes
        valid_rv_packets = np.zeros(num_RV, dtype=bool)
        for i in range(num_RV):
            valid_rv_packets[i] = crc_16(gpsmatrix[i,:-3]) == (gpsmatrix[i, -2]<<8) | gpsmatrix[i, -3]
        gpsmatrix = gpsmatrix[np.where(valid_rv_packets)].astype(np.uint64)

        # Get num_rv after eliminating frames that did not pass the checksum
        num_RV = np.shape(gpsmatrix)[0]

        # Signed position data
        gps_pos_ecef = (((gpsmatrix[:, [12, 20, 28]] << 32) |
                        (gpsmatrix[:, [11, 19, 27]] << 24) |
                        (gpsmatrix[:, [10, 18, 26]] << 16) |
                        (gpsmatrix[:, [ 9, 17, 25]] <<  8) |
                        (gpsmatrix[:, [16, 24, 32]])) - 
                        ((gpsmatrix[:, [12, 20, 28]]>=128)*(1<<40))).transpose()/10000
        
        gps_vel_ecef = (((gpsmatrix[:, [36, 40, 44]] << 20) |
                            (gpsmatrix[:, [35, 39, 43]] << 12) |
                            (gpsmatrix[:, [34, 38, 42]] << 4)  |
                            (gpsmatrix[:, [33, 37, 41]] >> 4)) -
                            ((gpsmatrix[:, [36, 40, 44]]>=128)*(1<<28))).transpose()/10000


        # Replace old data with new data from the start of the array
        gps_data["lat"][:num_RV], gps_data["lon"][:num_RV],  gps_data["alt"][:num_RV] = ecef2geodetic(*gps_pos_ecef) # Use ecef2geodetic to get position in lat, lon, alt

        gps_data["veast"][:num_RV], gps_data["vnorth"][:num_RV], gps_data["vup"][:num_RV] = ecef2enuv(*gps_vel_ecef, gps_data["lat"][:num_RV], gps_data["lon"][:num_RV]) # Use ecef2enuv to get velocity in east, north, up 

        gps_data["shorz"][:num_RV] = np.hypot(gps_data["veast"][:num_RV], gps_data["vnorth"][:num_RV]) # Get horizontal speed from the hypotonuse of east and north velocity

        gps_data["numsats"][:num_RV] = gpsmatrix[:, 15] & 0b00011111 # 0001-1111 -> take 5 digits
        
        # Shift data to the left by num_RV and set the last parsed value as text
        for val in GPS_NAMES_ID:
            gps_data[val] = np.roll(gps_data[val], -num_RV)
        
        '''
            gps_values[val].setText(f"{gps_data[val][-1] : .{DEC_PLACES}f}") #.rstrip('0') to remove zeros


        for gps_markers in gps2d_points:
            gps_markers.set_data(pos=np.transpose(np.array([gps_data["lon"], gps_data["lat"]])) ,face_color="#ff0000", edge_width=0, size=3, symbol='s')
        lon3d = (gps_data["lon"]-lonlim[0])/(lonlim[1]-lonlim[0])
        lat3d = (gps_data["lat"]-latlim[0])/(latlim[1]-latlim[0])
        alt3d = (gps_data["alt"]-altlim[0])/(altlim[1]-altlim[0])
        for gps_markers in gps3d_points:
            gps_markers.set_data(pos=np.transpose(np.array([lon3d, lat3d, alt3d])) ,face_color="#ff0000", edge_width=0, size=3, symbol='s')
        '''


    for name in data_channels.keys():
        dch = data_channels[name]
        all_data[name] = dch.new_data(protocol_minframes[dch.frame_ind])


    # Update digital accelerometer temperature
    '''
    acc_dig_temp_data[:len(protocol_minframes[2])] = ((protocol_minframes[2][:, 61]&15)<<8 | protocol_minframes[2][:, 62]).transpose()
    acc_dig_temp_data = np.roll(acc_dig_temp_data, -len(protocol_minframes[2]))
    
    if acc_dig_temp != None:
        acc_dig_temp.setText(f"{acc_dig_temp_data[-1]: .{DEC_PLACES}f}")
    '''

    #calc_time += time.perf_counter()-calc_start_time

    # Pause when reading a file
    return all_data
    '''
    if (read_mode == 1):
        pause_time = max((1/plot_hertz) - (time.perf_counter()-start_time), 0) 
        time.sleep(pause_time)
        start_time = time.perf_counter()
    '''

class Channel:
    def __init__(self, protocol, signed, byte_ind, bitmask):
        self.frame_ind = PROTOCOLS.index(protocol)
        self.signed = signed
        bit_num = 0
        self.byte_info = []

        for ind, mask in zip(byte_ind, bitmask): 
            # index and mask are given
            # infer bit shift from first bit in mask and how many bits have passed
            shift = int(bit_num - log2(mask & -mask))
            self.byte_info.append([ind, mask, shift])            
            bit_num += mask.bit_count()

        # self.xlims = [0, numpoints]
        # Infer y limits from number of bits
        if self.signed:
            self.ylims = [-2**(bit_num-1), 2**(bit_num-1)]
        else:
            self.ylims = [0, 2**bit_num]

        self.n = 0
        self.data = np.zeros(MAX_NUMPOINTS)
        #self.datax = np.arange(MAX_NUMPOINTS)

    def new_data(self, minframes):
        self.n = len(minframes)
        self.data[:self.n] = np.zeros(self.n)
        for ind, mask, shift in self.byte_info:
            if shift < 0:
                self.data[:self.n] += (minframes[:, ind] & mask) >> abs(shift)
            else:
                self.data[:self.n] += (minframes[:, ind] & mask) << shift
        
        if self.signed:
            self.data[:self.n] = self.datay[:self.n]+(self.datay[:self.n] >= self.ylims[1])*(2*self.ylims[0])

        #sself.datay = np.roll(self.datay, -l)

        #data = np.transpose(np.array([self.datax, self.datay]))
        #self.line.set_data(pos=data, edge_width=0, size=1, face_color=self.color)
        return self.data[:self.n]

    def reset(self):
        self.data = np.zeros(self.xlims[1])
        #self.line.set_data(pos=np.transpose(np.array([self.datax, self.datay])), edge_width=0, size=1, face_color=self.color)

class Housekeeping:
    def __init__(self, protocol, board_id, numpoints, b_ind, b_mask):
        self.frame_ind = PROTOCOLS.index(protocol)
        self.b_ind, self.b_mask = b_ind, b_mask 
        print(self.b_ind, self.b_mask)
        self.rate = 0
        for mask in b_mask:
            self.rate += mask.bit_count()


        print(self.rate)
        self.ipf = len(self.b_ind) # indexes per frame
        if self.rate == 8:
            self.board_id = board_id
            self.length = HK_LENGTH
        elif self.rate == 4:
            self.board_id = [board_id>>4, board_id&0xF]
            self.length = HK_LENGTH*2
        else:
            raise ValueError("Unsupported housekeeping rate")

        self.numpoints = int(numpoints*self.rate)

        self.indcol = np.array(np.arange(10)//self.rate, dtype=np.uint8)[:, None]+1
        self.data = np.zeros((10, self.numpoints))
        self.maxhkrange = AVG_NUMPOINTS

    def new_data(self, minframes):
        minframes = minframes.astype(np.uint8)
        self.n = len(minframes)
        databuffer = np.zeros(self.n*self.ipf, dtype=np.uint8)
        for i in range(self.ipf):
            databuffer[np.arange(self.n)*self.ipf+i] = minframes[:, self.b_ind[i]] & self.b_mask[i]
        if self.rate == 8: # ACC, mNLP, PIP
            inds = np.where(databuffer == self.board_id)[0]
            inds = inds[np.where(np.diff(inds) == self.length)[0]]
            if inds.size != 0:
                self.data[:, :inds.size] = databuffer[self.indcol + inds]
                

        elif self.rate == 4: # EFP
            inds = np.where( (databuffer==self.board_id[0])[:-1] & (databuffer==self.board_id[1])[1:])[0]
            inds = inds[np.where(np.diff(inds) == self.length)[0]][:-1]
            self.data[:, :inds.size] = databuffer[self.indcol + inds]<<4 | databuffer[self.indcol+1 + inds]

        if hkunits:
            self.data[:, :inds.size] = HK_COEF * (self.data[:, :inds.size]*2.5/256 - 0.5*2.5/256) + HK_ADD
         

        #self.data = np.roll(self.data, -inds.size, axis=1)
        
        '''
        hkrange = min(self.maxhkrange, inds.size)
    
        for edit, data_row in zip(self.values, self.data):
            if edit.isEnabled():
                if hkrange==0:
                    edit.setText("null")
                else:
                    edit.setText(f"{np.average(data_row[-hkrange:]): .{DEC_PLACES}f}")
        '''

    def reset(self):
        self.data = np.zeros((10, self.numpoints))
        for value in self.values:
            value.setText("")

if __name__ == "__main__":
    init(format_file="C://Users//ayash//Programming//vortex_parser//testing//mm.xlsx",
         udp_ip="127.0.0.1",
         udp_port=5000)

    parse()
