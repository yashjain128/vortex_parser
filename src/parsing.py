"""
Module to handle incoming data

Written for the Space and Atmospheric Instrumentation Laboratory at ERAU
by Yash Jain
"""
import numpy as np
from openpyxl import load_workbook

SYNC = [64, 40, 107, 254]
MINFRAME_LEN = 2 * 40
PACKET_LENGTH = MINFRAME_LEN + 44
bytes_ps = PACKET_LENGTH * 5000

RV_HEADER = [114, 86, 48, 50, 65]
RV_LEN = 48

# how many decimal places to round gps data
DEC_PLACES = 3
AVG_NUMPOINTS = 10


sock = None
# How long to wait for data before ending. 
sock_timeout = 5.0
sock_wait = 0.1
sock_rep = int(sock_timeout/sock_wait)

plot_width = 5
do_write = False
write_file = None

HK_NAMES = ["Temp1", "Temp2", "Temp3", "Int. Temp", "V Bat", "-12 V", "+12 V", "+5 V", "+3.3 V", "VBat Mon"]
GPS_NAMES = ["Longitude (deg)", "Latitude (deg)", "Altitude (km)", "vEast (m/s)", "vNorth (m/s)", "vUp (m/s)", "Horz. Speed (m/s)", "Num Sats"]
GPS_NAMES_ID = ["lon", "lat", "alt", "veast", "vnorth", "vup", "shorz", "numsats"]

acc_dig_temp = None
acc_dig_temp_data = np.zeros(25000, np.uint32)
# Housekeeping coefficients and constants for units
do_hkunits = True
HK_COEF = np.array([-76.9231    , -76.9231, -76.9231, -76.9231, 16, 6.15, 7.329, 3, 2, 2], dtype=np.float64) [:, None]
HK_ADD = np.array([202.54, 202.54, 202.54, 202.54, 0, -16.88, 0, 0, 0, 0], dtype=np.float64) [:, None]

PROTOCOLS = ['all', 'odd frame', 'even frame', 'odd sfid', 'even sfid']

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

def crc_16(arr):
    """
    Check sum
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
    
def init(format_file, read_mode=1, udp_ip="127.0.0.1", udp_port="5000", read_file="", write_mode=0, write_file="", hk_units=1, plot_hertz=5, plot_width=5) -> None:
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
    plot_hertz =5             parse() will read (1/plot_hertz) seconds of data
    plot_width =5             The amount of seconds of data to store 
    '''

    xl_sheet = load_workbook(format_file, data_only=True).active
    getval = lambda c: str(xl_sheet[c].value)

    # Bytes/second
    bps = int(getval("G3"))
    bytes_ps = bps

    # Plots
    graph_row_start, graph_row_end = getval("C3"), getval("D3")
    for row_num in range(int(graph_row_start), int(graph_row_end)+1):
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

    # Housekeeping
    hk_row_start, hk_row_end = getval("C5"), getval("D5")
    for row_num in range(int(hk_row_start), int(hk_row_end)+1):
        row = [getval("B"+str(row_num))]
        i = 1
        while row[-1] != "None":
            row.append( getval(chr(ord("B")+i) + str(row_num)) )
            i += 1

        # Remove last cell which is "None"
        row = row[:-1]

        if len(row) == 0:
            continue
        else:
            if (row[0]=="ACC"):
                hkValues = self.addHousekeeping(row[0], row[7:]+["True"], plotting.HK_NAMES+["Dig Temp"], )
                plotting.set_acc_dig_temp(hkValues[-1])
                plotting.add_housekeeping(*row[1:7], hkValues[:-1])
            else:
                hkValues = self.addHousekeeping(row[0], row[7:], plotting.HK_NAMES)
                plotting.add_housekeeping(*row[1:7], hkValues)
            
    
    self.valuesWidget.show()
    plotting.finish_creating()


    if read_mode == 0:
        print("Opening recording")
        read_file = open(read_file_name, "rb")
        raw_data = np.zeros(read_length)

    elif read_mode == 1:
        print("Connecting Socket...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
        sock.bind((udp_ip, udp_port))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,620000)
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
ss
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
        self.rate = self.b_mask[0].bit_count()/8

        self.bpf = len(self.b_ind) # bytes per frame
        if self.rate == 8/8:
            self.board_id = board_id
        elif self.rate == 4/8:
            self.board_id = [board_id>>4, board_id&0xF]
        else:
            raise ValueError("Unsupported housekeeping rate")

        self.numpoints = int(numpoints*self.rate)
        self.length = 11//self.rate

        self.indcol = np.array(np.arange(10)//self.rate, dtype=np.uint8)[:, None]+1
        self.data = np.zeros((10, self.numpoints))
        self.values = values
        self.maxhkrange = AVG_NUMPOINTS

    def new_data(self, minframes):
        minframes = minframes.astype(np.uint8)
        databuffer = np.zeros(len(minframes)*self.bpf, dtype=np.uint8)
        for i in range(self.bpf):
            databuffer[np.arange(len(minframes))*self.bpf+i] = minframes[:, self.b_ind[i]] & self.b_mask[i]
        if self.rate == 8/8: # ACC, mNLP, PIP
            inds = np.where(databuffer == self.board_id)[0]
            inds = inds[np.where(np.diff(inds) == self.length)[0]]
            if inds.size != 0:
                self.data[:, :inds.size] = databuffer[self.indcol + inds]
                

        elif self.rate == 4/8: # EFP
            inds = np.where( (databuffer==self.board_id[0])[:-1] & (databuffer==self.board_id[1])[1:])[0]
            inds = inds[np.where(np.diff(inds) == self.length)[0]][:-1]
            self.data[:, :inds.size] = databuffer[self.indcol + inds]<<4 | databuffer[self.indcol+1 + inds]

        if do_hkunits:
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
        self.data = np.zeros((10, self.numpoints))
        for value in self.values:
            value.setText("")
if __name__ == "__main__":
    #init()
    #parse
