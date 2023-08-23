'''
This script is for running 

'''
import time
import os, sys, platform
import ctypes
import socket
start_time = time.time()

import numpy as np
from pandas import read_excel

from PyQt5.QtWidgets import QApplication, QDialog, QWidget

from gui import Window
import newplotting as plotting
# True to turn on debugging
print(platform.system)
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(u'hi')
dbg = True
if dbg:
    print(f"[Debug] {time.time()-start_time} s")
    
# Constants
MINFRAME_LEN = 2 * 40
PACKET_LENGTH = MINFRAME_LEN + 44  
MAX_READ_LENGTH = PACKET_LENGTH * 5000  
SYNC = [64, 40, 107, 254]


e = np.arange(MINFRAME_LEN)
for i in range(0, MINFRAME_LEN, 4):
    e[i:i+4] = e[i:i+4][::-1]
print(e)
sync_arr = np.array(SYNC)
target_sync = np.dot(sync_arr, sync_arr)
def find_SYNC(seq):
    candidates = np.where(np.correlate(seq, sync_arr, mode='valid') == target_sync)[0]
    check = candidates[:, np.newaxis] + np.arange(4)
    mask = np.all((np.take(seq, check) == sync_arr), axis=-1)
    return candidates[mask]   



def parse():
    win.setupGroupBox.setEnabled(False)
    win.time_start()
    mode = win.read_mode
    read_file = None
    write_file = None
    if mode == 0:
        read_file = open(win.read_file, "rb")
    elif mode == 1:
        udp_ip = win.hostInputLine.text()
        port = win.portInputLine.text()

        if dbg:
            print(f"[Debug] Connected\nIP: {udp_ip}\n Port: {port}")    
        sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
        sock.bind((udp_ip, port))


    run = True
    while run:
        if mode == 0:
            raw_data = np.fromfile(read_file, dtype=np.uint8, count=MAX_READ_LENGTH)
        elif mode == 1:
            raw_data, addr = sock.recvfrom(MAX_READ_LENGTH)

        if len(raw_data) == 0:
            break
        
        if win.do_write:
            raw_data.tofile(win.write_file)

        inds = find_SYNC(raw_data)       
        prev_ind = inds[-1]
        inds = inds[:-1][(np.diff(inds) == PACKET_LENGTH)]
        inds[:-1] = inds[:-1][(np.diff(raw_data[inds + 6]) != 0)]

        minframes = raw_data[inds[:, None] + e].astype(int)

        oddframe = minframes[np.where(minframes[:, 57] & 3 == 1)]
        evenframe = minframes[np.where(minframes[:, 57] & 3 == 2)]
        oddsfid = minframes[np.where(minframes[:, 5] % 2 == 1)]
        evensfid = minframes[np.where(minframes[:, 5] % 2 == 0)]

if dbg:
    print(f"[Debug] {time.time()-start_time} s")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(u'hi')
    win = Window()
    win.readStart.clicked.connect(parse)

    win.show()
    sys.exit(app.exec_())
 