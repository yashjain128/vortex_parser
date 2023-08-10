import time
start_time = time.time()

import os, sys
import socket
import time

import numpy as np

from gui import win, app

MINFRAME_LEN = 2 * 40
PACKET_LENGTH = MINFRAME_LEN + 44  # 128
MAX_READ_LENGTH = PACKET_LENGTH * 5000  # 640000
SYNC = [64, 40, 107, 254]

sync_arr = np.array(SYNC)
target_sync = np.dot(sync_arr, sync_arr)
def find_SYNC(seq):
    candidates = np.where(np.correlate(seq, sync_arr, mode='valid') == target_sync)[0]
    check = candidates[:, np.newaxis] + np.arange(4)
    mask = np.all((np.take(seq, check) == sync_arr), axis=-1)
    return candidates[mask]   



def parse():
    win.setupGroupBox.setEnabled(False)

    mode = win.mode
    if mode == 0:
        read_file = open(win.read_file, "rb")
    elif mode == 1:
        udp_ip = win.hostInputLine.text()
        port = win.portInputLine.text()

        print(udp_ip, port)    
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
        inds = find_SYNC(raw_data)       
        prev_ind = inds[-1]
        inds = inds[:-1][(np.diff(inds) == packetlength)]
        inds[:-1] = inds[:-1][(np.diff(rawData[inds + 6]) != 0)]

        minframes = rawData[inds[:, None] + endianness].astype(int)

        oddframe = minframes[np.where(minframes[:, 57] & 3 == 1)]
        evenframe = minframes[np.where(minframes[:, 57] & 3 == 2)]
        oddsfid = minframes[np.where(minframes[:, 5] % 2 == 1)]
        evensfid = minframes[np.where(minframes[:, 5] % 2 == 0)]

 

    



win.readStart.clicked.connect(parse)
print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    sys.exit(app.exec_())