import os, sys
import socket

import numpy as np

from gui import win, app

MINFRAME_LEN = 2 * 40
PACKET_LENGTH = MINFRAME_LEN + 44  # 128
MAX_READ_LENGTH = PACKET_LENGTH * 5000  # 640000
SYNC = [64, 40, 107, 254]

sync_arr = np.array(SYNC)
target_sync = np.dot(sync_arr, sync_arr)
def find_SYNC(seq):
    candidates = np.where(np.correlate(seq, sync, mode='valid') == targetsync)[0]
    check = candidates[:, np.newaxis] + np.arange(4)
    mask = np.all((np.take(seq, check) == sync), axis=-1)
    return candidates[mask]   



def parse():
    mode = win.mode
    if mode == 0:
        read_file = open(win.read_file)
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
            raw_data = read_file.read(MAX_READ_LENGTH)
        elif mode == 1:
            raw_data, addr = sock.recvfrom(MAX_READ_LENGTH)

        inds = find_SYNC(raw_data)        
    
    



win.readStart.clicked.connect(parse)

if __name__ == '__main__':
    sys.exit(app.exec_())