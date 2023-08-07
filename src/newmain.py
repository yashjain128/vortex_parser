import os, sys

from gui import win, app


def parse():
    if win.mode == 0:
        pass
    elif win.mode == 1:
        udp_ip = win.hostInputLine.text()
        port = win.portInputLine.text()

        print(udp_ip, port)    
    



win.readStart.clicked.connect(parse)

if __name__ == '__main__':
    sys.exit(app.exec_())