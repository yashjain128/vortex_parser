import socket
import numpy as np

f = open("recordings/VortEx_test01.udp", "rb")
UDP_IP = "192.168.1.69"
UDP_PORT = 5005

print("UDP target IP: %s" % UDP_IP)
print("UDP target port: %s" % UDP_PORT)
print("Ctrl + C to end\n...")

try:

    while True:
        msg = f.read(1024)
        if len(msg) == 0:
            f.seek(0)
            msg = f.read(1024)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
        sock.sendto(msg, (UDP_IP, UDP_PORT))
except KeyboardInterrupt:
    print("Ended")