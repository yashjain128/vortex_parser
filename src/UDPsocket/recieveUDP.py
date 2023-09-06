import socket

UDP_IP = "192.168.1.72"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind(("", UDP_PORT))

while True:
    try:
        data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    except KeyboardInterrupt:
        print("Done")
        break 

    print("received message: %s" % data)
    