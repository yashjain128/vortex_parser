import socket

f = open("data/VortEx_test01.udp", "rb")
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

    

print("UDP target IP: %s" % UDP_IP)
print("UDP target port: %s" % UDP_PORT)
while True:
    msg = f.read(1000)
    if len(msg) == 0:
        f.seek(0)
        msg = f.read(1000)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
    sock.sendto(msg, (UDP_IP, UDP_PORT))
    
    #print("message: %s" % MESSAGE)

