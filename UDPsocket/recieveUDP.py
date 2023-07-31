import socket
import sys

if len(sys.argv) < 3:
    print("USAGE: client.py IP port")
    sys.exit()

try:
    ip = sys.argv[1]
    port = int(sys.argv[2])
except Exception:
    print("USAGE: client.py IP port")
    sys.exit()

def listen(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        print("UDP sending on Port:", port)
        sock.settimeout(5)
        sock.sendto("Hello".encode(), (ip, port))
        print("message sent")
        print("waiting for response on socket")
        data, addr = sock.recvfrom(1024)
        print("Received:", data.decode(), addr)
    except socket.timeout:
        print("ERROR: acknowledgement was not received")
    except Exception as ex:
        print("ERROR:", ex)
    finally:
        sock.close()

listen(ip, port)