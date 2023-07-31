import socket
import sys

if len(sys.argv) < 3:
    print("USAGE: server.py IP port")
    sys.exit()

ip = sys.argv[1]           # this is local host
port = int(sys.argv[2])    # start port here

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
    try:
        # server must bind to an ip address and port
        sock.bind((ip, port))
        print("Listening on Port:", port)
        break
    except Exception:
        print("ERROR: Cannot connect to Port:", port)
        port += 1

try:
    while True:
        message, addr = sock.recvfrom(1024)  # OK someone pinged me.
        print(f"received from {addr}: {message}")
        sock.sendto(b"Thank you!", addr)
except KeyboardInterrupt:
    pass
finally:
    sock.close()

    