import socket, time

host = "127.0.0.1"
port = 5000

buffer_size = 126
rep = 620000/buffer_size

file_name = "recordings\VortEx_test02.udp"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

f = open(file_name, "rb")
data = f.read(buffer_size)
cnt = 0

start_time = time.perf_counter()
print("Sending...")
while data:
    for i in range(0, rep, 1):
        if(sock.sendto(data, (host, port))):
            data = f.read(buffer_size)
    time.sleep(1)    
print(time.perf_counter()-start_time)
sock.close()
f.close()