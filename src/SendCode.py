import socket, time

host = "127.0.0.1"
port = 12000

buffer_size = 620000

file_name = "recordings\VortEx_test02.udp"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

f = open(file_name, "rb")
data = f.read(buffer_size)
cnt = 0

start_time = time.perf_counter()
while data:
    cnt+=1
    if(sock.sendto(data, (host, port))):
        data = f.read(buffer_size)
    time.sleep(1)    
print(time.perf_counter()-start_time)
sock.close()
f.close()