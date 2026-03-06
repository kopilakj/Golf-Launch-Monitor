import socket
LISTEN_IP = "0.0.0.0"
LISTEN_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((LISTEN_IP, LISTEN_PORT))

print(f"listening on {LISTEN_IP}:{LISTEN_PORT}")
data, addr = sock.recvfrom(2048)
print(f"received from {addr}: {data}")