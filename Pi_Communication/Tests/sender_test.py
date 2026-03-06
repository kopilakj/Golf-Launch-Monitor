import socket
DEST_IP = "192.168.50.2"   # IP of the receiver Pi
DEST_PORT = 5005

msg = b"hello from sender pi"
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(msg, (DEST_IP, DEST_PORT))
print(f"sent: {msg} to {DEST_IP}:{DEST_PORT}")