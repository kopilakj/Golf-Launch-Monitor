import socket
SENSOR_IP = "192.168.1.50.2"   # replace with your Sensor Pi ethernet IP
PORT = 5001
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((SENSOR_IP, PORT))
    print("[Main] Connected to sensor")
    # Main -> Sensor
    s.sendall(b"TRIGGER")
    print("[Main] Sent: TRIGGER")
    # Sensor -> Main
    reply1 = s.recv(1024).decode("utf-8")
    print(f"[Main] Received: {reply1}")
    reply2 = s.recv(1024).decode("utf-8")
    print(f"[Main] Received: {reply2}")
    # Main -> Sensor again
    s.sendall(b"PING")
    print("[Main] Sent: PING")
    reply3 = s.recv(1024).decode("utf-8")
    print(f"[Main] Received: {reply3}")