import socket
HOST = "0.0.0.0"   # listen on all interfaces
PORT = 5001
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    print(f"[Sensor] Listening on port {PORT}...")
    conn, addr = s.accept()
    with conn:
        print(f"[Sensor] Connected by {addr}")
        while True:
            data = conn.recv(1024)
            if not data:
                print("[Sensor] Connection closed")
                break
            msg = data.decode("utf-8")
            print(f"[Sensor] Received: {msg}")
            if msg == "TRIGGER":
                conn.sendall(b"ACK")
                conn.sendall(b"CSV_READY")
            elif msg == "PING":
                conn.sendall(b"PONG")
            else:
                conn.sendall(b"UNKNOWN_COMMAND")