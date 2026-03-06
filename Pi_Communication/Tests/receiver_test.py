# sensor_server_step2.py
import socket
HOST = "0.0.0.0"
PORT = 5001
def recv_line(conn):
    data = b""
    while not data.endswith(b"\n"):
        chunk = conn.recv(1)
        if not chunk:
            return None
        data += chunk
    return data.decode("utf-8").strip()
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    print(f"[Sensor] Listening on {PORT}...")
    conn, addr = s.accept()
    with conn:
        print(f"[Sensor] Connected by {addr}")
        while True:
            cmd = recv_line(conn)
            if cmd is None:
                print("[Sensor] Client disconnected")
                break
            print(f"[Sensor] Command: {cmd}")
            if cmd == "TRIGGER":
                # 1) ACK first
                conn.sendall(b"ACK\n")
                # 2) build fake CSV (later replace with real file content)
                csv_text = "time_ms,ball_speed,launch_angle\n0,152.3,12.1\n10,153.0,12.0\n"
                csv_bytes = csv_text.encode("utf-8")
                # 3) send length line
                header = f"CSV_LEN:{len(csv_bytes)}\n".encode("utf-8")
                conn.sendall(header)
                # 4) send exact CSV bytes
                conn.sendall(csv_bytes)
                print("[Sensor] CSV sent")
            else:
                conn.sendall(b"UNKNOWN\n")