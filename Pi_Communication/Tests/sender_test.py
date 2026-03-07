# main_client_step2.py
import socket
SENSOR_IP = "192.168.50.2"  # replace with Sensor Pi IP
PORT = 5001
def recv_line(sock):
    data = b""
    while not data.endswith(b"\n"):
        chunk = sock.recv(1)
        if not chunk:
            raise ConnectionError("Socket closed")
        data += chunk
    return data.decode("utf-8").strip()
def recv_exact(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Socket closed during payload receive")
        data += chunk
    return data
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((SENSOR_IP, PORT))
    print("[Main] Connected")
    # send trigger command
    s.sendall(b"TRIGGER\n")
    print("[Main] Sent TRIGGER")
    # read ACK
    ack = recv_line(s)
    print(f"[Main] Received: {ack}")
    if ack != "ACK":
        raise RuntimeError("Did not receive ACK")
    # read length header
    header = recv_line(s)
    print(f"[Main] Received: {header}")
    if not header.startswith("CSV_LEN:"):
        raise RuntimeError("Did not receive CSV length header")
    csv_len = int(header.split(":")[1])
    # read CSV payload
    csv_bytes = recv_exact(s, csv_len)
    # save file
    out_path = "received_shot.csv"
    with open(out_path, "wb") as f:
        f.write(csv_bytes)
    print(f"[Main] Saved CSV to {out_path}")