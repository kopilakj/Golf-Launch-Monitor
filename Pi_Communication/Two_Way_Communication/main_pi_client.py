import os
import socket
from uuid import uuid4
from protocol import send_frame, recv_frame, validate_header, utc_now, sha256_bytes, SHARED_TOKEN
SENSOR_IP = "192.168.50.2"
SENSOR_PORT = 5001
SAVE_DIR = "/home/pi/launch_data"
os.makedirs(SAVE_DIR, exist_ok=True)
def trigger_and_receive_csv(sock):
    request_id = str(uuid4())
    shot_id = str(uuid4())
    send_frame(sock, {
        "msg_type": "TRIGGER",
        "request_id": request_id,
        "shot_id": shot_id,
        "timestamp": utc_now(),
        "token": SHARED_TOKEN
    })
    print(f"[Main] TRIGGER sent shot_id={shot_id}")
    # Expect ACK
    header, payload = recv_frame(sock)
    validate_header(header)
    if header.get("msg_type") != "ACK":
        raise RuntimeError(f"Expected ACK, got {header.get('msg_type')}")
    print("[Main] ACK received")
    # Expect CSV_META
    header, payload = recv_frame(sock)
    validate_header(header)
    if header.get("msg_type") != "CSV_META":
        raise RuntimeError(f"Expected CSV_META, got {header.get('msg_type')}")
    file_size = int(header["file_size"])
    expected_sha = header["sha256"]
    total_chunks = int(header["total_chunks"])
    filename = header.get("filename", f"{shot_id}.csv")
    out_path = os.path.join(SAVE_DIR, filename)
    received = bytearray()
    for i in range(total_chunks):
        h, chunk = recv_frame(sock)
        validate_header(h)
        if h.get("msg_type") != "CSV_CHUNK":
            raise RuntimeError(f"Expected CSV_CHUNK, got {h.get('msg_type')}")
        if int(h["chunk_index"]) != i:
            raise RuntimeError(f"Chunk order mismatch: got {h.get('chunk_index')} expected {i}")
        received.extend(chunk)
    h, _ = recv_frame(sock)
    validate_header(h)
    if h.get("msg_type") != "CSV_DONE":
        raise RuntimeError(f"Expected CSV_DONE, got {h.get('msg_type')}")
    if len(received) != file_size:
        raise RuntimeError(f"Size mismatch: got {len(received)} expected {file_size}")
    got_sha = sha256_bytes(bytes(received))
    if got_sha != expected_sha:
        raise RuntimeError("SHA mismatch; file corrupted in transit")
    with open(out_path, "wb") as f:
        f.write(received)
    print(f"[Main] Saved CSV: {out_path}")
    return out_path, shot_id
def run():
    with socket.create_connection((SENSOR_IP, SENSOR_PORT), timeout=10) as sock:
        sock.settimeout(20)
        csv_path, shot_id = trigger_and_receive_csv(sock)
        print(f"[Main] shot_id={shot_id}, csv_path={csv_path}")
if __name__ == "__main__":
    run()