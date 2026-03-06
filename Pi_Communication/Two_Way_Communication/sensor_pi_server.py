import os
import math
import socket
import traceback
from uuid import uuid4
from protocol import send_frame, recv_frame, validate_header, utc_now, sha256_bytes, CHUNK_SIZE, SHARED_TOKEN
HOST = "192.168.50.2"   # Sensor Pi ethernet static IP
PORT = 5001
def capture_and_build_csv(shot_id: str) -> bytes:
    # Replace with your real capture pipeline:
    csv_text = "time_ms,ball_speed,launch_angle\n0,152.3,12.1\n10,153.0,12.0\n"
    return csv_text.encode("utf-8")
def handle_trigger(conn, trig_header):
    request_id = trig_header["request_id"]
    shot_id = trig_header["shot_id"]
    send_frame(conn, {
        "msg_type": "ACK",
        "request_id": request_id,
        "shot_id": shot_id,
        "timestamp": utc_now(),
        "token": SHARED_TOKEN,
        "status": "TRIGGER_RECEIVED"
    })
    csv_bytes = capture_and_build_csv(shot_id)
    digest = sha256_bytes(csv_bytes)
    total_chunks = math.ceil(len(csv_bytes) / CHUNK_SIZE) if csv_bytes else 0
    filename = f"{shot_id}.csv"
    send_frame(conn, {
        "msg_type": "CSV_META",
        "request_id": request_id,
        "shot_id": shot_id,
        "timestamp": utc_now(),
        "token": SHARED_TOKEN,
        "filename": filename,
        "file_size": len(csv_bytes),
        "sha256": digest,
        "total_chunks": total_chunks
    })
    for idx in range(total_chunks):
        start = idx * CHUNK_SIZE
        end = start + CHUNK_SIZE
        chunk = csv_bytes[start:end]
        send_frame(conn, {
            "msg_type": "CSV_CHUNK",
            "request_id": request_id,
            "shot_id": shot_id,
            "timestamp": utc_now(),
            "token": SHARED_TOKEN,
            "chunk_index": idx,
            "total_chunks": total_chunks
        }, payload=chunk)
    send_frame(conn, {
        "msg_type": "CSV_DONE",
        "request_id": request_id,
        "shot_id": shot_id,
        "timestamp": utc_now(),
        "token": SHARED_TOKEN,
        "sha256": digest,
        "file_size": len(csv_bytes)
    })
def client_loop(conn, addr):
    print(f"[Sensor] Connected: {addr}")
    try:
        while True:
            header, payload = recv_frame(conn)
            validate_header(header)
            if header.get("msg_type") == "TRIGGER":
                handle_trigger(conn, header)
            else:
                send_frame(conn, {
                    "msg_type": "ERROR",
                    "request_id": header.get("request_id", str(uuid4())),
                    "shot_id": header.get("shot_id", "unknown"),
                    "timestamp": utc_now(),
                    "token": SHARED_TOKEN,
                    "error": f"Unsupported msg_type: {header.get('msg_type')}"
                })
    except Exception as e:
        print(f"[Sensor] Client loop ended: {e}")
def run_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[Sensor] Listening on {HOST}:{PORT}")
        while True:
            conn, addr = s.accept()
            with conn:
                try:
                    client_loop(conn, addr)
                except Exception:
                    traceback.print_exc()
if __name__ == "__main__":
    run_server()