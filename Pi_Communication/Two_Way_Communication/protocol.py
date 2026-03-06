import json
import struct
import hashlib
from datetime import datetime, timezone
PROTOCOL_VERSION = 1
SHARED_TOKEN = "replace-with-strong-secret"
CHUNK_SIZE = 16 * 1024  # 16 KB
def utc_now():
    return datetime.now(timezone.utc).isoformat()
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
def recv_exact(sock, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        buf += chunk
    return buf
def send_frame(sock, header: dict, payload: bytes = b""):
    header = dict(header)
    header["protocol_version"] = PROTOCOL_VERSION
    header["payload_len"] = len(payload)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    sock.sendall(struct.pack(">I", len(header_bytes)))
    sock.sendall(header_bytes)
    if payload:
        sock.sendall(payload)
def recv_frame(sock):
    header_len_bytes = recv_exact(sock, 4)
    header_len = struct.unpack(">I", header_len_bytes)[0]
    header_bytes = recv_exact(sock, header_len)
    header = json.loads(header_bytes.decode("utf-8"))
    payload_len = int(header.get("payload_len", 0))
    payload = recv_exact(sock, payload_len) if payload_len > 0 else b""
    return header, payload
def validate_header(header: dict):
    if header.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Protocol version mismatch")
    if header.get("token") != SHARED_TOKEN:
        raise PermissionError("Invalid token")