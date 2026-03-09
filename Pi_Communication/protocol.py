"""
Protocol module for Pi-to-Pi communication
==========================================

Handles:
- Message framing (4-byte length prefix)
- JSON header encoding/decoding
- Binary payload handling (for CSV chunks)
- Checksum verification
- Token validation

Frame format:
  [4 bytes: header length (big-endian)] [header JSON bytes] [payload bytes]

Header always contains:
  - msg_type: Type of message
  - protocol_version: Must match on both sides
  - timestamp: ISO format UTC timestamp
  - request_id: UUID to track request/response pairs
  - token: Shared secret for basic security
  - payload_len: Length of binary payload (0 if none)
"""

import json
import struct
import hashlib
import socket
from datetime import datetime, timezone
from typing import Tuple, Optional, Dict, Any

from config import PROTOCOL_VERSION, SHARED_TOKEN, CHUNK_SIZE, SOCKET_TIMEOUT


class ProtocolError(Exception):
    """Base exception for protocol errors."""
    pass


class ValidationError(ProtocolError):
    """Raised when message validation fails."""
    pass


class ConnectionLostError(ProtocolError):
    """Raised when connection is unexpectedly closed."""
    pass


class TimeoutError(ProtocolError):
    """Raised when operation times out."""
    pass


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def utc_now() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    """Calculate SHA256 hash of bytes, return as hex string."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(filepath: str) -> str:
    """Calculate SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# LOW-LEVEL SOCKET OPERATIONS
# =============================================================================

def recv_exact(sock: socket.socket, n: int) -> bytes:
    """
    Receive exactly n bytes from socket.
    
    Raises:
        ConnectionLostError: If socket closes before receiving all bytes
    """
    buf = b""
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except socket.timeout:
            raise TimeoutError(f"Timed out waiting for {n - len(buf)} more bytes")
        
        if not chunk:
            raise ConnectionLostError(
                f"Connection closed after receiving {len(buf)}/{n} bytes"
            )
        buf += chunk
    return buf


# =============================================================================
# FRAME SENDING
# =============================================================================

def send_frame(sock: socket.socket, header: Dict[str, Any], payload: bytes = b"") -> None:
    """
    Send a framed message over socket.
    
    Args:
        sock: Connected socket
        header: Dictionary with message metadata (msg_type, etc.)
        payload: Optional binary payload (e.g., CSV chunk bytes)
    
    Frame format:
        [4 bytes header_len][header JSON][payload bytes]
    """
    # Add protocol fields to header
    header = dict(header)  # Don't modify original
    header["protocol_version"] = PROTOCOL_VERSION
    header["payload_len"] = len(payload)
    
    # Ensure timestamp is present
    if "timestamp" not in header:
        header["timestamp"] = utc_now()
    
    # Ensure token is present
    if "token" not in header:
        header["token"] = SHARED_TOKEN
    
    # Encode header to JSON bytes
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    
    # Send: [4-byte length][header][payload]
    sock.sendall(struct.pack(">I", len(header_bytes)))
    sock.sendall(header_bytes)
    if payload:
        sock.sendall(payload)


# =============================================================================
# FRAME RECEIVING
# =============================================================================

def recv_frame(sock: socket.socket) -> Tuple[Dict[str, Any], bytes]:
    """
    Receive a framed message from socket.
    
    Returns:
        Tuple of (header dict, payload bytes)
    
    Raises:
        ConnectionLostError: If connection closes unexpectedly
        TimeoutError: If waiting too long for data
        ValidationError: If message format is invalid
    """
    # Read 4-byte header length
    header_len_bytes = recv_exact(sock, 4)
    header_len = struct.unpack(">I", header_len_bytes)[0]
    
    # Sanity check header length (prevent memory issues)
    if header_len > 1024 * 1024:  # 1MB max header
        raise ValidationError(f"Header too large: {header_len} bytes")
    
    # Read header JSON
    header_bytes = recv_exact(sock, header_len)
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValidationError(f"Invalid header JSON: {e}")
    
    # Read payload if present
    payload_len = int(header.get("payload_len", 0))
    if payload_len > 0:
        payload = recv_exact(sock, payload_len)
    else:
        payload = b""
    
    return header, payload


# =============================================================================
# VALIDATION
# =============================================================================

def validate_header(header: Dict[str, Any], expected_types: Optional[list] = None) -> None:
    """
    Validate a received message header.
    
    Args:
        header: The header dictionary to validate
        expected_types: Optional list of acceptable msg_type values
    
    Raises:
        ValidationError: If validation fails
    """
    # Check protocol version
    version = header.get("protocol_version")
    if version != PROTOCOL_VERSION:
        raise ValidationError(
            f"Protocol version mismatch: got {version}, expected {PROTOCOL_VERSION}"
        )
    
    # Check token
    token = header.get("token")
    if token != SHARED_TOKEN:
        raise ValidationError("Invalid authentication token")
    
    # Check required fields
    required = ["msg_type", "timestamp", "request_id"]
    for field in required:
        if field not in header:
            raise ValidationError(f"Missing required field: {field}")
    
    # Check msg_type if expected_types specified
    if expected_types:
        msg_type = header.get("msg_type")
        if msg_type not in expected_types:
            raise ValidationError(
                f"Unexpected message type: {msg_type}, expected one of {expected_types}"
            )


def validate_shot_header(header: Dict[str, Any]) -> None:
    """
    Validate header for shot-related messages.
    Must have shot_id and club_id.
    
    Raises:
        ValidationError: If required shot fields are missing
    """
    shot_fields = ["shot_id", "club_id"]
    for field in shot_fields:
        if field not in header:
            raise ValidationError(f"Missing required shot field: {field}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def send_and_wait(
    sock: socket.socket,
    header: Dict[str, Any],
    payload: bytes = b"",
    expected_response: Optional[str] = None,
    timeout: float = SOCKET_TIMEOUT
) -> Tuple[Dict[str, Any], bytes]:
    """
    Send a frame and wait for response.
    
    Args:
        sock: Connected socket
        header: Message header to send
        payload: Optional payload bytes
        expected_response: Expected msg_type in response (raises if different)
        timeout: Seconds to wait for response
    
    Returns:
        Tuple of (response header, response payload)
    """
    old_timeout = sock.gettimeout()
    try:
        sock.settimeout(timeout)
        send_frame(sock, header, payload)
        resp_header, resp_payload = recv_frame(sock)
        validate_header(resp_header)
        
        if expected_response and resp_header.get("msg_type") != expected_response:
            raise ValidationError(
                f"Expected {expected_response}, got {resp_header.get('msg_type')}"
            )
        
        return resp_header, resp_payload
    finally:
        sock.settimeout(old_timeout)


def create_error_response(
    request_id: str,
    error_message: str,
    shot_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create a standard ERROR response header."""
    header = {
        "msg_type": "ERROR",
        "request_id": request_id,
        "timestamp": utc_now(),
        "token": SHARED_TOKEN,
        "error": error_message,
    }
    if shot_id:
        header["shot_id"] = shot_id
    return header


# =============================================================================
# DEBUG/LOGGING
# =============================================================================

def format_header_for_log(header: Dict[str, Any], max_len: int = 100) -> str:
    """Format header for logging (truncate long values)."""
    safe = {}
    for k, v in header.items():
        if k == "token":
            safe[k] = "***"  # Don't log token
        elif isinstance(v, str) and len(v) > max_len:
            safe[k] = v[:max_len] + "..."
        else:
            safe[k] = v
    return json.dumps(safe, indent=2)


if __name__ == "__main__":
    # Quick self-test
    print("Protocol module loaded successfully")
    print(f"Protocol version: {PROTOCOL_VERSION}")
    print(f"Chunk size: {CHUNK_SIZE} bytes")
    print(f"Current UTC: {utc_now()}")
    
    # Test hash function
    test_data = b"Hello, Golf Launch Monitor!"
    print(f"SHA256 of test data: {sha256_bytes(test_data)[:16]}...")
