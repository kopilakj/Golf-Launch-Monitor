#!/usr/bin/env python3
"""
Main Pi Client
==============

This runs on the MAIN PI (the one connected to the GUI).
It connects to Sensor Pi and:
1. Sends CLUB_PRESET when user selects a club
2. Sends TRIGGER when user swings
3. Receives CSV data from Sensor Pi
4. Parses CSV and forwards to Flask/GUI

Run this with: python3 main_client.py
"""

import os
import sys
import csv
import socket
import time
import traceback
from io import StringIO
from typing import Optional, Dict, Any, Tuple
from threading import Thread, Lock
from queue import Queue

# Import our modules
from config import (
    SENSOR_PI_IP, PI_COMM_PORT, CSV_SAVE_DIR,
    SOCKET_TIMEOUT, RECONNECT_DELAY, MAX_RETRIES,
    CLUB_PRESETS, DEFAULT_CLUB, validate_config
)
from protocol import (
    send_frame, recv_frame, validate_header, validate_shot_header,
    sha256_bytes, utc_now, format_header_for_log,
    ProtocolError, ValidationError, ConnectionLostError, TimeoutError
)
from messages import (
    MSG_CLUB_PRESET, MSG_ACK_PRESET, MSG_TRIGGER, MSG_ACK_TRIGGER,
    MSG_CSV_META, MSG_CSV_CHUNK, MSG_CSV_DONE, MSG_PING, MSG_PONG, MSG_ERROR,
    make_club_preset, make_trigger, make_ping,
    new_shot_id, new_request_id, get_preset_version
)


# =============================================================================
# MAIN PI STATE
# =============================================================================

class MainPiState:
    """Tracks current state of Main Pi communication."""
    
    def __init__(self):
        self.lock = Lock()
        self.connected = False
        self.current_club_id: str = DEFAULT_CLUB
        self.current_preset_version: int = 1
        self.preset_confirmed: bool = False
        self.shots_triggered: int = 0
        self.shots_received: int = 0
        
        # Queue for shot results (to send to Flask/GUI)
        self.result_queue: Queue = Queue()
    
    def set_club(self, club_id: str, preset_version: int):
        with self.lock:
            self.current_club_id = club_id
            self.current_preset_version = preset_version
            self.preset_confirmed = False
    
    def confirm_preset(self):
        with self.lock:
            self.preset_confirmed = True
    
    def get_current_club(self) -> Tuple[str, int]:
        with self.lock:
            return self.current_club_id, self.current_preset_version


# Global state
state = MainPiState()


# =============================================================================
# CONNECTION MANAGEMENT
# =============================================================================

class SensorConnection:
    """Manages connection to Sensor Pi."""
    
    def __init__(self):
        self.sock: Optional[socket.socket] = None
        self.connected = False
    
    def connect(self) -> bool:
        """Connect to Sensor Pi."""
        if self.connected and self.sock:
            return True
        
        print(f"[Main] Connecting to Sensor Pi at {SENSOR_PI_IP}:{PI_COMM_PORT}...")
        
        try:
            self.sock = socket.create_connection(
                (SENSOR_PI_IP, PI_COMM_PORT),
                timeout=SOCKET_TIMEOUT
            )
            self.sock.settimeout(SOCKET_TIMEOUT)
            self.connected = True
            print("[Main] Connected!")
            return True
        
        except socket.error as e:
            print(f"[Main] Connection failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Sensor Pi."""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
        self.connected = False
    
    def reconnect(self, max_attempts: int = MAX_RETRIES) -> bool:
        """Attempt to reconnect to Sensor Pi."""
        self.disconnect()
        
        for attempt in range(max_attempts):
            print(f"[Main] Reconnect attempt {attempt + 1}/{max_attempts}...")
            if self.connect():
                return True
            time.sleep(RECONNECT_DELAY)
        
        print("[Main] Failed to reconnect after all attempts")
        return False
    
    def send(self, header: Dict[str, Any], payload: bytes = b"") -> bool:
        """Send a frame, return True on success."""
        if not self.connected or not self.sock:
            print("[Main] Not connected, cannot send")
            return False
        
        try:
            send_frame(self.sock, header, payload)
            return True
        except Exception as e:
            print(f"[Main] Send error: {e}")
            self.connected = False
            return False
    
    def recv(self) -> Tuple[Optional[Dict[str, Any]], Optional[bytes]]:
        """Receive a frame, return (header, payload) or (None, None) on error."""
        if not self.connected or not self.sock:
            print("[Main] Not connected, cannot receive")
            return None, None
        
        try:
            header, payload = recv_frame(self.sock)
            validate_header(header)
            return header, payload
        except Exception as e:
            print(f"[Main] Receive error: {e}")
            self.connected = False
            return None, None


# Global connection
sensor_conn = SensorConnection()


# =============================================================================
# CLUB PRESET
# =============================================================================

def send_club_preset(club_id: str) -> bool:
    """
    Send club preset to Sensor Pi.
    
    Call this when user selects a club in the GUI.
    Returns True if preset was acknowledged.
    """
    if not sensor_conn.connected:
        if not sensor_conn.connect():
            return False
    
    # Get preset version for this club
    preset_version = get_preset_version(club_id)
    state.set_club(club_id, preset_version)
    
    print(f"[Main] Sending CLUB_PRESET for {club_id} (version {preset_version})")
    
    # Create and send preset message
    msg = make_club_preset(club_id)
    if not sensor_conn.send(msg):
        return False
    
    # Wait for ACK_PRESET
    header, _ = sensor_conn.recv()
    if not header:
        return False
    
    if header.get("msg_type") != MSG_ACK_PRESET:
        print(f"[Main] Expected ACK_PRESET, got {header.get('msg_type')}")
        return False
    
    # Verify it's for the right club/version
    if header.get("club_id") != club_id:
        print(f"[Main] Club mismatch in ACK_PRESET")
        return False
    
    state.confirm_preset()
    print(f"[Main] Preset confirmed for {club_id}")
    return True


# =============================================================================
# TRIGGER SHOT
# =============================================================================

def trigger_shot(club_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Trigger a shot capture on Sensor Pi.
    
    This is the main function to call when the user swings.
    
    Args:
        club_id: Optional club ID override. If None, uses current club.
    
    Returns:
        Shot result dictionary with metrics, or None on error.
    """
    if not sensor_conn.connected:
        if not sensor_conn.connect():
            print("[Main] Cannot trigger: not connected to Sensor Pi")
            return None
    
    # Get current club or use override
    if club_id is None:
        club_id, preset_version = state.get_current_club()
    else:
        preset_version = get_preset_version(club_id)
    
    # Generate shot ID
    shot_id = new_shot_id()
    state.shots_triggered += 1
    
    print(f"[Main] Triggering shot:")
    print(f"       shot_id: {shot_id}")
    print(f"       club_id: {club_id}")
    
    # Send TRIGGER
    msg = make_trigger(shot_id, club_id, preset_version)
    request_id = msg["request_id"]
    
    if not sensor_conn.send(msg):
        return None
    
    # Wait for ACK_TRIGGER
    header, _ = sensor_conn.recv()
    if not header:
        return None
    
    if header.get("msg_type") == MSG_ERROR:
        print(f"[Main] Sensor error: {header.get('error')}")
        return None
    
    if header.get("msg_type") != MSG_ACK_TRIGGER:
        print(f"[Main] Expected ACK_TRIGGER, got {header.get('msg_type')}")
        return None
    
    print("[Main] ACK_TRIGGER received, waiting for CSV...")
    
    # Receive CSV data
    csv_data = receive_csv(shot_id)
    if csv_data is None:
        return None
    
    # Parse CSV into metrics
    metrics = parse_csv(csv_data)
    
    # Create result
    result = {
        "shot_id": shot_id,
        "club_id": club_id,
        "preset_version": preset_version,
        "timestamp": utc_now(),
        "metrics": metrics,
        "raw_csv": csv_data.decode("utf-8"),
    }
    
    state.shots_received += 1
    state.result_queue.put(result)
    
    print(f"[Main] Shot complete! Metrics: {metrics}")
    return result


def receive_csv(shot_id: str) -> Optional[bytes]:
    """Receive CSV data from Sensor Pi."""
    
    # Expect CSV_META
    header, _ = sensor_conn.recv()
    if not header:
        return None
    
    if header.get("msg_type") == MSG_ERROR:
        print(f"[Main] Sensor error: {header.get('error')}")
        return None
    
    if header.get("msg_type") != MSG_CSV_META:
        print(f"[Main] Expected CSV_META, got {header.get('msg_type')}")
        return None
    
    # Extract metadata
    expected_size = int(header["file_size"])
    expected_sha = header["sha256"]
    total_chunks = int(header["total_chunks"])
    filename = header.get("filename", f"{shot_id}.csv")
    
    print(f"[Main] Expecting {expected_size} bytes in {total_chunks} chunks")
    
    # Receive chunks
    received = bytearray()
    for i in range(total_chunks):
        chunk_header, chunk_data = sensor_conn.recv()
        if chunk_header is None or chunk_data is None:
            print(f"[Main] Lost connection during chunk {i}")
            return None
        
        if chunk_header.get("msg_type") != MSG_CSV_CHUNK:
            print(f"[Main] Expected CSV_CHUNK, got {chunk_header.get('msg_type')}")
            return None
        
        chunk_idx = int(chunk_header["chunk_index"])
        if chunk_idx != i:
            print(f"[Main] Chunk order error: got {chunk_idx}, expected {i}")
            return None
        
        # chunk_data is guaranteed to be bytes here (not None)
        received.extend(bytes(chunk_data))
        print(f"[Main] Received chunk {i + 1}/{total_chunks}")
    
    # Expect CSV_DONE
    done_header, _ = sensor_conn.recv()
    if not done_header:
        return None
    
    if done_header.get("msg_type") != MSG_CSV_DONE:
        print(f"[Main] Expected CSV_DONE, got {done_header.get('msg_type')}")
        return None
    
    # Verify size
    if len(received) != expected_size:
        print(f"[Main] Size mismatch: got {len(received)}, expected {expected_size}")
        return None
    
    # Verify checksum
    actual_sha = sha256_bytes(bytes(received))
    if actual_sha != expected_sha:
        print("[Main] Checksum mismatch! Data may be corrupted.")
        return None
    
    # Save to file
    save_path = os.path.join(CSV_SAVE_DIR, filename)
    os.makedirs(CSV_SAVE_DIR, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(received)
    print(f"[Main] Saved CSV to {save_path}")
    
    return bytes(received)


def parse_csv(csv_bytes: bytes) -> Dict[str, Any]:
    """
    Parse CSV data into metrics dictionary.
    
    Expected format:
    metric,value,unit
    ball_speed,152.3,mph
    ...
    """
    metrics = {}
    
    try:
        text = csv_bytes.decode("utf-8")
        reader = csv.DictReader(StringIO(text))
        
        for row in reader:
            metric = row.get("metric", "").strip()
            value = row.get("value", "").strip()
            unit = row.get("unit", "").strip()
            
            if metric and value:
                # Try to convert to number
                try:
                    value = float(value)
                except ValueError:
                    pass
                
                metrics[metric] = {
                    "value": value,
                    "unit": unit
                }
    
    except Exception as e:
        print(f"[Main] CSV parse error: {e}")
    
    return metrics


# =============================================================================
# HEALTH CHECK
# =============================================================================

def ping_sensor() -> bool:
    """
    Send PING to Sensor Pi and wait for PONG.
    Returns True if Sensor Pi responds.
    """
    if not sensor_conn.connected:
        if not sensor_conn.connect():
            return False
    
    msg = make_ping()
    if not sensor_conn.send(msg):
        return False
    
    header, _ = sensor_conn.recv()
    if not header:
        return False
    
    if header.get("msg_type") == MSG_PONG:
        print("[Main] Sensor Pi responded to PING")
        return True
    
    return False


# =============================================================================
# INTERACTIVE TEST MODE
# =============================================================================

def interactive_test():
    """
    Simple interactive test mode.
    
    This lets you test the protocol without the GUI.
    """
    print()
    print("=" * 60)
    print("MAIN PI - INTERACTIVE TEST MODE")
    print("=" * 60)
    print()
    print("Commands:")
    print("  c <club>  - Send club preset (e.g., 'c driver')")
    print("  t         - Trigger a shot")
    print("  p         - Ping sensor")
    print("  s         - Show status")
    print("  q         - Quit")
    print()
    print("Available clubs:", list(CLUB_PRESETS.keys()))
    print()
    
    while True:
        try:
            cmd = input("> ").strip().lower()
            
            if not cmd:
                continue
            
            parts = cmd.split()
            action = parts[0]
            
            if action == "q":
                break
            
            elif action == "c":
                if len(parts) < 2:
                    print("Usage: c <club_id>")
                    continue
                club_id = parts[1]
                if club_id not in CLUB_PRESETS:
                    print(f"Unknown club. Available: {list(CLUB_PRESETS.keys())}")
                    continue
                send_club_preset(club_id)
            
            elif action == "t":
                result = trigger_shot()
                if result:
                    print()
                    print("Shot Result:")
                    for key, val in result["metrics"].items():
                        if isinstance(val, dict):
                            print(f"  {key}: {val['value']} {val['unit']}")
                        else:
                            print(f"  {key}: {val}")
            
            elif action == "p":
                if ping_sensor():
                    print("Sensor Pi is alive!")
                else:
                    print("No response from Sensor Pi")
            
            elif action == "s":
                club_id, version = state.get_current_club()
                print(f"Current club: {club_id} (v{version})")
                print(f"Preset confirmed: {state.preset_confirmed}")
                print(f"Shots triggered: {state.shots_triggered}")
                print(f"Shots received: {state.shots_received}")
                print(f"Connected: {sensor_conn.connected}")
            
            else:
                print("Unknown command. Use: c, t, p, s, or q")
        
        except KeyboardInterrupt:
            print()
            break
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
    
    sensor_conn.disconnect()
    print("Goodbye!")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    
    # Validate config
    if not validate_config():
        sys.exit(1)
    
    # Create save directory
    os.makedirs(CSV_SAVE_DIR, exist_ok=True)
    
    print("=" * 60)
    print("MAIN PI CLIENT")
    print("=" * 60)
    print(f"Will connect to: {SENSOR_PI_IP}:{PI_COMM_PORT}")
    print(f"CSV save dir: {CSV_SAVE_DIR}")
    print("=" * 60)
    print()
    
    # Try to connect
    if not sensor_conn.connect():
        print()
        print("Could not connect to Sensor Pi.")
        print("Make sure sensor_server.py is running on Sensor Pi!")
        print()
        print("Starting in offline test mode anyway...")
    
    # Run interactive test
    interactive_test()


if __name__ == "__main__":
    main()
