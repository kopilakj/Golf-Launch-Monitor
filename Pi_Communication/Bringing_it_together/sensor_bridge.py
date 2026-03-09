#!/usr/bin/env python3
"""
Sensor Bridge - Main Pi Communication Service
==============================================

This runs on the MAIN PI alongside your partner's app.py.
It acts as the bridge between:
  - The Web GUI (app.py on port 5000)
  - The Sensor Pi (sensor_server.py on port 5002)

What it does:
1. Receives club selection from GUI via /set_club endpoint
2. Sends CLUB_PRESET to Sensor Pi
3. Receives shot trigger from GUI via /trigger endpoint  
4. Sends TRIGGER to Sensor Pi, receives CSV data
5. Parses metrics and POSTs them to app.py:/api/upload_metrics
6. GUI displays the metrics

Run this with: python3 sensor_bridge.py

This replaces the old:
  - pi_server.py (club selection)
  - send_metrics.py (metrics forwarding)
"""

import os
import sys
import json
import time
import socket
import requests
import traceback
from threading import Thread, Lock
from typing import Optional, Dict, Any, Tuple
from io import StringIO
import csv

# Flask for the bridge API
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
except ImportError:
    print("ERROR: Flask not installed!")
    print("Run: pip install flask flask-cors requests")
    sys.exit(1)

# Import our protocol modules
from config import (
    SENSOR_PI_IP, PI_COMM_PORT, BRIDGE_PORT,
    FLASK_HOST, WEBAPP_METRICS_URL,
    CSV_SAVE_DIR, SOCKET_TIMEOUT, MAX_RETRIES, RECONNECT_DELAY,
    CLUB_PRESETS, DEFAULT_CLUB,
    normalize_club_name, PRESET_TO_GUI,
    validate_config
)
from protocol import (
    send_frame, recv_frame, validate_header, validate_shot_header,
    sha256_bytes, utc_now,
    ProtocolError, ValidationError, ConnectionLostError
)
from messages import (
    MSG_CLUB_PRESET, MSG_ACK_PRESET, MSG_TRIGGER, MSG_ACK_TRIGGER,
    MSG_CSV_META, MSG_CSV_CHUNK, MSG_CSV_DONE, MSG_PING, MSG_PONG, MSG_ERROR,
    make_club_preset, make_trigger, make_ping,
    new_shot_id, new_request_id, get_preset_version
)


# =============================================================================
# BRIDGE STATE
# =============================================================================

class BridgeState:
    """Tracks current state of the bridge."""
    
    def __init__(self):
        self.lock = Lock()
        self.current_club_gui: str = ""          # GUI name (e.g., "Driver")
        self.current_club_preset: str = DEFAULT_CLUB  # Internal name (e.g., "driver")
        self.current_preset_version: int = 1
        self.preset_confirmed: bool = False
        self.sensor_connected: bool = False
        self.shots_triggered: int = 0
        self.shots_received: int = 0
        self.last_error: str = ""
        self.last_metrics: Dict[str, Any] = {}
    
    def set_club(self, gui_name: str, preset_name: str, version: int):
        with self.lock:
            self.current_club_gui = gui_name
            self.current_club_preset = preset_name
            self.current_preset_version = version
            self.preset_confirmed = False
    
    def confirm_preset(self):
        with self.lock:
            self.preset_confirmed = True
    
    def get_club(self) -> Tuple[str, str, int]:
        """Returns (gui_name, preset_name, version)"""
        with self.lock:
            return self.current_club_gui, self.current_club_preset, self.current_preset_version
    
    def set_error(self, error: str):
        with self.lock:
            self.last_error = error
    
    def set_metrics(self, metrics: Dict[str, Any]):
        with self.lock:
            self.last_metrics = metrics


state = BridgeState()


# =============================================================================
# SENSOR PI CONNECTION
# =============================================================================

class SensorConnection:
    """Manages TCP connection to Sensor Pi."""
    
    def __init__(self):
        self.sock: Optional[socket.socket] = None
        self.connected = False
        self.lock = Lock()
    
    def connect(self) -> bool:
        """Connect to Sensor Pi."""
        with self.lock:
            if self.connected and self.sock:
                return True
            
            print(f"[Bridge] Connecting to Sensor Pi at {SENSOR_PI_IP}:{PI_COMM_PORT}...")
            
            try:
                self.sock = socket.create_connection(
                    (SENSOR_PI_IP, PI_COMM_PORT),
                    timeout=SOCKET_TIMEOUT
                )
                self.sock.settimeout(SOCKET_TIMEOUT)
                self.connected = True
                state.sensor_connected = True
                print("[Bridge] Connected to Sensor Pi!")
                return True
            
            except socket.error as e:
                print(f"[Bridge] Connection failed: {e}")
                self.connected = False
                state.sensor_connected = False
                state.set_error(f"Cannot connect to Sensor Pi: {e}")
                return False
    
    def disconnect(self):
        """Disconnect from Sensor Pi."""
        with self.lock:
            if self.sock:
                try:
                    self.sock.close()
                except:
                    pass
                self.sock = None
            self.connected = False
            state.sensor_connected = False
    
    def reconnect(self) -> bool:
        """Attempt to reconnect."""
        self.disconnect()
        for attempt in range(MAX_RETRIES):
            print(f"[Bridge] Reconnect attempt {attempt + 1}/{MAX_RETRIES}...")
            if self.connect():
                return True
            time.sleep(RECONNECT_DELAY)
        return False
    
    def send(self, header: Dict[str, Any], payload: bytes = b"") -> bool:
        """Send a frame to Sensor Pi."""
        with self.lock:
            if not self.connected or not self.sock:
                return False
            try:
                send_frame(self.sock, header, payload)
                return True
            except Exception as e:
                print(f"[Bridge] Send error: {e}")
                self.connected = False
                state.sensor_connected = False
                return False
    
    def recv(self) -> Tuple[Optional[Dict[str, Any]], Optional[bytes]]:
        """Receive a frame from Sensor Pi."""
        with self.lock:
            if not self.connected or not self.sock:
                return None, None
            try:
                header, payload = recv_frame(self.sock)
                validate_header(header)
                return header, payload
            except Exception as e:
                print(f"[Bridge] Receive error: {e}")
                self.connected = False
                state.sensor_connected = False
                return None, None


sensor = SensorConnection()


# =============================================================================
# COMMUNICATION FUNCTIONS
# =============================================================================

def send_club_preset(gui_club_name: str) -> Dict[str, Any]:
    """
    Send club preset to Sensor Pi.
    
    Args:
        gui_club_name: Club name from GUI (e.g., "Driver", "7 Iron")
    
    Returns:
        Dict with success status and details
    """
    # Normalize club name
    preset_name = normalize_club_name(gui_club_name)
    preset_version = get_preset_version(preset_name)
    
    print(f"[Bridge] Club selected: '{gui_club_name}' -> preset '{preset_name}'")
    
    # Update state
    state.set_club(gui_club_name, preset_name, preset_version)
    
    # Connect if needed
    if not sensor.connected:
        if not sensor.connect():
            return {
                "success": False,
                "error": "Cannot connect to Sensor Pi",
                "club": gui_club_name
            }
    
    # Send CLUB_PRESET
    msg = make_club_preset(preset_name)
    if not sensor.send(msg):
        sensor.reconnect()
        if not sensor.send(msg):
            return {
                "success": False,
                "error": "Failed to send preset to Sensor Pi",
                "club": gui_club_name
            }
    
    # Wait for ACK_PRESET
    header, _ = sensor.recv()
    if not header:
        return {
            "success": False,
            "error": "No response from Sensor Pi",
            "club": gui_club_name
        }
    
    if header.get("msg_type") == MSG_ERROR:
        return {
            "success": False,
            "error": header.get("error", "Unknown error"),
            "club": gui_club_name
        }
    
    if header.get("msg_type") != MSG_ACK_PRESET:
        return {
            "success": False,
            "error": f"Unexpected response: {header.get('msg_type')}",
            "club": gui_club_name
        }
    
    state.confirm_preset()
    print(f"[Bridge] Preset confirmed for {preset_name}")
    
    return {
        "success": True,
        "club": gui_club_name,
        "preset": preset_name,
        "version": preset_version
    }


def trigger_shot() -> Dict[str, Any]:
    """
    Trigger a shot capture on Sensor Pi.
    
    Returns:
        Dict with shot_id, metrics, and success status
    """
    gui_club, preset_name, preset_version = state.get_club()
    
    if not preset_name:
        return {
            "success": False,
            "error": "No club selected"
        }
    
    # Connect if needed
    if not sensor.connected:
        if not sensor.connect():
            return {
                "success": False,
                "error": "Cannot connect to Sensor Pi"
            }
    
    # Generate shot ID
    shot_id = new_shot_id()
    state.shots_triggered += 1
    
    print(f"[Bridge] Triggering shot {shot_id} with {preset_name}")
    
    # Send TRIGGER
    msg = make_trigger(shot_id, preset_name, preset_version)
    if not sensor.send(msg):
        return {
            "success": False,
            "error": "Failed to send trigger",
            "shot_id": shot_id
        }
    
    # Wait for ACK_TRIGGER
    header, _ = sensor.recv()
    if not header:
        return {
            "success": False,
            "error": "No ACK from Sensor Pi",
            "shot_id": shot_id
        }
    
    if header.get("msg_type") == MSG_ERROR:
        return {
            "success": False,
            "error": header.get("error", "Sensor error"),
            "shot_id": shot_id
        }
    
    if header.get("msg_type") != MSG_ACK_TRIGGER:
        return {
            "success": False,
            "error": f"Unexpected: {header.get('msg_type')}",
            "shot_id": shot_id
        }
    
    print("[Bridge] ACK_TRIGGER received, waiting for CSV...")
    
    # Receive CSV data
    csv_result = receive_csv(shot_id)
    if not csv_result["success"]:
        return csv_result
    
    csv_bytes = csv_result["data"]
    
    # Parse CSV into metrics
    metrics = parse_csv_to_metrics(csv_bytes)
    
    # Save CSV file
    save_csv(shot_id, csv_bytes)
    
    # Store metrics in state
    state.set_metrics(metrics)
    state.shots_received += 1
    
    # POST metrics to app.py
    post_result = post_metrics_to_webapp(metrics)
    
    print(f"[Bridge] Shot complete! Metrics sent to webapp: {post_result}")
    
    return {
        "success": True,
        "shot_id": shot_id,
        "club": gui_club,
        "preset": preset_name,
        "metrics": metrics,
        "webapp_notified": post_result
    }


def receive_csv(shot_id: str) -> Dict[str, Any]:
    """Receive CSV data from Sensor Pi."""
    
    # Expect CSV_META
    header, _ = sensor.recv()
    if not header:
        return {"success": False, "error": "Lost connection waiting for CSV_META"}
    
    if header.get("msg_type") == MSG_ERROR:
        return {"success": False, "error": header.get("error")}
    
    if header.get("msg_type") != MSG_CSV_META:
        return {"success": False, "error": f"Expected CSV_META, got {header.get('msg_type')}"}
    
    expected_size = int(header["file_size"])
    expected_sha = header["sha256"]
    total_chunks = int(header["total_chunks"])
    
    print(f"[Bridge] Expecting {expected_size} bytes in {total_chunks} chunks")
    
    # Receive chunks
    received = bytearray()
    for i in range(total_chunks):
        chunk_header, chunk_data = sensor.recv()
        if chunk_header is None or chunk_data is None:
            return {"success": False, "error": f"Lost connection at chunk {i}"}
        
        if chunk_header.get("msg_type") != MSG_CSV_CHUNK:
            return {"success": False, "error": f"Expected CSV_CHUNK, got {chunk_header.get('msg_type')}"}
        
        received.extend(chunk_data)
        print(f"[Bridge] Chunk {i + 1}/{total_chunks} received")
    
    # Expect CSV_DONE
    done_header, _ = sensor.recv()
    if not done_header:
        return {"success": False, "error": "Lost connection waiting for CSV_DONE"}
    
    if done_header.get("msg_type") != MSG_CSV_DONE:
        return {"success": False, "error": f"Expected CSV_DONE, got {done_header.get('msg_type')}"}
    
    # Verify
    if len(received) != expected_size:
        return {"success": False, "error": f"Size mismatch: {len(received)} vs {expected_size}"}
    
    actual_sha = sha256_bytes(bytes(received))
    if actual_sha != expected_sha:
        return {"success": False, "error": "Checksum mismatch"}
    
    print("[Bridge] CSV received and verified!")
    return {"success": True, "data": bytes(received)}


def parse_csv_to_metrics(csv_bytes: bytes) -> Dict[str, Any]:
    """
    Parse CSV data into metrics dict for the GUI.
    
    Expected CSV format from sensor:
        metric,value,unit
        ball_speed,152.3,mph
        launch_angle,12.5,degrees
        ...
    
    Returns format expected by app.py:
        {"ball_speed": 152.3, "launch_angle": 12.5, ...}
    """
    metrics = {}
    
    try:
        text = csv_bytes.decode("utf-8")
        reader = csv.DictReader(StringIO(text))
        
        for row in reader:
            metric = row.get("metric", "").strip()
            value = row.get("value", "").strip()
            
            if metric and value:
                # Skip metadata fields
                if metric in ("shot_id", "club_id", "preset_version"):
                    continue
                
                # Convert to number if possible
                try:
                    value = float(value)
                    # Round to reasonable precision
                    if value == int(value):
                        value = int(value)
                    else:
                        value = round(value, 1)
                except ValueError:
                    pass
                
                # Convert metric name to match GUI expectations
                # (e.g., "ball_speed" stays as "ball_speed")
                metrics[metric] = value
    
    except Exception as e:
        print(f"[Bridge] CSV parse error: {e}")
        metrics = {"error": str(e)}
    
    return metrics


def save_csv(shot_id: str, csv_bytes: bytes):
    """Save CSV to local file."""
    try:
        os.makedirs(CSV_SAVE_DIR, exist_ok=True)
        path = os.path.join(CSV_SAVE_DIR, f"{shot_id}.csv")
        with open(path, "wb") as f:
            f.write(csv_bytes)
        print(f"[Bridge] CSV saved to {path}")
    except Exception as e:
        print(f"[Bridge] Failed to save CSV: {e}")


def post_metrics_to_webapp(metrics: Dict[str, Any]) -> bool:
    """POST metrics to app.py:/api/upload_metrics"""
    try:
        response = requests.post(
            WEBAPP_METRICS_URL,
            json=metrics,
            timeout=5
        )
        if response.status_code == 200:
            print(f"[Bridge] Metrics posted to webapp successfully")
            return True
        else:
            print(f"[Bridge] Webapp returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"[Bridge] Failed to post metrics: {e}")
        return False


def ping_sensor() -> bool:
    """Ping Sensor Pi to check connection."""
    if not sensor.connected:
        if not sensor.connect():
            return False
    
    msg = make_ping()
    if not sensor.send(msg):
        return False
    
    header, _ = sensor.recv()
    if header and header.get("msg_type") == MSG_PONG:
        return True
    return False


# =============================================================================
# FLASK API (Bridge Server)
# =============================================================================

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    """Health check."""
    return jsonify({
        "service": "Golf Launch Monitor - Sensor Bridge",
        "status": "running",
        "sensor_connected": sensor.connected,
        "current_club": state.current_club_gui
    })


@app.route("/status", methods=["GET"])
def get_status():
    """Get detailed status."""
    gui_club, preset_name, version = state.get_club()
    return jsonify({
        "sensor_connected": sensor.connected,
        "current_club_gui": gui_club,
        "current_club_preset": preset_name,
        "preset_version": version,
        "preset_confirmed": state.preset_confirmed,
        "shots_triggered": state.shots_triggered,
        "shots_received": state.shots_received,
        "last_error": state.last_error
    })


@app.route("/set_club", methods=["POST"])
def set_club():
    """
    Receive club selection from GUI and forward to Sensor Pi.
    
    Expected POST body: {"club": "Driver"} or {"club": "7 Iron"}
    
    This endpoint is called by select_club.html when user clicks a club.
    """
    data = request.get_json() or {}
    club = data.get("club")
    
    if not club:
        return jsonify({"success": False, "error": "No club specified"}), 400
    
    print(f"[Bridge] /set_club called with: {club}")
    
    # Also save to file for compatibility with old system
    try:
        with open("current_club.json", "w") as f:
            json.dump({"club": club}, f)
    except:
        pass
    
    # Send to Sensor Pi
    result = send_club_preset(club)
    
    return jsonify(result)


@app.route("/get_club", methods=["GET"])
def get_club():
    """Get current club (for compatibility with old pi_server.py)."""
    gui_club, preset_name, version = state.get_club()
    return jsonify({
        "club": gui_club,
        "preset": preset_name,
        "version": version
    })


@app.route("/trigger", methods=["POST"])
def trigger():
    """
    Trigger a shot capture.
    
    Call this when the sensor detects a swing, or from a GUI button.
    """
    print("[Bridge] /trigger called")
    result = trigger_shot()
    return jsonify(result)


@app.route("/ping", methods=["GET"])
def ping():
    """Check if Sensor Pi is reachable."""
    success = ping_sensor()
    return jsonify({
        "success": success,
        "sensor_connected": sensor.connected,
        "sensor_ip": SENSOR_PI_IP,
        "sensor_port": PI_COMM_PORT
    })


@app.route("/connect", methods=["POST"])
def connect():
    """Manually connect to Sensor Pi."""
    success = sensor.connect()
    return jsonify({
        "success": success,
        "connected": sensor.connected
    })


@app.route("/disconnect", methods=["POST"])
def disconnect():
    """Disconnect from Sensor Pi."""
    sensor.disconnect()
    return jsonify({
        "success": True,
        "connected": False
    })


@app.route("/last_metrics", methods=["GET"])
def last_metrics():
    """Get the last received metrics."""
    return jsonify({
        "metrics": state.last_metrics,
        "shots_received": state.shots_received
    })


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Start the bridge server."""
    
    print()
    print("=" * 60)
    print("SENSOR BRIDGE - Main Pi Communication Service")
    print("=" * 60)
    print()
    print(f"Bridge API:     http://0.0.0.0:{BRIDGE_PORT}")
    print(f"Sensor Pi:      {SENSOR_PI_IP}:{PI_COMM_PORT}")
    print(f"Webapp metrics: {WEBAPP_METRICS_URL}")
    print()
    print("Endpoints:")
    print(f"  POST /set_club  - Set club (from GUI)")
    print(f"  GET  /get_club  - Get current club")
    print(f"  POST /trigger   - Trigger shot capture")
    print(f"  GET  /ping      - Check Sensor Pi connection")
    print(f"  GET  /status    - Get detailed status")
    print()
    print("=" * 60)
    print()
    
    # Validate config
    if not validate_config():
        print("WARNING: Config not fully set. Update config.py with correct IPs!")
        print()
    
    # Create directories
    os.makedirs(CSV_SAVE_DIR, exist_ok=True)
    
    # Try to connect to Sensor Pi
    print("Attempting to connect to Sensor Pi...")
    if sensor.connect():
        print("Connected to Sensor Pi!")
    else:
        print("Could not connect to Sensor Pi.")
        print("Make sure sensor_server.py is running on Sensor Pi!")
        print("Bridge will retry when requests come in.")
    print()
    
    # Start Flask
    print("Starting bridge server...")
    app.run(host=FLASK_HOST, port=BRIDGE_PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
