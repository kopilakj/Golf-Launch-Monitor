#!/usr/bin/env python3
"""
Integration Test Script
=======================

This script tests the full integration on localhost:
1. Starts a fake Sensor Pi server
2. Starts the sensor_bridge (Main Pi bridge)
3. Simulates what app.py does (club selection, triggering)
4. Verifies the complete flow

Run this BEFORE testing on actual Pis to verify the protocol works.

Usage:
    python test_integration.py
"""

import os
import sys
import time
import json
import math
import socket
import threading
import requests
from io import StringIO

# Override config for localhost testing
class LocalTestConfig:
    SENSOR_PI_IP = "127.0.0.1"
    MAIN_PI_IP = "127.0.0.1"
    PI_COMM_PORT = 5002           # Sensor Pi port
    BRIDGE_PORT = 5001            # Bridge port
    FLASK_HOST = "127.0.0.1"
    FLASK_PORT = 5000
    WEBAPP_METRICS_URL = "http://127.0.0.1:5000/api/upload_metrics"
    SHARED_TOKEN = "golf-launch-monitor-2024"
    PROTOCOL_VERSION = 1
    CHUNK_SIZE = 16 * 1024
    SOCKET_TIMEOUT = 10
    RECONNECT_DELAY = 1
    MAX_RETRIES = 3
    CSV_SAVE_DIR = "./test_output"
    CSV_SOURCE_DIR = "./test_output"
    
    CLUB_PRESETS = {
        "driver": {"preset_version": 1, "exposure": 100},
        "7iron": {"preset_version": 1, "exposure": 130},
    }
    DEFAULT_CLUB = "7iron"
    
    GUI_TO_PRESET = {
        "Driver": "driver",
        "7 Iron": "7iron",
    }
    PRESET_TO_GUI = {"driver": "Driver", "7iron": "7 Iron"}
    
    @staticmethod
    def normalize_club_name(name):
        if name in LocalTestConfig.GUI_TO_PRESET:
            return LocalTestConfig.GUI_TO_PRESET[name]
        if name in LocalTestConfig.CLUB_PRESETS:
            return name
        return LocalTestConfig.DEFAULT_CLUB
    
    @staticmethod
    def validate_config():
        return True


# Patch config module
sys.modules['config'] = LocalTestConfig

# Now import protocol (uses patched config)
from protocol import send_frame, recv_frame, validate_header, sha256_bytes, utc_now, SHARED_TOKEN
from messages import (
    MSG_CLUB_PRESET, MSG_ACK_PRESET, MSG_TRIGGER, MSG_ACK_TRIGGER,
    MSG_CSV_META, MSG_CSV_CHUNK, MSG_CSV_DONE, MSG_PING, MSG_PONG,
    make_ack_preset, make_ack_trigger, make_csv_meta, make_csv_chunk,
    make_csv_done, make_pong
)


# =============================================================================
# TEST RESULTS
# =============================================================================

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def ok(self, msg):
        self.passed += 1
        print(f"  [PASS] {msg}")
    
    def fail(self, msg, error=""):
        self.failed += 1
        self.errors.append((msg, error))
        print(f"  [FAIL] {msg}: {error}")


results = TestResults()


# =============================================================================
# FAKE SENSOR PI SERVER
# =============================================================================

def fake_sensor_server(ready_event, stop_event, received_messages):
    """Simulates Sensor Pi for testing."""
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", LocalTestConfig.PI_COMM_PORT))
    server.listen(1)
    server.settimeout(1.0)
    
    print(f"[FakeSensor] Started on port {LocalTestConfig.PI_COMM_PORT}")
    ready_event.set()
    
    current_club = "7iron"
    
    while not stop_event.is_set():
        try:
            conn, addr = server.accept()
            conn.settimeout(5.0)
            
            while not stop_event.is_set():
                try:
                    header, payload = recv_frame(conn)
                except socket.timeout:
                    continue
                except:
                    break
                
                msg_type = header.get("msg_type")
                request_id = header.get("request_id", "unknown")
                received_messages.append(header)
                
                print(f"[FakeSensor] Received: {msg_type}")
                
                if msg_type == MSG_PING:
                    send_frame(conn, make_pong(request_id))
                
                elif msg_type == MSG_CLUB_PRESET:
                    current_club = header.get("club_id", "7iron")
                    preset_version = header.get("preset_version", 1)
                    send_frame(conn, make_ack_preset(current_club, preset_version, request_id))
                    print(f"[FakeSensor] Preset set to {current_club}")
                
                elif msg_type == MSG_TRIGGER:
                    shot_id = header["shot_id"]
                    club_id = header["club_id"]
                    
                    # ACK
                    send_frame(conn, make_ack_trigger(shot_id, club_id, request_id))
                    
                    # Generate CSV
                    csv_text = f"metric,value,unit\nball_speed,155.5,mph\nlaunch_angle,12.3,degrees\ncarry_distance,250,yards\nclub_id,{club_id},\n"
                    csv_bytes = csv_text.encode()
                    sha = sha256_bytes(csv_bytes)
                    
                    # Send CSV
                    send_frame(conn, make_csv_meta(shot_id, club_id, f"{shot_id}.csv", len(csv_bytes), sha, 1, request_id))
                    send_frame(conn, make_csv_chunk(shot_id, club_id, 0, 1, request_id), payload=csv_bytes)
                    send_frame(conn, make_csv_done(shot_id, club_id, sha, len(csv_bytes), request_id))
                    print(f"[FakeSensor] Shot {shot_id} data sent")
            
            conn.close()
        except socket.timeout:
            continue
        except Exception as e:
            if not stop_event.is_set():
                print(f"[FakeSensor] Error: {e}")
    
    server.close()


# =============================================================================
# FAKE WEBAPP (simulates app.py)
# =============================================================================

received_metrics = []

def fake_webapp(ready_event, stop_event):
    """Simulates app.py for testing."""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    @app.route("/api/upload_metrics", methods=["POST"])
    def upload():
        data = request.get_json()
        received_metrics.append(data)
        print(f"[FakeWebapp] Received metrics: {data}")
        return jsonify({"status": "success"})
    
    @app.route("/")
    def index():
        return "Fake webapp running"
    
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    ready_event.set()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False)


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_bridge_club_selection():
    """Test club selection via bridge."""
    print("\nTest: Club Selection via Bridge")
    
    try:
        # Call bridge /set_club
        response = requests.post(
            "http://127.0.0.1:5001/set_club",
            json={"club": "Driver"},
            timeout=5
        )
        
        data = response.json()
        
        if data.get("success"):
            results.ok("Club selection sent to Sensor Pi")
        else:
            results.fail("Club selection", data.get("error", "Unknown error"))
        
        if data.get("preset") == "driver":
            results.ok("Club name normalized correctly")
        else:
            results.fail("Club name normalization", f"Got {data.get('preset')}")
            
    except Exception as e:
        results.fail("Club selection", str(e))


def test_bridge_trigger():
    """Test triggering a shot via bridge."""
    print("\nTest: Trigger Shot via Bridge")
    
    try:
        # First set a club
        requests.post("http://127.0.0.1:5001/set_club", json={"club": "7 Iron"}, timeout=5)
        
        # Trigger shot
        response = requests.post(
            "http://127.0.0.1:5001/trigger",
            timeout=15
        )
        
        data = response.json()
        
        if data.get("success"):
            results.ok("Shot triggered and data received")
        else:
            results.fail("Shot trigger", data.get("error", "Unknown error"))
        
        if "metrics" in data:
            metrics = data["metrics"]
            if "ball_speed" in metrics:
                results.ok(f"Metrics parsed correctly (ball_speed={metrics['ball_speed']})")
            else:
                results.fail("Metrics parsing", "Missing ball_speed")
        
        if data.get("webapp_notified"):
            results.ok("Webapp notified of new metrics")
        else:
            results.fail("Webapp notification", "Not notified")
            
    except Exception as e:
        results.fail("Shot trigger", str(e))


def test_webapp_received_metrics():
    """Check if webapp received the metrics."""
    print("\nTest: Webapp Received Metrics")
    
    time.sleep(0.5)  # Give time for async POST
    
    if len(received_metrics) > 0:
        results.ok(f"Webapp received {len(received_metrics)} metric update(s)")
        
        last = received_metrics[-1]
        if "ball_speed" in last:
            results.ok(f"Metrics contain ball_speed: {last['ball_speed']}")
        else:
            results.fail("Metrics format", "Missing ball_speed in webapp data")
    else:
        results.fail("Webapp metrics", "No metrics received by webapp")


def test_bridge_status():
    """Test bridge status endpoint."""
    print("\nTest: Bridge Status")
    
    try:
        response = requests.get("http://127.0.0.1:5001/status", timeout=5)
        data = response.json()
        
        if data.get("sensor_connected"):
            results.ok("Bridge reports sensor connected")
        else:
            results.fail("Sensor connection", "Bridge says sensor not connected")
        
        if data.get("shots_received", 0) > 0:
            results.ok(f"Bridge tracked {data['shots_received']} shot(s)")
        
    except Exception as e:
        results.fail("Bridge status", str(e))


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("=" * 60)
    print("INTEGRATION TEST")
    print("=" * 60)
    print()
    print("This tests the full flow on localhost.")
    print()
    
    os.makedirs("./test_output", exist_ok=True)
    
    # Start fake sensor
    sensor_ready = threading.Event()
    sensor_stop = threading.Event()
    sensor_messages = []
    
    sensor_thread = threading.Thread(
        target=fake_sensor_server,
        args=(sensor_ready, sensor_stop, sensor_messages)
    )
    sensor_thread.daemon = True
    sensor_thread.start()
    sensor_ready.wait(timeout=5)
    
    # Start fake webapp
    webapp_ready = threading.Event()
    webapp_stop = threading.Event()
    
    webapp_thread = threading.Thread(
        target=fake_webapp,
        args=(webapp_ready, webapp_stop)
    )
    webapp_thread.daemon = True
    webapp_thread.start()
    webapp_ready.wait(timeout=5)
    
    time.sleep(1)  # Let servers stabilize
    
    # Start bridge (import here after config patched)
    print("\nStarting sensor_bridge...")
    
    # We need to start the bridge in a subprocess or thread
    # For simplicity, let's import and run it
    import subprocess
    bridge_proc = subprocess.Popen(
        [sys.executable, "sensor_bridge.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    time.sleep(3)  # Wait for bridge to start
    
    try:
        # Check bridge is running
        try:
            resp = requests.get("http://127.0.0.1:5001/", timeout=2)
            print("[Test] Bridge is running!")
        except:
            print("[Test] ERROR: Bridge not responding. Check for errors above.")
            bridge_proc.terminate()
            return 1
        
        # Run tests
        test_bridge_club_selection()
        test_bridge_trigger()
        test_webapp_received_metrics()
        test_bridge_status()
        
    finally:
        # Cleanup
        bridge_proc.terminate()
        sensor_stop.set()
        webapp_stop.set()
    
    # Summary
    print()
    print("=" * 60)
    print(f"RESULTS: {results.passed} passed, {results.failed} failed")
    if results.errors:
        print()
        print("Failures:")
        for name, err in results.errors:
            print(f"  - {name}: {err}")
    print("=" * 60)
    
    if results.failed == 0:
        print()
        print("All tests passed! Ready to test on actual Pis.")
        return 0
    else:
        print()
        print("Some tests failed. Fix issues before deploying.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
