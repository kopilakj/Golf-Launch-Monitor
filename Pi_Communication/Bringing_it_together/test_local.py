#!/usr/bin/env python3
"""
Local Test Script
=================

Run this on your laptop/PC to test the protocol WITHOUT the Pis.
This simulates both Sensor Pi and Main Pi running on the same machine.

Usage:
    python test_local.py

This will:
1. Start a fake Sensor Pi server in a thread
2. Start a Main Pi client
3. Run through the full protocol flow
4. Report success/failure
"""

import os
import sys
import time
import socket
import threading
from io import StringIO

# Temporarily override config for local testing
# We'll use localhost instead of Pi IPs

class LocalConfig:
    """Override config for local testing."""
    SENSOR_PI_IP = "127.0.0.1"
    MAIN_PI_IP = "127.0.0.1"
    PI_COMM_PORT = 5001
    FLASK_HOST = "127.0.0.1"
    FLASK_PORT = 5000
    SHARED_TOKEN = "test-token-local"
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
    
    @staticmethod
    def validate_config():
        return True


# Patch the config module
sys.modules['config'] = LocalConfig

# Now import our modules (they'll use the patched config)
from protocol import (
    send_frame, recv_frame, validate_header, sha256_bytes, utc_now
)
from messages import (
    MSG_CLUB_PRESET, MSG_ACK_PRESET, MSG_TRIGGER, MSG_ACK_TRIGGER,
    MSG_CSV_META, MSG_CSV_CHUNK, MSG_CSV_DONE, MSG_PING, MSG_PONG,
    make_club_preset, make_ack_preset, make_trigger, make_ack_trigger,
    make_csv_meta, make_csv_chunk, make_csv_done, make_ping, make_pong,
    new_shot_id, new_request_id
)


# =============================================================================
# TEST RESULTS
# =============================================================================

class TestResults:
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.errors = []
    
    def passed(self, name):
        self.tests_run += 1
        self.tests_passed += 1
        print(f"  [PASS] {name}")
    
    def failed(self, name, error):
        self.tests_run += 1
        self.tests_failed += 1
        self.errors.append((name, error))
        print(f"  [FAIL] {name}: {error}")
    
    def summary(self):
        print()
        print("=" * 60)
        print(f"RESULTS: {self.tests_passed}/{self.tests_run} passed")
        if self.errors:
            print()
            print("Failures:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        print("=" * 60)
        return self.tests_failed == 0


results = TestResults()


# =============================================================================
# FAKE SENSOR PI SERVER
# =============================================================================

def fake_sensor_server(ready_event, stop_event):
    """Simulates Sensor Pi server for testing."""
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((LocalConfig.SENSOR_PI_IP, LocalConfig.PI_COMM_PORT))
    server.listen(1)
    server.settimeout(1.0)  # Allow checking stop_event
    
    print("[FakeSensor] Server started on 127.0.0.1:5001")
    ready_event.set()
    
    while not stop_event.is_set():
        try:
            conn, addr = server.accept()
            conn.settimeout(5.0)
            print(f"[FakeSensor] Client connected: {addr}")
            
            handle_fake_client(conn, stop_event)
            
        except socket.timeout:
            continue
        except Exception as e:
            if not stop_event.is_set():
                print(f"[FakeSensor] Error: {e}")
    
    server.close()
    print("[FakeSensor] Server stopped")


def handle_fake_client(conn, stop_event):
    """Handle messages from Main Pi."""
    
    current_club = "7iron"
    
    try:
        while not stop_event.is_set():
            try:
                header, payload = recv_frame(conn)
            except socket.timeout:
                continue
            except:
                break
            
            msg_type = header.get("msg_type")
            request_id = header.get("request_id")
            
            print(f"[FakeSensor] Received: {msg_type}")
            
            if msg_type == MSG_PING:
                pong = make_pong(request_id)
                send_frame(conn, pong)
                print("[FakeSensor] Sent PONG")
            
            elif msg_type == MSG_CLUB_PRESET:
                current_club = header.get("club_id", "7iron")
                preset_version = header.get("preset_version", 1)
                ack = make_ack_preset(current_club, preset_version, request_id)
                send_frame(conn, ack)
                print(f"[FakeSensor] Sent ACK_PRESET for {current_club}")
            
            elif msg_type == MSG_TRIGGER:
                shot_id = header["shot_id"]
                club_id = header["club_id"]
                
                # Send ACK_TRIGGER
                ack = make_ack_trigger(shot_id, club_id, request_id)
                send_frame(conn, ack)
                print("[FakeSensor] Sent ACK_TRIGGER")
                
                # Generate fake CSV
                csv_text = f"metric,value,unit\nball_speed,155.5,mph\nclub_id,{club_id},\nshot_id,{shot_id},\n"
                csv_bytes = csv_text.encode("utf-8")
                sha = sha256_bytes(csv_bytes)
                
                # Send CSV_META
                meta = make_csv_meta(
                    shot_id=shot_id,
                    club_id=club_id,
                    filename=f"{shot_id}.csv",
                    file_size=len(csv_bytes),
                    sha256=sha,
                    total_chunks=1,
                    request_id=request_id
                )
                send_frame(conn, meta)
                print("[FakeSensor] Sent CSV_META")
                
                # Send CSV_CHUNK
                chunk_header = make_csv_chunk(
                    shot_id=shot_id,
                    club_id=club_id,
                    chunk_index=0,
                    total_chunks=1,
                    request_id=request_id
                )
                send_frame(conn, chunk_header, payload=csv_bytes)
                print("[FakeSensor] Sent CSV_CHUNK")
                
                # Send CSV_DONE
                done = make_csv_done(
                    shot_id=shot_id,
                    club_id=club_id,
                    sha256=sha,
                    file_size=len(csv_bytes),
                    request_id=request_id
                )
                send_frame(conn, done)
                print("[FakeSensor] Sent CSV_DONE")
    
    except Exception as e:
        print(f"[FakeSensor] Client handler error: {e}")
    finally:
        conn.close()


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_connection():
    """Test basic connection."""
    print("\nTest: Basic Connection")
    
    try:
        sock = socket.create_connection(
            (LocalConfig.SENSOR_PI_IP, LocalConfig.PI_COMM_PORT),
            timeout=5
        )
        sock.close()
        results.passed("Connect to server")
    except Exception as e:
        results.failed("Connect to server", str(e))


def test_ping():
    """Test PING/PONG."""
    print("\nTest: PING/PONG")
    
    try:
        sock = socket.create_connection(
            (LocalConfig.SENSOR_PI_IP, LocalConfig.PI_COMM_PORT),
            timeout=5
        )
        sock.settimeout(5)
        
        # Send PING
        ping = make_ping()
        send_frame(sock, ping)
        
        # Receive PONG
        header, _ = recv_frame(sock)
        
        if header.get("msg_type") == MSG_PONG:
            results.passed("PING/PONG exchange")
        else:
            results.failed("PING/PONG exchange", f"Got {header.get('msg_type')}")
        
        sock.close()
    except Exception as e:
        results.failed("PING/PONG exchange", str(e))


def test_club_preset():
    """Test club preset flow."""
    print("\nTest: Club Preset")
    
    try:
        sock = socket.create_connection(
            (LocalConfig.SENSOR_PI_IP, LocalConfig.PI_COMM_PORT),
            timeout=5
        )
        sock.settimeout(5)
        
        # Send CLUB_PRESET
        preset = make_club_preset("driver")
        send_frame(sock, preset)
        
        # Receive ACK_PRESET
        header, _ = recv_frame(sock)
        
        if header.get("msg_type") == MSG_ACK_PRESET:
            if header.get("club_id") == "driver":
                results.passed("Club preset acknowledged")
            else:
                results.failed("Club preset acknowledged", "Wrong club_id")
        else:
            results.failed("Club preset acknowledged", f"Got {header.get('msg_type')}")
        
        sock.close()
    except Exception as e:
        results.failed("Club preset acknowledged", str(e))


def test_trigger_and_csv():
    """Test full trigger and CSV flow."""
    print("\nTest: Trigger and CSV Transfer")
    
    try:
        sock = socket.create_connection(
            (LocalConfig.SENSOR_PI_IP, LocalConfig.PI_COMM_PORT),
            timeout=10
        )
        sock.settimeout(10)
        
        shot_id = new_shot_id()
        club_id = "driver"
        
        # Send TRIGGER
        trigger = make_trigger(shot_id, club_id, 1)
        request_id = trigger["request_id"]
        send_frame(sock, trigger)
        
        # Receive ACK_TRIGGER
        header, _ = recv_frame(sock)
        if header.get("msg_type") != MSG_ACK_TRIGGER:
            results.failed("ACK_TRIGGER received", f"Got {header.get('msg_type')}")
            sock.close()
            return
        results.passed("ACK_TRIGGER received")
        
        # Receive CSV_META
        header, _ = recv_frame(sock)
        if header.get("msg_type") != MSG_CSV_META:
            results.failed("CSV_META received", f"Got {header.get('msg_type')}")
            sock.close()
            return
        results.passed("CSV_META received")
        
        file_size = header["file_size"]
        expected_sha = header["sha256"]
        total_chunks = header["total_chunks"]
        
        # Receive chunks
        received = bytearray()
        for i in range(total_chunks):
            header, payload = recv_frame(sock)
            if header.get("msg_type") != MSG_CSV_CHUNK:
                results.failed(f"CSV_CHUNK {i}", f"Got {header.get('msg_type')}")
                sock.close()
                return
            received.extend(payload)
        results.passed(f"CSV_CHUNK(s) received ({total_chunks} chunks)")
        
        # Receive CSV_DONE
        header, _ = recv_frame(sock)
        if header.get("msg_type") != MSG_CSV_DONE:
            results.failed("CSV_DONE received", f"Got {header.get('msg_type')}")
            sock.close()
            return
        results.passed("CSV_DONE received")
        
        # Verify checksum
        actual_sha = sha256_bytes(bytes(received))
        if actual_sha == expected_sha:
            results.passed("CSV checksum verified")
        else:
            results.failed("CSV checksum verified", "SHA256 mismatch")
        
        # Verify content
        csv_text = received.decode("utf-8")
        if "ball_speed" in csv_text:
            results.passed("CSV contains expected data")
        else:
            results.failed("CSV contains expected data", "Missing ball_speed")
        
        sock.close()
    except Exception as e:
        results.failed("Trigger and CSV flow", str(e))


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("=" * 60)
    print("LOCAL PROTOCOL TEST")
    print("=" * 60)
    print()
    print("This tests the protocol on localhost without needing Pis.")
    print()
    
    # Create output directory
    os.makedirs(LocalConfig.CSV_SAVE_DIR, exist_ok=True)
    
    # Start fake sensor server
    ready_event = threading.Event()
    stop_event = threading.Event()
    
    server_thread = threading.Thread(
        target=fake_sensor_server,
        args=(ready_event, stop_event)
    )
    server_thread.start()
    
    # Wait for server to be ready
    ready_event.wait(timeout=5)
    time.sleep(0.5)  # Extra delay for stability
    
    try:
        # Run tests
        test_connection()
        test_ping()
        test_club_preset()
        test_trigger_and_csv()
        
    finally:
        # Stop server
        stop_event.set()
        server_thread.join(timeout=3)
    
    # Print summary
    success = results.summary()
    
    if success:
        print()
        print("All tests passed! Protocol is working correctly.")
        print("Now test on your actual Pis with real IPs.")
    else:
        print()
        print("Some tests failed. Check the errors above.")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
