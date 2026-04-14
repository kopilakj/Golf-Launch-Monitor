#!/usr/bin/env python3
"""
Sensor Bridge - Main Pi Communication Service
==============================================

This runs on the MAIN PI alongside your partner's app.py.
It acts as the bridge between:
  - The Web GUI (app.py on port 5000)
  - The Sensor Pi (sensor_server.py on port 5002)
  - The motion detection camera (automatic swing triggers)

What it does:
1. Receives club selection from GUI via /set_club endpoint
2. Sends CLUB_PRESET to Sensor Pi
3. Monitors camera for swing motion (background thread)
4. When motion detected OR /trigger called: sends TRIGGER to Sensor Pi
5. Receives CSV data from Sensor Pi
6. Parses metrics and POSTs them to app.py:/api/upload_metrics
7. GUI displays the metrics

Run this with: python3 sensor_bridge.py

Endpoints:
  POST /set_club          - Set club (from GUI)
  POST /trigger           - Manual trigger shot capture
  POST /motion/enable     - Enable automatic motion detection
  POST /motion/disable    - Disable motion detection
  GET  /motion/status     - Get motion detection status
  GET  /status            - Get detailed system status
"""

import os
import sys
import json
import time
import socket
import requests
import traceback
from threading import Thread, Lock, Event
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

# Motion detection imports (Main Pi has camera)
try:
    import cv2
    import numpy as np
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("[Bridge] Warning: Camera libraries not available (cv2/picamera2)")
    print("[Bridge] Motion detection will be disabled")

# Import our protocol modules
from config import (
    SENSOR_PI_IP, PI_COMM_PORT, BRIDGE_PORT,
    FLASK_HOST, WEBAPP_METRICS_URL,
    CSV_SAVE_DIR, SOCKET_TIMEOUT, MAX_RETRIES, RECONNECT_DELAY,
    CLUB_PRESETS, DEFAULT_CLUB,
    normalize_club_name, PRESET_TO_GUI,
    validate_config,
    # Motion detection settings
    MOTION_CAMERA_SIZE, MOTION_TARGET_FPS, MOTION_EXPOSURE_US,
    MOTION_ANALOGUE_GAIN, MOTION_BUFFER_COUNT,
    MOTION_ROI, MOTION_PIXEL_DIFF_THRESH, MOTION_CHANGED_PIXELS_THRESH,
    MOTION_MIN_CONSECUTIVE, MOTION_WARMUP_SEC,
    # Ball-at-rest detection settings
    BALL_BRIGHTNESS_THRESHOLD, BALL_MIN_AREA_REST, BALL_MAX_AREA_REST,
    BALL_MIN_CIRCULARITY, BALL_CHECK_INTERVAL, BALL_READY_SECONDS,
)
from protocol import (
    send_frame, recv_frame, validate_header, validate_shot_header,
    sha256_bytes, utc_now,
    ProtocolError, ValidationError, ConnectionLostError
)
from messages import (
    MSG_CLUB_PRESET, MSG_ACK_PRESET, MSG_TRIGGER, MSG_ACK_TRIGGER,
    MSG_CSV_META, MSG_CSV_CHUNK, MSG_CSV_DONE, MSG_PING, MSG_PONG, MSG_ERROR,
    MSG_STROBE_ARM, MSG_STROBE_DISARM, MSG_ACK_STROBE,
    make_club_preset, make_trigger, make_ping,
    make_strobe_arm, make_strobe_disarm,
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
        
        # Motion detection state
        self.motion_enabled: bool = False
        self.motion_detecting: bool = False

        # Ball-at-rest state
        self.ball_state: str = "waiting"  # "waiting", "stabilizing", "ready"
        self.ball_position: Optional[Tuple[int, int]] = None
    
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
# MOTION DETECTION (Background Thread)
# =============================================================================

class MotionDetector:
    """
    Detects ball at rest + swing motion using the top Pi camera.

    State machine:
      "waiting"     -> Ball not found. Checking every BALL_CHECK_INTERVAL frames.
      "stabilizing" -> Ball found, counting consecutive detections.
      "ready"       -> Ball confirmed at rest. Motion detection armed.

    Swing detection only triggers when ball_state == "ready".
    After a swing, state resets to "waiting".
    """

    def __init__(self):
        self.camera = None
        self.running = False
        self.enabled = Event()
        self.stop_event = Event()
        self.thread: Optional[Thread] = None

        # Motion detection state
        self.prev_roi_frame = None
        self.consecutive_motion: int = 0
        self.warmup_start: float = 0
        self.is_warmed_up: bool = False

        # Ball-at-rest state
        self.ball_state: str = "waiting"
        self.ball_consecutive: int = 0
        self.ball_position: Optional[Tuple[int, int]] = None
        self.ball_frame_counter: int = 0
        self.ball_ready_count: int = int(
            BALL_READY_SECONDS / (BALL_CHECK_INTERVAL / MOTION_TARGET_FPS)
        )

        # Cooldown to prevent double-triggers
        self.last_trigger_time: float = 0
        self.trigger_cooldown_sec: float = 2.0

        # Trigger lock to prevent concurrent triggers
        self.trigger_lock = Lock()

    def setup_camera(self) -> bool:
        """Initialize the camera for motion detection."""
        if not CAMERA_AVAILABLE:
            print("[Motion] Camera libraries not available")
            return False

        try:
            print("[Motion] Setting up camera...")
            self.camera = Picamera2()

            config = self.camera.create_video_configuration(
                raw={"size": MOTION_CAMERA_SIZE, "format": "R8"},
                buffer_count=MOTION_BUFFER_COUNT,
            )
            self.camera.configure(config)
            self.camera.start()
            time.sleep(0.5)

            frame_us = int(1_000_000 / MOTION_TARGET_FPS)
            self.camera.set_controls({
                "FrameDurationLimits": (frame_us, frame_us),
                "ExposureTime": MOTION_EXPOSURE_US,
                "AnalogueGain": MOTION_ANALOGUE_GAIN,
            })

            print(f"[Motion] Camera ready: {MOTION_CAMERA_SIZE} @ {MOTION_TARGET_FPS}fps")
            print(f"[Motion] Exposure: {MOTION_EXPOSURE_US}us  ROI: {MOTION_ROI}")
            return True

        except Exception as e:
            print(f"[Motion] Camera setup failed: {e}")
            traceback.print_exc()
            return False

    def cleanup_camera(self):
        """Stop and close the camera."""
        if self.camera:
            try:
                self.camera.close()
            except:
                pass
            self.camera = None

    def reset_detection(self):
        """Reset all detection state for next shot."""
        self.prev_roi_frame = None
        self.consecutive_motion = 0
        self.warmup_start = time.time()
        self.is_warmed_up = False
        self.ball_state = "waiting"
        self.ball_consecutive = 0
        self.ball_position = None
        self.ball_frame_counter = 0
        state.ball_state = "waiting"
        state.ball_position = None

    # ---- Ball-at-rest detection ----

    @staticmethod
    def detect_ball_at_rest(frame) -> Optional[Tuple[int, int, int]]:
        """Search for a ball at rest inside the MOTION_ROI rectangle."""
        gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rx, ry, rw, rh = MOTION_ROI
        roi_crop = gray[ry:ry + rh, rx:rx + rw]

        best_candidate = None
        best_score = -1

        for thresh_val in [BALL_BRIGHTNESS_THRESHOLD,
                           BALL_BRIGHTNESS_THRESHOLD - 15,
                           BALL_BRIGHTNESS_THRESHOLD + 15]:
            if thresh_val < 10 or thresh_val > 240:
                continue
            _, thresh = cv2.threshold(roi_crop, thresh_val, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if not (BALL_MIN_AREA_REST < area < BALL_MAX_AREA_REST):
                    continue
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                if circularity < BALL_MIN_CIRCULARITY:
                    continue
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00']) + rx
                    cy = int(M['m01'] / M['m00']) + ry
                else:
                    continue
                score = circularity * 100 + area
                if score > best_score:
                    best_score = score
                    best_candidate = (cx, cy, int(area))

        return best_candidate

    def check_ball_at_rest(self, frame):
        """Run ball-at-rest detection periodically. Updates ball_state."""
        self.ball_frame_counter += 1
        if self.ball_frame_counter % BALL_CHECK_INTERVAL != 0:
            return

        result = self.detect_ball_at_rest(frame)
        prev_state = self.ball_state

        if result:
            self.ball_position = (result[0], result[1])
            self.ball_consecutive += 1

            if self.ball_state == "waiting":
                self.ball_state = "stabilizing"
                print(f"[Ball] Detected at ({result[0]}, {result[1]}) — stabilizing...")
                # Arm strobe as soon as ball is detected
                self._send_strobe_arm()

            if self.ball_consecutive >= self.ball_ready_count and self.ball_state != "ready":
                self.ball_state = "ready"
                print(f"[Ball] ======= READY FOR SWING =======")
        else:
            if self.ball_state != "waiting":
                print(f"[Ball] Lost — back to waiting (was {self.ball_state})")
                # Disarm strobe when ball is lost
                self._send_strobe_disarm()
            self.ball_state = "waiting"
            self.ball_consecutive = 0
            self.ball_position = None

        # Update shared state for API
        state.ball_state = self.ball_state
        state.ball_position = self.ball_position

    def _send_strobe_arm(self):
        """Tell bottom Pi to turn strobe on."""
        if not sensor.connected:
            return
        try:
            msg = make_strobe_arm()
            sensor.send(msg)
            # Read ACK but don't block the detection loop
            header, _ = sensor.recv()
            if header and header.get("msg_type") == MSG_ACK_STROBE:
                print("[Bridge] Strobe armed on Sensor Pi")
            else:
                print("[Bridge] Strobe arm — no ACK received")
        except Exception as e:
            print(f"[Bridge] Strobe arm failed: {e}")

    def _send_strobe_disarm(self):
        """Tell bottom Pi to turn strobe off."""
        if not sensor.connected:
            return
        try:
            msg = make_strobe_disarm()
            sensor.send(msg)
            header, _ = sensor.recv()
            if header and header.get("msg_type") == MSG_ACK_STROBE:
                print("[Bridge] Strobe disarmed on Sensor Pi")
            else:
                print("[Bridge] Strobe disarm — no ACK received")
        except Exception as e:
            print(f"[Bridge] Strobe disarm failed: {e}")

    # ---- Motion (swing) detection ----

    def check_frame(self, frame) -> Tuple[bool, int]:
        """
        Check a frame for swing motion.
        Returns (triggered, changed_pixels).
        """
        rx, ry, rw, rh = MOTION_ROI
        roi_frame = frame[ry:ry+rh, rx:rx+rw]

        if not self.is_warmed_up:
            if time.time() - self.warmup_start < MOTION_WARMUP_SEC:
                self.prev_roi_frame = roi_frame.copy()
                return False, 0
            else:
                self.is_warmed_up = True
                print("[Motion] Warmup complete")

        if self.prev_roi_frame is None:
            self.prev_roi_frame = roi_frame.copy()
            return False, 0

        diff = cv2.absdiff(roi_frame, self.prev_roi_frame)
        _, diff_bin = cv2.threshold(diff, MOTION_PIXEL_DIFF_THRESH, 255, cv2.THRESH_BINARY)
        changed_pixels = int(np.count_nonzero(diff_bin))

        if changed_pixels > MOTION_CHANGED_PIXELS_THRESH:
            self.consecutive_motion += 1
        else:
            self.consecutive_motion = 0

        self.prev_roi_frame = roi_frame.copy()

        return self.consecutive_motion >= MOTION_MIN_CONSECUTIVE, changed_pixels

    def on_motion_detected(self, changed_pixels: int):
        """Called when swing is detected - triggers shot on bottom Pi."""
        now = time.time()
        if now - self.last_trigger_time < self.trigger_cooldown_sec:
            print(f"[Motion] Cooldown active, ignoring trigger")
            return

        if not self.trigger_lock.acquire(blocking=False):
            print(f"[Motion] Trigger already in progress, ignoring")
            return

        try:
            self.last_trigger_time = now

            print(f"[Motion] ========================================")
            print(f"[Motion] SWING DETECTED! (changed={changed_pixels})")
            print(f"[Motion] Triggering shot on Sensor Pi...")
            print(f"[Motion] ========================================")

            result = trigger_shot()

            if result.get("success"):
                print(f"[Motion] Shot captured successfully!")
                print(f"[Motion] Shot ID: {result.get('shot_id')}")
            else:
                print(f"[Motion] Shot capture failed: {result.get('error')}")

        finally:
            self.trigger_lock.release()
            # Disarm strobe after shot, then reset to wait for next ball
            self._send_strobe_disarm()
            self.reset_detection()

    # ---- Main loop ----

    def detection_loop(self):
        """Main detection loop - runs in background thread."""
        print("[Motion] Detection thread started")

        while not self.stop_event.is_set():
            if not self.enabled.is_set():
                time.sleep(0.1)
                continue

            if self.camera is None:
                if not self.setup_camera():
                    print("[Motion] Camera setup failed, disabling")
                    self.enabled.clear()
                    state.motion_enabled = False
                    continue
                self.reset_detection()

            try:
                req = self.camera.capture_request()
                raw = req.make_array("raw")
                req.release()

                # Always check ball state
                self.check_ball_at_rest(raw)

                # Only check for swings when ball is confirmed at rest
                if self.ball_state == "ready":
                    triggered, changed = self.check_frame(raw)
                    if triggered:
                        self.on_motion_detected(changed)

            except Exception as e:
                print(f"[Motion] Error in detection loop: {e}")
                time.sleep(0.1)

        self.cleanup_camera()
        print("[Motion] Detection thread stopped")

    def start(self) -> bool:
        """Start the motion detection background thread."""
        if not CAMERA_AVAILABLE:
            print("[Motion] Cannot start - camera libraries not available")
            return False

        if self.thread and self.thread.is_alive():
            print("[Motion] Already running")
            return True

        self.stop_event.clear()
        self.thread = Thread(target=self.detection_loop, daemon=True)
        self.thread.start()
        self.running = True
        print("[Motion] Background thread started")
        return True

    def stop(self):
        """Stop the motion detection thread."""
        self.stop_event.set()
        self.enabled.clear()
        if self.thread:
            self.thread.join(timeout=2.0)
        self.running = False
        print("[Motion] Stopped")

    def enable(self) -> bool:
        """Enable motion detection."""
        if not CAMERA_AVAILABLE:
            print("[Motion] Cannot enable - camera libraries not available")
            return False

        if not self.running:
            self.start()

        self.enabled.set()
        state.motion_enabled = True
        state.motion_detecting = True
        print("[Motion] Enabled — watching for ball + swings")
        return True

    def disable(self):
        """Disable motion detection (keeps thread running)."""
        self.enabled.clear()
        state.motion_enabled = False
        state.motion_detecting = False
        self.ball_state = "waiting"
        state.ball_state = "waiting"
        state.ball_position = None
        print("[Motion] Disabled")

    def is_enabled(self) -> bool:
        return self.enabled.is_set()


# Global motion detector instance
motion_detector = MotionDetector()


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
        "last_error": state.last_error,
        # Motion detection status
        "camera_available": CAMERA_AVAILABLE,
        "motion_enabled": motion_detector.is_enabled() if CAMERA_AVAILABLE else False,
        "motion_thread_running": motion_detector.running if CAMERA_AVAILABLE else False,
        "ball_state": state.ball_state,
        "ball_position": state.ball_position
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
# MOTION DETECTION ENDPOINTS
# =============================================================================

@app.route("/motion/enable", methods=["POST"])
def motion_enable():
    """Enable automatic motion detection for swing triggers."""
    if not CAMERA_AVAILABLE:
        return jsonify({
            "success": False,
            "error": "Camera not available on this system"
        }), 503
    
    success = motion_detector.enable()
    return jsonify({
        "success": success,
        "motion_enabled": motion_detector.is_enabled(),
        "message": "Motion detection enabled - watching for swings" if success else "Failed to enable"
    })


@app.route("/motion/disable", methods=["POST"])
def motion_disable():
    """Disable automatic motion detection."""
    motion_detector.disable()
    return jsonify({
        "success": True,
        "motion_enabled": False,
        "message": "Motion detection disabled"
    })


@app.route("/motion/status", methods=["GET"])
def motion_status():
    """Get motion detection status including ball-at-rest state."""
    return jsonify({
        "camera_available": CAMERA_AVAILABLE,
        "motion_enabled": motion_detector.is_enabled(),
        "thread_running": motion_detector.running,
        "ball_state": state.ball_state,
        "ball_position": state.ball_position,
        "shots_triggered": state.shots_triggered,
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
    print(f"  POST /set_club       - Set club (from GUI)")
    print(f"  GET  /get_club       - Get current club")
    print(f"  POST /trigger        - Manual trigger shot capture")
    print(f"  GET  /ping           - Check Sensor Pi connection")
    print(f"  GET  /status         - Get detailed status")
    print()
    print("Motion Detection:")
    print(f"  POST /motion/enable  - Enable swing detection")
    print(f"  POST /motion/disable - Disable swing detection")
    print(f"  GET  /motion/status  - Get motion detection status")
    print()
    print(f"Camera available: {CAMERA_AVAILABLE}")
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
    
    # Start motion detection thread (but don't enable yet)
    if CAMERA_AVAILABLE:
        print("Starting motion detection thread...")
        motion_detector.start()
        print("Motion detection ready (disabled by default)")
        print("Call POST /motion/enable to start detecting swings")
    else:
        print("Motion detection not available (camera libraries missing)")
    print()
    
    # Start Flask
    print("Starting bridge server...")
    app.run(host=FLASK_HOST, port=BRIDGE_PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
