#!/usr/bin/env python3
"""
Sensor Pi Server
================

This runs on the SENSOR PI (the one with cameras/sensors).
It listens for connections from Main Pi and:
1. Receives CLUB_PRESET messages -> applies camera settings
2. Receives TRIGGER messages -> captures shot data using rpicam-vid
3. Sends CSV data back to Main Pi

Run this with: python3 sensor_server.py
"""

import os
import sys
import math
import socket
import traceback
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import cv2

# Import our modules
from config import (
    SENSOR_PI_IP, PI_COMM_PORT, CHUNK_SIZE,
    CSV_SOURCE_DIR, CLUB_PRESETS, DEFAULT_CLUB,
    CAPTURE_OUTPUT_DIR, CAPTURE_WIDTH, CAPTURE_HEIGHT,
    CAPTURE_FPS, CAPTURE_DURATION_MS,
    validate_config
)
from protocol import (
    send_frame, recv_frame, validate_header, validate_shot_header,
    sha256_bytes, utc_now, format_header_for_log,
    ProtocolError, ValidationError, ConnectionLostError
)
from messages import (
    MSG_CLUB_PRESET, MSG_ACK_PRESET, MSG_TRIGGER, MSG_ACK_TRIGGER,
    MSG_CSV_META, MSG_CSV_CHUNK, MSG_CSV_DONE, MSG_PING, MSG_ERROR,
    make_ack_preset, make_ack_trigger, make_csv_meta, make_csv_chunk,
    make_csv_done, make_error, make_pong
)


# =============================================================================
# GLOBAL STATE
# =============================================================================

class SensorState:
    """Tracks current sensor state."""
    def __init__(self):
        self.current_club_id: Optional[str] = None
        self.current_preset_version: int = 0
        self.current_preset_data: Dict[str, Any] = {}
        self.shots_processed: int = 0
        
        # Capture settings (can be overridden by preset)
        self.capture_width: int = CAPTURE_WIDTH
        self.capture_height: int = CAPTURE_HEIGHT
        self.capture_fps: int = CAPTURE_FPS
        self.capture_duration_ms: int = CAPTURE_DURATION_MS
    
    def apply_preset(self, club_id: str, preset_data: Dict[str, Any], version: int):
        """Apply a club preset (configure camera settings)."""
        self.current_club_id = club_id
        self.current_preset_version = version
        self.current_preset_data = preset_data
        
        # Apply capture settings from preset if available
        self.capture_duration_ms = preset_data.get('capture_duration_ms', CAPTURE_DURATION_MS)
        
        print(f"[Sensor] Applying preset for {club_id}:")
        print(f"         Version: {version}")
        print(f"         Capture duration: {self.capture_duration_ms}ms")
        print(f"         Resolution: {self.capture_width}x{self.capture_height} @ {self.capture_fps}fps")
    
    def run_capture(self, shot_id: str) -> Path:
        """
        Run rpicam-vid to capture high-speed video.
        
        Returns:
            Path to the output directory containing captured frames
        """
        # Create output directory for this shot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(CAPTURE_OUTPUT_DIR) / f"{shot_id}_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        pts_path = out_dir / "capture.pts"
        yuv_path = out_dir / "capture.yuv"
        
        # Build rpicam-vid command
        cmd = [
            "rpicam-vid",
            "--width", str(self.capture_width),
            "--height", str(self.capture_height),
            "--framerate", str(self.capture_fps),
            "--codec", "yuv420",
            "--denoise", "off",
            "--nopreview",
            "--save-pts", str(pts_path),
            "-t", str(self.capture_duration_ms),
            "-o", str(yuv_path),
        ]
        
        print(f"[Sensor] Running capture: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"[Sensor] Capture complete: {yuv_path}")
        except subprocess.CalledProcessError as e:
            print(f"[Sensor] Capture failed: {e.stderr}")
            raise RuntimeError(f"rpicam-vid failed: {e.stderr}")
        
        return out_dir
    
    def extract_frames(self, out_dir: Path) -> int:
        """
        Extract Y-channel frames from YUV420 capture.
        
        Returns:
            Number of frames extracted
        """
        frame_bytes = self.capture_width * self.capture_height * 3 // 2  # YUV420
        yuv_path = out_dir / "capture.yuv"
        
        if not yuv_path.exists():
            print(f"[Sensor] YUV file not found: {yuv_path}")
            return 0
        
        data = yuv_path.read_bytes()
        num_frames = len(data) // frame_bytes
        
        print(f"[Sensor] Extracting {num_frames} frames from {yuv_path}")
        
        for i in range(num_frames):
            start = i * frame_bytes
            y_plane = np.frombuffer(
                data[start:start + self.capture_width * self.capture_height],
                dtype=np.uint8
            ).reshape(self.capture_height, self.capture_width)
            
            frame_path = out_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(frame_path), y_plane)
        
        print(f"[Sensor] Extracted {num_frames} frames to {out_dir}")
        return num_frames
    
    def capture_shot(self, shot_id: str, club_id: str) -> bytes:
        """
        Capture and process a shot using rpicam-vid.
        
        This:
        1. Runs rpicam-vid to capture high-speed video
        2. Extracts frames from YUV data
        3. Returns placeholder metrics as CSV (real processing TBD)
        """
        self.shots_processed += 1
        
        print(f"[Sensor] Capturing shot {shot_id} with {club_id}")
        print(f"         Using preset version {self.current_preset_version}")
        
        # Run the capture
        try:
            out_dir = self.run_capture(shot_id)
            num_frames = self.extract_frames(out_dir)
        except Exception as e:
            print(f"[Sensor] Capture error: {e}")
            # Return error metrics
            csv_lines = [
                "metric,value,unit",
                f"error,{str(e)},",
                f"club_id,{club_id},",
                f"shot_id,{shot_id},",
            ]
            return ("\n".join(csv_lines) + "\n").encode("utf-8")
        
        # Generate placeholder metrics
        # TODO: Replace with actual ball tracking / analysis
        csv_lines = [
            "metric,value,unit",
            "ball_speed,152.3,mph",
            "launch_angle,12.5,degrees",
            "spin_rate,2800,rpm",
            "spin_axis,3.2,degrees",
            "carry_distance,245,yards",
            "total_distance,267,yards",
            f"frames_captured,{num_frames},",
            f"capture_fps,{self.capture_fps},",
            f"capture_duration_ms,{self.capture_duration_ms},",
            f"club_id,{club_id},",
            f"shot_id,{shot_id},",
            f"preset_version,{self.current_preset_version},",
            f"output_dir,{out_dir},",
        ]
        
        csv_text = "\n".join(csv_lines) + "\n"
        
        # Also save CSV to the capture directory
        csv_path = out_dir / "metrics.csv"
        csv_path.write_text(csv_text)
        print(f"[Sensor] Metrics saved to {csv_path}")
        
        return csv_text.encode("utf-8")


# Global state instance
state = SensorState()


# =============================================================================
# MESSAGE HANDLERS
# =============================================================================

def handle_club_preset(conn: socket.socket, header: Dict[str, Any]) -> None:
    """Handle CLUB_PRESET message from Main Pi."""
    club_id = header.get("club_id", DEFAULT_CLUB)
    preset_version = header.get("preset_version", 1)
    preset_data = header.get("preset_data", {})
    request_id = header["request_id"]
    
    print(f"[Sensor] Received CLUB_PRESET for {club_id}")
    
    # Apply the preset
    state.apply_preset(club_id, preset_data, preset_version)
    
    # Send acknowledgment
    ack = make_ack_preset(club_id, preset_version, request_id)
    send_frame(conn, ack)
    print(f"[Sensor] Sent ACK_PRESET")


def handle_trigger(conn: socket.socket, header: Dict[str, Any]) -> None:
    """Handle TRIGGER message from Main Pi."""
    request_id = header["request_id"]
    shot_id = header["shot_id"]
    club_id = header["club_id"]  # This is authoritative!
    preset_version = header.get("preset_version", state.current_preset_version)
    
    print(f"[Sensor] Received TRIGGER:")
    print(f"         shot_id: {shot_id}")
    print(f"         club_id: {club_id}")
    
    # Send immediate acknowledgment
    ack = make_ack_trigger(shot_id, club_id, request_id)
    send_frame(conn, ack)
    print(f"[Sensor] Sent ACK_TRIGGER")
    
    # Capture and process the shot
    try:
        csv_bytes = state.capture_shot(shot_id, club_id)
    except Exception as e:
        print(f"[Sensor] ERROR capturing shot: {e}")
        error_msg = make_error(f"Capture failed: {e}", request_id, shot_id)
        send_frame(conn, error_msg)
        return
    
    # Send CSV data
    send_csv_data(conn, shot_id, club_id, csv_bytes, request_id)


def send_csv_data(
    conn: socket.socket,
    shot_id: str,
    club_id: str,
    csv_bytes: bytes,
    request_id: str
) -> None:
    """Send CSV data to Main Pi in chunks."""
    
    # Calculate metadata
    digest = sha256_bytes(csv_bytes)
    total_chunks = math.ceil(len(csv_bytes) / CHUNK_SIZE) if csv_bytes else 1
    filename = f"{shot_id}.csv"
    
    print(f"[Sensor] Sending CSV: {len(csv_bytes)} bytes, {total_chunks} chunks")
    
    # Send CSV_META
    meta = make_csv_meta(
        shot_id=shot_id,
        club_id=club_id,
        filename=filename,
        file_size=len(csv_bytes),
        sha256=digest,
        total_chunks=total_chunks,
        request_id=request_id
    )
    send_frame(conn, meta)
    print(f"[Sensor] Sent CSV_META")
    
    # Send chunks
    for idx in range(total_chunks):
        start = idx * CHUNK_SIZE
        end = start + CHUNK_SIZE
        chunk = csv_bytes[start:end]
        
        chunk_header = make_csv_chunk(
            shot_id=shot_id,
            club_id=club_id,
            chunk_index=idx,
            total_chunks=total_chunks,
            request_id=request_id
        )
        send_frame(conn, chunk_header, payload=chunk)
        print(f"[Sensor] Sent chunk {idx + 1}/{total_chunks}")
    
    # Send CSV_DONE
    done = make_csv_done(
        shot_id=shot_id,
        club_id=club_id,
        sha256=digest,
        file_size=len(csv_bytes),
        request_id=request_id
    )
    send_frame(conn, done)
    print(f"[Sensor] Sent CSV_DONE")


def handle_ping(conn: socket.socket, header: Dict[str, Any]) -> None:
    """Handle PING health check."""
    pong = make_pong(header["request_id"])
    send_frame(conn, pong)
    print("[Sensor] Responded to PING")


# =============================================================================
# CONNECTION HANDLER
# =============================================================================

def handle_client(conn: socket.socket, addr: tuple) -> None:
    """Handle a connected client (Main Pi)."""
    print(f"[Sensor] Client connected: {addr}")
    
    try:
        while True:
            # Wait for message
            header, payload = recv_frame(conn)
            
            # Validate
            try:
                validate_header(header)
            except ValidationError as e:
                print(f"[Sensor] Validation error: {e}")
                error_msg = make_error(str(e), header.get("request_id"))
                send_frame(conn, error_msg)
                continue
            
            msg_type = header.get("msg_type")
            print(f"[Sensor] Received: {msg_type}")
            
            # Route to handler
            if msg_type == MSG_CLUB_PRESET:
                handle_club_preset(conn, header)
            
            elif msg_type == MSG_TRIGGER:
                validate_shot_header(header)
                handle_trigger(conn, header)
            
            elif msg_type == MSG_PING:
                handle_ping(conn, header)
            
            else:
                print(f"[Sensor] Unknown message type: {msg_type}")
                error_msg = make_error(
                    f"Unknown message type: {msg_type}",
                    header.get("request_id")
                )
                send_frame(conn, error_msg)
    
    except ConnectionLostError:
        print(f"[Sensor] Client disconnected: {addr}")
    except Exception as e:
        print(f"[Sensor] Error handling client: {e}")
        traceback.print_exc()


# =============================================================================
# SERVER MAIN
# =============================================================================

def run_server():
    """Start the sensor server."""
    
    # Validate config
    if not validate_config():
        sys.exit(1)
    
    # Create data directory
    os.makedirs(CSV_SOURCE_DIR, exist_ok=True)
    
    print("=" * 60)
    print("SENSOR PI SERVER")
    print("=" * 60)
    print(f"Listening on: {SENSOR_PI_IP}:{PI_COMM_PORT}")
    print(f"Protocol version: 1")
    print("=" * 60)
    print()
    print("Waiting for Main Pi to connect...")
    print()
    
    # Create server socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server.bind((SENSOR_PI_IP, PI_COMM_PORT))
        except OSError as e:
            print(f"ERROR: Cannot bind to {SENSOR_PI_IP}:{PI_COMM_PORT}")
            print(f"       {e}")
            print()
            print("Possible fixes:")
            print("1. Check that SENSOR_PI_IP in config.py matches this Pi's IP")
            print("2. Run: hostname -I to see this Pi's IP addresses")
            print("3. Make sure no other program is using port", PI_COMM_PORT)
            sys.exit(1)
        
        server.listen(1)
        
        # Accept connections forever
        while True:
            try:
                conn, addr = server.accept()
                with conn:
                    conn.settimeout(60)  # 60 second timeout
                    handle_client(conn, addr)
            except KeyboardInterrupt:
                print("\n[Sensor] Shutting down...")
                break
            except Exception as e:
                print(f"[Sensor] Server error: {e}")
                traceback.print_exc()
                time.sleep(1)  # Brief pause before accepting next connection


if __name__ == "__main__":
    run_server()
