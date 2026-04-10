#!/usr/bin/env python3
"""
Sensor Pi Server
================

This runs on the SENSOR PI (the one with cameras/sensors).
It listens for connections from Main Pi and:
1. Receives CLUB_PRESET messages -> applies camera settings
2. Runs continuous circular buffer capture in background
3. Receives TRIGGER messages -> saves pre/post trigger frames
4. Runs ball detection and metric calculation
5. Sends CSV data with real metrics back to Main Pi

Run this with: python3 sensor_server.py
"""

import os
import sys
import math
import socket
import traceback
import time
import csv as csv_module
from pathlib import Path
from datetime import datetime
from threading import Thread, Lock, Event, Timer
from collections import deque
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import cv2
from picamera2 import Picamera2
from smbus2 import SMBus, i2c_msg

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
# CIRCULAR BUFFER CAPTURE SETTINGS
# =============================================================================

PRE_TRIGGER_FRAMES = 15      # Frames to save before trigger
POST_TRIGGER_FRAMES = 20     # Frames to save after trigger
BUFFER_SIZE = PRE_TRIGGER_FRAMES + 10  # Circular buffer size

# Camera settings
CAMERA_SIZE = (CAPTURE_WIDTH, CAPTURE_HEIGHT)
TARGET_FPS = CAPTURE_FPS
EXPOSURE_US = 500
ANALOGUE_GAIN = 4.0
BUFFER_COUNT = 4


# =============================================================================
# METRIC CALCULATION CONSTANTS (from metric_calculation.py)
# =============================================================================

# Camera calibration
CAMERA_DISTANCE_M = 0.965          # Distance from camera to ball (meters)
HORIZONTAL_FOV_DEG = 70.0          # Horizontal field of view (degrees)
FRAME_RATE_FPS = CAPTURE_FPS       # Capture rate

# Derived calibration values
HORIZONTAL_FOV_RAD = np.radians(HORIZONTAL_FOV_DEG)
FIELD_WIDTH_M = 2 * CAMERA_DISTANCE_M * np.tan(HORIZONTAL_FOV_RAD / 2)
PIXELS_PER_METER = CAPTURE_WIDTH / FIELD_WIDTH_M
METERS_PER_PIXEL = FIELD_WIDTH_M / CAPTURE_WIDTH
SECONDS_PER_FRAME = 1.0 / FRAME_RATE_FPS

# Physics constants
GRAVITY_M_S2 = 9.81
AIR_DENSITY_KG_M3 = 1.225
BALL_MASS_KG = 0.04593
BALL_RADIUS_M = 0.02135
BALL_CROSS_SECTION_M2 = np.pi * BALL_RADIUS_M ** 2
DRAG_COEFFICIENT = 0.25

# Detection parameters
BRIGHTNESS_THRESHOLD = 80
MOTION_THRESHOLD = 25
MIN_BALL_AREA_REST = 40
MAX_BALL_AREA_REST = 500
MIN_BALL_AREA_FLIGHT = 15
MAX_BALL_AREA_FLIGHT = 400
MIN_CIRCULARITY_REST = 0.4
MIN_CIRCULARITY_FLIGHT = 0.35
REST_SEARCH_RADIUS = 40

# Known ball rest position (bottom camera view)
BALL_REST_X = 249
BALL_REST_Y = 216

# Dynamic ROI box for tracking
ROI_REST_HALF = 30
ROI_FLIGHT_HALF = 40
ROI_GROWTH_PER_MISS = 10
ROI_MAX_HALF = 80


# =============================================================================
# CIRCULAR BUFFER CAPTURE SYSTEM
# =============================================================================

class CircularBufferCapture:
    """
    Continuously captures frames into a circular buffer.
    When triggered, saves pre-trigger and post-trigger frames.
    """
    
    def __init__(self):
        self.camera: Optional[Picamera2] = None
        self.running = False
        self.stop_event = Event()
        self.trigger_event = Event()
        self.capture_thread: Optional[Thread] = None
        
        # Circular buffer for frames
        self.frame_buffer: deque = deque(maxlen=BUFFER_SIZE)
        self.meta_buffer: deque = deque(maxlen=BUFFER_SIZE)
        self.buffer_lock = Lock()
        
        # Captured frames after trigger
        self.captured_frames: List[np.ndarray] = []
        self.captured_meta: List[Dict] = []
        self.capture_complete = Event()

        # IR strobe I2C control
        self.i2c_bus: Optional[SMBus] = None
        self.strobe_on_msg = i2c_msg.write(0x60, [0x30, 0x09, 0x08])
        self.strobe_off_msg = i2c_msg.write(0x60, [0x30, 0x09, 0x00])
        self.strobe_armed = False
        self.strobe_timeout_s = 20
        self.strobe_timer: Optional[Timer] = None

    def setup_camera(self) -> bool:
        """Initialize the camera for continuous capture."""
        try:
            print("[Capture] Setting up camera...")
            self.camera = Picamera2()
            
            config = self.camera.create_video_configuration(
                raw={"size": CAMERA_SIZE, "format": "R8"},
                buffer_count=BUFFER_COUNT,
            )
            self.camera.configure(config)
            self.camera.start()
            time.sleep(0.5)
            
            # Set camera controls
            frame_us = int(1_000_000 / TARGET_FPS)
            self.camera.set_controls({
                "FrameDurationLimits": (frame_us, frame_us),
                "ExposureTime": EXPOSURE_US,
                "AnalogueGain": ANALOGUE_GAIN,
            })
            
            print(f"[Capture] Camera ready: {CAMERA_SIZE} @ {TARGET_FPS}fps")

            # Initialize IR strobe via I2C
            try:
                self.i2c_bus = SMBus(10)
                # Set FSTROBE pin as output (0x3006 = 0x0C)
                self.i2c_bus.i2c_rdwr(i2c_msg.write(0x60, [0x30, 0x06, 0x0C]))
                # Enable register-controlled strobe mode (0x3027 bit[3] = 1)
                self.i2c_bus.i2c_rdwr(i2c_msg.write(0x60, [0x30, 0x27, 0x08]))
                print("[Capture] IR strobe initialized on I2C bus 10")
            except Exception as e:
                print(f"[Capture] IR strobe init failed (continuing without strobe): {e}")
                self.i2c_bus = None

            return True

        except Exception as e:
            print(f"[Capture] Camera setup failed: {e}")
            traceback.print_exc()
            return False
    
    def cleanup_camera(self):
        """Stop and close the camera and IR strobe."""
        if self.i2c_bus:
            try:
                # Turn strobe off before closing
                self.i2c_bus.i2c_rdwr(self.strobe_off_msg)
                self.i2c_bus.close()
            except:
                pass
            self.i2c_bus = None
        if self.camera:
            try:
                self.camera.close()
            except:
                pass
            self.camera = None
    
    def _cancel_strobe_timer(self):
        """Cancel any pending auto-disarm timer."""
        if self.strobe_timer is not None:
            self.strobe_timer.cancel()
            self.strobe_timer = None

    def _start_strobe_timer(self):
        """Start (or restart) the auto-disarm timer."""
        self._cancel_strobe_timer()
        self.strobe_timer = Timer(self.strobe_timeout_s, self._auto_disarm)
        self.strobe_timer.daemon = True
        self.strobe_timer.start()

    def _auto_disarm(self):
        """Called by the timer when no swing happens within the timeout."""
        print(f"[Capture] No swing detected within {self.strobe_timeout_s}s — auto-disarming strobe")
        self.disarm_strobe()

    def arm_strobe(self):
        """Turn the IR strobe on. Called when system is armed for a swing.
        Pre-trigger frames captured after this point will be illuminated.
        Auto-disarms after strobe_timeout_s seconds if no trigger arrives."""
        if not self.i2c_bus:
            return
        if self.strobe_armed:
            # Already on — just reset the timeout timer
            self._start_strobe_timer()
            return
        try:
            self.i2c_bus.i2c_rdwr(self.strobe_on_msg)
            self.strobe_armed = True
            self._start_strobe_timer()
            print("[Capture] IR strobe ARMED (ON) — auto-disarm in %ds" % self.strobe_timeout_s)
        except Exception as e:
            print(f"[Capture] Failed to arm strobe: {e}")

    def disarm_strobe(self):
        """Turn the IR strobe off. Called after a shot is processed."""
        self._cancel_strobe_timer()
        if not self.i2c_bus:
            return
        if not self.strobe_armed:
            return
        try:
            self.i2c_bus.i2c_rdwr(self.strobe_off_msg)
            self.strobe_armed = False
            print("[Capture] IR strobe DISARMED (OFF)")
        except Exception as e:
            print(f"[Capture] Failed to disarm strobe: {e}")

    def capture_loop(self):
        """Main capture loop - runs continuously in background thread."""
        print("[Capture] Capture loop started")

        while not self.stop_event.is_set():
            try:
                req = self.camera.capture_request()
                meta = req.get_metadata()
                raw = req.make_array("raw").copy()
                req.release()

                timestamp_us = meta.get("SensorTimestamp", time.monotonic_ns() // 1000)
                
                # Add to circular buffer
                with self.buffer_lock:
                    self.frame_buffer.append(raw)
                    self.meta_buffer.append({
                        "timestamp_us": timestamp_us,
                        "frame_time": time.time()
                    })
                
                # Check if trigger was received
                if self.trigger_event.is_set():
                    self._handle_trigger()
                    self.trigger_event.clear()
                    
            except Exception as e:
                print(f"[Capture] Error in capture loop: {e}")
                time.sleep(0.01)
        
        print("[Capture] Capture loop stopped")
    
    def _handle_trigger(self):
        """Handle trigger - save pre-buffer and capture post-trigger frames."""
        print(f"[Capture] Trigger received! Saving frames...")
        
        # Copy pre-trigger frames from buffer
        with self.buffer_lock:
            self.captured_frames = list(self.frame_buffer)
            self.captured_meta = list(self.meta_buffer)
        
        pre_count = len(self.captured_frames)
        print(f"[Capture] Saved {pre_count} pre-trigger frames")
        
        # Capture post-trigger frames (strobe is already on from capture_loop startup)
        for i in range(POST_TRIGGER_FRAMES):
            try:
                req = self.camera.capture_request()
                meta = req.get_metadata()
                raw = req.make_array("raw").copy()
                req.release()

                timestamp_us = meta.get("SensorTimestamp", time.monotonic_ns() // 1000)
                
                self.captured_frames.append(raw)
                self.captured_meta.append({
                    "timestamp_us": timestamp_us,
                    "frame_time": time.time()
                })
            except Exception as e:
                print(f"[Capture] Error capturing post-trigger frame {i}: {e}")
                break
        
        post_count = len(self.captured_frames) - pre_count
        print(f"[Capture] Captured {post_count} post-trigger frames")
        print(f"[Capture] Total: {len(self.captured_frames)} frames")
        
        self.capture_complete.set()
    
    def start(self) -> bool:
        """Start continuous capture."""
        if self.running:
            return True
        
        if not self.setup_camera():
            return False
        
        self.stop_event.clear()
        self.capture_thread = Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()
        self.running = True
        
        print("[Capture] Continuous capture started")
        return True
    
    def stop(self):
        """Stop continuous capture."""
        self.stop_event.set()
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        self.cleanup_camera()
        self.running = False
        print("[Capture] Stopped")
    
    def trigger(self, timeout: float = 5.0) -> bool:
        """
        Trigger frame capture.
        
        Returns True if frames were captured successfully.
        """
        if not self.running:
            print("[Capture] Not running, cannot trigger")
            return False
        
        # Clear previous capture
        self.captured_frames = []
        self.captured_meta = []
        self.capture_complete.clear()
        
        # Signal trigger
        self.trigger_event.set()
        
        # Wait for capture to complete
        if self.capture_complete.wait(timeout=timeout):
            return len(self.captured_frames) > 0
        else:
            print("[Capture] Trigger timeout")
            return False
    
    def get_captured_frames(self) -> Tuple[List[np.ndarray], List[Dict]]:
        """Get the captured frames and metadata."""
        return self.captured_frames, self.captured_meta


# Global circular buffer capture instance
capture_system = CircularBufferCapture()


# =============================================================================
# BALL DETECTION AND METRIC CALCULATION
# =============================================================================

def get_contour_features(contour, gray_frame=None) -> dict:
    """Extract features from a contour."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    
    M = cv2.moments(contour)
    if M['m00'] > 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = 0, 0
    
    x, y, w, h = cv2.boundingRect(contour)
    
    brightness = 0
    if gray_frame is not None:
        mask = np.zeros(gray_frame.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        brightness = cv2.mean(gray_frame, mask=mask)[0]
    
    return {
        'area': area,
        'circularity': circularity,
        'cx': cx,
        'cy': cy,
        'bbox': (x, y, w, h),
        'brightness': brightness
    }


def apply_roi_box(gray: np.ndarray, cx: int, cy: int, half: int) -> np.ndarray:
    """Zero out everything outside an ROI box."""
    masked = np.zeros_like(gray)
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(gray.shape[1], cx + half)
    y2 = min(gray.shape[0], cy + half)
    masked[y1:y2, x1:x2] = gray[y1:y2, x1:x2]
    return masked


def detect_ball_at_rest(frame: np.ndarray, hint: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int, float]]:
    """
    Detect ball at rest position within a tight ROI box.
    Returns (x, y, area) or None if not found.
    """
    gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    target_x = hint[0] if hint else BALL_REST_X
    target_y = hint[1] if hint else BALL_REST_Y
    gray_roi = apply_roi_box(gray, target_x, target_y, ROI_REST_HALF)

    best_candidate = None
    best_score = -1

    for thresh_val in [BRIGHTNESS_THRESHOLD, BRIGHTNESS_THRESHOLD - 15, BRIGHTNESS_THRESHOLD + 15]:
        if thresh_val < 40 or thresh_val > 200:
            continue
        _, thresh = cv2.threshold(gray_roi, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            features = get_contour_features(contour, gray)
            if not (MIN_BALL_AREA_REST < features['area'] < MAX_BALL_AREA_REST):
                continue
            if features['circularity'] < MIN_CIRCULARITY_REST:
                continue
            dist = np.sqrt((features['cx'] - target_x)**2 + (features['cy'] - target_y)**2)
            if dist > REST_SEARCH_RADIUS:
                continue
            score = 100 - dist + features['circularity'] * 50
            if score > best_score:
                best_score = score
                best_candidate = (features['cx'], features['cy'], features['area'])

    return best_candidate


def detect_ball_in_flight(current_frame: np.ndarray, reference_frame: np.ndarray,
                          prev_position: Optional[Tuple[int, int]] = None,
                          roi_center: Optional[Tuple[int, int]] = None,
                          roi_half: int = ROI_FLIGHT_HALF) -> Optional[Tuple[int, int]]:
    """
    Detect ball in flight using frame differencing within a dynamic ROI box.
    Returns (x, y) or None if not found.
    """
    gray_curr = current_frame if len(current_frame.shape) == 2 else cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray_ref = reference_frame if len(reference_frame.shape) == 2 else cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray_curr, gray_ref)
    if roi_center:
        diff = apply_roi_box(diff, int(roi_center[0]), int(roi_center[1]), roi_half)

    best_candidate = None
    best_score = -1

    for thresh_val in [MOTION_THRESHOLD, MOTION_THRESHOLD - 5, MOTION_THRESHOLD + 10]:
        if thresh_val < 10:
            continue
        _, thresh = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            features = get_contour_features(contour, gray_curr)
            if not (MIN_BALL_AREA_FLIGHT < features['area'] < MAX_BALL_AREA_FLIGHT):
                continue
            if features['circularity'] < MIN_CIRCULARITY_FLIGHT:
                continue
            # Reject elongated shapes (club shaft)
            bbox_w = features['bbox'][2]
            bbox_h = features['bbox'][3]
            aspect = max(bbox_w, bbox_h) / max(1, min(bbox_w, bbox_h))
            if aspect > 3.0:
                continue

            score = features['circularity'] * 100
            if prev_position:
                dy = features['cy'] - prev_position[1]
                if dy < 0:  # Ball moves upward from bottom camera
                    score += 40
            if score > best_score:
                best_score = score
                best_candidate = (features['cx'], features['cy'])

    return best_candidate


def calculate_metrics(frames: List[np.ndarray], meta: List[Dict]) -> Dict[str, Any]:
    """
    Calculate golf metrics from captured frames.
    Uses dynamic ROI tracking and ball-still-at-rest detection.
    """
    if len(frames) < 5:
        return {"error": "Not enough frames for analysis"}

    print(f"[Metrics] Analyzing {len(frames)} frames...")

    # Find ball at rest in early frames
    rest_position = None
    rest_frame_idx = None

    for i in range(min(10, len(frames))):
        result = detect_ball_at_rest(frames[i])
        if result:
            rest_position = (result[0], result[1])
            rest_frame_idx = i
            break

    if not rest_position:
        print("[Metrics] Could not find ball at rest")
        return {
            "error": "Could not detect ball at rest",
            "_rest_position": None,
            "_detections": [],
            "_rest_frame_idx": None
        }

    print(f"[Metrics] Ball at rest: {rest_position} (frame {rest_frame_idx})")

    # Use latest pre-trigger frame as reference (will update if ball still at rest)
    reference_frame = frames[rest_frame_idx]

    # Track ball in post-trigger frames with dynamic ROI
    detections = []
    prev_pos = rest_position
    roi_center = rest_position
    roi_half = ROI_FLIGHT_HALF
    ball_departed = False
    frames_missed = 0

    trigger_idx = PRE_TRIGGER_FRAMES

    for i in range(trigger_idx, len(frames)):
        # Check if ball is still at rest before using motion detection
        if not ball_departed:
            rest_check = detect_ball_at_rest(frames[i], hint=rest_position)
            if rest_check:
                dist = np.sqrt((rest_check[0] - rest_position[0])**2 +
                               (rest_check[1] - rest_position[1])**2)
                if dist < 20:
                    # Ball still at rest — update reference frame and skip
                    reference_frame = frames[i]
                    continue
            # Ball gone from rest position — impact occurred
            ball_departed = True
            print(f"[Metrics] Ball departed at frame {i}")

        # Motion detection with dynamic ROI
        pos = detect_ball_in_flight(frames[i], reference_frame, prev_pos,
                                    roi_center=roi_center, roi_half=roi_half)

        if pos:
            dist_from_rest = np.sqrt((pos[0] - rest_position[0])**2 +
                                     (pos[1] - rest_position[1])**2)
            if dist_from_rest > 20:
                detections.append({
                    'frame_idx': i,
                    'x': pos[0],
                    'y': pos[1],
                    'timestamp_us': meta[i].get('timestamp_us', 0)
                })
                prev_pos = pos
                roi_center = pos  # Move ROI to follow ball
                roi_half = ROI_FLIGHT_HALF  # Reset ROI size on hit
                frames_missed = 0

                # Ball exiting frame
                if pos[0] > 620 or pos[1] < 20 or pos[0] < 20:
                    break
                continue

        # Missed detection — grow ROI, predict position
        frames_missed += 1
        roi_half = min(roi_half + ROI_GROWTH_PER_MISS, ROI_MAX_HALF)
        if frames_missed >= 3 and len(detections) > 0:
            break  # Lost the ball

    print(f"[Metrics] Found {len(detections)} post-impact detections")

    if len(detections) < 2:
        return {
            "error": "Not enough ball detections in flight",
            "_rest_position": rest_position,
            "_detections": detections,
            "_rest_frame_idx": rest_frame_idx
        }

    # Calculate ball speed from first few detections
    velocities_mps = []
    for i in range(min(4, len(detections) - 1)):
        d1 = detections[i]
        d2 = detections[i + 1]

        dx_px = d2['x'] - d1['x']
        dy_px = d2['y'] - d1['y']

        dx_m = dx_px * METERS_PER_PIXEL
        dy_m = -dy_px * METERS_PER_PIXEL  # Flip Y axis

        # Use timestamps if available, otherwise frame rate
        if d1['timestamp_us'] and d2['timestamp_us']:
            dt = (d2['timestamp_us'] - d1['timestamp_us']) / 1_000_000_000  # Nanoseconds to seconds
        else:
            dt = SECONDS_PER_FRAME

        if dt > 0:
            speed = np.sqrt(dx_m**2 + dy_m**2) / dt
            velocities_mps.append(speed)

    if not velocities_mps:
        return {
            "error": "Could not calculate velocity",
            "_rest_position": rest_position,
            "_detections": detections,
            "_rest_frame_idx": rest_frame_idx
        }

    # Use max speed (closest to impact)
    velocity_mps = max(velocities_mps)
    ball_speed_mph = velocity_mps * 2.237

    print(f"[Metrics] Ball speed: {ball_speed_mph:.1f} mph")

    # Calculate launch angle
    if len(detections) >= 2:
        x_vals = [d['x'] for d in detections[:5]]
        y_vals = [d['y'] for d in detections[:5]]

        x_m = [(x - rest_position[0]) * METERS_PER_PIXEL for x in x_vals]
        y_m = [-(y - rest_position[1]) * METERS_PER_PIXEL for y in y_vals]

        if len(x_m) >= 2 and (max(x_m) - min(x_m)) > 0:
            slope, _ = np.polyfit(x_m, y_m, 1)
            launch_angle_deg = np.degrees(np.arctan(slope))
        else:
            launch_angle_deg = 12.0
    else:
        launch_angle_deg = 12.0

    print(f"[Metrics] Launch angle: {launch_angle_deg:.1f} degrees")

    # Simulate trajectory for apex and carry distance
    apex_height_ft, carry_distance_yds = simulate_trajectory(velocity_mps, launch_angle_deg)

    return {
        "ball_speed": round(ball_speed_mph, 1),
        "launch_angle": round(launch_angle_deg, 1),
        "apex_height": round(apex_height_ft, 1),
        "carry_distance": round(carry_distance_yds, 1),
        "frames_analyzed": len(frames),
        "detections": len(detections),
        "_rest_position": rest_position,
        "_detections": detections,
        "_rest_frame_idx": rest_frame_idx
    }


def simulate_trajectory(velocity_mps: float, launch_angle_deg: float) -> Tuple[float, float]:
    """
    Simulate ball trajectory with air resistance.
    Returns (apex_height_ft, carry_distance_yds).
    """
    if velocity_mps <= 0 or launch_angle_deg <= 0:
        return 0, 0
    
    launch_angle_rad = np.radians(launch_angle_deg)
    
    x, y = 0.0, 0.0
    vx = velocity_mps * np.cos(launch_angle_rad)
    vy = velocity_mps * np.sin(launch_angle_rad)
    
    dt = 0.001
    max_time = 30
    max_height = 0.0
    
    drag_constant = (0.5 * AIR_DENSITY_KG_M3 * DRAG_COEFFICIENT * 
                     BALL_CROSS_SECTION_M2 / BALL_MASS_KG)
    
    t = 0
    has_risen = False
    
    while t < max_time:
        if y > max_height:
            max_height = y
            has_risen = True
        
        if has_risen and y <= 0:
            break
        
        v = np.sqrt(vx**2 + vy**2)
        
        if v > 0:
            drag_ax = -drag_constant * v * vx
            drag_ay = -drag_constant * v * vy
        else:
            drag_ax, drag_ay = 0, 0
        
        ax = drag_ax
        ay = -GRAVITY_M_S2 + drag_ay
        
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt
    
    apex_height_ft = max_height * 3.281
    carry_distance_yds = x * 1.094
    
    print(f"[Metrics] Apex: {apex_height_ft:.1f} ft, Carry: {carry_distance_yds:.1f} yds")
    
    return apex_height_ft, carry_distance_yds


# =============================================================================
# DEBUG OVERLAY GENERATION
# =============================================================================

def generate_debug_overlays(frames: List[np.ndarray], meta: List[Dict], 
                            metrics: Dict[str, Any], out_dir: Path) -> int:
    """
    Generate debug overlay images showing ball detection.
    
    Args:
        frames: List of captured frames
        meta: List of frame metadata
        metrics: Metrics dict from calculate_metrics (contains detection data)
        out_dir: Output directory for debug images
    
    Returns:
        Number of overlay images generated
    """
    # Create debug subdirectory
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(exist_ok=True)
    
    # Extract detection data from metrics
    rest_position = metrics.get("_rest_position")
    detections = metrics.get("_detections", [])
    rest_frame_idx = metrics.get("_rest_frame_idx", 0)
    
    # Build lookup for detections by frame index
    detection_by_frame = {d['frame_idx']: d for d in detections}
    
    # Colors (BGR format for OpenCV)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    CYAN = (255, 255, 0)
    WHITE = (255, 255, 255)
    MAGENTA = (255, 0, 255)
    
    trigger_idx = PRE_TRIGGER_FRAMES
    
    print(f"[Debug] Generating overlay images...")
    
    for i, frame in enumerate(frames):
        # Convert grayscale to BGR for colored overlays
        if len(frame.shape) == 2:
            overlay = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            overlay = frame.copy()
        
        # Determine frame phase
        if i < trigger_idx:
            phase = "pre"
            phase_color = CYAN
        else:
            phase = "post"
            phase_color = YELLOW
        
        # Draw rest position marker (blue circle)
        if rest_position:
            cv2.circle(overlay, rest_position, 20, BLUE, 2)
            cv2.putText(overlay, "REST", (rest_position[0] - 20, rest_position[1] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1)
        
        # Draw trajectory line (connecting all detections up to this frame)
        past_detections = [d for d in detections if d['frame_idx'] <= i]
        if len(past_detections) > 1:
            points = [(d['x'], d['y']) for d in past_detections]
            for j in range(1, len(points)):
                cv2.line(overlay, points[j-1], points[j], GREEN, 2)
        
        # Draw current detection (if this frame has one)
        if i in detection_by_frame:
            det = detection_by_frame[i]
            pos = (det['x'], det['y'])
            
            # Green filled circle for detected ball
            cv2.circle(overlay, pos, 15, GREEN, 2)
            cv2.circle(overlay, pos, 4, GREEN, -1)
            
            # Position label
            cv2.putText(overlay, f"({pos[0]}, {pos[1]})", (pos[0] + 18, pos[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1)
            
            # Calculate velocity to previous detection
            det_idx = detections.index(det)
            if det_idx > 0:
                prev_det = detections[det_idx - 1]
                vx = det['x'] - prev_det['x']
                vy = det['y'] - prev_det['y']
                vel = np.sqrt(vx**2 + vy**2)
                cv2.putText(overlay, f"v:{vel:.0f}px", (pos[0] + 18, pos[1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, CYAN, 1)
        
        # Draw status bar at top
        bar_height = 35
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], bar_height), (30, 30, 30), -1)
        
        # Frame info
        frame_text = f"Frame {i:02d} | {phase.upper()}"
        cv2.putText(overlay, frame_text, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, phase_color, 1)
        
        # Detection status
        if i in detection_by_frame:
            cv2.putText(overlay, "DETECTED", (520, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1)
        elif i < trigger_idx and rest_position and i == rest_frame_idx:
            cv2.putText(overlay, "REST FOUND", (510, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLUE, 1)
        else:
            cv2.putText(overlay, "---", (560, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1)
        
        # Metrics summary on second line (only if we have valid metrics)
        if "ball_speed" in metrics:
            metrics_text = f"Speed: {metrics['ball_speed']} mph | Angle: {metrics['launch_angle']} deg | Carry: {metrics['carry_distance']} yds"
            cv2.putText(overlay, metrics_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1)
        elif "error" in metrics:
            cv2.putText(overlay, f"Error: {metrics['error']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, RED, 1)
        
        # Save overlay image
        overlay_path = debug_dir / f"debug_{i:04d}.png"
        cv2.imwrite(str(overlay_path), overlay)
    
    print(f"[Debug] Saved {len(frames)} overlay images to {debug_dir}")
    return len(frames)


# =============================================================================
# SENSOR STATE
# =============================================================================

class SensorState:
    """Tracks current sensor state."""
    def __init__(self):
        self.current_club_id: Optional[str] = None
        self.current_preset_version: int = 0
        self.current_preset_data: Dict[str, Any] = {}
        self.shots_processed: int = 0
    
    def apply_preset(self, club_id: str, preset_data: Dict[str, Any], version: int):
        """Apply a club preset."""
        self.current_club_id = club_id
        self.current_preset_version = version
        self.current_preset_data = preset_data
        
        print(f"[Sensor] Applying preset for {club_id}:")
        print(f"         Version: {version}")
    
    def capture_shot(self, shot_id: str, club_id: str) -> bytes:
        """
        Capture and process a shot using circular buffer.
        
        1. Triggers the circular buffer to save pre/post frames
        2. Saves frames to disk
        3. Runs ball detection and metric calculation
        4. Returns CSV with real metrics
        """
        self.shots_processed += 1
        
        print(f"[Sensor] Capturing shot {shot_id} with {club_id}")
        
        # Trigger the circular buffer capture
        if not capture_system.trigger(timeout=5.0):
            csv_lines = [
                "metric,value,unit",
                "error,Capture trigger failed,",
                f"club_id,{club_id},",
                f"shot_id,{shot_id},",
            ]
            return ("\n".join(csv_lines) + "\n").encode("utf-8")
        
        # Get captured frames
        frames, meta = capture_system.get_captured_frames()
        num_frames = len(frames)
        
        print(f"[Sensor] Captured {num_frames} frames")
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(CAPTURE_OUTPUT_DIR) / f"{shot_id}_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save frames
        print(f"[Sensor] Saving frames to {out_dir}")
        for i, frame in enumerate(frames):
            frame_path = out_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(frame_path), frame)
        
        # Save metadata
        meta_path = out_dir / "burst_meta.csv"
        with open(meta_path, 'w', newline='') as f:
            writer = csv_module.writer(f)
            writer.writerow(["frame_idx", "timestamp_us", "phase"])
            trigger_idx = PRE_TRIGGER_FRAMES
            for i, m in enumerate(meta):
                phase = "pre" if i < trigger_idx else "post"
                writer.writerow([i, m.get('timestamp_us', 0), phase])
        
        # Calculate metrics
        metrics = calculate_metrics(frames, meta)
        
        # Generate debug overlay images
        try:
            generate_debug_overlays(frames, meta, metrics, out_dir)
        except Exception as e:
            print(f"[Sensor] Warning: Failed to generate debug overlays: {e}")
        
        # Build CSV response (exclude internal fields starting with _)
        csv_lines = ["metric,value,unit"]
        
        if "error" in metrics:
            csv_lines.append(f"error,{metrics['error']},")
        else:
            csv_lines.append(f"ball_speed,{metrics['ball_speed']},mph")
            csv_lines.append(f"launch_angle,{metrics['launch_angle']},degrees")
            csv_lines.append(f"apex_height,{metrics['apex_height']},feet")
            csv_lines.append(f"carry_distance,{metrics['carry_distance']},yards")
        
        csv_lines.append(f"frames_captured,{num_frames},")
        csv_lines.append(f"club_id,{club_id},")
        csv_lines.append(f"shot_id,{shot_id},")
        csv_lines.append(f"output_dir,{out_dir},")
        
        csv_text = "\n".join(csv_lines) + "\n"
        
        # Save metrics CSV
        metrics_path = out_dir / "metrics.csv"
        metrics_path.write_text(csv_text)
        print(f"[Sensor] Metrics saved to {metrics_path}")
        
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

    # Arm the IR strobe — system is now ready for a swing.
    # The circular buffer will roll over within ~100ms so by the time
    # the trigger arrives, all pre-trigger frames will be illuminated.
    capture_system.arm_strobe()

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

    # Defensive arm — in case no CLUB_PRESET was sent first.
    # Pre-trigger frames already in the buffer may be dark in that case,
    # but post-trigger frames will at least be illuminated.
    capture_system.arm_strobe()

    # Send immediate acknowledgment
    ack = make_ack_trigger(shot_id, club_id, request_id)
    send_frame(conn, ack)
    print(f"[Sensor] Sent ACK_TRIGGER")

    # Capture and process the shot
    try:
        csv_bytes = state.capture_shot(shot_id, club_id)
    except Exception as e:
        print(f"[Sensor] ERROR capturing shot: {e}")
        capture_system.disarm_strobe()
        error_msg = make_error(f"Capture failed: {e}", request_id, shot_id)
        send_frame(conn, error_msg)
        return

    # Shot processed — disarm strobe to save power and reduce LED heat.
    capture_system.disarm_strobe()

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
    
    # Create data directories
    os.makedirs(CSV_SOURCE_DIR, exist_ok=True)
    os.makedirs(CAPTURE_OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("SENSOR PI SERVER")
    print("=" * 60)
    print(f"Listening on: {SENSOR_PI_IP}:{PI_COMM_PORT}")
    print(f"Protocol version: 1")
    print(f"Capture: {CAMERA_SIZE} @ {TARGET_FPS}fps")
    print(f"Buffer: {PRE_TRIGGER_FRAMES} pre + {POST_TRIGGER_FRAMES} post frames")
    print("=" * 60)
    print()
    
    # Start continuous capture
    print("Starting continuous capture system...")
    if capture_system.start():
        print("Capture system running!")
    else:
        print("WARNING: Capture system failed to start")
        print("Shots will fail until camera is available")
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
            capture_system.stop()
            sys.exit(1)
        
        server.listen(1)
        
        # Accept connections forever
        try:
            while True:
                try:
                    conn, addr = server.accept()
                    with conn:
                        conn.settimeout(60)  # 60 second timeout
                        handle_client(conn, addr)
                except Exception as e:
                    print(f"[Sensor] Server error: {e}")
                    traceback.print_exc()
                    time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Sensor] Shutting down...")
        finally:
            capture_system.stop()
            print("[Sensor] Goodbye!")


if __name__ == "__main__":
    run_server()
