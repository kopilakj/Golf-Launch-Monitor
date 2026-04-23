#!/usr/bin/env python3
"""
Single Pi Golf Launch Monitor
==============================

All-in-one server that runs the entire golf launch monitor on a single
Raspberry Pi with one camera (no strobe).  Combines:

  - High-speed circular buffer capture
  - Motion detection for swing triggers
  - Ball detection and metric calculation
  - Flask web GUI

Run with:  python3 single_pi_server.py
Then open: http://<pi-ip>:5000
"""

import os
import sys
import time
import json
import sqlite3
import traceback
import csv as csv_module
from pathlib import Path
from datetime import datetime
from threading import Thread, Lock, Event
from collections import deque
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import cv2
from picamera2 import Picamera2
from flask import Flask, redirect, url_for, render_template, session, request, jsonify

# Optional I2C strobe control. If smbus2 isn't installed (or the strobe board
# isn't wired up) the system still works — ball detection just runs without
# strobe illumination.
try:
    from smbus2 import SMBus, i2c_msg
    SMBUS_AVAILABLE = True
except ImportError:
    SMBUS_AVAILABLE = False
    print("[Capture] smbus2 not available — strobe control disabled")

# Prefer waitress (production WSGI) over the Flask dev server. The dev server
# drops connections under concurrent polling, which is most of the reason the
# web UI feels flaky across browsers.
try:
    from waitress import serve as waitress_serve
    WAITRESS_AVAILABLE = True
except ImportError:
    WAITRESS_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
TEMPLATE_DIR = PROJECT_DIR / "WebApplication" / "templates"
STATIC_DIR = PROJECT_DIR / "WebApplication" / "static"
CAPTURE_OUTPUT_DIR = PROJECT_DIR / "Captures" / "camera_captures"
CSV_SAVE_DIR = PROJECT_DIR / "Captures" / "sensor_data"
DB_PATH = PROJECT_DIR / "WebApplication" / "player.db"

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

# Camera
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 400
CAMERA_SIZE = (CAPTURE_WIDTH, CAPTURE_HEIGHT)
TARGET_FPS = 300
# Two capture modes — switched live based on ball state:
#   DETECTION: finding the ball in ambient light (waiting/stabilizing, strobe off)
#   CAPTURE:   strobe-lit frames during a swing (ready onward, strobe armed)
# Gain is bumped up for CAPTURE mode because the 200us exposure + IR strobe
# pulse needs more amplification than the ambient 300us detection frame.
DETECTION_EXPOSURE_US = 300
DETECTION_GAIN = 4.0
CAPTURE_EXPOSURE_US = 200
CAPTURE_GAIN = 8.0
BUFFER_COUNT = 4

# Circular buffer
# BUFFER_SIZE must cover worst-case gap between ball leaving ROI and the next
# ball-check firing (BALL_CHECK_INTERVAL frames), plus enough pre-impact
# frames to establish rest position. 45 frames = 150ms @ 300fps.
PRE_TRIGGER_FRAMES = 15
POST_TRIGGER_FRAMES = 20
BUFFER_SIZE = 45

# Motion / ball detection
MOTION_ROI = (210, 270, 160, 40)
MOTION_COOLDOWN_SEC = 2.0

# Ball-at-rest verification
BALL_CHECK_INTERVAL = 30          # check every N frames (~10x/sec at 300fps)
BALL_READY_COUNT = int(2.0 / (BALL_CHECK_INTERVAL / TARGET_FPS))  # ~20 checks = ~2 seconds

# Swing confirmation (ball disappearance)
# When ball disappears, switch to fast checking and wait this long
# before confirming it's a real swing (not just occlusion)
BALL_CONFIRM_INTERVAL = 5         # check every N frames during confirmation (~60x/sec)
BALL_CONFIRM_SEC = 0.5            # ball must stay gone this long to confirm swing

# Metric calculation — camera calibration
CAMERA_DISTANCE_M = 0.965
HORIZONTAL_FOV_DEG = 70.0

HORIZONTAL_FOV_RAD = np.radians(HORIZONTAL_FOV_DEG)
FIELD_WIDTH_M = 2 * CAMERA_DISTANCE_M * np.tan(HORIZONTAL_FOV_RAD / 2)
METERS_PER_PIXEL = FIELD_WIDTH_M / CAPTURE_WIDTH
SECONDS_PER_FRAME = 1.0 / TARGET_FPS

# Physics
GRAVITY_M_S2 = 9.81
AIR_DENSITY_KG_M3 = 1.225
BALL_MASS_KG = 0.04593
BALL_RADIUS_M = 0.02135
BALL_CROSS_SECTION_M2 = np.pi * BALL_RADIUS_M ** 2
DRAG_COEFFICIENT = 0.25

# Detection parameters
BRIGHTNESS_THRESHOLD = 80
MOTION_THRESHOLD = 25
MIN_BALL_AREA_REST = 45
MAX_BALL_AREA_REST = 500
MIN_BALL_AREA_FLIGHT = 15
MAX_BALL_AREA_FLIGHT = 400
MIN_CIRCULARITY_REST = 0.4
MIN_CIRCULARITY_FLIGHT = 0.35
# Ball rest detection uses MOTION_ROI directly — no separate search area
ROI_FLIGHT_HALF = 40
ROI_GROWTH_PER_MISS = 10
ROI_MAX_HALF = 80

# Ball-vs-club discrimination and fit trust thresholds
MAX_VALID_DETECTIONS = 20        # more than this = club head, not ball
MIN_DETECTIONS_FOR_ANGLE = 5     # polyfit on fewer points is too noisy
MIN_ASCENT_PX = 15               # track must rise this far above rest_y
MIN_EARLY_ASCENT_PX = 5          # ball must rise by frame 3
RETURN_TO_REST_PX = 30           # detection inside this radius = whiff
BALLISTIC_OUTLIER_PX = 25        # reject detections this far from prediction
SPEED_JUMP_RATIO_MAX = 2.5       # reject if speed jumps > 2.5x
SPEED_JUMP_RATIO_MIN = 0.3       # or drops below 0.3x
ABSOLUTE_MAX_BALL_MPH = 230      # physical ceiling (driver ~200)

# Per-club expected top speeds (mph) — used as upper sanity cap = range_max * 1.3
EXPECTED_SPEED_MAX_MPH = {
    "driver": 180, "3wood": 160, "5wood": 155,
    "3hybrid": 145, "4hybrid": 140,
    "3iron": 140, "4iron": 135, "5iron": 140, "6iron": 130,
    "7iron": 130, "8iron": 125, "9iron": 120,
    "pitching_wedge": 110, "gap_wedge": 105,
    "sand_wedge": 100, "lob_wedge": 95, "putter": 50,
}

# Club presets
DEFAULT_CLUB = "7iron"

GUI_TO_PRESET = {
    "Driver": "driver", "3 Wood": "3wood", "5 Wood": "5wood",
    "3 Hybrid": "3hybrid", "4 Hybrid": "4hybrid",
    "3 Iron": "3iron", "4 Iron": "4iron", "5 Iron": "5iron",
    "6 Iron": "6iron", "7 Iron": "7iron", "8 Iron": "8iron",
    "9 Iron": "9iron", "Pitching Wedge": "pitching_wedge",
    "Gap Wedge": "gap_wedge", "Sand Wedge": "sand_wedge",
    "Lob Wedge": "lob_wedge", "Putter": "putter",
}


def normalize_club_name(club_name: str) -> str:
    if club_name in GUI_TO_PRESET:
        return GUI_TO_PRESET[club_name]
    return club_name.lower().replace(" ", "").replace("-", "_") or DEFAULT_CLUB


# =============================================================================
# CAPTURE SYSTEM (circular buffer + motion detection)
# =============================================================================

class CaptureSystem:
    """
    Continuous 300fps capture into a circular buffer with integrated
    motion detection.  When a swing is detected (or manually triggered),
    saves pre-trigger and post-trigger frames for metric calculation.
    """

    def __init__(self):
        self.camera: Optional[Picamera2] = None
        self.running = False
        self.stop_event = Event()
        self.trigger_event = Event()
        self.capture_thread: Optional[Thread] = None

        # Circular buffer
        self.frame_buffer: deque = deque(maxlen=BUFFER_SIZE)
        self.meta_buffer: deque = deque(maxlen=BUFFER_SIZE)
        self.buffer_lock = Lock()

        # Captured frames after trigger
        self.captured_frames: List[np.ndarray] = []
        self.captured_meta: List[Dict] = []
        self.capture_complete = Event()

        # Motion detection state
        self.motion_enabled = Event()
        self.last_trigger_time = 0.0

        # Ball-at-rest state machine
        # States: "waiting" -> "stabilizing" -> "ready" -> "confirming" -> trigger
        self.ball_state = "waiting"
        self.ball_consecutive = 0         # consecutive detections
        self.ball_position = None         # last detected (x, y)
        self.ball_frame_counter = 0       # counts frames for check interval
        self.ball_gone_time = 0.0         # timestamp when ball first disappeared

        # Latched buffer snapshot taken when ball first disappears.
        # Preserves impact-window frames against the 0.5s confirmation delay.
        self.pre_swing_frames: Optional[List[np.ndarray]] = None
        self.pre_swing_meta: Optional[List[Dict]] = None

        # Ball center (x, y) locked in when state reaches "ready". This is the
        # authoritative rest position for the shot — passed to metric calc so
        # it doesn't have to re-detect the rest position from strobed frames.
        self.rest_position_locked: Optional[Tuple[int, int]] = None
        self.pre_swing_rest_position: Optional[Tuple[int, int]] = None

        # IR strobe (I2C). Armed once ball is confirmed at rest and disarmed
        # after shot processing. Strobe off => detection exposure (ambient).
        self.i2c_bus: Optional["SMBus"] = None
        if SMBUS_AVAILABLE:
            self.strobe_on_msg = i2c_msg.write(0x60, [0x30, 0x09, 0x08])
            self.strobe_off_msg = i2c_msg.write(0x60, [0x30, 0x09, 0x00])
        self.strobe_armed = False
        self.current_exposure_us = DETECTION_EXPOSURE_US
        self.current_gain = DETECTION_GAIN

    def setup_camera(self) -> bool:
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

            frame_us = int(1_000_000 / TARGET_FPS)
            self.current_exposure_us = DETECTION_EXPOSURE_US
            self.current_gain = DETECTION_GAIN
            self.camera.set_controls({
                "FrameDurationLimits": (frame_us, frame_us),
                "ExposureTime": self.current_exposure_us,
                "AnalogueGain": self.current_gain,
            })
            print(f"[Capture] Camera ready: {CAMERA_SIZE} @ {TARGET_FPS}fps "
                  f"(detection: {DETECTION_EXPOSURE_US}us / gain {DETECTION_GAIN})")

            # Initialize IR strobe via I2C (graceful fallback if not present)
            if SMBUS_AVAILABLE:
                try:
                    self.i2c_bus = SMBus(10)
                    # FSTROBE pin as output (0x3006 = 0x0C)
                    self.i2c_bus.i2c_rdwr(i2c_msg.write(0x60, [0x30, 0x06, 0x0C]))
                    # Enable register-controlled strobe mode (0x3027 bit[3] = 1)
                    self.i2c_bus.i2c_rdwr(i2c_msg.write(0x60, [0x30, 0x27, 0x08]))
                    # Ensure strobe starts OFF
                    self.i2c_bus.i2c_rdwr(self.strobe_off_msg)
                    self.strobe_armed = False
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
        if self.i2c_bus:
            try:
                self.i2c_bus.i2c_rdwr(self.strobe_off_msg)
                self.i2c_bus.close()
            except:
                pass
            self.i2c_bus = None
            self.strobe_armed = False
        if self.camera:
            try:
                self.camera.close()
            except:
                pass
            self.camera = None

    # ---- Exposure + Strobe control ----

    def set_capture_mode(self, exposure_us: int, gain: float):
        """Switch camera exposure + gain live. Safe to call from the capture thread."""
        if self.camera is None:
            return
        if exposure_us == self.current_exposure_us and gain == self.current_gain:
            return
        try:
            self.camera.set_controls({
                "ExposureTime": exposure_us,
                "AnalogueGain": gain,
            })
            self.current_exposure_us = exposure_us
            self.current_gain = gain
            print(f"[Capture] Mode -> {exposure_us}us / gain {gain}")
        except Exception as e:
            print(f"[Capture] Failed to set mode {exposure_us}us/{gain}: {e}")

    def arm_strobe(self):
        """Turn IR strobe on. Called when ball reaches ready state."""
        if not self.i2c_bus or self.strobe_armed:
            return
        try:
            self.i2c_bus.i2c_rdwr(self.strobe_on_msg)
            self.strobe_armed = True
            print("[Capture] IR strobe ARMED")
        except Exception as e:
            print(f"[Capture] Failed to arm strobe: {e}")

    def disarm_strobe(self):
        """Turn IR strobe off. Called after shot processing or motion disable."""
        if not self.i2c_bus or not self.strobe_armed:
            return
        try:
            self.i2c_bus.i2c_rdwr(self.strobe_off_msg)
            self.strobe_armed = False
            print("[Capture] IR strobe DISARMED")
        except Exception as e:
            print(f"[Capture] Failed to disarm strobe: {e}")

    # ---- Ball Detection + Swing Detection ----

    def reset_motion(self):
        self.ball_state = "waiting"
        self.ball_consecutive = 0
        self.ball_position = None
        self.ball_frame_counter = 0
        self.ball_gone_time = 0.0
        self.pre_swing_frames = None
        self.pre_swing_meta = None
        self.rest_position_locked = None
        self.pre_swing_rest_position = None
        # Drop back to ambient ball-hunting mode
        self.disarm_strobe()
        self.set_capture_mode(DETECTION_EXPOSURE_US, DETECTION_GAIN)

    def check_ball_state(self, frame: np.ndarray) -> bool:
        """
        State machine for ball detection and swing confirmation.

        States:
          "waiting"     - Looking for ball every BALL_CHECK_INTERVAL frames
          "stabilizing" - Ball found, counting consecutive detections
          "ready"       - Ball confirmed at rest, checking it's still there
          "confirming"  - Ball disappeared, checking fast to confirm swing

        Returns True when a swing is confirmed (ball gone for BALL_CONFIRM_SEC).
        """
        self.ball_frame_counter += 1

        # Determine check interval based on state
        if self.ball_state == "confirming":
            interval = BALL_CONFIRM_INTERVAL
        else:
            interval = BALL_CHECK_INTERVAL

        if self.ball_frame_counter % interval != 0:
            return False

        result = detect_ball_at_rest(frame)

        if self.ball_state == "waiting":
            if result:
                self.ball_position = (result[0], result[1])
                self.ball_consecutive = 1
                self.ball_state = "stabilizing"
                print(f"[Ball] Detected at ({result[0]}, {result[1]}) — stabilizing...")

        elif self.ball_state == "stabilizing":
            if result:
                self.ball_position = (result[0], result[1])
                self.ball_consecutive += 1
                if self.ball_consecutive >= BALL_READY_COUNT:
                    self.ball_state = "ready"
                    # Lock in the rest position — this is the authoritative
                    # starting point for metric calculation
                    self.rest_position_locked = self.ball_position
                    # Ball locked in — switch camera to strobe-lit capture mode
                    self.set_capture_mode(CAPTURE_EXPOSURE_US, CAPTURE_GAIN)
                    self.arm_strobe()
                    print(f"[Ball] ======= READY FOR SWING at {self.rest_position_locked} =======")
            else:
                print(f"[Ball] Lost during stabilizing — back to waiting")
                self.ball_state = "waiting"
                self.ball_consecutive = 0
                self.ball_position = None

        elif self.ball_state == "ready":
            if result:
                # Ball still there — all good
                self.ball_position = (result[0], result[1])
            else:
                # Ball just disappeared — latch buffer snapshot NOW so impact
                # frames survive the 0.5s confirmation window, then start confirming
                with self.buffer_lock:
                    self.pre_swing_frames = list(self.frame_buffer)
                    self.pre_swing_meta = list(self.meta_buffer)
                # Latch the locked rest position too so metric calc doesn't
                # have to re-find it in the strobed frames after impact
                self.pre_swing_rest_position = self.rest_position_locked
                self.ball_state = "confirming"
                self.ball_gone_time = time.time()
                self.ball_frame_counter = 0  # reset so fast interval starts immediately
                print(f"[Ball] Disappeared — latched {len(self.pre_swing_frames)} pre-swing frames, confirming...")

        elif self.ball_state == "confirming":
            if result:
                # Ball came back — was just occlusion (person, club waggle)
                self.ball_position = (result[0], result[1])
                self.ball_state = "ready"
                self.pre_swing_frames = None
                self.pre_swing_meta = None
                self.pre_swing_rest_position = None
                print(f"[Ball] Came back — false alarm, still ready")
            else:
                # Ball still gone — check if confirmation time has elapsed
                elapsed = time.time() - self.ball_gone_time
                if elapsed >= BALL_CONFIRM_SEC:
                    print(f"[Ball] ========================================")
                    print(f"[Ball] SWING CONFIRMED! (gone for {elapsed:.2f}s)")
                    print(f"[Ball] ========================================")
                    return True

        return False

    def enable_motion(self):
        self.motion_enabled.set()
        self.reset_motion()
        print("[Motion] Enabled — watching for swings")

    def disable_motion(self):
        self.motion_enabled.clear()
        self.disarm_strobe()
        self.set_capture_mode(DETECTION_EXPOSURE_US, DETECTION_GAIN)
        print("[Motion] Disabled")

    # ---- Capture Loop ----

    def capture_loop(self):
        print("[Capture] Capture loop started")

        while not self.stop_event.is_set():
            try:
                req = self.camera.capture_request()
                meta = req.get_metadata()
                raw = req.make_array("raw").copy()
                req.release()

                timestamp_us = meta.get("SensorTimestamp", time.monotonic_ns() // 1000)

                with self.buffer_lock:
                    self.frame_buffer.append(raw)
                    self.meta_buffer.append({
                        "timestamp_us": timestamp_us,
                        "frame_time": time.time()
                    })

                # Ball detection + swing confirmation
                if (self.motion_enabled.is_set()
                        and not self.trigger_event.is_set()
                        and time.time() - self.last_trigger_time > MOTION_COOLDOWN_SEC):

                    if self.check_ball_state(raw):
                        self.last_trigger_time = time.time()
                        self.trigger_event.set()

                # Handle trigger
                if self.trigger_event.is_set():
                    self._handle_trigger()
                    self.trigger_event.clear()

                    # If motion-triggered, process the shot in a background thread
                    if self.motion_enabled.is_set():
                        Thread(target=_motion_process_shot, daemon=True).start()

            except Exception as e:
                print(f"[Capture] Error in capture loop: {e}")
                time.sleep(0.01)

        print("[Capture] Capture loop stopped")

    def _handle_trigger(self):
        print("[Capture] Trigger received! Saving frames...")

        # Prefer latched pre-swing snapshot (captured the instant ball disappeared).
        # Falls back to the current buffer for manual triggers.
        if self.pre_swing_frames is not None:
            self.captured_frames = self.pre_swing_frames
            self.captured_meta = self.pre_swing_meta
            self.pre_swing_frames = None
            self.pre_swing_meta = None
            print(f"[Capture] Using latched pre-swing snapshot")
        else:
            with self.buffer_lock:
                self.captured_frames = list(self.frame_buffer)
                self.captured_meta = list(self.meta_buffer)

        pre_count = len(self.captured_frames)
        print(f"[Capture] Saved {pre_count} pre-trigger frames")

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
        print(f"[Capture] Total: {pre_count} pre + {post_count} post = {len(self.captured_frames)} frames")
        self.capture_complete.set()

    # ---- Start / Stop / Trigger ----

    def start(self) -> bool:
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
        self.stop_event.set()
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        self.cleanup_camera()
        self.running = False
        print("[Capture] Stopped")

    def trigger(self, timeout: float = 5.0) -> bool:
        if not self.running:
            print("[Capture] Not running, cannot trigger")
            return False
        self.captured_frames = []
        self.captured_meta = []
        self.capture_complete.clear()
        self.trigger_event.set()
        if self.capture_complete.wait(timeout=timeout):
            return len(self.captured_frames) > 0
        print("[Capture] Trigger timeout")
        return False

    def get_captured_frames(self) -> Tuple[List[np.ndarray], List[Dict]]:
        return self.captured_frames, self.captured_meta


# Global capture system
capture_system = CaptureSystem()


# =============================================================================
# BALL DETECTION AND METRIC CALCULATION
# =============================================================================

def get_contour_features(contour, gray_frame=None) -> dict:
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
    cx_sub, cy_sub = float(cx), float(cy)
    if gray_frame is not None:
        mask = np.zeros(gray_frame.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        brightness = cv2.mean(gray_frame, mask=mask)[0]
        # Intensity-weighted sub-pixel centroid — better than the geometric
        # centroid for bright compact blobs like the strobed ball.
        ys_px, xs_px = np.where(mask > 0)
        if len(xs_px) > 0:
            I = gray_frame[ys_px, xs_px].astype(np.float64)
            tot = I.sum()
            if tot > 0:
                cx_sub = float((xs_px * I).sum() / tot)
                cy_sub = float((ys_px * I).sum() / tot)

    return {
        'area': area, 'circularity': circularity,
        'cx': cx, 'cy': cy,
        'cx_sub': cx_sub, 'cy_sub': cy_sub,
        'bbox': (x, y, w, h),
        'brightness': brightness
    }


def apply_roi_box(gray, cx, cy, half):
    masked = np.zeros_like(gray)
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(gray.shape[1], cx + half), min(gray.shape[0], cy + half)
    masked[y1:y2, x1:x2] = gray[y1:y2, x1:x2]
    return masked


def detect_ball_at_rest(frame):
    """Search for a ball at rest inside the MOTION_ROI rectangle."""
    gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rx, ry, rw, rh = MOTION_ROI

    # Crop to MOTION_ROI only
    roi_crop = gray[ry:ry + rh, rx:rx + rw]

    best_candidate = None
    best_score = -1

    for thresh_val in [BRIGHTNESS_THRESHOLD, BRIGHTNESS_THRESHOLD - 15, BRIGHTNESS_THRESHOLD + 15]:
        if thresh_val < 10 or thresh_val > 240:
            continue
        _, thresh = cv2.threshold(roi_crop, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            features = get_contour_features(contour, roi_crop)
            if not (MIN_BALL_AREA_REST < features['area'] < MAX_BALL_AREA_REST):
                continue
            if features['circularity'] < MIN_CIRCULARITY_REST:
                continue
            # Convert ROI-local coords back to full-frame coords
            full_cx = features['cx'] + rx
            full_cy = features['cy'] + ry
            score = features['circularity'] * 100 + features['area']
            if score > best_score:
                best_score = score
                best_candidate = (full_cx, full_cy, features['area'])

    return best_candidate


def detect_ball_in_flight(current_frame, reference_frame, prev_position=None,
                          roi_center=None, roi_half=ROI_FLIGHT_HALF,
                          min_x=None, max_y=None, top_n=1):
    """Returns (cx_sub, cy_sub) sub-pixel floats when top_n=1, else a list of
    up to top_n such tuples sorted by score descending (best first)."""
    gray_curr = current_frame if len(current_frame.shape) == 2 else cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray_ref = reference_frame if len(reference_frame.shape) == 2 else cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray_curr, gray_ref)
    if roi_center:
        diff = apply_roi_box(diff, int(roi_center[0]), int(roi_center[1]), roi_half)

    # Collect (score, cx_sub, cy_sub); dedupe candidates that appear at
    # multiple thresholds by keeping the highest-scoring one per ~3px bucket.
    bucket = {}  # (cx//3, cy//3) -> (score, cx_sub, cy_sub)

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
            bbox_w, bbox_h = features['bbox'][2], features['bbox'][3]
            aspect = max(bbox_w, bbox_h) / max(1, min(bbox_w, bbox_h))
            if aspect > 3.0:
                continue
            if min_x is not None and features['cx_sub'] < min_x:
                continue
            if max_y is not None and features['cy_sub'] > max_y:
                continue
            score = features['circularity'] * 100
            if prev_position:
                dy = features['cy_sub'] - prev_position[1]
                if dy < 0:
                    score += 40
            key = (int(features['cx']) // 3, int(features['cy']) // 3)
            entry = (score, features['cx_sub'], features['cy_sub'])
            if key not in bucket or bucket[key][0] < score:
                bucket[key] = entry

    ranked = sorted(bucket.values(), key=lambda e: e[0], reverse=True)
    if not ranked:
        return None if top_n == 1 else []
    if top_n == 1:
        return (ranked[0][1], ranked[0][2])
    return [(e[1], e[2]) for e in ranked[:top_n]]


def calculate_metrics(frames, meta, known_rest_position=None, club_id=None):
    if len(frames) < 5:
        return {"error": "Not enough frames for analysis"}

    print(f"[Metrics] Analyzing {len(frames)} frames...")

    rest_position = None
    rest_frame_idx = None

    if known_rest_position is not None:
        rest_position = (int(known_rest_position[0]), int(known_rest_position[1]))
        rest_frame_idx = 0
        print(f"[Metrics] Ball at rest (locked): {rest_position}")
    else:
        for i in range(min(10, len(frames))):
            result = detect_ball_at_rest(frames[i])
            if result:
                rest_position = (result[0], result[1])
                rest_frame_idx = i
                break
        if not rest_position:
            print("[Metrics] Could not find ball at rest")
            return {"error": "Could not detect ball at rest"}
        print(f"[Metrics] Ball at rest: {rest_position} (frame {rest_frame_idx})")

    reference_frame = frames[rest_frame_idx]

    detections = []
    prev_pos = rest_position
    roi_center = rest_position
    roi_half = ROI_FLIGHT_HALF
    ball_departed = False
    frames_missed = 0
    trigger_idx = PRE_TRIGGER_FRAMES

    for i in range(trigger_idx, len(frames)):
        if not ball_departed:
            rest_check = detect_ball_at_rest(frames[i])
            if rest_check:
                dist = np.sqrt((rest_check[0] - rest_position[0]) ** 2 +
                               (rest_check[1] - rest_position[1]) ** 2)
                if dist < 20:
                    reference_frame = frames[i]
                    continue
            ball_departed = True
            print(f"[Metrics] Ball departed at frame {i}")

        # Directional constraints passed into the detector so noise blobs that
        # would violate them never beat the real ball on score.
        if not detections:
            min_x = rest_position[0] + 2
        else:
            min_x = detections[-1]['x_sub'] - 3
        max_y = None
        if len(detections) >= 2:
            prev_dy = detections[-1]['y_sub'] - detections[-2]['y_sub']
            if prev_dy < -3:
                max_y = detections[-1]['y_sub'] + 3

        # Predict where the ball should be in this frame, if we have history.
        predicted_pos = None
        if len(detections) >= 2:
            last, prev = detections[-1], detections[-2]
            vx = last['x_sub'] - prev['x_sub']
            vy = last['y_sub'] - prev['y_sub']
            predicted_pos = (last['x_sub'] + vx, last['y_sub'] + vy)

        # Request top 3 candidates so we can do data association (pick the one
        # nearest our ballistic prediction) instead of blindly trusting score.
        candidates = detect_ball_in_flight(
            frames[i], reference_frame, prev_pos,
            roi_center=roi_center, roi_half=roi_half,
            min_x=min_x, max_y=max_y, top_n=3)

        pos = None
        if candidates:
            if predicted_pos is not None and len(candidates) > 1:
                # Prefer the candidate closest to prediction if the best-scoring
                # one is far off; tolerate BALLISTIC_OUTLIER_PX of wobble.
                best = candidates[0]
                best_d = np.hypot(best[0] - predicted_pos[0], best[1] - predicted_pos[1])
                if best_d > BALLISTIC_OUTLIER_PX:
                    for c in candidates[1:]:
                        d = np.hypot(c[0] - predicted_pos[0], c[1] - predicted_pos[1])
                        if d < best_d and d <= BALLISTIC_OUTLIER_PX:
                            best, best_d = c, d
                pos = best if best_d <= BALLISTIC_OUTLIER_PX else None
                if pos is None:
                    print(f"[Metrics] Frame {i}: all candidates off prediction (best d={best_d:.1f}px) — treated as miss")
            else:
                pos = candidates[0]

        # Speed-jump filter: reject if step size relative to previous is way off.
        if pos is not None and len(detections) >= 2:
            last, prev = detections[-1], detections[-2]
            prev_step = np.hypot(last['x_sub'] - prev['x_sub'], last['y_sub'] - prev['y_sub'])
            new_step = np.hypot(pos[0] - last['x_sub'], pos[1] - last['y_sub'])
            if prev_step > 3:
                ratio = new_step / prev_step
                if ratio > SPEED_JUMP_RATIO_MAX or ratio < SPEED_JUMP_RATIO_MIN:
                    print(f"[Metrics] Frame {i}: speed-jump ratio {ratio:.2f} — rejecting")
                    pos = None

        if pos is not None:
            dist_from_rest = np.hypot(pos[0] - rest_position[0], pos[1] - rest_position[1])
            if dist_from_rest > 20:
                detections.append({
                    'frame_idx': i,
                    'x': int(round(pos[0])), 'y': int(round(pos[1])),
                    'x_sub': float(pos[0]), 'y_sub': float(pos[1]),
                    'timestamp_us': meta[i].get('timestamp_us', 0)
                })
                prev_pos = pos
                # Predictive ROI for next frame.
                if len(detections) >= 2:
                    last, prev = detections[-1], detections[-2]
                    vx = last['x_sub'] - prev['x_sub']
                    vy = last['y_sub'] - prev['y_sub']
                    roi_center = (last['x_sub'] + vx, last['y_sub'] + vy)
                else:
                    roi_center = pos
                roi_half = ROI_FLIGHT_HALF
                frames_missed = 0
                if pos[0] > 620 or pos[1] < 20 or pos[0] < 20:
                    break
                continue
            else:
                print(f"[Metrics] Frame {i}: candidate {pos} rejected (too close to rest, d={dist_from_rest:.1f})")

        frames_missed += 1
        roi_half = min(roi_half + ROI_GROWTH_PER_MISS, ROI_MAX_HALF)
        if frames_missed >= 3 and len(detections) > 0:
            break

    print(f"[Metrics] Found {len(detections)} post-impact detections")

    # ---------------- Post-loop validation ----------------

    base = {
        "frames_analyzed": len(frames),
        "detections": len(detections),
        "rest_x": int(rest_position[0]),
        "rest_y": int(rest_position[1]),
        "_rest_position": rest_position,
        "_detections": detections,
        "_known_rest": known_rest_position is not None,
    }

    if len(detections) < 2:
        return {**base, "error": "Not enough ball detections in flight"}

    # Whiff / club-head tracked: too many detections = club head traced through
    if len(detections) > MAX_VALID_DETECTIONS:
        return {**base, "error": f"Too many detections ({len(detections)}) — likely tracked club head, not ball"}

    # Whiff: tracked object wandered back near rest position after initial move
    returned_near_rest = any(
        np.hypot(d['x_sub'] - rest_position[0], d['y_sub'] - rest_position[1]) < RETURN_TO_REST_PX
        for d in detections[2:]
    )
    if returned_near_rest:
        return {**base, "error": "Track returned near rest position — likely whiff / club head"}

    # Ground-hugging track: never rose meaningfully above rest_y
    max_ascent = max(rest_position[1] - d['y_sub'] for d in detections)
    if max_ascent < MIN_ASCENT_PX:
        return {**base, "error": f"Track never rose above rest (max ascent {max_ascent:.1f}px) — likely club head"}

    # Early ascent: by the third detection, ball should be off the ground
    if len(detections) >= 3:
        early_ascent = rest_position[1] - detections[2]['y_sub']
        if early_ascent < MIN_EARLY_ASCENT_PX:
            return {**base, "error": f"Ball did not ascend by frame 3 (ascent {early_ascent:.1f}px) — likely club head"}

    # ---------------- Velocity + launch angle (line fit in time) ----------------

    t0 = detections[0]['timestamp_us']
    ts = np.array([(d['timestamp_us'] - t0) / 1_000_000_000 for d in detections], dtype=np.float64)
    # If timestamps are unavailable (all zero / not populated), use frame index.
    if not np.any(ts):
        fr = np.array([d['frame_idx'] for d in detections], dtype=np.float64)
        ts = (fr - fr[0]) * SECONDS_PER_FRAME

    xs_m = np.array([(d['x_sub'] - rest_position[0]) * METERS_PER_PIXEL for d in detections], dtype=np.float64)
    ys_m = np.array([-(d['y_sub'] - rest_position[1]) * METERS_PER_PIXEL for d in detections], dtype=np.float64)

    if ts[-1] - ts[0] <= 0:
        return {**base, "error": "Detections have no time separation"}

    # x(t) linear fit → vx0
    vx_m = float(np.polyfit(ts, xs_m, 1)[0])

    # y(t) linear fit over first few frames — gravity drop is sub-millimeter over
    # this window, so linear is fine and more stable than parabolic.
    angle_n = min(MIN_DETECTIONS_FOR_ANGLE, len(ts))
    vy_m = float(np.polyfit(ts[:angle_n], ys_m[:angle_n], 1)[0])

    velocity_mps = float(np.hypot(vx_m, vy_m))
    ball_speed_mph = velocity_mps * 2.237
    print(f"[Metrics] Ball speed (fit): {ball_speed_mph:.1f} mph  (vx={vx_m:.2f}, vy={vy_m:.2f} m/s)")

    # --- Speed sanity check ---
    speed_cap = ABSOLUTE_MAX_BALL_MPH
    if club_id:
        expected = EXPECTED_SPEED_MAX_MPH.get(club_id)
        if expected is not None:
            speed_cap = min(speed_cap, expected * 1.3)
    if ball_speed_mph > speed_cap:
        return {**base, "error": f"Ball speed {ball_speed_mph:.1f} mph exceeds physical cap {speed_cap:.0f} — timing artifact"}

    # --- Launch angle (only trustworthy with enough detections) ---
    can_trust_angle = len(detections) >= MIN_DETECTIONS_FOR_ANGLE

    launch_angle_deg = None
    apex_height_ft = None
    carry_distance_yds = None

    if can_trust_angle:
        launch_angle_deg = float(np.degrees(np.arctan2(vy_m, vx_m)))
        print(f"[Metrics] Launch angle (fit): {launch_angle_deg:.1f} degrees")
        if launch_angle_deg < -10 or launch_angle_deg > 60:
            print(f"[Metrics] Implausible launch angle {launch_angle_deg:.1f}° — dropping angle-dependent metrics")
            launch_angle_deg = None
        elif launch_angle_deg <= 0:
            # Ball fitted to go down/flat — physically implausible for a real shot
            print(f"[Metrics] Non-positive launch angle {launch_angle_deg:.1f}° — dropping")
            launch_angle_deg = None

    if launch_angle_deg is not None:
        apex_height_ft, carry_distance_yds = simulate_trajectory(velocity_mps, launch_angle_deg)
        if apex_height_ft <= 0 or carry_distance_yds <= 0:
            apex_height_ft = None
            carry_distance_yds = None

    result = {
        **base,
        "ball_speed": round(ball_speed_mph, 1),
    }
    if launch_angle_deg is not None:
        result["launch_angle"] = round(launch_angle_deg, 1)
    if apex_height_ft is not None:
        result["apex_height"] = round(apex_height_ft, 1)
    if carry_distance_yds is not None:
        result["carry_distance"] = round(carry_distance_yds, 1)
    if not can_trust_angle:
        result["note"] = f"Launch angle suppressed: only {len(detections)} detections (need {MIN_DETECTIONS_FOR_ANGLE})"
    return result


def simulate_trajectory(velocity_mps, launch_angle_deg):
    if velocity_mps <= 0 or launch_angle_deg <= 0:
        return 0, 0

    launch_angle_rad = np.radians(launch_angle_deg)
    x, y = 0.0, 0.0
    vx = velocity_mps * np.cos(launch_angle_rad)
    vy = velocity_mps * np.sin(launch_angle_rad)
    dt = 0.001
    max_height = 0.0
    drag_constant = (0.5 * AIR_DENSITY_KG_M3 * DRAG_COEFFICIENT *
                     BALL_CROSS_SECTION_M2 / BALL_MASS_KG)
    t = 0
    has_risen = False

    while t < 30:
        if y > max_height:
            max_height = y
            has_risen = True
        if has_risen and y <= 0:
            break
        v = np.sqrt(vx ** 2 + vy ** 2)
        if v > 0:
            drag_ax = -drag_constant * v * vx
            drag_ay = -drag_constant * v * vy
        else:
            drag_ax, drag_ay = 0, 0
        vx += drag_ax * dt
        vy += (-GRAVITY_M_S2 + drag_ay) * dt
        x += vx * dt
        y += vy * dt
        t += dt

    return max_height * 3.281, x * 1.094


def generate_debug_overlays(frames, meta, metrics, out_dir):
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(exist_ok=True)

    rest_position = metrics.get("_rest_position")
    detections = metrics.get("_detections", [])
    detection_by_frame = {d['frame_idx']: d for d in detections}
    known_rest = metrics.get("_known_rest", False)

    # BGR color tuples
    GREEN     = (0, 255, 0)       # accepted flight detection
    BLUE      = (255, 0, 0)       # locked rest position
    YELLOW    = (0, 255, 255)     # post-trigger phase label
    CYAN      = (255, 255, 0)     # pre-trigger phase label
    WHITE     = (255, 255, 255)
    LIGHT_BLUE = (255, 200, 100)  # per-frame "ball is tracked" marker
    ORANGE    = (0, 140, 255)     # predicted search ROI for next frame

    # Pre-compute a reference frame for flight-detection visualization.
    # Use the first frame (pre-impact, ambient-lit) — same as calculate_metrics.
    reference_frame = frames[0] if frames else None

    for i, frame in enumerate(frames):
        overlay = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if len(frame.shape) == 2 else frame.copy()
        phase = "pre" if i < PRE_TRIGGER_FRAMES else "post"
        phase_color = CYAN if i < PRE_TRIGGER_FRAMES else YELLOW

        # Locked rest position (dark blue, large)
        if rest_position:
            cv2.circle(overlay, rest_position, 20, BLUE, 2)

        # Predicted search ROI for this frame (orange box) — mirrors the
        # roi_center / roi_half that calculate_metrics would have used when
        # searching frame i. Only show post-trigger.
        if rest_position and i >= PRE_TRIGGER_FRAMES:
            past_dets = [d for d in detections if d['frame_idx'] < i]
            if len(past_dets) >= 2:
                last, prev = past_dets[-1], past_dets[-2]
                vx = last['x'] - prev['x']
                vy = last['y'] - prev['y']
                roi_cx = last['x'] + vx
                roi_cy = last['y'] + vy
                roi_label = "pred"
            elif len(past_dets) == 1:
                roi_cx = past_dets[-1]['x']
                roi_cy = past_dets[-1]['y']
                roi_label = "last"
            else:
                roi_cx, roi_cy = rest_position
                roi_label = "rest"
            rh = ROI_FLIGHT_HALF
            x1 = max(0, roi_cx - rh); y1 = max(0, roi_cy - rh)
            x2 = min(overlay.shape[1] - 1, roi_cx + rh); y2 = min(overlay.shape[0] - 1, roi_cy + rh)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), ORANGE, 1)
            cv2.putText(overlay, f"ROI:{roi_label}", (x1 + 2, y1 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, ORANGE, 1)

        # --- Per-frame tracking indicator (light blue) ---
        # Try ball-at-rest detector first; if that misses and this frame has
        # an accepted flight detection, use that instead. No circle = no track.
        tracked_pos = None
        rest_det = detect_ball_at_rest(frame)
        if rest_det:
            tracked_pos = (rest_det[0], rest_det[1])
        elif i in detection_by_frame:
            det = detection_by_frame[i]
            tracked_pos = (det['x'], det['y'])
        elif reference_frame is not None and i >= PRE_TRIGGER_FRAMES:
            # Try flight detector for frames between trigger and first accepted detection
            flt = detect_ball_in_flight(frame, reference_frame, rest_position,
                                        roi_center=rest_position, roi_half=ROI_FLIGHT_HALF)
            if flt:
                tracked_pos = (int(round(flt[0])), int(round(flt[1])))

        if tracked_pos:
            cv2.circle(overlay, tracked_pos, 12, LIGHT_BLUE, 2)
            cv2.circle(overlay, tracked_pos, 2, LIGHT_BLUE, -1)
            cv2.putText(overlay, "TRACKED",
                        (tracked_pos[0] + 15, tracked_pos[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, LIGHT_BLUE, 1)
        else:
            cv2.putText(overlay, "no track", (10, overlay.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 200), 1)

        # Accepted flight-path line (green)
        past_dets = [d for d in detections if d['frame_idx'] <= i]
        if len(past_dets) > 1:
            pts = [(d['x'], d['y']) for d in past_dets]
            for j in range(1, len(pts)):
                cv2.line(overlay, pts[j - 1], pts[j], GREEN, 2)

        # Current accepted flight detection (green ring + dot)
        if i in detection_by_frame:
            det = detection_by_frame[i]
            cv2.circle(overlay, (det['x'], det['y']), 15, GREEN, 2)
            cv2.circle(overlay, (det['x'], det['y']), 4, GREEN, -1)

        # Header bar
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 35), (30, 30, 30), -1)
        lock_tag = " [locked]" if known_rest else ""
        cv2.putText(overlay, f"Frame {i:02d} | {phase.upper()}{lock_tag}", (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, phase_color, 1)

        if "ball_speed" in metrics:
            txt = (f"Speed: {metrics['ball_speed']} mph | Angle: {metrics['launch_angle']} deg "
                   f"| Carry: {metrics['carry_distance']} yds")
            cv2.putText(overlay, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1)

        cv2.imwrite(str(debug_dir / f"debug_{i:04d}.png"), overlay)

    print(f"[Debug] Saved {len(frames)} overlay images to {debug_dir}")


# =============================================================================
# SHOT PROCESSING
# =============================================================================

shots_processed = 0
current_club_gui = ""
current_club_preset = DEFAULT_CLUB


def _motion_process_shot():
    """Process a motion-triggered shot using already-captured frames."""
    global shots_processed
    try:
        shot_id = datetime.now().strftime("shot_%Y%m%d_%H%M%S")
        print(f"[Motion] Processing shot {shot_id}")

        frames, meta = capture_system.get_captured_frames()
        if not frames:
            print("[Motion] No frames captured")
            return

        out_dir = Path(CAPTURE_OUTPUT_DIR) / shot_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(frames):
            cv2.imwrite(str(out_dir / f"frame_{i:04d}.png"), frame)

        meta_path = out_dir / "burst_meta.csv"
        with open(meta_path, 'w', newline='') as f:
            writer = csv_module.writer(f)
            writer.writerow(["frame_idx", "timestamp_us", "phase"])
            for i, m in enumerate(meta):
                phase = "pre" if i < PRE_TRIGGER_FRAMES else "post"
                writer.writerow([i, m.get('timestamp_us', 0), phase])

        # Pass the locked rest position (latched when ball disappeared) so
        # metric calc doesn't have to re-find it in strobe-lit frames.
        rest_pos = capture_system.pre_swing_rest_position or capture_system.rest_position_locked
        metrics = calculate_metrics(frames, meta, known_rest_position=rest_pos,
                                    club_id=current_club_preset)

        # Surface the error to the UI rather than masking it with placeholders.
        # Any valid partial metrics (e.g. ball_speed without launch_angle) are
        # preserved alongside the error string so the UI can show what it has.
        if "error" in metrics:
            print(f"[Motion] Metric calculation failed: {metrics['error']}")

        # Build GUI metrics, converting numpy types to native Python
        gui_metrics = {}
        for k, v in metrics.items():
            if k.startswith("_"):
                continue
            if isinstance(v, (np.floating, np.integer)):
                v = v.item()
            gui_metrics[k] = v
        gui_metrics["shot_id"] = shot_id
        gui_metrics["club_id"] = current_club_preset

        # Write JSON BEFORE debug overlays so the web page picks it up fast
        metrics_json_path = PROJECT_DIR / "WebApplication" / "latest_metrics.json"
        _write_json_atomic(metrics_json_path, gui_metrics)

        print(f"[Motion] Shot complete: {gui_metrics}")

        (out_dir / "metrics.csv").write_text(
            "metric,value,unit\n" +
            "\n".join(f"{k},{v}," for k, v in gui_metrics.items()) + "\n"
        )

        shots_processed += 1
        capture_system.reset_motion()

        # Debug overlays last — slow but non-blocking for the web UI
        try:
            generate_debug_overlays(frames, meta, metrics, out_dir)
        except Exception as e:
            print(f"[Motion] Debug overlay error: {e}")

    except Exception as e:
        print(f"[Motion] Error processing shot: {e}")
        traceback.print_exc()


def process_shot_manual(club_id: str) -> Dict[str, Any]:
    """Manual trigger: trigger capture, then process frames."""
    if not capture_system.trigger(timeout=5.0):
        return {"success": False, "error": "Capture trigger failed"}

    frames, meta = capture_system.get_captured_frames()
    shot_id = datetime.now().strftime("shot_%Y%m%d_%H%M%S")

    out_dir = Path(CAPTURE_OUTPUT_DIR) / shot_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        cv2.imwrite(str(out_dir / f"frame_{i:04d}.png"), frame)

    rest_pos = capture_system.pre_swing_rest_position or capture_system.rest_position_locked
    metrics = calculate_metrics(frames, meta, known_rest_position=rest_pos, club_id=club_id)

    gui_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
    gui_metrics["shot_id"] = shot_id
    gui_metrics["club_id"] = club_id

    metrics_json_path = PROJECT_DIR / "WebApplication" / "latest_metrics.json"
    _write_json_atomic(metrics_json_path, gui_metrics)

    capture_system.reset_motion()
    print(f"[Shot] Manual shot complete: {gui_metrics}")
    return {"success": True, "shot_id": shot_id, "metrics": gui_metrics}


# =============================================================================
# FLASK WEB APPLICATION
# =============================================================================

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
)
app.secret_key = "dev"


@app.after_request
def _set_cache_headers(response):
    # Safari in particular caches JSON aggressively; stale /api/latest_metrics
    # and /api/motion/status responses make the UI look frozen. no-store is
    # safe here — every endpoint is state-dependent.
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
    else:
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


def _write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON via tmp+rename so concurrent readers never see a partial file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def db_connection():
    connection = sqlite3.connect(str(DB_PATH))
    connection.row_factory = sqlite3.Row
    return connection


def initialize_db():
    connection = sqlite3.connect(str(DB_PATH))
    cur = connection.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS player
                   (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL)""")
    cur.execute("SELECT COUNT(*) FROM player")
    if cur.fetchone()[0] == 0:
        for name in ["Player 1", "Player 2", "Player 3", "Player 4"]:
            cur.execute("INSERT INTO player (name) VALUES (?)", (name,))
    connection.commit()
    connection.close()


# ---- Page routes ----

@app.route("/")
def start():
    return render_template("start_page.html")


@app.route("/whos_playing")
def whos_playing():
    connection = db_connection()
    players = connection.execute("SELECT id, name FROM player ORDER BY id").fetchall()
    connection.close()
    return render_template("index.html", players=players)


@app.route("/select_player/<int:player_id>")
def select_player(player_id):
    session["player"] = player_id
    return redirect(url_for("select_club"))


@app.route("/selectclub")
def select_club():
    player_id = session.get("player")
    player_name = None
    if player_id:
        connection = db_connection()
        player = connection.execute("SELECT name FROM player WHERE id = ?",
                                    (player_id,)).fetchone()
        connection.close()
        if player:
            player_name = player["name"]
    return render_template("select_club.html", player_name=player_name)


@app.route("/select_gameplay_club/<path:club>")
def select_gameplay_club(club):
    global current_club_gui, current_club_preset

    session["club"] = club
    current_club_gui = club
    current_club_preset = normalize_club_name(club)

    # Auto-enable motion detection when a club is selected
    capture_system.enable_motion()
    print(f"[App] Club selected: {club} -> {current_club_preset}, motion detection enabled")

    return redirect(url_for("display_metrics"))


@app.route("/display_metrics")
def display_metrics():
    player_id = session.get("player")
    club = session.get("club")

    if not player_id or not club:
        return redirect(url_for("whos_playing"))

    connection = db_connection()
    player = connection.execute("SELECT name FROM player WHERE id = ?",
                                (player_id,)).fetchone()
    connection.close()

    # Load latest metrics if available (don't clear — the JS poll reads them)
    metrics = None
    metrics_path = PROJECT_DIR / "WebApplication" / "latest_metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                content = f.read().strip()
                if content:
                    metrics = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            metrics = None

    # Always render the page — JS will poll for new metrics in the background
    return render_template("metrics.html", player_name=player["name"],
                           club=club, metrics=metrics)


@app.route("/back_to_players")
def back_to_players():
    session.pop("club", None)
    return redirect(url_for("whos_playing"))


@app.route("/add_player", methods=["GET", "POST"])
def add_player():
    connection = db_connection()
    if request.method == "POST":
        name = request.form["name"]
        connection.execute("INSERT INTO player (name) VALUES (?)", (name,))
        connection.commit()
        connection.close()
        return redirect(url_for("whos_playing"))
    connection.close()
    return render_template("add_player.html")


@app.route("/edit_player/<int:player_id>", methods=["GET", "POST"])
def edit_player(player_id):
    connection = db_connection()
    if request.method == "POST":
        name = request.form["name"]
        connection.execute("UPDATE player SET name = ? WHERE id = ?", (name, player_id))
        connection.commit()
        connection.close()
        return redirect(url_for("whos_playing"))
    player = connection.execute("SELECT * FROM player WHERE id = ?", (player_id,)).fetchone()
    connection.close()
    return render_template("edit_player.html", player=player)


@app.route("/edit_users")
def edit_users():
    connection = db_connection()
    players = connection.execute("SELECT * FROM player ORDER BY id").fetchall()
    connection.close()
    return render_template("edit_users.html", players=players)


@app.route("/delete_player/<int:player_id>", methods=["GET", "POST"])
def delete_player(player_id):
    connection = db_connection()
    if request.method == "POST":
        connection.execute("DELETE FROM player WHERE id = ?", (player_id,))
        connection.commit()
        connection.close()
        if session.get("player") == player_id:
            session.pop("player")
        return redirect(url_for("whos_playing"))
    player = connection.execute("SELECT * FROM player WHERE id = ?", (player_id,)).fetchone()
    connection.close()
    return render_template("delete_player.html", player=player)


@app.route("/delete_users")
def delete_users():
    connection = db_connection()
    players = connection.execute("SELECT * FROM player ORDER BY id").fetchall()
    connection.close()
    return render_template("delete_users.html", players=players)


# ---- API routes ----

@app.route("/api/upload_metrics", methods=["POST"])
def api_upload_metrics():
    data = request.get_json()
    metrics_path = PROJECT_DIR / "WebApplication" / "latest_metrics.json"
    _write_json_atomic(metrics_path, data)
    return jsonify({"status": "success"})


@app.route("/api/latest_metrics", methods=["GET"])
def api_latest_metrics():
    """Return latest metrics as JSON (polled by the metrics page JS)."""
    metrics_path = PROJECT_DIR / "WebApplication" / "latest_metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                content = f.read().strip()
                if content:
                    return jsonify(json.loads(content))
        except (json.JSONDecodeError, ValueError):
            pass
    return jsonify({})


@app.route("/api/trigger_shot", methods=["POST"])
def api_trigger_shot():
    result = process_shot_manual(current_club_preset)
    return jsonify(result)


@app.route("/api/motion/enable", methods=["POST"])
def api_motion_enable():
    capture_system.enable_motion()
    return jsonify({"success": True, "motion_enabled": True,
                    "message": "Motion detection enabled — watching for swings"})


@app.route("/api/motion/disable", methods=["POST"])
def api_motion_disable():
    capture_system.disable_motion()
    return jsonify({"success": True, "motion_enabled": False})


@app.route("/api/motion/status", methods=["GET"])
def api_motion_status():
    return jsonify({
        "motion_enabled": capture_system.motion_enabled.is_set(),
        "camera_running": capture_system.running,
        "ball_state": capture_system.ball_state,
        "ball_position": capture_system.ball_position,
    })


@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({
        "mode": "single_pi",
        "camera_running": capture_system.running,
        "motion_enabled": capture_system.motion_enabled.is_set(),
        "current_club": current_club_gui,
        "current_preset": current_club_preset,
        "shots_processed": shots_processed,
    })


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("=" * 60)
    print("GOLF LAUNCH MONITOR — SINGLE PI MODE")
    print("=" * 60)
    print()
    print(f"Web UI:    http://0.0.0.0:{FLASK_PORT}")
    print(f"Camera:    {CAMERA_SIZE} @ {TARGET_FPS}fps "
          f"(detection {DETECTION_EXPOSURE_US}us/g{DETECTION_GAIN} "
          f"/ capture {CAPTURE_EXPOSURE_US}us/g{CAPTURE_GAIN})")
    print(f"Buffer:    {PRE_TRIGGER_FRAMES} pre + {POST_TRIGGER_FRAMES} post frames")
    print(f"Templates: {TEMPLATE_DIR}")
    print(f"Output:    {CAPTURE_OUTPUT_DIR}")
    print()
    print("=" * 60)
    print()

    CAPTURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    initialize_db()

    print("Starting capture system...")
    if capture_system.start():
        print("Capture system running!")
    else:
        print("WARNING: Capture system failed to start")
    print()

    print("Endpoints:")
    print(f"  POST /api/trigger_shot   - Manual trigger")
    print(f"  POST /api/motion/enable  - Enable swing detection")
    print(f"  POST /api/motion/disable - Disable swing detection")
    print(f"  GET  /api/motion/status  - Motion detection status")
    print(f"  GET  /api/status         - System status")
    print()
    print("Open the web UI in a browser to select a player and club.")
    print()

    try:
        if WAITRESS_AVAILABLE:
            print(f"[Server] Starting waitress on {FLASK_HOST}:{FLASK_PORT}...")
            waitress_serve(app, host=FLASK_HOST, port=FLASK_PORT, threads=8)
        else:
            print("[Server] waitress not installed -- falling back to Flask dev server")
            print("         install with: pip install waitress")
            app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
    finally:
        capture_system.stop()
        print("[Server] Goodbye!")


if __name__ == "__main__":
    main()
