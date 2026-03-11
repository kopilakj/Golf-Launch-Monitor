#!/usr/bin/env python3
"""
Motion Trigger - Main Pi Swing Detection Service
=================================================

This runs on the MAIN PI and monitors for golf swing motion.
When motion is detected (ball/club passing through ROI), it sends
a TRIGGER message to the Sensor Pi to capture high-speed video.

Based on the swing detection logic from swing_capture_test.py.

Run this with: python3 motion_trigger.py

Architecture:
    Main Pi (this script)          Sensor Pi
    =====================          ==========
    Camera monitors ROI    ---->   TRIGGER message
    Detects motion         <----   CSV metrics response
    Sends trigger
"""

import sys
import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
from picamera2 import Picamera2

# Import our communication modules
from config import (
    MOTION_CAMERA_SIZE, MOTION_TARGET_FPS, MOTION_EXPOSURE_US,
    MOTION_ANALOGUE_GAIN, MOTION_BUFFER_COUNT,
    MOTION_ROI, MOTION_PIXEL_DIFF_THRESH, MOTION_CHANGED_PIXELS_THRESH,
    MOTION_MIN_CONSECUTIVE, MOTION_WARMUP_SEC,
    validate_config
)
from main_client import (
    sensor_conn, state, send_club_preset, trigger_shot, ping_sensor
)


# =============================================================================
# MOTION DETECTOR
# =============================================================================

class MotionDetector:
    """
    Detects motion in camera frames using ROI-based frame differencing.
    
    This monitors a Region of Interest (ROI) in the camera feed and
    triggers when enough pixels change between consecutive frames.
    """
    
    def __init__(
        self,
        roi: Tuple[int, int, int, int] = MOTION_ROI,
        pixel_diff_thresh: int = MOTION_PIXEL_DIFF_THRESH,
        changed_pixels_thresh: int = MOTION_CHANGED_PIXELS_THRESH,
        min_consecutive: int = MOTION_MIN_CONSECUTIVE,
        warmup_sec: float = MOTION_WARMUP_SEC,
    ):
        """
        Initialize motion detector.
        
        Args:
            roi: Region of interest (x, y, width, height)
            pixel_diff_thresh: Pixel intensity difference to count as changed
            changed_pixels_thresh: Number of changed pixels to trigger
            min_consecutive: Consecutive frames with motion required to trigger
            warmup_sec: Seconds to wait before detecting motion
        """
        self.roi = roi  # (x, y, w, h)
        self.pixel_diff_thresh = pixel_diff_thresh
        self.changed_pixels_thresh = changed_pixels_thresh
        self.min_consecutive = min_consecutive
        self.warmup_sec = warmup_sec
        
        # State
        self.prev_roi_frame: Optional[np.ndarray] = None
        self.consecutive_motion: int = 0
        self.start_time: float = 0
        self.is_warmed_up: bool = False
    
    def reset(self):
        """Reset detector state for next shot."""
        self.prev_roi_frame = None
        self.consecutive_motion = 0
        self.start_time = time.time()
        self.is_warmed_up = False
    
    def check_frame(self, frame: np.ndarray) -> Tuple[bool, int]:
        """
        Check a frame for motion.
        
        Args:
            frame: Full camera frame (grayscale)
        
        Returns:
            Tuple of (triggered: bool, changed_pixels: int)
        """
        rx, ry, rw, rh = self.roi
        roi_frame = frame[ry:ry+rh, rx:rx+rw]
        
        # Warmup period - just store frame, don't detect
        if not self.is_warmed_up:
            if time.time() - self.start_time < self.warmup_sec:
                self.prev_roi_frame = roi_frame.copy()
                return False, 0
            else:
                self.is_warmed_up = True
                print("[Motion] Warmup complete, now detecting motion...")
        
        # Need previous frame to compare
        if self.prev_roi_frame is None:
            self.prev_roi_frame = roi_frame.copy()
            return False, 0
        
        # Calculate frame difference
        diff = cv2.absdiff(roi_frame, self.prev_roi_frame)
        _, diff_bin = cv2.threshold(diff, self.pixel_diff_thresh, 255, cv2.THRESH_BINARY)
        changed_pixels = int(np.count_nonzero(diff_bin))
        
        # Check if enough pixels changed
        if changed_pixels > self.changed_pixels_thresh:
            self.consecutive_motion += 1
        else:
            self.consecutive_motion = 0
        
        # Store current frame for next comparison
        self.prev_roi_frame = roi_frame.copy()
        
        # Trigger if enough consecutive frames with motion
        triggered = self.consecutive_motion >= self.min_consecutive
        
        return triggered, changed_pixels


# =============================================================================
# MAIN MOTION TRIGGER SERVICE
# =============================================================================

class MotionTriggerService:
    """
    Main service that monitors camera and triggers Sensor Pi on swing detection.
    """
    
    def __init__(self):
        self.camera: Optional[Picamera2] = None
        self.detector = MotionDetector()
        self.running = False
        self.shots_triggered = 0
        
        # Cooldown to prevent double-triggers
        self.last_trigger_time: float = 0
        self.trigger_cooldown_sec: float = 2.0  # Minimum seconds between triggers
    
    def setup_camera(self):
        """Initialize the camera for motion detection."""
        print("[Motion] Setting up camera...")
        
        self.camera = Picamera2()
        
        config = self.camera.create_video_configuration(
            raw={"size": MOTION_CAMERA_SIZE, "format": "R8"},
            buffer_count=MOTION_BUFFER_COUNT,
        )
        self.camera.configure(config)
        self.camera.start()
        time.sleep(0.5)  # Let camera stabilize
        
        # Set camera controls
        frame_us = int(1_000_000 / MOTION_TARGET_FPS)
        self.camera.set_controls({
            "FrameDurationLimits": (frame_us, frame_us),
            "ExposureTime": MOTION_EXPOSURE_US,
            "AnalogueGain": MOTION_ANALOGUE_GAIN,
        })
        
        print(f"[Motion] Camera ready: {MOTION_CAMERA_SIZE} @ {MOTION_TARGET_FPS}fps")
        print(f"[Motion] ROI: {MOTION_ROI}")
    
    def cleanup_camera(self):
        """Stop and close the camera."""
        if self.camera:
            try:
                self.camera.close()
            except:
                pass
            self.camera = None
    
    def connect_to_sensor(self) -> bool:
        """Connect to the Sensor Pi."""
        print("[Motion] Connecting to Sensor Pi...")
        
        if sensor_conn.connect():
            print("[Motion] Connected to Sensor Pi!")
            return True
        else:
            print("[Motion] Failed to connect to Sensor Pi")
            return False
    
    def on_motion_detected(self, changed_pixels: int):
        """
        Called when motion is detected - triggers shot capture on Sensor Pi.
        
        Args:
            changed_pixels: Number of pixels that changed (for logging)
        """
        # Check cooldown
        now = time.time()
        if now - self.last_trigger_time < self.trigger_cooldown_sec:
            print(f"[Motion] Cooldown active, ignoring trigger")
            return
        
        self.last_trigger_time = now
        self.shots_triggered += 1
        
        print(f"[Motion] ========================================")
        print(f"[Motion] SWING DETECTED! (changed={changed_pixels})")
        print(f"[Motion] Triggering shot #{self.shots_triggered}...")
        print(f"[Motion] ========================================")
        
        # Trigger shot on Sensor Pi
        result = trigger_shot()
        
        if result:
            print(f"[Motion] Shot captured successfully!")
            print(f"[Motion] Shot ID: {result.get('shot_id')}")
            metrics = result.get('metrics', {})
            for key, val in metrics.items():
                if isinstance(val, dict):
                    print(f"[Motion]   {key}: {val.get('value')} {val.get('unit', '')}")
                else:
                    print(f"[Motion]   {key}: {val}")
        else:
            print(f"[Motion] Shot capture failed!")
    
    def run(self):
        """Main loop - monitor camera and trigger on motion."""
        
        print()
        print("=" * 60)
        print("MOTION TRIGGER SERVICE")
        print("=" * 60)
        print()
        
        # Validate config
        if not validate_config():
            print("ERROR: Invalid configuration. Check config.py")
            sys.exit(1)
        
        # Connect to Sensor Pi
        if not self.connect_to_sensor():
            print()
            print("WARNING: Not connected to Sensor Pi!")
            print("Will retry when motion is detected.")
            print()
        
        # Setup camera
        self.setup_camera()
        
        print()
        print("Monitoring for swing motion...")
        print("Press Ctrl+C to stop")
        print()
        
        self.running = True
        self.detector.reset()
        
        try:
            while self.running:
                # Capture frame
                req = self.camera.capture_request()
                raw = req.make_array("raw")
                req.release()
                
                # Check for motion
                triggered, changed = self.detector.check_frame(raw)
                
                if triggered:
                    self.on_motion_detected(changed)
                    # Reset detector after trigger
                    self.detector.reset()
        
        except KeyboardInterrupt:
            print()
            print("[Motion] Shutting down...")
        
        finally:
            self.cleanup_camera()
            sensor_conn.disconnect()
            print(f"[Motion] Total shots triggered: {self.shots_triggered}")
            print("[Motion] Goodbye!")
    
    def run_single_shot(self) -> bool:
        """
        Wait for a single motion event and trigger.
        
        Useful for testing or integration with other code.
        
        Returns:
            True if shot was triggered successfully
        """
        if not self.camera:
            self.setup_camera()
        
        if not sensor_conn.connected:
            self.connect_to_sensor()
        
        print("[Motion] Waiting for motion...")
        self.detector.reset()
        
        while True:
            req = self.camera.capture_request()
            raw = req.make_array("raw")
            req.release()
            
            triggered, changed = self.detector.check_frame(raw)
            
            if triggered:
                self.on_motion_detected(changed)
                return True


# =============================================================================
# STANDALONE USAGE
# =============================================================================

def main():
    """Run the motion trigger service."""
    service = MotionTriggerService()
    service.run()


if __name__ == "__main__":
    main()
