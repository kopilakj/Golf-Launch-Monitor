#!/usr/bin/env python3
"""
Ball-at-Rest Detection Calibration Tool
========================================

Captures a live frame from the camera, runs ball detection on it,
and saves an annotated debug image so you can see exactly:

  - What the camera sees (raw grayscale)
  - The motion ROI (cyan box)
  - The ball-rest search area (green box)
  - Every contour found at each threshold (labeled with area, circularity, brightness)
  - Whether ball detection passed or failed, and why each contour was rejected

Usage:
  1. Place the ball on the tee / hitting mat
  2. Run:  python3 calibrate_ball_detection.py
  3. Open the saved image:  calibration_output/cal_<timestamp>.png
  4. Adjust and re-run until detection is solid

Optional flags:
  --no-camera        Use an existing frame image instead of live capture
  --frame PATH       Path to a .png frame to analyze (implies --no-camera)
  --brightness N     Override BRIGHTNESS_THRESHOLD (default 80)
  --exposure N       Override EXPOSURE_US (default 994)
  --gain N           Override ANALOGUE_GAIN (default 4.0)
  --continuous       Keep capturing every 2 seconds until Ctrl+C
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Camera / detection defaults (mirrored from single_pi_server.py)
# ---------------------------------------------------------------------------
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 400
CAMERA_SIZE = (CAPTURE_WIDTH, CAPTURE_HEIGHT)
TARGET_FPS = 300
EXPOSURE_US = 994
ANALOGUE_GAIN = 4.0
BUFFER_COUNT = 4

# Motion ROI (x, y, w, h)
MOTION_ROI = (210, 270, 160, 40)

# Ball detection
BRIGHTNESS_THRESHOLD = 65
MIN_BALL_AREA_REST = 40
MAX_BALL_AREA_REST = 500
MIN_CIRCULARITY_REST = 0.4

# Output directory
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "calibration_output"


def capture_frame(exposure_us=EXPOSURE_US, gain=ANALOGUE_GAIN):
    """Capture a single raw frame from the camera."""
    from picamera2 import Picamera2

    cam = Picamera2()
    config = cam.create_video_configuration(
        raw={"size": CAMERA_SIZE, "format": "R8"},
        buffer_count=BUFFER_COUNT,
    )
    cam.configure(config)
    cam.start()
    time.sleep(0.5)

    frame_us = int(1_000_000 / TARGET_FPS)
    cam.set_controls({
        "FrameDurationLimits": (frame_us, frame_us),
        "ExposureTime": exposure_us,
        "AnalogueGain": gain,
    })
    # Let auto-exposure settle, then drain buffer to get a fresh frame
    time.sleep(0.3)
    for _ in range(BUFFER_COUNT + 1):
        req = cam.capture_request()
        req.release()

    req = cam.capture_request()
    frame = req.make_array("raw").copy()
    req.release()
    cam.close()
    return frame


def get_contour_features(contour, gray_frame=None):
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
        'area': area, 'circularity': circularity,
        'cx': cx, 'cy': cy, 'bbox': (x, y, w, h),
        'brightness': brightness,
    }


def analyze_frame(gray, brightness_threshold):
    """
    Run ball-at-rest detection within the MOTION_ROI only.
    Returns (best_candidate, all_contour_info).
    """
    rx, ry, rw, rh = MOTION_ROI
    roi_crop = gray[ry:ry + rh, rx:rx + rw]

    best_candidate = None
    best_score = -1
    all_contours = []

    thresholds = [brightness_threshold - 15, brightness_threshold, brightness_threshold + 15]
    thresholds = [t for t in thresholds if 10 <= t <= 240]

    for thresh_val in thresholds:
        _, thresh_img = cv2.threshold(roi_crop, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            features = get_contour_features(contour, roi_crop)
            # Convert ROI-local coords to full-frame coords
            full_cx = features['cx'] + rx
            full_cy = features['cy'] + ry

            rejection = None
            if features['area'] <= MIN_BALL_AREA_REST:
                rejection = f"area too small ({features['area']:.0f} <= {MIN_BALL_AREA_REST})"
            elif features['area'] >= MAX_BALL_AREA_REST:
                rejection = f"area too large ({features['area']:.0f} >= {MAX_BALL_AREA_REST})"
            elif features['circularity'] < MIN_CIRCULARITY_REST:
                rejection = f"not circular ({features['circularity']:.2f} < {MIN_CIRCULARITY_REST})"

            info = {
                **features,
                'cx': full_cx, 'cy': full_cy,
                'thresh_val': thresh_val,
                'rejection': rejection,
                'contour': contour,
                'contour_offset': (rx, ry),  # for drawing on full frame
            }
            all_contours.append(info)

            if rejection is None:
                score = features['circularity'] * 100 + features['area']
                if score > best_score:
                    best_score = score
                    best_candidate = info

    return best_candidate, all_contours


def draw_debug_image(gray, best_candidate, all_contours, brightness_threshold):
    """Draw an annotated BGR debug image."""
    # Convert grayscale to BGR so we can draw colored annotations
    debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Brighten the image for visibility (the raw frame may be very dark)
    # Create a brightened version for display only
    display = debug.copy()
    max_val = gray.max()
    if max_val > 0 and max_val < 100:
        scale = min(255.0 / max_val, 5.0)
        display = np.clip(debug.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    # --- Draw MOTION_ROI (cyan) — this IS the search area ---
    rx, ry, rw, rh = MOTION_ROI
    cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 2)
    cv2.putText(display, "MOTION ROI (search area)", (rx, ry - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # --- Draw all rejected contours (red) with labels ---
    for info in all_contours:
        if info.get('rejection') and info is not best_candidate:
            ox, oy = info.get('contour_offset', (0, 0))
            shifted = info['contour'] + np.array([ox, oy])
            cv2.drawContours(display, [shifted], 0, (0, 0, 255), 1)
            label = f"A={info['area']:.0f} C={info['circularity']:.2f}"
            cv2.putText(display, label,
                        (info['bbox'][0] + ox, info['bbox'][1] + oy - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    # --- Draw accepted candidate (bright green, thick) ---
    if best_candidate:
        ox, oy = best_candidate.get('contour_offset', (0, 0))
        shifted = best_candidate['contour'] + np.array([ox, oy])
        cv2.drawContours(display, [shifted], 0, (0, 255, 0), 2)
        cx, cy = best_candidate['cx'], best_candidate['cy']
        cv2.circle(display, (cx, cy), 5, (0, 255, 0), -1)
        label = (f"BALL ({cx},{cy}) A={best_candidate['area']:.0f} "
                 f"C={best_candidate['circularity']:.2f} B={best_candidate['brightness']:.0f}")
        cv2.putText(display, label, (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # --- Info panel at top ---
    panel_lines = [
        f"Brightness thresh: {brightness_threshold} (tries {brightness_threshold-15}, {brightness_threshold}, {brightness_threshold+15})",
        f"Frame stats: min={gray.min()} max={gray.max()} mean={gray.mean():.1f}",
        f"MOTION_ROI: x={rx} y={ry} w={rw} h={rh}",
        f"Area filter: {MIN_BALL_AREA_REST} - {MAX_BALL_AREA_REST}  |  Min circularity: {MIN_CIRCULARITY_REST}",
        f"Contours found: {len(all_contours)}  |  Accepted: {'YES' if best_candidate else 'NO'}",
    ]
    for i, line in enumerate(panel_lines):
        cv2.putText(display, line, (10, 15 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return display


def print_report(gray, best_candidate, all_contours, brightness_threshold):
    """Print a text summary to the terminal."""
    rx, ry, rw, rh = MOTION_ROI
    print("=" * 65)
    print("  BALL-AT-REST CALIBRATION REPORT")
    print("=" * 65)
    print(f"  Frame size:        {gray.shape[1]} x {gray.shape[0]}")
    print(f"  Pixel stats:       min={gray.min()}  max={gray.max()}  mean={gray.mean():.1f}")
    print(f"  Brightness thresh: {brightness_threshold}")
    print(f"  MOTION_ROI:        x={rx} y={ry} w={rw} h={rh}")
    print(f"  Area range:        {MIN_BALL_AREA_REST} - {MAX_BALL_AREA_REST}")
    print(f"  Min circularity:   {MIN_CIRCULARITY_REST}")
    print()

    # ROI pixel stats
    roi_pixels = gray[ry:ry + rh, rx:rx + rw]
    print(f"  ROI pixel stats:   min={roi_pixels.min()}  max={roi_pixels.max()}  mean={roi_pixels.mean():.1f}")

    if roi_pixels.max() < brightness_threshold - 15:
        print()
        print(f"  *** WARNING: ROI max pixel ({roi_pixels.max()}) is below the lowest")
        print(f"      threshold tried ({brightness_threshold - 15}). Nothing can possibly")
        print(f"      be detected. Lower --brightness to ~{max(10, roi_pixels.max() - 10)}")
    print()

    if not all_contours:
        print("  No contours found at any threshold level.")
        print("  The ball is likely too dark to detect. Try:")
        print(f"    --brightness {max(10, int(roi_pixels.mean()))}    (match the ROI mean)")
        print(f"    --exposure 2000                (double the exposure)")
        print(f"    --gain 8.0                     (increase analog gain)")
    else:
        print(f"  Total contours found: {len(all_contours)}")
        print()
        sorted_c = sorted(all_contours, key=lambda c: c['area'], reverse=True)
        for i, c in enumerate(sorted_c[:10]):
            status = "ACCEPTED" if c is best_candidate else f"REJECTED: {c['rejection']}"
            print(f"  [{i+1}] pos=({c['cx']},{c['cy']}) area={c['area']:.0f} "
                  f"circ={c['circularity']:.2f} bright={c['brightness']:.0f} "
                  f"thresh={c['thresh_val']} -> {status}")

    print()
    if best_candidate:
        print(f"  RESULT: Ball detected at ({best_candidate['cx']}, {best_candidate['cy']})")
        print(f"          area={best_candidate['area']:.0f}  circularity={best_candidate['circularity']:.2f}")
    else:
        print("  RESULT: Ball NOT detected")
    print("=" * 65)


def run_calibration(frame, brightness_threshold):
    """Run analysis on a frame and save output."""
    gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    best, all_contours = analyze_frame(gray, brightness_threshold)
    print_report(gray, best, all_contours, brightness_threshold)

    debug_img = draw_debug_image(gray, best, all_contours, brightness_threshold)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = OUTPUT_DIR / f"cal_raw_{timestamp}.png"
    debug_path = OUTPUT_DIR / f"cal_debug_{timestamp}.png"

    cv2.imwrite(str(raw_path), gray)
    cv2.imwrite(str(debug_path), debug_img)
    print(f"\n  Saved raw frame:   {raw_path}")
    print(f"  Saved debug image: {debug_path}")
    return best


def main():
    parser = argparse.ArgumentParser(description="Calibrate ball-at-rest detection")
    parser.add_argument("--no-camera", action="store_true",
                        help="Skip camera capture (use --frame instead)")
    parser.add_argument("--frame", type=str, default=None,
                        help="Path to an existing frame image to analyze")
    parser.add_argument("--brightness", type=int, default=BRIGHTNESS_THRESHOLD,
                        help=f"Brightness threshold (default {BRIGHTNESS_THRESHOLD})")
    parser.add_argument("--exposure", type=int, default=EXPOSURE_US,
                        help=f"Exposure in microseconds (default {EXPOSURE_US})")
    parser.add_argument("--gain", type=float, default=ANALOGUE_GAIN,
                        help=f"Analogue gain (default {ANALOGUE_GAIN})")
    parser.add_argument("--continuous", action="store_true",
                        help="Keep capturing every 2 seconds until Ctrl+C")
    args = parser.parse_args()

    if args.frame:
        frame = cv2.imread(args.frame, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            print(f"Error: could not read image '{args.frame}'")
            sys.exit(1)
        print(f"Loaded frame from {args.frame}")
        run_calibration(frame, args.brightness)
        return

    if args.no_camera:
        print("Error: --no-camera requires --frame")
        sys.exit(1)

    if args.continuous:
        print("Continuous mode — press Ctrl+C to stop\n")
        from picamera2 import Picamera2

        cam = Picamera2()
        config = cam.create_video_configuration(
            raw={"size": CAMERA_SIZE, "format": "R8"},
            buffer_count=BUFFER_COUNT,
        )
        cam.configure(config)
        cam.start()
        time.sleep(0.5)
        frame_us = int(1_000_000 / TARGET_FPS)
        cam.set_controls({
            "FrameDurationLimits": (frame_us, frame_us),
            "ExposureTime": args.exposure,
            "AnalogueGain": args.gain,
        })
        time.sleep(0.3)

        try:
            iteration = 0
            while True:
                # Drain buffer for fresh frame
                for _ in range(BUFFER_COUNT + 1):
                    req = cam.capture_request()
                    req.release()
                req = cam.capture_request()
                frame = req.make_array("raw").copy()
                req.release()

                iteration += 1
                print(f"\n--- Capture #{iteration} ---")
                run_calibration(frame, args.brightness)
                time.sleep(2)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            cam.close()
    else:
        print("Capturing single frame...")
        frame = capture_frame(exposure_us=args.exposure, gain=args.gain)
        run_calibration(frame, args.brightness)


if __name__ == "__main__":
    main()
