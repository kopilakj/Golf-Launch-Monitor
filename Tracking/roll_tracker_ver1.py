## This code file is a step up from version 0. Instead of saving images, it utilizes the data from the images that are processed in RAM and sends the necessary data out to flash to then be computed 
## after. The goal with this being to save CPU usage and reduce latency of the camera's imaging and processing.

## TARGET_FPS: higher is better
## STORE_EVERY_NTH_FRAME: keep at 1 for max temporal resolution
## SHOW_PREVIEW: turn off for max FPS during real trials

from picamera2 import Picamera2
import cv2
import time
import os
from datetime import datetime
import math
import csv
from collections import deque

# ================= CONFIG =================

FRAME_SIZE = (640, 400)      # (W, H)
ROI = (80, 80, 480, 240)     # (x, y, w, h) in full-frame coords

# Motion detection (used only to detect roll start/end in realtime)
DIFF_THRESHOLD = 12
DILATE_ITERS = 1
MOTION_PIXELS_THRESHOLD = 250

# Offline centroid extraction
MIN_CONTOUR_AREA = 30
CENTER_EMA_ALPHA = 0.2

# Roll grouping
ROLL_END_GRACE_SEC = 0.35

# RAM buffer limits (safety)
MAX_ROLL_FRAMES = 600        # stop buffering if roll goes too long
STORE_EVERY_NTH_FRAME = 1    # store every frame; set to 2 to store every other frame

# Output
OUTPUT_DIR = "motion_captures_ver1"
MIN_FRAMES_TO_SAVE_PER_ROLL = 3
SAVE_ALL_BUFFERED_FRAMES = False   # if True, saves every buffered frame; else saves 3 spaced frames

# Preview (throttle to avoid FPS bottleneck)
SHOW_PREVIEW = False
SHOW_EVERY_N_FRAMES = 3

# Camera controls (best-effort)
SET_EXPOSURE = True
EXPOSURE_US = 2000
ANALOG_GAIN = 4.0

SET_TARGET_FPS = True
TARGET_FPS = 60

# ========================================


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def timestamp_str_ms():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def make_roll_dir(base_dir: str):
    roll_dir = os.path.join(base_dir, f"roll_{timestamp_str_ms()}")
    os.makedirs(roll_dir, exist_ok=True)
    return roll_dir


def ema_point(prev, curr, alpha):
    if prev is None:
        return curr
    return (
        int(alpha * prev[0] + (1 - alpha) * curr[0]),
        int(alpha * prev[1] + (1 - alpha) * curr[1]),
    )


def choose_spaced_indices(n: int, k: int):
    """Pick k indices spaced across [0, n-1]. Always unique if n >= k."""
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    # evenly spaced including endpoints
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def offline_extract_centers_and_speeds(roi_frames, times):
    """
    roi_frames: list of uint8 2D arrays (ROI grayscale)
    times: list of timestamps (seconds)
    Returns: centers (list of (cx, cy) or None), speeds_px_s (list of float)
    """
    centers = [None] * len(roi_frames)
    speeds = [0.0] * len(roi_frames)

    prev_center = None

    # Use frame-to-frame diff inside ROI
    prev_gray = None

    for i, gray in enumerate(roi_frames):
        # Light blur for stability
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

        cxcy = None

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray_blur)
            _, thresh = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
            if DILATE_ITERS > 0:
                thresh = cv2.dilate(thresh, None, iterations=DILATE_ITERS)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_cnt = None
            best_area = 0
            for c in contours:
                area = cv2.contourArea(c)
                if area < MIN_CONTOUR_AREA:
                    continue
                if area > best_area:
                    best_area = area
                    best_cnt = c

            if best_cnt is not None:
                M = cv2.moments(best_cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cxcy_raw = (cx, cy)
                    cxcy = ema_point(prev_center, cxcy_raw, CENTER_EMA_ALPHA)

        centers[i] = cxcy

        # speed
        if cxcy is not None and prev_center is not None:
            dt = times[i] - times[i - 1]
            if dt > 0:
                dx = cxcy[0] - prev_center[0]
                dy = cxcy[1] - prev_center[1]
                speeds[i] = math.hypot(dx, dy) / dt

        if cxcy is not None:
            prev_center = cxcy

        prev_gray = gray_blur

    return centers, speeds


def save_roll_outputs(roll_dir, roi_frames, times, centers, speeds, rx, ry):
    """
    Saves either all frames or >=3 spaced frames.
    Saves images as BGR for overlays.
    Writes manifest.csv for saved frames.
    """
    manifest_path = os.path.join(roll_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["saved_idx", "buffer_idx", "t", "cx", "cy", "speed_px_s", "filename"])

        n = len(roi_frames)
        if SAVE_ALL_BUFFERED_FRAMES:
            save_indices = list(range(n))
        else:
            save_indices = choose_spaced_indices(n, MIN_FRAMES_TO_SAVE_PER_ROLL)

        for saved_idx, i in enumerate(save_indices):
            roi = roi_frames[i]
            t = times[i]
            c = centers[i]
            spd = speeds[i]

            # Convert ROI grayscale to BGR
            out = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

            # Draw center (in ROI coordinates)
            if c is not None:
                cv2.circle(out, (c[0], c[1]), 4, (0, 0, 255), -1)

            # Overlay text
            cv2.putText(
                out,
                f"buf:{i}/{n-1}  t:{t:.6f}  speed:{spd:.1f}px/s",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

            filename = f"roi_{i:05d}.jpg"
            path = os.path.join(roll_dir, filename)
            cv2.imwrite(path, out)

            cx = c[0] if c is not None else ""
            cy = c[1] if c is not None else ""
            w.writerow([saved_idx, i, f"{t:.6f}", cx, cy, f"{spd:.2f}", filename])

    # Also write a tiny summary
    summary_path = os.path.join(roll_dir, "summary.txt")
    max_speed = max(speeds) if speeds else 0.0
    with open(summary_path, "w") as sf:
        sf.write(f"Buffered frames: {len(roi_frames)}\n")
        sf.write(f"Saved frames: {'ALL' if SAVE_ALL_BUFFERED_FRAMES else MIN_FRAMES_TO_SAVE_PER_ROLL}\n")
        sf.write(f"Max speed (px/s): {max_speed:.2f}\n")


def main():
    ensure_dir(OUTPUT_DIR)

    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": FRAME_SIZE, "format": "YUV420"})
    picam2.configure(config)
    picam2.start()
    time.sleep(0.5)

    controls = {}
    if SET_EXPOSURE:
        controls["ExposureTime"] = int(EXPOSURE_US)
        controls["AnalogueGain"] = float(ANALOG_GAIN)
    if SET_TARGET_FPS:
        dur = int(1_000_000 / TARGET_FPS)
        controls["FrameDurationLimits"] = (dur, dur)

    if controls:
        try:
            picam2.set_controls(controls)
            print("Applied controls:", controls)
        except Exception as e:
            print("Warning: could not apply some controls:", e)

    W, H = FRAME_SIZE
    rx, ry, rw, rh = ROI

    # Real-time motion detection state (for roll start/end)
    prev_gray_rt = None
    last_motion_time = 0.0
    roll_active = False

    # RAM buffers (ROI only)
    roi_buf = deque()
    t_buf = deque()

    # For preview / FPS
    prev_time = time.time()
    fps_ema = None
    fps_alpha = 0.9
    frame_count = 0
    stored_counter = 0

    print("Running (ROI buffered to RAM). Press ESC or 'q' to quit.")

    try:
        while True:
            now = time.time()
            dt = now - prev_time if prev_time else 0.0

            yuv = picam2.capture_array()
            y_plane = yuv[:H, :W]  # grayscale
            roi = y_plane[ry:ry + rh, rx:rx + rw]

            # Real-time motion detection uses a small blur
            gray_rt = cv2.GaussianBlur(roi, (3, 3), 0)
            motion_pixels = 0

            if prev_gray_rt is not None:
                diff = cv2.absdiff(prev_gray_rt, gray_rt)
                _, thresh = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
                if DILATE_ITERS > 0:
                    thresh = cv2.dilate(thresh, None, iterations=DILATE_ITERS)
                motion_pixels = cv2.countNonZero(thresh)

            motion_present = (motion_pixels > MOTION_PIXELS_THRESHOLD)

            # ---- Roll start ----
            if (not roll_active) and motion_present:
                roll_active = True
                last_motion_time = now
                roi_buf.clear()
                t_buf.clear()
                stored_counter = 0
                print("[ROLL START] buffering ROI frames in RAM")

            # ---- During roll: buffer ROI frames in RAM (no disk writes) ----
            if roll_active:
                if motion_present:
                    last_motion_time = now

                # store every Nth frame to control memory / compute
                stored_counter += 1
                if (stored_counter % STORE_EVERY_NTH_FRAME) == 0:
                    if len(roi_buf) < MAX_ROLL_FRAMES:
                        roi_buf.append(roi.copy())  # copy so next frame doesn't overwrite
                        t_buf.append(now)
                    else:
                        # Safety: stop growing forever
                        print("[WARN] MAX_ROLL_FRAMES reached; stopping buffer growth")
                        # still keep roll_active to end naturally

                # ---- Roll end ----
                if (now - last_motion_time) > ROLL_END_GRACE_SEC:
                    roll_active = False

                    frames = list(roi_buf)
                    times = list(t_buf)

                    print(f"[ROLL END] buffered {len(frames)} ROI frames. Processing offline...")

                    if len(frames) >= 2:
                        centers, speeds = offline_extract_centers_and_speeds(frames, times)
                        roll_dir = make_roll_dir(OUTPUT_DIR)
                        save_roll_outputs(roll_dir, frames, times, centers, speeds, rx, ry)
                        print(f"[ROLL SAVED] {roll_dir}")
                    else:
                        print("[ROLL SKIPPED] not enough buffered frames to analyze")

                    roi_buf.clear()
                    t_buf.clear()

            # FPS estimate
            if dt > 0:
                fps = 1.0 / dt
                fps_ema = fps if fps_ema is None else (fps_alpha * fps_ema + (1 - fps_alpha) * fps)
            
            if fps_ema is not None and frame_count % 30 == 0:
                print(f"FPS: {fps_ema:.1f}")

            # Preview (throttled)
            frame_count += 1
            if SHOW_PREVIEW and (frame_count % SHOW_EVERY_N_FRAMES == 0):
                # Show full frame lightly for aiming ROI, but keep work minimal
                preview = cv2.cvtColor(y_plane, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 1)

                status = "ON" if roll_active else "OFF"
                cv2.putText(
                    preview,
                    f"Roll:{status}  MotionPix:{motion_pixels}  Buf:{len(roi_buf)}  FPS:{0 if fps_ema is None else fps_ema:.1f}",
                    (10, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("ROI RAM Buffer Capture", preview)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

            prev_gray_rt = gray_rt
            prev_time = now

    finally:
        if SHOW_PREVIEW:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        try:
            picam2.close()
        except:
            pass

        print("Exited cleanly.")


if __name__ == "__main__":
    main()
