## This code file simply tracks the ball when it is rolled and saves the necessary images to a folder that can be viewed later. It is an initial step to computing the ball speed, but fails in efficiency
## and effectiveness. It is a good starting point and shows how we can use principles of this code to compute the more necessary information later. 

from picamera2 import Picamera2
import cv2
import time
import os
from datetime import datetime
import math
import csv

# ================= CONFIG =================

# Camera output size (W, H)
FRAME_SIZE = (640, 400)

# ROI in (x, y, w, h) relative to FRAME_SIZE
ROI = (80, 80, 480, 240)

# Motion detection thresholds
DIFF_THRESHOLD = 12
DILATE_ITERS = 1
MIN_CONTOUR_AREA = 30

# "Motion present" threshold (changed pixels inside ROI)
MOTION_PIXELS_THRESHOLD = 250

# Speed estimate threshold (for slow roll tests set to 0.0)
SPEED_THRESHOLD_PX_PER_SEC = 0.0

# Centroid smoothing (0 = none, closer to 1 = smoother)
CENTER_EMA_ALPHA = 0.2

# Output
OUTPUT_DIR = "motion_captures"

# Preview throttling (imshow can be an FPS bottleneck)
SHOW_PREVIEW = True
SHOW_EVERY_N_FRAMES = 3  # show 1 out of N frames

# Camera controls (optional)
SET_EXPOSURE = True
EXPOSURE_US = 2000        # microseconds (try 1000–3000)
ANALOG_GAIN = 4.0

SET_TARGET_FPS = True
TARGET_FPS = 60           # will clamp if unsupported

# ---------- Roll grouping ----------
ROLL_END_GRACE_SEC = 0.35       # how long motion can go quiet before roll ends
MIN_FRAMES_TO_KEEP = 3          # keep rolls even if short; target >= 3 frames
SAVE_EVERY_SEC = 0.04           # save ~25 images/sec while roll active (try 0.02–0.06)
# ---------------------------------

# =================================


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def timestamp_str_ms():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def ema_point(prev, curr, alpha):
    if prev is None:
        return curr
    return (
        int(alpha * prev[0] + (1 - alpha) * curr[0]),
        int(alpha * prev[1] + (1 - alpha) * curr[1]),
    )


def make_roll_dir(base_dir):
    roll_dir = os.path.join(base_dir, f"roll_{timestamp_str_ms()}")
    os.makedirs(roll_dir, exist_ok=True)
    return roll_dir


def open_manifest_csv(roll_dir):
    manifest_path = os.path.join(roll_dir, "manifest.csv")
    f = open(manifest_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["frame_idx", "t", "dt", "motion_pixels", "cx", "cy", "speed_px_s", "filename"])
    f.flush()
    return f, w


def main():
    ensure_output_dir(OUTPUT_DIR)

    picam2 = Picamera2()

    # Use YUV420 for speed; process only the Y plane
    config = picam2.create_video_configuration(
        main={"size": FRAME_SIZE, "format": "YUV420"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.5)

    # Apply camera controls (best-effort)
    controls = {}
    if SET_EXPOSURE:
        controls["ExposureTime"] = int(EXPOSURE_US)
        controls["AnalogueGain"] = float(ANALOG_GAIN)

    if SET_TARGET_FPS:
        dur = int(1_000_000 / TARGET_FPS)  # microseconds per frame
        controls["FrameDurationLimits"] = (dur, dur)

    if controls:
        try:
            picam2.set_controls(controls)
            print("Applied controls:", controls)
        except Exception as e:
            print("Warning: could not apply some controls:", e)

    W, H = FRAME_SIZE
    rx, ry, rw, rh = ROI

    prev_gray = None
    prev_center = None
    prev_time = time.time()

    # FPS tracking
    fps_ema = None
    fps_alpha = 0.9

    # ---- Roll state ----
    roll_active = False
    roll_dir = None
    manifest_f = None
    manifest_w = None

    roll_frame_idx = 0
    saved_count = 0
    last_motion_time = 0.0
    last_saved_time = 0.0

    frame_count = 0

    print("Running. Press ESC or 'q' to quit.")
    print("Tip: If FPS is low, set SHOW_PREVIEW=False or increase SHOW_EVERY_N_FRAMES.")

    try:
        while True:
            frame_time = time.time()
            dt = frame_time - prev_time if prev_time else 0.0

            # Capture YUV; Y plane is top H rows, full W columns
            yuv = picam2.capture_array()
            y_plane = yuv[:H, :W]

            # ROI crop
            roi_gray = y_plane[ry:ry + rh, rx:rx + rw]

            # Light blur for noise
            gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)

            motion_pixels = 0
            current_center = None
            estimated_speed = 0.0

            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                _, thresh = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
                if DILATE_ITERS > 0:
                    thresh = cv2.dilate(thresh, None, iterations=DILATE_ITERS)

                motion_pixels = cv2.countNonZero(thresh)

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

                        current_center_raw = (cx, cy)
                        current_center = ema_point(prev_center, current_center_raw, CENTER_EMA_ALPHA)

                        if prev_center is not None and dt > 0:
                            dx = current_center[0] - prev_center[0]
                            dy = current_center[1] - prev_center[1]
                            dist = math.hypot(dx, dy)
                            estimated_speed = dist / dt

            # Motion present for roll grouping
            motion_present = (motion_pixels > MOTION_PIXELS_THRESHOLD)

            # ---- Roll start ----
            if (not roll_active) and motion_present:
                roll_active = True
                roll_dir = make_roll_dir(OUTPUT_DIR)
                manifest_f, manifest_w = open_manifest_csv(roll_dir)

                roll_frame_idx = 0
                saved_count = 0
                last_motion_time = frame_time
                last_saved_time = 0.0  # forces an immediate save on first eligible frame

                print(f"[ROLL START] {roll_dir}")

            # ---- Roll active: save frames on a time cadence ----
            if roll_active:
                if motion_present:
                    last_motion_time = frame_time

                roll_frame_idx += 1

                # Optional speed gating (leave SPEED_THRESHOLD_PX_PER_SEC=0 to disable)
                speed_ok = (estimated_speed >= SPEED_THRESHOLD_PX_PER_SEC)

                # Save based on time cadence (more robust than frame counting)
                if speed_ok and ((frame_time - last_saved_time) >= SAVE_EVERY_SEC):
                    last_saved_time = frame_time

                    filename = f"frame_{roll_frame_idx:05d}.jpg"
                    path = os.path.join(roll_dir, filename)

                    # Convert to BGR only when saving (fast path stays grayscale)
                    debug_bgr = cv2.cvtColor(y_plane, cv2.COLOR_GRAY2BGR)

                    # Draw ROI + center
                    cv2.rectangle(debug_bgr, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 1)
                    if current_center is not None:
                        cv2.circle(
                            debug_bgr,
                            (rx + current_center[0], ry + current_center[1]),
                            4,
                            (0, 0, 255),
                            -1,
                        )

                    cv2.putText(
                        debug_bgr,
                        f"Roll frame {roll_frame_idx}  Motion:{motion_pixels}  Speed:{estimated_speed:.1f}px/s",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                    cv2.imwrite(path, debug_bgr)
                    saved_count += 1

                    cx = current_center[0] if current_center is not None else ""
                    cy = current_center[1] if current_center is not None else ""
                    manifest_w.writerow([
                        roll_frame_idx,
                        f"{frame_time:.6f}",
                        f"{dt:.6f}",
                        int(motion_pixels),
                        cx,
                        cy,
                        f"{estimated_speed:.2f}",
                        filename
                    ])
                    manifest_f.flush()

                # ---- Roll end (grace period) ----
                if (frame_time - last_motion_time) > ROLL_END_GRACE_SEC:
                    roll_active = False

                    if manifest_f:
                        try:
                            manifest_f.close()
                        except:
                            pass
                        manifest_f = None
                        manifest_w = None

                    if saved_count < MIN_FRAMES_TO_KEEP:
                        # Keep it anyway (per your request), but tell you it was short
                        print(f"[ROLL END] saved {saved_count} frames (short roll) in {roll_dir}")
                    else:
                        print(f"[ROLL END] saved {saved_count} frames in {roll_dir}")

                    roll_dir = None

            # FPS overlay estimate
            if dt > 0:
                fps = 1.0 / dt
                fps_ema = fps if fps_ema is None else (fps_alpha * fps_ema + (1 - fps_alpha) * fps)

            # Preview (throttled)
            frame_count += 1
            if SHOW_PREVIEW and (frame_count % SHOW_EVERY_N_FRAMES == 0):
                preview = cv2.cvtColor(y_plane, cv2.COLOR_GRAY2BGR)

                cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 1)
                if current_center is not None:
                    cv2.circle(preview, (rx + current_center[0], ry + current_center[1]), 4, (0, 0, 255), -1)

                roll_txt = "ON" if roll_active else "OFF"
                cv2.putText(
                    preview,
                    f"Roll:{roll_txt}  Saved:{saved_count}  MotionPix:{motion_pixels}  Speed:{estimated_speed:.1f}px/s  FPS:{0 if fps_ema is None else fps_ema:.1f}",
                    (10, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )

                cv2.imshow("Ball Tracking + Roll Grouping", preview)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

            # Update previous state
            prev_gray = gray
            prev_time = frame_time
            if current_center is not None:
                prev_center = current_center  # don't overwrite with None

    finally:
        # Clean up
        if manifest_f:
            try:
                manifest_f.close()
            except:
                pass
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
