import csv
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2

# --- Editable settings ---
SIZE = (640, 400)
TARGET_FPS = 300
PRE_FRAMES = 10
POST_FRAMES = 15

WARMUP_SEC = 1.0
MAX_WAIT_SEC = 0 # Timeout 

ROI = (210, 250, 160, 50) # default is (0, 220, 640, 120)  # x, y, w, h
PIXEL_DIFF_THRESH = 10
CHANGED_PIXELS_THRESHOLD = 500
MIN_CONSECUTIVE = 2

ANALOGUE_GAIN = 4.0
EXPOSURE_US = 994
BUFFER_COUNT = 4

SAVE_DIR_NAME = "raw_burst"
SAVE_FRAMES = True
SAVE_STATS = True
COPY_FRAMES = True


def _timestamp_us(meta):
    ts = None
    if meta is not None:
        ts = meta.get("SensorTimestamp")
    return int(ts // 1000) if ts is not None else time.monotonic_ns() // 1000


def main():
    save_dir = Path(__file__).resolve().parent / SAVE_DIR_NAME
    save_dir.mkdir(parents=True, exist_ok=True)

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        raw={"size": SIZE, "format": "R8"},
        buffer_count=BUFFER_COUNT,
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.5)

    frame_us = int(1_000_000 / TARGET_FPS)
    picam2.set_controls(
        {
            "FrameDurationLimits": (frame_us, frame_us),
            "ExposureTime": EXPOSURE_US,
            "AnalogueGain": ANALOGUE_GAIN,
        }
    )

    rx, ry, rw, rh = ROI
    t_start = time.time()

    prebuffer = deque(maxlen=PRE_FRAMES)
    pre_meta = deque(maxlen=PRE_FRAMES)
    prev_roi = None
    consec = 0
    triggered = False
    trigger_changed = 0

    captured_frames = []
    captured_meta = []

    print("Waiting for trigger...")

    while True:
        req = picam2.capture_request()
        meta = req.get_metadata()
        raw = req.make_array("raw")
        if COPY_FRAMES:
            raw = raw.copy()
        req.release()

        prebuffer.append(raw)
        pre_meta.append(meta)

        roi = raw[ry : ry + rh, rx : rx + rw]

        if time.time() - t_start < WARMUP_SEC:
            prev_roi = roi.copy()
            continue

        if not triggered and prev_roi is not None:
            diff = cv2.absdiff(roi, prev_roi)
            _, diff_bin = cv2.threshold(diff, PIXEL_DIFF_THRESH, 255, cv2.THRESH_BINARY)
            changed = int(np.count_nonzero(diff_bin))

            consec = consec + 1 if changed > CHANGED_PIXELS_THRESHOLD else 0

            if consec >= MIN_CONSECUTIVE:
                triggered = True
                trigger_changed = changed
                print(f"Triggered (changed={changed}). Capturing burst to RAM...")

                captured_frames.extend(list(prebuffer))
                captured_meta.extend(list(pre_meta))

                for _ in range(POST_FRAMES):
                    req2 = picam2.capture_request()
                    meta2 = req2.get_metadata()
                    raw2 = req2.make_array("raw")
                    if COPY_FRAMES:
                        raw2 = raw2.copy()
                    req2.release()
                    captured_frames.append(raw2)
                    captured_meta.append(meta2)

                break

        prev_roi = roi.copy()

        if MAX_WAIT_SEC and time.time() - t_start > MAX_WAIT_SEC:
            print("Timeout waiting for trigger.")
            break

    picam2.close()

    if not captured_frames:
        print("No frames captured.")
        return

    print(f"Captured {len(captured_frames)} frames. Saving...")

    if SAVE_FRAMES:
        for i, frame in enumerate(captured_frames):
            out_path = save_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(out_path), frame)

    if SAVE_STATS:
        trigger_index = max(0, len(captured_frames) - POST_FRAMES)
        stats_path = save_dir / "burst_meta.csv"
        with stats_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_idx", "timestamp_us", "phase", "trigger_changed"])
            for i, meta in enumerate(captured_meta):
                phase = "pre" if i < trigger_index else "post"
                writer.writerow([i, _timestamp_us(meta), phase, trigger_changed])

    print("Done.")


if __name__ == "__main__":
    main()
