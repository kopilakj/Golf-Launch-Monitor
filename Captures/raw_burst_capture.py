import os, time
import numpy as np
from collections import deque
from picamera2 import Picamera2
import cv2

SAVE_DIR = "raw_burst"
os.makedirs(SAVE_DIR, exist_ok=True)

SIZE = (640, 400)
TARGET_FPS = 300

PRE_FRAMES  = 30
POST_FRAMES = 90

# Trigger tuning
WARMUP_SEC = 1.0
ROI = (0, 220, 640, 120)  # x,y,w,h
PIXEL_DIFF_THRESH = 15
CHANGED_PIXELS_THRESHOLD = 2500
MIN_CONSECUTIVE = 2

picam2 = Picamera2()
config = picam2.create_video_configuration(
    raw={"size": SIZE, "format": "R8"},
    buffer_count=4
)
picam2.configure(config)
picam2.start()
time.sleep(0.5)

frame_us = int(1_000_000 / TARGET_FPS)
picam2.set_controls({
    "FrameDurationLimits": (frame_us, frame_us),
    "ExposureTime": min(1000, frame_us),
    "AnalogueGain": 4.0
})

rx, ry, rw, rh = ROI
t_start = time.time()

prebuffer = deque(maxlen=PRE_FRAMES)  # store full raw frames
prev_roi = None
consec = 0
triggered = False

print("Waiting for trigger...")

captured_frames = []  # will hold pre + post frames in RAM

while True:
    req = picam2.capture_request()
    raw = req.make_array("raw")
    req.release()

    prebuffer.append(raw)

    roi = raw[ry:ry+rh, rx:rx+rw]

    if time.time() - t_start < WARMUP_SEC:
        prev_roi = roi.copy()
        continue

    if prev_roi is not None and not triggered:
        diff = cv2.absdiff(roi, prev_roi)
        _, diff_bin = cv2.threshold(diff, PIXEL_DIFF_THRESH, 255, cv2.THRESH_BINARY)
        changed = int(np.count_nonzero(diff_bin))

        consec = consec + 1 if changed > CHANGED_PIXELS_THRESHOLD else 0

        if consec >= MIN_CONSECUTIVE:
            triggered = True
            print(f"Triggered (changed={changed}). Capturing burst to RAM...")

            # Grab pre-trigger frames from RAM
            captured_frames.extend(list(prebuffer))

            # Capture post-trigger frames to RAM
            for _ in range(POST_FRAMES):
                req2 = picam2.capture_request()
                raw2 = req2.make_array("raw")
                req2.release()
                captured_frames.append(raw2)

            break

    prev_roi = roi.copy()

picam2.close()
print(f"Captured {len(captured_frames)} frames. Now saving to disk...")

# Save AFTER capture so we don't drop frames mid-burst
for i, frame in enumerate(captured_frames):
    cv2.imwrite(f"{SAVE_DIR}/frame_{i:04d}.png", frame)

print("Done saving.")
