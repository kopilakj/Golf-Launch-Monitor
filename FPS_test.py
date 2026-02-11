##This file is a simple test of the frame rate using different camera configuration modes. This one specifically was tested to yield the highest FPS, that is RAW8 mode set at max frames being 300.
##In the end it yielded about 257fps, which is the highest I could get it to achieve. 

from picamera2 import Picamera2
import time

SIZE = (640, 400)
TARGET_FPS = 300
DURATION_SEC = 2.0

picam2 = Picamera2()

# ---- RAW8 configuration ----
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
    "ExposureTime": 994,
    "AnalogueGain": 4.0
})

t0 = time.time()
n = 0
while time.time() - t0 < DURATION_SEC:
    req = picam2.capture_request()
    _ = req.make_array("raw")   # <-- RAW8 frames
    req.release()
    n += 1

elapsed = time.time() - t0
print("Measured RAW FPS:", n / elapsed)

md = picam2.capture_metadata()
print("FrameDuration (us):", md.get("FrameDuration"))
print("ExposureTime (us):", md.get("ExposureTime"))

picam2.close()
