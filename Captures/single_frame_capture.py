import time
from pathlib import Path
import cv2
from picamera2 import Picamera2

SIZE = (640, 400)
TARGET_FPS = 300
EXPOSURE_US = 100
ANALOGUE_GAIN = 4.0
BUFFER_COUNT = 4
SAVE_PATH = Path(__file__).resolve().parent / "pov_test.png"
def main():
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        raw={"size": SIZE, "format": "R8"},
        buffer_count=BUFFER_COUNT,
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.5)
    frame_us = int(1_000_000 / TARGET_FPS)
    picam2.set_controls({
        "FrameDurationLimits": (frame_us, frame_us),
        "ExposureTime": EXPOSURE_US,
        "AnalogueGain": ANALOGUE_GAIN,
    })
    req = picam2.capture_request()
    raw = req.make_array("raw")
    req.release()
    cv2.imwrite(str(SAVE_PATH), raw)
    picam2.close()
    print(f"Saved: {SAVE_PATH}")
if __name__ == "__main__":
    main()