"""                                                                  
Single frame capture with IR strobe pulse.  
Emulates how the final system fires the strobe for a capture:
- Camera and strobe are initialized once at startup.                     
- On trigger, strobe is turned ON, buffered frames are drained,
    one fresh (illuminated) frame is captured, then strobe is OFF.         
Run on Sensor Pi: python3 strobe_single_capture.py
"""

import time
from picamera2 import Picamera2
from smbus2 import SMBus, i2c_msg
import cv2

# ---------- Camera settings ----------
WIDTH, HEIGHT = 640, 400
TARGET_FPS = 300
EXPOSURE_US = 500
ANALOGUE_GAIN = 4.0
BUFFER_COUNT = 4

# ---------- Strobe (OV9281 I2C) ----------
OV9281_ADDR = 0x60
I2C_BUS = 10

STROBE_INIT_PIN  = [0x30, 0x06, 0x0C]  # FSTROBE pad as output
STROBE_INIT_MODE = [0x30, 0x27, 0x08]  # register-control mode
STROBE_ON        = [0x30, 0x09, 0x08]
STROBE_OFF       = [0x30, 0x09, 0x00]


def init_camera():
    cam = Picamera2()
    config = cam.create_video_configuration(
        raw={"size": (WIDTH, HEIGHT), "format": "R8"},
        buffer_count=BUFFER_COUNT,
    )
    cam.configure(config)
    cam.start()

    frame_us = int(1_000_000 / TARGET_FPS)
    cam.set_controls({
        "FrameDurationLimits": (frame_us, frame_us),
        "ExposureTime": EXPOSURE_US,
        "AnalogueGain": ANALOGUE_GAIN,
    })
    time.sleep(1.0)
    print("Camera ready: %dx%d @ %dfps, exposure %dus, gain %.1f" %
        (WIDTH, HEIGHT, TARGET_FPS, EXPOSURE_US, ANALOGUE_GAIN))
    return cam


def init_strobe(bus):
    bus.i2c_rdwr(i2c_msg.write(OV9281_ADDR, STROBE_INIT_PIN))
    bus.i2c_rdwr(i2c_msg.write(OV9281_ADDR, STROBE_INIT_MODE))
    bus.i2c_rdwr(i2c_msg.write(OV9281_ADDR, STROBE_OFF))
    print("Strobe initialized")


def strobe_on(bus):
    bus.i2c_rdwr(i2c_msg.write(OV9281_ADDR, STROBE_ON))


def strobe_off(bus):
    bus.i2c_rdwr(i2c_msg.write(OV9281_ADDR, STROBE_OFF))


def capture_illuminated_frame(cam, bus):
    """Turn strobe on, drain stale buffered frames, grab one fresh
frame."""
    strobe_on(bus)
    time.sleep(3)
    try:
        # Discard frames captured before the strobe actually turned on.
        for _ in range(BUFFER_COUNT + 1):
            r = cam.capture_request()
            r.release()

        req = cam.capture_request()
        raw = req.make_array("raw").copy()
        req.release()
    finally:
        strobe_off(bus)
    return raw


def main():
    cam = init_camera()
    bus = SMBus(I2C_BUS)
    try:
        init_strobe(bus)

        # In the final system this would be triggered by motion detection.
        raw = capture_illuminated_frame(cam, bus)

        print("Frame captured: shape=%s, min=%d, max=%d, mean=%.1f" %
            (raw.shape, raw.min(), raw.max(), raw.mean()))

        filename = "strobe_capture.png"
        cv2.imwrite(filename, raw)
        print("Saved to %s" % filename)
    finally:
        bus.close()
        cam.close()


if __name__ == "__main__":
    main()