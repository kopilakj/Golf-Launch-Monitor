#!/usr/bin/env python3
"""
Single frame capture with IR strobe pulse.
Captures one image with the strobe firing, saves it as a PNG.
Run on Sensor Pi: python3 strobe_single_capture.py
"""

import time
from picamera2 import Picamera2
from smbus2 import SMBus, i2c_msg
import cv2
import numpy as np

# Camera settings
WIDTH, HEIGHT = 640, 400
TARGET_FPS = 300
EXPOSURE_US = 994

# Setup camera
cam = Picamera2()
config = cam.create_video_configuration(
    raw={"size": (WIDTH, HEIGHT), "format": "R8"},
    buffer_count=4,
)
cam.configure(config)
cam.start()

frame_us = int(1_000_000 / TARGET_FPS)
cam.set_controls({
    "FrameDurationLimits": (frame_us, frame_us),
    "ExposureTime": EXPOSURE_US,
    "AnalogueGain": 4.0,
})
time.sleep(1)
print("Camera ready: %dx%d @ %dfps, exposure %dus" % (WIDTH, HEIGHT, TARGET_FPS, EXPOSURE_US))

# Setup strobe
bus = SMBus(10)
on_msg = i2c_msg.write(0x60, [0x30, 0x09, 0x08])
off_msg = i2c_msg.write(0x60, [0x30, 0x09, 0x00])

bus.i2c_rdwr(i2c_msg.write(0x60, [0x30, 0x06, 0x0C]))  # strobe pin as output
bus.i2c_rdwr(i2c_msg.write(0x60, [0x30, 0x27, 0x08]))   # register control mode
print("Strobe initialized")

# Capture one frame with strobe
bus.i2c_rdwr(on_msg)
req = cam.capture_request()
raw = req.make_array("raw").copy()
req.release()
bus.i2c_rdwr(off_msg)

print("Frame captured: shape=%s, min=%d, max=%d, mean=%.1f" % (raw.shape, raw.min(), raw.max(), raw.mean()))

# Save the image
filename = "strobe_capture.png"
cv2.imwrite(filename, raw)
print("Saved to %s" % filename)

# Cleanup
bus.close()
cam.close()
