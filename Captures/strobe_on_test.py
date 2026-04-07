#!/usr/bin/env python3
"""Manual strobe ON test - turns the IR LED on for 3 seconds.
The camera must be initialized first so the OV9281 powers up
and its I2C registers become accessible."""

from picamera2 import Picamera2
from smbus2 import SMBus, i2c_msg
import time

# Start the camera so the sensor powers on (required for I2C access)
cam = Picamera2()
config = cam.create_video_configuration(
    raw={"size": (640, 400), "format": "R8"},
    buffer_count=4,
)
cam.configure(config)
cam.start()
time.sleep(1)
print("Camera started")

# Now the OV9281 will respond on I2C
bus = SMBus(10)

# Init: configure FSTROBE pin and enable register control mode
bus.i2c_rdwr(i2c_msg.write(0x60, [0x30, 0x06, 0x0C]))
bus.i2c_rdwr(i2c_msg.write(0x60, [0x30, 0x27, 0x08]))
print("Strobe initialized")

# Turn ON
print("Strobe ON - point a phone camera at the LED for 3 seconds")
bus.i2c_rdwr(i2c_msg.write(0x60, [0x30, 0x09, 0x08]))

time.sleep(3)

# Turn OFF
bus.i2c_rdwr(i2c_msg.write(0x60, [0x30, 0x09, 0x00]))
print("Strobe OFF")

bus.close()
cam.close()