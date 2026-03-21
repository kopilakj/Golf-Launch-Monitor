from picamera2 import Picamera2
from smbus2 import i2c_msg, SMBus
import time

cam = Picamera2()
config = cam.create_video_configuration(raw={"size": (640, 400), "format": 
"R8"}, buffer_count=4)
cam.configure(config)
cam.start()

frame_us = int(1_000_000 / 30)
cam.set_controls({
    "FrameDurationLimits": (frame_us, frame_us),
    "ExposureTime": 200,
})
time.sleep(1)

bus = SMBus(10)

def write_reg(addr, reg, val):
    hi = (reg >> 8) & 0xFF
    lo = reg & 0xFF
    msg = i2c_msg.write(addr, [hi, lo, val])
    bus.i2c_rdwr(msg)

def read_reg(addr, reg):
    hi = (reg >> 8) & 0xFF
    lo = reg & 0xFF
    w = i2c_msg.write(addr, [hi, lo])
    r = i2c_msg.read(addr, 1)
    bus.i2c_rdwr(w, r)
    return list(r)[0]

configs = [0x0D, 0x09, 0x0C, 0x08]

for val in configs:
    write_reg(0x60, 0x3006, val)
    write_reg(0x60, 0x3027, 0x00)
    v1 = read_reg(0x60, 0x3006)
    v2 = read_reg(0x60, 0x3027)
    print("0x3006=" + hex(v1) + " 0x3027=" + hex(v2))
    print("Check scope for 10 seconds...")
    time.sleep(10)

bus.close()
cam.close()