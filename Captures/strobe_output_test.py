from picamera2 import Picamera2
from smbus2 import i2c_msg, SMBus
import time

cam = Picamera2()
config = cam.create_video_configuration(
    raw={"size": (640, 400), "format": "R8"},
    buffer_count=4
)
cam.configure(config)
cam.start()

frame_us = int(1_000_000 / 30)
cam.set_controls({
    "FrameDurationLimits": (frame_us, frame_us),
    "ExposureTime": 1000,
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

# Read all strobe registers first
print("Before:")
for reg in range(0x3920, 0x3930):
    val = read_reg(0x60, reg)
    print("  0x%04X = 0x%02X" % (reg, val))

# Enable strobe output on pin
write_reg(0x60, 0x3006, 0x0C)

# Set large strobe span (pulse width)
# strobe_frame_span[31:0] across 0x3925-0x3928
write_reg(0x60, 0x3925, 0x00)
write_reg(0x60, 0x3926, 0x01)
write_reg(0x60, 0x3927, 0x00)
write_reg(0x60, 0x3928, 0x00)

# Set shift to 0 (pulse starts at integration start)
write_reg(0x60, 0x3921, 0x00)
write_reg(0x60, 0x3922, 0x00)
write_reg(0x60, 0x3923, 0x00)
write_reg(0x60, 0x3924, 0x00)

# Try enabling strobe via 0x3920 and 0x3929
write_reg(0x60, 0x3920, 0x01)
write_reg(0x60, 0x3929, 0x01)

print("\nAfter:")
for reg in range(0x3920, 0x3930):
    val = read_reg(0x60, reg)
    print("  0x%04X = 0x%02X" % (reg, val))

print("\nStrobe configured. Check scope for 30 sec...")
time.sleep(30)

bus.close()
cam.close()