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

frame_us = int(1_000_000 / 300)
cam.set_controls({
    "FrameDurationLimits": (frame_us, frame_us),
    "ExposureTime": 200,
})
time.sleep(1)

bus = SMBus(10)

on_msg = i2c_msg.write(0x60, [0x30, 0x09, 0x08])
off_msg = i2c_msg.write(0x60, [0x30, 0x09, 0x00])

bus.i2c_rdwr(i2c_msg.write(0x60, [0x30, 0x06, 0x0C]))
bus.i2c_rdwr(i2c_msg.write(0x60, [0x30, 0x27, 0x08]))

print("Strobing for 30 seconds. Check scope...")
print("Press Ctrl+C to stop")

count = 0
start = time.time()
try:
    while time.time() - start < 30:
        bus.i2c_rdwr(on_msg)
        req = cam.capture_request()
        raw = req.make_array("raw")
        req.release()
        bus.i2c_rdwr(off_msg)
        count += 1
except KeyboardInterrupt:
    pass

elapsed = time.time() - start
print("%d frames in %.1f sec = %.1f fps" % (count, elapsed, count/elapsed))

bus.close()
cam.close()