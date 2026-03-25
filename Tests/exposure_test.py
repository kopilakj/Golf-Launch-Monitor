import subprocess
import sys
import os
import numpy as np
import cv2

WIDTH = 640
HEIGHT = 400
DURATION = 1000
OUTPUT = "/tmp/exposure_test.yuv"

exposure = int(sys.argv[1]) if len(sys.argv) > 1 else 994

print("Capturing at %dus exposure..." % exposure)

cmd = [
    "rpicam-vid", "-t", str(DURATION),
    "--width", str(WIDTH), "--height", str(HEIGHT),
    "--codec", "yuv420", "--framerate", "300",
    "--shutter", str(exposure),
    "--gain", "4.0",
    "-o", OUTPUT
]

result = subprocess.run(cmd, capture_output=True, text=True)
if result.stderr:
    for line in result.stderr.strip().split("\n"):
        if "ERROR" in line or "fps" in line.lower():
            print(line)

if not os.path.exists(OUTPUT):
    print("No output file - capture failed")
    sys.exit(1)

filesize = os.path.getsize(OUTPUT)
# YUV420: 1.5 bytes per pixel
frame_size = WIDTH * HEIGHT * 3 // 2
frame_count = filesize // frame_size
duration_s = DURATION / 1000
fps = frame_count / duration_s

print("%d frames in %.0fs = %.1f fps" % (frame_count, duration_s, fps))    

# Extract first frame (Y plane only = grayscale)
data = np.fromfile(OUTPUT, dtype=np.uint8)
y_plane = data[:WIDTH * HEIGHT].reshape((HEIGHT, WIDTH))

out_name = "exposure_%dus.png" % exposure
cv2.imwrite(out_name, y_plane)
print("Saved %s (min=%d, max=%d, mean=%.1f)" % (out_name, y_plane.min(),   
y_plane.max(), y_plane.mean()))

os.remove(OUTPUT)