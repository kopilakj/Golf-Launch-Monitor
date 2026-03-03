import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
# --- CONFIG ---
W, H = 640, 400
FPS = 300
DURATION_MS = 1000  # 1 second
USE_SEGMENTS = False  # single file mode
# --------------
def run_capture(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    pts_path = out_dir / "capture.pts"
    out_file = out_dir / "capture.yuv"
    cmd = [
        "rpicam-vid",
        "--width", str(W),
        "--height", str(H),
        "--framerate", str(FPS),
        "--codec", "yuv420",
        "--denoise", "off",
        "--nopreview",
        "--save-pts", str(pts_path),
        "-t", str(DURATION_MS),
        "-o", str(out_file),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
def extract_frames(out_dir: Path):
    frame_bytes = W * H * 3 // 2  # yuv420
    yuv_path = out_dir / "capture.yuv"
    data = yuv_path.read_bytes()
    num_frames = len(data) // frame_bytes
    print(f"capture.yuv: {num_frames} frames")
    for i in range(num_frames):
        start = i * frame_bytes
        y = np.frombuffer(data[start:start + W*H], dtype=np.uint8).reshape(H, W)
        out_path = out_dir / f"frame_{i:04d}.png"
        cv2.imwrite(str(out_path), y)
    print("Done extracting frames.")
def main():
    tag = datetime.now().strftime("capture_%Y%m%d_%H%M%S")
    out_dir = Path(".") / tag
    run_capture(out_dir)
    extract_frames(out_dir)
if __name__ == "__main__":
    main()