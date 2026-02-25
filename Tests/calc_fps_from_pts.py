from pathlib import Path
PTS_FILE = "capture.pts"
pts = []
for line in Path(PTS_FILE).read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    token = line.split()[0]
    pts.append(float(token))
if len(pts) < 2:
    raise SystemExit("Not enough PTS values")
elapsed_s = (pts[-1] - pts[0]) / 1000.0  # convert ms -> seconds
fps = (len(pts) - 1) / elapsed_s
print(f"Frames: {len(pts)}")
print(f"Elapsed (s): {elapsed_s:.6f}")
print(f"FPS: {fps:.2f}")