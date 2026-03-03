import csv
from pathlib import Path
import math
import cv2
import numpy as np
# ---- CONFIG ----
FRAMES_DIR = Path("/home/jkopila1/project/Golf-Launch-Monitor/Captures/raw_burst")
META_CSV = FRAMES_DIR / "burst_meta.csv"
OUT_TRACK = FRAMES_DIR / "ball_track.csv"
OUT_METRICS = FRAMES_DIR / "metrics.csv"
W, H = 640, 400  # frame size
# Crop: right 2/3, top 2/3
CROP_X = W // 6
CROP_Y = 0
CROP_W = W 
CROP_H = (3 * H) // 4
# Ball detection params (tune later)
AREA_MIN = 150
AREA_MAX = 1000
CIRC_MIN = 0.4
BLUR_K = 3  # 3x3 blur
# Metrics params
LAUNCH_WINDOW_FRAMES = 8  # number of early frames for launch metrics
# Debug overlay
DEBUG_OVERLAY = True
DEBUG_DIR = FRAMES_DIR / "debug_overlay"
# ----------------
def load_timestamps(meta_path):
    ts = {}
    with meta_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["frame_idx"])
            ts[idx] = float(row["timestamp_us"])
    return ts
def circularity(area, perimeter):
    if perimeter <= 0:
        return 0.0
    return 4.0 * math.pi * area / (perimeter * perimeter)
def find_ball(frame_gray, last_center=None):
    crop = frame_gray[CROP_Y:CROP_Y + CROP_H, CROP_X:CROP_X + CROP_W]
    blur = cv2.GaussianBlur(crop, (BLUR_K, BLUR_K), 0)
    # Otsu threshold (ball is brighter)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < AREA_MIN or area > AREA_MAX:
            continue
        peri = cv2.arcLength(cnt, True)
        circ = circularity(area, peri)
        if circ < CIRC_MIN:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # Convert crop coords to full-frame coords
        fx = cx + CROP_X
        fy = cy + CROP_Y
        score = circ
        if last_center is not None:
            dx = fx - last_center[0]
            dy = fy - last_center[1]
            dist2 = dx*dx + dy*dy
            score = circ - (dist2 * 1e-4)
        if score > best_score:
            best_score = score
            best = (fx, fy, area, circ)
    return best  # (cx, cy, area, circ) or None
def compute_metrics(track_rows):
    # track_rows: list of (idx, t_us, cx, cy, valid)
    valid = [r for r in track_rows if r[4] == 1]
    if len(valid) < 2:
        return None
    valid.sort(key=lambda r: r[0])
    vxs, vys, speeds = [], [], []
    for i in range(1, len(valid)):
        t0 = valid[i-1][1] / 1e6
        t1 = valid[i][1] / 1e6
        dt = t1 - t0
        if dt <= 0:
            continue
        dx = valid[i][2] - valid[i-1][2]
        dy = valid[i][3] - valid[i-1][3]
        vx = dx / dt
        vy = dy / dt
        vxs.append(vx)
        vys.append(vy)
        speeds.append(math.hypot(vx, vy))
    if not speeds:
        return None
    n = min(LAUNCH_WINDOW_FRAMES, len(speeds))
    speed_med = float(np.median(speeds[:n]))
    vx_med = float(np.median(vxs[:n]))
    vy_med = float(np.median(vys[:n]))
    launch_angle_deg = math.degrees(math.atan2(-vy_med, vx_med))
    cy0 = valid[0][3]
    apex_y = min([r[3] for r in valid])
    apex_height_px = cy0 - apex_y
    cx0 = valid[0][2]
    cx_last = valid[-1][2]
    carry_px = cx_last - cx0
    return {
        "ball_speed_pxps": speed_med,
        "launch_angle_deg": launch_angle_deg,
        "apex_height_px": apex_height_px,
        "carry_px": carry_px,
        "valid_frames": len(valid),
    }
def main():
    ts = load_timestamps(META_CSV)
    frames = sorted(FRAMES_DIR.glob("frame_*.png"))
    if DEBUG_OVERLAY:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    last_center = None
    track_rows = []
    for frame_path in frames:
        idx = int(frame_path.stem.split("_")[1])
        frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            continue
        result = find_ball(frame, last_center=last_center)
        if result is None:
            track_rows.append([idx, ts.get(idx, None), "", "", 0])
        else:
            cx, cy, area, circ = result
            track_rows.append([idx, ts.get(idx, None), cx, cy, 1])
            last_center = (cx, cy)
        if DEBUG_OVERLAY:
            vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # Draw crop box
            cv2.rectangle(
                vis,
                (CROP_X, CROP_Y),
                (CROP_X + CROP_W, CROP_Y + CROP_H),
                (255, 255, 0),
                1,
            )
            if result is not None:
                radius = int(np.sqrt(area / np.pi))
                cv2.circle(vis, (cx, cy), radius, (0, 255, 0), 1)
                cv2.circle(vis, (cx, cy), 2, (0, 0, 255), -1)
            out_vis = DEBUG_DIR / f"{frame_path.stem}_overlay.png"
            cv2.imwrite(str(out_vis), vis)
    with OUT_TRACK.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "timestamp_us", "cx", "cy", "valid"])
        writer.writerows(track_rows)
    metrics = compute_metrics(track_rows)
    if metrics is None:
        print("Not enough valid detections to compute metrics.")
        return
    with OUT_METRICS.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(metrics.keys())
        writer.writerow(metrics.values())
    print(f"Wrote: {OUT_TRACK}")
    print(f"Wrote: {OUT_METRICS}")
    if DEBUG_OVERLAY:
        print(f"Wrote overlays to: {DEBUG_DIR}")
if __name__ == "__main__":
    main()