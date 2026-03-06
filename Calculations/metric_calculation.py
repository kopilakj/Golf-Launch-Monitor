import csv
from pathlib import Path
from enum import Enum, auto
from collections import deque
import math
import cv2
import numpy as np
# ========================== CONFIG ==========================
FRAMES_DIR = Path("/home/jkopila1/project/Golf-Launch-Monitor/Captures/raw_burst")
META_CSV = FRAMES_DIR / "burst_meta.csv"
OUT_TRACK = Path("/home/jkopila1/project/Golf-Launch-Monitor/Calculations/ball_track.csv")
OUT_METRICS = Path("/home/jkopila1/project/Golf-Launch-Monitor/Calculations/metrics.csv")
W, H = 640, 400
# ROI: bottom third, shifted right by 1/6
CROP_X = W // 6
CROP_W = W - CROP_X
BASE_CROP_Y = (3 * H) // 5
CROP_H = H // 3
# Detection thresholds
AREA_MIN = 20
AREA_MAX = 2000
CIRC_MIN = 0.03
BLUR_K = 5
RELAXED_AREA_MIN = 12
RELAXED_AREA_MAX = 2600
RELAXED_CIRC_MIN = 0.04
# Brightness / shape filters
MIN_MEAN_BRIGHTNESS = 75.0
ASPECT_MIN = 0.20
ASPECT_MAX = 4.0
# Thresholding (dual path)
PERCENTILE_THRESH = 90
ADAPTIVE_BLOCK = 21
ADAPTIVE_C = -4
# Motion/continuity gating
REST_GATE_RADIUS = 28
POST_GATE_RADIUS = 75
MAX_JUMP_PX = 90
DIST_PENALTY = 0.0035
# Rest / launch
STABLE_WINDOW = 5
STABLE_TOL_PX = 2
STABLE_AREA_STD_MAX = 55.0
REST_MISS_TOL = 2
LAUNCH_THRESHOLD_PX = 5
LAUNCH_CONFIRM_FRAMES = 1
LAUNCH_MIN_VX = 0     # px/s
LAUNCH_MAX_VY = 999999     # px/s (up in image is negative y)
POST_LAUNCH_FRAMES = 5
# ROI shift
SHIFT_PER_FRAME = 25
MAX_SHIFT_PX = 120
# Gap / reacquire
MAX_GAP_FRAMES = 6
REACQUIRE_TOL_PX = 60
REACQUIRE_CONFIRM = 2
# Debug
DEBUG_OVERLAY = True
DEBUG_THRESHOLD = True
DEBUG_DIR = Path("/home/jkopila1/project/Golf-Launch-Monitor/Calculations/debug_overlay")
DEBUG_THRESH_DIR = Path("/home/jkopila1/project/Golf-Launch-Monitor/Calculations/debug_threshold")
# ============================================================
class State(Enum):
    SEEKING_REST = auto()
    REST_FOUND = auto()
    LAUNCH_DETECTED = auto()
    IMPACT_GAP = auto()
    TRACKING_POST = auto()
    DONE = auto()
    FAILED = auto()
def load_timestamps(meta_path):
    ts = {}
    with meta_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts[int(row["frame_idx"])] = float(row["timestamp_us"])
    return ts
def circularity(area, perimeter):
    if perimeter <= 0:
        return 0.0
    return 4.0 * math.pi * area / (perimeter * perimeter)
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])
def clamp_crop_y(y):
    return max(0, min(y, H - CROP_H))
def build_masks(blur):
    # Mask 1: percentile threshold
    pval = np.percentile(blur, PERCENTILE_THRESH)
    _, m1 = cv2.threshold(blur, pval, 255, cv2.THRESH_BINARY)
    # Mask 2: adaptive threshold
    m2 = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ADAPTIVE_BLOCK, ADAPTIVE_C
    )
    # Small cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m1 = cv2.morphologyEx(m1, cv2.MORPH_OPEN, k)
    m2 = cv2.morphologyEx(m2, cv2.MORPH_OPEN, k)
    return m1, m2
def contour_candidates(crop, mask, crop_y, area_min, area_max, circ_min, source):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_min or area > area_max:
            continue
        peri = cv2.arcLength(cnt, True)
        circ = circularity(area, peri)
        if circ < circ_min:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if h <= 0:
            continue
        aspect = w / float(h)
        if aspect < ASPECT_MIN or aspect > ASPECT_MAX:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # Mean brightness inside contour
        local_mask = np.zeros(crop.shape, dtype=np.uint8)
        cv2.drawContours(local_mask, [cnt], -1, 255, -1)
        mean_b = cv2.mean(crop, mask=local_mask)[0]
        if mean_b < MIN_MEAN_BRIGHTNESS:
            continue
        out.append({
            "cx": cx + CROP_X,
            "cy": cy + crop_y,
            "area": float(area),
            "circ": float(circ),
            "aspect": float(aspect),
            "mean_b": float(mean_b),
            "source": source,
        })
    return out
def dedupe_candidates(cands, px=6):
    deduped = []
    for c in cands:
        keep = True
        for d in deduped:
            if abs(c["cx"] - d["cx"]) <= px and abs(c["cy"] - d["cy"]) <= px:
                # Keep stronger circularity candidate
                if c["circ"] > d["circ"]:
                    d.update(c)
                keep = False
                break
        if keep:
            deduped.append(c)
    return deduped
def select_candidate(cands, state, rest_center, pred_center, last_center):
    if not cands:
        return None, 0.0
    best = None
    best_score = -1e18
    for c in cands:
        p = (c["cx"], c["cy"])
        # Hard gates
        if state in (State.SEEKING_REST, State.REST_FOUND) and rest_center is not None:
            if dist(p, rest_center) > REST_GATE_RADIUS:
                continue
        if state in (State.LAUNCH_DETECTED, State.IMPACT_GAP, State.TRACKING_POST) and pred_center is not None:
            if dist(p, pred_center) > POST_GATE_RADIUS:
                continue
        if last_center is not None and dist(p, last_center) > MAX_JUMP_PX:
            continue
        # Score
        s = 0.0
        s += 2.0 * c["circ"]
        s += 0.003 * c["mean_b"]
        s += 0.002 * min(c["area"], 500.0)
        if pred_center is not None:
            s -= DIST_PENALTY * (dist(p, pred_center) ** 2)
        elif last_center is not None:
            s -= DIST_PENALTY * (dist(p, last_center) ** 2)
        if rest_center is not None and state in (State.SEEKING_REST, State.REST_FOUND):
            s -= 0.001 * (dist(p, rest_center) ** 2)
        if s > best_score:
            best_score = s
            best = c
    if best is None:
        return None, 0.0
    return best, float(best_score)
def detect_ball(frame_gray, crop_y, state, rest_center, pred_center, last_center, relaxed, frame_idx=None):
    crop_y = clamp_crop_y(crop_y)
    crop = frame_gray[crop_y:crop_y + CROP_H, CROP_X:CROP_X + CROP_W]
    blur = cv2.GaussianBlur(crop, (BLUR_K, BLUR_K), 0)
    area_min = RELAXED_AREA_MIN if relaxed else AREA_MIN
    area_max = RELAXED_AREA_MAX if relaxed else AREA_MAX
    circ_min = RELAXED_CIRC_MIN if relaxed else CIRC_MIN
    m1, m2 = build_masks(blur)
    c1 = contour_candidates(crop, m1, crop_y, area_min, area_max, circ_min, "percentile")
    c2 = contour_candidates(crop, m2, crop_y, area_min, area_max, circ_min, "adaptive")
    cands = dedupe_candidates(c1 + c2)
    picked, score = select_candidate(cands, state, rest_center, pred_center, last_center)
    if DEBUG_THRESHOLD and frame_idx is not None:
        DEBUG_THRESH_DIR.mkdir(parents=True, exist_ok=True)
        combo = cv2.bitwise_or(m1, m2)
        cv2.imwrite(str(DEBUG_THRESH_DIR / f"thresh_{frame_idx:04d}.png"), combo)
    return picked, score, len(cands)
def stable_window_ok(rest_samples):
    if len(rest_samples) < STABLE_WINDOW:
        return False, None, None
    pts = [(r["cx"], r["cy"]) for r in rest_samples]
    areas = [r["area"] for r in rest_samples]
    mx = float(np.median([p[0] for p in pts]))
    my = float(np.median([p[1] for p in pts]))
    mc = (mx, my)
    dmax = max(dist(p, mc) for p in pts)
    area_std = float(np.std(areas))
    if dmax <= STABLE_TOL_PX and area_std <= STABLE_AREA_STD_MAX:
        return True, mc, float(np.median(areas))
    return False, None, None
def compute_metrics(post_valid, rest_center):
    if len(post_valid) < 2:
        return None
    post = sorted(post_valid, key=lambda r: r["idx"])[:POST_LAUNCH_FRAMES]
    if len(post) < 2:
        return None
    vxs, vys, speeds = [], [], []
    for i in range(1, len(post)):
        t0 = post[i - 1]["ts"] / 1e6
        t1 = post[i]["ts"] / 1e6
        dt = t1 - t0
        if dt <= 0:
            continue
        dx = post[i]["cx"] - post[i - 1]["cx"]
        dy = post[i]["cy"] - post[i - 1]["cy"]
        vx = dx / dt
        vy = dy / dt
        vxs.append(vx)
        vys.append(vy)
        speeds.append(math.hypot(vx, vy))
    if not speeds:
        return None
    vx_med = float(np.median(vxs))
    vy_med = float(np.median(vys))
    speed_med = float(np.median(speeds))
    apex_y = min(r["cy"] for r in post)
    apex_height_px = rest_center[1] - apex_y
    carry_px = post[-1]["cx"] - rest_center[0]
    launch_angle_deg = math.degrees(math.atan2(-vy_med, vx_med))
    return {
        "ball_speed_pxps": round(speed_med, 2),
        "launch_angle_deg": round(launch_angle_deg, 2),
        "apex_height_px": round(apex_height_px, 2),
        "carry_px": round(carry_px, 2),
        "post_launch_frames": len(post),
        "vx_pxps": round(vx_med, 2),
        "vy_pxps": round(vy_med, 2),
    }
def draw_overlay(frame_gray, crop_y, state, detection, rest_center, pred_center, frame_idx, gap_counter, post_count, cand_count):
    vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    y = clamp_crop_y(crop_y)
    cv2.rectangle(vis, (CROP_X, y), (CROP_X + CROP_W, y + CROP_H), (255, 255, 0), 1)
    if rest_center is not None:
        rc = (int(rest_center[0]), int(rest_center[1]))
        cv2.drawMarker(vis, rc, (255, 0, 0), cv2.MARKER_CROSS, 10, 1)
    if pred_center is not None:
        pc = (int(pred_center[0]), int(pred_center[1]))
        cv2.drawMarker(vis, pc, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 8, 1)
    if detection is not None:
        cx, cy = int(detection["cx"]), int(detection["cy"])
        r = int(max(3, math.sqrt(max(1.0, detection["area"]) / math.pi)))
        cv2.circle(vis, (cx, cy), r, (0, 255, 0), 1)
        cv2.circle(vis, (cx, cy), 2, (0, 0, 255), -1)
        txt = f"{detection['source']} a:{detection['area']:.0f} c:{detection['circ']:.2f}"
        cv2.putText(vis, txt, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.putText(vis, f"State: {state.name}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(vis, f"Frame: {frame_idx}", (W - 130, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(vis, f"ROI shift: {BASE_CROP_Y - y}px", (10, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(vis, f"Gap: {gap_counter}/{MAX_GAP_FRAMES}", (10, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(vis, f"Post-launch: {post_count}/{POST_LAUNCH_FRAMES}", (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(vis, f"Candidates: {cand_count}", (W - 180, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return vis
def main():
    ts_map = load_timestamps(META_CSV)
    frames = sorted(FRAMES_DIR.glob("frame_*.png"))
    if not frames:
        print("No frames found.")
        return
    if DEBUG_OVERLAY:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    if DEBUG_THRESHOLD:
        DEBUG_THRESH_DIR.mkdir(parents=True, exist_ok=True)
    state = State.SEEKING_REST
    rest_samples = deque(maxlen=STABLE_WINDOW)
    rest_miss = 0
    rest_center = None
    rest_area_ref = None
    rest_frame_idx = None
    launch_frame_idx = None
    launch_confirm = 0
    current_crop_y = BASE_CROP_Y
    frames_since_launch = 0
    gap_counter = 0
    reacq_counter = 0
    last_center = None
    last_ts = None
    vx_f = 0.0
    vy_f = 0.0
    post_valid = []
    track_rows = []
    for fp in frames:
        idx = int(fp.stem.split("_")[1])
        t_us = ts_map.get(idx, None)
        frame = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if frame is None or t_us is None:
            continue
        # Predict center from filtered velocity
        pred_center = None
        if last_center is not None and last_ts is not None:
            dt = (t_us - last_ts) / 1e6
            if dt > 0:
                pred_center = (last_center[0] + vx_f * dt, last_center[1] + vy_f * dt)
        relaxed = state in (State.LAUNCH_DETECTED, State.IMPACT_GAP, State.TRACKING_POST)
        det, score, cand_count = detect_ball(
            frame, current_crop_y, state, rest_center, pred_center, last_center, relaxed, frame_idx=idx
        )
        valid = 0
        cx = ""
        cy = ""
        source = ""
        area = ""
        circ = ""
        mean_b = ""
        conf = 0.0
        # ---- State machine ----
        if state == State.SEEKING_REST:
            if det is not None:
                rest_samples.append(det)
                last_center = (det["cx"], det["cy"])
                last_ts = t_us
                rest_miss = 0
                ok, c, a_ref = stable_window_ok(rest_samples)
                if ok:
                    rest_center = c
                    rest_area_ref = a_ref
                    rest_frame_idx = idx
                    state = State.REST_FOUND
                    print(f"[{idx}] REST_FOUND at ({rest_center[0]:.1f}, {rest_center[1]:.1f})")
            else:
                rest_miss += 1
                if rest_miss > REST_MISS_TOL:
                    rest_samples.clear()
        elif state == State.REST_FOUND:
            if det is not None:
                p = (det["cx"], det["cy"])
                d_rest = dist(p, rest_center)
                # velocity estimate
                inst_vx = 0.0
                inst_vy = 0.0
                if last_center is not None and last_ts is not None:
                    dt = (t_us - last_ts) / 1e6
                    if dt > 0:
                        inst_vx = (p[0] - last_center[0]) / dt
                        inst_vy = (p[1] - last_center[1]) / dt
                last_center = p
                last_ts = t_us
                # Launch confirmation uses displacement + directional velocity
                if d_rest >= LAUNCH_THRESHOLD_PX and inst_vx >= LAUNCH_MIN_VX and inst_vy <= LAUNCH_MAX_VY:
                    launch_confirm += 1
                else:
                    launch_confirm = 0
                if launch_confirm >= LAUNCH_CONFIRM_FRAMES:
                    state = State.LAUNCH_DETECTED
                    launch_frame_idx = idx
                    frames_since_launch = 0
                    gap_counter = 0
                    reacq_counter = 0
                    post_valid.append({"idx": idx, "ts": t_us, "cx": p[0], "cy": p[1], "conf": score})
                    print(f"[{idx}] LAUNCH_DETECTED")
            else:
                rest_miss += 1
                if rest_miss > REST_MISS_TOL:
                    state = State.SEEKING_REST
                    launch_confirm = 0
                    rest_samples.clear()
                    rest_miss = 0
        elif state in (State.LAUNCH_DETECTED, State.IMPACT_GAP, State.TRACKING_POST):
            frames_since_launch += 1
            shift_px = min(frames_since_launch * SHIFT_PER_FRAME, MAX_SHIFT_PX)
            current_crop_y = clamp_crop_y(BASE_CROP_Y - shift_px)
            if det is not None:
                p = (det["cx"], det["cy"])
                # continuity check in post states
                pass_continuity = True
                if pred_center is not None and dist(p, pred_center) > REACQUIRE_TOL_PX:
                    pass_continuity = False
                if pass_continuity:
                    # velocity update
                    if last_center is not None and last_ts is not None:
                        dt = (t_us - last_ts) / 1e6
                        if dt > 0:
                            vx_i = (p[0] - last_center[0]) / dt
                            vy_i = (p[1] - last_center[1]) / dt
                            alpha = 0.65
                            vx_f = alpha * vx_f + (1.0 - alpha) * vx_i
                            vy_f = alpha * vy_f + (1.0 - alpha) * vy_i
                    last_center = p
                    last_ts = t_us
                    if state == State.IMPACT_GAP:
                        reacq_counter += 1
                        if reacq_counter >= REACQUIRE_CONFIRM:
                            state = State.TRACKING_POST
                            gap_counter = 0
                    else:
                        state = State.TRACKING_POST
                    post_valid.append({"idx": idx, "ts": t_us, "cx": p[0], "cy": p[1], "conf": score})
                    gap_counter = 0
                else:
                    det = None
            if det is None:
                gap_counter += 1
                reacq_counter = 0
                state = State.IMPACT_GAP
                if gap_counter > MAX_GAP_FRAMES:
                    if len(post_valid) >= 2:
                        state = State.DONE
                    else:
                        state = State.FAILED
            if len(post_valid) >= POST_LAUNCH_FRAMES:
                state = State.DONE
        # ---- row bookkeeping ----
        if det is not None:
            valid = 1
            cx = det["cx"]
            cy = det["cy"]
            source = det["source"]
            area = round(det["area"], 2)
            circ = round(det["circ"], 4)
            mean_b = round(det["mean_b"], 2)
            conf = round(score, 4)
        track_rows.append([
            idx, t_us, cx, cy, valid, state.name, source, area, circ, mean_b, conf, cand_count
        ])
        if DEBUG_OVERLAY:
            vis = draw_overlay(
                frame_gray=frame,
                crop_y=current_crop_y,
                state=state,
                detection=det,
                rest_center=rest_center,
                pred_center=pred_center,
                frame_idx=idx,
                gap_counter=gap_counter,
                post_count=len(post_valid),
                cand_count=cand_count,
            )
            cv2.imwrite(str(DEBUG_DIR / f"frame_{idx:04d}_overlay.png"), vis)
    # Write track CSV
    with OUT_TRACK.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_idx", "timestamp_us", "cx", "cy", "valid", "state",
            "source", "area", "circ", "mean_brightness", "score", "candidate_count"
        ])
        w.writerows(track_rows)
    print(f"Wrote {OUT_TRACK}")
    # Metrics
    if rest_center is None:
        print("No rest center found; cannot compute metrics.")
        return
    metrics = compute_metrics(post_valid, rest_center)
    if metrics is None:
        print("Insufficient valid post-launch points for metrics.")
        return
    metrics["rest_frame_idx"] = rest_frame_idx if rest_frame_idx is not None else ""
    metrics["launch_frame_idx"] = launch_frame_idx if launch_frame_idx is not None else ""
    with OUT_METRICS.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(metrics.keys())
        w.writerow(metrics.values())
    print(f"Wrote {OUT_METRICS}")
    print("Done.")
if __name__ == "__main__":
    main()