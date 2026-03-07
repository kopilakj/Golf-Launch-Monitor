import csv
from pathlib import Path
from enum import Enum, auto
from collections import deque
import math
import cv2
import numpy as np
# ========================== CONFIG ==========================
# Paths (Pi paths - adjust if running locally)
FRAMES_DIR = Path("/home/jkopila1/project/Golf-Launch-Monitor/Captures/raw_burst")
META_CSV = FRAMES_DIR / "burst_meta.csv"
OUT_TRACK = Path("/home/jkopila1/project/Golf-Launch-Monitor/Calculations/ball_track.csv")
OUT_METRICS = Path("/home/jkopila1/project/Golf-Launch-Monitor/Calculations/metrics.csv")
# Frame dimensions
W, H = 640, 400
# Base crop region (bottom third, shifted right by 1/6)
CROP_X = W // 6              # 106 - left edge starts 1/6 from left
CROP_W = W - CROP_X          # 534 - extends to right edge
BASE_CROP_Y = (2 * H) // 3   # 266 - top edge at 2/3 down (bottom third)
CROP_H = H // 3              # 133 - height is 1/3 of frame
# Ball detection thresholds
AREA_MIN = 50            # Min contour area - lowered for ball at distance
AREA_MAX = 1200          # Max contour area (increased slightly for motion blur)
CIRC_MIN = 0.15          # Min circularity (relaxed for motion blur)
BLUR_K = 5               # Gaussian blur kernel size
PERCENTILE_THRESH = 92   # Top 8% brightness for threshold
# Relaxed detection (for post-impact when ball may be blurred)
RELAXED_AREA_MIN = 30    # Even lower for motion-blurred ball
RELAXED_AREA_MAX = 1800
RELAXED_CIRC_MIN = 0.10
# Rest detection
STABLE_WINDOW = 5        # Frames needed to confirm rest
STABLE_TOL_PX = 2        # Max movement during rest window
# Launch detection
LAUNCH_THRESHOLD_PX = 5  # Min movement from rest to trigger launch
LAUNCH_CONFIRM_FRAMES = 1  # Consecutive frames above threshold (reduced for fast ball)
# Post-launch collection
POST_LAUNCH_FRAMES = 5   # Valid frames to collect after launch
# ROI shift (ball goes UP after launch, so we shift ROI up = decrease crop_y)
SHIFT_PER_FRAME = 25     # Pixels to shift ROI up each frame after launch
MAX_SHIFT_PX = 120       # Maximum total shift (don't go above frame top)
# Impact gap handling
MAX_GAP_FRAMES = 6       # Max consecutive frames without detection before giving up
REACQUIRE_TOL_PX = 40    # Max distance from predicted position to accept reacquisition
REACQUIRE_CONFIRM = 2    # Consecutive detections needed to confirm reacquisition
# Debug output
DEBUG_OVERLAY = True
DEBUG_THRESHOLD = True  # Set to True to save binary threshold images
DEBUG_DIR = Path("/home/jkopila1/project/Golf-Launch-Monitor/Calculations/debug_overlay")
DEBUG_THRESH_DIR = Path("/home/jkopila1/project/Golf-Launch-Monitor/Calculations/debug_threshold")
# ========================== STATE MACHINE ==========================
class State(Enum):
    SEEKING_REST = auto()
    REST_FOUND = auto()
    LAUNCH_DETECTED = auto()
    IMPACT_GAP = auto()
    TRACKING_POST = auto()
    DONE = auto()
    FAILED = auto()
# ========================== HELPER FUNCTIONS ==========================
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
def find_ball(frame_gray, crop_y, last_center=None, relaxed=False, frame_idx=None):
    """
    Detect ball in the cropped region.
    
    Args:
        frame_gray: Full grayscale frame
        crop_y: Current top edge of ROI (shifts upward after launch)
        last_center: Previous ball position for scoring nearby candidates
        relaxed: If True, use relaxed thresholds (for post-impact detection)
        frame_idx: Frame index for debug output
    
    Returns:
        (cx, cy, area, circ) in full-frame coordinates, or None if not found
    """
    # Clamp crop_y to valid range
    crop_y = max(0, min(crop_y, H - CROP_H))
    
    # Extract crop region
    crop = frame_gray[crop_y:crop_y + CROP_H, CROP_X:CROP_X + CROP_W]
    blur = cv2.GaussianBlur(crop, (BLUR_K, BLUR_K), 0)
    
    # Percentile-based threshold (ball is brightest object in ROI)
    # More robust than Otsu when bright lights are present in frame
    threshold_val = np.percentile(blur, PERCENTILE_THRESH)
    _, thresh = cv2.threshold(blur, threshold_val, 255, cv2.THRESH_BINARY)
    
    # Save threshold debug image if enabled
    if DEBUG_THRESHOLD and frame_idx is not None:
        DEBUG_THRESH_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(DEBUG_THRESH_DIR / f"thresh_{frame_idx:04d}.png"), thresh)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Select thresholds based on relaxed mode
    area_min = RELAXED_AREA_MIN if relaxed else AREA_MIN
    area_max = RELAXED_AREA_MAX if relaxed else AREA_MAX
    circ_min = RELAXED_CIRC_MIN if relaxed else CIRC_MIN
    
    best = None
    best_score = -1
    
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
        
        # Centroid in crop coordinates
        cx_crop = int(M["m10"] / M["m00"])
        cy_crop = int(M["m01"] / M["m00"])
        
        # Convert to full-frame coordinates
        fx = cx_crop + CROP_X
        fy = cy_crop + crop_y  # Use dynamic crop_y
        
        # Score: prefer circular, and near last known position
        score = circ
        if last_center is not None:
            dx = fx - last_center[0]
            dy = fy - last_center[1]
            dist2 = dx * dx + dy * dy
            # Penalize distance from last position
            score = circ - (dist2 * 1e-4)
        
        if score > best_score:
            best_score = score
            best = (fx, fy, area, circ)
    
    return best
def distance(p1, p2):
    """Euclidean distance between two points"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
def check_stable_window(positions):
    """
    Check if the last STABLE_WINDOW positions are within STABLE_TOL_PX.
    
    Args:
        positions: List of (cx, cy) tuples
    
    Returns:
        (is_stable, center) where center is median position if stable
    """
    if len(positions) < STABLE_WINDOW:
        return False, None
    
    window = positions[-STABLE_WINDOW:]
    xs = [p[0] for p in window]
    ys = [p[1] for p in window]
    
    cx_med = float(np.median(xs))
    cy_med = float(np.median(ys))
    center = (cx_med, cy_med)
    
    # Check all points are within tolerance of median
    for p in window:
        if distance(p, center) > STABLE_TOL_PX:
            return False, None
    
    return True, center
def compute_metrics(post_launch_data, rest_center):
    """
    Compute ball flight metrics from post-launch detections.
    
    Args:
        post_launch_data: List of (idx, timestamp_us, cx, cy) for post-launch frames
        rest_center: (cx, cy) of ball at rest
    
    Returns:
        Dict of metrics, or None if insufficient data
    """
    if len(post_launch_data) < 2:
        print(f"Not enough post-launch data: {len(post_launch_data)} frames")
        return None
    
    # Sort by frame index
    post_launch_data.sort(key=lambda r: r[0])
    
    # Calculate velocities between consecutive frames
    vxs, vys, speeds = [], [], []
    
    for i in range(1, len(post_launch_data)):
        idx0, t0_us, x0, y0 = post_launch_data[i - 1]
        idx1, t1_us, x1, y1 = post_launch_data[i]
        
        dt = (t1_us - t0_us) / 1e6  # Convert to seconds
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
    
    # Apex: highest point (lowest Y value) in post-launch data
    apex_y = min(r[3] for r in post_launch_data)
    apex_height_px = rest_center[1] - apex_y  # Positive = above rest
    
    # Carry: horizontal distance from rest to last post-launch position
    last_x = post_launch_data[-1][2]
    carry_px = last_x - rest_center[0]
    
    return {
        "ball_speed_pxps": round(speed_med, 2),
        "launch_angle_deg": round(launch_angle_deg, 2),
        "apex_height_px": round(apex_height_px, 2),
        "carry_px": round(carry_px, 2),
        "post_launch_frames": len(post),
        "vx_pxps": round(vx_med, 2),
        "vy_pxps": round(vy_med, 2),
    }
def draw_overlay(frame_gray, crop_y, state, detection, rest_center, frame_idx,
                 gap_counter, post_launch_count, last_center):
    """
    Draw debug overlay showing ROI, detection, and state.
    """
    vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    
    # Clamp crop_y for drawing
    draw_crop_y = max(0, min(crop_y, H - CROP_H))
    
    # Draw ROI rectangle (cyan)
    cv2.rectangle(vis, (CROP_X, draw_crop_y),
                  (CROP_X + CROP_W, draw_crop_y + CROP_H), (255, 255, 0), 1)
    
    # Draw rest center if known (blue cross)
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
    print("=" * 60)
    print("Golf Launch Monitor - Metric Calculation")
    print("=" * 60)
    
    # Load timestamps
    ts = load_timestamps(META_CSV)
    print(f"Loaded {len(ts)} timestamps from {META_CSV}")
    
    # Get frame files
    frames = sorted(FRAMES_DIR.glob("frame_*.png"))
    print(f"Found {len(frames)} frames in {FRAMES_DIR}")
    
    if not frames:
        print("No frames found.")
        return
    
    # Create debug output directories
    if DEBUG_OVERLAY:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    if DEBUG_THRESHOLD:
        DEBUG_THRESH_DIR.mkdir(parents=True, exist_ok=True)
    state = State.SEEKING_REST
    
    # Detection history for rest detection
    detection_history = []  # List of (cx, cy) for recent valid detections
    
    # Rest position
    rest_center = None
    rest_area_ref = None
    rest_frame_idx = None
    launch_frame_idx = None
    frames_since_launch = 0
    
    # Current ROI position (shifts upward after launch)
    current_crop_y = BASE_CROP_Y
    frames_since_launch = 0
    gap_counter = 0
    reacq_counter = 0
    last_center = None
    last_velocity = (0, 0)  # (vx, vy) in px/frame for prediction
    
    # Track all detections for CSV output
    track_rows = []
    for fp in frames:
        idx = int(fp.stem.split("_")[1])
        t_us = ts_map.get(idx, None)
        frame = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if frame is None or t_us is None:
            continue
        
        # ---- Determine if we should use relaxed detection ----
        use_relaxed = state in (State.IMPACT_GAP, State.TRACKING_POST, State.LAUNCH_DETECTED)
        
        # ---- Detect ball ----
        detection = find_ball(frame, current_crop_y, last_center, relaxed=use_relaxed, frame_idx=idx)
        
        # ---- State machine logic ----
        
        if state == State.SEEKING_REST:
            if detection is not None:
                cx, cy, area, circ = detection
                detection_history.append((cx, cy))
                last_center = (cx, cy)
                
                # Check if we have a stable window
                is_stable, center = check_stable_window(detection_history)
                if is_stable:
                    rest_center = center
                    rest_frame_idx = idx
                    state = State.REST_FOUND
                    print(f"[Frame {idx}] REST_FOUND at ({rest_center[0]:.1f}, {rest_center[1]:.1f})")
            else:
                # Reset history if detection fails during seeking
                detection_history = []
        
        elif state == State.REST_FOUND:
            if detection is not None:
                cx, cy, area, circ = detection
                last_center = (cx, cy)
                
                # Check for launch (movement from rest)
                dist_from_rest = distance((cx, cy), rest_center)
                
                if dist_from_rest >= LAUNCH_THRESHOLD_PX:
                    # Ball has moved - launch detected!
                    state = State.LAUNCH_DETECTED
                    launch_frame_idx = idx
                    frames_since_launch = 1
                    
                    # Start collecting post-launch data
                    post_launch_data.append((idx, timestamp_us, cx, cy))
                    
                    # Estimate initial velocity for prediction
                    last_velocity = (cx - rest_center[0], cy - rest_center[1])
                    
                    print(f"[Frame {idx}] LAUNCH_DETECTED! Moved {dist_from_rest:.1f}px from rest")
        
        elif state == State.LAUNCH_DETECTED:
            frames_since_launch += 1
            
            # Shift ROI upward (ball is rising)
            shift_amount = min(frames_since_launch * SHIFT_PER_FRAME, MAX_SHIFT_PX)
            current_crop_y = max(0, BASE_CROP_Y - shift_amount)
            
            if detection is not None:
                cx, cy, area, circ = detection
                
                # Validate detection is reasonable (not too far from predicted)
                detection_valid = True
                if last_center is not None:
                    # Predict where ball should be
                    pred_x = last_center[0] + last_velocity[0]
                    pred_y = last_center[1] + last_velocity[1]
                    dist_from_pred = distance((cx, cy), (pred_x, pred_y))
                    
                    # Accept if within tolerance
                    if dist_from_pred > REACQUIRE_TOL_PX * 2:
                        # Detection too far from expected - might be noise
                        detection_valid = False
                        detection = None  # Mark as invalid
                        gap_counter += 1
                        consecutive_reacquire = 0
                        if gap_counter > MAX_GAP_FRAMES:
                            state = State.FAILED
                            print(f"[Frame {idx}] FAILED - Lost ball for too long")
                        else:
                            state = State.IMPACT_GAP
                            print(f"[Frame {idx}] IMPACT_GAP - Detection too far from expected")
                
                if detection_valid:
                    # Valid detection - update tracking
                    if last_center is not None:
                        last_velocity = (cx - last_center[0], cy - last_center[1])
                    
                    last_center = (cx, cy)
                    post_launch_data.append((idx, timestamp_us, cx, cy))
                    gap_counter = 0
                    
                    # Check if we have enough post-launch frames
                    if len(post_launch_data) >= POST_LAUNCH_FRAMES:
                        state = State.DONE
                        print(f"[Frame {idx}] DONE - Collected {len(post_launch_data)} post-launch frames")
            else:
                rest_miss += 1
                if rest_miss > REST_MISS_TOL:
                    state = State.SEEKING_REST
                    launch_confirm = 0
                    rest_samples.clear()
                    rest_miss = 0
        elif state in (State.LAUNCH_DETECTED, State.IMPACT_GAP, State.TRACKING_POST):
            frames_since_launch += 1
            gap_counter += 1
            
            # Continue shifting ROI upward
            shift_amount = min(frames_since_launch * SHIFT_PER_FRAME, MAX_SHIFT_PX)
            current_crop_y = max(0, BASE_CROP_Y - shift_amount)
            
            if gap_counter > MAX_GAP_FRAMES:
                state = State.FAILED
                print(f"[Frame {idx}] FAILED - Gap exceeded {MAX_GAP_FRAMES} frames")
            elif detection is not None:
                cx, cy, area, circ = detection
                
                # Check if detection is near predicted position
                if last_center is not None:
                    # Predict position based on last velocity * gap frames
                    pred_x = last_center[0] + last_velocity[0] * gap_counter
                    pred_y = last_center[1] + last_velocity[1] * gap_counter
                    dist_from_pred = distance((cx, cy), (pred_x, pred_y))
                    
                    if dist_from_pred <= REACQUIRE_TOL_PX:
                        consecutive_reacquire += 1
                        print(f"[Frame {idx}] Reacquire candidate {consecutive_reacquire}/{REACQUIRE_CONFIRM} "
                              f"(dist={dist_from_pred:.1f}px)")
                        
                        if consecutive_reacquire >= REACQUIRE_CONFIRM:
                            # Reacquired!
                            state = State.TRACKING_POST
                            gap_counter = 0
                    else:
                        # Detection too far - reset consecutive counter
                        consecutive_reacquire = 0
                else:
                    # No last center - accept detection
                    consecutive_reacquire += 1
                    if consecutive_reacquire >= REACQUIRE_CONFIRM:
                        state = State.TRACKING_POST
                        last_center = (cx, cy)
                        post_launch_data.append((idx, timestamp_us, cx, cy))
                        gap_counter = 0
                        consecutive_reacquire = 0
            else:
                consecutive_reacquire = 0
        
        elif state == State.TRACKING_POST:
            frames_since_launch += 1
            
            # Continue shifting ROI upward
            shift_amount = min(frames_since_launch * SHIFT_PER_FRAME, MAX_SHIFT_PX)
            current_crop_y = max(0, BASE_CROP_Y - shift_amount)
            
            if detection is not None:
                cx, cy, area, circ = detection
                
                # Update velocity estimate
                if last_center is not None:
                    last_velocity = (cx - last_center[0], cy - last_center[1])
                
                last_center = (cx, cy)
                post_launch_data.append((idx, timestamp_us, cx, cy))
                gap_counter = 0
                
                # Check if we have enough post-launch frames
                if len(post_launch_data) >= POST_LAUNCH_FRAMES:
                    state = State.DONE
                    print(f"[Frame {idx}] DONE - Collected {len(post_launch_data)} post-launch frames")
            else:
                # Lost again - back to gap
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
            vis = draw_overlay(frame, current_crop_y, state, detection, rest_center,
                               idx, gap_counter, len(post_launch_data), last_center)
            out_vis = DEBUG_DIR / f"frame_{idx:04d}_overlay.png"
            cv2.imwrite(str(out_vis), vis)
        
        # ---- Early exit if done or failed ----
        if state in (State.DONE, State.FAILED):
            # Continue processing remaining frames for overlay but don't update state
            pass
    
    # ---- Write tracking CSV ----
    print("\n" + "=" * 60)
    print("Writing outputs...")
    
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
