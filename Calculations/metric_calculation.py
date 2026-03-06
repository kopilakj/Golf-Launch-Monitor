import csv
from pathlib import Path
from enum import Enum, auto
import math
import cv2
import numpy as np
# ========================== CONFIG ==========================
# Path resolution (prefers local raw_burst outside repo)
SCRIPT_PATH = Path(__file__).resolve()
CALC_DIR = SCRIPT_PATH.parent
REPO_ROOT = CALC_DIR.parent
PROJECT_ROOT = REPO_ROOT.parent

FRAMES_DIR_CANDIDATES = [
    PROJECT_ROOT / "raw_burst",
    REPO_ROOT / "Captures" / "raw_burst",
    Path("/home/jkopila1/project/Golf-Launch-Monitor/Captures/raw_burst"),
]

OUT_TRACK = CALC_DIR / "ball_track.csv"
OUT_METRICS = CALC_DIR / "metrics.csv"
# Frame dimensions
W, H = 640, 400
# Base crop region (bottom third, shifted right by 1/6)
CROP_X = W // 6              # 106 - left edge starts 1/6 from left
CROP_W = W - (CROP_X + 200)          # 534 - extends to right edge
BASE_CROP_Y = (3 * H) // 5   # 266 - top edge at 2/3 down (bottom third)
CROP_H = H // 3              # 133 - height is 1/3 of frame
REST_EXTRA_H = 24            # Slightly taller ROI while in REST_FOUND
# Ball detection thresholds
AREA_MIN = 20            # Min contour area - low enough for blurred in-flight ball
AREA_MAX = 1400          # Max contour area
CIRC_MIN = 0.08          # Min circularity
BLUR_K = 5               # Gaussian blur kernel size
PERCENTILE_THRESH = 90   # Top 10% brightness for threshold
# Relaxed detection (for post-impact when ball may be blurred)
RELAXED_AREA_MIN = 12    # Even lower for motion-blurred ball
RELAXED_AREA_MAX = 2200
RELAXED_CIRC_MIN = 0.04
MOTION_DIFF_THRESH = 12   # Used for post-launch motion-based fallback detection
BG_DIFF_THRESH = 16       # Difference from static background threshold
# Rest detection
STABLE_WINDOW = 5        # Frames needed to confirm rest
STABLE_TOL_PX = 4        # Max movement during rest window
SEEKING_MISS_TOL = 6     # Allow brief misses while forming rest window
REST_BOOTSTRAP_LAST_FRAME = 9   # Use early frames to lock rest center
REST_BOOTSTRAP_MIN_POINTS = 3   # Min detections needed for bootstrap lock
# Launch detection
LAUNCH_THRESHOLD_PX = 12  # Min movement from rest to trigger launch
LAUNCH_CONFIRM_FRAMES = 1  # Consecutive frames above threshold (reduced for fast ball)
REST_LOST_LAUNCH_FRAMES = 1  # fallback launch trigger when impact occludes ball
# Post-launch collection
POST_LAUNCH_FRAMES = 3   # Frames used for metric calculations
TRACK_POST_MIN_FRAMES = 8  # Keep tracking longer for overlay/debug quality
# ROI shift (ball goes UP after launch, so we shift ROI up = decrease crop_y)
SHIFT_PER_FRAME = 25     # Pixels to shift ROI up each frame after launch
MAX_SHIFT_PX = 120       # Maximum total shift (don't go above frame top)
# Impact gap handling
MAX_GAP_FRAMES = 8       # Max consecutive frames without detection before giving up
REACQUIRE_TOL_PX = 90    # Max distance from predicted position to accept reacquisition
REACQUIRE_CONFIRM = 1    # Consecutive detections needed to confirm reacquisition
MAX_REST_JUMP_PX = 35    # Reject obvious pre-launch false jumps
DISTANCE_PENALTY = 1e-3  # Stronger tie to previous ball position
REST_GATE_RADIUS_PX = 70 # Keep pre-launch detections near rest center

# Rest search box (full-frame coords) to avoid locking onto floor reflections.
# Keep this tied to ROI geometry so it stays valid when constants change.
REST_SEARCH_X_MIN = CROP_X + 40
REST_SEARCH_X_MAX = CROP_X + min(CROP_W - 30, 190)
REST_SEARCH_Y_MIN = BASE_CROP_Y + 20
REST_SEARCH_Y_MAX = BASE_CROP_Y + CROP_H - 10
POST_MIN_DX_FROM_REST = 15   # Post-launch detections should move right from rest point
POST_MAX_DY_FROM_REST = 25   # Allow some downward noise, but reject large drops
POST_MIN_PROGRESS_X = 8      # Require rightward progress frame-to-frame post-launch
POST_EXCLUDE_BOTTOM_PX = 28  # Reject post-launch detections too close to ROI bottom (club/floor)
POST_SEARCH_X_MIN = 220
POST_SEARCH_X_MAX = 520
POST_SEARCH_Y_MIN = 140
POST_SEARCH_Y_MAX = 320
POST_BACKTRACK_TOL_X = 4
POST_DROP_TOL_Y = 40
POST_AREA_TARGET = 130.0
POST_AREA_MAX = 500.0
POST_MAX_DIST_PRED = 85
POST_MAX_DIST_PRED_GAP = 150
# Debug output
DEBUG_OVERLAY = True
DEBUG_THRESHOLD = True  # Set to True to save binary threshold images
DEBUG_DIR = CALC_DIR / "debug_overlay"
DEBUG_THRESH_DIR = CALC_DIR / "debug_threshold"
# ========================== STATE MACHINE ==========================
class State(Enum):
    SEEKING_REST = auto()
    REST_FOUND = auto()
    LAUNCH_DETECTED = auto()
    IMPACT_GAP = auto()
    TRACKING_POST = auto()
    DONE = auto()
    FAILED = auto()


def resolve_input_paths():
    for candidate in FRAMES_DIR_CANDIDATES:
        meta = candidate / "burst_meta.csv"
        if candidate.exists() and meta.exists() and any(candidate.glob("frame_*.png")):
            return candidate, meta
    raise FileNotFoundError(
        "Could not find raw burst frames. Checked: "
        + ", ".join(str(p) for p in FRAMES_DIR_CANDIDATES)
    )


def build_background(frames, max_frames=5):
    """Build static background from early pre-launch frames (median image)."""
    samples = []
    for fp in frames[:max_frames]:
        img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            samples.append(img)
    if not samples:
        return None
    stack = np.stack(samples, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)

# ========================== HELPER FUNCTIONS ==========================
def load_timestamps(meta_path):
    """Load frame timestamps from burst_meta.csv"""
    ts = {}
    with meta_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["frame_idx"])
            ts[idx] = float(row["timestamp_us"])
    return ts
def circularity(area, perimeter):
    """Calculate circularity: 1.0 = perfect circle"""
    if perimeter <= 0:
        return 0.0
    return 4.0 * math.pi * area / (perimeter * perimeter)
def find_ball(frame_gray, crop_y, crop_h=CROP_H, last_center=None, relaxed=False, frame_idx=None):
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
    # Clamp crop bounds to valid range
    crop_h = max(20, min(crop_h, H))
    crop_y = max(0, min(crop_y, H - crop_h))
    
    # Extract crop region
    crop = frame_gray[crop_y:crop_y + crop_h, CROP_X:CROP_X + CROP_W]
    blur = cv2.GaussianBlur(crop, (BLUR_K, BLUR_K), 0)
    
    # Dual thresholding: percentile + Otsu. This improves robustness when
    # lighting changes and the ball is not always in the top percentile.
    threshold_val = np.percentile(blur, PERCENTILE_THRESH)
    _, thresh_pct = cv2.threshold(blur, threshold_val, 255, cv2.THRESH_BINARY)
    _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_or(thresh_pct, thresh_otsu)
    
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
            score = circ - (dist2 * DISTANCE_PENALTY)
        
        if score > best_score:
            best_score = score
            best = (fx, fy, area, circ)
    
    return best


def find_ball_motion(frame_gray, prev_gray, crop_y, last_center=None, rest_center=None):
    """Fallback detector for post-launch: detect moving bright blob via frame differencing."""
    crop_y = max(0, min(crop_y, H - CROP_H))
    curr = frame_gray[crop_y:crop_y + CROP_H, CROP_X:CROP_X + CROP_W]
    prev = prev_gray[crop_y:crop_y + CROP_H, CROP_X:CROP_X + CROP_W]

    curr_blur = cv2.GaussianBlur(curr, (BLUR_K, BLUR_K), 0)
    prev_blur = cv2.GaussianBlur(prev, (BLUR_K, BLUR_K), 0)
    diff = cv2.absdiff(curr_blur, prev_blur)
    _, motion = cv2.threshold(diff, MOTION_DIFF_THRESH, 255, cv2.THRESH_BINARY)

    k = np.ones((3, 3), np.uint8)
    motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, k)
    motion = cv2.dilate(motion, k, iterations=1)

    contours, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1e9
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 15 or area > POST_AREA_MAX:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if h <= 0:
            continue
        aspect = w / float(h)
        if aspect < 0.45 or aspect > 3.5:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        fx = cx + CROP_X
        fy = cy + crop_y

        if fx < POST_SEARCH_X_MIN or fx > POST_SEARCH_X_MAX:
            continue
        if fy < POST_SEARCH_Y_MIN or fy > POST_SEARCH_Y_MAX:
            continue

        if rest_center is not None:
            dx_rest = fx - rest_center[0]
            dy_rest = fy - rest_center[1]
            up_rest = rest_center[1] - fy
            if (
                dx_rest < POST_MIN_DX_FROM_REST
                or dy_rest > POST_MAX_DY_FROM_REST
            ):
                continue

        if last_center is not None and fx < (last_center[0] - 15):
            continue

        # Motion blobs are less circular; keep a weak circularity prior only.
        peri = cv2.arcLength(cnt, True)
        circ = circularity(area, peri)
        score = 1.2 * circ - 0.01 * abs(area - POST_AREA_TARGET)
        if rest_center is not None:
            score += 0.030 * (fx - rest_center[0]) + 0.012 * (rest_center[1] - fy)
        if last_center is not None:
            d = distance((fx, fy), last_center)
            score -= 0.02 * d

        if score > best_score:
            best_score = score
            best = (fx, fy, area, max(circ, 0.01))

    return best


def find_ball_post_bg_motion(
    frame_gray,
    prev_gray,
    bg_gray,
    crop_y,
    rest_center,
    pred_center=None,
    gap_counter=0,
):
    """
    Post-launch detector using BOTH background subtraction and frame differencing.
    This strongly suppresses static bright objects and reduces club/background jumps.
    """
    if prev_gray is None or bg_gray is None or rest_center is None:
        return None

    crop_y = max(0, min(crop_y, H - CROP_H))
    curr = frame_gray[crop_y:crop_y + CROP_H, CROP_X:CROP_X + CROP_W]
    prev = prev_gray[crop_y:crop_y + CROP_H, CROP_X:CROP_X + CROP_W]
    bg = bg_gray[crop_y:crop_y + CROP_H, CROP_X:CROP_X + CROP_W]

    curr_blur = cv2.GaussianBlur(curr, (BLUR_K, BLUR_K), 0)
    prev_blur = cv2.GaussianBlur(prev, (BLUR_K, BLUR_K), 0)
    bg_blur = cv2.GaussianBlur(bg, (BLUR_K, BLUR_K), 0)

    diff_motion = cv2.absdiff(curr_blur, prev_blur)
    diff_bg = cv2.absdiff(curr_blur, bg_blur)

    _, m_motion = cv2.threshold(diff_motion, MOTION_DIFF_THRESH, 255, cv2.THRESH_BINARY)
    _, m_bg = cv2.threshold(diff_bg, BG_DIFF_THRESH, 255, cv2.THRESH_BINARY)

    # Must be both moving and different from static background.
    mask = cv2.bitwise_and(m_motion, m_bg)
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.dilate(mask, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1e9
    search_radius = REACQUIRE_TOL_PX + (gap_counter * 20)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 12 or area > 700:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if h <= 0:
            continue
        aspect = w / float(h)
        if aspect < 0.4 or aspect > 3.8:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        fx = cx + CROP_X
        fy = cy + crop_y

        if fx < POST_SEARCH_X_MIN or fx > POST_SEARCH_X_MAX:
            continue
        if fy < POST_SEARCH_Y_MIN or fy > POST_SEARCH_Y_MAX:
            continue

        dx_rest = fx - rest_center[0]
        dy_rest = fy - rest_center[1]
        if dx_rest < POST_MIN_DX_FROM_REST or dy_rest > POST_MAX_DY_FROM_REST:
            continue

        if pred_center is not None:
            d_pred = distance((fx, fy), pred_center)
            if d_pred > search_radius:
                continue
        else:
            d_pred = 0.0

        peri = cv2.arcLength(cnt, True)
        circ = circularity(area, peri)

        score = 0.0
        score += 0.03 * dx_rest
        score += 0.012 * (rest_center[1] - fy)
        score += 0.002 * area
        score += 0.8 * max(circ, 0.0)
        score -= 0.015 * d_pred

        if score > best_score:
            best_score = score
            best = (fx, fy, area, max(circ, 0.01))

    return best


def find_ball_post(frame_gray, crop_y, rest_center):
    """Post-launch detector that prefers forward/upward flight candidates."""
    crop_y = max(0, min(crop_y, H - CROP_H))
    crop = frame_gray[crop_y:crop_y + CROP_H, CROP_X:CROP_X + CROP_W]
    blur = cv2.GaussianBlur(crop, (BLUR_K, BLUR_K), 0)

    pval = np.percentile(blur, PERCENTILE_THRESH)
    _, t1 = cv2.threshold(blur, pval, 255, cv2.THRESH_BINARY)
    _, t2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_or(t1, t2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1e9
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 8 or area > 3000:
            continue

        peri = cv2.arcLength(cnt, True)
        circ = circularity(area, peri)
        if circ < 0.02:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        fx = cx + CROP_X
        fy = cy + crop_y

        dx_rest = fx - rest_center[0]
        up = rest_center[1] - fy

        if fx < POST_SEARCH_X_MIN or fx > POST_SEARCH_X_MAX:
            continue
        if fy < POST_SEARCH_Y_MIN or fy > POST_SEARCH_Y_MAX:
            continue

        # Flight must move right and generally up from rest.
        if dx_rest < POST_MIN_DX_FROM_REST:
            continue
        if fy > (rest_center[1] + POST_MAX_DY_FROM_REST):
            continue

        score = (0.04 * dx_rest) + (0.02 * up) + (0.3 * circ) + (0.0005 * area)
        if score > best_score:
            best_score = score
            best = (fx, fy, area, max(circ, 0.01))

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
    if len(post_launch_data) < POST_LAUNCH_FRAMES:
        print(f"Not enough post-launch data: {len(post_launch_data)} frames")
        return None

    # Sort by frame index and use only the first metric frames
    post_launch_data.sort(key=lambda r: r[0])
    post = post_launch_data[:POST_LAUNCH_FRAMES]
    
    # Calculate velocities between consecutive frames
    vxs, vys, speeds = [], [], []
    
    for i in range(1, len(post)):
        idx0, t0_us, x0, y0 = post[i - 1]
        idx1, t1_us, x1, y1 = post[i]
        
        dt = (t1_us - t0_us) / 1e6  # Convert to seconds
        if dt <= 0:
            continue
        
        dx = x1 - x0
        dy = y1 - y0
        vx = dx / dt  # pixels per second
        vy = dy / dt
        
        vxs.append(vx)
        vys.append(vy)
        speeds.append(math.hypot(vx, vy))
    
    if not speeds:
        print("No valid velocity calculations")
        return None
    
    # Use median to reduce outlier impact
    speed_med = float(np.median(speeds))
    vx_med = float(np.median(vxs))
    vy_med = float(np.median(vys))
    
    # Launch angle: negative vy means ball going UP in image coords
    # atan2(-vy, vx) gives angle above horizontal
    launch_angle_deg = math.degrees(math.atan2(-vy_med, vx_med))
    
    # Apex: highest point (lowest Y value) in post-launch data
    apex_y = min(r[3] for r in post)
    apex_height_px = rest_center[1] - apex_y  # Positive = above rest
    
    # Carry: horizontal distance from rest to last post-launch position
    last_x = post[-1][2]
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
                 gap_counter, post_launch_count, last_center, crop_h=CROP_H):
    """
    Draw debug overlay showing ROI, detection, and state.
    """
    vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    
    # Clamp crop_y for drawing
    crop_h = max(20, min(crop_h, H))
    draw_crop_y = max(0, min(crop_y, H - crop_h))
    
    # Draw ROI rectangle (cyan)
    cv2.rectangle(vis, (CROP_X, draw_crop_y),
                  (CROP_X + CROP_W, draw_crop_y + crop_h), (255, 255, 0), 1)
    
    # Draw rest center if known (blue cross)
    if rest_center is not None:
        rc = (int(rest_center[0]), int(rest_center[1]))
        cv2.drawMarker(vis, rc, (255, 0, 0), cv2.MARKER_CROSS, 10, 1)
    
    # Draw last known position if in gap (yellow circle)
    if state == State.IMPACT_GAP and last_center is not None:
        lc = (int(last_center[0]), int(last_center[1]))
        cv2.circle(vis, lc, 8, (0, 255, 255), 1)
    
    # Draw current detection (green circle with red center)
    if detection is not None:
        cx, cy, area, circ = detection
        radius = int(math.sqrt(area / math.pi))
        cv2.circle(vis, (cx, cy), radius, (0, 255, 0), 1)
        cv2.circle(vis, (cx, cy), 2, (0, 0, 255), -1)
    
    # State text (top left)
    state_text = f"State: {state.name}"
    cv2.putText(vis, state_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Frame info (top right)
    info_text = f"Frame: {frame_idx}"
    cv2.putText(vis, info_text, (W - 120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Additional info (bottom left)
    if state == State.IMPACT_GAP:
        gap_text = f"Gap: {gap_counter}/{MAX_GAP_FRAMES}"
        cv2.putText(vis, gap_text, (10, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    post_text = f"Post-launch: {post_launch_count}/{POST_LAUNCH_FRAMES}"
    cv2.putText(vis, post_text, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # ROI shift amount
    shift = BASE_CROP_Y - crop_y
    shift_text = f"ROI shift: {shift}px"
    cv2.putText(vis, shift_text, (10, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    return vis
# ========================== MAIN PROCESSING ==========================
def main():
    print("=" * 60)
    print("Golf Launch Monitor - Metric Calculation")
    print("=" * 60)

    frames_dir, meta_csv = resolve_input_paths()
    print(f"Using frames directory: {frames_dir}")
    
    # Load timestamps
    ts = load_timestamps(meta_csv)
    print(f"Loaded {len(ts)} timestamps from {meta_csv}")
    
    # Get frame files
    frames = sorted(frames_dir.glob("frame_*.png"))
    print(f"Found {len(frames)} frames in {frames_dir}")
    
    if not frames:
        print("ERROR: No frames found!")
        return

    background_gray = build_background(frames, max_frames=5)
    
    # Create debug output directories
    if DEBUG_OVERLAY:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    if DEBUG_THRESHOLD:
        DEBUG_THRESH_DIR.mkdir(parents=True, exist_ok=True)
    
    # ---- State machine variables ----
    state = State.SEEKING_REST
    
    # Detection history for rest detection
    detection_history = []  # List of (cx, cy) for recent valid detections
    seeking_miss_counter = 0
    bootstrap_points = []
    
    # Rest position
    rest_center = None
    rest_frame_idx = None
    
    # Launch tracking
    launch_frame_idx = None
    frames_since_launch = 0
    rest_lost_counter = 0
    
    # Current ROI position (shifts upward after launch)
    current_crop_y = BASE_CROP_Y
    
    # Gap handling
    gap_counter = 0
    consecutive_reacquire = 0
    
    # Post-launch data collection
    post_launch_data = []  # List of (idx, timestamp_us, cx, cy)
    
    # Last known position (for gap handling and reacquisition)
    last_center = None
    last_velocity = (0, 0)  # (vx, vy) in px/frame for prediction
    prev_frame_gray = None
    
    # Track all detections for CSV output
    track_rows = []
    
    print("\nProcessing frames...")
    print("-" * 60)
    
    for frame_path in frames:
        # Extract frame index from filename (e.g., "frame_0001.png" -> 1)
        idx = int(frame_path.stem.split("_")[1])
        timestamp_us = ts.get(idx, None)
        
        # Load frame
        frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            print(f"WARNING: Could not load {frame_path}")
            continue

        # After terminal state, keep writing overlays but stop further detection updates.
        if state in (State.DONE, State.FAILED):
            track_rows.append([idx, timestamp_us, "", "", 0, state.name])
            if DEBUG_OVERLAY:
                done_crop_h = CROP_H
                vis = draw_overlay(frame, current_crop_y, state, None, rest_center,
                                   idx, gap_counter, len(post_launch_data), last_center,
                                   crop_h=done_crop_h)
                out_vis = DEBUG_DIR / f"frame_{idx:04d}_overlay.png"
                cv2.imwrite(str(out_vis), vis)
            prev_frame_gray = frame
            continue
        
        # ---- Determine if we should use relaxed detection ----
        use_relaxed = state in (State.IMPACT_GAP, State.TRACKING_POST, State.LAUNCH_DETECTED)

        # Slightly taller ROI while in REST_FOUND state
        active_crop_h = CROP_H + REST_EXTRA_H if state == State.REST_FOUND else CROP_H
        active_crop_h = min(active_crop_h, H - current_crop_y)

        # ---- Detect ball ----
        detection = find_ball(
            frame,
            current_crop_y,
            crop_h=active_crop_h,
            last_center=last_center,
            relaxed=use_relaxed,
            frame_idx=idx,
        )
        pred_center = None
        if state in (State.LAUNCH_DETECTED, State.IMPACT_GAP, State.TRACKING_POST):
            launch_age = (idx - launch_frame_idx) if launch_frame_idx is not None else 0
            motion_det = None
            post_det = None
            if last_center is not None:
                step = max(1, gap_counter)
                pred_center = (
                    last_center[0] + last_velocity[0] * step,
                    last_center[1] + last_velocity[1] * step,
                )

            bg_motion_det = find_ball_post_bg_motion(
                frame,
                prev_frame_gray,
                background_gray,
                current_crop_y,
                rest_center,
                pred_center=pred_center,
                gap_counter=gap_counter,
            )
            if prev_frame_gray is not None:
                motion_det = find_ball_motion(
                    frame,
                    prev_frame_gray,
                    current_crop_y,
                    last_center=last_center,
                    rest_center=rest_center,
                )
            if rest_center is not None:
                post_det = find_ball_post(frame, current_crop_y, rest_center)

            # Prefer background+motion candidate first (most robust to static clutter),
            # then pure motion, then threshold-based post detector.
            chosen = None
            chosen_d = 0.0
            for det in (bg_motion_det, motion_det, post_det):
                if det is None:
                    continue
                if pred_center is None:
                    d_pred = 0.0
                else:
                    d_pred = distance((det[0], det[1]), pred_center)
                chosen = det
                chosen_d = d_pred
                break

            if chosen is not None:
                max_pred = POST_MAX_DIST_PRED_GAP if state == State.IMPACT_GAP else POST_MAX_DIST_PRED
                if pred_center is None or chosen_d <= max_pred:
                    detection = chosen
                else:
                    detection = None

        # Continuity guard for post-launch states
        if detection is not None and pred_center is not None and state in (State.LAUNCH_DETECTED, State.IMPACT_GAP, State.TRACKING_POST):
            max_pred = POST_MAX_DIST_PRED_GAP if state == State.IMPACT_GAP else POST_MAX_DIST_PRED
            if distance((detection[0], detection[1]), pred_center) > max_pred:
                detection = None

        # Pre-launch spatial filter: if rest center unknown, only consider detections
        # inside the expected tee region to avoid locking onto floor reflections.
        if detection is not None and state in (State.SEEKING_REST, State.REST_FOUND):
            dx, dy = detection[0], detection[1]
            if rest_center is None:
                if not (
                    REST_SEARCH_X_MIN <= dx <= REST_SEARCH_X_MAX
                    and REST_SEARCH_Y_MIN <= dy <= REST_SEARCH_Y_MAX
                ):
                    detection = None
            else:
                if distance((dx, dy), rest_center) > REST_GATE_RADIUS_PX:
                    detection = None

        # Pre-launch false-jump filter (keeps rest detection stable)
        if detection is not None and state in (State.SEEKING_REST, State.REST_FOUND) and last_center is not None:
            jump = distance((detection[0], detection[1]), last_center)
            if jump > MAX_REST_JUMP_PX:
                detection = None

        # Post-launch directional gate: ball should generally move right from rest
        # and not drop far below rest height.
        if detection is not None and state in (State.LAUNCH_DETECTED, State.IMPACT_GAP, State.TRACKING_POST) and rest_center is not None:
            dx_rest = detection[0] - rest_center[0]
            dy_rest = detection[1] - rest_center[1]
            bottom_cutoff_y = current_crop_y + CROP_H - POST_EXCLUDE_BOTTOM_PX
            required_dx = max(POST_MIN_DX_FROM_REST, 15 * max(1, frames_since_launch - 1))
            if (
                dx_rest < required_dx
                or dy_rest > POST_MAX_DY_FROM_REST
                or detection[1] > bottom_cutoff_y
            ):
                detection = None
            elif frames_since_launch >= 6:
                late_min_x = rest_center[0] + 150 + 15 * (frames_since_launch - 6)
                if detection[0] < late_min_x:
                    detection = None
            elif state == State.TRACKING_POST and post_launch_data:
                prev_x, prev_y = post_launch_data[-1][2], post_launch_data[-1][3]
                if detection[0] < (prev_x - POST_BACKTRACK_TOL_X):
                    detection = None
                elif detection[1] > (prev_y + POST_DROP_TOL_Y):
                    detection = None
            elif state == State.TRACKING_POST and last_center is not None and detection[0] < (last_center[0] - 10):
                detection = None
        
        # ---- State machine logic ----
        
        if state == State.SEEKING_REST:
            if detection is not None:
                cx, cy, area, circ = detection
                detection_history.append((cx, cy))
                if idx <= REST_BOOTSTRAP_LAST_FRAME:
                    bootstrap_points.append((cx, cy))
                last_center = (cx, cy)
                seeking_miss_counter = 0

                # Bootstrap rest lock from early pre-impact frames
                if idx <= REST_BOOTSTRAP_LAST_FRAME and len(bootstrap_points) >= REST_BOOTSTRAP_MIN_POINTS:
                    xs = [p[0] for p in bootstrap_points]
                    ys = [p[1] for p in bootstrap_points]
                    rest_center = (float(np.median(xs)), float(np.median(ys)))
                    rest_frame_idx = idx
                    state = State.REST_FOUND
                    print(f"[Frame {idx}] REST_FOUND (bootstrap) at ({rest_center[0]:.1f}, {rest_center[1]:.1f})")
                    continue
                
                # Check if we have a stable window
                is_stable, center = check_stable_window(detection_history)
                if is_stable and center is not None:
                    rest_center = center
                    rest_frame_idx = idx
                    state = State.REST_FOUND
                    print(f"[Frame {idx}] REST_FOUND at ({center[0]:.1f}, {center[1]:.1f})")
            else:
                # Allow short misses without hard reset
                seeking_miss_counter += 1
                if seeking_miss_counter > SEEKING_MISS_TOL:
                    detection_history = []

        elif state == State.REST_FOUND:
            if rest_center is None:
                state = State.SEEKING_REST
                continue

            if detection is not None:
                rest_lost_counter = 0
                cx, cy, area, circ = detection
                last_center = (cx, cy)
                
                # Check for launch (movement from rest)
                dist_from_rest = distance((cx, cy), rest_center)
                
                if dist_from_rest >= LAUNCH_THRESHOLD_PX:
                    # Ball has moved - launch detected!
                    state = State.LAUNCH_DETECTED
                    launch_frame_idx = idx
                    frames_since_launch = 1
                    current_crop_y = max(0, BASE_CROP_Y - SHIFT_PER_FRAME)  # immediate shift
                    
                    # Start collecting post-launch data
                    post_launch_data.append((idx, timestamp_us, cx, cy))
                    
                    # Estimate initial velocity for prediction
                    last_velocity = (cx - rest_center[0], cy - rest_center[1])
                    
                    print(f"[Frame {idx}] LAUNCH_DETECTED! Moved {dist_from_rest:.1f}px from rest")
            else:
                # If ball disappears right after rest lock, treat as impact occlusion launch
                rest_lost_counter += 1
                if rest_lost_counter >= REST_LOST_LAUNCH_FRAMES:
                    state = State.LAUNCH_DETECTED
                    launch_frame_idx = idx
                    frames_since_launch = 1
                    current_crop_y = max(0, BASE_CROP_Y - SHIFT_PER_FRAME)
                    print(f"[Frame {idx}] LAUNCH_DETECTED (occlusion fallback)")
        
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
                    if len(post_launch_data) >= TRACK_POST_MIN_FRAMES:
                        state = State.DONE
                        print(f"[Frame {idx}] DONE - Collected {len(post_launch_data)} post-launch frames")
            else:
                # Detection failed - enter impact gap
                gap_counter += 1
                consecutive_reacquire = 0
                
                if gap_counter > MAX_GAP_FRAMES:
                    state = State.FAILED
                    print(f"[Frame {idx}] FAILED - Lost ball for {gap_counter} frames")
                else:
                    state = State.IMPACT_GAP
                    print(f"[Frame {idx}] IMPACT_GAP (gap={gap_counter})")
        
        elif state == State.IMPACT_GAP:
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

                    directional_ok = False
                    if rest_center is not None:
                        dx_rest = cx - rest_center[0]
                        dy_rest = cy - rest_center[1]
                        directional_ok = (dx_rest >= 90 and dy_rest <= -40)

                    if dist_from_pred <= REACQUIRE_TOL_PX or directional_ok:
                        consecutive_reacquire += 1
                        print(f"[Frame {idx}] Reacquire candidate {consecutive_reacquire}/{REACQUIRE_CONFIRM} "
                              f"(dist={dist_from_pred:.1f}px)")
                        
                        if consecutive_reacquire >= REACQUIRE_CONFIRM:
                            # Reacquired!
                            state = State.TRACKING_POST
                            last_center = (cx, cy)
                            post_launch_data.append((idx, timestamp_us, cx, cy))
                            gap_counter = 0
                            consecutive_reacquire = 0
                            print(f"[Frame {idx}] TRACKING_POST - Reacquired ball!")
                    else:
                        # Detection too far - reset consecutive counter
                        consecutive_reacquire = 0
                        detection = None
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
                if len(post_launch_data) >= TRACK_POST_MIN_FRAMES:
                    state = State.DONE
                    print(f"[Frame {idx}] DONE - Collected {len(post_launch_data)} post-launch frames")
            else:
                # Lost again - back to gap
                gap_counter += 1
                consecutive_reacquire = 0
                
                if gap_counter > MAX_GAP_FRAMES:
                    # Check if we have enough data anyway
                    if len(post_launch_data) >= 2:
                        state = State.DONE
                        print(f"[Frame {idx}] DONE (early) - Have {len(post_launch_data)} frames")
                    else:
                        state = State.FAILED
                        print(f"[Frame {idx}] FAILED - Lost ball again")
                else:
                    state = State.IMPACT_GAP
        
        # ---- Record tracking data ----
        if detection is not None:
            cx, cy, area, circ = detection
            track_rows.append([idx, timestamp_us, cx, cy, 1, state.name])
        else:
            track_rows.append([idx, timestamp_us, "", "", 0, state.name])
        
        # ---- Draw debug overlay ----
        if DEBUG_OVERLAY:
            vis = draw_overlay(frame, current_crop_y, state, detection, rest_center,
                               idx, gap_counter, len(post_launch_data), last_center,
                               crop_h=active_crop_h)
            out_vis = DEBUG_DIR / f"frame_{idx:04d}_overlay.png"
            cv2.imwrite(str(out_vis), vis)
        
        # ---- Early exit if done or failed ----
        if state in (State.DONE, State.FAILED):
            # Continue processing remaining frames for overlay but don't update state
            pass

        prev_frame_gray = frame
    
    # ---- Write tracking CSV ----
    print("\n" + "=" * 60)
    print("Writing outputs...")
    
    with OUT_TRACK.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "timestamp_us", "cx", "cy", "valid", "state"])
        writer.writerows(track_rows)
    print(f"Wrote: {OUT_TRACK}")
    
    # ---- Compute and write metrics ----
    if state == State.DONE and rest_center is not None:
        metrics = compute_metrics(post_launch_data, rest_center)
        
        if metrics is not None:
            # Add frame indices to metrics
            metrics["rest_frame_idx"] = rest_frame_idx
            metrics["launch_frame_idx"] = launch_frame_idx
            
            with OUT_METRICS.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(metrics.keys())
                writer.writerow(metrics.values())
            print(f"Wrote: {OUT_METRICS}")
            
            print("\n" + "=" * 60)
            print("METRICS SUMMARY")
            print("=" * 60)
            for k, v in metrics.items():
                print(f"  {k}: {v}")
        else:
            print("ERROR: Could not compute metrics from post-launch data")
    else:
        print(f"ERROR: Processing ended in state {state.name}")
        if not post_launch_data:
            print("  No post-launch data collected")
        else:
            print(f"  Post-launch frames collected: {len(post_launch_data)}")
    
    if DEBUG_OVERLAY:
        print(f"\nDebug overlays written to: {DEBUG_DIR}")
    if DEBUG_THRESHOLD:
        print(f"Debug threshold images written to: {DEBUG_THRESH_DIR}")
    
    print("\nDone!")
if __name__ == "__main__":
    main()
