"""
Golf Ball Detection - Bottom Camera Implementation
===================================================
Two-pass approach:
  Pass 1: Collect all candidate blobs from every post-impact frame
  Pass 2: Reject static features, build smoothest monotonic-x trajectory

Key insight: The real ball has BOTH high raw-frame brightness AND high
frame-difference brightness (raw * diff product). Static background
features have diff ≈ 0, and body/club movement lacks the concentrated
brightness of the ball.
"""

import cv2
import numpy as np
import os
import csv
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

CAPTURES_BASE_PATH = r"C:\Users\kopil\OneDrive\Senior Project\camera_captures_bottompi"
DEBUG_BASE_PATH = r"C:\Users\kopil\OneDrive\Senior Project\Golf-Launch-Monitor\Calculations\debug_overlay"

FRAME_WIDTH = 640
FRAME_HEIGHT = 400
FRAME_RATE_FPS = 288

# =============================================================================
# CAMERA CALIBRATION
# =============================================================================
CAMERA_DISTANCE_M = 0.965
CAMERA_HEIGHT_M = 0.181
HORIZONTAL_FOV_DEG = 70.0

HORIZONTAL_FOV_RAD = np.radians(HORIZONTAL_FOV_DEG)
FIELD_WIDTH_M = 2 * CAMERA_DISTANCE_M * np.tan(HORIZONTAL_FOV_RAD / 2)
PIXELS_PER_METER = FRAME_WIDTH / FIELD_WIDTH_M
METERS_PER_PIXEL = FIELD_WIDTH_M / FRAME_WIDTH
SECONDS_PER_FRAME = 1.0 / FRAME_RATE_FPS

# Physics
GRAVITY_M_S2 = 9.81
AIR_DENSITY_KG_M3 = 1.225
BALL_MASS_KG = 0.04593
BALL_RADIUS_M = 0.02135
BALL_CROSS_SECTION_M2 = np.pi * BALL_RADIUS_M ** 2
DRAG_COEFFICIENT = 0.25

# Detection
DIFF_THRESHOLD = 15       # Min diff value for blob detection
MIN_BLOB_AREA = 5
MAX_BLOB_AREA = 500
MIN_RAW_BRIGHTNESS = 50   # Ball must be bright in raw frame
MIN_DIFF_BRIGHTNESS = 15  # Ball must show change from reference
STATIC_FEATURE_DIST = 15  # px — blobs closer than this across frames = static
STATIC_FEATURE_MIN_FRAMES = 3  # must appear in this many frames to be "static"
MAX_Y_JUMP = 25           # Max y change between consecutive trajectory points
MIN_TRAJECTORY_LENGTH = 3 # Minimum detections for a valid trajectory


# =============================================================================
# UTILITY
# =============================================================================

def to_gray(frame):
    if len(frame.shape) == 2:
        return frame
    if len(frame.shape) == 3 and frame.shape[2] == 1:
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def measure_contrast(gray, cx, cy, inner_r=10, outer_r=22):
    h, w = gray.shape[:2]
    y_coords, x_coords = np.ogrid[:h, :w]
    dist_sq = (x_coords - cx)**2 + (y_coords - cy)**2
    inner_mask = dist_sq <= inner_r**2
    outer_mask = (dist_sq > inner_r**2) & (dist_sq <= outer_r**2)
    inner_pixels = gray[inner_mask]
    outer_pixels = gray[outer_mask]
    if len(inner_pixels) == 0 or len(outer_pixels) == 0:
        return 0
    return float(np.mean(inner_pixels)) - float(np.mean(outer_pixels))


# =============================================================================
# REST DETECTION (averaged pre-impact frames)
# =============================================================================

def detect_ball_at_rest(frames_list):
    """Find ball at rest by averaging multiple pre-impact frames.
    Static objects (ball) stay bright; moving objects (club) blur out.

    Also uses "disappearance detection": compares early pre-impact frames
    against a late pre-impact frame (when club is close) to find what moved.

    Returns dict with cx, cy, area, contrast or None."""
    if not frames_list or len(frames_list) < 10:
        return None

    grays = [to_gray(f).astype(np.float32) for f in frames_list]

    # Average EARLY pre-impact frames (0-8) where club is far away
    early_avg = np.mean(grays[:min(9, len(grays))], axis=0).astype(np.uint8)

    # Search region: turf area where ball typically sits
    # Ball is usually at x=230-310, y=195-225
    y_min, y_max = 190, 230
    x_min, x_max = 220, 320
    roi = early_avg[y_min:y_max, x_min:x_max]

    best = None
    best_score = 0

    for thresh_val in [35, 40, 50, 60, 70, 30]:
        _, thresh = cv2.threshold(roi, thresh_val, 255, cv2.THRESH_BINARY)
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < 15 or area > 600:
                continue
            per = cv2.arcLength(c, True)
            circ = 4 * np.pi * area / (per**2) if per > 0 else 0
            if circ < 0.25:
                continue
            _, _, bw, bh = cv2.boundingRect(c)
            if max(bw, bh) / max(1, min(bw, bh)) > 2.5:
                continue

            M = cv2.moments(c)
            if M['m00'] <= 0:
                continue
            cx = int(M['m10'] / M['m00']) + x_min
            cy = int(M['m01'] / M['m00']) + y_min

            contrast = measure_contrast(early_avg, cx, cy)
            brightness = float(early_avg[cy, cx])

            # Check below is dark (turf, not shoe)
            below_y = min(cy + 15, FRAME_HEIGHT - 1)
            below_region = early_avg[below_y:min(below_y+15, FRAME_HEIGHT), max(0,cx-10):min(cx+10, FRAME_WIDTH)]
            below_bright = float(below_region.mean()) if below_region.size > 0 else 0

            # Score: contrast + circularity + brightness - below penalty
            score = contrast * 2.0 + circ * 50 + brightness * 0.3
            if below_bright > 50:
                score -= 30  # Bright below = likely shoe, not ball on turf

            if score > best_score:
                best_score = score
                best = {'cx': cx, 'cy': cy, 'area': area, 'contrast': contrast,
                        'circularity': circ, 'brightness': brightness, 'score': score}

    return best


# =============================================================================
# FLIGHT CANDIDATE COLLECTION (Pass 1)
# =============================================================================

def collect_frame_candidates(gray_frame, gray_reference, frame_idx, rest_x=0):
    """Find all candidate blobs in a single frame using frame differencing.
    Returns list of candidate dicts, scored by raw*diff product."""
    diff = cv2.absdiff(gray_frame, gray_reference)

    candidates = []
    seen = set()

    for thresh_val in [DIFF_THRESHOLD, 20, 25, 30]:
        _, thresh = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_BLOB_AREA or area > MAX_BLOB_AREA:
                continue

            M = cv2.moments(c)
            if M['m00'] <= 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Must be to the right of rest position
            if cx <= rest_x:
                continue

            # Deduplicate
            key = (cx // 8, cy // 8)
            if key in seen:
                continue
            seen.add(key)

            # Get brightness values
            raw_bright = float(gray_frame[cy, cx]) if 0 <= cy < FRAME_HEIGHT and 0 <= cx < FRAME_WIDTH else 0
            diff_bright = float(diff[cy, cx]) if 0 <= cy < FRAME_HEIGHT and 0 <= cx < FRAME_WIDTH else 0

            if raw_bright < MIN_RAW_BRIGHTNESS or diff_bright < MIN_DIFF_BRIGHTNESS:
                continue

            per = cv2.arcLength(c, True)
            circ = 4 * np.pi * area / (per**2) if per > 0 else 0

            product = raw_bright * diff_bright

            candidates.append({
                'frame_idx': frame_idx,
                'cx': cx, 'cy': cy, 'area': area,
                'circularity': circ,
                'raw_bright': raw_bright, 'diff_bright': diff_bright,
                'product': product
            })

    candidates.sort(key=lambda c: c['product'], reverse=True)
    return candidates[:15]  # Keep top 15 per frame


# =============================================================================
# STATIC FEATURE REJECTION (Pass 2a)
# =============================================================================

def find_static_features(all_candidates_by_frame):
    """Find positions that appear at roughly the same location across many frames.
    These are static background features, not the moving ball."""
    # Collect all candidate positions with their frame index
    all_positions = []
    for frame_idx, candidates in all_candidates_by_frame.items():
        for c in candidates:
            all_positions.append((c['cx'], c['cy'], frame_idx))

    # Cluster by position
    static_positions = []
    used = set()

    for i, (x1, y1, f1) in enumerate(all_positions):
        if i in used:
            continue
        cluster_frames = {f1}
        cluster_x = [x1]
        cluster_y = [y1]
        for j, (x2, y2, f2) in enumerate(all_positions):
            if j in used or j == i:
                continue
            if abs(x2 - x1) < STATIC_FEATURE_DIST and abs(y2 - y1) < STATIC_FEATURE_DIST:
                cluster_frames.add(f2)
                cluster_x.append(x2)
                cluster_y.append(y2)
                used.add(j)
        used.add(i)

        if len(cluster_frames) >= STATIC_FEATURE_MIN_FRAMES:
            cx = int(np.mean(cluster_x))
            cy = int(np.mean(cluster_y))
            static_positions.append((cx, cy, len(cluster_frames)))

    return static_positions


def remove_static_candidates(all_candidates_by_frame, static_positions):
    """Remove candidates that are near known static feature positions."""
    cleaned = {}
    for frame_idx, candidates in all_candidates_by_frame.items():
        filtered = []
        for c in candidates:
            is_static = False
            for sx, sy, _ in static_positions:
                if abs(c['cx'] - sx) < STATIC_FEATURE_DIST and abs(c['cy'] - sy) < STATIC_FEATURE_DIST:
                    is_static = True
                    break
            if not is_static:
                filtered.append(c)
        cleaned[frame_idx] = filtered
    return cleaned


# =============================================================================
# TRAJECTORY BUILDING (Pass 2b)
# =============================================================================

def build_best_trajectory(candidates_by_frame, rest_position):
    """Build the smoothest monotonically-increasing-x trajectory.

    Strategy:
    1. Try starting from each frame that has candidates
    2. Extend forward greedily: pick candidate with highest product that
       has cx > last_cx and |cy - last_cy| < MAX_Y_JUMP
    3. Keep the longest valid trajectory
    """
    frame_indices = sorted(candidates_by_frame.keys())
    if not frame_indices:
        return []

    rest_x = rest_position[0] if rest_position else 0
    rest_y = rest_position[1] if rest_position else FRAME_HEIGHT // 2

    best_trajectory = []
    best_score = 0

    # Try starting from different frames
    for start_frame in frame_indices:
        for start_cand in candidates_by_frame[start_frame][:3]:  # Try top 3 in each start frame
            # Must be right of rest
            if start_cand['cx'] <= rest_x + 10:
                continue
            # Must be above rest (lower y = higher in image = ball going up)
            if start_cand['cy'] > rest_y + 5:
                continue

            trajectory = [start_cand]
            last_x = start_cand['cx']
            last_y = start_cand['cy']

            # Extend forward
            for fwd_frame in frame_indices:
                if fwd_frame <= start_frame:
                    continue

                best_next = None
                best_prod = 0
                for c in candidates_by_frame[fwd_frame]:
                    # PHYSICS: must move right (monotonic x)
                    if c['cx'] <= last_x:
                        continue
                    # PHYSICS: must stay above rest
                    if c['cy'] > rest_y + 5:
                        continue
                    # PHYSICS: must move upward (monotonic decreasing y)
                    # Ball is always ascending within this frame size
                    if c['cy'] > last_y + 3:  # +3px tolerance for noise
                        continue
                    # SMOOTHNESS: no sudden y jumps
                    if abs(c['cy'] - last_y) > MAX_Y_JUMP:
                        continue
                    # Among valid candidates, pick highest product
                    if c['product'] > best_prod:
                        best_prod = c['product']
                        best_next = c

                if best_next:
                    trajectory.append(best_next)
                    last_x = best_next['cx']
                    last_y = best_next['cy']

            # Score: length * average product (prefer longer AND brighter)
            if len(trajectory) >= MIN_TRAJECTORY_LENGTH:
                avg_prod = np.mean([d['product'] for d in trajectory])
                score = len(trajectory) * avg_prod
                if score > best_score:
                    best_score = score
                    best_trajectory = trajectory

    return best_trajectory


# =============================================================================
# BALL TRACKER (orchestrates the two passes)
# =============================================================================

class BallTracker:
    def __init__(self):
        self.rest_position = None
        self.rest_area = None
        self.rest_contrast = None
        self.trajectory = []       # Final clean trajectory
        self.all_candidates = {}   # frame_idx -> [candidates]
        self.static_features = []
        self.frame_results = {}    # frame_idx -> result dict (for debug overlay)

    def process_burst(self, burst_path, frames_meta):
        """Process an entire burst: load frames, detect rest, collect
        candidates, reject statics, build trajectory."""

        # --- Load all frames ---
        frames = {}
        for fm in frames_meta:
            idx = fm['idx']
            path = os.path.join(burst_path, f"frame_{idx:04d}.png")
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                frames[idx] = img

        if not frames:
            return

        # --- Detect rest position using averaged pre-impact frames ---
        pre_frames = [frames[fm['idx']] for fm in frames_meta
                      if fm['phase'] == 'pre' and fm['idx'] in frames]

        rest_result = detect_ball_at_rest(pre_frames)
        if rest_result:
            self.rest_position = (rest_result['cx'], rest_result['cy'])
            self.rest_area = rest_result['area']
            self.rest_contrast = rest_result['contrast']
            print(f"  Rest: ({rest_result['cx']},{rest_result['cy']}) "
                  f"area={rest_result['area']:.0f} contrast={rest_result['contrast']:.1f} "
                  f"brightness={rest_result['brightness']:.0f}")
        else:
            print("  Rest: NOT DETECTED — using default (260, 205)")
            self.rest_position = (260, 205)
            self.rest_area = 100
            self.rest_contrast = 0

        # --- Get reference frame (frame 5, before club enters) ---
        ref_idx = 5
        if ref_idx not in frames:
            ref_idx = min(frames.keys())
        gray_ref = to_gray(frames[ref_idx])

        # --- Pass 1: Collect candidates from all post-impact frames ---
        post_frames = [fm for fm in frames_meta if fm['phase'] == 'post']
        rest_x = self.rest_position[0] if self.rest_position else 0

        for fm in post_frames:
            idx = fm['idx']
            if idx not in frames:
                continue
            gray = to_gray(frames[idx])
            candidates = collect_frame_candidates(gray, gray_ref, idx, rest_x)
            self.all_candidates[idx] = candidates

        total_cands = sum(len(v) for v in self.all_candidates.values())
        print(f"  Pass 1: {total_cands} candidates across {len(self.all_candidates)} frames")

        # --- Pass 2a: Identify and remove static features ---
        self.static_features = find_static_features(self.all_candidates)
        if self.static_features:
            print(f"  Static features rejected: {[(sx,sy,n) for sx,sy,n in self.static_features]}")
        cleaned = remove_static_candidates(self.all_candidates, self.static_features)

        remaining = sum(len(v) for v in cleaned.values())
        print(f"  After static removal: {remaining} candidates")

        # --- Pass 2b: Build best trajectory ---
        self.trajectory = build_best_trajectory(cleaned, self.rest_position)
        print(f"  Trajectory: {len(self.trajectory)} detections")

        # --- Build per-frame results for debug overlay ---
        traj_set = {d['frame_idx'] for d in self.trajectory}
        traj_by_frame = {d['frame_idx']: d for d in self.trajectory}

        for fm in frames_meta:
            idx = fm['idx']
            result = {
                'frame_idx': idx,
                'phase': fm['phase'],
                'detected': False,
                'position': None,
                'candidates': self.all_candidates.get(idx, []),
                'is_rest': fm['phase'] == 'pre',
            }

            if fm['phase'] == 'pre' and self.rest_position:
                result['detected'] = True
                result['position'] = self.rest_position

            if idx in traj_by_frame:
                d = traj_by_frame[idx]
                result['detected'] = True
                result['position'] = (d['cx'], d['cy'])
                result['area'] = d['area']
                result['product'] = d['product']

            self.frame_results[idx] = result

        # Print trajectory details
        for d in self.trajectory:
            vel_str = ""
            prev_idx = None
            for prev_d in self.trajectory:
                if prev_d['frame_idx'] < d['frame_idx']:
                    prev_idx = prev_d
            if prev_idx:
                vx = d['cx'] - prev_idx['cx']
                vy = d['cy'] - prev_idx['cy']
                vel_str = f" v=({vx:+d},{vy:+d})"
            print(f"    F{d['frame_idx']:02d}: ({d['cx']:3d},{d['cy']:3d}) "
                  f"A={d['area']:.0f} raw={d['raw_bright']:.0f} "
                  f"diff={d['diff_bright']:.0f} prod={d['product']:.0f}{vel_str}")

    def get_clean_trajectory(self):
        return self.trajectory


# =============================================================================
# DEBUG OVERLAY
# =============================================================================

def draw_debug_overlay(frame, result, tracker):
    gray = to_gray(frame)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    BLUE = (255, 0, 0)
    CYAN = (255, 255, 0)
    WHITE = (255, 255, 255)
    MAGENTA = (255, 0, 255)
    DIM_GREEN = (0, 120, 0)

    # Mark static features with red X
    for sx, sy, n in tracker.static_features:
        cv2.drawMarker(overlay, (sx, sy), RED, cv2.MARKER_TILTED_CROSS, 8, 1)

    # Rest position
    if tracker.rest_position:
        cv2.circle(overlay, tracker.rest_position, 18, BLUE, 1)
        cv2.putText(overlay, f"REST({tracker.rest_position[0]},{tracker.rest_position[1]})",
                    (tracker.rest_position[0]-30, tracker.rest_position[1]-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, BLUE, 1)

    # Candidates (show top 5 in yellow)
    for i, cand in enumerate(result.get('candidates', [])[:5]):
        color = YELLOW if i == 0 else (100, 100, 0)
        cv2.circle(overlay, (cand['cx'], cand['cy']), 8, color, 1)
        cv2.putText(overlay, f"p:{cand['product']:.0f}",
                    (cand['cx']+10, cand['cy']),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.22, color, 1)

    # Detected position (trajectory point)
    if result['detected'] and result['position']:
        pos = result['position']
        cv2.circle(overlay, pos, 14, GREEN, 2)
        cv2.circle(overlay, pos, 3, GREEN, -1)

        info = f"({pos[0]},{pos[1]})"
        if 'area' in result:
            info += f" A:{result['area']:.0f}"
        if 'product' in result:
            info += f" P:{result['product']:.0f}"
        cv2.putText(overlay, info, (pos[0]+15, pos[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, GREEN, 1)

    # Draw full trajectory line
    if len(tracker.trajectory) > 1:
        points = [(d['cx'], d['cy']) for d in tracker.trajectory]
        for i in range(1, len(points)):
            cv2.line(overlay, points[i-1], points[i], GREEN, 1)
        # Draw all trajectory points as small dots
        for p in points:
            cv2.circle(overlay, p, 2, DIM_GREEN, -1)

    # Status bar
    status_bg = np.zeros((30, overlay.shape[1], 3), dtype=np.uint8)
    status_bg[:] = (40, 40, 40)

    phase = result.get('phase', '?')
    det_text = "BALL" if result['detected'] else "---"
    det_color = GREEN if result['detected'] else RED
    frame_text = f"F{result['frame_idx']:02d} | {phase}"
    cv2.putText(status_bg, frame_text, (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
    cv2.putText(status_bg, det_text, (580, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, det_color, 1)

    n_traj = len(tracker.trajectory)
    if n_traj > 0:
        cv2.putText(status_bg, f"traj:{n_traj}", (480, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, CYAN, 1)

    return np.vstack([status_bg, overlay])


# =============================================================================
# METRICS CALCULATION
# =============================================================================

class MetricsCalculator:
    def __init__(self, tracker, frame_timestamps=None):
        self.tracker = tracker
        self.frame_timestamps = frame_timestamps or {}
        self.metrics = {}
        self.confidence = "LOW"

    def calculate_all(self):
        clean = self.tracker.get_clean_trajectory()

        if len(clean) < 2:
            print(f"  Only {len(clean)} detections — insufficient for metrics")
            return {}

        self.confidence = "HIGH" if len(clean) >= 5 else "MEDIUM" if len(clean) >= 3 else "LOW"

        ball_speed_mph, velocity_mps = self._calculate_ball_speed(clean)
        launch_angle_deg = self._calculate_launch_angle(clean)
        apex_height_ft, carry_distance_yds = self._simulate_trajectory(
            velocity_mps, launch_angle_deg
        )

        self.metrics = {
            'ball_speed_mph': ball_speed_mph,
            'launch_angle_deg': launch_angle_deg,
            'apex_height_ft': apex_height_ft,
            'carry_distance_yds': carry_distance_yds,
            'velocity_mps': velocity_mps,
            'num_clean_detections': len(clean),
            'confidence': self.confidence,
        }
        return self.metrics

    def _calculate_ball_speed(self, detections):
        # Skip first 1-2 noisy near-impact points
        start_idx = 0
        if len(detections) > 6:
            start_idx = 2
        elif len(detections) > 4:
            start_idx = 1

        velocities_mps = []
        for i in range(start_idx, min(start_idx + 5, len(detections) - 1)):
            d1 = detections[i]
            d2 = detections[i + 1]

            dx_m = (d2['cx'] - d1['cx']) * METERS_PER_PIXEL
            dy_m = -(d2['cy'] - d1['cy']) * METERS_PER_PIXEL

            if self.frame_timestamps:
                t1 = self.frame_timestamps.get(d1['frame_idx'], 0)
                t2 = self.frame_timestamps.get(d2['frame_idx'], 0)
                dt = (t2 - t1) / 1_000_000_000
            else:
                dt = (d2['frame_idx'] - d1['frame_idx']) * SECONDS_PER_FRAME

            if dt > 0:
                speed = np.sqrt((dx_m/dt)**2 + (dy_m/dt)**2)
                velocities_mps.append(speed)

        if not velocities_mps:
            return 0, 0

        velocities_mps.sort()
        initial_speed = velocities_mps[len(velocities_mps) // 2]
        speed_mph = initial_speed * 2.237
        print(f"  Speed samples (m/s): {[f'{v:.1f}' for v in velocities_mps]}")
        print(f"  Selected speed: {initial_speed:.1f} m/s ({speed_mph:.1f} mph)")

        return speed_mph, initial_speed

    def _calculate_launch_angle(self, detections):
        rest = self.tracker.rest_position
        rest_x = rest[0] if rest else detections[0]['cx']
        rest_y = rest[1] if rest else detections[0]['cy']

        x_m = [(d['cx'] - rest_x) * METERS_PER_PIXEL for d in detections]
        y_m = [-(d['cy'] - rest_y) * METERS_PER_PIXEL for d in detections]

        if len(x_m) >= 3 and (max(x_m) - min(x_m)) > 0.01:
            slope, _ = np.polyfit(x_m, y_m, 1)
            angle_fit = np.degrees(np.arctan(slope))

            dx_total = x_m[-1] - x_m[0]
            dy_total = y_m[-1] - y_m[0]
            angle_vec = np.degrees(np.arctan2(dy_total, max(dx_total, 0.001)))

            angle = (angle_fit + angle_vec) / 2.0
            print(f"  Angle: fit={angle_fit:.1f} vec={angle_vec:.1f} avg={angle:.1f}")
        elif len(detections) >= 2:
            dx = (detections[-1]['cx'] - detections[0]['cx']) * METERS_PER_PIXEL
            dy = -(detections[-1]['cy'] - detections[0]['cy']) * METERS_PER_PIXEL
            angle = np.degrees(np.arctan2(dy, max(dx, 0.001)))
        else:
            angle = 0

        angle = max(0, min(75, angle))
        print(f"  Trajectory (m): x=[{x_m[0]:.3f}..{x_m[-1]:.3f}] "
              f"y=[{y_m[0]:.3f}..{y_m[-1]:.3f}] ({len(x_m)} pts)")
        print(f"  Launch angle: {angle:.1f} deg")
        return angle

    def _simulate_trajectory(self, velocity_mps, launch_angle_deg):
        if velocity_mps <= 0 or launch_angle_deg <= 0:
            return 0, 0

        rad = np.radians(launch_angle_deg)
        x, y = 0.0, 0.0
        vx = velocity_mps * np.cos(rad)
        vy = velocity_mps * np.sin(rad)
        dt = 0.001
        max_height = 0.0
        drag_c = 0.5 * AIR_DENSITY_KG_M3 * DRAG_COEFFICIENT * BALL_CROSS_SECTION_M2 / BALL_MASS_KG
        t = 0
        has_risen = False

        while t < 30:
            if y > max_height:
                max_height = y
                has_risen = True
            if has_risen and y <= 0:
                break
            v = np.sqrt(vx**2 + vy**2)
            if v > 0:
                drag_ax = -drag_c * v * vx
                drag_ay = -drag_c * v * vy
            else:
                drag_ax, drag_ay = 0, 0
            vx += drag_ax * dt
            vy += (-GRAVITY_M_S2 + drag_ay) * dt
            x += vx * dt
            y += vy * dt
            t += dt

        apex_ft = max_height * 3.281
        carry_yds = x * 1.094
        print(f"  Sim: {velocity_mps:.1f}m/s at {launch_angle_deg:.1f}deg -> "
              f"apex {max_height:.1f}m, carry {x:.1f}m ({carry_yds:.1f}yds)")
        return apex_ft, carry_yds

    def print_summary(self):
        if not self.metrics:
            return
        m = self.metrics
        print(f"  [{m['confidence']}] Speed:{m['ball_speed_mph']:.1f}mph  "
              f"Angle:{m['launch_angle_deg']:.1f}deg  "
              f"Carry:{m['carry_distance_yds']:.1f}yds  "
              f"({m['num_clean_detections']} detections)")


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_single_capture(capture_path, debug_output_path):
    os.makedirs(debug_output_path, exist_ok=True)

    meta_path = os.path.join(capture_path, "burst_meta.csv")
    if not os.path.exists(meta_path):
        print(f"  No burst_meta.csv, skipping")
        return None, None

    frames_meta = []
    with open(meta_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames_meta.append({
                'idx': int(row['frame_idx']),
                'timestamp_us': int(row['timestamp_us']),
                'phase': row['phase']
            })

    tracker = BallTracker()
    tracker.process_burst(capture_path, frames_meta)

    # Generate debug overlays
    for fm in frames_meta:
        idx = fm['idx']
        path = os.path.join(capture_path, f"frame_{idx:04d}.png")
        frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if frame is None:
            continue

        result = tracker.frame_results.get(idx, {
            'frame_idx': idx, 'phase': fm['phase'],
            'detected': False, 'position': None, 'candidates': []
        })

        overlay = draw_debug_overlay(frame, result, tracker)
        cv2.imwrite(os.path.join(debug_output_path, f"debug_{idx:04d}.png"), overlay)

    # Calculate metrics
    timestamps = {m['idx']: m['timestamp_us'] for m in frames_meta}
    calculator = MetricsCalculator(tracker, timestamps)
    metrics = calculator.calculate_all()

    if metrics:
        calculator.print_summary()

    return metrics, tracker


def run_all_captures():
    if not os.path.isdir(CAPTURES_BASE_PATH):
        print(f"ERROR: Not found: {CAPTURES_BASE_PATH}")
        return

    folders = sorted([
        d for d in os.listdir(CAPTURES_BASE_PATH)
        if os.path.isdir(os.path.join(CAPTURES_BASE_PATH, d))
    ])

    print(f"Found {len(folders)} capture folders\n")
    all_results = []

    for folder in folders:
        capture_path = os.path.join(CAPTURES_BASE_PATH, folder)
        debug_path = os.path.join(DEBUG_BASE_PATH, folder)

        parts = folder.split('_')
        short_name = parts[-1] if len(parts) >= 3 else folder[:20]

        print("=" * 70)
        print(f"SHOT: {short_name} ({folder[:8]}...)")
        print("=" * 70)

        metrics, tracker = process_single_capture(capture_path, debug_path)

        all_results.append({
            'folder': folder, 'short_name': short_name,
            'metrics': metrics,
            'rest_position': tracker.rest_position if tracker else None,
            'num_detections': len(tracker.trajectory) if tracker else 0,
        })
        print()

    # Summary table
    print("\n" + "=" * 80)
    print("BATCH SUMMARY")
    print("=" * 80)
    print(f"{'Shot':>8} {'Rest Pos':>12} {'Dets':>5} {'Speed':>8} {'Angle':>7} {'Carry':>8} {'Conf'}")
    print("-" * 80)

    for r in all_results:
        rest = f"({r['rest_position'][0]},{r['rest_position'][1]})" if r['rest_position'] else "NONE"
        m = r['metrics']
        if m:
            speed = f"{m['ball_speed_mph']:.1f}mph"
            angle = f"{m['launch_angle_deg']:.1f}deg"
            carry = f"{m['carry_distance_yds']:.1f}yds"
            conf = m['confidence']
        else:
            speed = angle = carry = "---"
            conf = "FAIL"

        print(f"{r['short_name']:>8} {rest:>12} {r['num_detections']:>5} "
              f"{speed:>8} {angle:>7} {carry:>8} {conf}")

    # Save CSV
    csv_path = os.path.join(DEBUG_BASE_PATH, "batch_results.csv")
    os.makedirs(DEBUG_BASE_PATH, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['folder', 'rest_x', 'rest_y', 'detections',
                         'ball_speed_mph', 'launch_angle_deg', 'carry_distance_yds', 'confidence'])
        for r in all_results:
            rest = r['rest_position']
            m = r['metrics'] or {}
            writer.writerow([
                r['folder'],
                rest[0] if rest else '', rest[1] if rest else '',
                r['num_detections'],
                f"{m.get('ball_speed_mph', 0):.1f}",
                f"{m.get('launch_angle_deg', 0):.1f}",
                f"{m.get('carry_distance_yds', 0):.1f}",
                m.get('confidence', 'FAIL')
            ])
    print(f"\nResults: {csv_path}")
    print(f"Debug overlays: {DEBUG_BASE_PATH}")


if __name__ == "__main__":
    run_all_captures()
