"""
Golf Ball Detection - Bottom Camera Implementation
===================================================
Detects golf ball at rest and tracks it through post-impact frames.
Uses dynamic ROI box tracking to reject club/body noise.
Generates debug overlay images to visualize detection.
"""

import cv2
import numpy as np
import os
import csv
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

# Frame source path (Windows) - point to a specific capture folder
RAW_BURST_PATH = r"C:\Users\kopil\OneDrive\Senior Project\camera_captures_bottompi\exposure=500"

# Output path for debug overlays
DEBUG_OUTPUT_PATH = r"C:\Users\kopil\OneDrive\Senior Project\Golf-Launch-Monitor\Calculations\debug_overlay"

# Dynamic ROI box - starts centered on ball at rest, moves with tracking
# Ball is ~20px diameter, so keep ROI tight to avoid picking up shoes/legs
ROI_REST_HALF_W = 30   # Half-width of ROI box when searching for ball at rest
ROI_REST_HALF_H = 30   # Half-height of ROI box when searching for ball at rest
ROI_FLIGHT_HALF_W = 40 # Half-width of ROI box when tracking ball in flight
ROI_FLIGHT_HALF_H = 40 # Half-height of ROI box when tracking ball in flight
ROI_GROWTH_PER_MISS = 10  # ROI grows by this many pixels each frame without detection
ROI_MAX_HALF = 80      # Cap ROI growth to prevent picking up body parts

# Frame dimensions
FRAME_WIDTH = 640
FRAME_HEIGHT = 400

# Camera parameters
FRAME_RATE_FPS = 288  # Approximate capture rate

# =============================================================================
# CAMERA CALIBRATION - Physical Setup
# =============================================================================
CAMERA_DISTANCE_M = 0.965          # Distance from camera to ball (meters)
CAMERA_HEIGHT_M = 0.181            # Height of camera lens center (meters)
BALL_DIAMETER_M = 0.04267          # Golf ball diameter (meters)
HORIZONTAL_FOV_DEG = 70.0          # Horizontal field of view (degrees)

# Derived calibration values
HORIZONTAL_FOV_RAD = np.radians(HORIZONTAL_FOV_DEG)
FIELD_WIDTH_M = 2 * CAMERA_DISTANCE_M * np.tan(HORIZONTAL_FOV_RAD / 2)
PIXELS_PER_METER = FRAME_WIDTH / FIELD_WIDTH_M
METERS_PER_PIXEL = FIELD_WIDTH_M / FRAME_WIDTH

# Time between frames
SECONDS_PER_FRAME = 1.0 / FRAME_RATE_FPS

# Physics constants
GRAVITY_M_S2 = 9.81
AIR_DENSITY_KG_M3 = 1.225
BALL_MASS_KG = 0.04593
BALL_RADIUS_M = 0.02135
BALL_CROSS_SECTION_M2 = np.pi * BALL_RADIUS_M ** 2
DRAG_COEFFICIENT = 0.25

# Known ball rest position (bottom camera view)
BALL_REST_X = 249
BALL_REST_Y = 216

# Detection parameters
BRIGHTNESS_THRESHOLD_DEFAULT = 80
MOTION_THRESHOLD_DEFAULT = 25

# Ball size - ball is ~20px diameter (~314px area), keep range tight to reject club
MIN_BALL_AREA_REST = 40
MAX_BALL_AREA_REST = 500
MIN_BALL_AREA_FLIGHT = 15
MAX_BALL_AREA_FLIGHT = 400

# Circularity ranges - strict to reject club shaft (long thin contour)
MIN_CIRCULARITY_REST = 0.4
MIN_CIRCULARITY_FLIGHT = 0.35

# Search regions
REST_SEARCH_RADIUS = 40
POST_IMPACT_MIN_X = 0  # Disabled for bottom camera

# Velocity constraints
MIN_VELOCITY_PX_FRAME = 5
MAX_VELOCITY_PX_FRAME = 150


# =============================================================================
# ADAPTIVE PARAMETERS CLASS
# =============================================================================

class AdaptiveParams:
    """Learns and adapts detection parameters based on observed ball behavior."""

    def __init__(self):
        self.brightness_threshold = BRIGHTNESS_THRESHOLD_DEFAULT
        self.motion_threshold = MOTION_THRESHOLD_DEFAULT
        self.rest_ball_area = None
        self.rest_ball_circularity = None
        self.rest_ball_brightness = None
        self.velocities = []
        self.avg_velocity = None
        self.velocity_magnitude = None
        self.search_radius = 60
        self.primary_direction = None

    def learn_from_rest(self, area: float, circularity: float, brightness: float):
        self.rest_ball_area = area
        self.rest_ball_circularity = circularity
        self.rest_ball_brightness = brightness

    def add_velocity_sample(self, vx: float, vy: float):
        self.velocities.append((vx, vy))
        if len(self.velocities) >= 2:
            recent = self.velocities[-3:] if len(self.velocities) >= 3 else self.velocities
            avg_vx = sum(v[0] for v in recent) / len(recent)
            avg_vy = sum(v[1] for v in recent) / len(recent)
            self.avg_velocity = (avg_vx, avg_vy)
            self.velocity_magnitude = np.sqrt(avg_vx**2 + avg_vy**2)
            self.primary_direction = (1 if avg_vx > 0 else -1, 1 if avg_vy > 0 else -1)
            self.search_radius = max(40, min(150, self.velocity_magnitude * 1.5))

    def predict_next_position(self, current_pos: tuple) -> tuple:
        if self.avg_velocity:
            return (
                current_pos[0] + self.avg_velocity[0],
                current_pos[1] + self.avg_velocity[1]
            )
        return None

    def get_area_range(self, in_flight: bool) -> tuple:
        if in_flight:
            base_min = MIN_BALL_AREA_FLIGHT
            base_max = MAX_BALL_AREA_FLIGHT
        else:
            base_min = MIN_BALL_AREA_REST
            base_max = MAX_BALL_AREA_REST
        if self.rest_ball_area:
            if in_flight:
                learned_min = self.rest_ball_area * 0.2
                learned_max = self.rest_ball_area * 3.0
                return (max(15, learned_min), max(learned_max, base_max))
            else:
                return (self.rest_ball_area * 0.5, self.rest_ball_area * 1.5)
        return (base_min, base_max)

    def get_velocity_range(self) -> tuple:
        if self.velocity_magnitude:
            return (
                max(3, self.velocity_magnitude * 0.3),
                max(self.velocity_magnitude * 2.5, MAX_VELOCITY_PX_FRAME)
            )
        return (MIN_VELOCITY_PX_FRAME, MAX_VELOCITY_PX_FRAME)


# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================

def load_frame(frame_idx: int) -> np.ndarray:
    """Load a frame by index. Loads as-is for R8 raw frames."""
    path = os.path.join(RAW_BURST_PATH, f"frame_{frame_idx:04d}.png")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load frame: {path}")
    return img


def load_burst_meta() -> list:
    """Load burst metadata CSV."""
    meta_path = os.path.join(RAW_BURST_PATH, "burst_meta.csv")
    frames = []
    with open(meta_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append({
                'idx': int(row['frame_idx']),
                'timestamp_us': int(row['timestamp_us']),
                'phase': row['phase']
            })
    return frames


def to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert frame to grayscale, handling both BGR and single-channel R8 formats."""
    if len(frame.shape) == 2:
        return frame
    if len(frame.shape) == 3 and frame.shape[2] == 1:
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def apply_roi_box(gray: np.ndarray, center_x: int, center_y: int,
                   half_w: int, half_h: int) -> np.ndarray:
    """Zero out everything outside an ROI box centered at (center_x, center_y)."""
    masked = np.zeros_like(gray)
    x1 = max(0, center_x - half_w)
    y1 = max(0, center_y - half_h)
    x2 = min(gray.shape[1], center_x + half_w)
    y2 = min(gray.shape[0], center_y + half_h)
    masked[y1:y2, x1:x2] = gray[y1:y2, x1:x2]
    return masked


def get_roi_bounds(center_x: int, center_y: int, half_w: int, half_h: int,
                   frame_w: int, frame_h: int) -> tuple:
    """Return clamped (x1, y1, x2, y2) for the ROI box."""
    return (max(0, center_x - half_w), max(0, center_y - half_h),
            min(frame_w, center_x + half_w), min(frame_h, center_y + half_h))


def get_contour_features(contour, gray_frame=None) -> dict:
    """Extract features from a contour."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

    M = cv2.moments(contour)
    if M['m00'] > 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = 0, 0

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 0

    brightness = 0
    if gray_frame is not None:
        mask = np.zeros(gray_frame.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        brightness = cv2.mean(gray_frame, mask=mask)[0]

    return {
        'contour': contour,
        'area': area,
        'circularity': circularity,
        'cx': cx,
        'cy': cy,
        'bbox': (x, y, w, h),
        'aspect_ratio': aspect_ratio,
        'brightness': brightness
    }


def detect_ball_at_rest(frame: np.ndarray, params: AdaptiveParams,
                        hint_position: tuple = None) -> dict:
    """
    Detect ball at rest using direct thresholding within a tight ROI box.
    Uses hint_position if provided, otherwise searches near expected position.
    """
    gray = to_gray(frame)

    target_x = hint_position[0] if hint_position else BALL_REST_X
    target_y = hint_position[1] if hint_position else BALL_REST_Y
    gray_roi = apply_roi_box(gray, target_x, target_y, ROI_REST_HALF_W, ROI_REST_HALF_H)

    thresholds = [params.brightness_threshold,
                  params.brightness_threshold - 15,
                  params.brightness_threshold + 15]

    all_candidates = []

    for thresh_val in thresholds:
        if thresh_val < 40 or thresh_val > 200:
            continue

        _, thresh = cv2.threshold(gray_roi, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        area_min, area_max = params.get_area_range(in_flight=False)

        for contour in contours:
            features = get_contour_features(contour, gray)

            if not (area_min < features['area'] < area_max):
                continue
            if features['circularity'] < MIN_CIRCULARITY_REST:
                continue

            dist = np.sqrt((features['cx'] - target_x)**2 + (features['cy'] - target_y)**2)
            if dist > REST_SEARCH_RADIUS:
                continue

            score = 100 - dist + features['circularity'] * 50
            features['score'] = score
            features['dist_to_expected'] = dist
            all_candidates.append(features)

    if not all_candidates:
        return None

    all_candidates.sort(key=lambda x: x['score'], reverse=True)
    return all_candidates[0]


def detect_ball_motion(current_frame: np.ndarray, reference_frame: np.ndarray,
                       params: AdaptiveParams, prev_position: tuple = None,
                       rest_position: tuple = None,
                       roi_center: tuple = None, roi_half_w: int = ROI_FLIGHT_HALF_W,
                       roi_half_h: int = ROI_FLIGHT_HALF_H) -> list:
    """
    Detect ball using frame differencing within a dynamic ROI box.
    Returns list of candidates sorted by likelihood.
    """
    gray_curr = to_gray(current_frame)
    gray_ref = to_gray(reference_frame)

    thresholds = [params.motion_threshold,
                  params.motion_threshold - 5,
                  params.motion_threshold + 10]

    all_candidates = []
    seen_positions = set()

    for thresh_val in thresholds:
        if thresh_val < 10:
            continue

        diff = cv2.absdiff(gray_curr, gray_ref)
        if roi_center:
            diff = apply_roi_box(diff, int(roi_center[0]), int(roi_center[1]),
                                 roi_half_w, roi_half_h)
        _, thresh = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        area_min, area_max = params.get_area_range(in_flight=True)

        for contour in contours:
            features = get_contour_features(contour, gray_curr)

            # Filter by area
            if not (area_min < features['area'] < area_max):
                continue

            # Strict circularity filter - rejects club shaft
            if features['circularity'] < MIN_CIRCULARITY_FLIGHT:
                continue

            # Reject elongated shapes (club shaft has high aspect ratio)
            bbox_w = features['bbox'][2]
            bbox_h = features['bbox'][3]
            aspect = max(bbox_w, bbox_h) / max(1, min(bbox_w, bbox_h))
            if aspect > 3.0:
                continue

            # Avoid duplicate detections
            pos_key = (features['cx'] // 10, features['cy'] // 10)
            if pos_key in seen_positions:
                continue
            seen_positions.add(pos_key)

            score = calculate_candidate_score(features, params, prev_position, rest_position)
            features['score'] = score
            all_candidates.append(features)

    all_candidates.sort(key=lambda x: x['score'], reverse=True)
    return all_candidates


def calculate_candidate_score(features: dict, params: AdaptiveParams,
                              prev_position: tuple, rest_position: tuple) -> float:
    """Calculate likelihood score for a ball candidate."""
    score = 0

    score += features['circularity'] * 100

    if prev_position:
        dx = features['cx'] - prev_position[0]
        dy = features['cy'] - prev_position[1]
        velocity = np.sqrt(dx**2 + dy**2)

        vel_min, vel_max = params.get_velocity_range()

        if vel_min <= velocity <= vel_max:
            score += 50
        elif velocity < vel_min:
            score -= 30
        elif velocity > vel_max * 1.5:
            score -= 100

        if params.primary_direction:
            expected_dx_sign, expected_dy_sign = params.primary_direction
            actual_dx_sign = 1 if dx > 0 else -1
            actual_dy_sign = 1 if dy > 0 else -1

            if actual_dx_sign == expected_dx_sign:
                score += 40
            else:
                score -= 80
            if actual_dy_sign == expected_dy_sign:
                score += 20
        else:
            # From bottom camera, ball moves upward in frame
            if dy < 0:
                score += 40

        predicted = params.predict_next_position(prev_position)
        if predicted:
            dist_to_predicted = np.sqrt((features['cx'] - predicted[0])**2 +
                                        (features['cy'] - predicted[1])**2)
            if dist_to_predicted < params.search_radius:
                score += 30 * (1 - dist_to_predicted / params.search_radius)

    if rest_position:
        dist_from_rest = np.sqrt((features['cx'] - rest_position[0])**2 +
                                 (features['cy'] - rest_position[1])**2)
        if dist_from_rest > 30:
            score += 20
        else:
            score -= 50

    if params.rest_ball_area:
        area_ratio = features['area'] / params.rest_ball_area
        if 0.3 <= area_ratio <= 2.5:
            score += 15
        else:
            score -= 20

    return score


# =============================================================================
# TRACKING STATE MACHINE
# =============================================================================

class BallTracker:
    """
    Tracks ball through pre-impact rest, impact, and post-impact flight.
    Uses dynamic ROI box that follows the ball to reject club/body noise.
    """

    def __init__(self):
        self.state = "INIT"
        self.rest_position = None
        self.rest_frames = []
        self.post_impact_detections = []
        self.reference_frame = None
        self.last_position = None
        self.params = AdaptiveParams()

        self.frames_since_last_detection = 0
        self.max_gap_frames = 3

        # Dynamic ROI tracking
        self.roi_center = (BALL_REST_X, BALL_REST_Y)
        self.roi_half_w = ROI_FLIGHT_HALF_W
        self.roi_half_h = ROI_FLIGHT_HALF_H

    def process_frame(self, frame_idx: int, frame: np.ndarray, phase: str) -> dict:
        result = {
            'frame_idx': frame_idx,
            'phase': phase,
            'state': self.state,
            'detected': False,
            'position': None,
            'candidates': [],
            'predicted_position': None,
            'velocity': None
        }

        if self.state == "BALL_EXIT":
            result['state'] = self.state
            return result

        if phase == "pre":
            result = self._process_pre_impact(frame_idx, frame, result)
        elif phase == "post":
            result = self._process_post_impact(frame_idx, frame, result)

        result['state'] = self.state
        return result

    def _process_pre_impact(self, frame_idx: int, frame: np.ndarray, result: dict) -> dict:
        """Process pre-impact frame to find ball at rest."""
        hint = self.rest_position if self.rest_position else None

        target_x = hint[0] if hint else BALL_REST_X
        target_y = hint[1] if hint else BALL_REST_Y
        result['roi_center'] = (target_x, target_y)
        result['roi_half_w'] = ROI_REST_HALF_W
        result['roi_half_h'] = ROI_REST_HALF_H

        # Once found, lock rest position for all pre-impact frames
        if self.rest_position is not None:
            result['detected'] = True
            result['position'] = self.rest_position
            self.rest_frames.append(frame_idx)
            self.reference_frame = frame.copy()
            return result

        detection = detect_ball_at_rest(frame, self.params, hint)

        if detection:
            result['detected'] = True
            result['position'] = (detection['cx'], detection['cy'])
            result['area'] = detection['area']
            result['circularity'] = detection['circularity']

            self.params.learn_from_rest(
                detection['area'],
                detection['circularity'],
                detection['brightness']
            )

            self.rest_position = (detection['cx'], detection['cy'])
            self.rest_frames.append(frame_idx)
            self.state = "REST_FOUND"
            self.reference_frame = frame.copy()

        return result

    def _process_post_impact(self, frame_idx: int, frame: np.ndarray, result: dict) -> dict:
        """Process post-impact frame to track ball in flight."""
        if self.reference_frame is None:
            try:
                self.reference_frame = load_frame(5)
            except:
                self.reference_frame = frame

        # Check if ball is still at rest before switching to motion detection
        if self.state == "REST_FOUND" and self.rest_position:
            rest_check = detect_ball_at_rest(frame, self.params, self.rest_position)
            if rest_check:
                dist = np.sqrt((rest_check['cx'] - self.rest_position[0])**2 +
                               (rest_check['cy'] - self.rest_position[1])**2)
                if dist < 20:
                    # Ball still at rest - update reference frame
                    result['detected'] = True
                    result['position'] = self.rest_position
                    result['roi_center'] = self.rest_position
                    result['roi_half_w'] = ROI_REST_HALF_W
                    result['roi_half_h'] = ROI_REST_HALF_H
                    self.reference_frame = frame.copy()
                    return result

            # Ball gone from rest - impact occurred
            self.state = "IMPACT"

        # Determine ROI center
        predicted = None
        if self.last_position:
            predicted = self.params.predict_next_position(self.last_position)
            result['predicted_position'] = predicted

        if predicted:
            self.roi_center = (int(predicted[0]), int(predicted[1]))
        elif self.last_position:
            self.roi_center = self.last_position
        elif self.rest_position:
            self.roi_center = self.rest_position

        result['roi_center'] = self.roi_center
        result['roi_half_w'] = self.roi_half_w
        result['roi_half_h'] = self.roi_half_h

        # Detect candidates within dynamic ROI
        candidates = detect_ball_motion(
            frame, self.reference_frame, self.params,
            self.last_position or self.rest_position,
            self.rest_position,
            roi_center=self.roi_center,
            roi_half_w=self.roi_half_w,
            roi_half_h=self.roi_half_h
        )

        result['candidates'] = candidates

        if candidates:
            best = candidates[0]
            valid = self._validate_candidate(best, frame_idx)

            if valid:
                result['detected'] = True
                result['position'] = (best['cx'], best['cy'])
                result['area'] = best['area']
                result['circularity'] = best['circularity']

                if self.last_position:
                    vx = best['cx'] - self.last_position[0]
                    vy = best['cy'] - self.last_position[1]
                    result['velocity'] = (vx, vy)
                    self.params.add_velocity_sample(vx, vy)

                self.post_impact_detections.append({
                    'frame_idx': frame_idx,
                    'x': best['cx'],
                    'y': best['cy'],
                    'area': best['area'],
                    'circularity': best['circularity'],
                    'velocity': result['velocity']
                })

                self.last_position = (best['cx'], best['cy'])
                self.state = "TRACKING_POST"
                self.frames_since_last_detection = 0

                # Reset ROI size on successful detection
                self.roi_half_w = ROI_FLIGHT_HALF_W
                self.roi_half_h = ROI_FLIGHT_HALF_H

                if best['cx'] > 620 or best['cy'] < 20 or best['cx'] < 20:
                    self.state = "BALL_EXIT"
            else:
                self.frames_since_last_detection += 1
                self.roi_half_w = min(self.roi_half_w + ROI_GROWTH_PER_MISS, ROI_MAX_HALF)
                self.roi_half_h = min(self.roi_half_h + ROI_GROWTH_PER_MISS, ROI_MAX_HALF)
        else:
            self.frames_since_last_detection += 1
            self.roi_half_w = min(self.roi_half_w + ROI_GROWTH_PER_MISS, ROI_MAX_HALF)
            self.roi_half_h = min(self.roi_half_h + ROI_GROWTH_PER_MISS, ROI_MAX_HALF)

        if self.frames_since_last_detection >= self.max_gap_frames:
            if len(self.post_impact_detections) > 0:
                self.state = "BALL_EXIT"

        return result

    def _validate_candidate(self, candidate: dict, frame_idx: int) -> bool:
        """Validate a candidate detection."""
        if self.rest_position:
            dist_from_rest = np.sqrt((candidate['cx'] - self.rest_position[0])**2 +
                                     (candidate['cy'] - self.rest_position[1])**2)
            if dist_from_rest < 15:
                return False

        if not self.last_position or len(self.post_impact_detections) == 0:
            return True

        dx = candidate['cx'] - self.last_position[0]
        dy = candidate['cy'] - self.last_position[1]
        velocity = np.sqrt(dx**2 + dy**2)

        vel_min, vel_max = self.params.get_velocity_range()

        if velocity < vel_min * 0.2:
            return False
        if velocity > vel_max * 2:
            return False

        if self.params.primary_direction:
            expected_dx_sign = self.params.primary_direction[0]
            actual_dx_sign = 1 if dx > 0 else -1
            if actual_dx_sign != expected_dx_sign and abs(dx) > 10:
                return False

        return True


# =============================================================================
# DEBUG OVERLAY GENERATION
# =============================================================================

def draw_debug_overlay(frame: np.ndarray, result: dict, tracker: BallTracker) -> np.ndarray:
    """Draw debug visualization on frame."""
    # Convert single-channel to BGR for colored annotations
    if len(frame.shape) == 2:
        overlay = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif len(frame.shape) == 3 and frame.shape[2] == 1:
        overlay = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    else:
        overlay = frame.copy()

    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    BLUE = (255, 0, 0)
    CYAN = (255, 255, 0)
    WHITE = (255, 255, 255)
    MAGENTA = (255, 0, 255)

    # Draw dynamic ROI box
    if result.get('roi_center'):
        rc = result['roi_center']
        rw = result.get('roi_half_w', ROI_FLIGHT_HALF_W)
        rh = result.get('roi_half_h', ROI_FLIGHT_HALF_H)
        x1, y1, x2, y2 = get_roi_bounds(rc[0], rc[1], rw, rh,
                                          overlay.shape[1], overlay.shape[0])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), YELLOW, 1)
        cv2.putText(overlay, "ROI", (x1 + 2, y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, YELLOW, 1)

    # Draw rest position reference
    if tracker.rest_position:
        cv2.circle(overlay, tracker.rest_position, 20, BLUE, 1)
        cv2.putText(overlay, "REST", (tracker.rest_position[0]-15, tracker.rest_position[1]-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLUE, 1)

    # Draw predicted position
    if result.get('predicted_position'):
        pred = result['predicted_position']
        pred_int = (int(pred[0]), int(pred[1]))
        cv2.circle(overlay, pred_int, 10, MAGENTA, 1)
        cv2.putText(overlay, "PRED", (pred_int[0]+12, pred_int[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, MAGENTA, 1)

    # Draw all candidates (top 5)
    for i, cand in enumerate(result.get('candidates', [])[:5]):
        color = GREEN if i == 0 else YELLOW
        cv2.circle(overlay, (cand['cx'], cand['cy']), 12, color, 1)
        if i > 0:
            cv2.putText(overlay, f"c{i}", (cand['cx']+10, cand['cy']),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, YELLOW, 1)

    # Draw detected ball position
    if result['detected'] and result['position']:
        pos = result['position']
        cv2.circle(overlay, pos, 15, GREEN, 2)
        cv2.circle(overlay, pos, 3, GREEN, -1)

        info_text = f"({pos[0]}, {pos[1]})"
        if 'area' in result:
            info_text += f" A:{result['area']:.0f}"
        if 'circularity' in result:
            info_text += f" C:{result['circularity']:.2f}"
        cv2.putText(overlay, info_text, (pos[0]+15, pos[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, GREEN, 1)

        if result.get('velocity'):
            vx, vy = result['velocity']
            vel_mag = np.sqrt(vx**2 + vy**2)
            vel_text = f"v:{vel_mag:.0f}px/f"
            cv2.putText(overlay, vel_text, (pos[0]+15, pos[1]+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, CYAN, 1)

    # Draw trajectory line
    if len(tracker.post_impact_detections) > 1:
        points = [(d['x'], d['y']) for d in tracker.post_impact_detections]
        for i in range(1, len(points)):
            cv2.line(overlay, points[i-1], points[i], GREEN, 1)

    # Status bar at top
    status_bg = np.zeros((40, overlay.shape[1], 3), dtype=np.uint8)
    status_bg[:] = (40, 40, 40)

    frame_text = f"Frame {result['frame_idx']:02d} | Phase: {result['phase']} | State: {result['state']}"
    cv2.putText(status_bg, frame_text, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

    if tracker.params.velocity_magnitude:
        params_text = f"Learned vel: {tracker.params.velocity_magnitude:.1f} px/f | Search: {tracker.params.search_radius:.0f}px"
    else:
        params_text = "Learning velocity..."
    cv2.putText(status_bg, params_text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.35, CYAN, 1)

    if result['detected']:
        cv2.putText(status_bg, "DETECTED", (540, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)
    else:
        cv2.putText(status_bg, "NOT FOUND", (540, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)

    combined = np.vstack([status_bg, overlay])
    return combined


# =============================================================================
# MAIN DETECTION PIPELINE
# =============================================================================

def run_detection():
    """Run ball detection on all frames and generate debug overlays."""
    os.makedirs(DEBUG_OUTPUT_PATH, exist_ok=True)

    meta = load_burst_meta()
    print(f"Loaded {len(meta)} frames from burst_meta.csv")

    tracker = BallTracker()
    results = []

    for frame_info in meta:
        frame_idx = frame_info['idx']
        phase = frame_info['phase']

        frame = load_frame(frame_idx)
        result = tracker.process_frame(frame_idx, frame, phase)
        results.append(result)

        overlay = draw_debug_overlay(frame, result, tracker)
        overlay_path = os.path.join(DEBUG_OUTPUT_PATH, f"debug_{frame_idx:04d}.png")
        cv2.imwrite(overlay_path, overlay)

        status = "DETECTED" if result['detected'] else "---"
        pos_str = f"({result['position'][0]:3d}, {result['position'][1]:3d})" if result['position'] else "(---, ---)"
        vel_str = ""
        if result.get('velocity'):
            vx, vy = result['velocity']
            vel_str = f" v=({vx:+.0f},{vy:+.0f})"
        print(f"Frame {frame_idx:02d} [{phase:4s}] {result['state']:15s} {status:8s} {pos_str}{vel_str}")

    # Summary
    print("\n" + "="*70)
    print("DETECTION SUMMARY")
    print("="*70)

    print(f"\nRest position: {tracker.rest_position}")
    print(f"Rest frames detected: {tracker.rest_frames}")

    if tracker.params.rest_ball_area:
        print(f"Learned ball area at rest: {tracker.params.rest_ball_area:.0f} px")

    if tracker.params.velocity_magnitude:
        print(f"Learned velocity: {tracker.params.velocity_magnitude:.1f} px/frame")
        if tracker.params.avg_velocity:
            print(f"  Direction: vx={tracker.params.avg_velocity[0]:.1f}, vy={tracker.params.avg_velocity[1]:.1f}")

    print(f"\nPost-impact detections: {len(tracker.post_impact_detections)}")
    for d in tracker.post_impact_detections:
        vel_str = ""
        if d.get('velocity'):
            vx, vy = d['velocity']
            vel_str = f" v=({vx:+.0f},{vy:+.0f})"
        print(f"  Frame {d['frame_idx']:02d}: ({d['x']:3d}, {d['y']:3d}) area={d['area']:.0f} circ={d['circularity']:.2f}{vel_str}")

    consecutive = 0
    max_consecutive = 0
    for i, d in enumerate(tracker.post_impact_detections):
        if i == 0:
            consecutive = 1
        else:
            prev = tracker.post_impact_detections[i-1]
            if d['frame_idx'] == prev['frame_idx'] + 1:
                consecutive += 1
            else:
                consecutive = 1
        max_consecutive = max(max_consecutive, consecutive)

    print(f"\nMax consecutive post-impact frames: {max_consecutive}")

    if max_consecutive >= 3:
        print("\n*** SUCCESS: Detected ball in 3+ consecutive post-impact frames! ***")
    else:
        print("\n*** NEEDS IMPROVEMENT: Less than 3 consecutive detections ***")

    print(f"\nDebug overlays saved to: {DEBUG_OUTPUT_PATH}")

    return results, tracker


# =============================================================================
# METRICS CALCULATION
# =============================================================================

class MetricsCalculator:
    """Calculates golf swing metrics from tracked ball positions."""

    def __init__(self, tracker: BallTracker, frame_timestamps: dict = None):
        self.tracker = tracker
        self.frame_timestamps = frame_timestamps or {}
        self.metrics = {}

    def calculate_all(self) -> dict:
        detections = self.tracker.post_impact_detections

        if len(detections) < 2:
            print("ERROR: Need at least 2 post-impact detections to calculate metrics")
            return {}

        ball_speed_mph, velocity_mps = self._calculate_ball_speed(detections)
        launch_angle_deg = self._calculate_launch_angle(detections)
        apex_height_ft, carry_distance_yds = self._simulate_trajectory(
            velocity_mps, launch_angle_deg
        )

        self.metrics = {
            'ball_speed_mph': ball_speed_mph,
            'launch_angle_deg': launch_angle_deg,
            'apex_height_ft': apex_height_ft,
            'carry_distance_yds': carry_distance_yds,
            'velocity_mps': velocity_mps,
            'num_detections': len(detections),
        }

        return self.metrics

    def _calculate_ball_speed(self, detections: list) -> tuple:
        """Calculate ball speed from initial post-impact detections."""
        num_samples = min(5, len(detections) - 1)
        velocities_mps = []

        for i in range(num_samples):
            d1 = detections[i]
            d2 = detections[i + 1]

            dx_px = d2['x'] - d1['x']
            dy_px = d2['y'] - d1['y']

            dx_m = dx_px * METERS_PER_PIXEL
            dy_m = -dy_px * METERS_PER_PIXEL

            if self.frame_timestamps:
                t1 = self.frame_timestamps.get(d1['frame_idx'], 0)
                t2 = self.frame_timestamps.get(d2['frame_idx'], 0)
                dt = (t2 - t1) / 1_000_000_000  # Nanoseconds to seconds
            else:
                dt = SECONDS_PER_FRAME

            if dt > 0:
                vx = dx_m / dt
                vy = dy_m / dt
                speed = np.sqrt(vx**2 + vy**2)
                velocities_mps.append(speed)

        if not velocities_mps:
            return 0, 0

        initial_speed_mps = max(velocities_mps)
        initial_speed_mph = initial_speed_mps * 2.237

        print(f"  Velocity samples (m/s): {[f'{v:.1f}' for v in velocities_mps]}")
        print(f"  Using max initial speed: {initial_speed_mps:.1f} m/s ({initial_speed_mph:.1f} mph)")

        return initial_speed_mph, initial_speed_mps

    def _calculate_launch_angle(self, detections: list) -> float:
        """Calculate launch angle from trajectory."""
        num_points = min(5, len(detections))

        x_vals = [d['x'] for d in detections[:num_points]]
        y_vals = [d['y'] for d in detections[:num_points]]

        rest_x = self.tracker.rest_position[0] if self.tracker.rest_position else x_vals[0]
        rest_y = self.tracker.rest_position[1] if self.tracker.rest_position else y_vals[0]

        x_m = [(x - rest_x) * METERS_PER_PIXEL for x in x_vals]
        y_m = [-(y - rest_y) * METERS_PER_PIXEL for y in y_vals]

        if len(x_m) >= 2 and (max(x_m) - min(x_m)) > 0:
            slope, intercept = np.polyfit(x_m, y_m, 1)
            launch_angle_deg = np.degrees(np.arctan(slope))
        else:
            if len(detections) >= 2:
                dx = (detections[1]['x'] - detections[0]['x']) * METERS_PER_PIXEL
                dy = -(detections[1]['y'] - detections[0]['y']) * METERS_PER_PIXEL
                if dx != 0:
                    launch_angle_deg = np.degrees(np.arctan(dy / dx))
                else:
                    launch_angle_deg = 90 if dy > 0 else -90
            else:
                launch_angle_deg = 0

        print(f"  Trajectory points (m): x={[f'{v:.3f}' for v in x_m]}")
        print(f"  Trajectory points (m): y={[f'{v:.3f}' for v in y_m]}")
        print(f"  Launch angle: {launch_angle_deg:.1f} degrees")

        return launch_angle_deg

    def _simulate_trajectory(self, velocity_mps: float, launch_angle_deg: float) -> tuple:
        """Simulate ball trajectory with air resistance."""
        if velocity_mps <= 0 or launch_angle_deg <= 0:
            return 0, 0

        launch_angle_rad = np.radians(launch_angle_deg)

        x = 0.0
        y = 0.0
        vx = velocity_mps * np.cos(launch_angle_rad)
        vy = velocity_mps * np.sin(launch_angle_rad)

        dt = 0.001
        max_time = 30
        max_height = 0.0

        drag_constant = (0.5 * AIR_DENSITY_KG_M3 * DRAG_COEFFICIENT *
                         BALL_CROSS_SECTION_M2 / BALL_MASS_KG)

        t = 0
        has_risen = False

        while t < max_time:
            if y > max_height:
                max_height = y
                has_risen = True

            if has_risen and y <= 0:
                break

            v = np.sqrt(vx**2 + vy**2)

            if v > 0:
                drag_ax = -drag_constant * v * vx
                drag_ay = -drag_constant * v * vy
            else:
                drag_ax = 0
                drag_ay = 0

            ax = drag_ax
            ay = -GRAVITY_M_S2 + drag_ay

            vx += ax * dt
            vy += ay * dt

            x += vx * dt
            y += vy * dt

            t += dt

        apex_height_ft = max_height * 3.281
        carry_distance_yds = x * 1.094

        print(f"  Trajectory simulation (with air resistance):")
        print(f"    Initial velocity: {velocity_mps:.1f} m/s at {launch_angle_deg:.1f} deg")
        print(f"    Flight time: {t:.2f} seconds")
        print(f"    Apex height: {max_height:.2f} m ({apex_height_ft:.1f} ft)")
        print(f"    Carry distance: {x:.2f} m ({carry_distance_yds:.1f} yds)")

        return apex_height_ft, carry_distance_yds

    def save_to_csv(self, filepath: str):
        if not self.metrics:
            print("ERROR: No metrics calculated. Run calculate_all() first.")
            return

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value', 'Unit'])
            writer.writerow(['Ball Speed', f"{self.metrics['ball_speed_mph']:.1f}", 'mph'])
            writer.writerow(['Launch Angle', f"{self.metrics['launch_angle_deg']:.1f}", 'degrees'])
            writer.writerow(['Apex Height', f"{self.metrics['apex_height_ft']:.1f}", 'feet'])
            writer.writerow(['Carry Distance', f"{self.metrics['carry_distance_yds']:.1f}", 'yards'])

        print(f"\nMetrics saved to: {filepath}")

    def print_summary(self):
        if not self.metrics:
            print("No metrics calculated.")
            return

        print("\n" + "="*50)
        print("CALCULATED METRICS")
        print("="*50)
        print(f"  Ball Speed:       {self.metrics['ball_speed_mph']:.1f} mph")
        print(f"  Launch Angle:     {self.metrics['launch_angle_deg']:.1f} degrees")
        print(f"  Apex Height:      {self.metrics['apex_height_ft']:.1f} feet")
        print(f"  Carry Distance:   {self.metrics['carry_distance_yds']:.1f} yards")
        print("="*50)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results, tracker = run_detection()

    meta = load_burst_meta()
    timestamps = {m['idx']: m['timestamp_us'] for m in meta}

    print("\n" + "="*70)
    print("CALCULATING METRICS")
    print("="*70)

    calculator = MetricsCalculator(tracker, timestamps)
    metrics = calculator.calculate_all()

    if metrics:
        calculator.print_summary()

        csv_path = os.path.join(
            r"C:\Users\kopil\OneDrive\Senior Project\Golf-Launch-Monitor\Calculations",
            "metrics.csv"
        )
        calculator.save_to_csv(csv_path)
