"""
Golf Ball Detection - Adaptive Implementation
==============================================
Detects golf ball at rest and tracks it through post-impact frames.
Adapts to different swing speeds by learning velocity from initial detections.
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

# Frame source path (Windows)
RAW_BURST_PATH = r"C:\Users\kopil\OneDrive\Senior Project\raw_burst"

# Output path for debug overlays
DEBUG_OUTPUT_PATH = r"C:\Users\kopil\OneDrive\Senior Project\Golf-Launch-Monitor\Calculations\debug_overlay"

# Frame dimensions
FRAME_WIDTH = 640
FRAME_HEIGHT = 400

# Camera parameters
FRAME_RATE_FPS = 288  # Approximate capture rate

# =============================================================================
# CAMERA CALIBRATION - Physical Setup
# =============================================================================
# Camera: Arducam OV9281 with 2.8mm lens (~70 degree horizontal FOV)
# Distance from ball: ~38 inches (0.965 meters)
# Lens center height: ~7.125 inches (0.181 meters) above ground
# Ball diameter: 1.68 inches (42.67mm)

CAMERA_DISTANCE_M = 0.965          # Distance from camera to ball (meters)
CAMERA_HEIGHT_M = 0.181            # Height of camera lens center (meters)
BALL_DIAMETER_M = 0.04267          # Golf ball diameter (meters)
HORIZONTAL_FOV_DEG = 70.0          # Horizontal field of view (degrees)

# Derived calibration values
HORIZONTAL_FOV_RAD = np.radians(HORIZONTAL_FOV_DEG)
# Pixels per meter at the ball's distance
# FOV covers FRAME_WIDTH pixels, representing 2 * distance * tan(FOV/2) meters
FIELD_WIDTH_M = 2 * CAMERA_DISTANCE_M * np.tan(HORIZONTAL_FOV_RAD / 2)
PIXELS_PER_METER = FRAME_WIDTH / FIELD_WIDTH_M
METERS_PER_PIXEL = FIELD_WIDTH_M / FRAME_WIDTH

# Time between frames
SECONDS_PER_FRAME = 1.0 / FRAME_RATE_FPS

# Physics constants
GRAVITY_M_S2 = 9.81  # Acceleration due to gravity (m/s^2)

# Air density and drag parameters (for carry distance calculation)
AIR_DENSITY_KG_M3 = 1.225      # Standard air density at sea level (kg/m^3)
BALL_MASS_KG = 0.04593         # Golf ball mass (45.93 grams)
BALL_RADIUS_M = 0.02135        # Golf ball radius (meters)
BALL_CROSS_SECTION_M2 = np.pi * BALL_RADIUS_M ** 2  # Cross-sectional area
DRAG_COEFFICIENT = 0.25        # Typical golf ball drag coefficient (dimpled ball)

# Known ball rest position (approximate - used as initial search hint)
BALL_REST_X = 223
BALL_REST_Y = 270

# Detection parameters - ADAPTIVE RANGES
# These are wide ranges to accommodate different conditions
BRIGHTNESS_THRESHOLD_DEFAULT = 80
MOTION_THRESHOLD_DEFAULT = 25

# Ball size can vary with distance and motion blur
MIN_BALL_AREA_REST = 40        # At rest, ball is clear
MAX_BALL_AREA_REST = 300
MIN_BALL_AREA_FLIGHT = 20      # In flight, can be smaller due to blur or partial visibility
MAX_BALL_AREA_FLIGHT = 500     # Can be larger due to motion blur trails

# Circularity ranges (motion blur makes ball less circular)
MIN_CIRCULARITY_REST = 0.4
MIN_CIRCULARITY_FLIGHT = 0.15

# Search regions
REST_SEARCH_RADIUS = 80        # Wider search for rest position
POST_IMPACT_MIN_X = 150        # Ball must be right of this after impact (allow for slower shots)

# Velocity constraints (will be learned and adapted)
# These are INITIAL estimates, adjusted based on actual detections
# Slow swing: ~50 mph = ~22 m/s = ~580 px/s at 288fps = ~2 px/frame
# Fast swing: ~180 mph = ~80 m/s = ~2100 px/s at 288fps = ~7.3 px/frame (ball speed, not club)
# Accounting for camera distance and angle, expect 15-80 px/frame displacement
MIN_VELOCITY_PX_FRAME = 5      # Minimum expected ball movement per frame
MAX_VELOCITY_PX_FRAME = 150    # Maximum expected (very fast + motion blur center shift)


# =============================================================================
# ADAPTIVE PARAMETERS CLASS
# =============================================================================

class AdaptiveParams:
    """
    Learns and adapts detection parameters based on observed ball behavior.
    This allows the system to handle different swing speeds automatically.
    """
    
    def __init__(self):
        # Thresholds (can be auto-tuned)
        self.brightness_threshold = BRIGHTNESS_THRESHOLD_DEFAULT
        self.motion_threshold = MOTION_THRESHOLD_DEFAULT
        
        # Learned ball characteristics from rest detection
        self.rest_ball_area = None
        self.rest_ball_circularity = None
        self.rest_ball_brightness = None
        
        # Learned velocity from first few post-impact frames
        self.velocities = []  # List of (vx, vy) measurements
        self.avg_velocity = None
        self.velocity_magnitude = None
        
        # Adaptive search window
        self.search_radius = 60  # Initial search radius around predicted position
        
        # Direction consistency
        self.primary_direction = None  # (dx_sign, dy_sign) expected direction
        
    def learn_from_rest(self, area: float, circularity: float, brightness: float):
        """Learn ball characteristics from rest detection."""
        self.rest_ball_area = area
        self.rest_ball_circularity = circularity
        self.rest_ball_brightness = brightness
        
    def add_velocity_sample(self, vx: float, vy: float):
        """Add a velocity measurement from consecutive frame tracking."""
        self.velocities.append((vx, vy))
        
        # Update running average
        if len(self.velocities) >= 2:
            recent = self.velocities[-3:] if len(self.velocities) >= 3 else self.velocities
            avg_vx = sum(v[0] for v in recent) / len(recent)
            avg_vy = sum(v[1] for v in recent) / len(recent)
            self.avg_velocity = (avg_vx, avg_vy)
            self.velocity_magnitude = np.sqrt(avg_vx**2 + avg_vy**2)
            
            # Learn primary direction
            self.primary_direction = (1 if avg_vx > 0 else -1, 1 if avg_vy > 0 else -1)
            
            # Adapt search radius based on velocity magnitude
            # Faster ball = larger search window for next frame
            self.search_radius = max(40, min(150, self.velocity_magnitude * 1.5))
    
    def predict_next_position(self, current_pos: tuple) -> tuple:
        """Predict where ball will be in next frame based on learned velocity."""
        if self.avg_velocity:
            return (
                current_pos[0] + self.avg_velocity[0],
                current_pos[1] + self.avg_velocity[1]
            )
        return None
    
    def get_area_range(self, in_flight: bool) -> tuple:
        """Get acceptable area range, adapted to observed ball size."""
        if in_flight:
            base_min = MIN_BALL_AREA_FLIGHT
            base_max = MAX_BALL_AREA_FLIGHT
        else:
            base_min = MIN_BALL_AREA_REST
            base_max = MAX_BALL_AREA_REST
            
        # If we learned the rest area, use it to refine expectations
        if self.rest_ball_area:
            # In flight, ball area can vary from 0.3x to 2x rest area
            if in_flight:
                learned_min = self.rest_ball_area * 0.2
                learned_max = self.rest_ball_area * 3.0
                return (max(15, learned_min), max(learned_max, base_max))
            else:
                # At rest, should be close to learned area
                return (self.rest_ball_area * 0.5, self.rest_ball_area * 1.5)
        
        return (base_min, base_max)
    
    def get_velocity_range(self) -> tuple:
        """Get acceptable velocity range based on learned data."""
        if self.velocity_magnitude:
            # Allow 0.3x to 2x of observed velocity
            return (
                max(3, self.velocity_magnitude * 0.3),
                max(self.velocity_magnitude * 2.5, MAX_VELOCITY_PX_FRAME)
            )
        return (MIN_VELOCITY_PX_FRAME, MAX_VELOCITY_PX_FRAME)


# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================

def load_frame(frame_idx: int) -> np.ndarray:
    """Load a frame by index."""
    path = os.path.join(RAW_BURST_PATH, f"frame_{frame_idx:04d}.png")
    img = cv2.imread(path)
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


def get_contour_features(contour, gray_frame=None) -> dict:
    """Extract features from a contour."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Circularity
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter ** 2)
    else:
        circularity = 0
    
    # Centroid
    M = cv2.moments(contour)
    if M['m00'] > 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = 0, 0
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Aspect ratio
    aspect_ratio = w / h if h > 0 else 0
    
    # Mean brightness within contour (if frame provided)
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
    Detect ball at rest using direct thresholding.
    Uses hint_position if provided, otherwise searches near expected position.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Try multiple threshold levels to handle different lighting
    thresholds = [params.brightness_threshold, 
                  params.brightness_threshold - 15,
                  params.brightness_threshold + 15]
    
    all_candidates = []
    
    for thresh_val in thresholds:
        if thresh_val < 40 or thresh_val > 200:
            continue
            
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        area_min, area_max = params.get_area_range(in_flight=False)
        
        for contour in contours:
            features = get_contour_features(contour, gray)
            
            # Filter by area
            if not (area_min < features['area'] < area_max):
                continue
            
            # Filter by circularity (ball at rest should be fairly round)
            if features['circularity'] < MIN_CIRCULARITY_REST:
                continue
            
            # Calculate distance to hint position or expected position
            target_x = hint_position[0] if hint_position else BALL_REST_X
            target_y = hint_position[1] if hint_position else BALL_REST_Y
            dist = np.sqrt((features['cx'] - target_x)**2 + (features['cy'] - target_y)**2)
            
            # Must be within search radius
            if dist > REST_SEARCH_RADIUS:
                continue
            
            # Score: prefer closer to expected, higher circularity, moderate area
            score = 100 - dist + features['circularity'] * 50
            features['score'] = score
            features['dist_to_expected'] = dist
            all_candidates.append(features)
    
    if not all_candidates:
        return None
    
    # Return best candidate
    all_candidates.sort(key=lambda x: x['score'], reverse=True)
    return all_candidates[0]


def detect_ball_motion(current_frame: np.ndarray, reference_frame: np.ndarray,
                       params: AdaptiveParams, prev_position: tuple = None,
                       rest_position: tuple = None) -> list:
    """
    Detect ball using frame differencing with adaptive parameters.
    Returns list of candidates sorted by likelihood.
    """
    gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    
    # Try multiple motion thresholds
    thresholds = [params.motion_threshold,
                  params.motion_threshold - 5,
                  params.motion_threshold + 10]
    
    all_candidates = []
    seen_positions = set()  # Avoid duplicates from different thresholds
    
    for thresh_val in thresholds:
        if thresh_val < 10:
            continue
            
        # Frame difference
        diff = cv2.absdiff(gray_curr, gray_ref)
        _, thresh = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Clean up noise
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get expected area range for flight
        area_min, area_max = params.get_area_range(in_flight=True)
        
        # If near edge, allow smaller areas
        for contour in contours:
            features = get_contour_features(contour, gray_curr)
            
            # Near-edge adjustment
            near_edge = features['cx'] > 580 or features['cy'] < 60 or features['cx'] < 60
            effective_area_min = area_min * 0.3 if near_edge else area_min
            effective_circ_min = 0.1 if near_edge else MIN_CIRCULARITY_FLIGHT
            
            # Filter by area
            if not (effective_area_min < features['area'] < area_max):
                continue
            
            # Filter by circularity
            if features['circularity'] < effective_circ_min:
                continue
            
            # Avoid duplicate detections (same position from different thresholds)
            pos_key = (features['cx'] // 10, features['cy'] // 10)
            if pos_key in seen_positions:
                continue
            seen_positions.add(pos_key)
            
            # Calculate score
            score = calculate_candidate_score(features, params, prev_position, rest_position)
            features['score'] = score
            all_candidates.append(features)
    
    # Sort by score (highest first)
    all_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    return all_candidates


def calculate_candidate_score(features: dict, params: AdaptiveParams,
                              prev_position: tuple, rest_position: tuple) -> float:
    """
    Calculate likelihood score for a ball candidate.
    Higher score = more likely to be the ball.
    """
    score = 0
    
    # Base score from circularity (rounder is better)
    score += features['circularity'] * 100
    
    # If we have previous position, analyze trajectory
    if prev_position:
        dx = features['cx'] - prev_position[0]
        dy = features['cy'] - prev_position[1]
        velocity = np.sqrt(dx**2 + dy**2)
        
        # Get expected velocity range
        vel_min, vel_max = params.get_velocity_range()
        
        # Velocity within expected range
        if vel_min <= velocity <= vel_max:
            score += 50
        elif velocity < vel_min:
            score -= 30  # Too slow
        elif velocity > vel_max * 1.5:
            score -= 100  # Way too fast, probably noise
        
        # Direction consistency
        if params.primary_direction:
            expected_dx_sign, expected_dy_sign = params.primary_direction
            actual_dx_sign = 1 if dx > 0 else -1
            actual_dy_sign = 1 if dy > 0 else -1
            
            if actual_dx_sign == expected_dx_sign:
                score += 40  # Moving in expected horizontal direction
            else:
                score -= 80  # Wrong direction is very suspicious
            
            # Vertical direction (ball going up initially)
            if actual_dy_sign == expected_dy_sign:
                score += 20
        else:
            # No learned direction yet - assume ball moves right and up
            if dx > 0:
                score += 30
            if dy < 0:  # Up in image coordinates
                score += 20
        
        # Predicted position bonus
        predicted = params.predict_next_position(prev_position)
        if predicted:
            dist_to_predicted = np.sqrt((features['cx'] - predicted[0])**2 + 
                                        (features['cy'] - predicted[1])**2)
            # Closer to predicted position = higher score
            if dist_to_predicted < params.search_radius:
                score += 30 * (1 - dist_to_predicted / params.search_radius)
    
    # Distance from rest position (should be moving away)
    if rest_position:
        dist_from_rest = np.sqrt((features['cx'] - rest_position[0])**2 + 
                                 (features['cy'] - rest_position[1])**2)
        if dist_from_rest > 30:
            score += 20  # Good, ball has moved from rest
        else:
            score -= 50  # Still too close to rest, might be false detection
    
    # Ball should be in valid region of frame
    if features['cx'] < POST_IMPACT_MIN_X:
        score -= 100  # Too far left
    
    # Area consistency with learned ball size
    if params.rest_ball_area:
        area_ratio = features['area'] / params.rest_ball_area
        if 0.3 <= area_ratio <= 2.5:
            score += 15  # Area is reasonable compared to rest
        else:
            score -= 20  # Area is suspiciously different
    
    return score


# =============================================================================
# TRACKING STATE MACHINE
# =============================================================================

class BallTracker:
    """
    Tracks ball through pre-impact rest, impact, and post-impact flight.
    Adapts to different swing speeds by learning from initial detections.
    """
    
    def __init__(self):
        self.state = "INIT"
        self.rest_position = None
        self.rest_frames = []
        self.post_impact_detections = []
        self.reference_frame = None
        self.last_position = None
        self.params = AdaptiveParams()
        
        # For detecting when ball exits frame
        self.frames_since_last_detection = 0
        self.max_gap_frames = 3  # Allow up to 3 frames without detection before giving up
        
    def process_frame(self, frame_idx: int, frame: np.ndarray, phase: str) -> dict:
        """Process a single frame and update tracking state."""
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
        
        # Don't process more frames if ball has exited
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
        # Use previous detection as hint if available
        hint = self.rest_position if self.rest_position else None
        
        detection = detect_ball_at_rest(frame, self.params, hint)
        
        if detection:
            result['detected'] = True
            result['position'] = (detection['cx'], detection['cy'])
            result['area'] = detection['area']
            result['circularity'] = detection['circularity']
            
            # Learn from this detection
            self.params.learn_from_rest(
                detection['area'],
                detection['circularity'],
                detection['brightness']
            )
            
            # Only lock in rest position from early, stable frames
            # This adapts based on when the phase changes
            if self.rest_position is None or frame_idx <= 6:
                self.rest_position = (detection['cx'], detection['cy'])
                self.rest_frames.append(frame_idx)
                self.state = "REST_FOUND"
                self.reference_frame = frame.copy()
            else:
                # Later frames: verify detection is consistent with established rest
                dist = np.sqrt((detection['cx'] - self.rest_position[0])**2 + 
                               (detection['cy'] - self.rest_position[1])**2)
                if dist < 15:  # Close to established rest
                    self.rest_frames.append(frame_idx)
                else:
                    # This might be the club, not the ball
                    result['detected'] = False
                    result['position'] = None
                    result['note'] = "Rejected: too far from established rest position"
        
        return result
    
    def _process_post_impact(self, frame_idx: int, frame: np.ndarray, result: dict) -> dict:
        """Process post-impact frame to track ball in flight."""
        if self.reference_frame is None:
            # Fallback: try to load a known good frame
            try:
                self.reference_frame = load_frame(5)
            except:
                self.reference_frame = frame  # Last resort
        
        # Get predicted position if we have velocity data
        predicted = None
        if self.last_position:
            predicted = self.params.predict_next_position(self.last_position)
            result['predicted_position'] = predicted
        
        # Detect candidates
        candidates = detect_ball_motion(
            frame,
            self.reference_frame,
            self.params,
            self.last_position or self.rest_position,
            self.rest_position
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
                
                # Calculate and store velocity
                if self.last_position:
                    vx = best['cx'] - self.last_position[0]
                    vy = best['cy'] - self.last_position[1]
                    result['velocity'] = (vx, vy)
                    self.params.add_velocity_sample(vx, vy)
                
                # Record detection
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
                
                # Check if ball is exiting frame
                if best['cx'] > 620 or best['cy'] < 20 or best['cx'] < 20:
                    self.state = "BALL_EXIT"
            else:
                self.frames_since_last_detection += 1
        else:
            self.frames_since_last_detection += 1
        
        # If too many frames without detection, ball has probably left
        if self.frames_since_last_detection >= self.max_gap_frames:
            if len(self.post_impact_detections) > 0:
                self.state = "BALL_EXIT"
        
        return result
    
    def _validate_candidate(self, candidate: dict, frame_idx: int) -> bool:
        """
        Validate a candidate detection.
        Uses adaptive criteria based on learned parameters.
        """
        # Must have moved from rest
        if self.rest_position:
            dx_from_rest = candidate['cx'] - self.rest_position[0]
            # For right-handed golfer, ball moves right
            # Allow some tolerance for different setups
            if abs(dx_from_rest) < 15:
                return False  # Still at rest position
        
        # First detection after rest
        if not self.last_position or len(self.post_impact_detections) == 0:
            # First post-impact detection: just verify it's away from rest
            return True
        
        # Subsequent detections: check trajectory consistency
        dx = candidate['cx'] - self.last_position[0]
        dy = candidate['cy'] - self.last_position[1]
        velocity = np.sqrt(dx**2 + dy**2)
        
        vel_min, vel_max = self.params.get_velocity_range()
        
        # Velocity sanity check
        if velocity < vel_min * 0.2:
            return False  # Suspiciously slow
        if velocity > vel_max * 2:
            return False  # Impossibly fast
        
        # Direction check (learned direction)
        if self.params.primary_direction:
            expected_dx_sign = self.params.primary_direction[0]
            actual_dx_sign = 1 if dx > 0 else -1
            
            # Allow occasional small backward motion (edge cases, noise)
            if actual_dx_sign != expected_dx_sign and abs(dx) > 10:
                return False  # Significant backward movement
        
        # Passed all checks
        return True


# =============================================================================
# DEBUG OVERLAY GENERATION
# =============================================================================

def draw_debug_overlay(frame: np.ndarray, result: dict, tracker: BallTracker) -> np.ndarray:
    """Draw debug visualization on frame."""
    overlay = frame.copy()
    
    # Colors
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    BLUE = (255, 0, 0)
    CYAN = (255, 255, 0)
    WHITE = (255, 255, 255)
    MAGENTA = (255, 0, 255)
    
    # Draw rest position reference
    if tracker.rest_position:
        cv2.circle(overlay, tracker.rest_position, 20, BLUE, 1)
        cv2.putText(overlay, "REST", (tracker.rest_position[0]-15, tracker.rest_position[1]-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLUE, 1)
    
    # Draw predicted position (if available)
    if result.get('predicted_position'):
        pred = result['predicted_position']
        pred_int = (int(pred[0]), int(pred[1]))
        cv2.circle(overlay, pred_int, 10, MAGENTA, 1)
        cv2.putText(overlay, "PRED", (pred_int[0]+12, pred_int[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, MAGENTA, 1)
    
    # Draw search radius around last position
    if tracker.last_position and result['phase'] == 'post':
        cv2.circle(overlay, tracker.last_position, int(tracker.params.search_radius), CYAN, 1)
    
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
        
        # Show position and metrics
        info_text = f"({pos[0]}, {pos[1]})"
        if 'area' in result:
            info_text += f" A:{result['area']:.0f}"
        if 'circularity' in result:
            info_text += f" C:{result['circularity']:.2f}"
        cv2.putText(overlay, info_text, (pos[0]+15, pos[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, GREEN, 1)
        
        # Show velocity if available
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
    
    # Frame info (line 1)
    frame_text = f"Frame {result['frame_idx']:02d} | Phase: {result['phase']} | State: {result['state']}"
    cv2.putText(status_bg, frame_text, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
    
    # Adaptive params info (line 2)
    if tracker.params.velocity_magnitude:
        params_text = f"Learned vel: {tracker.params.velocity_magnitude:.1f} px/f | Search: {tracker.params.search_radius:.0f}px"
    else:
        params_text = "Learning velocity..."
    cv2.putText(status_bg, params_text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.35, CYAN, 1)
    
    # Detection status
    if result['detected']:
        cv2.putText(status_bg, "DETECTED", (540, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)
    else:
        cv2.putText(status_bg, "NOT FOUND", (540, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
    
    # Combine status bar with overlay
    combined = np.vstack([status_bg, overlay])
    
    return combined


# =============================================================================
# MAIN DETECTION PIPELINE
# =============================================================================

def run_detection():
    """Run ball detection on all frames and generate debug overlays."""
    
    # Create output directory
    os.makedirs(DEBUG_OUTPUT_PATH, exist_ok=True)
    
    # Load metadata
    meta = load_burst_meta()
    print(f"Loaded {len(meta)} frames from burst_meta.csv")
    
    # Initialize tracker
    tracker = BallTracker()
    
    # Process each frame
    results = []
    
    for frame_info in meta:
        frame_idx = frame_info['idx']
        phase = frame_info['phase']
        
        # Load frame
        frame = load_frame(frame_idx)
        
        # Process
        result = tracker.process_frame(frame_idx, frame, phase)
        results.append(result)
        
        # Generate debug overlay
        overlay = draw_debug_overlay(frame, result, tracker)
        
        # Save overlay
        overlay_path = os.path.join(DEBUG_OUTPUT_PATH, f"debug_{frame_idx:04d}.png")
        cv2.imwrite(overlay_path, overlay)
        
        # Print status
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
    
    # Check success criteria
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
    """
    Calculates golf swing metrics from tracked ball positions.
    
    Metrics calculated:
    - Ball Speed (mph): Initial velocity of the ball after impact
    - Launch Angle (degrees): Angle of ball trajectory relative to horizontal
    - Apex Height (feet): Maximum height the ball would reach (with air resistance)
    - Carry Distance (yards): How far the ball travels before landing (with air resistance)
    """
    
    def __init__(self, tracker: BallTracker, frame_timestamps: dict = None):
        self.tracker = tracker
        self.frame_timestamps = frame_timestamps or {}
        self.metrics = {}
        
    def calculate_all(self) -> dict:
        """Calculate all metrics from tracked data."""
        detections = self.tracker.post_impact_detections
        
        if len(detections) < 2:
            print("ERROR: Need at least 2 post-impact detections to calculate metrics")
            return {}
        
        # Calculate initial velocity (from first few frames after impact)
        ball_speed_mph, velocity_mps = self._calculate_ball_speed(detections)
        
        # Calculate launch angle
        launch_angle_deg = self._calculate_launch_angle(detections)
        
        # Simulate trajectory with air resistance to get apex and carry
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
        """
        Calculate ball speed from initial post-impact detections.
        Uses average velocity from first few frames for stability.
        
        Returns: (speed_mph, speed_mps)
        """
        # Use first 3-5 detections for initial velocity (before significant slowdown)
        num_samples = min(5, len(detections) - 1)
        
        velocities_mps = []
        
        for i in range(num_samples):
            d1 = detections[i]
            d2 = detections[i + 1]
            
            # Pixel displacement
            dx_px = d2['x'] - d1['x']
            dy_px = d2['y'] - d1['y']
            
            # Convert to meters
            dx_m = dx_px * METERS_PER_PIXEL
            dy_m = -dy_px * METERS_PER_PIXEL  # Negative because Y increases downward in image
            
            # Calculate time between frames
            if self.frame_timestamps:
                t1 = self.frame_timestamps.get(d1['frame_idx'], 0)
                t2 = self.frame_timestamps.get(d2['frame_idx'], 0)
                dt = (t2 - t1) / 1_000_000  # Convert microseconds to seconds
            else:
                # Use frame rate if timestamps not available
                dt = SECONDS_PER_FRAME
            
            if dt > 0:
                vx = dx_m / dt
                vy = dy_m / dt
                speed = np.sqrt(vx**2 + vy**2)
                velocities_mps.append(speed)
        
        if not velocities_mps:
            return 0, 0
        
        # Use maximum of first few samples (closest to true initial speed)
        # Ball slows down due to air resistance, so early frames have highest speed
        initial_speed_mps = max(velocities_mps)
        
        # Also calculate average for comparison
        avg_speed_mps = sum(velocities_mps) / len(velocities_mps)
        
        # Convert to mph (1 m/s = 2.237 mph)
        initial_speed_mph = initial_speed_mps * 2.237
        
        print(f"  Velocity samples (m/s): {[f'{v:.1f}' for v in velocities_mps]}")
        print(f"  Using max initial speed: {initial_speed_mps:.1f} m/s ({initial_speed_mph:.1f} mph)")
        
        return initial_speed_mph, initial_speed_mps
    
    def _calculate_launch_angle(self, detections: list) -> float:
        """
        Calculate launch angle from trajectory.
        Uses linear regression on first few points for best estimate.
        
        Returns: angle in degrees (positive = upward)
        """
        # Use first few detections (before gravity significantly curves the path)
        num_points = min(5, len(detections))
        
        # Extract x, y positions
        x_vals = [d['x'] for d in detections[:num_points]]
        y_vals = [d['y'] for d in detections[:num_points]]
        
        # Convert to real-world coordinates (meters)
        # X: horizontal distance from rest position
        rest_x = self.tracker.rest_position[0] if self.tracker.rest_position else x_vals[0]
        rest_y = self.tracker.rest_position[1] if self.tracker.rest_position else y_vals[0]
        
        x_m = [(x - rest_x) * METERS_PER_PIXEL for x in x_vals]
        y_m = [-(y - rest_y) * METERS_PER_PIXEL for y in y_vals]  # Flip Y axis
        
        # Linear regression to find slope
        # Using numpy polyfit for line of best fit
        if len(x_m) >= 2 and (max(x_m) - min(x_m)) > 0:
            slope, intercept = np.polyfit(x_m, y_m, 1)
            
            # Slope = rise/run = tan(angle)
            launch_angle_rad = np.arctan(slope)
            launch_angle_deg = np.degrees(launch_angle_rad)
        else:
            # Fallback: use first two points
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
        """
        Simulate ball trajectory with air resistance to calculate apex height and carry distance.
        
        Uses numerical integration (Euler method) with drag force:
        - F_drag = 0.5 * rho * v^2 * Cd * A
        
        Assumes:
        - No wind
        - Standard air density at sea level
        - No spin effects (simplified model)
        
        Returns: (apex_height_ft, carry_distance_yds)
        """
        if velocity_mps <= 0 or launch_angle_deg <= 0:
            return 0, 0
        
        launch_angle_rad = np.radians(launch_angle_deg)
        
        # Initial conditions
        x = 0.0  # Horizontal position (m)
        y = 0.0  # Vertical position (m) - starting at ground level
        vx = velocity_mps * np.cos(launch_angle_rad)  # Horizontal velocity (m/s)
        vy = velocity_mps * np.sin(launch_angle_rad)  # Vertical velocity (m/s)
        
        # Simulation parameters
        dt = 0.001  # Time step (seconds) - small for accuracy
        max_time = 30  # Maximum simulation time (seconds)
        
        # Track apex
        max_height = 0.0
        
        # Drag constant: k = 0.5 * rho * Cd * A / m
        drag_constant = (0.5 * AIR_DENSITY_KG_M3 * DRAG_COEFFICIENT * 
                         BALL_CROSS_SECTION_M2 / BALL_MASS_KG)
        
        # Simulate until ball hits ground (y < 0 after rising)
        t = 0
        has_risen = False
        
        while t < max_time:
            # Update max height
            if y > max_height:
                max_height = y
                has_risen = True
            
            # Check if ball has landed (y < 0 after rising)
            if has_risen and y <= 0:
                break
            
            # Current speed
            v = np.sqrt(vx**2 + vy**2)
            
            # Drag acceleration (opposes velocity direction)
            if v > 0:
                drag_ax = -drag_constant * v * vx
                drag_ay = -drag_constant * v * vy
            else:
                drag_ax = 0
                drag_ay = 0
            
            # Total acceleration
            ax = drag_ax
            ay = -GRAVITY_M_S2 + drag_ay
            
            # Update velocity (Euler integration)
            vx += ax * dt
            vy += ay * dt
            
            # Update position
            x += vx * dt
            y += vy * dt
            
            t += dt
        
        # Convert results
        apex_height_ft = max_height * 3.281  # meters to feet
        carry_distance_yds = x * 1.094  # meters to yards
        
        print(f"  Trajectory simulation (with air resistance):")
        print(f"    Initial velocity: {velocity_mps:.1f} m/s at {launch_angle_deg:.1f} deg")
        print(f"    Air density: {AIR_DENSITY_KG_M3} kg/m^3")
        print(f"    Drag coefficient: {DRAG_COEFFICIENT}")
        print(f"    Flight time: {t:.2f} seconds")
        print(f"    Apex height: {max_height:.2f} m ({apex_height_ft:.1f} ft)")
        print(f"    Carry distance: {x:.2f} m ({carry_distance_yds:.1f} yds)")
        
        return apex_height_ft, carry_distance_yds
    
    def save_to_csv(self, filepath: str):
        """Save metrics to CSV file."""
        if not self.metrics:
            print("ERROR: No metrics calculated. Run calculate_all() first.")
            return
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Metric', 'Value', 'Unit'])
            
            # Data rows
            writer.writerow(['Ball Speed', f"{self.metrics['ball_speed_mph']:.1f}", 'mph'])
            writer.writerow(['Launch Angle', f"{self.metrics['launch_angle_deg']:.1f}", 'degrees'])
            writer.writerow(['Apex Height', f"{self.metrics['apex_height_ft']:.1f}", 'feet'])
            writer.writerow(['Carry Distance', f"{self.metrics['carry_distance_yds']:.1f}", 'yards'])
        
        print(f"\nMetrics saved to: {filepath}")
    
    def print_summary(self):
        """Print a summary of calculated metrics."""
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
    
    # Build timestamp lookup from metadata
    meta = load_burst_meta()
    timestamps = {m['idx']: m['timestamp_us'] for m in meta}
    
    # Calculate metrics
    print("\n" + "="*70)
    print("CALCULATING METRICS")
    print("="*70)
    
    calculator = MetricsCalculator(tracker, timestamps)
    metrics = calculator.calculate_all()
    
    if metrics:
        calculator.print_summary()
        
        # Save to CSV
        csv_path = os.path.join(
            r"C:\Users\kopil\OneDrive\Senior Project\Golf-Launch-Monitor\Calculations",
            "metrics.csv"
        )
        calculator.save_to_csv(csv_path)
