"""
Configuration file for Pi Communication System
==============================================

IMPORTANT: Update the IP addresses below before running!

How to find your IP addresses:
1. Connect both Pis to your phone hotspot
2. On each Pi, run: hostname -I
3. Update the IPs below

For Ethernet (direct cable between Pis):
- These are your static IPs you already configured
- Example: SENSOR_PI_IP = "192.168.50.2"
"""

# =============================================================================
# NETWORK CONFIGURATION - UPDATE THESE!
# =============================================================================

# Option 1: Phone Hotspot Testing (update these when you connect)
# Run 'hostname -I' on each Pi to find their IPs

SENSOR_PI_IP = "CHANGE_ME"      # <-- Sensor Pi's IP on hotspot (e.g., "192.168.43.101")
MAIN_PI_IP = "CHANGE_ME"        # <-- Main Pi's IP on hotspot (e.g., "192.168.43.102")

# Option 2: Ethernet (direct cable) - uncomment these if using Ethernet
# SENSOR_PI_IP = "192.168.50.2"   # Sensor Pi static IP (eth0)
# MAIN_PI_IP = "192.168.50.1"     # Main Pi static IP (eth0)

# Port for Pi-to-Pi communication (Sensor Pi server)
PI_COMM_PORT = 5002             # Changed from 5001 to avoid conflict with bridge

# =============================================================================
# FLASK/GUI CONFIGURATION
# =============================================================================

# Main Flask app (your partner's app.py)
FLASK_HOST = "0.0.0.0"          # Listen on all interfaces
FLASK_PORT = 5000               # Port for main Flask web app

# Sensor Bridge Flask server (runs alongside app.py on Main Pi)
BRIDGE_PORT = 5001              # Port for sensor_bridge.py

# URL where app.py receives metrics (sensor_bridge posts here)
WEBAPP_METRICS_URL = "http://127.0.0.1:5000/api/upload_metrics"

# =============================================================================
# SECURITY
# =============================================================================

# Shared secret token - both Pis must have the same value
# Change this to something unique for your project!
SHARED_TOKEN = "golf-launch-monitor-2024"

# =============================================================================
# PROTOCOL SETTINGS
# =============================================================================

PROTOCOL_VERSION = 1
CHUNK_SIZE = 16 * 1024          # 16 KB chunks for CSV transfer
SOCKET_TIMEOUT = 30             # Seconds to wait for responses
RECONNECT_DELAY = 2             # Seconds between reconnection attempts
MAX_RETRIES = 3                 # Max retries for failed operations

# =============================================================================
# FILE PATHS
# =============================================================================

# Where Main Pi saves received CSV files
CSV_SAVE_DIR = "/home/pi/launch_data"

# Where Sensor Pi reads/generates CSV files (your capture pipeline output)
CSV_SOURCE_DIR = "/home/pi/sensor_data"

# =============================================================================
# CLUB NAME MAPPING (GUI names -> internal preset names)
# =============================================================================

# The GUI uses friendly names like "Driver", "3 Wood", "7 Iron"
# This maps them to our internal preset keys
GUI_TO_PRESET = {
    "Driver": "driver",
    "3 Wood": "3wood",
    "5 Wood": "5wood",
    "3 Hybrid": "3hybrid",
    "4 Hybrid": "4hybrid",
    "3 Iron": "3iron",
    "4 Iron": "4iron",
    "5 Iron": "5iron",
    "6 Iron": "6iron",
    "7 Iron": "7iron",
    "8 Iron": "8iron",
    "9 Iron": "9iron",
    "Pitching Wedge": "pitching_wedge",
    "Gap Wedge": "gap_wedge",
    "Sand Wedge": "sand_wedge",
    "Lob Wedge": "lob_wedge",
    "Putter": "putter",
}

# Reverse mapping (internal -> GUI)
PRESET_TO_GUI = {v: k for k, v in GUI_TO_PRESET.items()}


def normalize_club_name(club_name: str) -> str:
    """Convert GUI club name to internal preset name."""
    # First check direct mapping
    if club_name in GUI_TO_PRESET:
        return GUI_TO_PRESET[club_name]
    # Already a preset name?
    if club_name in CLUB_PRESETS:
        return club_name
    # Try lowercase
    lower = club_name.lower().replace(" ", "").replace("-", "_")
    if lower in CLUB_PRESETS:
        return lower
    # Default
    return DEFAULT_CLUB


# =============================================================================
# CLUB PRESETS
# =============================================================================

# Default club presets (camera settings, thresholds, etc.)
# Your sensor code should use these when processing shots
CLUB_PRESETS = {
    "driver": {
        "preset_version": 1,
        "exposure": 100,
        "threshold": 0.8,
        "expected_speed_range": (140, 180),
    },
    "3wood": {
        "preset_version": 1,
        "exposure": 110,
        "threshold": 0.75,
        "expected_speed_range": (130, 160),
    },
    "5wood": {
        "preset_version": 1,
        "exposure": 110,
        "threshold": 0.75,
        "expected_speed_range": (125, 155),
    },
    "3hybrid": {
        "preset_version": 1,
        "exposure": 115,
        "threshold": 0.72,
        "expected_speed_range": (115, 145),
    },
    "4hybrid": {
        "preset_version": 1,
        "exposure": 115,
        "threshold": 0.72,
        "expected_speed_range": (110, 140),
    },
    "3iron": {
        "preset_version": 1,
        "exposure": 120,
        "threshold": 0.7,
        "expected_speed_range": (105, 140),
    },
    "4iron": {
        "preset_version": 1,
        "exposure": 120,
        "threshold": 0.7,
        "expected_speed_range": (100, 135),
    },
    "5iron": {
        "preset_version": 1,
        "exposure": 120,
        "threshold": 0.7,
        "expected_speed_range": (100, 140),
    },
    "6iron": {
        "preset_version": 1,
        "exposure": 125,
        "threshold": 0.68,
        "expected_speed_range": (95, 130),
    },
    "7iron": {
        "preset_version": 1,
        "exposure": 130,
        "threshold": 0.65,
        "expected_speed_range": (90, 130),
    },
    "8iron": {
        "preset_version": 1,
        "exposure": 135,
        "threshold": 0.62,
        "expected_speed_range": (85, 125),
    },
    "9iron": {
        "preset_version": 1,
        "exposure": 140,
        "threshold": 0.6,
        "expected_speed_range": (80, 120),
    },
    "pitching_wedge": {
        "preset_version": 1,
        "exposure": 150,
        "threshold": 0.55,
        "expected_speed_range": (70, 110),
    },
    "gap_wedge": {
        "preset_version": 1,
        "exposure": 155,
        "threshold": 0.52,
        "expected_speed_range": (65, 105),
    },
    "sand_wedge": {
        "preset_version": 1,
        "exposure": 160,
        "threshold": 0.5,
        "expected_speed_range": (60, 100),
    },
    "lob_wedge": {
        "preset_version": 1,
        "exposure": 165,
        "threshold": 0.48,
        "expected_speed_range": (55, 95),
    },
    "putter": {
        "preset_version": 1,
        "exposure": 200,
        "threshold": 0.4,
        "expected_speed_range": (10, 50),
    },
}

# Default club if none selected
DEFAULT_CLUB = "7iron"


def validate_config():
    """Check that config is properly set up."""
    errors = []
    
    if SENSOR_PI_IP == "CHANGE_ME":
        errors.append("SENSOR_PI_IP not configured in config.py")
    if MAIN_PI_IP == "CHANGE_ME":
        errors.append("MAIN_PI_IP not configured in config.py")
    
    if errors:
        print("=" * 60)
        print("CONFIGURATION ERROR")
        print("=" * 60)
        for e in errors:
            print(f"  - {e}")
        print()
        print("Please edit config.py and set the correct IP addresses.")
        print("Run 'hostname -I' on each Pi to find its IP.")
        print("=" * 60)
        return False
    
    return True


if __name__ == "__main__":
    # Quick test when running directly
    print("Current Configuration:")
    print(f"  SENSOR_PI_IP: {SENSOR_PI_IP}")
    print(f"  MAIN_PI_IP: {MAIN_PI_IP}")
    print(f"  PI_COMM_PORT: {PI_COMM_PORT}")
    print(f"  FLASK_PORT: {FLASK_PORT}")
    print()
    
    if validate_config():
        print("Configuration looks good!")
    else:
        print("Please fix the configuration errors above.")
