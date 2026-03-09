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

SENSOR_PI_IP = "192.168.50.1"      # <-- Sensor Pi's IP on hotspot (e.g., "192.168.43.101")
MAIN_PI_IP = "192.168.50.2"        # <-- Main Pi's IP on hotspot (e.g., "192.168.43.102")

# Option 2: Ethernet (direct cable) - uncomment these if using Ethernet
# SENSOR_PI_IP = "192.168.50.2"   # Sensor Pi static IP (eth0)
# MAIN_PI_IP = "192.168.50.1"     # Main Pi static IP (eth0)

# Port for Pi-to-Pi communication
PI_COMM_PORT = 5001

# =============================================================================
# FLASK/GUI CONFIGURATION
# =============================================================================

# Flask app runs on Main Pi, GUI connects to it
FLASK_HOST = "0.0.0.0"          # Listen on all interfaces
FLASK_PORT = 5000               # Port for Flask web app

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
CSV_SAVE_DIR = "/home/elumbis/main_data"

# Where Sensor Pi reads/generates CSV files (your capture pipeline output)
CSV_SOURCE_DIR = "/home/jkopila1/project/sensor_data"

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
    "5iron": {
        "preset_version": 1,
        "exposure": 120,
        "threshold": 0.7,
        "expected_speed_range": (100, 140),
    },
    "7iron": {
        "preset_version": 1,
        "exposure": 130,
        "threshold": 0.65,
        "expected_speed_range": (90, 130),
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
    "sand_wedge": {
        "preset_version": 1,
        "exposure": 160,
        "threshold": 0.5,
        "expected_speed_range": (60, 100),
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
