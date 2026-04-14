"""
Message definitions for Pi Communication Protocol
==================================================

This module defines all message types and helper functions
to create properly formatted messages.

Message Flow:
1. GUI -> Main: CLUB_SELECT
2. Main -> Sensor: CLUB_PRESET
3. Sensor -> Main: ACK_PRESET
4. User swings
5. Main -> Sensor: TRIGGER
6. Sensor -> Main: ACK_TRIGGER
7. Sensor computes
8. Sensor -> Main: CSV_META, CSV_CHUNK(s), CSV_DONE
9. Main -> GUI: SHOT_RESULT (via Flask)
"""

from uuid import uuid4
from typing import Dict, Any, Optional

from protocol import utc_now, SHARED_TOKEN
from config import CLUB_PRESETS, DEFAULT_CLUB


# =============================================================================
# MESSAGE TYPES (constants)
# =============================================================================

# Club selection flow
MSG_CLUB_SELECT = "CLUB_SELECT"     # GUI -> Main
MSG_CLUB_PRESET = "CLUB_PRESET"     # Main -> Sensor
MSG_ACK_PRESET = "ACK_PRESET"       # Sensor -> Main

# Shot capture flow
MSG_TRIGGER = "TRIGGER"             # Main -> Sensor
MSG_ACK_TRIGGER = "ACK_TRIGGER"     # Sensor -> Main

# CSV transfer flow
MSG_CSV_META = "CSV_META"           # Sensor -> Main (file info)
MSG_CSV_CHUNK = "CSV_CHUNK"         # Sensor -> Main (data chunks)
MSG_CSV_DONE = "CSV_DONE"           # Sensor -> Main (transfer complete)

# Results
MSG_SHOT_RESULT = "SHOT_RESULT"     # Main -> GUI (via Flask)

# Strobe control
MSG_STROBE_ARM = "STROBE_ARM"       # Main -> Sensor (turn strobe on)
MSG_STROBE_DISARM = "STROBE_DISARM" # Main -> Sensor (turn strobe off)
MSG_ACK_STROBE = "ACK_STROBE"       # Sensor -> Main

# Utility
MSG_ERROR = "ERROR"                 # Any direction (error response)
MSG_PING = "PING"                   # Health check
MSG_PONG = "PONG"                   # Health check response


# =============================================================================
# MESSAGE BUILDERS
# =============================================================================

def new_request_id() -> str:
    """Generate a new unique request ID."""
    return str(uuid4())


def new_shot_id() -> str:
    """Generate a new unique shot ID."""
    return str(uuid4())


def base_header(msg_type: str, request_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Create base header with common fields.
    
    All messages should start with this.
    """
    return {
        "msg_type": msg_type,
        "request_id": request_id or new_request_id(),
        "timestamp": utc_now(),
        "token": SHARED_TOKEN,
    }


# =============================================================================
# CLUB SELECTION MESSAGES
# =============================================================================

def make_club_select(club_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Create CLUB_SELECT message (GUI -> Main).
    
    Sent when user selects a club in the GUI.
    """
    header = base_header(MSG_CLUB_SELECT)
    header["club_id"] = club_id
    if user_id:
        header["user_id"] = user_id
    return header


def make_club_preset(
    club_id: str,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create CLUB_PRESET message (Main -> Sensor).
    
    Tells Sensor Pi to prepare for shots with this club.
    Includes preset settings for the club.
    """
    header = base_header(MSG_CLUB_PRESET, request_id)
    header["club_id"] = club_id
    
    # Get preset for this club (or default)
    preset = CLUB_PRESETS.get(club_id, CLUB_PRESETS.get(DEFAULT_CLUB, {}))
    header["preset_version"] = preset.get("preset_version", 1)
    header["preset_data"] = preset
    
    return header


def make_ack_preset(
    club_id: str,
    preset_version: int,
    request_id: str
) -> Dict[str, Any]:
    """
    Create ACK_PRESET message (Sensor -> Main).
    
    Confirms Sensor Pi received and applied club preset.
    """
    header = base_header(MSG_ACK_PRESET, request_id)
    header["club_id"] = club_id
    header["preset_version"] = preset_version
    header["status"] = "PRESET_APPLIED"
    return header


# =============================================================================
# TRIGGER MESSAGES
# =============================================================================

def make_trigger(
    shot_id: str,
    club_id: str,
    preset_version: int,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create TRIGGER message (Main -> Sensor).
    
    Sent when user swings. Tells Sensor Pi to capture shot.
    club_id in TRIGGER is authoritative (final truth).
    """
    header = base_header(MSG_TRIGGER, request_id)
    header["shot_id"] = shot_id
    header["club_id"] = club_id
    header["preset_version"] = preset_version
    return header


def make_ack_trigger(
    shot_id: str,
    club_id: str,
    request_id: str
) -> Dict[str, Any]:
    """
    Create ACK_TRIGGER message (Sensor -> Main).
    
    Confirms Sensor Pi received trigger and started processing.
    """
    header = base_header(MSG_ACK_TRIGGER, request_id)
    header["shot_id"] = shot_id
    header["club_id"] = club_id
    header["status"] = "PROCESSING"
    return header


# =============================================================================
# CSV TRANSFER MESSAGES
# =============================================================================

def make_csv_meta(
    shot_id: str,
    club_id: str,
    filename: str,
    file_size: int,
    sha256: str,
    total_chunks: int,
    request_id: str
) -> Dict[str, Any]:
    """
    Create CSV_META message (Sensor -> Main).
    
    Sent before CSV data to tell Main Pi what to expect.
    """
    header = base_header(MSG_CSV_META, request_id)
    header["shot_id"] = shot_id
    header["club_id"] = club_id
    header["filename"] = filename
    header["file_size"] = file_size
    header["sha256"] = sha256
    header["total_chunks"] = total_chunks
    return header


def make_csv_chunk(
    shot_id: str,
    club_id: str,
    chunk_index: int,
    total_chunks: int,
    request_id: str
) -> Dict[str, Any]:
    """
    Create CSV_CHUNK header (Sensor -> Main).
    
    Note: The actual chunk data is sent as payload, not in header.
    """
    header = base_header(MSG_CSV_CHUNK, request_id)
    header["shot_id"] = shot_id
    header["club_id"] = club_id
    header["chunk_index"] = chunk_index
    header["total_chunks"] = total_chunks
    return header


def make_csv_done(
    shot_id: str,
    club_id: str,
    sha256: str,
    file_size: int,
    request_id: str
) -> Dict[str, Any]:
    """
    Create CSV_DONE message (Sensor -> Main).
    
    Signals end of CSV transfer. Main Pi should verify checksum.
    """
    header = base_header(MSG_CSV_DONE, request_id)
    header["shot_id"] = shot_id
    header["club_id"] = club_id
    header["sha256"] = sha256
    header["file_size"] = file_size
    header["status"] = "TRANSFER_COMPLETE"
    return header


# =============================================================================
# RESULT MESSAGES
# =============================================================================

def make_shot_result(
    shot_id: str,
    club_id: str,
    metrics: Dict[str, Any],
    csv_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create SHOT_RESULT message (Main -> GUI via Flask).
    
    Contains parsed shot metrics for display.
    """
    header = base_header(MSG_SHOT_RESULT)
    header["shot_id"] = shot_id
    header["club_id"] = club_id
    header["metrics"] = metrics
    if csv_path:
        header["csv_path"] = csv_path
    return header


# =============================================================================
# ERROR MESSAGES
# =============================================================================

def make_error(
    error_message: str,
    request_id: Optional[str] = None,
    shot_id: Optional[str] = None,
    error_code: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create ERROR message.
    
    Can be sent in any direction to report errors.
    """
    header = base_header(MSG_ERROR, request_id)
    header["error"] = error_message
    if shot_id:
        header["shot_id"] = shot_id
    if error_code:
        header["error_code"] = error_code
    return header


# =============================================================================
# STROBE CONTROL MESSAGES
# =============================================================================

def make_strobe_arm(request_id: Optional[str] = None) -> Dict[str, Any]:
    """Create STROBE_ARM message (Main -> Sensor). Turn strobe on."""
    return base_header(MSG_STROBE_ARM, request_id)


def make_strobe_disarm(request_id: Optional[str] = None) -> Dict[str, Any]:
    """Create STROBE_DISARM message (Main -> Sensor). Turn strobe off."""
    return base_header(MSG_STROBE_DISARM, request_id)


def make_ack_strobe(armed: bool, request_id: str) -> Dict[str, Any]:
    """Create ACK_STROBE message (Sensor -> Main)."""
    header = base_header(MSG_ACK_STROBE, request_id)
    header["strobe_armed"] = armed
    return header


# =============================================================================
# HEALTH CHECK MESSAGES
# =============================================================================

def make_ping() -> Dict[str, Any]:
    """Create PING message for health check."""
    return base_header(MSG_PING)


def make_pong(request_id: str) -> Dict[str, Any]:
    """Create PONG response to PING."""
    header = base_header(MSG_PONG, request_id)
    header["status"] = "OK"
    return header


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def is_valid_club_id(club_id: str) -> bool:
    """Check if club_id is in our preset list."""
    return club_id in CLUB_PRESETS


def get_preset_version(club_id: str) -> int:
    """Get preset version for a club."""
    preset = CLUB_PRESETS.get(club_id, CLUB_PRESETS.get(DEFAULT_CLUB, {}))
    return preset.get("preset_version", 1)


if __name__ == "__main__":
    # Test message creation
    print("Testing message builders...")
    print()
    
    # Club selection
    msg = make_club_select("driver", "user123")
    print(f"CLUB_SELECT: {msg}")
    print()
    
    # Preset
    msg = make_club_preset("driver")
    print(f"CLUB_PRESET: {msg}")
    print()
    
    # Trigger
    shot_id = new_shot_id()
    msg = make_trigger(shot_id, "driver", 1)
    print(f"TRIGGER: {msg}")
    print()
    
    print("All message builders working!")
