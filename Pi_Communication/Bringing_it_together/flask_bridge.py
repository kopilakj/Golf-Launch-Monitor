#!/usr/bin/env python3
"""
Flask Bridge for Main Pi
========================

This module provides integration between the Pi communication system
and your partner's Flask web app.

There are two ways to use this:

1. IMPORT MODE: Import the functions into your existing Flask app
   from flask_bridge import handle_club_select, handle_trigger, get_shot_result

2. STANDALONE MODE: Run this file to start a simple Flask server
   python3 flask_bridge.py

The standalone mode is useful for testing before integrating with
your partner's app.
"""

import os
import sys
import json
import time
from threading import Thread, Lock
from queue import Queue, Empty

# Flask imports (install with: pip install flask flask-cors)
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Warning: Flask not installed. Run: pip install flask flask-cors")

# Import our Pi communication modules
from config import FLASK_HOST, FLASK_PORT, CLUB_PRESETS, DEFAULT_CLUB
from main_client import (
    sensor_conn, state, send_club_preset, trigger_shot, ping_sensor
)
from messages import new_shot_id


# =============================================================================
# BRIDGE STATE
# =============================================================================

class BridgeState:
    """Tracks state for Flask bridge."""
    
    def __init__(self):
        self.lock = Lock()
        self.last_result = None
        self.pending_shots = {}  # shot_id -> status
        self.error_log = []
    
    def store_result(self, result: dict):
        with self.lock:
            self.last_result = result
            shot_id = result.get("shot_id")
            if shot_id in self.pending_shots:
                del self.pending_shots[shot_id]
    
    def get_last_result(self) -> dict:
        with self.lock:
            return self.last_result
    
    def add_pending(self, shot_id: str):
        with self.lock:
            self.pending_shots[shot_id] = "pending"
    
    def get_pending(self) -> dict:
        with self.lock:
            return dict(self.pending_shots)
    
    def log_error(self, error: str):
        with self.lock:
            self.error_log.append({
                "time": time.time(),
                "error": error
            })
            # Keep only last 100 errors
            self.error_log = self.error_log[-100:]


bridge_state = BridgeState()


# =============================================================================
# API FUNCTIONS (for import into Flask app)
# =============================================================================

def handle_club_select(club_id: str, user_id: str = None) -> dict:
    """
    Handle club selection from GUI.
    
    Call this when user selects a club in your Flask app.
    
    Args:
        club_id: The club selected (e.g., "driver", "7iron")
        user_id: Optional user identifier
    
    Returns:
        dict with status and details
    
    Example:
        result = handle_club_select("driver")
        if result["success"]:
            print("Preset sent!")
    """
    if club_id not in CLUB_PRESETS:
        return {
            "success": False,
            "error": f"Unknown club: {club_id}",
            "available_clubs": list(CLUB_PRESETS.keys())
        }
    
    success = send_club_preset(club_id)
    
    if success:
        return {
            "success": True,
            "club_id": club_id,
            "preset_version": CLUB_PRESETS[club_id].get("preset_version", 1),
            "message": f"Preset applied for {club_id}"
        }
    else:
        bridge_state.log_error(f"Failed to send preset for {club_id}")
        return {
            "success": False,
            "error": "Failed to send preset to Sensor Pi",
            "connected": sensor_conn.connected
        }


def handle_trigger(club_id: str = None) -> dict:
    """
    Handle shot trigger from GUI.
    
    Call this when user swings/triggers a shot.
    
    Args:
        club_id: Optional club override. If None, uses current club.
    
    Returns:
        dict with shot_id and status
    
    Example:
        result = handle_trigger()
        if result["success"]:
            shot_id = result["shot_id"]
            # Poll for results or wait for callback
    """
    # Use provided club or current
    if club_id is None:
        club_id, _ = state.get_current_club()
    
    # Trigger shot (this blocks until CSV is received)
    result = trigger_shot(club_id)
    
    if result:
        bridge_state.store_result(result)
        
        # Format metrics for GUI
        formatted_metrics = {}
        for key, val in result.get("metrics", {}).items():
            if isinstance(val, dict):
                formatted_metrics[key] = val.get("value")
            else:
                formatted_metrics[key] = val
        
        return {
            "success": True,
            "shot_id": result["shot_id"],
            "club_id": result["club_id"],
            "metrics": formatted_metrics,
            "timestamp": result.get("timestamp")
        }
    else:
        bridge_state.log_error("Shot trigger failed")
        return {
            "success": False,
            "error": "Failed to capture shot",
            "connected": sensor_conn.connected
        }


def get_shot_result(shot_id: str = None) -> dict:
    """
    Get the result of a shot.
    
    Args:
        shot_id: Specific shot to get, or None for latest
    
    Returns:
        dict with shot result or status
    """
    result = bridge_state.get_last_result()
    
    if result is None:
        return {
            "success": False,
            "error": "No shot results available"
        }
    
    if shot_id and result.get("shot_id") != shot_id:
        return {
            "success": False,
            "error": f"Shot {shot_id} not found",
            "latest_shot_id": result.get("shot_id")
        }
    
    # Format for GUI
    formatted_metrics = {}
    for key, val in result.get("metrics", {}).items():
        if isinstance(val, dict):
            formatted_metrics[key] = val.get("value")
        else:
            formatted_metrics[key] = val
    
    return {
        "success": True,
        "shot_id": result["shot_id"],
        "club_id": result["club_id"],
        "metrics": formatted_metrics,
        "timestamp": result.get("timestamp")
    }


def get_status() -> dict:
    """
    Get current system status.
    
    Returns connection status, current club, etc.
    """
    club_id, preset_version = state.get_current_club()
    
    return {
        "connected": sensor_conn.connected,
        "current_club": club_id,
        "preset_version": preset_version,
        "preset_confirmed": state.preset_confirmed,
        "shots_triggered": state.shots_triggered,
        "shots_received": state.shots_received,
        "available_clubs": list(CLUB_PRESETS.keys())
    }


def check_connection() -> dict:
    """
    Check if Sensor Pi is reachable.
    
    Sends a PING and waits for PONG.
    """
    success = ping_sensor()
    
    return {
        "success": success,
        "connected": sensor_conn.connected,
        "sensor_ip": sensor_conn.sock.getpeername() if sensor_conn.sock else None
    }


# =============================================================================
# STANDALONE FLASK SERVER
# =============================================================================

def create_app():
    """Create Flask app with all routes."""
    
    if not FLASK_AVAILABLE:
        print("ERROR: Flask not installed!")
        print("Run: pip install flask flask-cors")
        sys.exit(1)
    
    app = Flask(__name__)
    CORS(app)  # Enable CORS for GUI access
    
    @app.route("/")
    def index():
        """Health check endpoint."""
        return jsonify({
            "service": "Golf Launch Monitor - Pi Bridge",
            "status": "running",
            "connected_to_sensor": sensor_conn.connected
        })
    
    @app.route("/api/status", methods=["GET"])
    def api_status():
        """Get system status."""
        return jsonify(get_status())
    
    @app.route("/api/clubs", methods=["GET"])
    def api_clubs():
        """Get available clubs."""
        return jsonify({
            "clubs": list(CLUB_PRESETS.keys()),
            "current": state.get_current_club()[0]
        })
    
    @app.route("/api/club/select", methods=["POST"])
    def api_club_select():
        """
        Select a club.
        
        POST body: {"club_id": "driver"}
        """
        data = request.get_json() or {}
        club_id = data.get("club_id")
        user_id = data.get("user_id")
        
        if not club_id:
            return jsonify({"success": False, "error": "club_id required"}), 400
        
        result = handle_club_select(club_id, user_id)
        return jsonify(result)
    
    @app.route("/api/trigger", methods=["POST"])
    def api_trigger():
        """
        Trigger a shot.
        
        POST body: {} or {"club_id": "driver"} to override
        """
        data = request.get_json() or {}
        club_id = data.get("club_id")
        
        result = handle_trigger(club_id)
        return jsonify(result)
    
    @app.route("/api/result", methods=["GET"])
    def api_result():
        """Get latest shot result."""
        shot_id = request.args.get("shot_id")
        result = get_shot_result(shot_id)
        return jsonify(result)
    
    @app.route("/api/ping", methods=["GET"])
    def api_ping():
        """Ping sensor pi."""
        result = check_connection()
        return jsonify(result)
    
    @app.route("/api/connect", methods=["POST"])
    def api_connect():
        """Connect to sensor pi."""
        success = sensor_conn.connect()
        return jsonify({
            "success": success,
            "connected": sensor_conn.connected
        })
    
    @app.route("/api/disconnect", methods=["POST"])
    def api_disconnect():
        """Disconnect from sensor pi."""
        sensor_conn.disconnect()
        return jsonify({
            "success": True,
            "connected": False
        })
    
    return app


def run_standalone():
    """Run standalone Flask server."""
    
    print()
    print("=" * 60)
    print("FLASK BRIDGE - STANDALONE MODE")
    print("=" * 60)
    print()
    print(f"Server will run on: http://{FLASK_HOST}:{FLASK_PORT}")
    print()
    print("API Endpoints:")
    print("  GET  /api/status      - System status")
    print("  GET  /api/clubs       - Available clubs")
    print("  POST /api/club/select - Select club {club_id}")
    print("  POST /api/trigger     - Trigger shot")
    print("  GET  /api/result      - Get last result")
    print("  GET  /api/ping        - Ping sensor")
    print("  POST /api/connect     - Connect to sensor")
    print("  POST /api/disconnect  - Disconnect")
    print()
    print("=" * 60)
    print()
    
    # Try to connect to sensor pi first
    print("Attempting to connect to Sensor Pi...")
    if sensor_conn.connect():
        print("Connected to Sensor Pi!")
    else:
        print("Could not connect to Sensor Pi.")
        print("Make sure sensor_server.py is running.")
        print("Starting server anyway...")
    print()
    
    # Create and run app
    app = create_app()
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True, threaded=True)


# =============================================================================
# EXAMPLE USAGE FOR YOUR PARTNER
# =============================================================================

"""
INTEGRATION GUIDE FOR YOUR PARTNER
==================================

Your partner can integrate this with their Flask app in two ways:

OPTION 1: Import functions directly
-----------------------------------

In their Flask routes, add:

    from flask_bridge import handle_club_select, handle_trigger, get_shot_result

    @app.route('/club/select', methods=['POST'])
    def select_club():
        club_id = request.json.get('club_id')
        result = handle_club_select(club_id)
        return jsonify(result)

    @app.route('/shot/trigger', methods=['POST'])  
    def trigger():
        result = handle_trigger()
        return jsonify(result)


OPTION 2: Call this server's API
--------------------------------

If running flask_bridge.py separately, your partner's GUI can
make HTTP requests to it:

    fetch('http://main-pi-ip:5000/api/club/select', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({club_id: 'driver'})
    })

    fetch('http://main-pi-ip:5000/api/trigger', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        console.log('Ball speed:', data.metrics.ball_speed);
    });
"""


if __name__ == "__main__":
    run_standalone()
