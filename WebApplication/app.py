from flask import Flask, redirect, url_for, render_template, session, request, flash, jsonify
import requests
import sqlite3
import os
import json
import requests

app = Flask(__name__)
app.secret_key = "dev"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok = True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'json'}
latest_metrics = {}

# =============================================================================
# SENSOR BRIDGE CONFIGURATION
# =============================================================================
# The sensor_bridge.py runs on port 5001 and handles communication with Sensor Pi
BRIDGE_URL = "http://127.0.0.1:5001"


def notify_bridge_club_selected(club: str) -> bool:
    """
    Notify the sensor bridge when a club is selected.
    This sends the CLUB_PRESET to Sensor Pi.
    
    Returns True if successful, False otherwise.
    """
    try:
        response = requests.post(
            f"{BRIDGE_URL}/set_club",
            json={"club": club},
            timeout=5
        )
        if response.status_code == 200:
            result = response.json()
            print(f"[App] Bridge notified of club {club}: {result}")
            return result.get("success", False)
        else:
            print(f"[App] Bridge returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"[App] Could not connect to bridge at {BRIDGE_URL}")
        print("[App] Make sure sensor_bridge.py is running!")
        return False
    except Exception as e:
        print(f"[App] Error notifying bridge: {e}")
        return False

## Get Database Connection
def db_connection():
    connection = sqlite3.connect("player.db")
    connection.row_factory = sqlite3.Row
    return connection

def initialize_db():
    ## Create Connection to SQLite Database
    connection = sqlite3.connect("player.db")

    ## Create Cursor Object
    cur = connection.cursor()

    ## Execute SQL Statement to Create Table
    cur.execute(""" CREATE TABLE IF NOT EXISTS player 
                (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL)""")
    
    ## Check if Table is Empty
    cur.execute("SELECT COUNT(*) FROM player")
    count = cur.fetchone()[0]

    ## Add Default Players
    if count == 0:
        default_players = [
            "Player 1",
            "Player 2",
            "Player 3",
            "Player 4"
        ]
        for name in default_players:
            cur.execute("INSERT INTO player (name) VALUES (?)",
                        (name,))
        
    connection.commit()
    connection.close()

## Initialize Database
initialize_db()

## Define Routes
@app.route("/")
def start():
    return render_template("start_page.html")

@app.route("/whos_playing")
def whos_playing():
    connection = sqlite3.connect("player.db")
    connection.row_factory = sqlite3.Row
    cur = connection.cursor()

    players = cur.execute("SELECT id, name FROM player ORDER BY id"
                          ).fetchall()
    
    connection.close()
    return render_template("index.html", players=players)

@app.route("/select_player/<int:player_id>")
def select_player(player_id):
    session["player"] = player_id
    return redirect(url_for("select_club"))

@app.route('/selectclub')
def select_club():
    player_id = session.get("player")
    player_name = None

    if player_id:
        connection = db_connection()
        player = connection.execute("SELECT name FROM player WHERE id = ?",
                              (player_id,)).fetchone()
        connection.close()

        if player:
            player_name = player["name"]

    return render_template("select_club.html", player_name=player_name)


@app.route("/select_gameplay_club/<path:club>")
def select_gameplay_club(club):
    session["club"] = club
    
    # Notify the sensor bridge about the club selection
    # This will send CLUB_PRESET to Sensor Pi
    bridge_success = notify_bridge_club_selected(club)
    if not bridge_success:
        print(f"[App] Warning: Could not notify bridge about club {club}")
        # Continue anyway - user can still see metrics page

    requests.post(
        "http://172.20.10.2:5001/set_club",
        json = {"club": club}
    )

    print("Club Selection Saved: ", club)

    return redirect(url_for("display_metrics"))

@app.route("/back_to_players")
def back_to_players():
    session.pop("club", None)
    return redirect(url_for("whos_playing"))

@app.route("/add_player", methods=["GET", "POST"])
def add_player():
    connection = db_connection()
    
    if request.method == "POST":
        name = request.form["name"]
        connection.execute("INSERT INTO player (name) VALUES (?)"
                           , (name,))
        
        connection.commit()
        connection.close()
        return redirect(url_for("whos_playing"))
    
    connection.close()
    return render_template("add_player.html")

## Edits Player Name
@app.route("/edit_player/<int:player_id>", methods=["GET", "POST"])
def edit_player(player_id):
    connection = db_connection()
    
    if request.method == "POST":
        name = request.form["name"]
        connection.execute("UPDATE player SET name = ? WHERE id = ?",
                           (name, player_id))
        
        connection.commit()
        connection.close()
        return redirect(url_for("whos_playing"))
    
    player=connection.execute("SELECT * FROM player WHERE id = ?",
                              (player_id,)).fetchone()
    
    connection.close()
    return render_template("edit_player.html", player=player)

## Chooses Which Player To Edit
@app.route("/edit_users")
def edit_users():
    connection = db_connection()
    players = connection.execute("SELECT * FROM player ORDER BY " \
    "id").fetchall()

    connection.close()
    return render_template("edit_users.html", players=players)

## Deletes Player Name
@app.route("/delete_player/<int:player_id>", methods=["GET", "POST"])
def delete_player(player_id):
    connection = db_connection()
    
    if request.method == "POST":
        connection.execute("DELETE FROM player WHERE id = ?",
                           (player_id,))
        
        connection.commit()
        connection.close()

        if session.get("player") == player_id:
            session.pop("player")

        return redirect(url_for("whos_playing"))
    
    player=connection.execute("SELECT * FROM player WHERE id = ?",
                              (player_id,)).fetchone()
    
    connection.close()
    return render_template("delete_player.html", player=player)

## Chooses Which Player To Delete
@app.route("/delete_users")
def delete_users():
    connection = db_connection()
    players = connection.execute("SELECT * FROM player ORDER BY " \
    "id").fetchall()

    connection.close()
    return render_template("delete_users.html", players=players)

## Upload Metrics (called by sensor_bridge.py when shot data arrives)
@app.route("/api/upload_metrics", methods=["POST"])
def api_upload_metrics():
    data = request.get_json()

    print("Received metrics:", data)

    with open("latest_metrics.json", "w") as f:
        json.dump(data, f)

    return jsonify({"status": "success"})


## API endpoint to trigger a shot (for testing or future auto-trigger)
@app.route("/api/trigger_shot", methods=["POST"])
def api_trigger_shot():
    """
    Trigger a shot capture via the sensor bridge.
    Call this when the sensor detects a swing.
    """
    try:
        response = requests.post(
            f"{BRIDGE_URL}/trigger",
            timeout=30  # Shot capture can take time
        )
        return jsonify(response.json())
    except requests.exceptions.ConnectionError:
        return jsonify({
            "success": False, 
            "error": "Bridge not running. Start sensor_bridge.py!"
        }), 503
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


## API endpoint to check bridge/sensor status
@app.route("/api/bridge_status", methods=["GET"])
def api_bridge_status():
    """Check if the sensor bridge and Sensor Pi are connected."""
    try:
        response = requests.get(f"{BRIDGE_URL}/status", timeout=5)
        return jsonify(response.json())
    except requests.exceptions.ConnectionError:
        return jsonify({
            "bridge_running": False,
            "error": "Bridge not running. Start sensor_bridge.py!"
        }), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


## API endpoint to enable motion detection
@app.route("/api/motion/enable", methods=["POST"])
def api_motion_enable():
    """Enable automatic swing detection via motion sensor."""
    try:
        response = requests.post(f"{BRIDGE_URL}/motion/enable", timeout=5)
        return jsonify(response.json())
    except requests.exceptions.ConnectionError:
        return jsonify({
            "success": False,
            "error": "Bridge not running. Start sensor_bridge.py!"
        }), 503
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


## API endpoint to disable motion detection
@app.route("/api/motion/disable", methods=["POST"])
def api_motion_disable():
    """Disable automatic swing detection."""
    try:
        response = requests.post(f"{BRIDGE_URL}/motion/disable", timeout=5)
        return jsonify(response.json())
    except requests.exceptions.ConnectionError:
        return jsonify({
            "success": False,
            "error": "Bridge not running. Start sensor_bridge.py!"
        }), 503
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


## API endpoint to get motion detection status
@app.route("/api/motion/status", methods=["GET"])
def api_motion_status():
    """Get current motion detection status."""
    try:
        response = requests.get(f"{BRIDGE_URL}/motion/status", timeout=5)
        return jsonify(response.json())
    except requests.exceptions.ConnectionError:
        return jsonify({
            "motion_enabled": False,
            "camera_available": False,
            "error": "Bridge not running"
        }), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

## Display Metrics
@app.route("/display_metrics")
def display_metrics():

    player_id = session.get("player")
    club = session.get("club")

    if not player_id or not club:
        return redirect(url_for("whos_playing"))

    connection = db_connection()
    player = connection.execute(
        "SELECT name FROM player WHERE id = ?",
        (player_id,)
    ).fetchone()
    connection.close()

    metrics = None
    if os.path.exists("latest_metrics.json"):
        try:
            with open("latest_metrics.json", "r") as f:
                metrics = json.load(f)
        except json.JSONDecodeError:
            metrics = None

    if not metrics:
        return "<h1>No metrics received yet. Waiting for player to swing...</h1>"

    with open("latest_metrics.json", "w") as f:
        f.write("")

    return render_template(
        "metrics.html",
        player_name=player["name"],
        club=club,
        metrics=metrics
    )

## Run Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)