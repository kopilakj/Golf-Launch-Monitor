from flask import Flask, redirect, url_for, render_template, session, request, flash, jsonify
import sqlite3
import os
import json

app = Flask(__name__)
app.secret_key = "dev"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok = True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'json'}
latest_metrics = {}

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

        return redirect(url_for("delete_users"))
    
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

## Upload Metrics
@app.route("/api/upload_metrics", methods=["POST"])
def api_upload_metrics():
    data = request.get_json()

    print("Received metrics:", data)

    with open("latest_metrics.json", "w") as f:
        json.dump(data, f)

    return jsonify({"status": "success"})

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

    if not os.path.exists("latest_metrics.json"):
        return "<h1>No metrics received yet. Waiting for player to swing...</h1>"

    with open("latest_metrics.json", "r") as f:
        metrics = json.load(f)

    return render_template(
        "metrics.html",
        player_name=player["name"],
        club=club,
        metrics=metrics
    )

## Run Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)