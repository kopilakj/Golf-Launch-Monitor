from flask import Flask, request, jsonify
import json

app = Flask(__name__)

current_club = None

@app.route("/set_club", methods=["POST"])
def set_club():
	global current_club

	data = request.get_json()
	current_club = data.get("club")

	print("Club received: ", current_club)

	with open("current_club.json", "w") as f:
		json.dump({"club": current_club}, f)

	return jsonify({"status": "club saved"})

@app.route("/get_club", methods=["GET"])
def get_club():
	return jsonify({"club": current_club})

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5001)
