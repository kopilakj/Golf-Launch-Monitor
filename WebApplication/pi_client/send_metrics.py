import json
import urllib.request
import os
import requests

url = "http://10.42.0.233:5000/api/upload_metrics"

def get_club():	
	try:
		response = requests.get("http://127.0.0.1:5001/get_club")
		club = response.json()["club"]

		if club is None:
			print("No club selected yet.")
			return None
		return club

	except Exception as e:
		print("Error getting club from pi server:", e)
		return None

def send_metrics():
	club = get_club()

	if club is None:
		print("Cannot send metrics, no club selected.")
		return

	data = {"club_speed": 110,
		"ball_speed": 160,
		"launch_angle": 14.5,
		"spin_rate": 3200,
		"carry_distance": 265,
		"total_distance": 280
	}

	json_data = json.dumps(data).encode("utf-8")
	
	req = urllib.request.Request(
		url,
	 	data=json_data,
	 	headers={"Content-Type": "application/json"}
	)

	try:
		with urllib.request.urlopen(req, timeout=5) as response:
			print("Server response:", response.read().decode())
	except Exception as e:
		print("Error sending metrics:", e)

if __name__ == "__main__":
	send_metrics()

