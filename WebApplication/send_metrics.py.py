import requests

# Replace with the IP of the computer running your Flask app
url = requests.get("http://10.42.0.1/api/upload_metrics")

# Example metrics
metrics = {
    "player_id": 1,
    "club_speed": 112,
    "ball_speed": 160,
    "launch_angle": 14.5,
    "spin_rate": 3200,
    "carry_distance": 265,
    "total_distance": 280
}

response = requests.post(url, json = metrics)
print(response.json())