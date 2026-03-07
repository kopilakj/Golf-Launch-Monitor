import json
import urllib.request
import os
import requests
import csv

url = "http://127.0.0.1:5000/api/upload_metrics"

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


def load_metrics_from_csv(csv_path):
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            data = {
                "ball_speed": float(row["ball_speed_pxps"]),
                "launch_angle": float(row["launch_angle_deg"]),
                "apex_height": float(row["apex_height_px"]),
                "carry_distance": float(row["carry_px"]),
                "vx": float(row["vx_pxps"]),
                "vy": float(row["vy_pxps"])
            }

            return data


def send_metrics():
    club = get_club()

    if club is None:
        print("Cannot send metrics, no club selected.")
        return

    data = load_metrics_from_csv("metrics.csv")

    if data is None:
        print("No metrics found.")
        return

    data["club"] = club

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