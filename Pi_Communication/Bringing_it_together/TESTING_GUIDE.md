# Golf Launch Monitor - Complete Testing Guide

This guide walks you through testing the integrated system step by step.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                              MAIN PI                                 │
│  ┌───────────────┐                    ┌───────────────────────┐     │
│  │   app.py      │◄──POST metrics─────│   sensor_bridge.py    │     │
│  │  (Port 5000)  │                    │     (Port 5001)       │     │
│  │               │                    │                       │     │
│  │  - GUI/Web    │──notify club──────►│  - Talks to Sensor Pi │     │
│  │  - Players    │                    │  - Receives CSV       │     │
│  │  - Metrics    │                    │  - Parses metrics     │     │
│  └───────────────┘                    └───────────┬───────────┘     │
│         ▲                                         │                  │
│         │                                         │ TCP              │
│     Browser                                       │ Port 5002        │
└─────────┼─────────────────────────────────────────┼──────────────────┘
          │                                         │
          │                                         ▼
    ┌─────┴─────┐                         ┌─────────────────┐
    │   Phone   │                         │   SENSOR PI     │
    │  Browser  │                         │                 │
    │           │                         │ sensor_server.py│
    └───────────┘                         │   (Port 5002)   │
                                          └─────────────────┘
```

## Files Overview

### On SENSOR PI:
- `config.py` - IP addresses and settings
- `protocol.py` - Low-level communication
- `messages.py` - Message definitions
- `sensor_server.py` - **RUN THIS** - listens for commands

### On MAIN PI:
- `config.py` - Same file, same IPs
- `protocol.py` - Same file
- `messages.py` - Same file
- `sensor_bridge.py` - **RUN THIS** - connects to Sensor Pi
- `app.py` - **RUN THIS** - web interface (your partner's code)

---

## Step 1: Test on Your Laptop First

Before touching the Pis, verify everything works locally.

### 1.1 Open Terminal in the Bringing_it_together folder

```bash
cd "C:\Users\kopil\OneDrive\Senior Project\Golf-Launch-Monitor\Pi_Communication\Bringing_it_together"
```

### 1.2 Install dependencies (if not already)

```bash
pip install flask flask-cors requests
```

### 1.3 Run the local protocol test

```bash
python test_local.py
```

Expected output:
```
============================================================
LOCAL PROTOCOL TEST
============================================================
[FakeSensor] Server started on 127.0.0.1:5001

Test: Basic Connection
  [PASS] Connect to server
Test: PING/PONG
  [PASS] PING/PONG exchange
...
RESULTS: 9/9 passed
```

If all 9 tests pass, the protocol is working!

---

## Step 2: Copy Files to the Pis

### 2.1 Files to copy to SENSOR PI:

```
config.py
protocol.py
messages.py
sensor_server.py
```

Copy to: `/home/jkopila1/project/golf_comm/` (or wherever you prefer)

### 2.2 Files to copy to MAIN PI:

Option A - Copy to webapplication/pi_client/ (recommended):
```
config.py
protocol.py
messages.py
sensor_bridge.py
```

These are already there! I copied them for you.

Option B - Use the Bringing_it_together folder:
```
Copy entire Bringing_it_together folder to Main Pi
```

### 2.3 Using SCP from your laptop:

```bash
# To Sensor Pi
scp config.py protocol.py messages.py sensor_server.py pi@SENSOR_IP:/home/jkopila1/project/golf_comm/

# To Main Pi (if needed)
scp config.py protocol.py messages.py sensor_bridge.py pi@MAIN_IP:/path/to/webapplication/pi_client/
```

---

## Step 3: Configure IP Addresses

### 3.1 Connect both Pis to your phone hotspot

### 3.2 Find IP addresses

On each Pi:
```bash
hostname -I
```

Example output:
- Sensor Pi: `192.168.43.101`
- Main Pi: `192.168.43.102`

### 3.3 Edit config.py on BOTH Pis

```python
# In config.py
SENSOR_PI_IP = "192.168.43.101"   # <-- Sensor Pi's actual IP
MAIN_PI_IP = "192.168.43.102"     # <-- Main Pi's actual IP
```

**CRITICAL:** Both Pis must have identical IP values!

---

## Step 4: Start the System (Correct Order!)

### 4.1 FIRST - Start Sensor Pi Server

SSH to Sensor Pi and run:
```bash
cd /home/jkopila1/project/golf_comm
python3 sensor_server.py
```

You should see:
```
============================================================
SENSOR PI SERVER
============================================================
Listening on: 192.168.43.101:5002
Protocol version: 1
============================================================

Waiting for Main Pi to connect...
```

**Leave this running!**

### 4.2 SECOND - Start Sensor Bridge on Main Pi

SSH to Main Pi (new terminal) and run:
```bash
cd /path/to/webapplication/pi_client
python3 sensor_bridge.py
```

You should see:
```
============================================================
SENSOR BRIDGE - Main Pi Communication Service
============================================================

Bridge API:     http://0.0.0.0:5001
Sensor Pi:      192.168.43.101:5002
Webapp metrics: http://127.0.0.1:5000/api/upload_metrics

[Bridge] Connecting to Sensor Pi at 192.168.43.101:5002...
[Bridge] Connected to Sensor Pi!

Starting bridge server...
```

**Leave this running!**

### 4.3 THIRD - Start Web App on Main Pi

SSH to Main Pi (another terminal) and run:
```bash
cd /path/to/webapplication
python3 app.py
```

You should see:
```
 * Running on http://0.0.0.0:5000
```

**Leave this running!**

---

## Step 5: Test the Full Flow

### 5.1 Open the GUI in your phone browser

Go to: `http://MAIN_PI_IP:5000`

Example: `http://192.168.43.102:5000`

### 5.2 Select a Player

Click any player name.

### 5.3 Select a Club

Click any club (e.g., "Driver").

**Watch the terminal windows!**

You should see:
- **app.py**: `[App] Bridge notified of club Driver: {'success': True, ...}`
- **sensor_bridge.py**: `[Bridge] Club selected: 'Driver' -> preset 'driver'`
- **sensor_server.py**: `[Sensor] Received CLUB_PRESET for driver`

### 5.4 Trigger a Shot

On the metrics page, click the **"TRIGGER SHOT"** button.

**Watch the terminals!**

You should see:
- **sensor_bridge.py**: `[Bridge] Triggering shot ... with driver`
- **sensor_server.py**: `[Sensor] Received TRIGGER`, `[Sensor] Sent CSV_DONE`
- **sensor_bridge.py**: `[Bridge] CSV received and verified!`, `[Bridge] Metrics posted to webapp`
- **app.py**: `Received metrics: {'ball_speed': 155.5, ...}`

### 5.5 See the Metrics

The page should refresh and show the shot metrics!

---

## Step 6: Test API Endpoints Directly (Optional)

You can also test the bridge API directly using curl or your browser:

### Check bridge status:
```bash
curl http://MAIN_PI_IP:5001/status
```

### Set club manually:
```bash
curl -X POST http://MAIN_PI_IP:5001/set_club \
  -H "Content-Type: application/json" \
  -d '{"club": "7 Iron"}'
```

### Trigger shot manually:
```bash
curl -X POST http://MAIN_PI_IP:5001/trigger
```

### Ping sensor:
```bash
curl http://MAIN_PI_IP:5001/ping
```

---

## Troubleshooting

### "Cannot connect to Sensor Pi"

1. Check Sensor Pi server is running: `python3 sensor_server.py`
2. Verify IPs in config.py match: `hostname -I`
3. Test network: `ping SENSOR_PI_IP` from Main Pi
4. Check firewall: `sudo ufw allow 5002`

### "Bridge not running"

1. Check sensor_bridge.py is running
2. Check for Python errors in the terminal
3. Make sure port 5001 is free: `sudo lsof -i :5001`

### "No metrics received"

1. Check app.py is running on port 5000
2. Verify the WEBAPP_METRICS_URL in config.py is correct
3. Check app.py terminal for errors

### "Club preset failed"

1. Check the club name mapping in config.py
2. Verify Sensor Pi received the message (check its terminal)
3. Check for token mismatch (SHARED_TOKEN must match)

### Port already in use

Kill existing processes:
```bash
# Find what's using the port
sudo lsof -i :5001
sudo lsof -i :5002

# Kill by PID
kill -9 <PID>
```

---

## Expected Terminal Output (Success Case)

### Sensor Pi (sensor_server.py):
```
============================================================
SENSOR PI SERVER
============================================================
Listening on: 192.168.43.101:5002

[Sensor] Client connected: ('192.168.43.102', 54321)
[Sensor] Received: CLUB_PRESET
[Sensor] Applying preset for driver:
         Version: 1
         Settings: {'preset_version': 1, 'exposure': 100, ...}
[Sensor] Sent ACK_PRESET
[Sensor] Received: TRIGGER
[Sensor] Capturing shot abc-123... with driver
[Sensor] Sent ACK_TRIGGER
[Sensor] Sending CSV: 156 bytes, 1 chunks
[Sensor] Sent CSV_META
[Sensor] Sent chunk 1/1
[Sensor] Sent CSV_DONE
```

### Main Pi (sensor_bridge.py):
```
============================================================
SENSOR BRIDGE - Main Pi Communication Service
============================================================

[Bridge] Connected to Sensor Pi!
[Bridge] /set_club called with: Driver
[Bridge] Club selected: 'Driver' -> preset 'driver'
[Bridge] Preset confirmed for driver
[Bridge] /trigger called
[Bridge] Triggering shot abc-123... with driver
[Bridge] ACK_TRIGGER received, waiting for CSV...
[Bridge] Expecting 156 bytes in 1 chunks
[Bridge] Chunk 1/1 received
[Bridge] CSV received and verified!
[Bridge] CSV saved to /home/pi/launch_data/abc-123....csv
[Bridge] Metrics posted to webapp successfully
[Bridge] Shot complete! Metrics sent to webapp: True
```

### Main Pi (app.py):
```
[App] Bridge notified of club Driver: {'success': True, 'club': 'Driver', 'preset': 'driver', 'version': 1}
Received metrics: {'ball_speed': 155.5, 'launch_angle': 12.3, 'carry_distance': 250}
```

---

## Summary of Running Commands

| Device | Terminal | Command |
|--------|----------|---------|
| Sensor Pi | 1 | `python3 sensor_server.py` |
| Main Pi | 1 | `python3 sensor_bridge.py` |
| Main Pi | 2 | `python3 app.py` |
| Phone | Browser | `http://MAIN_PI_IP:5000` |

**Start order: Sensor Pi first, then Bridge, then App.**

Good luck on Monday!
