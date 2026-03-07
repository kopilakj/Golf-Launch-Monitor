# Golf Launch Monitor - Pi Communication System

Complete two-way communication system between Raspberry Pis for the golf launch monitor.

## Quick Start

### Step 1: Test on Your Laptop First

Before going to the Pis, verify the protocol works:

```bash
cd Pi_Communication/Bringing_it_together
python test_local.py
```

This runs a fake sensor and client on localhost. All tests should pass.

### Step 2: Copy Files to Both Pis

Copy ALL these files to BOTH Pis:
- `config.py`
- `protocol.py`
- `messages.py`
- `sensor_server.py` (runs on Sensor Pi)
- `main_client.py` (runs on Main Pi)
- `flask_bridge.py` (runs on Main Pi, optional)

You can use SCP or a USB drive:

```bash
# From your laptop to Sensor Pi
scp *.py pi@<sensor-pi-ip>:/home/pi/golf_comm/

# From your laptop to Main Pi  
scp *.py pi@<main-pi-ip>:/home/pi/golf_comm/
```

### Step 3: Configure IP Addresses

**On BOTH Pis**, edit `config.py`:

1. Connect both Pis to your phone hotspot
2. On each Pi, run: `hostname -I` to find its IP
3. Edit config.py on BOTH Pis:

```python
# Example IPs (yours will be different!)
SENSOR_PI_IP = "192.168.43.101"   # Sensor Pi's IP
MAIN_PI_IP = "192.168.43.102"     # Main Pi's IP
```

**IMPORTANT:** Both Pis must have the SAME IPs in their config.py files!

### Step 4: Start the Sensor Pi Server

On the **Sensor Pi**, run:

```bash
cd /home/pi/golf_comm
python3 sensor_server.py
```

You should see:
```
============================================================
SENSOR PI SERVER
============================================================
Listening on: 192.168.43.101:5001
Protocol version: 1
============================================================

Waiting for Main Pi to connect...
```

### Step 5: Start the Main Pi Client

On the **Main Pi**, run:

```bash
cd /home/pi/golf_comm
python3 main_client.py
```

You should see it connect and get an interactive prompt:
```
============================================================
MAIN PI CLIENT
============================================================
[Main] Connecting to Sensor Pi at 192.168.43.101:5001...
[Main] Connected!

Commands:
  c <club>  - Send club preset (e.g., 'c driver')
  t         - Trigger a shot
  p         - Ping sensor
  s         - Show status
  q         - Quit

Available clubs: ['driver', '3wood', '5iron', '7iron', ...]
>
```

### Step 6: Test the Protocol

In the Main Pi interactive mode:

```
> p              # Test connection
Sensor Pi is alive!

> c driver       # Select driver
[Main] Sending CLUB_PRESET for driver (version 1)
[Main] Preset confirmed for driver

> t              # Trigger a shot
[Main] Triggering shot:
       shot_id: abc-123-...
       club_id: driver
[Main] ACK_TRIGGER received, waiting for CSV...
[Main] Saved CSV to /home/pi/launch_data/abc-123-....csv

Shot Result:
  ball_speed: 152.3 mph
  launch_angle: 12.5 degrees
  ...
```

## File Structure

```
Bringing_it_together/
  config.py          # IP addresses and settings (EDIT THIS!)
  protocol.py        # Low-level framing (send/recv frames)
  messages.py        # Message type definitions
  sensor_server.py   # Runs on SENSOR PI
  main_client.py     # Runs on MAIN PI
  flask_bridge.py    # Optional Flask integration
  test_local.py      # Localhost testing
  README.md          # This file
```

## Communication Flow

```
GUI                    MAIN PI                  SENSOR PI
 |                        |                          |
 |--CLUB_SELECT---------->|                          |
 |                        |--CLUB_PRESET------------>|
 |                        |<---------ACK_PRESET------|
 |                        |                          |
 |  (user swings)         |                          |
 |                        |--TRIGGER---------------->|
 |                        |<---------ACK_TRIGGER-----|
 |                        |                          |
 |                        |          (sensor computes)
 |                        |                          |
 |                        |<---------CSV_META--------|
 |                        |<---------CSV_CHUNK(s)----|
 |                        |<---------CSV_DONE--------|
 |                        |                          |
 |<---SHOT_RESULT---------|                          |
```

## Integrating with Your Partner's Flask App

### Option 1: Import Functions

In your Flask routes:

```python
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

@app.route('/shot/result', methods=['GET'])
def get_result():
    result = get_shot_result()
    return jsonify(result)
```

### Option 2: Run Standalone Flask Server

```bash
python3 flask_bridge.py
```

This starts a REST API on port 5000. Your GUI can call:
- `POST /api/club/select` with `{"club_id": "driver"}`
- `POST /api/trigger`
- `GET /api/result`

## Troubleshooting

### "SENSOR_PI_IP not configured"
Edit `config.py` and set the correct IP addresses.

### "Connection refused"
1. Make sure `sensor_server.py` is running on Sensor Pi
2. Check that IPs are correct: `hostname -I`
3. Check firewall: `sudo ufw allow 5001`
4. Test connectivity: `ping <sensor-pi-ip>`

### "Invalid authentication token"
Both Pis must have the same `SHARED_TOKEN` in `config.py`.

### Sensor Pi shows "Cannot bind"
1. Check IP matches this Pi: `hostname -I`
2. Check if port is in use: `sudo lsof -i :5001`
3. Kill any old processes: `pkill -f sensor_server.py`

### No response from Sensor Pi
1. Check both Pis are on same network
2. On phone hotspot: make sure both connected
3. Run `ping` between Pis to verify network

## Testing Without Pis

You can test the protocol on any computer:

```bash
python test_local.py
```

This simulates both Pis on localhost.

## Adding Your Real Sensor Code

In `sensor_server.py`, find the `capture_shot()` method in the `SensorState` class.
Replace the dummy data with your actual capture pipeline:

```python
def capture_shot(self, shot_id: str, club_id: str) -> bytes:
    # YOUR REAL CODE HERE:
    # 1. Trigger cameras
    # 2. Capture images  
    # 3. Run ball tracking algorithm
    # 4. Calculate metrics
    # 5. Format as CSV
    
    csv_text = "metric,value,unit\n"
    csv_text += f"ball_speed,{your_speed},mph\n"
    # ... add all metrics
    
    return csv_text.encode("utf-8")
```

## Available Clubs

Defined in `config.py` under `CLUB_PRESETS`:
- driver
- 3wood
- 5iron
- 7iron
- 9iron
- pitching_wedge
- sand_wedge
- putter

Add more as needed!
