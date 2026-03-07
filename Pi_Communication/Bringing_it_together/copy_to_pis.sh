#!/bin/bash
# Copy files to Raspberry Pis
# Usage: ./copy_to_pis.sh <sensor-ip> <main-ip>
#
# Example: ./copy_to_pis.sh 192.168.43.101 192.168.43.102

if [ $# -lt 2 ]; then
    echo "Usage: ./copy_to_pis.sh <sensor-pi-ip> <main-pi-ip>"
    echo "Example: ./copy_to_pis.sh 192.168.43.101 192.168.43.102"
    exit 1
fi

SENSOR_IP=$1
MAIN_IP=$2
DEST_DIR="/home/pi/golf_comm"

FILES="config.py protocol.py messages.py sensor_server.py main_client.py flask_bridge.py"

echo "============================================"
echo "Copying files to Raspberry Pis"
echo "============================================"
echo "Sensor Pi: $SENSOR_IP"
echo "Main Pi: $MAIN_IP"
echo "Destination: $DEST_DIR"
echo ""

# Create directory on both Pis
echo "Creating directories..."
ssh pi@$SENSOR_IP "mkdir -p $DEST_DIR" 2>/dev/null
ssh pi@$MAIN_IP "mkdir -p $DEST_DIR" 2>/dev/null

# Copy to Sensor Pi
echo ""
echo "Copying to Sensor Pi ($SENSOR_IP)..."
for f in $FILES; do
    scp $f pi@$SENSOR_IP:$DEST_DIR/
done

# Copy to Main Pi
echo ""
echo "Copying to Main Pi ($MAIN_IP)..."
for f in $FILES; do
    scp $f pi@$MAIN_IP:$DEST_DIR/
done

echo ""
echo "============================================"
echo "Done!"
echo "============================================"
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. SSH to BOTH Pis and edit config.py:"
echo "   SENSOR_PI_IP = \"$SENSOR_IP\""
echo "   MAIN_PI_IP = \"$MAIN_IP\""
echo ""
echo "2. On SENSOR Pi, run:"
echo "   cd $DEST_DIR && python3 sensor_server.py"
echo ""
echo "3. On MAIN Pi, run:"
echo "   cd $DEST_DIR && python3 main_client.py"
echo ""
