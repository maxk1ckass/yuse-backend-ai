#!/bin/bash

echo "Starting YUSE AI Servers..."
echo

echo "[1/2] Starting DashScope Relay Server..."
cd qwen_cloud_relay
python dashscope.py &
RELAY_PID=$!

echo "[2/2] Starting FastRTC Server..."
cd ../qwen_fastrtc
uvicorn main:app --host 0.0.0.0 --port 8000 &
FASTRTC_PID=$!

echo
echo "✅ Both servers started!"
echo
echo "📡 DashScope Relay: ws://localhost:8001 (PID: $RELAY_PID)"
echo "🤖 FastRTC Server:  http://localhost:8000 (PID: $FASTRTC_PID)"
echo
echo "Press Ctrl+C to stop both servers..."

# Function to cleanup on exit
cleanup() {
    echo
    echo "Stopping servers..."
    kill $RELAY_PID 2>/dev/null
    kill $FASTRTC_PID 2>/dev/null
    echo "Servers stopped."
    exit 0
}

# Set up signal handler
trap cleanup SIGINT SIGTERM

# Wait for user interrupt
wait
