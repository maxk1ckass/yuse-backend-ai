@echo off
echo Starting YUSE AI Servers...
echo.

echo [1/2] Starting DashScope Relay Server...
start "DashScope Relay" cmd /k "cd qwen_cloud_relay && python dashscope.py"

echo [2/2] Starting FastRTC Server...
start "FastRTC Server" cmd /k "cd qwen_fastrtc && uvicorn main:app --host 0.0.0.0 --port 8000"

echo.
echo ✅ Both servers started!
echo.
echo 📡 DashScope Relay: ws://localhost:8001
echo 🤖 FastRTC Server:  http://localhost:8000
echo.
echo Press any key to close this window...
pause > nul
