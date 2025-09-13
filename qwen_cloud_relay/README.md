# DashScope WebSocket Relay Server

This server acts as a proxy between the frontend and DashScope cloud API, enabling WebSocket connections that would otherwise be blocked by CORS policies.

## Architecture

```
Frontend (Browser) ←→ WebSocket ←→ Relay Server (Python) ←→ WebSocket ←→ DashScope Cloud
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create .env file:**
   ```bash
   # Create .env file in qwen_cloud_relay directory
   echo "DASHSCOPE_API_KEY=sk-your-actual-api-key-here" > .env
   echo "RELAY_PORT=8001" >> .env
   ```

3. **Run the relay server:**
   ```bash
   python dashscope.py
   ```

## Usage

- **Relay Server**: Runs on `ws://localhost:8001`
- **Frontend connects to**: `ws://localhost:8001`
- **Relay connects to**: `wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime`

## Features

- ✅ Bidirectional message forwarding
- ✅ Connection management
- ✅ Error handling and logging
- ✅ Graceful shutdown
- ✅ API key validation

## Ports

- **FastRTC Server**: `8000` (for local model)
- **DashScope Relay**: `8001` (for cloud model)
