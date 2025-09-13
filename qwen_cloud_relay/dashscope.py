#!/usr/bin/env python3
"""
DashScope WebSocket Relay Server
Relays WebSocket connections between frontend and DashScope cloud API
"""

import asyncio
import websockets
import json
import os
import base64
import logging
from typing import Dict, Set
import signal
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DashScope configuration
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', 'sk-your-api-key-here')
DASHSCOPE_WS_URL = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"

# Store active connections
active_connections: Set[websockets.WebSocketServerProtocol] = set()
dashscope_connections: Dict[websockets.WebSocketServerProtocol, websockets.WebSocketClientProtocol] = {}

class DashScopeRelay:
    def __init__(self):
        self.port = int(os.getenv('RELAY_PORT', 8001))  # Different from FastRTC (8000)
        
    async def handle_frontend_connection(self, websocket, path):
        """Handle connection from frontend"""
        logger.info(f"Frontend connected: {websocket.remote_address}")
        active_connections.add(websocket)
        
        try:
            # Connect to DashScope
            dashscope_ws = await websockets.connect(
                DASHSCOPE_WS_URL,
                extra_headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}"}
            )
            dashscope_connections[websocket] = dashscope_ws
            logger.info("Connected to DashScope cloud")
            
            # Start bidirectional message forwarding
            await asyncio.gather(
                self.forward_to_dashscope(websocket, dashscope_ws),
                self.forward_to_frontend(websocket, dashscope_ws)
            )
            
        except Exception as e:
            logger.error(f"Error handling frontend connection: {e}")
        finally:
            # Cleanup
            active_connections.discard(websocket)
            if websocket in dashscope_connections:
                await dashscope_connections[websocket].close()
                del dashscope_connections[websocket]
            logger.info(f"Frontend disconnected: {websocket.remote_address}")
    
    async def forward_to_dashscope(self, frontend_ws, dashscope_ws):
        """Forward messages from frontend to DashScope"""
        try:
            async for message in frontend_ws:
                logger.debug(f"Frontend → DashScope: {message[:100]}...")
                await dashscope_ws.send(message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("Frontend connection closed")
        except Exception as e:
            logger.error(f"Error forwarding to DashScope: {e}")
    
    async def forward_to_frontend(self, frontend_ws, dashscope_ws):
        """Forward messages from DashScope to frontend"""
        try:
            async for message in dashscope_ws:
                logger.debug(f"DashScope → Frontend: {message[:100]}...")
                await frontend_ws.send(message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("DashScope connection closed")
        except Exception as e:
            logger.error(f"Error forwarding to frontend: {e}")
    
    async def start_server(self):
        """Start the relay server"""
        logger.info(f"Starting DashScope relay server on port {self.port}")
        logger.info(f"DashScope API Key: {DASHSCOPE_API_KEY[:10]}...")
        
        # Start WebSocket server
        server = await websockets.serve(
            self.handle_frontend_connection,
            "0.0.0.0",
            self.port,
            ping_interval=20,
            ping_timeout=10
        )
        
        logger.info(f"✅ DashScope relay server running on ws://localhost:{self.port}")
        logger.info("Ready to relay connections to DashScope cloud")
        
        # Keep server running
        await server.wait_closed()

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    logger.info("Shutting down relay server...")
    sys.exit(0)

async def main():
    """Main entry point"""
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check API key
    if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == 'sk-your-api-key-here':
        logger.error("❌ DASHSCOPE_API_KEY not set!")
        logger.error("Please set DASHSCOPE_API_KEY environment variable")
        sys.exit(1)
    
    # Start relay server
    relay = DashScopeRelay()
    await relay.start_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
