#!/usr/bin/env python3
"""
DashScope WebSocket Relay Server
Relays WebSocket connections between frontend and DashScope cloud API using official SDK
"""

import asyncio
import http
from websockets.http11 import Response  # <-- key import
import websockets
import json
import os
import base64
import logging
from typing import Dict, Set, Any
import signal
import sys
from dotenv import load_dotenv
import dashscope
from dashscope.audio.qwen_omni import *

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DashScope configuration
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', 'sk-your-api-key-here')
DASHSCOPE_BACKEND_URL = os.getenv('DASHSCOPE_BACKEND_URL', 'dashscope-intl.aliyuncs.com')
dashscope.api_key = DASHSCOPE_API_KEY

# Build WebSocket URL from backend URL
DASHSCOPE_WS_URL = f"wss://{DASHSCOPE_BACKEND_URL}/api-ws/v1/realtime"

# Debug configuration
logger.info(f"API Key loaded: {DASHSCOPE_API_KEY[:10]}...{DASHSCOPE_API_KEY[-4:] if len(DASHSCOPE_API_KEY) > 14 else '...'}")
logger.info(f"API Key length: {len(DASHSCOPE_API_KEY)}")
logger.info(f"Backend URL: {DASHSCOPE_BACKEND_URL}")
logger.info(f"WebSocket URL: {DASHSCOPE_WS_URL}")

# Store active connections
active_connections: Set[Any] = set()
dashscope_conversations: Dict[Any, OmniRealtimeConversation] = {}

class DashScopeRelay:
    def __init__(self):
        self.port = int(os.getenv('RELAY_PORT', 8001))  # Different from FastRTC (8000)
        self.health_probe_health_message = "ok"  # whatever class data you want
        
    async def handle_frontend_connection(self, websocket, path=None):
        """Handle connection from frontend"""
        logger.info(f"Frontend connected: {websocket.remote_address}")
        active_connections.add(websocket)
        
        try:
            # Create DashScope conversation with callback
            callback = DashScopeCallback(websocket, asyncio.get_event_loop())
            conversation = OmniRealtimeConversation(
                model='qwen-omni-turbo-realtime-latest',
                callback=callback,
                url=DASHSCOPE_WS_URL
            )
            
            logger.info("Attempting to connect to DashScope...")
            # Connect to DashScope
            conversation.connect()
            dashscope_conversations[websocket] = conversation
            logger.info("Successfully connected to DashScope")
            
            # Configure session with restaurant ordering context
            conversation.update_session(
                output_modalities=[MultiModality.AUDIO, MultiModality.TEXT],
                voice='Chelsie',
                input_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
                output_audio_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                enable_input_audio_transcription=True,
                input_audio_transcription_model='gummy-realtime-v1',
                enable_turn_detection=True,
                turn_detection_type='server_vad',
                instructions="""You are Yuni, a friendly English instructor helping students practice restaurant ordering scenarios. 

CONTEXT: You are teaching English through a restaurant ordering roleplay. The student is learning how to order food, ask questions about the menu, and interact with restaurant staff.

THE SCRIPT:
Waiter: Good evening! Welcome to Golden Dragon Restaurant. How many people are dining today?
Customer: Just one, thank you.
Waiter: Of course. Would you like to sit by the window or closer to the bar?
Customer: By the window, please.
Waiter: Here you are. Can I get you something to drink while you look at the menu?
Customer: Yes, I’ll have a glass of water and a lemonade, please.
Waiter: Certainly. Our soup of the day is pumpkin soup, and we also have a chef’s special stir-fry chicken. Would you like me to go over the menu highlights?
Customer: Yes, that would be helpful.
Waiter: For starters, we recommend dumplings, spring rolls, or the mixed platter. For mains, popular dishes are the grilled salmon, beef noodles, and the vegetarian fried rice.
Customer: I think I’ll start with the dumplings.
Waiter: Excellent choice. And for your main course?
Customer: The grilled salmon with lemon butter, please.
Waiter: Very good. Would you like any side dishes, such as steamed vegetables or extra rice?
Customer: Steamed vegetables, please.
Waiter: Noted. Do you have any dietary restrictions or allergies I should be aware of?
Customer: No, I’m fine with everything.
Waiter: Perfect. I’ll place your order.
Customer: Thank you.

THE WAY TO INTERACT WITH THE STUDENT:
- Everytime, you speek a sentence, and wait for the student to speak the next turn
- Finish the scenario script with the student turn by turn
- Ask if the student want to switch roles after the scenario is finished
- For each turn, you sentence should be short and not too complicated, the student is a beginner
- We allow some open ended conversation, so if the student didn't stick to the scenario script, you can adjust your response

START BY::
- greeting the student warmly
- let the student know we are going to recap the restaurant ordering task
- ask the student if they want to play the waiter role or the customer role, then start the roleplay.."""
            )
            
            logger.info("Connected to DashScope cloud")
            
            # Handle messages from frontend
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get('type') == 'session.update':
                        # Frontend is sending session config - we already configured it
                        logger.info("Session configuration received from frontend")
                    elif data.get('type') == 'input_audio_buffer.append':
                        # Forward audio data to DashScope
                        audio_b64 = data.get('audio')
                        if audio_b64:
                            conversation.append_audio(audio_b64)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from frontend: {message}")
                except Exception as e:
                    logger.error(f"Error processing frontend message: {e}")
            
        except Exception as e:
            logger.error(f"Error handling frontend connection: {e}")
        finally:
            # Cleanup
            active_connections.discard(websocket)
            if websocket in dashscope_conversations:
                dashscope_conversations[websocket].close()
                del dashscope_conversations[websocket]
            logger.info(f"Frontend disconnected: {websocket.remote_address}")
    
    async def start_server(self):
        """Start the relay server"""
        logger.info(f"Starting DashScope relay server on port {self.port}")
        logger.info(f"DashScope API Key: {DASHSCOPE_API_KEY[:10]}...")

        async def process_request(protocol, request):
            try:
                path = getattr(request, "path", "/")
                logger.info(f"HTTP probe on {path}")

                if path == "/health":
                    body = getattr(self, "health_message", "ok").encode("utf-8")
                    return Response(
                        200,
                        headers=[
                            (b"content-type", b"text/plain"),
                            (b"cache-control", b"no-store"),
                        ],
                        body=body,
                    )

                if path == "/":
                    return Response(
                        200,
                        headers=[
                            (b"content-type", b"text/plain"),
                            (b"cache-control", b"no-store"),
                        ],
                        body=b"WS backend\n",
                    )

                # Non-WS HTTP on other paths -> hint upgrade
                return Response(
                    426,  # UPGRADE_REQUIRED
                    headers=[
                        (b"content-type", b"text/plain"),
                        (b"connection", b"close"),
                    ],
                    body=b"WebSocket endpoint. Use WS/WSS.\n",
                )
            except Exception as e:
                logger.exception(f"process_request error: {e}")
                return Response(
                    500,
                    headers=[(b"content-type", b"text/plain")],
                    body=b"internal error\n",
                )
        
        # Start WebSocket server
        server = await websockets.serve(
            self.handle_frontend_connection,
            "0.0.0.0",
            self.port,
            ping_interval=20,
            ping_timeout=10,
            process_request=process_request
        )
        
        logger.info(f"✅ DashScope relay server running on ws://localhost:{self.port}")
        logger.info("Ready to relay connections to DashScope cloud")
        
        # Keep server running
        await server.wait_closed()


class DashScopeCallback(OmniRealtimeCallback):
    def __init__(self, frontend_websocket, event_loop):
        self.frontend_ws = frontend_websocket
        self.event_loop = event_loop
        
    def on_open(self) -> None:
        logger.info("DashScope connection opened")
        
    def on_close(self, close_status_code, close_msg) -> None:
        logger.info(f"DashScope connection closed: {close_status_code}, {close_msg}")
        
    def on_error(self, error) -> None:
        logger.error(f"DashScope connection error: {error}")
        
    def on_event(self, response: dict) -> None:
        try:
            # Forward DashScope events to frontend using the event loop
            if self.event_loop and not self.event_loop.is_closed():
                asyncio.run_coroutine_threadsafe(
                    self.send_to_frontend(response), 
                    self.event_loop
                )
        except Exception as e:
            logger.error(f"Error in DashScope callback: {e}")
    
    async def send_to_frontend(self, response: dict):
        """Send DashScope response to frontend"""
        try:
            if self.frontend_ws:
                await self.frontend_ws.send(json.dumps(response))
        except Exception as e:
            logger.error(f"Error sending to frontend: {e}")

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
        logger.error("Please set DASHSCOPE_API_KEY in your .env file")
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
