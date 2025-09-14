#!/usr/bin/env python3
"""
DashScope WebSocket Relay Server
- Accepts a browser WebSocket (no headers) on ws://<host>:8001
- Opens a DashScope Realtime connection with Authorization header
- Sets persistent "instructions" (system prompt) in session.update
- Enables server VAD + create_response so the model replies after user pauses
- Relays events bidirectionally

ENV (.env):
  DASHSCOPE_API_KEY=sk-...
  DASHSCOPE_BACKEND_URL=dashscope-intl.aliyuncs.com   # or dashscope.aliyuncs.com (CN)
  RELAY_PORT=8001
  DASHSCOPE_MODEL=qwen-omni-turbo-realtime-latest
"""

import asyncio
import json
import logging
import os
import signal
import sys
from typing import Any, Dict, Set

import websockets
from websockets.server import WebSocketServerProtocol
from dotenv import load_dotenv

import dashscope
from dashscope.audio.qwen_omni import (
    OmniRealtimeConversation,
    OmniRealtimeCallback,
    MultiModality,
    AudioFormat,
)

# -----------------------------------------------------------------------------
# Env & logging
# -----------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dashscope-relay")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BACKEND_URL = os.getenv("DASHSCOPE_BACKEND_URL", "dashscope-intl.aliyuncs.com")
DASHSCOPE_WS_URL = f"wss://{DASHSCOPE_BACKEND_URL}/api-ws/v1/realtime"
DASHSCOPE_MODEL = os.getenv("DASHSCOPE_MODEL", "qwen-omni-turbo-realtime-latest")
RELAY_PORT = int(os.getenv("RELAY_PORT", "8001"))

if not DASHSCOPE_API_KEY:
    logger.error("DASHSCOPE_API_KEY is not set.")
    sys.exit(1)

dashscope.api_key = DASHSCOPE_API_KEY

logger.info("Backend: %s", DASHSCOPE_BACKEND_URL)
logger.info("Realtime WS URL: %s", DASHSCOPE_WS_URL)
logger.info("Model: %s", DASHSCOPE_MODEL)

# -----------------------------------------------------------------------------
# Global connection tracking (optional, handy for cleanup/metrics)
# -----------------------------------------------------------------------------
active_connections: Set[WebSocketServerProtocol] = set()
dashscope_conversations: Dict[WebSocketServerProtocol, OmniRealtimeConversation] = {}

# -----------------------------------------------------------------------------
# Fixed system context for the session
# -----------------------------------------------------------------------------
INSTRUCTIONS = """You are Yuni, a friendly English instructor helping students practice restaurant ordering scenarios.

CONTEXT: You are teaching English through a restaurant ordering roleplay. The student is learning how to order food, ask questions about the menu, and interact with restaurant staff.

THE WAY TO INTERACT WITH THE STUDENT:
- Ask if the student wants to play the waiter/waitress role or the customer role
- Once the student chooses, you start the roleplay
- Every time you speak, keep it to 1–2 short sentences and then wait for the student
- Finish the scenario turn by turn
- After finishing, ask if they'd like to switch roles
- Keep vocabulary simple (A2–B1), be encouraging, add tiny inline corrections in [brackets] if helpful
"""

# -----------------------------------------------------------------------------
# DashScope callback → forwards server events to the browser
# -----------------------------------------------------------------------------
class RelayCallback(OmniRealtimeCallback):
    def __init__(self, client_ws: WebSocketServerProtocol, loop: asyncio.AbstractEventLoop):
        self.client_ws = client_ws
        self.loop = loop

    def on_open(self) -> None:
        logger.info("[DashScope] connection opened")

    def on_close(self, code, msg) -> None:
        logger.info("[DashScope] connection closed: %s %s", code, msg)

    def on_error(self, error) -> None:
        logger.error("[DashScope] error: %s", error)

    def on_event(self, response: dict) -> None:
        # Forward every event to the browser client
        try:
            if self.loop and not self.loop.is_closed():
                asyncio.run_coroutine_threadsafe(
                    self._send_to_client(response), self.loop
                )
        except Exception as e:
            logger.error("Callback forward error: %s", e)

    async def _send_to_client(self, obj: dict):
        if self.client_ws.closed:
            return
        try:
            await self.client_ws.send(json.dumps(obj))
        except Exception as e:
            logger.error("send to client failed: %s", e)

# -----------------------------------------------------------------------------
# Per-client handler
# -----------------------------------------------------------------------------
async def handle_client(client_ws: WebSocketServerProtocol, path: str):
    """Bridge a single browser WS to a single DashScope Realtime conversation."""
    active_connections.add(client_ws)
    loop = asyncio.get_event_loop()
    logger.info("Client connected: %s %s", client_ws.remote_address, path)

    # Create DashScope conversation
    conversation = None
    try:
        callback = RelayCallback(client_ws, loop)
        conversation = OmniRealtimeConversation(
            model=DASHSCOPE_MODEL,
            callback=callback,
            url=DASHSCOPE_WS_URL,  # picks region from env
        )
        logger.info("Connecting to DashScope...")
        conversation.connect()  # this sets up the auth header internally (via sdk)
        dashscope_conversations[client_ws] = conversation
        logger.info("Connected to DashScope.")

        # Set session configuration with SYSTEM context + server VAD
        # IMPORTANT: instructions goes here (system prompt), not append_text
        conversation.update_session(
            instructions=INSTRUCTIONS,
            output_modalities=[MultiModality.AUDIO, MultiModality.TEXT],
            voice="Chelsie",
            input_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
            output_audio_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
            enable_input_audio_transcription=True,
            input_audio_transcription_model="gummy-realtime-v1",
            enable_turn_detection=True,
            turn_detection_type="server_vad",
            turn_detection_threshold=0.5,
            turn_detection_prefix_padding_ms=300,
            turn_detection_silence_duration_ms=600,
            create_response=True,   # auto-generate replies on user pause
            interrupt_response=True # allow barge-in
        )

        # OPTIONAL: have Yuni greet immediately (comment out if you prefer silence until user talks)
        try:
            conversation.append_text(
                "Greet the student warmly and ask how many people are in their party today."
            )
            # Some SDK builds also expose an explicit trigger; if available, you can call it:
            # conversation.create_response()
        except Exception:
            pass

        # Relay messages from browser to DashScope
        async for raw in client_ws:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Non-JSON frame from client, ignoring.")
                continue

            msg_type = data.get("type")

            # Allow the client to override/extend session settings at runtime
            if msg_type == "session.update":
                session = data.get("session", {})
                # If frontend supplies its own instructions, merge/override
                instructions = session.get("instructions", INSTRUCTIONS)

                # Build kwargs carefully, falling back to current defaults
                kwargs = {
                    "instructions": instructions,
                    "output_modalities": session.get("modalities", [MultiModality.AUDIO, MultiModality.TEXT]),
                    "voice": session.get("voice", "Chelsie"),
                    "input_audio_format": AudioFormat.PCM_16000HZ_MONO_16BIT,
                    "output_audio_format": AudioFormat.PCM_24000HZ_MONO_16BIT,
                    "enable_input_audio_transcription": True,
                    "input_audio_transcription_model": session.get("input_audio_transcription", {}).get("model", "gummy-realtime-v1"),
                    "enable_turn_detection": True,
                    "turn_detection_type": session.get("turn_detection", {}).get("type", "server_vad"),
                    "turn_detection_threshold": session.get("turn_detection", {}).get("threshold", 0.5),
                    "turn_detection_prefix_padding_ms": session.get("turn_detection", {}).get("prefix_padding_ms", 300),
                    "turn_detection_silence_duration_ms": session.get("turn_detection", {}).get("silence_duration_ms", 600),
                    "create_response": True,
                    "interrupt_response": True,
                }
                try:
                    conversation.update_session(**kwargs)
                except Exception as e:
                    logger.error("session.update failed: %s", e)

            elif msg_type == "input_audio_buffer.append":
                # Audio is base64 PCM16 from the browser
                b64 = data.get("audio")
                if b64:
                    try:
                        conversation.append_audio(b64)
                    except Exception as e:
                        logger.error("append_audio failed: %s", e)

            elif msg_type == "input_audio_buffer.commit":
                # Needed only if you run WITHOUT server_vad or want manual control
                try:
                    conversation.commit_audio()
                except Exception as e:
                    logger.error("commit_audio failed: %s", e)

            elif msg_type == "response.create":
                # Manually trigger a response (useful if server_vad disabled)
                try:
                    conversation.create_response()
                except Exception as e:
                    logger.error("create_response failed: %s", e)

            elif msg_type == "conversation.append_text":
                # Optional helper to add a text message from the user side
                txt = data.get("text", "")
                if txt:
                    try:
                        conversation.append_text(txt)
                    except Exception as e:
                        logger.error("append_text failed: %s", e)

            else:
                # Unknown or UI-only messages are ignored
                pass

    except Exception as e:
        logger.error("Client handler error: %s", e)

    finally:
        # Cleanup
        if conversation is not None:
            try:
                conversation.close()
            except Exception:
                pass
        dashscope_conversations.pop(client_ws, None)
        active_connections.discard(client_ws)
        try:
            await client_ws.close()
        except Exception:
            pass
        logger.info("Client disconnected.")

# -----------------------------------------------------------------------------
# Server bootstrap
# -----------------------------------------------------------------------------
async def main():
    logger.info("Starting relay on ws://0.0.0.0:%d", RELAY_PORT)
    server = await websockets.serve(
        handle_client,
        host="0.0.0.0",
        port=RELAY_PORT,
        ping_interval=20,
        ping_timeout=10,
        max_size=2**22,  # ~4MB; enough for audio frames
    )
    logger.info("✅ Relay running. Forwarding to %s (model=%s)", DASHSCOPE_WS_URL, DASHSCOPE_MODEL)
    await server.wait_closed()

def _sigint(sig, frame):
    logger.info("Shutting down...")
    for ws in list(active_connections):
        try:
            asyncio.get_event_loop().create_task(ws.close())
        except Exception:
            pass
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _sigint)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        _sigint(None, None)
