#!/usr/bin/env python3
"""
DashScope WebSocket Relay Server
Relays WebSocket connections between frontend and DashScope cloud API using official SDK
"""

import asyncio
import http
from websockets.http11 import Response  # <-- key import
from websockets.datastructures import Headers # <-- add this
import websockets
import json
import os
import base64
import logging
import time
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
current_prompts: Dict[Any, Dict[str, Any]] = {}  # Store current prompts per connection

# Store conversation scripts per session
conversation_scripts: Dict[Any, list] = {}  # Store conversation turns per connection

# Store session parameters per connection to preserve them during updates
session_parameters: Dict[Any, Dict[str, Any]] = {}  # Store session config per connection

class DashScopeRelay:
    def __init__(self):
        self.port = int(os.getenv('RELAY_PORT', 8001))  # Different from FastRTC (8000)
        self.health_probe_health_message = "ok"  # whatever class data you want
        
    def generate_instructions_from_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """Generate full instructions from prompt data"""
        system_prompt = prompt_data.get('systemPrompt', 'You are Yuni, a friendly English instructor helping students practicing dialogue roleplay in real life scenarios.')
        instructions = prompt_data.get('instructions', 'Be encouraging and provide gentle corrections when needed.')
        context = prompt_data.get('context', '')
        personality = prompt_data.get('personality', 'friendly, patient, encouraging')
        difficulty = prompt_data.get('difficulty', 'beginner')
        scenario = prompt_data.get('scenario', '')
        role = prompt_data.get('role')  # Optional - let conversation determine role
        
        # Build base instructions
        full_instructions = f"{system_prompt} {instructions}"
        if context:
            full_instructions += f" {context}"
        full_instructions += "."
        
        # Add role-specific instructions only if role is explicitly provided
        if role:
            if scenario:
                role_instructions = f" You are the {role} in this {scenario} scenario."
            else:
                role_instructions = f" You are the {role} in this scenario."
            
            if role == 'waiter':
                role_instructions += " Take orders, suggest menu items, and provide excellent service. Be helpful and professional."
            elif role == 'customer':
                role_instructions += " Order food, ask questions about the menu, and interact naturally with the waiter."
            full_instructions += role_instructions
        
        # Add personality and interaction guidelines
        if personality:
            full_instructions += f" Personality: {personality}."
        full_instructions += f" Turn-taking: reply in 1�? short sentences, then stop so the student can speak. Be encouraging; if needed, give tiny inline corrections in brackets. Keep vocabulary {difficulty} level, natural and conversational."
        
        # Add role flexibility instruction when no specific role is set
        if not role:
            full_instructions += f" Adapt your role based on the conversation context as needed."
        
        return full_instructions
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for better readability"""
        if not text:
            return text
        
        # Remove extra whitespace and normalize
        text = text.strip()
        
        # Add spaces between words that might be concatenated
        import re
        # Add space before capital letters that follow lowercase letters (camelCase)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        # Add space before numbers that follow letters
        text = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', text)
        # Add space after numbers that are followed by letters
        text = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', text)
        
        # Normalize multiple spaces to single spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    async def handle_session_end(self, websocket):
        """Handle session end and send conversation script"""
        try:
            if websocket in conversation_scripts:
                script = conversation_scripts[websocket]
                
                # Clean up text in the script
                for turn in script:
                    turn['text'] = self.clean_text(turn['text'])
                
                # Format the script for frontend
                script_data = {
                    "type": "conversation.script",
                    "session_id": id(websocket),  # Use websocket id as session identifier
                    "script": script,
                    "total_turns": len(script),
                    "timestamp": int(time.time() * 1000)  # Unix timestamp in milliseconds
                }
                
                await websocket.send(json.dumps(script_data))
                logger.info(f"Sent conversation script to {websocket.remote_address}: {len(script)} turns")
                
                # Clear the script for this session
                del conversation_scripts[websocket]
            else:
                logger.warning(f"No conversation script found for session {websocket.remote_address}")
                
        except Exception as e:
            logger.error(f"Error handling session end: {e}")
    
    async def handle_initial_greeting_request(self, websocket):
        """Handle request for initial AI greeting"""
        try:
            if websocket in dashscope_conversations:
                conversation = dashscope_conversations[websocket]
                logger.info("Triggering initial AI greeting...")
                
                try:
                    # Method 1: Send a very short silent audio chunk to trigger conversation
                    import base64
                    import numpy as np
                    
                    # Create a minimal silent audio chunk (50ms of silence)
                    sample_rate = 16000
                    silence_duration = 0.05  # 50ms
                    silent_samples = np.zeros(int(sample_rate * silence_duration), dtype=np.int16)
                    silent_audio_b64 = base64.b64encode(silent_samples.tobytes()).decode('utf-8')
                    
                    # Send the silent audio to trigger conversation
                    conversation.append_audio(silent_audio_b64)
                    logger.info("�?Triggered initial AI greeting with silent audio")
                    
                except Exception as e:
                    logger.warning(f"Silent audio trigger failed: {e}")
                    
                    # Method 2: Try to send a minimal audio signal
                    try:
                        # Create a very quiet tone (not silence) to trigger the AI
                        import numpy as np
                        
                        sample_rate = 16000
                        duration = 0.1  # 100ms
                        frequency = 440  # A note
                        t = np.linspace(0, duration, int(sample_rate * duration))
                        tone = np.sin(2 * np.pi * frequency * t) * 0.01  # Very quiet
                        audio_samples = (tone * 32767).astype(np.int16)
                        
                        tone_audio_b64 = base64.b64encode(audio_samples.tobytes()).decode('utf-8')
                        conversation.append_audio(tone_audio_b64)
                        logger.info("�?Triggered initial AI greeting with quiet tone")
                        
                    except Exception as e2:
                        logger.error(f"All greeting trigger methods failed: {e2}")
                        
                        # Send error response to frontend
                        await websocket.send(json.dumps({
                            "type": "initial_greeting.error",
                            "success": False,
                            "message": "Failed to trigger initial greeting"
                        }))
                        return
                
                # Send success response to frontend
                await websocket.send(json.dumps({
                    "type": "initial_greeting.triggered",
                    "success": True,
                    "message": "Initial greeting triggered"
                }))
                logger.info("Sent initial greeting trigger confirmation to frontend")
                
            else:
                logger.warning(f"No conversation found for websocket {websocket.remote_address}")
                await websocket.send(json.dumps({
                    "type": "initial_greeting.error",
                    "success": False,
                    "message": "No active conversation found"
                }))
                
        except Exception as e:
            logger.error(f"Error handling initial greeting request: {e}")
            try:
                await websocket.send(json.dumps({
                    "type": "initial_greeting.error",
                    "success": False,
                    "message": f"Error triggering initial greeting: {str(e)}"
                }))
            except:
                pass
    
    async def handle_prompt_update(self, websocket, prompt_data: Dict[str, Any]):
        """Handle prompt update message"""
        try:
            logger.info("=== BACKEND: Handling prompt update ===")
            logger.info(f"Received prompt data: {prompt_data}")
            
            # If conversation exists, update it
            if websocket in dashscope_conversations:
                conversation = dashscope_conversations[websocket]
                
                # Get current prompt data to preserve context before updating
                current_prompt = current_prompts.get(websocket, {})
                logger.info(f"Current prompt data: {current_prompt}")
                
                # Merge current prompt with new prompt data (new data takes precedence)
                merged_prompt = {**current_prompt, **prompt_data}
                logger.info(f"Merged prompt data: {merged_prompt}")
                
                # Store the merged prompt data
                current_prompts[websocket] = merged_prompt
                
                # Generate comprehensive instructions that include all context
                new_instructions = self.generate_instructions_from_prompt(merged_prompt)
                logger.info(f"Generated new instructions length: {len(new_instructions)}")
                logger.info(f"Generated new instructions preview: {new_instructions[:300]}...")
                
                # Update session with new instructions (this replaces the previous instructions)
                logger.info("Updating DashScope conversation with new instructions...")
                
                if websocket in session_parameters:
                    # Use stored session parameters and only update instructions
                    stored_params = session_parameters[websocket].copy()
                    stored_params['instructions'] = new_instructions
                    logger.info(f"Updating with stored parameters: {list(stored_params.keys())}")
                    conversation.update_session(**stored_params)
                    # Update stored parameters
                    session_parameters[websocket] = stored_params
                else:
                    # Fallback to default parameters (shouldn't happen in normal flow)
                    logger.warning("No stored session parameters found, using fallback")
                    conversation.update_session(
                        output_modalities=[MultiModality.AUDIO, MultiModality.TEXT],
                        voice='Chelsie',
                        input_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
                        output_audio_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                        enable_input_audio_transcription=True,
                        input_audio_transcription_model='gummy-realtime-v1',
                        enable_turn_detection=True,
                        turn_detection_type='server_vad',
                        instructions=new_instructions
                    )
                
                logger.info(f"�?Updated instructions for connection {websocket.remote_address}")
                logger.debug(f"New instructions: {new_instructions}")
                
                # Send confirmation back to frontend
                response = {
                    "type": "control.prompt.updated",
                    "success": True,
                    "message": "Prompt updated successfully",
                    "appliedPrompt": {
                        "systemPrompt": prompt_data.get('systemPrompt'),
                        "instructions": prompt_data.get('instructions'),
                        "context": prompt_data.get('context')
                    },
                    "instructions_length": len(new_instructions)
                }
                await websocket.send(json.dumps(response))
                logger.info("Sent prompt update confirmation to frontend")
            else:
                logger.warning(f"No conversation found for websocket {websocket.remote_address}")
                
        except Exception as e:
            logger.error(f"�?Error handling prompt update: {e}")
            error_response = {
                "type": "error",
                "code": "prompt_update_failed",
                "message": f"Failed to update prompt: {str(e)}"
            }
            await websocket.send(json.dumps(error_response))
        
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
            
            # Configure session with generic default instructions that include initial greeting
            default_instructions = """You are Yuni, a friendly English instructor helping students practicing dialogue roleplay in real life scenarios. Be encouraging and provide gentle corrections when needed. Turn-taking: reply in 1�? short sentences, then stop so the student can speak. Be encouraging; if needed, give tiny inline corrections in brackets. Keep vocabulary beginner level, natural and conversational. Adapt your role based on the conversation context as needed.

IMPORTANT: Start every conversation by greeting the student warmly and explaining what we're going to practice. Begin with: "Hello! I'm Yuni, your English instructor. Today we're going to practice [scenario]. Are you ready to start?"""""

            # Store the session parameters for future updates
            session_params = {
                'output_modalities': [MultiModality.AUDIO, MultiModality.TEXT],
                'voice': 'Chelsie',
                'input_audio_format': AudioFormat.PCM_16000HZ_MONO_16BIT,
                'output_audio_format': AudioFormat.PCM_24000HZ_MONO_16BIT,
                'enable_input_audio_transcription': True,
                'input_audio_transcription_model': 'gummy-realtime-v1',
                'enable_turn_detection': True,
                'turn_detection_type': 'server_vad',
                'instructions': default_instructions
            }
            session_parameters[websocket] = session_params

            conversation.update_session(**session_params)
            
            # Send a message to frontend indicating session is ready for initial greeting
            logger.info("Session configured, ready for initial AI greeting")
            try:
                await websocket.send(json.dumps({
                    "type": "session.ready",
                    "message": "Session is ready, AI can now greet the user",
                    "ready_for_initial_greeting": True
                }))
                logger.info("�?Sent session ready message to frontend")
            except Exception as e:
                logger.warning(f"Failed to send session ready message: {e}")
            
            logger.info("Connected to DashScope cloud")
            
            # Handle messages from frontend
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get('type')
                    
                    if message_type == 'session.update':
                        # Frontend is sending session config - update with new instructions
                        logger.info("=== BACKEND: Received session.update from frontend ===")
                        session_data = data.get('session', {})
                        frontend_instructions = session_data.get('instructions', '')
                        
                        logger.info(f"Frontend session data: {session_data}")
                        logger.info(f"Frontend instructions length: {len(frontend_instructions)}")
                        logger.info(f"Frontend instructions preview: {frontend_instructions[:200]}...")
                        
                        if frontend_instructions and websocket in dashscope_conversations and websocket in session_parameters:
                            conversation = dashscope_conversations[websocket]
                            stored_params = session_parameters[websocket].copy()
                            
                            logger.info("Updating DashScope conversation with frontend instructions...")
                            logger.info(f"Stored session parameters: {list(stored_params.keys())}")
                            
                            try:
                                # Update only the instructions while preserving all other parameters
                                stored_params['instructions'] = frontend_instructions
                                logger.info("Calling conversation.update_session with preserved parameters...")
                                
                                conversation.update_session(**stored_params)
                                logger.info("�?Successfully updated DashScope session with frontend instructions")
                                
                                # Update stored parameters with new instructions
                                session_parameters[websocket] = stored_params
                                
                                # Send confirmation back to frontend
                                response = {
                                    "type": "session.updated",
                                    "success": True,
                                    "message": "Session updated with new instructions",
                                    "instructions_length": len(frontend_instructions)
                                }
                                await websocket.send(json.dumps(response))
                                logger.info("Sent session update confirmation to frontend")
                                
                            except Exception as e:
                                logger.error(f"�?Failed to update DashScope session: {e}")
                                error_response = {
                                    "type": "session.update.error",
                                    "success": False,
                                    "message": f"Failed to update session: {str(e)}"
                                }
                                await websocket.send(json.dumps(error_response))
                        else:
                            logger.warning("No instructions provided or conversation/session parameters not found")
                            logger.warning(f"Has instructions: {bool(frontend_instructions)}")
                            logger.warning(f"Has conversation: {websocket in dashscope_conversations}")
                            logger.warning(f"Has session parameters: {websocket in session_parameters}")
                    elif message_type == 'input_audio_buffer.append':
                        # Forward audio data to DashScope
                        audio_b64 = data.get('audio')
                        if audio_b64:
                            conversation.append_audio(audio_b64)
                    elif message_type == 'control.prompt.update':
                        # Handle prompt update
                        prompt_data = data.get('prompt', {})
                        await self.handle_prompt_update(websocket, prompt_data)
                    elif message_type == 'control.session.config':
                        # Handle session configuration update
                        config_data = data.get('config', {})
                        logger.info(f"Session config update received: {config_data}")
                    elif message_type == 'control.role.switch':
                        # Handle role switching
                        roles = data.get('roles', {})
                        logger.info(f"Role switch requested: {roles}")
                        # Convert role switch to prompt update
                        role_prompt = {
                            'context': f"We are practicing restaurant ordering scenarios. You are the {roles.get('ai', 'waiter')}, the student is the {roles.get('user', 'customer')}.",
                            'role': roles.get('ai', 'waiter')
                        }
                        await self.handle_prompt_update(websocket, role_prompt)
                    elif message_type == 'session.end':
                        # Handle session end - send conversation script
                        logger.info(f"Session end requested from {websocket.remote_address}")
                        await self.handle_session_end(websocket)
                    elif message_type == 'request.initial_greeting':
                        # Handle request for initial AI greeting
                        logger.info(f"Initial greeting requested from {websocket.remote_address}")
                        await self.handle_initial_greeting_request(websocket)
                    else:
                        logger.info(f"Unhandled message type: {message_type}")
                        
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
            if websocket in current_prompts:
                del current_prompts[websocket]
            if websocket in conversation_scripts:
                del conversation_scripts[websocket]
            if websocket in session_parameters:
                del session_parameters[websocket]
            logger.info(f"Frontend disconnected: {websocket.remote_address}")
    
    async def start_server(self):
        """Start the relay server"""
        logger.info(f"Starting DashScope relay server on port {self.port}")
        logger.info(f"DashScope API Key: {DASHSCOPE_API_KEY[:10]}...")

        def make_response(status: int | http.HTTPStatus, headers: Headers | None = None, body: bytes = b""):
            # Build a websockets.http11.Response the way websockets>=12 expects it.
            if isinstance(status, http.HTTPStatus):
                code = status.value
                reason = status.phrase.encode("ascii")
            else:
                code = int(status)
                reason = http.HTTPStatus(code).phrase.encode("ascii")

            if headers is None:
                headers = Headers()
            # Ensure required headers exist when body present (optional but nice)
            if body and b"content-length" not in {k.lower().encode() if isinstance(k, str) else k.lower() for k, _ in headers.raw_items()}:
                headers["content-length"] = str(len(body))

            return Response(code, reason, headers, body)

        async def process_request(protocol, request):
            try:
                path = getattr(request, "path", "/")
                logger.info(f"HTTP probe on {path}")

                # 1) Health check: still 200 OK (Front Door / your curl)
                if path == "/health":
                    body = getattr(self, "health_probe_health_message", "ok").encode("utf-8")
                    h = Headers()
                    h["content-type"] = "text/plain"
                    h["cache-control"] = "no-store"
                    return make_response(http.HTTPStatus.OK, h, body)

                # 2) WebSocket endpoint on "/" �?DO NOT return an HTTP response.
                # Returning None tells websockets to continue the WS handshake.
                if path == "/":
                    # If it's a true WS handshake, proceed:
                    # (Optional) Only gate on Upgrade header:
                    # up = (request.headers.get("upgrade") or request.headers.get("Upgrade") or "").lower()
                    # if up == "websocket":
                    #     return None
                    return None  # <�?let WS handshake happen

                # 3) Any other plain HTTP paths: hint upgrade
                h = Headers()
                h["content-type"] = "text/plain"
                h["connection"] = "close"
                return make_response(
                    http.HTTPStatus.UPGRADE_REQUIRED,
                    h,
                    b"WebSocket endpoint. Use WS/WSS.\n",
                )

            except Exception as e:
                logger.exception(f"process_request error: {e}")
                h = Headers()
                h["content-type"] = "text/plain"
                return make_response(http.HTTPStatus.INTERNAL_SERVER_ERROR, h, b"internal error\n")

        # Start WebSocket server
        server = await websockets.serve(
            self.handle_frontend_connection,
            "0.0.0.0",
            self.port,
            ping_interval=20,
            ping_timeout=10,
            process_request=process_request
        )
        
        logger.info(f"�?DashScope relay server running on ws://localhost:{self.port}")
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
            # Track conversation text for script generation
            self.track_conversation_text(response)
            
            # Forward DashScope events to frontend using the event loop
            if self.event_loop and not self.event_loop.is_closed():
                asyncio.run_coroutine_threadsafe(
                    self.send_to_frontend(response), 
                    self.event_loop
                )
        except Exception as e:
            logger.error(f"Error in DashScope callback: {e}")
    
    def track_conversation_text(self, response: dict) -> None:
        """Track conversation text for script generation"""
        try:
            response_type = response.get('type', '')
            
            # Debug: Log all response types to understand the data flow
            logger.debug(f"DashScope response type: {response_type}")
            
            # Track user speech transcription
            if response_type == 'conversation.item.input_audio_transcription.completed':
                transcript = response.get('transcript', '').strip()
                if transcript:
                    turn = {
                        'speaker': 'user',
                        'text': transcript,
                        'timestamp': int(time.time() * 1000),
                        'type': 'user_speech'
                    }
                    if self.frontend_ws not in conversation_scripts:
                        conversation_scripts[self.frontend_ws] = []
                    conversation_scripts[self.frontend_ws].append(turn)
                    logger.info(f"Tracked user speech: '{transcript}'")
            
            # Track AI response text - try multiple possible response types
            elif response_type in ['response.audio_transcript.delta', 'response.text.delta']:
                ai_text = response.get('delta', '').strip()
                logger.debug(f"AI text delta: '{ai_text}' (length: {len(ai_text)})")
                
                if ai_text:
                    # Check if this is a new response or continuation
                    if (self.frontend_ws in conversation_scripts and 
                        conversation_scripts[self.frontend_ws] and 
                        conversation_scripts[self.frontend_ws][-1]['speaker'] == 'ai'):
                        # Append to existing AI turn with better word boundary handling
                        existing_text = conversation_scripts[self.frontend_ws][-1]['text']
                        
                        # Smart spacing logic
                        if existing_text:
                            # Check if we need to add a space
                            needs_space = (
                                not existing_text.endswith(' ') and 
                                not existing_text.endswith('.') and 
                                not existing_text.endswith(',') and 
                                not existing_text.endswith('!') and 
                                not existing_text.endswith('?') and 
                                not existing_text.endswith(':') and 
                                not existing_text.endswith(';') and
                                not ai_text.startswith(' ') and
                                not ai_text.startswith('.') and
                                not ai_text.startswith(',') and
                                not ai_text.startswith('!') and
                                not ai_text.startswith('?')
                            )
                            
                            if needs_space:
                                conversation_scripts[self.frontend_ws][-1]['text'] += ' ' + ai_text
                            else:
                                conversation_scripts[self.frontend_ws][-1]['text'] += ai_text
                        else:
                            conversation_scripts[self.frontend_ws][-1]['text'] += ai_text
                        
                        logger.info(f"Appended AI response: '{ai_text}' -> Total: '{conversation_scripts[self.frontend_ws][-1]['text']}'")
                    else:
                        # Create new AI turn
                        turn = {
                            'speaker': 'ai',
                            'text': ai_text,
                            'timestamp': int(time.time() * 1000),
                            'type': 'ai_response'
                        }
                        if self.frontend_ws not in conversation_scripts:
                            conversation_scripts[self.frontend_ws] = []
                        conversation_scripts[self.frontend_ws].append(turn)
                        logger.info(f"New AI response: '{ai_text}'")
            
            # Also check for complete AI responses
            elif response_type == 'response.audio_transcript.completed':
                complete_text = response.get('transcript', '').strip()
                if complete_text:
                    logger.info(f"Complete AI transcript: '{complete_text}'")
                    # Use the complete transcript instead of delta chunks
                    if (self.frontend_ws in conversation_scripts and 
                        conversation_scripts[self.frontend_ws] and 
                        conversation_scripts[self.frontend_ws][-1]['speaker'] == 'ai'):
                        # Replace the last AI turn with the complete text
                        conversation_scripts[self.frontend_ws][-1]['text'] = complete_text
                        logger.info(f"Replaced AI response with complete transcript: '{complete_text}'")
                    else:
                        # Create new AI turn with complete text
                        turn = {
                            'speaker': 'ai',
                            'text': complete_text,
                            'timestamp': int(time.time() * 1000),
                            'type': 'ai_response'
                        }
                        if self.frontend_ws not in conversation_scripts:
                            conversation_scripts[self.frontend_ws] = []
                        conversation_scripts[self.frontend_ws].append(turn)
                        logger.info(f"New AI response (complete): '{complete_text}'")
                    
        except Exception as e:
            logger.error(f"Error tracking conversation text: {e}")
    
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
        logger.error("�?DASHSCOPE_API_KEY not set!")
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
