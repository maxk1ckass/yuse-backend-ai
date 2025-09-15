# main.py
import os
import re
import time
import threading
import json
from typing import Generator, Tuple, Dict, Any

import numpy as np
from fastapi import FastAPI
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------
# Env & device
# ----------------------------
load_dotenv()

HF_MODEL = os.getenv("QWEN_FASTRTC_MODEL", "../models/qwen2.5-omni-7b-gptq-int4")

# Check GPU availability
if torch.cuda.is_available():
    print(f"✅ GPU acceleration enabled ({torch.cuda.get_device_name(0)})")
else:
    print("⚠️  Running on CPU (GPU not available)")

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda") if USE_CUDA else torch.device("cpu")
DTYPE = torch.float16 if USE_CUDA else torch.float32

# ----------------------------
# Optional FastRTC & VAD deps
# ----------------------------
Stream = None
ReplyOnPause = None
fastrtc_ok = False
vad_ok = False

try:
    from fastrtc import Stream, ReplyOnPause, get_stt_model, get_tts_model
    print("✅ Voice processing ready")
    fastrtc_ok = True
except Exception as e:
    print(f"⚠️  Voice processing limited: {e}")

if fastrtc_ok:
    try:
        import onnxruntime  # noqa: F401
        vad_ok = True
        print("✅ Smart conversation mode enabled")
    except Exception as e:
        print(f"⚠️  Basic conversation mode: {e}")
        vad_ok = False

# ----------------------------
# STT / TTS (optional)
# ----------------------------
stt = None
tts = None
if fastrtc_ok:
    try:
        print("INFO:     Warming up STT model.")
        stt = get_stt_model()
        print("INFO:     STT model warmed up.")
        print("✅ Speech recognition ready")
    except Exception as e:
        print(f"⚠️  Speech recognition unavailable: {e}")
        stt = None

    try:
        print("INFO:     Warming up TTS model.")
        tts = get_tts_model()
        print("INFO:     TTS model warmed up.")
        print("✅ Voice synthesis ready")
    except Exception as e:
        print(f"⚠️  Voice synthesis unavailable: {e}")
        tts = None

# ----------------------------
# Qwen LLM
# ----------------------------
print(f"Loading Qwen model from: {HF_MODEL}")
try:
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        dtype=DTYPE,  # replaces torch_dtype
        device_map="auto" if USE_CUDA else None,
        trust_remote_code=True,
    )
    if not USE_CUDA:
        model = model.to(DEVICE)

    try:
        model.generation_config.use_cache = False
    except Exception:
        pass

    print("Qwen model loaded successfully")
except Exception as e:
    print(f"Failed to load Qwen model: {e}")
    tokenizer = None
    model = None

# ----------------------------
# SYSTEM PROMPT & PROMPT MANAGEMENT
# ----------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are a restraurant waiter and I'm the customer. Please greeting me, ask me questions and serve me like a real waiter. Please keep in mind you never speak long sentences. Each of your response should be less than 20 words."
)

# Store current prompts per session
current_prompts: Dict[str, Dict[str, Any]] = {}

def get_current_prompt(session_id: str = "default") -> Dict[str, Any]:
    """Get current prompt for session"""
    return current_prompts.get(session_id, {
        "systemPrompt": "You are Yuni, a friendly English instructor helping students practicing dialogue roleplay in real life scenarios.",
        "instructions": "Be encouraging and provide gentle corrections when needed.",
        "context": "",
        "scenario": "",
        "personality": "friendly, patient, encouraging",
        "language": "english",
        "difficulty": "beginner"
        # Note: role is optional - let conversation determine role naturally
    })

def update_prompt(session_id: str, prompt_data: Dict[str, Any]):
    """Update prompt for session - merges with existing prompt data"""
    current_prompt = get_current_prompt(session_id)
    merged_prompt = {**current_prompt, **prompt_data}
    current_prompts[session_id] = merged_prompt
    print(f"Updated prompt for session {session_id}: {merged_prompt}")

def initialize_prompt_from_offer(session_id: str, prompt_config: Dict[str, Any]):
    """Initialize prompt from WebRTC offer"""
    if prompt_config:
        current_prompts[session_id] = prompt_config
        print(f"Initialized prompt for session {session_id} from offer: {current_prompts[session_id]}")
    else:
        print(f"No prompt config in offer, using default for session {session_id}")

def generate_system_prompt_from_data(prompt_data: Dict[str, Any]) -> str:
    """Generate system prompt from prompt data"""
    system_prompt = prompt_data.get('systemPrompt', 'You are Yuni, a friendly English instructor helping students practicing dialogue roleplay in real life scenarios.')
    instructions = prompt_data.get('instructions', 'Be encouraging and provide gentle corrections when needed.')
    context = prompt_data.get('context', '')
    personality = prompt_data.get('personality', 'friendly, patient, encouraging')
    difficulty = prompt_data.get('difficulty', 'beginner')
    role = prompt_data.get('role')  # Optional - let conversation determine role
    scenario = prompt_data.get('scenario', '')
    
    # Build base prompt
    full_prompt = f"{system_prompt} {instructions}"
    if context:
        full_prompt += f" {context}"
    full_prompt += "."
    
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
        full_prompt += role_instructions
    
    # Add personality and response guidelines
    if personality:
        full_prompt += f" Personality: {personality}."
    full_prompt += f" Keep vocabulary {difficulty} level. Each response should be less than 20 words."
    
    # Add role flexibility instruction when no specific role is set
    if not role:
        full_prompt += f" Adapt your role based on the conversation context as needed."
    
    return full_prompt

# ----------------------------
# Anti-feedback + reentrancy state
# ----------------------------
SPEAKING = False
LAST_AI_TEXT = ""
TTS_COOLDOWN_S = 0.8
last_tts_end_time = 0.0

# Serialize handler calls to avoid "generator already executing"
RESPOND_LOCK = threading.Lock()

def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def is_self_echo(stt_text: str, ai_text: str) -> bool:
    if not stt_text or not ai_text:
        return False
    a, b = _norm(stt_text), _norm(ai_text)
    if not a or not b:
        return False
    return a == b or a in b or b in a

# ----------------------------
# Helpers
# ----------------------------
def silence_chunk(seconds: float = 0.25, sr: int = 16000) -> Tuple[int, np.ndarray]:
    return (sr, np.zeros(int(sr * seconds), dtype=np.int16))

def run_llm(user_text: str, session_id: str = "default") -> str:
    """Memory-frugal generation to avoid second-turn VRAM spikes."""
    if not user_text or not user_text.strip():
        return "Sorry, I didn't catch that. Please try again."
    if model is None or tokenizer is None:
        return "AI model is not available. Please check your model installation."

    # Get current prompt for session
    current_prompt_data = get_current_prompt(session_id)
    system_prompt = generate_system_prompt_from_data(current_prompt_data)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": "Session memory: Last time we practiced ordering food; the student asked about salmon vs pasta and requested a side salad."},
        {"role": "user", "content": user_text},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    attention_mask = torch.ones_like(inputs)

    # Trim context for small KV cache
    MAX_INPUT_TOKENS = 256
    if inputs.shape[-1] > MAX_INPUT_TOKENS:
        inputs = inputs[:, -MAX_INPUT_TOKENS:]
        attention_mask = attention_mask[:, -MAX_INPUT_TOKENS:]

    with torch.inference_mode():
        output_ids = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=96,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            use_cache=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output_ids[0][inputs.shape[-1]:], skip_special_tokens=True)

    try:
        del inputs, attention_mask, output_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return text

# ----------------------------
# Data Channel Handling
# ----------------------------
def handle_data_channel_message(message: str, session_id: str = "default"):
    """Handle control messages from data channel"""
    try:
        data = json.loads(message)
        message_type = data.get('type')
        
        if message_type == 'control.prompt.update':
            prompt_data = data.get('prompt', {})
            update_prompt(session_id, prompt_data)
            print(f"Prompt updated for session {session_id}")
        elif message_type == 'control.role.switch':
            roles = data.get('roles', {})
            role_prompt = {
                'context': f"We are practicing restaurant ordering scenarios. You are the {roles.get('ai', 'waiter')}, the student is the {roles.get('user', 'customer')}.",
                'role': roles.get('ai', 'waiter')
            }
            update_prompt(session_id, role_prompt)
            print(f"Role switched for session {session_id}: {roles}")
        elif message_type == 'control.session.init':
            # Handle initial session setup with prompt config
            prompt_config = data.get('prompt_config', {})
            if prompt_config:
                initialize_prompt_from_offer(session_id, prompt_config)
                print(f"Session initialized with prompt config for {session_id}")
        else:
            print(f"Unhandled data channel message type: {message_type}")
            
    except json.JSONDecodeError:
        print(f"Invalid JSON in data channel message: {message}")
    except Exception as e:
        print(f"Error handling data channel message: {e}")

# ----------------------------
# Audio pipeline
# ----------------------------
last_response_time = 0
MIN_RESPONSE_INTERVAL = 2.0  # seconds

def respond(audio: Tuple[int, np.ndarray]) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Input:  (sample_rate:int, mono_int16_audio: np.ndarray shape [N])
    Yields: (sample_rate:int, mono_int16_audio:int16[1,N]) chunks for TTS
    """
    global last_response_time, LAST_AI_TEXT, SPEAKING, last_tts_end_time

    # ----- Reentrancy guard -----
    if not RESPOND_LOCK.acquire(blocking=False):
        # Another invocation is still running (LLM or TTS). Drop this one.
        print("Reentry blocked: handler already running")
        yield silence_chunk(0.1)
        return

    try:
        sr, arr = audio
        now = time.time()

        # Gate 1: if bot speaking, ignore mic
        if SPEAKING:
            yield silence_chunk(0.1)
            return

        # Gate 2: cooldown after TTS ends
        if now - last_tts_end_time < TTS_COOLDOWN_S:
            yield silence_chunk(0.1)
            return

        print(f"=== Audio Processing Debug ===")
        print(f"Audio shape: {arr.shape}, sample rate: {sr}")
        audio_length = arr.shape[-1] if len(arr.shape) > 1 else len(arr)
        print(f"Audio duration: {audio_length / sr:.2f} seconds")

        # Gate 3: throttle
        if now - last_response_time < MIN_RESPONSE_INTERVAL:
            print("⏰ Too soon since last response, ignoring audio")
            yield silence_chunk(0.2)
            return

        # 1) STT
        if stt is None:
            text = "Speech-to-text is not available. Please check your installation."
            print(f"STT not available, using fallback text: {text}")
        else:
            print("Processing speech-to-text...")
            try:
                text = stt.stt(audio)
                print(f"STT result: '{text}'")
            except Exception as e:
                print(f"STT error: {e}")
                text = ""

        # Skip empty STT with gentle nudge
        if not text or not text.strip():
            print("Empty STT; skipping LLM.")
            if tts is not None:
                SPEAKING = True
                for chunk in tts.stream_tts_sync("Sorry, I didn't catch that. Please try again."):
                    yield chunk
                SPEAKING = False
                last_tts_end_time = time.time()
            else:
                yield silence_chunk(0.2)
            return

        # 2) Self-echo filter
        if is_self_echo(text, LAST_AI_TEXT):
            print("Detected self-echo; dropping this turn.")
            yield silence_chunk(0.1)
            return

        # 3) LLM
        print(f"Running LLM with input: '{text}'")
        output_text = run_llm(text)
        print(f"LLM output: '{output_text}'")

        # 4) TTS or silence
        if tts is not None:
            print("Generating TTS audio...")
            last_response_time = now
            LAST_AI_TEXT = output_text
            SPEAKING = True
            for chunk in tts.stream_tts_sync(output_text):
                yield chunk
            SPEAKING = False
            last_tts_end_time = time.time()
        else:
            print("TTS not available, returning silence")
            yield silence_chunk(0.2)

        print("=== End Audio Processing ===")

    finally:
        # Always release lock to avoid deadlock if an exception occurs
        if RESPOND_LOCK.locked():
            RESPOND_LOCK.release()

# ----------------------------
# Build Stream + FastAPI
# ----------------------------
app = FastAPI()

if fastrtc_ok:
    try:
        if vad_ok:
            # Keep non-interruptible to avoid reentry during TTS
            stream = Stream(
                handler=ReplyOnPause(respond, can_interrupt=False),
                modality="audio",
                mode="send-receive",
                data_channel_handler=handle_data_channel_message,  # Add data channel handler
            )
            print("Using ReplyOnPause with VAD and data channel support")
        else:
            def no_vad_handler(audio):
                try:
                    for ch in respond(audio):
                        yield ch
                except Exception as ex:
                    print(f"no_vad_handler error: {ex}")
                    yield silence_chunk(0.2)

            stream = Stream(
                handler=no_vad_handler,
                modality="audio",
                mode="send-receive",
                data_channel_handler=handle_data_channel_message,  # Add data channel handler
            )
            print("Using simple no-VAD handler with data channel support")
        stream.mount(app)
    except Exception as e:
        print(f"Failed to initialize FastRTC Stream: {e}")
else:
    print("FastRTC is unavailable; starting FastAPI without WebRTC endpoints.")

# Health
@app.get("/health")
def health():
    return {
        "ok": True,
        "cuda": USE_CUDA,
        "vad_enabled": bool(fastrtc_ok and vad_ok),
        "fastrtc": bool(fastrtc_ok),
        "model": HF_MODEL,
    }
