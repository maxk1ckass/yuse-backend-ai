# main.py
import os
import numpy as np
import time
from typing import Generator, Tuple, Optional

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
# We try to import FastRTC & ONNX (GPU) for VAD. If anything fails, we run without VAD.
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

# Try to confirm ONNX Runtime availability (GPU preferred, CPU acceptable)
if fastrtc_ok:
    try:
        # Importing onnxruntime triggers DLL checks on Windows. If it fails, we won't use VAD.
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
        stt = get_stt_model()   # e.g., Whisper
        print("✅ Speech recognition ready")
    except Exception as e:
        print(f"⚠️  Speech recognition unavailable: {e}")
        stt = None

    try:
        tts = get_tts_model()   # e.g., Kokoro / XTTS
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
        dtype=DTYPE,  # <- modern arg (replaces torch_dtype)
        device_map="auto" if USE_CUDA else None,
        trust_remote_code=True,
    )
    # If no device_map given (CPU path), move to CPU explicitly; for GPU w/ device_map="auto" it’s already placed
    if not USE_CUDA:
        model = model.to(DEVICE)
    print("Qwen model loaded successfully")
except Exception as e:
    print(f"Failed to load Qwen model: {e}")
    tokenizer = None
    model = None

SYSTEM_PROMPT = (
    "You are a concise, helpful voice assistant. "
    "Keep answers short and conversational. If asked for code, summarize verbally."
)

# ----------------------------
# Helpers
# ----------------------------
def silence_chunk(seconds: float = 0.25, sr: int = 16000) -> Tuple[int, np.ndarray]:
    return (sr, np.zeros(int(sr * seconds), dtype=np.int16))

def run_llm(user_text: str) -> str:
    if not user_text or not user_text.strip():
        return "Sorry, I didn't catch that. Please try again."
    if model is None or tokenizer is None:
        return "AI model is not available. Please check your model installation."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    text = tokenizer.decode(output_ids[0][inputs.shape[-1]:], skip_special_tokens=True)
    return text

# ----------------------------
# English Instructor Functions
# ----------------------------
def get_initial_greeting() -> str:
    """AI speaks first - English instructor greeting"""
    return ("Hello! I'm your English instructor. I'm excited to practice with you today! "
            "I have a fun roleplay scenario ready. Would you like to hear about it? "
            "Just say 'yes' or 'tell me about it' when you're ready!")

def get_scenario_script() -> dict:
    """Returns a sample dialogue scenario for roleplay"""
    return {
        "title": "Restaurant Order",
        "description": "Practice ordering food at a restaurant",
        "roles": {
            "customer": "You are a customer at a nice restaurant. You want to order food and ask questions about the menu.",
            "waiter": "You are a friendly waiter. Help the customer with their order and make recommendations."
        },
        "dialogue": [
            {"role": "waiter", "text": "Good evening! Welcome to our restaurant. How many people are in your party?"},
            {"role": "customer", "text": "Just one, please. Do you have a table by the window?"},
            {"role": "waiter", "text": "Of course! Right this way. Here's your menu. Can I get you something to drink while you decide?"},
            {"role": "customer", "text": "I'll have a glass of water, please. What do you recommend for dinner?"},
            {"role": "waiter", "text": "Our salmon is excellent tonight, and the pasta is very popular. What sounds good to you?"},
            {"role": "customer", "text": "I'll try the salmon. Is it fresh?"},
            {"role": "waiter", "text": "Yes, it was caught this morning. Would you like it grilled or pan-seared?"},
            {"role": "customer", "text": "Grilled, please. And could I have a side salad?"},
            {"role": "waiter", "text": "Perfect! I'll put that order in right away. Anything else I can get you?"},
            {"role": "customer", "text": "That's all for now, thank you!"}
        ]
    }

def process_instructor_response(user_text: str) -> str:
    """Process user input and generate appropriate instructor response"""
    global conversation_state
    
    user_text_lower = user_text.lower()
    
    # Check if user wants to hear about the scenario
    if any(phrase in user_text_lower for phrase in ["yes", "tell me", "scenario", "roleplay", "practice"]):
        scenario = get_scenario_script()
        conversation_state["current_scenario"] = scenario
        
        response = (f"Great! Today we'll practice a {scenario['title']} scenario. "
                   f"{scenario['description']} "
                   f"I'll be the {list(scenario['roles'].keys())[0]} and you'll be the {list(scenario['roles'].keys())[1]}. "
                   f"Are you ready to start? Just say 'ready' or 'let's begin'!")
        return response
    
    # Check if user is ready to start
    if any(phrase in user_text_lower for phrase in ["ready", "begin", "start", "let's go"]):
        if conversation_state["current_scenario"]:
            scenario = conversation_state["current_scenario"]
            conversation_state["scene_step"] = 0
            conversation_state["ai_role"] = list(scenario["roles"].keys())[0]
            conversation_state["user_role"] = list(scenario["roles"].keys())[1]
            
            first_line = scenario["dialogue"][0]
            response = (f"Perfect! Let's begin. I'm the {conversation_state['ai_role']} and you're the {conversation_state['user_role']}. "
                       f"Here we go: {first_line['text']}")
            return response
        else:
            return "Let me first tell you about our scenario. Would you like to hear about it?"
    
    # During roleplay - check if user is following the script
    if conversation_state["current_scenario"] and conversation_state["scene_step"] > 0:
        return process_roleplay_response(user_text)
    
    # Default response
    return ("I'm here to help you practice English! Would you like to try our roleplay scenario? "
            "Just say 'yes' to hear about it!")

def process_roleplay_response(user_text: str) -> str:
    """Process responses during roleplay"""
    global conversation_state
    
    scenario = conversation_state["current_scenario"]
    current_step = conversation_state["scene_step"]
    
    # Check if we've completed the scenario
    if current_step >= len(scenario["dialogue"]) - 1:
        return ("Excellent work! We've completed the scenario. You did great! "
                "Would you like to try it again, or would you like to practice a different scenario?")
    
    # Move to next step
    conversation_state["scene_step"] += 1
    next_line = scenario["dialogue"][conversation_state["scene_step"]]
    
    # If it's the AI's turn, provide the line
    if next_line["role"] == conversation_state["ai_role"]:
        return next_line["text"]
    else:
        # It's the user's turn, give them a hint
        return (f"Great! Now it's your turn as the {conversation_state['user_role']}. "
                f"Try responding naturally. If you need help, just ask!")

# ----------------------------
# Audio pipeline
# ----------------------------
# Track last response time to prevent feedback loops
last_response_time = 0
MIN_RESPONSE_INTERVAL = 2.0  # Minimum 2 seconds between responses

# Conversation state for English instructor
conversation_state = {
    "is_first_interaction": True,
    "current_scenario": None,
    "user_role": None,
    "ai_role": None,
    "scene_step": 0
}

def respond(audio: Tuple[int, np.ndarray]) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Input:  (sample_rate:int, mono_int16_audio: np.ndarray shape [N])
    Yields: (sample_rate:int, mono_int16_audio:int16[1,N]) chunks for TTS
    """
    global last_response_time, conversation_state
    
    print(f"=== Audio Processing Debug ===")
    print(f"Audio shape: {audio[1].shape}, sample rate: {audio[0]}")
    # Handle both 1D and 2D audio arrays
    audio_length = audio[1].shape[-1] if len(audio[1].shape) > 1 else len(audio[1])
    print(f"Audio duration: {audio_length / audio[0]:.2f} seconds")
    
    # Check if we should respond (prevent feedback loops)
    current_time = time.time()
    if current_time - last_response_time < MIN_RESPONSE_INTERVAL:
        print(f"⏰ Too soon since last response, ignoring audio")
        yield silence_chunk(0.2)
        return
    
    # Special case: First interaction - AI speaks first
    if conversation_state["is_first_interaction"]:
        conversation_state["is_first_interaction"] = False
        greeting_text = get_initial_greeting()
        print(f"🎓 AI Instructor greeting: '{greeting_text}'")
        if tts is not None:
            last_response_time = current_time
            for chunk in tts.stream_tts_sync(greeting_text):
                yield chunk
        else:
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
            text = "Sorry, I didn't catch that. Please try again."

    # 2) Process with English Instructor System
    print(f"Processing user input: '{text}'")
    output_text = process_instructor_response(text)
    print(f"Instructor response: '{output_text}'")

    # 3) TTS or silence
    if tts is not None:
        print("Generating TTS audio...")
        last_response_time = current_time  # Update response time
        for chunk in tts.stream_tts_sync(output_text):
            yield chunk
    else:
        print("TTS not available, returning silence")
        # No TTS → return brief silence (keeps pipeline valid)
        yield silence_chunk(0.2)
    
    print("=== End Audio Processing ===")

# ----------------------------
# Build Stream + FastAPI
# ----------------------------
app = FastAPI()

if fastrtc_ok:
    try:
        if vad_ok:
            # Best path: VAD-enabled turn-taking w/ interruption
            stream = Stream(
                handler=ReplyOnPause(respond, can_interrupt=True),
                modality="audio",
                mode="send-receive",
            )
            print("Using ReplyOnPause with VAD")
        else:
            # No VAD → simple handler that just calls respond() per request
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
            )
            print("Using simple no-VAD handler")
        stream.mount(app)  # exposes /webrtc/offer & /websocket/offer on this FastAPI app
    except Exception as e:
        print(f"Failed to initialize FastRTC Stream: {e}")
        # App will still start (REST only), but no WebRTC endpoints.
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
