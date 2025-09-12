# main.py
import os
import time
from typing import Generator, Tuple

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
    print(f"[OK] GPU acceleration enabled ({torch.cuda.get_device_name(0)})")
else:
    print("[WARN] Running on CPU (GPU not available)")

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda") if USE_CUDA else torch.device("cpu")
DTYPE = torch.float16 if USE_CUDA else torch.float32

# ----------------------------
# Optional FastRTC & VAD deps
# ----------------------------
# We try to import FastRTC & ONNX for VAD. If anything fails, we run without VAD.
Stream = None
ReplyOnPause = None
fastrtc_ok = False
vad_ok = False

try:
    from fastrtc import Stream, ReplyOnPause, get_stt_model, get_tts_model
    print("[OK] Voice processing ready")
    fastrtc_ok = True
except Exception as e:
    print(f"[WARN] Voice processing limited: {e}")

# Try to confirm ONNX Runtime availability (GPU or CPU wheel both import as `onnxruntime`)
if fastrtc_ok:
    try:
        import onnxruntime  # noqa: F401
        vad_ok = True
        print("[OK] Smart conversation mode enabled")
    except Exception as e:
        print(f"[WARN] Basic conversation mode (no VAD): {e}")
        vad_ok = False

# ----------------------------
# STT / TTS (optional)
# ----------------------------
stt = None
tts = None
if fastrtc_ok:
    try:
        stt = get_stt_model()   # e.g., Whisper
        print("[OK] Speech recognition ready")
    except Exception as e:
        print(f"[WARN] Speech recognition unavailable: {e}")
        stt = None

    try:
        tts = get_tts_model()   # e.g., Kokoro / XTTS
        print("[OK] Voice synthesis ready")
    except Exception as e:
        print(f"[WARN] Voice synthesis unavailable: {e}")
        tts = None

# ----------------------------
# Qwen LLM
# ----------------------------
print(f"Loading Qwen model from: {HF_MODEL}")
try:
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        dtype=DTYPE,  # modern arg (replaces torch_dtype)
        device_map="auto" if USE_CUDA else None,
        trust_remote_code=True,
    )
    # If CPU path (no device_map), move explicitly
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
    greeting_prompt = (
        "You are a friendly, enthusiastic English instructor. A student just connected "
        "to practice English with you. Greet them warmly and naturally. Mention that "
        "you're excited to help them practice English conversation. You can suggest fun "
        "activities like roleplay scenarios, but keep it conversational and natural. "
        "Don't be robotic or template-like."
    )
    return run_llm(greeting_prompt)

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

def create_instructor_prompt(user_text: str, state: dict) -> str:
    """Create a contextual prompt for the English instructor AI"""
    base_prompt = (
        "You are a friendly, patient English instructor helping a student practice "
        "conversational English. You should:\n"
        "1. Respond naturally and conversationally\n"
        "2. Be encouraging and supportive\n"
        "3. Help with vocabulary and grammar when needed\n"
        "4. Guide roleplay scenarios when appropriate\n"
        "5. Understand what the student is saying and respond meaningfully\n\n"
        "Current conversation state:"
    )

    if state["is_first_interaction"]:
        return (
            f"{base_prompt}\nThis is the first interaction. The student just connected. "
            "Greet them warmly and offer to help with English practice."
        )

    if not state["current_scenario"]:
        return (
            f"{base_prompt}\n"
            f'No active scenario. The student said: "{user_text}"\n'
            "Respond naturally to what they said. If they seem interested in practice, "
            "suggest a roleplay scenario like ordering at a restaurant, asking for directions, "
            "or making small talk."
        )

    if state["current_scenario"] and state["scene_step"] == 0:
        scenario = state["current_scenario"]
        return (
            f"{base_prompt}\n"
            f"You're about to start a roleplay scenario: {scenario['title']}\n"
            f"Description: {scenario['description']}\n"
            f"Your role: {state['ai_role']}\n"
            f"Student's role: {state['user_role']}\n"
            f'The student said: "{user_text}"\n'
            "Start the roleplay naturally as your character. Don't be rigid - respond to what "
            "they actually said while staying in character."
        )

    if state["current_scenario"] and state["scene_step"] > 0:
        scenario = state["current_scenario"]
        return (
            f"{base_prompt}\n"
            f"You're in the middle of a roleplay scenario: {scenario['title']}\n"
            f"Your role: {state['ai_role']}\n"
            f"Student's role: {state['user_role']}\n"
            f"Current step: {state['scene_step']}\n"
            f'The student said: "{user_text}"\n'
            "Respond naturally as your character. Keep the conversation flowing. If they make "
            "mistakes, gently correct them. If they're doing well, encourage them."
        )

    return f"{base_prompt}\nThe student said: '{user_text}'. Respond naturally and helpfully."

def update_conversation_state(user_text: str, ai_response: str, state: dict):
    """Update conversation state based on the interaction"""
    user_text_lower = user_text.lower()

    # Start a scenario?
    if any(phrase in user_text_lower for phrase in ["scenario", "roleplay", "practice", "restaurant", "order"]):
        if not state["current_scenario"]:
            state["current_scenario"] = get_scenario_script()
            state["ai_role"] = "waiter"
            state["user_role"] = "customer"
            state["scene_step"] = 0

    # Ready to begin roleplay?
    if any(phrase in user_text_lower for phrase in ["ready", "begin", "start", "let's go", "yes"]):
        if state["current_scenario"] and state["scene_step"] == 0:
            state["scene_step"] = 1

    # During roleplay, advance step
    if state["current_scenario"] and state["scene_step"] > 0:
        state["scene_step"] += 1

        # Complete?
        if state["scene_step"] >= len(state["current_scenario"]["dialogue"]):
            state["scene_step"] = 0
            state["current_scenario"] = None

    # Reset?
    if any(phrase in user_text_lower for phrase in ["reset", "start over", "new conversation", "clear"]):
        state["current_scenario"] = None
        state["scene_step"] = 0
        state["ai_role"] = None
        state["user_role"] = None

def process_instructor_response(user_text: str) -> str:
    """Process user input with intelligent English instructor responses"""
    global conversation_state
    context_prompt = create_instructor_prompt(user_text, conversation_state)
    llm_response = run_llm(context_prompt)
    update_conversation_state(user_text, llm_response, conversation_state)
    return llm_response

# ----------------------------
# Audio pipeline
# ----------------------------
last_response_time = 0
MIN_RESPONSE_INTERVAL = 2.0  # seconds

conversation_state = {
    "is_first_interaction": True,
    "current_scenario": None,
    "user_role": None,
    "ai_role": None,
    "scene_step": 0
}

def respond(audio: Tuple[int, np.ndarray]) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Input:  (sample_rate:int, mono_int16_audio: np.ndarray)
    Yields: (sample_rate:int, mono_int16_audio:int16[1,N]) chunks
    """
    global last_response_time, conversation_state

    print("=== Audio Processing Debug ===")
    print(f"Audio shape: {audio[1].shape}, sample rate: {audio[0]}")
    audio_length = audio[1].shape[-1] if len(audio[1].shape) > 1 else len(audio[1])
    print(f"Audio duration: {audio_length / audio[0]:.2f} seconds")

    # throttle to avoid loops
    current_time = time.time()
    if current_time - last_response_time < MIN_RESPONSE_INTERVAL:
        print("[TIMER] Too soon since last response, ignoring audio")
        yield silence_chunk(0.2)
        return

    # First interaction: AI greets
    if conversation_state["is_first_interaction"]:
        conversation_state["is_first_interaction"] = False
        greeting_text = get_initial_greeting()
        print(f"[AI] Instructor greeting: '{greeting_text}'")
        if tts is not None:
            last_response_time = current_time
            for chunk in tts.stream_tts_sync(greeting_text):
                yield chunk
        else:
            yield silence_chunk(0.2)
        return

    # STT
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

    # Instructor response
    print(f"Processing user input: '{text}'")
    output_text = process_instructor_response(text)
    print(f"Instructor response: '{output_text}'")

    # TTS or silence
    if tts is not None:
        print("Generating TTS audio...")
        last_response_time = current_time
        for chunk in tts.stream_tts_sync(output_text):
            yield chunk
    else:
        print("TTS not available, returning silence")
        yield silence_chunk(0.2)

    print("=== End Audio Processing ===")

# ----------------------------
# Build Stream + FastAPI
# ----------------------------
app = FastAPI()

if fastrtc_ok:
    try:
        if vad_ok:
            # VAD-enabled turn taking
            stream = Stream(
                handler=ReplyOnPause(respond, can_interrupt=True),
                modality="audio",
                mode="send-receive",
            )
            print("Using ReplyOnPause with VAD")
        else:
            # No VAD: simple handler that just calls respond()
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

        # Mount endpoints: /webrtc/offer & /websocket/offer
        stream.mount(app)

    except Exception as e:
        print(f"Failed to initialize FastRTC Stream: {e}")
        # App still starts; only REST endpoints available.
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
