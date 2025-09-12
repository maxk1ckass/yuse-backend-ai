# main.py
import os
import time
from typing import Generator, Tuple

import numpy as np
from fastapi import FastAPI
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================
# Env & device
# ============================
load_dotenv()

HF_MODEL = os.getenv("QWEN_FASTRTC_MODEL", "../models/qwen2.5-omni-7b-gptq-int4")

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda") if USE_CUDA else torch.device("cpu")
DTYPE = torch.float16 if USE_CUDA else torch.float32

if USE_CUDA:
    print(f"[OK] GPU acceleration enabled ({torch.cuda.get_device_name(0)})")
else:
    print("[WARN] Running on CPU (GPU not available)")

# ============================
# Optional FastRTC & VAD deps
# ============================
Stream = None
ReplyOnPause = None
fastrtc_ok = False
vad_ok = False

try:
    from fastrtc import Stream, ReplyOnPause, get_stt_model, get_tts_model
    print("[OK] FastRTC imported")
    fastrtc_ok = True
except Exception as e:
    print(f"[WARN] FastRTC import failed: {e}")

# Prefer CPU onnxruntime to avoid CUDA runtime contention; GPU ORT is OK if it imports.
if fastrtc_ok:
    try:
        import onnxruntime  # noqa: F401
        vad_ok = True
        print("[OK] VAD available (onnxruntime)")
    except Exception as e:
        print(f"[WARN] VAD disabled (onnxruntime import failed): {e}")
        vad_ok = False

# ============================
# STT / TTS (optional)
# ============================
stt = None
tts = None
if fastrtc_ok:
    try:
        stt = get_stt_model()
        print("[OK] STT ready")
    except Exception as e:
        print(f"[WARN] STT unavailable: {e}")
        stt = None

    try:
        tts = get_tts_model()
        print("[OK] TTS ready")
    except Exception as e:
        print(f"[WARN] TTS unavailable: {e}")
        tts = None

# ============================
# Qwen LLM (single-device load to avoid meta tensors)
# ============================
print(f"Loading Qwen model from: {HF_MODEL}")
try:
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)

    # IMPORTANT: no device_map="auto" and no low_cpu_mem_usage=True
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        dtype=DTYPE,              # modern arg; avoids deprecation
        device_map=None,          # <- force no Accelerate/meta init
        low_cpu_mem_usage=False,  # <- avoid meta tensors
        trust_remote_code=True,
    ).to(DEVICE)
    model.eval()
    print("[OK] LLM loaded on", DEVICE)

    # ---- Warm-up: compile kernels to avoid first-turn stalls ----
    try:
        with torch.inference_mode():
            warm_msgs = [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Warm up."},
            ]
            warm_input_ids = tokenizer.apply_chat_template(
                warm_msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
            warm_attention_mask = torch.ones_like(warm_input_ids)
            _ = model.generate(
                warm_input_ids,
                attention_mask=warm_attention_mask,
                max_new_tokens=8,
                do_sample=False,
            )
            if USE_CUDA:
                torch.cuda.synchronize()
        print("[OK] Warm-up done")
    except Exception as e:
        print(f"[WARN] Warm-up failed: {e}")

except Exception as e:
    print(f"[ERR] LLM load failed: {e}")
    tokenizer = None
    model = None

SYSTEM_PROMPT = (
    "You are a concise, helpful voice assistant. "
    "Keep answers short and conversational. If asked for code, summarize verbally."
)

# ============================
# Helpers
# ============================
def silence_chunk(seconds: float = 0.25, sr: int = 16000) -> Tuple[int, np.ndarray]:
    return (sr, np.zeros(int(sr * seconds), dtype=np.int16))

def run_llm(user_text: str, max_new_tokens: int = 128) -> str:
    if not user_text or not user_text.strip():
        return "Sorry, I didn't catch that. Please try again."
    if model is None or tokenizer is None:
        return "AI model is not available. Please check your model installation."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    attention_mask = torch.ones_like(input_ids)  # no padding; silence warning

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    text = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return text

# ============================
# English Instructor Helpers
# ============================
def get_initial_greeting() -> str:
    prompt = (
        "You are a friendly, enthusiastic English instructor. A student just connected "
        "to practice English with you. Greet them warmly and naturally. Mention that "
        "you're excited to help them practice English conversation. You can suggest fun "
        "activities like roleplay scenarios, but keep it conversational and natural. "
        "Don't be robotic or template-like."
    )
    return run_llm(prompt, max_new_tokens=64)

def get_scenario_script() -> dict:
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
        ]
    }

def create_instructor_prompt(user_text: str, state: dict) -> str:
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

def update_conversation_state(user_text: str, ai_response: str, state: dict):
    s = user_text.lower()

    if any(k in s for k in ["scenario", "roleplay", "practice", "restaurant", "order"]):
        if not state["current_scenario"]:
            state["current_scenario"] = get_scenario_script()
            state["ai_role"] = "waiter"
            state["user_role"] = "customer"
            state["scene_step"] = 0

    if any(k in s for k in ["ready", "begin", "start", "let's go", "yes"]):
        if state["current_scenario"] and state["scene_step"] == 0:
            state["scene_step"] = 1

    if state["current_scenario"] and state["scene_step"] > 0:
        state["scene_step"] += 1
        if state["scene_step"] >= len(state["current_scenario"]["dialogue"]):
            state["scene_step"] = 0
            state["current_scenario"] = None

    if any(k in s for k in ["reset", "start over", "new conversation", "clear"]):
        state["current_scenario"] = None
        state["scene_step"] = 0
        state["ai_role"] = None
        state["user_role"] = None

def process_instructor_response(user_text: str) -> str:
    global conversation_state
    context_prompt = create_instructor_prompt(user_text, conversation_state)
    llm_response = run_llm(context_prompt, max_new_tokens=128)
    update_conversation_state(user_text, llm_response, conversation_state)
    return llm_response

# ============================
# Audio pipeline
# ============================
last_response_time = 0.0
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

    try:
        sr, arr = audio
        arr_len = arr.shape[-1] if arr.ndim > 1 else len(arr)
        print(f"=== Audio Processing === sr={sr}, dur={arr_len/sr:.2f}s ===")

        now = time.time()
        if now - last_response_time < MIN_RESPONSE_INTERVAL:
            print("[TIMER] Throttled; too soon since last response")
            yield silence_chunk(0.2)
            return

        # First interaction: AI greets (no STT)
        if conversation_state["is_first_interaction"]:
            conversation_state["is_first_interaction"] = False
            greeting_text = get_initial_greeting()
            print(f"[AI] Greeting: {greeting_text[:100]}...")
            last_response_time = now
            if tts is not None:
                for chunk in tts.stream_tts_sync(greeting_text):
                    yield chunk
            else:
                yield silence_chunk(0.2)
            return

        # STT
        if stt is not None:
            try:
                text = stt.stt(audio)
                print(f"[STT] '{text}'")
            except Exception as e:
                print(f"[STT err] {e}")
                text = ""
        else:
            text = ""
            print("[STT] unavailable; empty text")

        # If STT empty, don't call LLM (prevents timeouts & nonsense)
        if not text or not text.strip():
            print("[Info] Empty STT result; skipping LLM")
            last_response_time = now
            if tts is not None:
                for chunk in tts.stream_tts_sync("Sorry, I didn't catch that. Please try again."):
                    yield chunk
            else:
                yield silence_chunk(0.2)
            return

        # LLM (English instructor flow)
        reply = process_instructor_response(text)
        print(f"[LLM] {reply[:120]}...")

        # TTS
        last_response_time = now
        if tts is not None:
            for chunk in tts.stream_tts_sync(reply):
                yield chunk
        else:
            yield silence_chunk(0.2)

        print("=== End Audio Processing ===")

    except Exception as e:
        print(f"[respond err] {e}")
        yield silence_chunk(0.2)

# ============================
# FastAPI & Stream
# ============================
app = FastAPI()

if fastrtc_ok:
    try:
        if vad_ok:
            stream = Stream(
                handler=ReplyOnPause(respond, can_interrupt=True),
                modality="audio",
                mode="send-receive",
            )
            print("[OK] Using ReplyOnPause with VAD")
        else:
            def no_vad_handler(audio):
                for ch in respond(audio):
                    yield ch

            stream = Stream(
                handler=no_vad_handler,
                modality="audio",
                mode="send-receive",
            )
            print("[OK] Using simple no-VAD handler")

        # Exposes /webrtc/offer & /websocket/offer
        stream.mount(app)

    except Exception as e:
        print(f"[WARN] Stream init failed: {e}")
else:
    print("[WARN] FastRTC unavailable; REST only")

@app.get("/health")
def health():
    return {
        "ok": True,
        "cuda": USE_CUDA,
        "vad_enabled": bool(fastrtc_ok and vad_ok),
        "fastrtc": bool(fastrtc_ok),
        "model": HF_MODEL,
    }
