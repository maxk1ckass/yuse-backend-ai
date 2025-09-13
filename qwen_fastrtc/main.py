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
    "You are Yuni from YUSE (pronounced as \"use\"), a friendly, patient English instructor.\n"
    "- We previously practiced a restaurant ordering roleplay (you were the waiter, the student was the customer).\n"
    "- We're meeting again to repeat and improve that scenario.\n"
    "- The student will greeting you first, then you should ask if we are ready to recap the learning task.\n"
    "- If the student asks to start, begin like a real waiter greeting at a restaurant.\n"
    "- Default roles: YOU = waiter; STUDENT = customer. If the student says 'switch roles', then YOU = customer; STUDENT = waiter.\n"
    "- Turn-taking: reply in 1–2 short sentences, then stop so the student can speak.\n"
    "- Be encouraging; if needed, give tiny inline corrections in brackets.\n"
    "- Keep vocabulary B1–B2 level, natural and conversational.\n"
    "\n"
    "Here is the exact dialogue transcript of the scenario we practiced:\n"
    "Waiter: Good evening! Welcome to our restaurant. How many people are in your party?\n"
    "Customer: Just one, please. Do you have a table by the window?\n"
    "Waiter: Of course! Here's your menu. Can I get you something to drink?\n"
    "Customer: I'll have water, please. What do you recommend for dinner?\n"
    "Waiter: The salmon is excellent tonight, and the pasta is very popular.\n"
    "Customer: I'll try the salmon. Is it fresh?\n"
    "Waiter: Yes, it was caught this morning. Would you like it grilled or pan-seared?\n"
    "Customer: Grilled, please. And a side salad.\n"
    "Waiter: Perfect! I'll put that order in right away.\n"
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
        # Optional: add a compact memory of the last session so the model leans into the same scene
        {"role": "system", "content": "Session memory: Last time we practiced ordering food; the student asked about salmon vs pasta and requested a side salad."},
        {"role": "user", "content": user_text},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    # Provide an attention mask (no padding from chat template, but this removes warnings and is robust)
    attention_mask = torch.ones_like(inputs)

    with torch.inference_mode():
        output_ids = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=128,   # a bit lower to keep responses concise
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    text = tokenizer.decode(output_ids[0][inputs.shape[-1]:], skip_special_tokens=True)
    return text

# ----------------------------
# Audio pipeline
# ----------------------------
# Track last response time to prevent feedback loops
last_response_time = 0
MIN_RESPONSE_INTERVAL = 2.0  # Minimum 2 seconds between responses

def respond(audio: Tuple[int, np.ndarray]) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Input:  (sample_rate:int, mono_int16_audio: np.ndarray shape [N])
    Yields: (sample_rate:int, mono_int16_audio:int16[1,N]) chunks for TTS
    """
    global last_response_time
    
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

    # Skip LLM when STT returns empty
    if not text or not text.strip():
        print("Empty STT result; skipping LLM.")
        if tts is not None:
            for chunk in tts.stream_tts_sync("Sorry, I didn't catch that. Please try again."):
                yield chunk
        else:
            yield silence_chunk(0.2)
        return

    # 2) LLM
    print(f"Running LLM with input: '{text}'")
    output_text = run_llm(text)
    print(f"LLM output: '{output_text}'")

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
