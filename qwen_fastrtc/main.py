# main.py
import os
import re
import numpy as np
import time
from typing import Generator, Tuple

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

# Try to confirm ONNX Runtime availability (GPU or CPU is fine)
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
        stt = get_stt_model()   # e.g., Whisper
        print("INFO:     STT model warmed up.")
        print("✅ Speech recognition ready")
    except Exception as e:
        print(f"⚠️  Speech recognition unavailable: {e}")
        stt = None

    try:
        print("INFO:     Warming up TTS model.")
        tts = get_tts_model()   # e.g., Kokoro / XTTS
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
        dtype=DTYPE,  # modern arg (replaces torch_dtype)
        device_map="auto" if USE_CUDA else None,
        trust_remote_code=True,
    )
    if not USE_CUDA:
        model = model.to(DEVICE)

    # Reduce peak VRAM on follow-up turns
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
# SYSTEM PROMPT (English instructor, restaurant scenario)
# ----------------------------
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
# Anti-feedback state & helpers
# ----------------------------
SPEAKING = False
LAST_AI_TEXT = ""
TTS_COOLDOWN_S = 0.8
last_tts_end_time = 0.0

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
    # simple containment or equality covers most speaker-bleed cases
    return a == b or a in b or b in a

# ----------------------------
# Helpers
# ----------------------------
def silence_chunk(seconds: float = 0.25, sr: int = 16000) -> Tuple[int, np.ndarray]:
    return (sr, np.zeros(int(sr * seconds), dtype=np.int16))

def run_llm(user_text: str) -> str:
    """Memory-frugal generation to avoid 2nd-turn VRAM spikes on V100-16GB."""
    if not user_text or not user_text.strip():
        return "Sorry, I didn't catch that. Please try again."
    if model is None or tokenizer is None:
        return "AI model is not available. Please check your model installation."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": "Session memory: Last time we practiced ordering food; the student asked about salmon vs pasta and requested a side salad."},
        {"role": "user", "content": user_text},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    attention_mask = torch.ones_like(inputs)

    # Trim context so KV cache stays small
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
            use_cache=False,  # big win for VRAM headroom
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output_ids[0][inputs.shape[-1]:], skip_special_tokens=True)

    # cleanup
    try:
        del inputs, attention_mask, output_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return text

# ----------------------------
# Audio pipeline
# ----------------------------
last_response_time = 0
MIN_RESPONSE_INTERVAL = 2.0  # Minimum 2 seconds between responses

def respond(audio: Tuple[int, np.ndarray]) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Input:  (sample_rate:int, mono_int16_audio: np.ndarray shape [N])
    Yields: (sample_rate:int, mono_int16_audio:int16[1,N]) chunks for TTS
    """
    global last_response_time, LAST_AI_TEXT, SPEAKING, last_tts_end_time

    sr, arr = audio
    now = time.time()

    # Gate 1: if the bot is currently speaking, ignore mic frames
    if SPEAKING:
        yield silence_chunk(0.1)
        return

    # Gate 2: small cooldown after TTS finishes to avoid speaker bleed
    if now - last_tts_end_time < TTS_COOLDOWN_S:
        yield silence_chunk(0.1)
        return

    print(f"=== Audio Processing Debug ===")
    print(f"Audio shape: {arr.shape}, sample rate: {sr}")
    audio_length = arr.shape[-1] if len(arr.shape) > 1 else len(arr)
    print(f"Audio duration: {audio_length / sr:.2f} seconds")

    # Gate 3: response throttle
    if now - last_response_time < MIN_RESPONSE_INTERVAL:
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
            text = ""

    # If empty STT, nudge user and return
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

# ----------------------------
# Build Stream + FastAPI
# ----------------------------
app = FastAPI()

if fastrtc_ok:
    try:
        if vad_ok:
            # Use non-interruptible VAD at first to avoid mid-TTS reentry
            stream = Stream(
                handler=ReplyOnPause(respond, can_interrupt=False),
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
