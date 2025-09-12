# main.py
import os
import numpy as np
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

# Debug CUDA detection
print("=== CUDA Debug Info ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("CUDA not available - checking why...")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
    print(f"PATH contains CUDA: {'cuda' in os.environ.get('PATH', '').lower()}")
print("========================")

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
    print("FastRTC imported successfully")
    fastrtc_ok = True
except Exception as e:
    print(f"FastRTC import failed (limited WebRTC): {e}")

# Try to confirm ONNX Runtime availability (GPU preferred, CPU acceptable)
if fastrtc_ok:
    try:
        # Importing onnxruntime triggers DLL checks on Windows. If it fails, we won't use VAD.
        import onnxruntime  # noqa: F401
        vad_ok = True
        print("onnxruntime available; VAD can be enabled.")
    except Exception as e:
        print(f"onnxruntime not available; VAD will be disabled: {e}")
        vad_ok = False

# ----------------------------
# STT / TTS (optional)
# ----------------------------
stt = None
tts = None
if fastrtc_ok:
    try:
        stt = get_stt_model()   # e.g., Whisper
        print("STT model loaded successfully")
    except Exception as e:
        print(f"Failed to load STT model: {e}")
        stt = None

    try:
        tts = get_tts_model()   # e.g., Kokoro / XTTS
        print("TTS model loaded successfully")
    except Exception as e:
        print(f"Failed to load TTS model: {e}")
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
# Audio pipeline
# ----------------------------
def respond(audio: Tuple[int, np.ndarray]) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Input:  (sample_rate:int, mono_int16_audio: np.ndarray shape [N])
    Yields: (sample_rate:int, mono_int16_audio:int16[1,N]) chunks for TTS
    """
    print(f"=== Audio Processing Debug ===")
    print(f"Audio shape: {audio[1].shape}, sample rate: {audio[0]}")
    print(f"Audio duration: {len(audio[1]) / audio[0]:.2f} seconds")
    
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

    # 2) LLM
    print(f"Running LLM with input: '{text}'")
    output_text = run_llm(text)
    print(f"LLM output: '{output_text}'")

    # 3) TTS or silence
    if tts is not None:
        print("Generating TTS audio...")
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
