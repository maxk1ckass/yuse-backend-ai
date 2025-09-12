import os
import numpy as np
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ----------------------------
# 1) Load models
# ----------------------------
# Speech-to-text & Text-to-speech via FastRTC helpers (swap to your favorites later)
stt = None
tts = None

try:
    from fastrtc import Stream, ReplyOnPause, get_stt_model, get_tts_model
    print("FastRTC imported successfully")
    
    try:
        stt = get_stt_model()   # chooses a sensible default (e.g., Whisper); see gallery
        print("STT model loaded successfully")
    except Exception as e:
        print(f"Failed to load STT model: {e}")
        print("Falling back to a simple text input for testing...")
        stt = None

    try:
        tts = get_tts_model()   # chooses a sensible default (e.g., Kokoro/XTTS)
        print("TTS model loaded successfully")
    except Exception as e:
        print(f"Failed to load TTS model: {e}")
        print("Falling back to text-only response for testing...")
        tts = None
        
except Exception as e:
    print(f"Failed to import FastRTC: {e}")
    print("FastRTC is not available. The service will start but WebRTC functionality will be limited.")
    # We'll need to create dummy classes for Stream and ReplyOnPause
    class DummyStream:
        def __init__(self, *args, **kwargs):
            pass
        def mount(self, app):
            pass
    
    class DummyReplyOnPause:
        def __init__(self, *args, **kwargs):
            pass
    
    Stream = DummyStream
    ReplyOnPause = DummyReplyOnPause

# Qwen text LLM (fast & simple). Use any Qwen instruct model you prefer.
QWEN_FASTRTC_MODEL = os.getenv("QWEN_FASTRTC_MODEL", "../models/qwen2.5-omni-7b-gptq-int4")
device = 0 if torch.cuda.is_available() else "cpu"

tokenizer = None
model = None

try:
    print(f"Loading Qwen model from: {QWEN_FASTRTC_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_FASTRTC_MODEL, trust_remote_code=True)
    
    # Try loading with trust_remote_code=True to use the model's own config
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_FASTRTC_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    ).to(device if isinstance(device, str) else "cuda")
    print("Qwen model loaded successfully")
except Exception as e:
    print(f"Failed to load Qwen model: {e}")
    print("Trying alternative loading method...")
    
    try:
        # Try loading as a generic model
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            QWEN_FASTRTC_MODEL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        ).to(device if isinstance(device, str) else "cuda")
        print("Qwen model loaded as generic model")
    except Exception as e2:
        print(f"Alternative loading also failed: {e2}")
        print("The service will start but AI text generation will not be available.")
        print("You can still test the WebRTC connection and basic API endpoints.")
        tokenizer = None
        model = None

SYSTEM_PROMPT = (
    "You are a concise, helpful voice assistant. "
    "Keep answers short and conversational. If asked for code, summarize verbally."
)

# ----------------------------
# 2) Define the audio->audio handler
# ----------------------------
def respond(audio: tuple[int, np.ndarray]):
    """
    Called automatically after FastRTC detects the user paused speaking.
    Input: (sample_rate, mono_int16_audio)
    Yield: (sample_rate, mono_int16_audio) chunks for TTS streaming back.
    """
    # 2.1 ASR
    if stt is None:
        # Fallback: return a simple response indicating STT is not available
        response_text = "Speech-to-text is not available. Please check your installation."
        if tts is not None:
            for chunk in tts.stream_tts_sync(response_text):
                yield chunk
        return
    
    user_text = stt.stt(audio)  # returns transcribed text
    if not user_text or not user_text.strip():
        # Say "sorry, didn't catch that" with your TTS model
        if tts is not None:
            for chunk in tts.stream_tts_sync("Sorry, I didn't catch that. Please try again."):
                yield chunk
        return

    # 2.2 LLM (Qwen) — build a chat-style prompt
    if model is None or tokenizer is None:
        output_text = "AI model is not available. Please check your model installation."
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        output_text = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)

    # 2.3 TTS (stream back audio chunks)
    if tts is not None:
        for audio_chunk in tts.stream_tts_sync(output_text):
            # Each chunk must be (sample_rate:int, np.ndarray[int16][1, N])
            yield audio_chunk
    else:
        # Fallback: return silence if TTS is not available
        print(f"TTS not available. Would have said: {output_text}")
        # Return a short silence
        sample_rate = 16000
        silence = np.zeros((sample_rate,), dtype=np.int16)
        yield (sample_rate, silence)

# ----------------------------
# 3) Build the FastRTC stream & mount to FastAPI
# ----------------------------
# Try to use ReplyOnPause if available, otherwise create a minimal stream
try:
    stream = Stream(
        handler=ReplyOnPause(respond, can_interrupt=True),
        modality="audio",
        mode="send-receive",
    )
    print("Using ReplyOnPause with VAD")
except Exception as e:
    print(f"ReplyOnPause failed (likely due to ONNX Runtime): {e}")
    print("Creating minimal stream without VAD")
    
    # Create a minimal stream that just passes audio through
    try:
        from fastrtc import StreamHandlerBase
        
        class SimpleAudioHandler(StreamHandlerBase):
            def __init__(self, respond_func):
                super().__init__()
                self.respond_func = respond_func
            
            def handle_audio(self, audio):
                return self.respond_func(audio)
        
        stream = Stream(
            handler=SimpleAudioHandler(respond),
            modality="audio",
            mode="send-receive",
        )
        print("Using SimpleAudioHandler")
    except Exception as e2:
        print(f"SimpleAudioHandler also failed: {e2}")
        print("Creating basic stream without custom handler")
        
        # Last resort: create a basic stream
        stream = Stream(
            modality="audio",
            mode="send-receive",
        )
        print("Using basic stream (no custom handler)")

app = FastAPI()
stream.mount(app)  # exposes /webrtc/offer (and /websocket/offer, etc.) on this FastAPI app  :contentReference[oaicite:3]{index=3}

# Optional: a quick health check
@app.get("/health")
def health():
    return {"ok": True, "model": QWEN_FASTRTC_MODEL}
