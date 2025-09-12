import os
import numpy as np
from fastapi import FastAPI
from fastrtc import Stream, ReplyOnPause, get_stt_model, get_tts_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ----------------------------
# 1) Load models
# ----------------------------
# Speech-to-text & Text-to-speech via FastRTC helpers (swap to your favorites later)
stt = get_stt_model()   # chooses a sensible default (e.g., Whisper); see gallery
tts = get_tts_model()   # chooses a sensible default (e.g., Kokoro/XTTS)           :contentReference[oaicite:1]{index=1}

# Qwen text LLM (fast & simple). Use any Qwen instruct model you prefer.
QWEN_FASTRTC_MODEL = os.getenv("QWEN_FASTRTC_MODEL", "../models/qwen2.5-omni-7b-gptq-int4")
device = 0 if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(QWEN_FASTRTC_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    QWEN_FASTRTC_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
).to(device if isinstance(device, str) else "cuda")

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
    user_text = stt.stt(audio)  # returns transcribed text
    if not user_text or not user_text.strip():
        # Say "sorry, didn't catch that" with your TTS model
        for chunk in tts.stream_tts_sync("Sorry, I didn't catch that. Please try again."):
            yield chunk
        return

    # 2.2 LLM (Qwen) — build a chat-style prompt
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
    for audio_chunk in tts.stream_tts_sync(output_text):
        # Each chunk must be (sample_rate:int, np.ndarray[int16][1, N])
        yield audio_chunk

# ----------------------------
# 3) Build the FastRTC stream & mount to FastAPI
# ----------------------------
# ReplyOnPause = built-in turn-taking (VAD); speak–pause→response; supports interruption. :contentReference[oaicite:2]{index=2}
stream = Stream(
    handler=ReplyOnPause(respond, can_interrupt=True),
    modality="audio",
    mode="send-receive",
)

app = FastAPI()
stream.mount(app)  # exposes /webrtc/offer (and /websocket/offer, etc.) on this FastAPI app  :contentReference[oaicite:3]{index=3}

# Optional: a quick health check
@app.get("/health")
def health():
    return {"ok": True, "model": QWEN_FASTRTC_MODEL}
