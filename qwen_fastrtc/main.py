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

# Reduce CUDA allocator fragmentation & avoid ORT using GPU
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("ORT_DISABLE_GPU", "1")  # keep onnxruntime on CPU

HF_MODEL = os.getenv("QWEN_FASTRTC_MODEL", "../models/qwen2.5-omni-7b-gptq-int4")  # change as needed

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda") if USE_CUDA else torch.device("cpu")
DTYPE = torch.float16 if USE_CUDA else torch.float32

if USE_CUDA:
    print(f"[OK] GPU acceleration enabled ({torch.cuda.get_device_name(0)})")
else:
    print("[WARN] Running on CPU (GPU not available)")

def vram(msg=""):
    if not USE_CUDA:
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    print(f"[VRAM]{' ' + msg if msg else ''} allocated={alloc:.1f} MiB reserved={reserved:.1f} MiB")

# ----------------------------
# LLM FIRST — 4-bit quantization to fit V100 16GB
# ----------------------------
print(f"Loading LLM from: {HF_MODEL}")

bnb_available = False
try:
    from transformers import BitsAndBytesConfig
    bnb_available = True
except Exception:
    pass

tokenizer = None
model = None

try:
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)

    if bnb_available:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16 if USE_CUDA else torch.float32,
        )
        # Cap GPU memory so the rest offloads to CPU if needed
        max_memory = {0: "14GiB", "cpu": "64GiB"} if USE_CUDA else {"cpu": "64GiB"}

        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL,
            quantization_config=bnb_config,
            device_map="auto" if USE_CUDA else None,  # auto-place on GPU, offload overflow to CPU
            max_memory=max_memory,
            trust_remote_code=True,
        )
        print("[OK] LLM loaded in 4-bit with bitsandbytes")
    else:
        # Fallback (may OOM on 16GB if other models load on GPU)
        print("[WARN] bitsandbytes not installed; loading fp16/fp32 — may OOM.")
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL,
            dtype=DTYPE,
            device_map=None,           # single device to avoid meta tensors
            low_cpu_mem_usage=False,
            trust_remote_code=True,
        ).to(DEVICE)

    model.eval()
    vram("after LLM load")

    # Warm-up small generation to compile kernels
    try:
        with torch.inference_mode():
            warm_msgs = [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Warm up."},
            ]
            input_ids = tokenizer.apply_chat_template(
                warm_msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
            attention_mask = torch.ones_like(input_ids)
            _ = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=8, do_sample=False)
            if USE_CUDA:
                torch.cuda.synchronize()
        print("[OK] Warm-up done")
        vram("after warm-up")
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
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    text = tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return text

# ----------------------------
# FastRTC (load AFTER LLM so VRAM is prioritized for the LLM)
# ----------------------------
Stream = None
ReplyOnPause = None
fastrtc_ok = False
vad_ok = False
stt = None
tts = None

try:
    from fastrtc import Stream, ReplyOnPause, get_stt_model, get_tts_model
    print("[OK] FastRTC imported")
    fastrtc_ok = True
except Exception as e:
    print(f"[WARN] FastRTC import failed: {e}")

# Keep VAD on CPU (ORT_DISABLE_GPU=1 already set). If onnxruntime not installed, we just skip VAD.
if fastrtc_ok:
    try:
        import onnxruntime  # noqa: F401
        vad_ok = True
        print("[OK] VAD available (onnxruntime on CPU)")
    except Exception as e:
        print(f"[WARN] VAD disabled (onnxruntime import failed): {e}")
        vad_ok = False

# STT/TTS AFTER LLM; try to keep them on CPU to save VRAM
if fastrtc_ok:
    try:
        print("INFO:     Warming up STT model.")
        stt = get_stt_model()  # library chooses backend; ORT is CPU due to env var
        print("INFO:     STT model warmed up.")
        print("[OK] STT ready")
    except Exception as e:
        print(f"[WARN] STT unavailable: {e}")
        stt = None

    try:
        print("INFO:     Warming up TTS model.")
        tts = get_tts_model()
        print("INFO:     TTS model warmed up.")
        print("[OK] TTS ready")
    except Exception as e:
        print(f"[WARN] TTS unavailable: {e}")
        tts = None

vram("after FastRTC/STT/TTS init")

# ----------------------------
# Instructor helpers
# ----------------------------
def get_initial_greeting() -> str:
    prompt = (
        """You are a friendly, enthusiastic English instructor. A student just connected
        to practice English with you. Greet them warmly and naturally. Mention that
        you're excited to help them practice English conversation. You can suggest fun
        activities like roleplay scenarios, but keep it conversational and natural.
        Don't be robotic or template-like."""
    )
    return run_llm(prompt, max_new_tokens=64)
