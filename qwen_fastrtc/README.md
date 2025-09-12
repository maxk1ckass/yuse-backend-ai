# Qwen FastRTC Voice Assistant

A real-time voice assistant built with FastAPI, FastRTC, and Qwen language model. This service provides audio-to-audio conversation capabilities with speech-to-text, text generation, and text-to-speech functionality.

## Features

- **Real-time Voice Interaction**: Audio input/output with WebRTC streaming
- **Speech-to-Text**: Automatic speech recognition using FastRTC's STT models
- **Text Generation**: Powered by Qwen language model
- **Text-to-Speech**: Audio response generation using FastRTC's TTS models
- **Turn-taking**: Built-in voice activity detection and interruption support
- **Environment Configuration**: Supports `.env` file for configuration

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU
- At least 8GB RAM (16GB+ recommended for GPU)

## Setup Instructions

### 1. Create Python Virtual Environment

```bash
# Navigate to the project directory
cd backend_ai/qwen_fastrtc

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### 3. Download Qwen Model

Download the Qwen model to the `../models/` directory:

```bash
# Create models directory (if it doesn't exist)
mkdir -p ../models

# Download Qwen model (example - adjust path as needed)
# You can download from Hugging Face or use your preferred method
# Example: git lfs clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
```

### 4. Environment Configuration

Create a `.env` file in the project directory:

```bash
# Create .env file
touch .env
```

Add the following configuration to `.env`:

```env
# Qwen model path (adjust to your actual model location)
QWEN_FASTRTC_MODEL=../models/qwen2.5-omni-7b-gptq-int4

# Optional: Add other environment variables as needed
# CUDA_VISIBLE_DEVICES=0
# LOG_LEVEL=INFO
```

## Usage

### 1. Start the Service

```bash
# Make sure virtual environment is activated
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Start the FastAPI server with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The service will be available at:
- **API**: http://localhost:8000
- **Health Check**: http://localhost:8000/health
- **WebRTC Endpoint**: http://localhost:8000/webrtc/offer

### 2. Test the Service

You can test the health endpoint:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "ok": true,
  "model": "../models/qwen2.5-omni-7b-gptq-int4"
}
```

### 3. WebRTC Integration

The service exposes WebRTC endpoints for real-time audio streaming:
- `/webrtc/offer` - WebRTC offer endpoint
- `/websocket/offer` - WebSocket alternative

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `QWEN_FASTRTC_MODEL` | Path to Qwen model directory | `../models/qwen2.5-omni-7b-gptq-int4` |

### Model Configuration

The service uses:
- **STT**: FastRTC's default speech-to-text model (Whisper)
- **TTS**: FastRTC's default text-to-speech model (Kokoro/XTTS)
- **LLM**: Qwen language model (configurable via environment)

## Development

### Project Structure

```
qwen_fastrtc/
├── main.py              # Main FastAPI application
├── requirements.txt     # Python dependencies
├── .env                 # Environment configuration
├── README.md           # This file
└── venv/               # Virtual environment (created during setup)
```

### Key Components

- **FastAPI App**: Main web server
- **FastRTC Stream**: WebRTC audio streaming
- **Qwen Model**: Language model for text generation
- **STT/TTS**: Speech processing via FastRTC

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce model size or use CPU mode
   - Set `device = "cpu"` in the code

2. **Model Not Found**
   - Verify the model path in `.env`
   - Ensure the model is downloaded and accessible

3. **Port Already in Use**
   - Change the port: `uvicorn main:app --port 8001`
   - Kill existing processes using the port

4. **Virtual Environment Issues**
   - Ensure you're in the correct directory
   - Reactivate the virtual environment
   - Reinstall dependencies if needed

### Performance Tips

- Use GPU for better performance
- Ensure sufficient RAM (16GB+ recommended)
- Use quantized models for lower memory usage
- Consider model size vs. performance trade-offs

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /webrtc/offer` - WebRTC offer for audio streaming
- `GET /websocket/offer` - WebSocket alternative

## License

This project uses various open-source components. Please check individual component licenses for details.
