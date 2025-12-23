"""Configuration settings for Liquid Reachy."""

import torch

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Model settings
LFM2_AUDIO_MODEL = "LiquidAI/LFM2-Audio-1.5B"
YOLO_MODEL = "yolov8n.pt"  # Nano model for speed

# Audio settings
SAMPLE_RATE = 16000  # Input sample rate for VAD/recording
OUTPUT_SAMPLE_RATE = 24000  # LFM2-Audio outputs 24kHz
CHANNELS = 1

# VAD settings
VAD_THRESHOLD = 0.5  # Speech probability threshold
SILENCE_DURATION = 0.5  # Seconds of silence before stopping recording (lower = faster response)
MIN_SPEECH_DURATION = 0.3  # Minimum speech duration to process

# Recording settings
CHUNK_DURATION = 0.032  # 32ms chunks (512 samples at 16kHz) - required by Silero VAD
MAX_RECORDING_DURATION = 30.0  # Maximum recording length in seconds

# Vision settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
DETECTION_CONFIDENCE = 0.5

# Generation settings
MAX_NEW_TOKENS = 512
AUDIO_TEMPERATURE = 1.0
AUDIO_TOP_K = 4

# Voice presets - natural language descriptions for LFM2-Audio voice customization
VOICE_PRESETS = {
    "default": "A friendly, clear voice with a warm tone.",
    "professional": "A male speaker with a calm, professional tone and clear enunciation.",
    "friendly": "A cheerful, upbeat voice with an animated and warm tone.",
    "calm": "A soft, soothing voice with a relaxed pace and gentle tone.",
    "robot": "A slightly robotic, neutral voice with precise articulation.",
    "energetic": "An enthusiastic, high-energy voice with dynamic intonation.",
}

# Default voice preset
VOICE_PRESET = "professional"

# System prompt template - includes voice description
def get_system_prompt(voice_preset: str = None) -> str:
    """Generate system prompt with voice description."""
    preset = voice_preset or VOICE_PRESET
    voice_desc = VOICE_PRESETS.get(preset, VOICE_PRESETS["default"])
    return f"""Respond with interleaved text and audio.
Use the following voice: {voice_desc}
You are a friendly robot companion. Keep responses concise."""

# Legacy system prompt (for compatibility)
SYSTEM_PROMPT = get_system_prompt()
