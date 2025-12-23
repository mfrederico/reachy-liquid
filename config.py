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

# System prompt - must instruct model to respond with audio
SYSTEM_PROMPT = """Respond with interleaved text and audio. You are a friendly robot companion. Keep responses concise."""
