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

# Simple system prompt for lite mode (small LLMs)
def get_lite_system_prompt(has_camera: bool = False, has_tools: bool = False) -> str:
    """Generate simple system prompt for small LLMs.

    Args:
        has_camera: Whether PTZ camera is available
        has_tools: Whether keyword tools are enabled
    """
    prompt = "You are Lili, a friendly robot assistant. IMPORTANT: Keep ALL responses to 1-2 sentences maximum. Be concise and conversational."

    if has_tools:
        prompt += " State time/date info briefly."

    if has_camera:
        prompt += " You can look around with your camera."

    return prompt


# System prompt template - includes voice description and capabilities
def get_system_prompt(voice_preset: str = None, has_camera: bool = False, has_tools: bool = False) -> str:
    """Generate system prompt with voice description and capabilities.

    Args:
        voice_preset: Voice style to use
        has_camera: Whether PTZ camera is available
        has_tools: Whether keyword tools are enabled
    """
    preset = voice_preset or VOICE_PRESET
    voice_desc = VOICE_PRESETS.get(preset, VOICE_PRESETS["default"])

    base_prompt = f"""Respond with interleaved text and audio.
Use the following voice: {voice_desc}
You are Liquid Lili, a friendly robot companion. Keep responses concise and natural."""

    capabilities = []

    if has_tools:
        capabilities.append("You can tell the current time and date. When SYSTEM INFO provides time/date, state it in your response.")

    if has_camera:
        capabilities.append("You have a camera with PTZ (pan-tilt-zoom) controls. When SYSTEM INFO says you looked somewhere, confirm you moved the camera. You CAN physically look around.")

    # Add general instruction about SYSTEM INFO
    if has_tools or has_camera:
        capabilities.append("IMPORTANT: When you see 'SYSTEM INFO:', that information is REAL and comes from your tools. Always include it in your response.")

    if capabilities:
        base_prompt += "\n\nYour capabilities:\n- " + "\n- ".join(capabilities)

    return base_prompt

# Legacy system prompt (for compatibility)
SYSTEM_PROMPT = get_system_prompt()
