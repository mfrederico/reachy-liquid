"""Lightweight conversation manager for Raspberry Pi 5.

Uses Whisper (ASR) + LiteLLM (text generation) + Piper (TTS).
Much lighter than LFM2-Audio, suitable for edge devices.
"""

import numpy as np
from typing import Generator, Optional

import config


class LiteConversationManager:
    """Lightweight conversation manager using ASR + LLM + TTS pipeline.

    Unlike LFM2-Audio (end-to-end speech model), this uses a pipeline:
    1. Whisper: Audio → Text
    2. LiteLLM: Text → Response text
    3. Piper TTS: Response text → Audio

    This approach uses much less memory and runs on Pi 5.

    Example:
        >>> manager = LiteConversationManager()
        >>> for audio_chunk in manager.generate_response_streaming(audio_input):
        ...     player.play_chunk(audio_chunk)
    """

    def __init__(
        self,
        system_prompt: str = None,
        model_path: str = None,
        model_name: str = "tinyllama",
        voice: str = "amy",
        whisper_model: str = "tiny",
    ):
        """Initialize lite conversation manager.

        Args:
            system_prompt: System prompt for the LLM
            model_path: Path to GGUF model file (optional)
            model_name: Preset model name if model_path not provided
            voice: Piper voice preset
            whisper_model: Whisper model size (tiny, base, small)
        """
        self.system_prompt = system_prompt or config.SYSTEM_PROMPT
        self._vision_context = []
        self._tool_context = None

        # Lazy-loaded components
        self._llm = None
        self._tts = None
        self._transcriber = None

        self._model_path = model_path
        self._model_name = model_name
        self._voice = voice
        self._whisper_model = whisper_model

        # Output sample rate (Piper native is 22050, but we may resample)
        self.output_sample_rate = 24000  # Match LFM2-Audio for compatibility

    def _ensure_llm(self):
        """Lazy-load LLM."""
        if self._llm is not None:
            return

        from .lite_llm import LiteLLM

        print(f"Loading LiteLLM ({self._model_name})...")
        self._llm = LiteLLM(
            model_path=self._model_path,
            model_name=self._model_name,
        )
        self._llm.set_system_prompt(self.system_prompt)

    def _ensure_tts(self):
        """Lazy-load TTS."""
        if self._tts is not None:
            return

        from audio.tts import PiperTTS

        print(f"Loading Piper TTS ({self._voice})...")
        self._tts = PiperTTS(voice=self._voice)

    def _ensure_transcriber(self):
        """Lazy-load transcriber."""
        if self._transcriber is not None:
            return

        from audio import create_transcriber

        print(f"Loading Whisper ({self._whisper_model})...")
        self._transcriber = create_transcriber(
            enabled=True,
            model_size=self._whisper_model,
            device="cpu",  # Pi 5 has no CUDA
        )

    def add_vision_context(self, detections: list[dict]):
        """Add detected objects as context for the conversation.

        Args:
            detections: List of detection dicts from ObjectDetector
        """
        self._vision_context = detections

    def add_tool_context(self, tool_results: list[dict], pre_speak: bool = True):
        """Add tool execution results to the conversation context.

        Args:
            tool_results: List of tool result dicts
            pre_speak: If True, print the message before model response
        """
        if not tool_results:
            self._tool_context = None
            return

        messages = []
        for result in tool_results:
            if result.get("success"):
                msg = result.get("message", result.get("action", "done"))
                messages.append(msg)

        if messages:
            self._tool_context = "; ".join(messages)
            if pre_speak:
                print(f"[{self._tool_context}] ", end="", flush=True)
        else:
            self._tool_context = None

    def _format_context(self) -> str:
        """Format vision and tool context as text."""
        parts = []

        # Vision context
        if self._vision_context:
            objects = [d.get("name", "object") for d in self._vision_context[:5]]
            parts.append(f"[Currently visible: {', '.join(objects)}]")

        # Tool context
        if self._tool_context:
            parts.append(f"[{self._tool_context}]")

        return " ".join(parts)

    def generate_response_streaming(
        self,
        audio_input: np.ndarray,
        sample_rate: int = None,
        user_text: str = None,
    ) -> Generator[np.ndarray, None, None]:
        """Process speech input and generate streaming audio response.

        Pipeline:
        1. Transcribe audio → text (if user_text not provided)
        2. Generate LLM response → text
        3. Synthesize response → audio

        Args:
            audio_input: Audio waveform as numpy array
            sample_rate: Sample rate of input (default: config.SAMPLE_RATE)
            user_text: Pre-transcribed text (skip transcription if provided)

        Yields:
            Audio chunks as numpy arrays
        """
        sample_rate = sample_rate or config.SAMPLE_RATE

        # Step 1: Transcribe if needed (only if not already provided)
        if user_text is None:
            self._ensure_transcriber()
            user_text = self._transcriber.transcribe(audio_input, sample_rate)
            if user_text:
                print(f"You: {user_text}")  # Only print if we did the transcription

        if not user_text or not user_text.strip():
            return

        # Step 2: Build prompt with context
        context = self._format_context()
        if context:
            full_prompt = f"{context}\n\nUser: {user_text}"
        else:
            full_prompt = user_text

        # Clear tool context after use
        self._tool_context = None

        # Step 3: Generate LLM response
        self._ensure_llm()
        self._ensure_tts()

        # Collect response text (for TTS)
        # We stream text first, then synthesize audio
        print("", end="", flush=True)  # Prepare for streaming output

        response_text = ""
        for chunk in self._llm.generate_streaming(full_prompt):
            print(chunk, end="", flush=True)
            response_text += chunk

        print()  # Newline after response

        # Step 4: Synthesize audio from response
        if response_text.strip():
            audio = self._tts.synthesize(response_text)

            # Resample to output rate if needed
            if self._tts.sample_rate != self.output_sample_rate:
                audio = self._tts.resample_to(audio, self.output_sample_rate)

            # Yield in chunks for streaming playback
            chunk_size = int(self.output_sample_rate * 0.1)  # 100ms chunks
            for i in range(0, len(audio), chunk_size):
                yield audio[i:i + chunk_size]

        # Store for access after iteration
        self.last_text_response = response_text

    def reset_conversation(self):
        """Reset the conversation state."""
        if self._llm:
            self._llm.reset()
        self._vision_context = []
        self._tool_context = None


def create_lite_conversation_manager(
    system_prompt: str = None,
    model_name: str = "tinyllama",
    voice: str = "amy",
    whisper_model: str = "tiny",
) -> LiteConversationManager:
    """Create a LiteConversationManager.

    Args:
        system_prompt: System prompt for the LLM
        model_name: LLM model preset
        voice: Piper voice preset
        whisper_model: Whisper model size

    Returns:
        Configured LiteConversationManager
    """
    return LiteConversationManager(
        system_prompt=system_prompt,
        model_name=model_name,
        voice=voice,
        whisper_model=whisper_model,
    )
