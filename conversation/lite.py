"""Lightweight conversation manager for Raspberry Pi 5.

Uses Whisper (ASR) + LiteLLM (text generation) + Piper (TTS).
Much lighter than LFM2-Audio, suitable for edge devices.
"""

import numpy as np
import time
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

    def _normalize_time_for_tts(self, text: str) -> str:
        """Convert time formats to spoken form."""
        import re

        def time_to_spoken(match):
            hour = int(match.group(1))
            minute = int(match.group(2))
            second = match.group(3)  # May be None
            period = match.group(4) or ""  # AM/PM

            # Build spoken time
            parts = []

            # Hour
            if hour == 0:
                parts.append("twelve")
            elif hour <= 12:
                parts.append(str(hour))
            else:
                parts.append(str(hour - 12))

            # Minutes
            if minute == 0:
                if not period:
                    parts.append("o'clock")
            elif minute < 10:
                parts.append(f"oh {minute}")
            else:
                parts.append(str(minute))

            # Skip seconds for speech (too verbose)

            # AM/PM
            if period:
                parts.append(period.upper().replace(".", ""))

            return " ".join(parts)

        # Match times like 01:08:21 PM, 13:45, 1:30 AM
        text = re.sub(
            r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM|am|pm|A\.M\.|P\.M\.)?",
            time_to_spoken,
            text
        )

        return text

    def _clean_for_tts(self, text: str) -> str:
        """Clean response text for TTS (remove role prefixes, normalize text)."""
        import re

        # Remove any single-word prefix followed by colon at start
        # Catches: "Robot:", "A:", "Response:", "Assistant:", etc.
        text = re.sub(r"^\s*\w+:\s*", "", text)

        # Also remove bracketed prefixes like "[Billboard Baggins]"
        text = re.sub(r"^\[[^\]]+\]\s*", "", text)

        # Remove "responds/says" patterns
        text = re.sub(r"^(responds?|says?)\s*:?\s*", "", text, flags=re.IGNORECASE)

        # Remove "as per SYSTEM INFO" and similar
        text = re.sub(r",?\s*as per (SYSTEM INFO|system info|the system)\.?", "", text, flags=re.IGNORECASE)

        # Normalize times for natural speech
        text = self._normalize_time_for_tts(text)

        return text.strip()

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
            t_stt_start = time.perf_counter()
            user_text = self._transcriber.transcribe(audio_input, sample_rate)
            t_stt_end = time.perf_counter()
            if user_text:
                print(f"\nYou: {user_text}")
                print(f"[STT: {(t_stt_end - t_stt_start)*1000:.0f}ms]")

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

        # Step 3: Generate LLM response with sentence-based TTS streaming
        self._ensure_llm()
        self._ensure_tts()

        import re
        chunk_size = int(self.output_sample_rate * 0.1)  # 100ms audio chunks

        response_text = ""
        sentence_buffer = ""
        t_llm_start = time.perf_counter()
        t_first_token = None
        token_count = 0
        tts_total_ms = 0

        def is_sentence_end(text: str) -> bool:
            """Check if text ends with a sentence boundary (not a time colon)."""
            text = text.rstrip()
            if not text:
                return False

            # Must end with sentence punctuation
            if text[-1] not in ".!?":
                return False

            # Make sure it's not an abbreviation like "Dr." or "Mr."
            abbrevs = ["Mr.", "Mrs.", "Ms.", "Dr.", "Jr.", "Sr.", "vs.", "etc.", "e.g.", "i.e."]
            for abbr in abbrevs:
                if text.endswith(abbr):
                    return False

            return True

        print("A: ", end="", flush=True)
        for token in self._llm.generate_streaming(full_prompt):
            if t_first_token is None:
                t_first_token = time.perf_counter()
            token_count += 1
            print(token, end="", flush=True)
            response_text += token
            sentence_buffer += token

            # Check if we have a complete sentence
            has_sentence = is_sentence_end(sentence_buffer)

            if has_sentence:
                # Extract complete sentence and synthesize
                sentence = self._clean_for_tts(sentence_buffer.strip())
                if sentence:
                    t_tts_start = time.perf_counter()
                    audio = self._tts.synthesize(sentence)
                    t_tts_end = time.perf_counter()
                    tts_total_ms += (t_tts_end - t_tts_start) * 1000

                    # Resample if needed
                    if self._tts.sample_rate != self.output_sample_rate:
                        audio = self._tts.resample_to(audio, self.output_sample_rate)

                    # Yield complete sentence audio as one array
                    # Caller handles pacing and barge-in between sentences
                    yield audio

                sentence_buffer = ""

        t_llm_end = time.perf_counter()
        print()  # Newline after response

        # Print timing metrics
        llm_total_ms = (t_llm_end - t_llm_start) * 1000
        first_token_ms = (t_first_token - t_llm_start) * 1000 if t_first_token else 0
        print(f"[LLM: {llm_total_ms:.0f}ms total, {first_token_ms:.0f}ms to first token, {token_count} tokens]")
        print(f"[TTS: {tts_total_ms:.0f}ms total]")

        # Synthesize any remaining text
        if sentence_buffer.strip():
            sentence = self._clean_for_tts(sentence_buffer.strip())
            if sentence:
                t_tts_start = time.perf_counter()
                audio = self._tts.synthesize(sentence)
                t_tts_end = time.perf_counter()
                print(f"[TTS remaining: {(t_tts_end - t_tts_start)*1000:.0f}ms]")

                if self._tts.sample_rate != self.output_sample_rate:
                    audio = self._tts.resample_to(audio, self.output_sample_rate)

                yield audio

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
