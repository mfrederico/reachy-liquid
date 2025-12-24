"""Lightweight speech-to-text using Whisper.

Supports two backends:
- faster-whisper: CTranslate2-based, 4-10x faster on CPU (recommended for Pi 5)
- openai-whisper: PyTorch-based, slower but widely compatible

Optimized for Pi 5 with the tiny model (~39M params).
"""

import numpy as np
import torch


class FasterTranscriber:
    """Transcribe audio using faster-whisper (CTranslate2-based).

    Much faster than OpenAI Whisper on CPU - typically 4-10x speedup.
    Recommended for Raspberry Pi 5.

    Example:
        >>> transcriber = FasterTranscriber()
        >>> text = transcriber.transcribe(audio_array)
        >>> print(text)
        "Hello, how are you?"
    """

    def __init__(self, model_size: str = "tiny", device: str = None):
        """Initialize faster-whisper transcriber.

        Args:
            model_size: Whisper model size ('tiny', 'base', 'small')
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load faster-whisper model."""
        try:
            from faster_whisper import WhisperModel

            # Map device to faster-whisper format
            compute_type = "float16" if self.device == "cuda" else "int8"
            device_str = self.device if self.device in ("cuda", "cpu") else "cpu"

            print(f"Loading faster-whisper {self.model_size} on {device_str}...")
            self.model = WhisperModel(
                self.model_size,
                device=device_str,
                compute_type=compute_type,
            )
            print(f"Faster-whisper {self.model_size} loaded")
        except ImportError:
            print("faster-whisper not installed. Install with: pip install faster-whisper")
            raise

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str = "en",
    ) -> str:
        """Transcribe audio to text.

        Args:
            audio: Audio waveform as numpy array (float32, mono)
            sample_rate: Sample rate of the audio
            language: Language code (e.g., 'en', 'es', 'fr')

        Returns:
            Transcribed text
        """
        if self.model is None:
            return ""

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import torchaudio
            audio_tensor = torch.from_numpy(audio)
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, sample_rate, 16000
            )
            audio = audio_tensor.numpy()

        # Transcribe with faster-whisper
        segments, _ = self.model.transcribe(
            audio,
            language=language,
            beam_size=1,  # Faster with beam_size=1
            vad_filter=True,  # Skip silence
        )

        # Combine all segments
        text = " ".join(segment.text for segment in segments)
        return text.strip()


class Transcriber:
    """Transcribe audio to text using Whisper.

    Uses the tiny model by default for speed on constrained devices.

    Example:
        >>> transcriber = Transcriber()
        >>> text = transcriber.transcribe(audio_array)
        >>> print(text)
        "Hello, how are you?"
    """

    def __init__(self, model_size: str = "tiny", device: str = None):
        """Initialize transcriber.

        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
                       tiny = 39M params, ~1GB VRAM, fastest
                       base = 74M params, ~1GB VRAM
                       small = 244M params, ~2GB VRAM
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load Whisper model."""
        try:
            import whisper
            print(f"Loading Whisper {self.model_size} on {self.device}...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            print(f"Whisper {self.model_size} loaded")
        except ImportError:
            print("Whisper not installed. Install with: pip install openai-whisper")
            raise

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str = "en",
    ) -> str:
        """Transcribe audio to text.

        Args:
            audio: Audio waveform as numpy array (float32, mono)
            sample_rate: Sample rate of the audio
            language: Language code (e.g., 'en', 'es', 'fr')

        Returns:
            Transcribed text
        """
        if self.model is None:
            return ""

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)

        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            import torchaudio
            audio_tensor = torch.from_numpy(audio)
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, sample_rate, 16000
            )
            audio = audio_tensor.numpy()

        # Transcribe
        result = self.model.transcribe(
            audio,
            language=language,
            fp16=(self.device == "cuda"),
        )

        return result["text"].strip()


class DummyTranscriber:
    """Placeholder transcriber that returns empty string.

    Used when Whisper is not available or not needed.
    """

    def transcribe(self, audio: np.ndarray, **kwargs) -> str:
        return ""


def create_transcriber(
    enabled: bool = True,
    model_size: str = "tiny",
    device: str = None,
    backend: str = "auto",
) -> FasterTranscriber | Transcriber | DummyTranscriber:
    """Create a transcriber (or dummy if disabled).

    Args:
        enabled: Whether to actually load Whisper
        model_size: Whisper model size
        device: Device to run on
        backend: 'auto' (try faster-whisper first), 'faster', or 'openai'

    Returns:
        FasterTranscriber, Transcriber, or DummyTranscriber
    """
    if not enabled:
        return DummyTranscriber()

    # Try faster-whisper first (much faster on CPU)
    if backend in ("auto", "faster"):
        try:
            return FasterTranscriber(model_size=model_size, device=device)
        except ImportError:
            if backend == "faster":
                print("Warning: faster-whisper not available")
                return DummyTranscriber()
            # Fall through to try OpenAI whisper
            print("faster-whisper not found, falling back to openai-whisper")

    # Fall back to OpenAI Whisper
    try:
        return Transcriber(model_size=model_size, device=device)
    except ImportError:
        print("Warning: Whisper not available, transcription disabled")
        return DummyTranscriber()
