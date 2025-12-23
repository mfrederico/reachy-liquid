"""Text-to-speech using Piper TTS.

Lightweight, fast TTS suitable for Raspberry Pi 5.
"""

import io
import wave
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional, Generator

import config


class PiperTTS:
    """Text-to-speech using Piper.

    Piper is a fast, local neural TTS that runs well on Pi 5.

    Example:
        >>> tts = PiperTTS()
        >>> audio = tts.synthesize("Hello, how are you?")
        >>> # audio is a numpy array at 22050 Hz
    """

    # Default voice models (can be downloaded from HuggingFace)
    VOICES = {
        "amy": "en_US-amy-medium",
        "lessac": "en_US-lessac-medium",
        "ryan": "en_US-ryan-medium",
        "arctic": "en_US-arctic-medium",
        "jenny": "en_GB-jenny_dioco-medium",
        "alan": "en_GB-alan-medium",
    }

    def __init__(
        self,
        voice: str = "amy",
        model_dir: Optional[Path] = None,
        sample_rate: int = 22050,
    ):
        """Initialize Piper TTS.

        Args:
            voice: Voice preset name or full model name
            model_dir: Directory containing model files (default: ~/.local/share/piper)
            sample_rate: Output sample rate (Piper native is 22050)
        """
        self.sample_rate = sample_rate
        self.model_dir = model_dir or Path.home() / ".local" / "share" / "piper"

        # Resolve voice name
        if voice in self.VOICES:
            self.model_name = self.VOICES[voice]
        else:
            self.model_name = voice

        self._voice = None
        self._use_cli = True  # Start with CLI, try Python API

        # Try to use Python API if available
        try:
            from piper.voice import PiperVoice
            self._use_cli = False
            self._load_voice()
        except ImportError:
            print("Piper Python API not available, using CLI")

    def _load_voice(self):
        """Load voice model for Python API."""
        from piper.voice import PiperVoice

        model_path = self.model_dir / f"{self.model_name}.onnx"
        config_path = self.model_dir / f"{self.model_name}.onnx.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Download with: piper --download-model {self.model_name}"
            )

        self._voice = PiperVoice.load(str(model_path), str(config_path))
        print(f"Loaded Piper voice: {self.model_name}")

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize speech from text.

        Args:
            text: Text to speak

        Returns:
            Audio samples as float32 numpy array
        """
        if self._use_cli:
            return self._synthesize_cli(text)
        else:
            return self._synthesize_api(text)

    def _synthesize_api(self, text: str) -> np.ndarray:
        """Synthesize using Python API."""
        # Create in-memory WAV file
        wav_buffer = io.BytesIO()

        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)

            # Synthesize
            self._voice.synthesize(text, wav_file)

        # Read back and convert to float32
        wav_buffer.seek(0)
        with wave.open(wav_buffer, 'rb') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            audio /= 32768.0  # Normalize to [-1, 1]

        return audio

    def _synthesize_cli(self, text: str) -> np.ndarray:
        """Synthesize using Piper CLI."""
        try:
            # Run piper and capture raw audio output
            result = subprocess.run(
                [
                    "piper",
                    "--model", self.model_name,
                    "--output-raw",
                ],
                input=text.encode(),
                capture_output=True,
                check=True,
            )

            # Parse raw 16-bit PCM
            audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)
            audio /= 32768.0

            return audio

        except subprocess.CalledProcessError as e:
            print(f"Piper CLI error: {e.stderr.decode()}")
            return np.array([], dtype=np.float32)
        except FileNotFoundError:
            print("Piper CLI not found. Install with: pip install piper-tts")
            return np.array([], dtype=np.float32)

    def synthesize_streaming(self, text: str) -> Generator[np.ndarray, None, None]:
        """Synthesize speech with streaming output.

        Yields chunks as they're generated for lower latency.

        Args:
            text: Text to speak

        Yields:
            Audio chunks as float32 numpy arrays
        """
        # For now, synthesize whole thing then chunk it
        # TODO: True streaming with sentence splitting
        audio = self.synthesize(text)

        if len(audio) == 0:
            return

        # Yield in chunks (~100ms each)
        chunk_size = int(self.sample_rate * 0.1)
        for i in range(0, len(audio), chunk_size):
            yield audio[i:i + chunk_size]

    def resample_to(self, audio: np.ndarray, target_rate: int) -> np.ndarray:
        """Resample audio to target sample rate.

        Args:
            audio: Input audio at self.sample_rate
            target_rate: Desired sample rate

        Returns:
            Resampled audio
        """
        if target_rate == self.sample_rate:
            return audio

        try:
            import torchaudio
            import torch

            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            resampled = torchaudio.functional.resample(
                audio_tensor, self.sample_rate, target_rate
            )
            return resampled.squeeze(0).numpy()
        except ImportError:
            # Fallback: simple linear interpolation
            ratio = target_rate / self.sample_rate
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def create_tts(voice: str = "amy") -> PiperTTS:
    """Create a TTS instance.

    Args:
        voice: Voice preset name

    Returns:
        Configured PiperTTS instance
    """
    return PiperTTS(voice=voice)
