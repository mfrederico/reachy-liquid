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

    # Default voice models with HuggingFace download URLs
    VOICES = {
        "amy": "en_US-amy-medium",
        "lessac": "en_US-lessac-medium",
        "ryan": "en_US-ryan-medium",
        "arctic": "en_US-arctic-medium",
        "jenny": "en_GB-jenny_dioco-medium",
        "alan": "en_GB-alan-medium",
    }

    # HuggingFace base URL for Piper voices
    HF_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

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

    def _download_voice(self, model_path: Path, config_path: Path):
        """Download voice model from HuggingFace."""
        import urllib.request
        import sys

        # Parse model name to build URL path
        # e.g., en_US-amy-medium -> en/en_US/amy/medium/en_US-amy-medium
        parts = self.model_name.split("-")
        if len(parts) >= 3:
            lang_region = parts[0]  # en_US
            lang = lang_region.split("_")[0]  # en
            voice_name = parts[1]  # amy
            quality = parts[2]  # medium

            base_path = f"{lang}/{lang_region}/{voice_name}/{quality}/{self.model_name}"
        else:
            # Fallback: try direct path
            base_path = self.model_name

        model_url = f"{self.HF_BASE_URL}/{base_path}.onnx"
        config_url = f"{self.HF_BASE_URL}/{base_path}.onnx.json"

        self.model_dir.mkdir(parents=True, exist_ok=True)

        def download_file(url: str, dest: Path, desc: str):
            """Download with progress."""
            def progress_hook(count, block_size, total_size):
                if total_size > 0:
                    percent = min(100, count * block_size * 100 // total_size)
                    mb_done = count * block_size / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    sys.stdout.write(f"\r{desc}: {percent}% ({mb_done:.1f}/{mb_total:.1f} MB)")
                    sys.stdout.flush()

            urllib.request.urlretrieve(url, dest, progress_hook)
            print()  # Newline after progress

        print(f"Downloading Piper voice: {self.model_name}")
        try:
            download_file(model_url, model_path, "Model")
            download_file(config_url, config_path, "Config")
            print(f"Voice downloaded to: {self.model_dir}")
        except Exception as e:
            # Clean up partial downloads
            if model_path.exists():
                model_path.unlink()
            if config_path.exists():
                config_path.unlink()
            raise RuntimeError(f"Failed to download voice: {e}")

    def _load_voice(self):
        """Load voice model for Python API."""
        from piper.voice import PiperVoice

        model_path = self.model_dir / f"{self.model_name}.onnx"
        config_path = self.model_dir / f"{self.model_name}.onnx.json"

        # Auto-download if not found
        if not model_path.exists():
            self._download_voice(model_path, config_path)

        self._voice = PiperVoice.load(str(model_path), str(config_path))

        # Get actual sample rate from model config
        if hasattr(self._voice, 'config') and hasattr(self._voice.config, 'sample_rate'):
            self.sample_rate = self._voice.config.sample_rate

        print(f"Loaded Piper voice: {self.model_name} ({self.sample_rate} Hz)")

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
        # Collect raw audio bytes from synthesize (returns AudioChunk objects)
        audio_bytes = b""
        for chunk in self._voice.synthesize(text):
            audio_bytes += chunk.audio_int16_bytes

        if not audio_bytes:
            print("[TTS] Warning: No audio generated")
            return np.array([], dtype=np.float32)

        # Parse raw 16-bit PCM
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
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
        # Handle empty audio
        if len(audio) == 0:
            return audio

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
