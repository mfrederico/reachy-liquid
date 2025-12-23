"""Audio playback for LFM2-Audio output."""

import threading
import time
import numpy as np
import sounddevice as sd
import torch

import config


def find_supported_sample_rate(preferred: int = 24000) -> int:
    """Find a sample rate the default output device supports.

    Args:
        preferred: Preferred sample rate

    Returns:
        Supported sample rate (preferred if supported, else fallback)
    """
    # Common sample rates in order of preference
    candidates = [preferred, 48000, 44100, 22050, 16000]

    for rate in candidates:
        try:
            sd.check_output_settings(samplerate=rate, channels=1)
            return rate
        except Exception:
            continue

    # Last resort: let sounddevice pick
    return None


class AudioPlayer:
    """Low-latency streaming audio player with robust buffering.

    Uses a large circular buffer and a PERSISTENT audio stream to avoid
    repeated open/close cycles that cause PipeWire/Wireplumber issues.
    """

    def __init__(self, prebuffer_seconds: float = 0.8, buffer_seconds: float = 10.0):
        """Initialize audio player.

        Args:
            prebuffer_seconds: Seconds of audio to buffer before starting playback
            buffer_seconds: Total buffer capacity in seconds
        """
        # Find a sample rate the hardware supports
        self.input_sample_rate = config.OUTPUT_SAMPLE_RATE  # What we receive
        self.sample_rate = find_supported_sample_rate(self.input_sample_rate)

        if self.sample_rate != self.input_sample_rate:
            print(f"[Audio] Output device doesn't support {self.input_sample_rate}Hz, using {self.sample_rate}Hz")
            self._needs_resample = True
        else:
            self._needs_resample = False

        self.prebuffer_samples = int(self.sample_rate * prebuffer_seconds)
        self.buffer_size = int(self.sample_rate * buffer_seconds)

        # Circular buffer (pre-allocated for performance)
        self._buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self._write_pos = 0
        self._read_pos = 0
        self._samples_available = 0
        self._lock = threading.Lock()

        # State
        self._stream = None
        self._streaming_active = False  # Whether we're actively playing a response
        self._total_written = 0

        # Create persistent stream on init
        self._init_stream()

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to prevent clipping."""
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val
        return audio.astype(np.float32)

    def _init_stream(self):
        """Initialize persistent audio stream (called once at startup)."""
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self._audio_callback,
            blocksize=1024,
            latency='low',
        )
        self._stream.start()

    def play(self, audio: np.ndarray | torch.Tensor, blocking: bool = True):
        """Play complete audio (non-streaming)."""
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        if audio.ndim > 1:
            audio = audio.squeeze()
        audio = self._normalize(audio)

        # Resample if needed
        if self._needs_resample:
            audio = self._resample(audio, self.input_sample_rate, self.sample_rate)

        sd.play(audio, self.sample_rate)
        if blocking:
            sd.wait()

    def start_stream(self):
        """Prepare for streaming playback (resets buffer, stream stays open)."""
        with self._lock:
            self._buffer.fill(0)
            self._write_pos = 0
            self._read_pos = 0
            self._samples_available = 0
        self._streaming_active = False  # Will be set True after prebuffer
        self._total_written = 0

    def _audio_callback(self, outdata, frames, time_info, status):
        """Audio output callback - reads from circular buffer using fast numpy ops.

        Stream is always running; outputs silence when not actively streaming.
        """
        with self._lock:
            available = self._samples_available

            if available >= frames:
                # Fast numpy read from circular buffer
                end_pos = self._read_pos + frames
                if end_pos <= self.buffer_size:
                    # Contiguous read
                    outdata[:, 0] = self._buffer[self._read_pos:end_pos]
                else:
                    # Wrap-around read
                    first_part = self.buffer_size - self._read_pos
                    outdata[:first_part, 0] = self._buffer[self._read_pos:]
                    outdata[first_part:, 0] = self._buffer[:frames - first_part]
                self._read_pos = end_pos % self.buffer_size
                self._samples_available -= frames
            elif available > 0:
                # Partial read with numpy
                end_pos = self._read_pos + available
                if end_pos <= self.buffer_size:
                    outdata[:available, 0] = self._buffer[self._read_pos:end_pos]
                else:
                    first_part = self.buffer_size - self._read_pos
                    outdata[:first_part, 0] = self._buffer[self._read_pos:]
                    outdata[first_part:available, 0] = self._buffer[:available - first_part]
                outdata[available:, 0] = 0
                self._read_pos = end_pos % self.buffer_size
                self._samples_available = 0
            else:
                # Buffer empty - output silence (don't stop, stream is persistent)
                outdata.fill(0)

    def _resample(self, audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Resample audio using linear interpolation (fast, good enough for speech)."""
        if from_rate == to_rate:
            return audio

        ratio = to_rate / from_rate
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def play_chunk(self, chunk: np.ndarray | torch.Tensor):
        """Add a chunk to the buffer for streaming playback."""
        if isinstance(chunk, torch.Tensor):
            chunk = chunk.cpu().numpy()
        if chunk.ndim > 1:
            chunk = chunk.squeeze()

        chunk = self._normalize(chunk)

        # Resample if hardware doesn't support input sample rate
        if self._needs_resample:
            chunk = self._resample(chunk, self.input_sample_rate, self.sample_rate)

        # Write to circular buffer using fast numpy ops
        with self._lock:
            chunk_len = len(chunk)

            # Check for buffer overflow
            if self._samples_available + chunk_len > self.buffer_size:
                print("[Audio] Buffer overflow, dropping old samples")
                # Skip oldest samples
                skip = (self._samples_available + chunk_len) - self.buffer_size
                self._read_pos = (self._read_pos + skip) % self.buffer_size
                self._samples_available -= skip

            # Fast numpy write to circular buffer
            end_pos = self._write_pos + chunk_len
            if end_pos <= self.buffer_size:
                # Contiguous write
                self._buffer[self._write_pos:end_pos] = chunk
            else:
                # Wrap-around write
                first_part = self.buffer_size - self._write_pos
                self._buffer[self._write_pos:] = chunk[:first_part]
                self._buffer[:chunk_len - first_part] = chunk[first_part:]
            self._write_pos = end_pos % self.buffer_size
            self._samples_available += chunk_len
            self._total_written += chunk_len

        # Mark as actively streaming once we have enough buffered
        if not self._streaming_active and self._samples_available >= self.prebuffer_samples:
            self._streaming_active = True

    def finish_stream(self):
        """Wait for playback to complete (stream stays open for next response)."""
        # Wait for buffer to drain
        while self._samples_available > 0:
            time.sleep(0.05)

        # Let last samples play out
        time.sleep(0.1)

        self._streaming_active = False

    def stop(self):
        """Stop playback immediately (stream stays open)."""
        with self._lock:
            self._samples_available = 0
            self._write_pos = 0
            self._read_pos = 0
        self._streaming_active = False

    def close(self):
        """Close the persistent stream (call on shutdown)."""
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


class BufferedAudioPlayer:
    """Collects all audio first, then plays (guaranteed smooth, higher latency)."""

    def __init__(self):
        self.sample_rate = config.OUTPUT_SAMPLE_RATE
        self._chunks = []

    def play(self, audio, blocking=True):
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        if audio.ndim > 1:
            audio = audio.squeeze()
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val
        sd.play(audio.astype(np.float32), self.sample_rate)
        if blocking:
            sd.wait()

    def start_stream(self):
        self._chunks = []

    def play_chunk(self, chunk):
        if isinstance(chunk, torch.Tensor):
            chunk = chunk.cpu().numpy()
        if chunk.ndim > 1:
            chunk = chunk.squeeze()
        self._chunks.append(chunk.astype(np.float32))

    def finish_stream(self):
        if not self._chunks:
            return
        full_audio = np.concatenate(self._chunks)
        max_val = np.abs(full_audio).max()
        if max_val > 1.0:
            full_audio = full_audio / max_val
        sd.play(full_audio, self.sample_rate)
        sd.wait()
        self._chunks = []

    def stop(self):
        sd.stop()
        self._chunks = []
