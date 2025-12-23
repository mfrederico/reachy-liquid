"""Audio recording with Voice Activity Detection."""

import time
import threading
import queue
import numpy as np
import sounddevice as sd

import config
from .vad import VoiceActivityDetector


class AudioRecorder:
    """Records audio from microphone using VAD to detect speech boundaries.

    Uses a persistent input stream to avoid rapid device open/close cycles
    that cause PipeWire/Wireplumber issues.
    """

    def __init__(
        self,
        vad: VoiceActivityDetector = None,
        gain: float = 1.0,
        auto_gain: bool = True,
        target_level: float = 0.3,
        max_gain: float = 10.0,
        preroll_chunks: int = 10,
    ):
        """Initialize audio recorder.

        Args:
            vad: Voice activity detector (or None for default)
            gain: Initial/fixed gain multiplier (1.0 = no gain)
            auto_gain: Enable automatic gain control
            target_level: Target RMS level for AGC (0.0-1.0)
            max_gain: Maximum gain multiplier for AGC
            preroll_chunks: Number of chunks to keep before VAD triggers (~320ms at 10 chunks)
        """
        self.vad = vad or VoiceActivityDetector()
        self.chunk_samples = int(config.SAMPLE_RATE * config.CHUNK_DURATION)
        self.base_gain = gain
        self.auto_gain = auto_gain
        self.target_level = target_level
        self.max_gain = max_gain
        self.preroll_chunks = preroll_chunks

        # AGC state
        self._current_gain = gain
        self._gain_smoothing = 0.1  # How fast gain changes (0-1, lower = smoother)

        # Pre-roll buffer (circular buffer of recent audio before speech detected)
        self._preroll_buffer = []

        # Stop flag for graceful shutdown
        self._stop_requested = False

        # Barge-in detection state
        self._barge_in_detected = threading.Event()
        self._monitoring = False
        self._captured_audio = []
        self._consecutive_speech = 0
        self._monitor_start_time = 0
        self._delay = 0.5

        # Persistent input stream
        self._stream = None
        self._chunk_queue = queue.Queue()
        self._init_stream()

    def _init_stream(self):
        """Initialize persistent input stream (called once at startup)."""
        self._stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            dtype=np.float32,
            blocksize=self.chunk_samples,
            callback=self._audio_callback,
        )
        self._stream.start()

    def _audio_callback(self, indata, frames, time_info, status):
        """Audio input callback - puts chunks in queue for processing."""
        # Copy data and put in queue (callback must be fast)
        self._chunk_queue.put(indata.copy().flatten())

    def stop(self):
        """Request stop - causes recording methods to return early."""
        self._stop_requested = True
        # Put None in queue to unblock any waiting get()
        self._chunk_queue.put(None)

    def close(self):
        """Close the persistent stream (call on shutdown)."""
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def reset_stop(self):
        """Reset stop flag for new recording session."""
        self._stop_requested = False

    def _record_chunk(self) -> np.ndarray | None:
        """Get next audio chunk from persistent stream.

        Applies automatic gain control if enabled.

        Returns:
            Audio chunk or None if stopped
        """
        if self._stop_requested:
            return None

        try:
            # Get chunk from queue (blocks until available)
            chunk = self._chunk_queue.get(timeout=1.0)
            if chunk is None:
                return None

            # Automatic gain control
            if self.auto_gain:
                # Calculate RMS level
                rms = np.sqrt(np.mean(chunk ** 2))

                if rms > 0.001:  # Avoid division by zero on silence
                    # Calculate desired gain to reach target level
                    desired_gain = self.target_level / rms
                    desired_gain = min(desired_gain, self.max_gain)  # Limit max gain
                    desired_gain = max(desired_gain, 0.1)  # Minimum gain

                    # Smooth gain changes (attack/release)
                    self._current_gain = (
                        self._gain_smoothing * desired_gain +
                        (1 - self._gain_smoothing) * self._current_gain
                    )

                # Apply current gain
                chunk = chunk * self._current_gain
            else:
                # Apply fixed gain
                chunk = chunk * self.base_gain

            # Soft limiter to prevent harsh clipping
            chunk = np.tanh(chunk)  # Soft saturation

            return chunk.astype(np.float32)
        except Exception:
            return None

    def wait_for_speech(self, timeout: float = None) -> bool:
        """
        Wait until speech is detected.

        Maintains a pre-roll buffer so we don't lose the start of speech.

        Args:
            timeout: Maximum time to wait (None for infinite)

        Returns:
            True if speech detected, False if timeout or stopped
        """
        print("Listening...", end="", flush=True)
        start_time = time.time()
        self.vad.reset()
        self._preroll_buffer = []

        while not self._stop_requested:
            if timeout and (time.time() - start_time) > timeout:
                print(" (timeout)")
                return False

            chunk = self._record_chunk()
            if chunk is None:
                return False

            # Keep pre-roll buffer (circular)
            self._preroll_buffer.append(chunk)
            if len(self._preroll_buffer) > self.preroll_chunks:
                self._preroll_buffer.pop(0)

            if self.vad.is_speech(chunk):
                print(" Speech detected!")
                return True

        return False

    def record_until_silence(self) -> np.ndarray | None:
        """
        Record audio until silence is detected.

        Includes pre-roll buffer from wait_for_speech to capture speech start.

        Returns:
            Recorded audio as numpy array (mono, 16kHz), or None if stopped
        """
        print("Recording...", end="", flush=True)
        self.vad.reset()

        # Start with pre-roll buffer (audio from before VAD triggered)
        chunks = self._preroll_buffer.copy()
        self._preroll_buffer = []

        silence_start = None
        recording_start = time.time()

        while not self._stop_requested:
            # Check max duration
            if (time.time() - recording_start) > config.MAX_RECORDING_DURATION:
                print(" (max duration reached)")
                break

            chunk = self._record_chunk()
            if chunk is None:
                break

            chunks.append(chunk)

            # Check for speech/silence
            if self.vad.is_speech(chunk):
                silence_start = None
            else:
                if silence_start is None:
                    silence_start = time.time()
                elif (time.time() - silence_start) > config.SILENCE_DURATION:
                    print(" Done.")
                    break

        if not chunks:
            return None

        # Concatenate all chunks
        full_audio = np.concatenate(chunks)

        # Check minimum duration
        duration = len(full_audio) / config.SAMPLE_RATE
        if duration < config.MIN_SPEECH_DURATION:
            print(f" (too short: {duration:.2f}s)")
            return None

        print(f" Recorded {duration:.2f}s")
        return full_audio

    def record_speech(self, timeout: float = None) -> np.ndarray | None:
        """
        Wait for speech and record until silence.

        Args:
            timeout: Maximum time to wait for speech to start

        Returns:
            Recorded audio, or None if timeout/too short/stopped
        """
        if self._stop_requested:
            return None

        if not self.wait_for_speech(timeout):
            return None

        return self.record_until_silence()

    def start_barge_in_monitor(self, delay: float = 0.5):
        """
        Start monitoring for user speech (barge-in detection).

        Args:
            delay: Seconds to wait before starting detection (avoids echo)
        """
        self._barge_in_detected.clear()
        self._captured_audio = []
        self._monitoring = True
        self._consecutive_speech = 0
        self._monitor_start_time = time.time()
        self._delay = delay
        self.vad.reset()

    def check_barge_in(self) -> bool:
        """
        Non-blocking check for barge-in.

        Returns:
            True if barge-in detected
        """
        if not self._monitoring or self._stop_requested:
            return False

        # Honor the delay
        if time.time() - self._monitor_start_time < self._delay:
            return False

        # Already detected
        if self._barge_in_detected.is_set():
            return True

        required_consecutive = 4

        try:
            chunk = self._record_chunk()
            if chunk is None:
                return False

            if self.vad.is_speech(chunk):
                self._consecutive_speech += 1
                self._captured_audio.append(chunk)

                if self._consecutive_speech >= required_consecutive:
                    self._barge_in_detected.set()
                    return True
            else:
                if self._barge_in_detected.is_set():
                    self._captured_audio.append(chunk)
                else:
                    self._consecutive_speech = 0
                    self._captured_audio = []

        except Exception:
            pass

        return False

    def stop_barge_in_monitor(self) -> bool:
        """
        Stop barge-in monitoring.

        Returns:
            True if barge-in was detected
        """
        self._monitoring = False
        return self._barge_in_detected.is_set()

    def was_barge_in_detected(self) -> bool:
        """Check if barge-in was detected."""
        return self._barge_in_detected.is_set()

    def record_after_barge_in(self) -> np.ndarray | None:
        """
        Continue recording after barge-in.

        Returns:
            Full recorded audio including pre-captured barge-in audio
        """
        if self._stop_requested:
            return None

        print("Recording (barge-in)...", end="", flush=True)

        chunks = self._captured_audio.copy()
        self._captured_audio = []

        silence_start = None
        recording_start = time.time()

        while not self._stop_requested:
            if (time.time() - recording_start) > config.MAX_RECORDING_DURATION:
                print(" (max duration reached)")
                break

            chunk = self._record_chunk()
            if chunk is None:
                break

            chunks.append(chunk)

            if self.vad.is_speech(chunk):
                silence_start = None
            else:
                if silence_start is None:
                    silence_start = time.time()
                elif (time.time() - silence_start) > config.SILENCE_DURATION:
                    print(" Done.")
                    break

        if not chunks:
            return None

        full_audio = np.concatenate(chunks)
        duration = len(full_audio) / config.SAMPLE_RATE

        if duration < config.MIN_SPEECH_DURATION:
            print(f" (too short: {duration:.2f}s)")
            return None

        print(f" Recorded {duration:.2f}s")
        return full_audio
