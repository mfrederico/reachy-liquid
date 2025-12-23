"""Audio recording with Voice Activity Detection."""

import time
import threading
import numpy as np
import sounddevice as sd

import config
from .vad import VoiceActivityDetector


class AudioRecorder:
    """Records audio from microphone using VAD to detect speech boundaries."""

    def __init__(self, vad: VoiceActivityDetector = None):
        self.vad = vad or VoiceActivityDetector()
        self.chunk_samples = int(config.SAMPLE_RATE * config.CHUNK_DURATION)

        # Barge-in detection state
        self._barge_in_detected = threading.Event()
        self._monitoring = False
        self._captured_audio = []  # Audio captured during barge-in
        self._consecutive_speech = 0
        self._monitor_start_time = 0
        self._delay = 0.5

    def wait_for_speech(self, timeout: float = None) -> bool:
        """
        Wait until speech is detected.

        Args:
            timeout: Maximum time to wait (None for infinite)

        Returns:
            True if speech detected, False if timeout
        """
        print("Listening...", end="", flush=True)
        start_time = time.time()
        self.vad.reset()

        while True:
            if timeout and (time.time() - start_time) > timeout:
                print(" (timeout)")
                return False

            # Record a small chunk
            audio = sd.rec(
                self.chunk_samples,
                samplerate=config.SAMPLE_RATE,
                channels=config.CHANNELS,
                dtype=np.float32,
            )
            sd.wait()

            if self.vad.is_speech(audio.flatten()):
                print(" Speech detected!")
                return True

    def record_until_silence(self) -> np.ndarray:
        """
        Record audio until silence is detected.

        Returns:
            Recorded audio as numpy array (mono, 16kHz)
        """
        print("Recording...", end="", flush=True)
        self.vad.reset()

        chunks = []
        silence_start = None
        recording_start = time.time()

        while True:
            # Check max duration
            if (time.time() - recording_start) > config.MAX_RECORDING_DURATION:
                print(" (max duration reached)")
                break

            # Record chunk
            audio = sd.rec(
                self.chunk_samples,
                samplerate=config.SAMPLE_RATE,
                channels=config.CHANNELS,
                dtype=np.float32,
            )
            sd.wait()

            chunk = audio.flatten()
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

        # Concatenate all chunks
        full_audio = np.concatenate(chunks)

        # Check minimum duration
        duration = len(full_audio) / config.SAMPLE_RATE
        if duration < config.MIN_SPEECH_DURATION:
            print(f" (too short: {duration:.2f}s)")
            return None

        print(f" Recorded {duration:.2f}s")
        return full_audio

    def record_speech(self, timeout: float = None) -> np.ndarray:
        """
        Wait for speech and record until silence.

        Args:
            timeout: Maximum time to wait for speech to start

        Returns:
            Recorded audio, or None if timeout/too short
        """
        if not self.wait_for_speech(timeout):
            return None

        return self.record_until_silence()

    def start_barge_in_monitor(self, delay: float = 0.5):
        """
        Start monitoring for user speech (barge-in detection).
        Uses non-blocking audio check to avoid CUDA conflicts.

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
        Non-blocking check for barge-in. Call this periodically during response generation.

        Returns:
            True if barge-in detected
        """
        if not self._monitoring:
            return False

        # Honor the delay
        if time.time() - self._monitor_start_time < self._delay:
            return False

        # Already detected
        if self._barge_in_detected.is_set():
            return True

        # Number of consecutive speech chunks required (~128ms at 32ms chunks)
        required_consecutive = 4

        try:
            # Non-blocking record
            audio = sd.rec(
                self.chunk_samples,
                samplerate=config.SAMPLE_RATE,
                channels=config.CHANNELS,
                dtype=np.float32,
            )
            sd.wait()

            chunk = audio.flatten()

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
        """Check if barge-in was detected (non-blocking)."""
        return self._barge_in_detected.is_set()

    def record_after_barge_in(self) -> np.ndarray:
        """
        Continue recording after barge-in, including already captured audio.

        Returns:
            Full recorded audio including pre-captured barge-in audio
        """
        print("Recording (barge-in)...", end="", flush=True)

        # Start with already captured audio
        chunks = self._captured_audio.copy()
        self._captured_audio = []

        silence_start = None
        recording_start = time.time()

        while True:
            if (time.time() - recording_start) > config.MAX_RECORDING_DURATION:
                print(" (max duration reached)")
                break

            audio = sd.rec(
                self.chunk_samples,
                samplerate=config.SAMPLE_RATE,
                channels=config.CHANNELS,
                dtype=np.float32,
            )
            sd.wait()

            chunk = audio.flatten()
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
